import os
import re
import json
import numpy as np
import cv2

from fractions import Fraction
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'unicedit'

SAVE_DATASET_DIR_NAME = 'UnicEdit'

# ==============================================================================
# 【这个数据集只能产出图像编辑数据集，不能产出文生图数据集】
# 上游017解包出来的每一行只有20个key，其中真正的文本字段只有
# prompt_en(英文编辑指令) / prompt_cn(中文编辑指令) / instruction 这三个，
# 而instruction实测与prompt_en**逐字100%相同**(全库2025000行，0条不同)，是纯冗余。
# 整个数据集**没有任何一列是整图内容描述(caption)**: 既没有编辑前原图的描述、
# 也没有编辑后图的描述。编辑指令只说"要改什么"、不说"整张图是什么"，
# 形如"Change the large pink cotton candy-like object in the center of the bowl
# to a rainbow-colored sphere. ... Ensure the rest of the objects and environment
# remain unchanged."，拿它当t2i的prompt会得到完全错误的图文对。
# 所以本数据集只走ti2i这一条链路、不另写t2i脚本
# (与014.resave_scaleedit_12m_ti2i_dataset.py、
#  016.resave_crispedit_2m_ti2i_dataset.py同样的处置)。
# 上游017自己也把dataset_task_type写死成image_edit，
# HF README的task_categories是image-to-image。
#
# ==============================================================================
# 【为什么数据集叫"10M"、实测却只有2025000对(四方对账结论)】
# 不是本地没下全，而是**上游官方只上传了完整数据集的135/681片(19.8%)**:
#   每片行数(135/135片完全一致、无尾片)            15000
#   文件名声称的总片数 train-%05d-of-00681          681
#       681 × 15000 = 10215000  <- "10M"这个名字的来源，精确匹配
#   上游实际上传的片数                              135
#       135 × 15000 =  2025000  <- 本地拿到的全部，正好是宣称规模的1/5
# 四方对账(全部一致，证明135片就是线上全部)：
#   a. 本地data/目录                                     135个parquet;
#   b. HF线上tree API /tree/main/data?recursive=1(翻页拉全) 135个parquet,
#      编号min=0 / max=240;
#   c. HF仓库元信息siblings                              137个文件
#      (135 parquet + README.md + .gitattributes);
#   d. 本地下载缓存.cache/huggingface/ 里 *.incomplete **0个**(无中断下载)，
#      且本地字节数与线上usedStorage(5.14 TB)100%一致。
# 缺失编号的形态(0~240之间严重不连续、241~680整段全无)是典型的"分批上传中途终止":
#   已传 0-25,30-35,37-60,62-67,69-70,75-113,115-127,129-140,160-162,227-229,240
#   缺失 26-29,36,61,68,71-74,114,128,141-159,163-226,230-239,241-680 (共546个)
# 很可能是踩到了存储配额: 上游把**未压缩PNG原图**直接内嵌进parquet(单片约39GB)，
# 已传135片就5.14TB，完整681片推算要25.9TB。
# README的YAML里size_categories写的是10M<n<100M，那是按论文规模填的标签
# (CVPR 2026, arXiv:2512.02790)，不是按已上传文件填的。
# 【对本脚本的影响】拿到的2025000对**每一对都是完整可训练的**(上游017自校验
# 0隔离样本/0错误、4050000张图全部落盘且逐张校验过落盘字节数)，不存在半个样本对。
# 所以这里把135与2025000写死成硬对账常量: 上游哪天补传了新片，重跑会立刻硬失败、
# 提醒先重跑017解包，而不是静默用一个过期的子集继续训。
# 落盘名用<parquet名>_<行号>合成，新片有新的parquet名、天然不会与现有样本撞名，
# 未来补片可以增量追加而不需要重命名已有产物。
#
# ==============================================================================
# 【上游017.unzip_unicedit_10m_dataset.py的产物规格(全量实测，非抽样)】
# UnicEdit-10M/
# ├── unzip_annotations/<4个edit_task>/<21个edit_subtask>/<parquet名>.jsonl
# │                                                2556个文件 / 2025000行
# ├── unzip_images/<edit_task>/<edit_subtask>/<parquet名>/
# │       <parquet名>_%08d_src.png    参考图(编辑前原图)
# │       <parquet名>_%08d_edit.png   编辑后图
# │                                                4050000张(约5.2T)
# └── unzip_check_missing_images.json  上游自校验: 0隔离样本 / 0错误 / 1条warning
#                                      (那条warning就是下面的unknown_subtask脏值)
#
# 【逐行20个key的全量核对结论】
#   prompt_en          英文编辑指令，全非空，长28~4691(p50=213 / p99=420)  -> ti2i_caption
#   prompt_cn          中文编辑指令，全非空，长8~1342(p50=56 / p99=114)    -> ti2i_caption
#   instruction        **与prompt_en逐字100%相同**(0条不同)，纯冗余         -> 丢弃
#   task_name          4类粗粒度任务族，全非空                              -> 只做硬对账
#   subtask_name       20类细粒度任务族 + unknown_subtask(1行)              -> 子集名
#   edit_task          带空格的原值(如"Attribute Editing")，与task_name同义 -> 丢弃
#   edit_subtask       带空格的原值，与subtask_name同义                     -> 丢弃
#   parquet_name       分片名，与row_index组合成**全局唯一**的样本定位       -> 拼保存图像名
#   row_index          片内行号，同上                                       -> 拼保存图像名
#   sample_key         "<task>/<subtask>/<parquet>_<行号>"，与上两列同义     -> 只做交叉校验
#   md5_key            2019819个唯一 + 5181个重复(见下面说明)               -> 丢弃
#   reference_image_path_list  恒为长度1的list，全非空                      -> 定位参考图
#   edited_image_path          全非空                                       -> 定位编辑后图
#   reference_image_num        **恒为1**(2025000/2025000)                   -> 不采信，现算
#   input_image_shape          [宽, 高]                                     -> 长宽比预筛
#   edited_image_shape         [宽, 高]                                     -> 长宽比预筛
#   input_image_suffix         **恒为.png**                                 -> 丢弃
#   edited_image_suffix        **恒为.png**                                 -> 丢弃
#   dataset_task_type          恒为image_edit(整库单值)                     -> 丢弃
# 上游的unzip_check_missing_images.json也不读不搬(它的实测值已写死成本脚本的
# EXPECTED_*常量用于硬对账)。
#
# 【为什么5181个重复md5_key不做去重】
# 上游017已逐行核对过: 这5181个key各出现2次，两行的src_image字节相同、
# 但prompt与edit_image不同(同一张源图的多条不同编辑)，
# **两行都是完整有效的独立编辑样本对**，去重会平白丢掉5181对。
# 落盘名用<parquet名>_<行号>合成(全库2025000个组合**0重名**，已全量验证)，
# 与md5_key无关，所以重复key既不会覆盖也不需要处理，只在校验报告里记一笔。
#
# 【图像规格实测】
# 全库input_image_shape与edited_image_shape**2025000行完全相同**(0条不同)，
# 即参考图与编辑后图天然像素对齐; 编辑后图短边最小672、最大宽高比2.33
# (分辨率全是16的倍数，1024x1024最多、占57.6%)。
# 抽样150对(300张)逐张PIL打开: mode **100%是RGB**、0缺图、0坏图、
# 宽高与标注100%一致、参考图与编辑后图尺寸150/150完全相同。
# ==============================================================================

# 本脚本只读unzip_annotations这一套标注(2556个jsonl / 2025000行)定位样本对，
# **绝不os.walk图像目录**: 上游解出405万个小文件，扫目录树在NAS上不可接受
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 上游标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如unzip_images/Attribute_Editing/Color_Alteration/train-00000-of-00681/
#  train-00000-of-00681_00000000_src.png)，不需要再往前拼任何子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 上游落盘图像名的后缀: 参考图是_src、编辑后图是_edit
# (扩展名实测恒为.png，但取前缀时仍然先去扩展名再去这个后缀，
#  上游哪天换成JPEG也不会取错)
LOAD_REFERENCE_IMAGE_NAME_SUFFIX = '_src'

LOAD_EDITED_IMAGE_NAME_SUFFIX = '_edit'

# 粗粒度任务族(4类)，只用于解析阶段的硬对账，不参与子集划分
ANNOTATION_TASK_NAME_KEY_NAME = 'task_name'

# 细粒度任务族(20类 + 1行脏值)，**子集就是按它划分的**。
# 不用task_name分子集的原因: 4类太粗，长尾任务(counting_change 125对、
# object_extraction 151对)会被淹没在几十万对的大类里，没法单独做过采样/限流
ANNOTATION_SUBTASK_NAME_KEY_NAME = 'subtask_name'

# 英文编辑指令。注意不用instruction: 它与prompt_en逐字100%相同(全库0条不同)，
# 直接读prompt_en语义更明确、也和prompt_cn对称
ANNOTATION_ENGLISH_CAPTION_KEY_NAME = 'prompt_en'

# 中文编辑指令
ANNOTATION_CHINESE_CAPTION_KEY_NAME = 'prompt_cn'

# 参考图相对路径(list，本数据集恒为长度1)与编辑后图相对路径(字符串)
ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'reference_image_path_list'

ANNOTATION_EDITED_IMAGE_KEY_NAME = 'edited_image_path'

# 上游记录的样本唯一键，形如
# "Attribute_Editing/Color_Alteration/train-00000-of-00681_00000000"。
# 只用来和从图像文件名现取的前缀交叉比对，不写进新标注
ANNOTATION_SAMPLE_KEY_KEY_NAME = 'sample_key'

# 上游的分片名与片内行号，两者组合出全局唯一的样本定位，用于拼保存图像名
ANNOTATION_PARQUET_NAME_KEY_NAME = 'parquet_name'

ANNOTATION_ROW_INDEX_KEY_NAME = 'row_index'

# 上游解图像header得到的参考图/编辑后图真实宽高([宽, 高])。
# **只用于长宽比预筛与统计，绝不采信**: 写进json的width/height一律取自
# 实际写盘数组的shape
ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME = 'input_image_shape'

ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME = 'edited_image_shape'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 新标注固定只存这七个key，多一个少一个都在收尾自校验里报错
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# ==============================================================================
# 【本数据集特有: ti2i_caption与ti2i_caption_length这两个key的value都是字典】
# 这是本项目第一个**中英双语指令**的图像编辑数据集: prompt_en与prompt_cn
# 全库2025000行都非空、且是同一条编辑语义的两种语言表述，两条都有训练价值，
# 丢掉任何一条都是信息损失; 而json的key是编辑后图像名、一个样本对只能有一条记录，
# 没法像单语数据集那样把caption直接写成字符串。
# 所以本数据集把这两个key的value都改成字典(其余数据集仍是字符串/整数):
#   "ti2i_caption": {
#       "english_ti2i_caption": "<prompt_en原文，已strip>",
#       "chinese_ti2i_caption": "<prompt_cn原文，已strip>"
#   },
#   "ti2i_caption_length": {
#       "english_ti2i_caption_length": len(english_ti2i_caption),
#       "chinese_ti2i_caption_length": len(chinese_ti2i_caption)
#   }
# 下游读取时必须显式感知这个结构差异。
# 收尾自校验会专门校验: 两个字典的key集合严格等于下面两个列表、
# 两个length都等于对应字符串的实际len()、两条指令都在[MIN, MAX]区间内、
# 都不含[Vn*]占位符、都不是null字面量、都含有至少一个文字字符。
# ==============================================================================
SAVE_TI2I_CAPTION_KEY_NAME = 'ti2i_caption'

SAVE_TI2I_CAPTION_LENGTH_KEY_NAME = 'ti2i_caption_length'

SAVE_ENGLISH_CAPTION_KEY_NAME = 'english_ti2i_caption'

SAVE_CHINESE_CAPTION_KEY_NAME = 'chinese_ti2i_caption'

SAVE_ENGLISH_CAPTION_LENGTH_KEY_NAME = 'english_ti2i_caption_length'

SAVE_CHINESE_CAPTION_LENGTH_KEY_NAME = 'chinese_ti2i_caption_length'

SAVE_TI2I_CAPTION_KEY_NAME_LIST = [
    SAVE_ENGLISH_CAPTION_KEY_NAME,
    SAVE_CHINESE_CAPTION_KEY_NAME,
]

SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST = [
    SAVE_ENGLISH_CAPTION_LENGTH_KEY_NAME,
    SAVE_CHINESE_CAPTION_LENGTH_KEY_NAME,
]

# 本数据集每个编辑对只有编辑前原图这1张参考图(上游reference_image_num
# 全库2025000行恒为1)，所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 上游subtask_name -> 归一化后的任务名(即子集名)，最终产出20个子集。
# 归一化规则就是全小写(上游已经把空格换成下划线了)，
# 其中Multi-object_Coordination与Shape-Size_Alteration带中划线，
# VALID_IMAGE_NAME_PATTERN已允许中划线。
# 显式写死而不是每行现lower()的原因: 归一化规则一改就会静默把几十万个样本对
# 写进错误子集，写死之后任何上游取值变化都会在check_load_annotation_count里硬失败。
# 实测20个子集名互不相同、与subtask_name严格一对一，不存在跨组归并的情况
GET_SET_NAME_DICT = {
    'Background_Change': 'background_change',
    'Color_Alteration': 'color_alteration',
    'Compound_Operation_Edits': 'compound_operation_edits',
    'Counting_Change': 'counting_change',
    'Material_Modification': 'material_modification',
    'Motion_Change': 'motion_change',
    'Multi-object_Coordination': 'multi-object_coordination',
    'Object_Extraction': 'object_extraction',
    'Portrait_Editing': 'portrait_editing',
    'Relation_Change': 'relation_change',
    'Shape-Size_Alteration': 'shape-size_alteration',
    'Spatial_Reasoning_Edits': 'spatial_reasoning_edits',
    'Style_Transfer': 'style_transfer',
    'Subject_Addition': 'subject_addition',
    'Subject_Removal': 'subject_removal',
    'Subject_Replacement': 'subject_replacement',
    'Text_Modification': 'text_modification',
    'Texture_Editing': 'texture_editing',
    'Tone_Transformation': 'tone_transformation',
    'Viewpoint_Transformation': 'viewpoint_transformation',
}

# 上游subtask_name的兜底目录名。
# 实测恰好有1行(train-00067-of-00681第7069行)的上游edit_subtask列被误填成
# 一整句中文prompt("修复精灵球的破裂部分，并在其旁边添加一个小型的、漂浮的精灵球。")，
# 上游017把它归到了unknown_subtask兜底目录。
# 按方案确认: 这一行的任务类型不可知，**整对丢弃**(只丢1对，占0.00005%)，
# 不为它单独开一个只有1对的子集、也不并进任何已知子集
SKIP_SUBTASK_NAME_LIST = [
    'unknown_subtask',
]

EXPECTED_SKIP_SUBTASK_COUNT = 1

# 找不到任务类型时才用的兜底子集名。
# 本数据集20个子集的任务类型全部可知、且unknown_subtask那1行已整对丢弃，
# 一条都不会落进mix，所以最终产出的就是上面20个子集。
# 保留这个常量只是为了和002/005/015.0的口径保持一致，
# 并防止上游之后新增subtask取值时被静默漏处理
# (新取值会被check_load_annotation_count的"unknown subtask"硬拦下来)
MIX_SET_NAME = 'mix'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 保存名形如
# unicedit_color_alteration_train-00000-of-00681_00000000_edited.jpg(最长77字符)，
# 20个子集名全是ASCII小写、分片名是train-%05d-of-%05d、行号是8位数字，
# 所以不会出现CJK字符或其它异常字符，也远低于文件系统单文件名255字节的上限
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 上游落盘图像名去掉扩展名与_src/_edit后缀之后的主干，
# 必须是"<分片名>_<8位片内行号>"这个形态。
# 保存图像名的前缀就取自它，一旦它和标注里的parquet_name/row_index对不上，
# 落盘的图和标注就张冠李戴了，所以必须交叉比对
VALID_LOAD_IMAGE_NAME_PREFIX_PATTERN = re.compile(
    r'^train-\d{5}-of-\d{5}_\d{8}$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃。
# 实测抽样300张(150对的参考图+编辑后图)100%是RGB模式的PNG
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 上游标注文件数与总行数，解析阶段硬对账。
# 少一个分片/少一行都说明上游017没跑完或产物被改动过;
# 多出来则说明上游补传了新的parquet分片(见文件头"10M vs 2M"的分析)，
# 这时必须先重跑017解包、再把本脚本的全部EXPECTED_*常量按新实测值更新
EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 2556

EXPECTED_TOTAL_ANNOTATION_COUNT = 2025000

# 上游全量task_name分布(合计2025000)，解析阶段逐项硬对账。
# 本脚本不按它分子集，但它是"上游产物没被改动过"的一道独立证据:
# 4个粗粒度数字全对上、而某个细粒度数字不对，就能立刻定位到是子集映射出了问题
EXPECTED_TASK_ANNOTATION_COUNT_DICT = {
    'Attribute_Editing': 676977,
    'Object_Editing': 518729,
    'Reasoning_Editing': 280932,
    'Scene_Editing': 548362,
}

# 上游全量subtask_name分布(20个合法子类合计2024999 + unknown_subtask 1行 =
# 2025000)，即"过滤之前"每个子集分到的条数，解析阶段逐项硬对账。
# 这能拦住"某个subtask被映射进错误子集"这种总数级对账看不出来的问题
EXPECTED_SUBTASK_ANNOTATION_COUNT_DICT = {
    'Background_Change': 528872,
    'Color_Alteration': 520303,
    'Compound_Operation_Edits': 19698,
    'Counting_Change': 125,
    'Material_Modification': 86647,
    'Motion_Change': 4451,
    'Multi-object_Coordination': 53516,
    'Object_Extraction': 151,
    'Portrait_Editing': 32763,
    'Relation_Change': 793,
    'Shape-Size_Alteration': 19254,
    'Spatial_Reasoning_Edits': 206924,
    'Style_Transfer': 4473,
    'Subject_Addition': 471956,
    'Subject_Removal': 375,
    'Subject_Replacement': 11686,
    'Text_Modification': 1673,
    'Texture_Editing': 46322,
    'Tone_Transformation': 14856,
    'Viewpoint_Transformation': 161,
}

# 文本层各类不合格样本对的实测精确条数，解析阶段逐项硬对账。
# 判定顺序严格按下面process_single_annotation_file里的顺序执行
# (annotation_dir_name_not_match -> skip_subtask -> unknown_subtask ->
#  empty_caption -> null_like_caption -> no_word_char_caption ->
#  too_short_caption -> too_long_caption -> invalid_placeholder_caption ->
#  invalid_annotation_image_shape -> different_aspect_ratio ->
#  missing_image -> invalid_save_image_name)，
# 换顺序会让这些数字互相搬家，所以顺序不能改。
#
# 【双语联合判定口径】empty / null_like / no_word_char / too_short / too_long /
# invalid_placeholder 这六项都是**英文与中文只要有任意一条不合格，
# 就整对丢弃**(而不是只丢那一种语言)。理由: 落盘的json里两条指令都要写，
# 留下一条坏的等于把坏数据混进训练集。
#   too_short 62条 : **全部是中文短指令**(中文min=8、英文min=28，
#                    英文一条都不会被10这个阈值砍掉)
#   too_long 1327条: 英文超长1327条、其中1条中文也同时超长
#                    (中文max=1342、超过512的只有这1条)，两者是包含关系，
#                    所以联合判定的总数仍是1327
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 0,
    'null_like_caption_count': 0,
    'no_word_char_caption_count': 0,
    'too_short_caption_count': 62,
    'too_long_caption_count': 1327,
    'invalid_placeholder_caption_count': 0,
}

# 除了子集丢弃与指令过滤之外剩下的几项丢弃计数的实测值，逐项硬对账。
# 这六项实测都是0(上游017自校验已保证0隔离样本、4050000张图全部落盘)，
# 但必须一并算进过滤链路恒等式，否则上游一旦出现缺图/坏行，恒等式会误报
EXPECTED_OTHER_FILTER_COUNT_DICT = {
    'illegal_line_count': 0,
    'annotation_dir_name_not_match_count': 0,
    'unknown_subtask_count': 0,
    'invalid_annotation_image_shape_count': 0,
    'different_aspect_ratio_count': 0,
    'missing_image_count': 0,
    'invalid_save_image_name_count': 0,
}

# 被"参考图与编辑后图长宽比不同"这条规则丢弃的实测条数。
# 本数据集全库2025000行的input_image_shape与edited_image_shape**完全相同**，
# 所以这个数字实测为0。
# 保留这条链路(而不是因为实测0就删掉)是必须的: 上游哪天换版、两张图不再同尺寸时，
# 这条规则会立刻把形变样本挡在几十小时的解码重编码之前，而不是静默落盘成错位样本对
EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT = 0

# 文本层全部过滤之后、图像解码校验之前的样本对数:
# 2025000 - 1(unknown_subtask) - 62(too_short) - 1327(too_long) == 2023610。
# 全量实测(非抽样)的精确数字，解析阶段硬对账
EXPECTED_VALID_ANNOTATION_COUNT = 2023610

# 20个子集过滤后的实测条数，合计2023610，解析阶段逐个硬对账。
# 掉得最多的是style_transfer(4473 -> 4410，-1.41%)与
# background_change(528872 -> 528229，-0.12%)，都是被超长英文指令砍的
EXPECTED_SET_ANNOTATION_COUNT_DICT = {
    'background_change': 528229,
    'color_alteration': 520098,
    'compound_operation_edits': 19678,
    'counting_change': 125,
    'material_modification': 86631,
    'motion_change': 4450,
    'multi-object_coordination': 53455,
    'object_extraction': 151,
    'portrait_editing': 32763,
    'relation_change': 791,
    'shape-size_alteration': 19249,
    'spatial_reasoning_edits': 206668,
    'style_transfer': 4410,
    'subject_addition': 471900,
    'subject_removal': 374,
    'subject_replacement': 11678,
    'text_modification': 1673,
    'texture_editing': 46308,
    'tone_transformation': 14818,
    'viewpoint_transformation': 161,
}

# 最终产出的子集数，必须与GET_SET_NAME_DICT严格一一对应(不多也不少)。
# 20个子集里有5个不足10000对(counting_change 125 / object_extraction 151 /
# viewpoint_transformation 161 / subject_removal 374 / relation_change 791)，
# 这5个子集各只会切出1个不满10000对的文件夹，这是允许的
# (每个子集的最后一个文件夹允许不满)。
# 按ceil(条数/10000)逐子集累加，实测合计215个文件夹
EXPECTED_SAVE_SET_COUNT = 20

# 上游实测的重复md5 key数(5181个key各出现2次)。
# 落盘名不用md5 key，所以重复key不会覆盖丢样本，这里只写进校验报告备查，
# 不参与任何过滤与硬对账
EXPECTED_DUPLICATE_MD5_KEY_COUNT = 5181

PROCESS_NUM = 32

PER_FOLDER_EDIT_PAIR_NUM = 10000

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# ==============================================================================
# 【参考图与编辑后图的尺寸对齐规格(全部ti2i数据集统一口径)】
# 编辑后图**原分辨率落盘、不做任何缩放**，它是这个样本对唯一的尺寸基准
# (json里的width/height就是它)。参考图按下面的规则对齐:
#
#   reference_image[0](编辑前原图):
#       长宽比与编辑后图严格相同 -> LANCZOS resize到编辑后图尺寸(等比缩放、零形变)
#       长宽比不同               -> **整个样本对丢弃**(resize会把画面拉伸变形)
#
#   reference_image[k>=1](第二张视觉条件图/主体图/物体图):
#       长宽比与编辑后图严格相同 -> resize到编辑后图尺寸
#       长宽比不同               -> 按**长边对齐**等比resize(不裁剪、不形变、不丢弃)
#
# 长宽比判定用Fraction最简分数比、**不留任何容差**。
#
# 【本数据集的长宽比过滤按015.0.resave_inter_edit_ti2i_dataset.py的口径做两层】
# 第一层(文本解析阶段，主力): 只读标注里的input_image_shape / edited_image_shape
# 就能判、**完全不解图**，所以能把形变样本挡在几十小时的解码重编码之前;
# 第二层(图像校验阶段，兜底): 拿真解码出来的shape再判一次，
# 防止上游标注里记的宽高与磁盘上的实际图像不一致。
# 两层各自独立计数(different_aspect_ratio_count / image_different_aspect_ratio_count)，
# 本数据集两个数字实测都是0(全库2025000行两张图分辨率完全相同)。
#
# 【为什么第一张参考图必须与编辑后图尺寸严格一致】
# 下游TorchAspectRatioBucketResize会按"这张参考图解码后的宽高比是否等于编辑后图
# 解码后的宽高比"逐张走两条分支: 相等走"与GT完全相同的各向异性resize到bucket
# 分辨率"(h/w RoPE逐像素对齐，结构/ID保持类编辑的关键)，不相等则退化成"保住自身
# 宽高比、长边对齐bucket长边"的独立主体图分支。
# 也就是说尺寸不一致的编辑样本会被静默降级成主体参考样本来训，
# 所以这个不变式必须在resave阶段就硬保证。
# ==============================================================================

# 参考图resize到目标尺寸时用的重采样方式。
# 按方案确认用PIL的LANCZOS而不是cv2.resize: 与项目既定口径保持一致
SAVE_IMAGE_RESIZE_RESAMPLING = Image.Resampling.LANCZOS

# 豁免尺寸对齐的子集: 这些子集的编辑后图与全部参考图**完全原样落盘**，
# 不判长宽比、不resize、不丢弃、也不做长边对齐。
# 本数据集20个子集全部都是"参考图与编辑后图像素对齐"的局部编辑任务，
# 没有任何子集需要豁免，所以这里是空列表(保留这个常量只为与004/005口径一致)
EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST = []

# 收尾自校验时是否真解一次reference_image[0]、硬校验它的shape等于json里的
# width/height。按方案置True: "第一张参考图与编辑后图尺寸必须一致"是核心不变式，
# 而只对账json里的数字是查不出resize有没有真的生效的，必须真解一次图。
# 代价是收尾自校验要多解约202万张参考图，在NAS上会明显变慢
CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG = True

# 指令长度阈值(英文与中文用同一套阈值，任一条越界即整对丢弃)。
# 下限10: 与003/005/014/016等英文ti2i脚本一致。
#   英文实测min=28，一条都不会被砍;
#   中文实测min=8，被砍掉62条(形如极短的中文指令)，只占0.003%，按方案确认一并砍掉。
#   这里没有像004/015.0那样为中文单独放宽到4: 那两个数据集的中文指令p50只有7~12、
#   放宽是为了不砍掉几十万条正常样本; 本数据集中文p50=56、p99=114，
#   分布完全不同，用10这个统一阈值只会砍到真正的异常值
MIN_CAPTION_LENGTH = 10

# 上限512: 与003/004/014/016等ti2i脚本一致。
#   英文实测p50=213 / p99=420 / max=4691，超过512的1327条(0.066%);
#   中文实测p50=56 / p99=114 / max=1342，超过512的只有1条(且这条的英文也超长)。
#   英文超长的那批是模型把整段场景描述写进了指令，语义冗长、不适合做编辑监督
MAX_CAPTION_LENGTH = 512

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 中英两条都要各自命中，实测0条被判掉，只作防御性拦截
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)，与004/005/015.0口径一致。
# 中英两条只要任意一条命中就整对丢弃。实测本数据集0条命中，只作防御性拦截
NULL_LIKE_CAPTION_LIST = [
    'null',
    'none',
    'nan',
    'n/a',
    'na',
    'nil',
    'undefined',
    'unknown',
    'empty',
    'blank',
    'no change',
    'no changes',
    'nochange',
    '无',
    '空',
    '无指令',
    '无变化',
    '不变',
    '没有',
    '无需修改',
    '未修改',
]

# 带编号的视觉参考图占位符，编号从"非原图的第1张参考图"起算:
# [V1*]指代reference_image[1]、[V2*]指代reference_image[2]...[VN*]指代
# reference_image[N]，其中N == reference_image_num - 1。
# 编辑前原图reference_image[0]永远隐式、不写进指令、不占编号。
# 这套写法与002/004/005/015.0完全一致，保证跨数据集口径统一。
# 本数据集恒为1张参考图(N == 0)，所以中英两条指令里都不允许出现任何占位符
# (实测2025000行里含[Vn*]形态的0条)，这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2

# jpg重编码质量与色度采样方式，与005.resave_foundir_ti2i_dataset.py、
# 016.resave_crispedit_2m_ti2i_dataset.py口径一致(质量97 + 色度4:4:4)，
# 而不是cv2.imencode的默认值(质量95 + 色度4:2:0)。
#
# 【本数据集必须用这一档的理由】
# 1) 源图**100%是PNG无损**(全库4050000张，input_image_suffix与
#    edited_image_suffix恒为.png)，不存在"源图本来就是jpg、重编码有量化表幂等性"
#    这种可以省质量的情况: 这里每一次重编码都是**从无损到有损的第一次损失**，
#    是编码器损失的真实度量，质量档位越低损失越直接。
# 2) 真正的瓶颈是色度下采样而不是质量值: OpenCV默认的4:2:0会把色度分辨率直接砍半，
#    而本数据集有color_alteration(52.0万对)、tone_transformation(1.5万对)、
#    texture_editing(4.6万对)这些**直接以颜色/色调/纹理为编辑目标**的子集，
#    合计约57万对(28%)。GT自己带色度模糊，等于要求模型学一个"改颜色但颜色是糊的"
#    的自相矛盾目标，监督信号会被直接污染。
# 3) 分辨率普遍偏大(1024x1024占57.6%、最小短边672)，色度下采样的损失在大图上
#    更容易被下游的bucket resize放大。
# 4) 参考图与编辑后图用完全相同的编码参数，避免两条编码链路引入
#    "参考图多一层压缩"这种参考图/GT不对称的伪偏差。
SAVE_IMAGE_JPEG_QUALITY = 97

SAVE_IMAGE_JPEG_SAMPLING_FACTOR = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444

# 落盘时统一使用的jpg编码参数，编辑后图和所有参考图都走这一套，
# 保证编码链路完全一致
SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST = [
    int(cv2.IMWRITE_JPEG_QUALITY),
    int(SAVE_IMAGE_JPEG_QUALITY),
    int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
    int(SAVE_IMAGE_JPEG_SAMPLING_FACTOR),
]


def get_set_name(per_subtask_name):
    """把上游subtask_name映射成子集名(即图像编辑任务类型)

    本数据集20个细粒度任务类型全部可知，优先用写死的映射表
    (映射规则一改就会静默把几十万个样本对写进错误子集);
    表里没有的取值退回到mix兜底子集，这条路径只在上游新增subtask时才会走到，
    且会在check_load_annotation_count里被"unknown subtask"硬拦下来。
    """
    per_subtask_name = str(per_subtask_name).strip()

    if per_subtask_name in GET_SET_NAME_DICT:
        return GET_SET_NAME_DICT[per_subtask_name]

    return MIX_SET_NAME


def check_skip_subtask(per_subtask_name):
    """判定这个subtask是不是按方案整体丢弃的，返回True表示丢弃

    见SKIP_SUBTASK_NAME_LIST的注释: unknown_subtask那1行的任务类型不可知
    (上游edit_subtask列被误填成一整句中文prompt)，整对丢弃。
    这个判定必须排在"未知subtask"判定之前，否则那1行会被算成未知取值、
    污染unknown_subtask_count的硬对账。
    """
    return str(per_subtask_name).strip() in SKIP_SUBTASK_NAME_LIST


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    这个数据集每个编辑对只有编辑前原图这一张参考图(上游reference_image_num
    全库恒为1)，所有子集恒为1。
    保留这个函数是为了和004/005的收尾自校验口径保持一致。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令

    这个数据集恒为1张参考图、不引入任何额外的视觉条件图，
    指令里既没有占位符(实测0条)也没有"the reference image"这类自然语言指代
    (只有1张参考图、指令从不指代它)，所以这里只做strip，不做任何占位符改写。
    中英两条指令都走这同一个函数，保证两条的归一化口径完全一致。
    已经带编号的占位符原样保留，不做任何改动(便于后续多参考图数据集复用本函数)。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    return per_ti2i_caption


def check_invalid_caption(per_ti2i_caption, per_reference_image_num):
    """判定占位符编号与参考图数量不自洽的坏指令，返回True表示这条指令不合格

    参考图里第0张永远是编辑前原图(隐式、不占编号)，所以一条指令应该带的占位符编号
    正好是1...N，其中N = reference_image_num - 1。这里做三条校验:
    1. 同一个编号最多重复2次，超过就是逐字符插占位符的坏指令;
    2. 最大编号必须正好等于N，多了就是指代了不存在的参考图;
    3. 1...N每个编号都必须至少出现一次，不允许跳号，也不允许有图没被指代。
    本数据集N恒为0，即要求指令里完全没有占位符。
    另外还禁止无编号与带编号混用，只做防御性拦截。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    per_placeholder_index_list = CAPTION_VISUAL_PLACEHOLDER_PATTERN.findall(
        per_ti2i_caption)

    # 归一化之后不允许再出现无编号的[V*]
    if '' in per_placeholder_index_list:
        return True

    per_placeholder_index_count_dict = {}
    for per_placeholder_index in per_placeholder_index_list:
        per_placeholder_index = int(per_placeholder_index)
        per_placeholder_index_count_dict[
            per_placeholder_index] = per_placeholder_index_count_dict.get(
                per_placeholder_index, 0) + 1

    # 校验1: 同一个编号最多重复MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM次
    for per_placeholder_index, per_placeholder_count in per_placeholder_index_count_dict.items(
    ):
        if per_placeholder_count > MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM:
            return True

    per_expect_placeholder_num = per_reference_image_num - 1
    if per_expect_placeholder_num < 0:
        per_expect_placeholder_num = 0

    # 校验2: 最大编号必须正好等于N(N为0时不允许有任何占位符)
    per_max_placeholder_index = max(per_placeholder_index_count_dict.keys(
    )) if len(per_placeholder_index_count_dict) > 0 else 0
    if per_max_placeholder_index != per_expect_placeholder_num:
        return True

    # 校验3: 1...N每个编号都必须至少出现一次，不允许跳号
    for per_placeholder_index in range(1, per_expect_placeholder_num + 1):
        if per_placeholder_index not in per_placeholder_index_count_dict:
            return True

    return False


def check_null_like_caption(per_ti2i_caption):
    """判定指令是不是null字面量或"不做任何修改"这类无意义指令

    先小写化再剥掉两端标点空白，然后与黑名单做全串精确匹配，
    这样"None"/"None."/"无"/"无。"能被同一条规则一次判掉。
    实测本数据集0条命中(中英两条都是)，只作防御性拦截。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().strip(
        CAPTION_STRIP_CHAR)

    return per_ti2i_caption in NULL_LIKE_CAPTION_LIST


def get_annotation_text_value(per_annotation, per_key_name):
    """从标注里取一个文本字段并strip，取不到或类型不对时返回空串

    上游文本列实测全库无null、无空串、无前后空白，这里的类型兼容只做防御性拦截。
    """
    per_text_value = per_annotation.get(per_key_name, '')
    if isinstance(per_text_value, (list, tuple)):
        per_text_value = per_text_value[0] if len(per_text_value) > 0 else ''
    if not isinstance(per_text_value, str):
        per_text_value = ''

    return per_text_value.strip()


def get_annotation_image_shape(per_image_shape):
    """把标注里的宽高字段规整成[宽, 高]，不合法时返回None

    上游input_image_shape / edited_image_shape实测2025000行全是2元正int list，
    这里的类型校验只做防御性拦截。
    注意bool是int的子类，必须显式排掉，否则True会被当成宽度1。
    """
    if not isinstance(per_image_shape,
                      (list, tuple)) or len(per_image_shape) != 2:
        return None

    per_image_w, per_image_h = per_image_shape[0], per_image_shape[1]
    if not isinstance(per_image_w, int) or isinstance(per_image_w, bool):
        return None
    if not isinstance(per_image_h, int) or isinstance(per_image_h, bool):
        return None
    if per_image_w <= 0 or per_image_h <= 0:
        return None

    return [per_image_w, per_image_h]


def check_same_image_aspect_ratio(per_reference_image_shape,
                                  per_edited_image_shape):
    """判定参考图与编辑后图的长宽比是否严格相同，返回True表示相同(可以等比resize)

    两个入参都是[宽, 高]。
    用Fraction的最简分数比精确判定、**不留任何容差**，不用浮点相除:
    浮点比较要么因为精度误差把本该相同的判成不同(如1056/1584与832/1248)，
    要么需要引入一个人为的容差阈值，而容差一旦放开就会让"几乎一样但不严格相等"的
    样本被各向异性拉伸落盘。最简分数比是精确的、可复现的。
    实测本数据集2025000行两张图的分辨率完全相同，所以这里恒为True。
    """
    if not per_reference_image_shape or not per_edited_image_shape:
        return False

    return Fraction(per_reference_image_shape[0],
                    per_reference_image_shape[1]) == Fraction(
                        per_edited_image_shape[0], per_edited_image_shape[1])


def get_long_side_aligned_shape(per_reference_image_shape,
                                per_edited_image_shape):
    """按长边与编辑后图长边对齐，算出参考图应该被resize到的[宽, 高]

    只有reference_image[k>=1](第二张视觉条件图/主体图/物体图)在长宽比与编辑后图
    不同时才会走到这里: 这类参考图是独立主体/材质样例，本来就不要求与编辑后图
    像素对齐，硬resize到编辑后图尺寸会把画面拉伸变形，所以改成保持它自己的长宽比、
    只把长边缩放到与编辑后图长边相同(等比缩放、零形变、不裁剪)。
    本数据集恒为1张参考图，这个函数不会被走到，只为与004/005口径一致而保留。
    """
    per_reference_image_w, per_reference_image_h = per_reference_image_shape
    per_edited_image_w, per_edited_image_h = per_edited_image_shape

    per_scale = max(per_edited_image_w, per_edited_image_h) / max(
        per_reference_image_w, per_reference_image_h)

    per_save_image_w = max(1, int(round(per_reference_image_w * per_scale)))
    per_save_image_h = max(1, int(round(per_reference_image_h * per_scale)))

    return [
        per_save_image_w,
        per_save_image_h,
    ]


def get_save_reference_image_shape(per_reference_image_index,
                                   per_reference_image_shape,
                                   per_edited_image_shape):
    """算出这张参考图应该被resize到的[宽, 高]，返回None表示整个样本对必须丢弃

    统一口径(见文件头的尺寸对齐规格):
      长宽比与编辑后图严格相同 -> 一律resize到编辑后图尺寸(等比缩放、零形变);
      长宽比不同且是reference_image[0](编辑前原图) -> 返回None，整对丢弃;
      长宽比不同且是reference_image[k>=1]          -> 按长边对齐resize。
    """
    if check_same_image_aspect_ratio(per_reference_image_shape,
                                     per_edited_image_shape):
        return list(per_edited_image_shape)

    # 第一张参考图是编辑前原图，它必须与编辑后图像素对齐(这是编辑类样本的根本要求)，
    # 长宽比不同就无法在不形变的前提下对齐，整对丢弃
    if per_reference_image_index == 0:
        return None

    return get_long_side_aligned_shape(per_reference_image_shape,
                                       per_edited_image_shape)


def get_load_image_name_prefix(per_image_relative_path,
                               per_load_image_name_suffix):
    """从上游图像相对路径取出样本主干(去掉扩展名与_src/_edit后缀)

    上游图像名形如train-00000-of-00681_00000000_src.png，
    扩展名实测恒为.png，但这里仍然先splitext再去后缀，
    上游哪天换成JPEG也不会取错。
    后缀对不上时返回空串，由调用方计入invalid_save_image_name。
    """
    per_image_name_prefix = os.path.splitext(
        os.path.basename(per_image_relative_path))[0].strip().lower()

    if not per_image_name_prefix.endswith(per_load_image_name_suffix):
        return ''

    return per_image_name_prefix[:-len(per_load_image_name_suffix)]


def process_single_annotation_file(annotation_file_pair):
    """解析单个标注文件，组装图像编辑对(参考图+编辑后图+中英双语编辑指令)的列表

    这一步只做纯文本层面 + 标注里已有宽高的过滤，判定顺序**严格固定**为:
      标注目录名与行内任务名不一致 -> 整体丢弃的subtask -> 未知subtask ->
      指令为空 -> null字面量 -> 无文字字符 -> 指令过短 -> 指令过长 ->
      坏占位符指令 -> 标注宽高非法 -> **参考图与编辑后图长宽比不同** ->
      缺图 -> 保存名非法
    换顺序会让EXPECTED_INVALID_CAPTION_COUNT_DICT、
    EXPECTED_OTHER_FILTER_COUNT_DICT与EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT里
    那些数字互相搬家，所以顺序不能改。

    指令的六项过滤(空/null/无文字/过短/过长/坏占位符)都是**中英联合判定**:
    英文与中文只要有任意一条不合格就整对丢弃，因为落盘的json里两条指令都要写，
    留下一条坏的等于把坏数据混进训练集。

    长宽比这条只读标注里的input_image_shape / edited_image_shape就能判、
    **完全不解图**(按015.0的口径)，所以能把形变样本挡在几十小时的解码重编码之前。
    图像本身的解码校验和分辨率过滤留到后面多进程里做。

    图像是否存在这里用os.path.isfile逐个判，没有按目录缓存os.listdir:
    上游图像按<task>/<subtask>/<parquet名>/分到2556个目录里、每个目录约15000×2个
    文件，缓存一个目录的文件名集合就要几MB，32个worker叠起来反而更亏;
    而且每张图后面都要真解码一遍，真缺图在解码阶段一定会被判出来，
    这里的存在性判定只是为了把"缺图"和"图坏"分开统计。
    """

    per_annotation_path, per_dir_task_name, per_dir_subtask_name, root_image_path = annotation_file_pair

    annotation_list = []
    illegal_line_count = 0
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_jsonl_file:
            for per_line in load_jsonl_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                try:
                    annotation_list.append(json.loads(per_line))
                except Exception as e:
                    illegal_line_count += 1
                    print('2222', per_annotation_path, e)
                    continue
    except Exception as e:
        print('2222', per_annotation_path, e)

    total_annotation_count = len(annotation_list) + illegal_line_count
    annotation_dir_name_not_match_count = 0
    skip_subtask_count, unknown_subtask_count = 0, 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_annotation_image_shape_count = 0
    different_aspect_ratio_count = 0
    missing_image_count = 0
    invalid_save_image_name_count = 0
    task_annotation_count_dict = {}
    subtask_annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    for per_annotation in annotation_list:
        if not isinstance(per_annotation, dict):
            illegal_line_count += 1
            print('2222', per_annotation_path)
            continue

        per_task_name = get_annotation_text_value(
            per_annotation, ANNOTATION_TASK_NAME_KEY_NAME)
        per_subtask_name = get_annotation_text_value(
            per_annotation, ANNOTATION_SUBTASK_NAME_KEY_NAME)

        # 按上游原值统计"过滤之前"的分布，与实测ground truth硬对账。
        # 这里统计的是每一行的原值(含unknown_subtask那1行)，
        # 所以必须在任何过滤之前累加
        task_annotation_count_dict[
            per_task_name] = task_annotation_count_dict.get(per_task_name,
                                                            0) + 1
        subtask_annotation_count_dict[
            per_subtask_name] = subtask_annotation_count_dict.get(
                per_subtask_name, 0) + 1

        # 行内任务名必须和这个标注文件所在的两级目录一致:
        # 不一致说明上游产物被搬动过，继续跑会把样本对写进错误子集(实测0条)
        if per_task_name != per_dir_task_name or per_subtask_name != per_dir_subtask_name:
            annotation_dir_name_not_match_count += 1
            print('3333', per_annotation_path, per_task_name, per_subtask_name)
            continue

        # 任务类型不可知的那1行整对丢弃(见SKIP_SUBTASK_NAME_LIST)。
        # 必须排在未知subtask判定之前，否则它会被算成未知取值
        if check_skip_subtask(per_subtask_name):
            skip_subtask_count += 1
            continue

        # 上游新增了subtask取值时会走到这里，
        # 会被check_load_annotation_count的"unknown subtask"硬拦下来(实测0条)
        if per_subtask_name not in GET_SET_NAME_DICT:
            unknown_subtask_count += 1
            print('3333', per_annotation_path, per_subtask_name[:50])
            continue

        per_set_name = get_set_name(per_subtask_name)
        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)

        per_english_ti2i_caption = get_annotation_text_value(
            per_annotation, ANNOTATION_ENGLISH_CAPTION_KEY_NAME)
        per_chinese_ti2i_caption = get_annotation_text_value(
            per_annotation, ANNOTATION_CHINESE_CAPTION_KEY_NAME)

        # 【以下六项指令过滤全部是中英联合判定: 任一条不合格就整对丢弃】
        # 空指令、全空格指令视为不合格图像编辑对(实测0条，只作防御)
        if not per_english_ti2i_caption or not per_chinese_ti2i_caption:
            empty_caption_count += 1
            continue

        # null字面量与"不做任何修改"这类无意义指令同样丢弃(实测0条，只作防御)
        if check_null_like_caption(
                per_english_ti2i_caption) or check_null_like_caption(
                    per_chinese_ti2i_caption):
            null_like_caption_count += 1
            print('3333', per_annotation_path, per_english_ti2i_caption[:50])
            continue

        # 只剩标点、没有任何数字/字母/汉字的指令也丢弃(实测0条，只作防御)
        if not CAPTION_WORD_CHAR_PATTERN.search(
                per_english_ti2i_caption
        ) or not CAPTION_WORD_CHAR_PATTERN.search(per_chinese_ti2i_caption):
            no_word_char_caption_count += 1
            print('3333', per_annotation_path, per_english_ti2i_caption[:50])
            continue

        # 过短指令视为不合格图像编辑对(实测62条，全部是中文短指令、英文0条)
        if len(per_english_ti2i_caption) < MIN_CAPTION_LENGTH or len(
                per_chinese_ti2i_caption) < MIN_CAPTION_LENGTH:
            too_short_caption_count += 1
            continue

        # 过长指令同样视为不合格图像编辑对
        # (实测1327条: 英文1327条、其中1条中文也同时超长，两者是包含关系)
        if len(per_english_ti2i_caption) > MAX_CAPTION_LENGTH or len(
                per_chinese_ti2i_caption) > MAX_CAPTION_LENGTH:
            too_long_caption_count += 1
            continue

        # 本数据集的指令不需要任何占位符改写，这里只做strip，
        # 写进json的一定是归一化后的指令。
        # 归一化放在长度过滤之后、占位符校验之前: 归一化只做strip，
        # 而上面取值时已经strip过，所以这里不会改变长度、两处口径完全一致
        per_english_ti2i_caption = get_normalized_ti2i_caption(
            per_english_ti2i_caption)
        per_chinese_ti2i_caption = get_normalized_ti2i_caption(
            per_chinese_ti2i_caption)

        # 占位符编号与参考图数量不自洽的指令也丢弃。
        # 本数据集恒1张参考图，即中英两条指令里都不允许出现任何占位符(实测0条)
        if check_invalid_caption(
                per_english_ti2i_caption,
                per_expect_reference_image_num) or check_invalid_caption(
                    per_chinese_ti2i_caption, per_expect_reference_image_num):
            invalid_placeholder_caption_count += 1
            print('3333', per_annotation_path, per_english_ti2i_caption[:100])
            continue

        # 【参考图与编辑后图必须能对齐到同一尺寸(第一层，只读标注不解图)】
        # 长宽比相同的等比resize到编辑后图尺寸(零形变、保留)，
        # 长宽比不同的resize会把画面拉伸变形，整对丢弃。
        # 本数据集实测2025000行两张图分辨率完全相同，这里恒为0，
        # 但这条链路必须在: 上游换版时它能把形变样本挡在解码重编码之前
        per_annotation_reference_image_shape = get_annotation_image_shape(
            per_annotation.get(ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME,
                               None))
        per_annotation_edited_image_shape = get_annotation_image_shape(
            per_annotation.get(ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME, None))

        # 两个宽高字段实测2025000行全非空全合法，取不到只可能是上游规格变了
        if per_annotation_reference_image_shape is None or per_annotation_edited_image_shape is None:
            invalid_annotation_image_shape_count += 1
            print(
                '3333', per_annotation_path,
                per_annotation.get(ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME,
                                   None),
                per_annotation.get(ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME,
                                   None))
            continue

        if not check_same_image_aspect_ratio(
                per_annotation_reference_image_shape,
                per_annotation_edited_image_shape):
            different_aspect_ratio_count += 1
            continue

        per_edited_image_relative_path = per_annotation.get(
            ANNOTATION_EDITED_IMAGE_KEY_NAME, '')
        if not isinstance(per_edited_image_relative_path, str):
            per_edited_image_relative_path = ''
        per_edited_image_relative_path = per_edited_image_relative_path.replace(
            '\\', '/').strip().lstrip('/')
        if not per_edited_image_relative_path:
            missing_image_count += 1
            continue

        per_edited_image_path = os.path.join(root_image_path,
                                             per_edited_image_relative_path)
        if not os.path.isfile(per_edited_image_path):
            missing_image_count += 1
            continue

        # 参考图顺序固定，第0张一定是编辑前原图(这个数据集也只有这一张)。
        # 上游reference_image_path_list恒为长度1的list，
        # 这里仍然按list遍历，保证与004/005的多参考图写法完全同构
        per_annotation_reference_image_relative_path_list = per_annotation.get(
            ANNOTATION_REFERENCE_IMAGE_KEY_NAME, [])
        if not isinstance(per_annotation_reference_image_relative_path_list,
                          (list, tuple)):
            per_annotation_reference_image_relative_path_list = []

        per_reference_image_path_list = []
        per_reference_image_relative_path_list = []
        per_missing_reference_image_count = 0
        for per_reference_image_relative_path in per_annotation_reference_image_relative_path_list:
            if not isinstance(per_reference_image_relative_path, str):
                per_reference_image_relative_path = ''
            per_reference_image_relative_path = per_reference_image_relative_path.replace(
                '\\', '/').strip().lstrip('/')

            if not per_reference_image_relative_path:
                per_missing_reference_image_count += 1
                continue

            per_reference_image_path = os.path.join(
                root_image_path, per_reference_image_relative_path)
            if not os.path.isfile(per_reference_image_path):
                per_missing_reference_image_count += 1
                continue

            per_reference_image_path_list.append(per_reference_image_path)
            per_reference_image_relative_path_list.append(
                per_reference_image_relative_path)

        # 参考图缺任意一张都会让这个编辑对的条件信息不完整，整对丢弃
        if per_missing_reference_image_count > 0 or len(
                per_reference_image_path_list
        ) != per_expect_reference_image_num:
            missing_image_count += 1
            continue

        # 【保存图像名的前缀必须与真正被读取的那张图严格对应】
        # 上游图像名主干形如train-00000-of-00681_00000000，
        # 它由<分片名>_<8位片内行号>组成，全库2025000个组合**0重名**(已全量验证)，
        # 所以拼上数据集名与子集名后100%唯一，不需要再塞别的字段。
        # 这里从编辑后图像的文件名现取主干(而不是直接用parquet_name/row_index字段)，
        # 再和标注里的parquet_name/row_index/sample_key三方交叉比对，
        # 任何一处对不上都说明上游成员错位，整对丢弃
        per_edited_image_name_prefix = get_load_image_name_prefix(
            per_edited_image_relative_path, LOAD_EDITED_IMAGE_NAME_SUFFIX)
        per_reference_image_name_prefix = get_load_image_name_prefix(
            per_reference_image_relative_path_list[0],
            LOAD_REFERENCE_IMAGE_NAME_SUFFIX)

        per_parquet_name = get_annotation_text_value(
            per_annotation, ANNOTATION_PARQUET_NAME_KEY_NAME)
        per_row_index = per_annotation.get(ANNOTATION_ROW_INDEX_KEY_NAME, None)
        if not isinstance(per_row_index, int) or isinstance(
                per_row_index, bool) or per_row_index < 0:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_row_index)
            continue

        per_expect_image_name_prefix = f'{per_parquet_name}_{per_row_index:08d}'.lower(
        )
        per_sample_key = get_annotation_text_value(
            per_annotation, ANNOTATION_SAMPLE_KEY_KEY_NAME)
        per_expect_sample_key = f'{per_task_name}/{per_subtask_name}/{per_parquet_name}_{per_row_index:08d}'

        if not VALID_LOAD_IMAGE_NAME_PREFIX_PATTERN.match(
                per_edited_image_name_prefix
        ) or per_edited_image_name_prefix != per_expect_image_name_prefix or per_reference_image_name_prefix != per_expect_image_name_prefix or per_sample_key != per_expect_sample_key:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_edited_image_name_prefix,
                  per_reference_image_name_prefix, per_sample_key)
            continue

        # 保存图像名统一全小写，形如
        # unicedit_color_alteration_train-00000-of-00681_00000000_edited.jpg
        per_save_image_name_prefix = (f'{DATASET_NAME}_{per_set_name}_'
                                      f'{per_edited_image_name_prefix}')
        per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
        per_save_reference_image_name_list = [
            f'{per_save_image_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}'
        ]
        # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的前缀
        # (即带_edited那一段)，和004/005的写法保持一致，
        # 收尾自校验也是按edited_image去掉.jpg来反推这个文件夹名的
        per_save_pair_folder_name = os.path.splitext(
            per_save_edited_image_name)[0]

        # 保存名里出现路径分隔符或其它异常字符会写坏目录结构，整对丢弃(实测0条)
        if not VALID_IMAGE_NAME_PATTERN.match(per_save_edited_image_name):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_save_edited_image_name)
            continue

        per_invalid_save_reference_image_name_count = 0
        for per_save_reference_image_name in per_save_reference_image_name_list:
            if not VALID_IMAGE_NAME_PATTERN.match(
                    per_save_reference_image_name):
                per_invalid_save_reference_image_name_count += 1

        if per_invalid_save_reference_image_name_count > 0:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path,
                  per_save_reference_image_name_list)
            continue

        # 同一个样本对里两张参考图撞名会互相覆盖，整对丢弃
        # (本数据集只有1张参考图，这里只做防御性拦截)
        if len(set(per_save_reference_image_name_list)) != len(
                per_save_reference_image_name_list):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path,
                  per_save_reference_image_name_list)
            continue

        set_annotation_count_dict[
            per_set_name] = set_annotation_count_dict.get(per_set_name, 0) + 1

        # 中英双语指令一起带下去，落盘时写成两个字典
        # (见文件头SAVE_TI2I_CAPTION_KEY_NAME的注释)
        per_ti2i_caption_dict = {
            SAVE_ENGLISH_CAPTION_KEY_NAME: per_english_ti2i_caption,
            SAVE_CHINESE_CAPTION_KEY_NAME: per_chinese_ti2i_caption,
        }

        edit_annotation_pair_list.append([
            per_set_name,
            per_save_pair_folder_name,
            per_edited_image_path,
            per_save_edited_image_name,
            per_reference_image_path_list,
            per_save_reference_image_name_list,
            per_ti2i_caption_dict,
            per_expect_reference_image_num,
        ])

    return [
        edit_annotation_pair_list,
        task_annotation_count_dict,
        subtask_annotation_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        annotation_dir_name_not_match_count,
        skip_subtask_count,
        unknown_subtask_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_annotation_image_shape_count,
        different_aspect_ratio_count,
        missing_image_count,
        invalid_save_image_name_count,
    ]


def get_all_annotation_file_pair(root_dataset_path):
    """收集上游全部标注文件，返回[标注文件任务列表, 每个任务族的标注文件数]

    上游标注按<edit_task>/<edit_subtask>两级分目录、组内按parquet名分文件
    (实测4个task × 21个subtask下共2556个jsonl)，
    这里按"标注文件"这一粒度出任务，正好能把多进程铺满。
    两级目录名一并带给worker，用于和行内的task_name/subtask_name交叉校验。
    """
    root_image_path = os.path.join(root_dataset_path,
                                   *LOAD_IMAGE_DIR_NAME_LIST)

    load_annotation_dir_path = os.path.join(root_dataset_path,
                                            LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    task_annotation_file_count_dict = {}
    for per_task_name in sorted(os.listdir(load_annotation_dir_path)):
        per_task_dir_path = os.path.join(load_annotation_dir_path,
                                         per_task_name)
        if not os.path.isdir(per_task_dir_path):
            continue

        per_task_annotation_file_count = 0
        for per_subtask_name in sorted(os.listdir(per_task_dir_path)):
            per_subtask_dir_path = os.path.join(per_task_dir_path,
                                                per_subtask_name)
            if not os.path.isdir(per_subtask_dir_path):
                continue

            per_subtask_annotation_file_name_list = sorted([
                per_annotation_file_name for per_annotation_file_name in
                os.listdir(per_subtask_dir_path)
                if per_annotation_file_name.endswith(
                    LOAD_ANNOTATION_FILE_NAME_SUFFIX)
            ])

            for per_annotation_file_name in per_subtask_annotation_file_name_list:
                annotation_file_pair_list.append([
                    os.path.join(per_subtask_dir_path,
                                 per_annotation_file_name),
                    per_task_name,
                    per_subtask_name,
                    root_image_path,
                ])

            per_task_annotation_file_count += len(
                per_subtask_annotation_file_name_list)

        task_annotation_file_count_dict[
            per_task_name] = per_task_annotation_file_count

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list, task_annotation_file_count_dict


def get_all_edit_annotation_pair(root_dataset_path):
    """按标注文件粒度多进程组装全部图像编辑对的列表

    上游有2556个标注文件、合计2025000行，逐行还要判2张图像文件是否存在，
    所以这里按标注文件开多进程解析，最后按保存的编辑后图像名统一排序。
    """
    annotation_file_pair_list, task_annotation_file_count_dict = get_all_annotation_file_pair(
        root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list),
          'annotation task:', len(task_annotation_file_count_dict))

    total_annotation_count = 0
    illegal_line_count = 0
    annotation_dir_name_not_match_count = 0
    skip_subtask_count, unknown_subtask_count = 0, 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_annotation_image_shape_count = 0
    different_aspect_ratio_count = 0
    missing_image_count = 0
    invalid_save_image_name_count = 0
    task_annotation_count_dict = {}
    subtask_annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            for per_task_name, per_task_count in per_load_result[1].items():
                task_annotation_count_dict[
                    per_task_name] = task_annotation_count_dict.get(
                        per_task_name, 0) + per_task_count
            for per_subtask_name, per_subtask_count in per_load_result[
                    2].items():
                subtask_annotation_count_dict[
                    per_subtask_name] = subtask_annotation_count_dict.get(
                        per_subtask_name, 0) + per_subtask_count
            for per_set_name, per_set_count in per_load_result[3].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

            total_annotation_count += per_load_result[4]
            illegal_line_count += per_load_result[5]
            annotation_dir_name_not_match_count += per_load_result[6]
            skip_subtask_count += per_load_result[7]
            unknown_subtask_count += per_load_result[8]
            empty_caption_count += per_load_result[9]
            null_like_caption_count += per_load_result[10]
            no_word_char_caption_count += per_load_result[11]
            too_short_caption_count += per_load_result[12]
            too_long_caption_count += per_load_result[13]
            invalid_placeholder_caption_count += per_load_result[14]
            invalid_annotation_image_shape_count += per_load_result[15]
            different_aspect_ratio_count += per_load_result[16]
            missing_image_count += per_load_result[17]
            invalid_save_image_name_count += per_load_result[18]

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        task_annotation_file_count_dict,
        task_annotation_count_dict,
        subtask_annotation_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        annotation_dir_name_not_match_count,
        skip_subtask_count,
        unknown_subtask_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_annotation_image_shape_count,
        different_aspect_ratio_count,
        missing_image_count,
        invalid_save_image_name_count,
    ]


def check_load_annotation_count(
        task_annotation_count_dict, subtask_annotation_count_dict,
        set_annotation_count_dict, total_annotation_count,
        total_annotation_file_count, valid_annotation_count,
        skip_subtask_count, different_aspect_ratio_count,
        invalid_caption_count_dict, other_filter_count_dict,
        edit_annotation_pair_list):
    """解析完标注后按task/subtask/子集三级硬对账，并检查保存图像名是否唯一

    上游标注是017一步跑出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账能额外拦住"某个subtask被映射进错误子集"这种task级对账看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对。
    """
    check_error_message_list = []

    # 上游标注文件数: 多出来说明上游补传了新的parquet分片(见文件头"10M vs 2M")，
    # 必须先重跑017解包、再更新本脚本的全部EXPECTED_*常量
    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    # 4个粗粒度任务族逐项硬对账(本脚本不按它分子集，但它是一道独立证据)
    for per_task_name in sorted(task_annotation_count_dict.keys()):
        if per_task_name not in EXPECTED_TASK_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown task {per_task_name}')
            continue

        per_expect_task_annotation_count = EXPECTED_TASK_ANNOTATION_COUNT_DICT[
            per_task_name]
        if task_annotation_count_dict[
                per_task_name] != per_expect_task_annotation_count:
            check_error_message_list.append(
                f'{per_task_name} task annotation count not match '
                f'{task_annotation_count_dict[per_task_name]} != '
                f'{per_expect_task_annotation_count}')

    for per_task_name in sorted(EXPECTED_TASK_ANNOTATION_COUNT_DICT.keys()):
        if per_task_name not in task_annotation_count_dict:
            check_error_message_list.append(f'missing task {per_task_name}')

    # 20个细粒度任务族逐项硬对账(整体丢弃的unknown_subtask单独对账，见下面)
    for per_subtask_name in sorted(subtask_annotation_count_dict.keys()):
        if per_subtask_name in SKIP_SUBTASK_NAME_LIST:
            continue

        if per_subtask_name not in EXPECTED_SUBTASK_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown subtask {per_subtask_name[:50]}')
            continue

        per_expect_subtask_annotation_count = EXPECTED_SUBTASK_ANNOTATION_COUNT_DICT[
            per_subtask_name]
        if subtask_annotation_count_dict[
                per_subtask_name] != per_expect_subtask_annotation_count:
            check_error_message_list.append(
                f'{per_subtask_name} subtask annotation count not match '
                f'{subtask_annotation_count_dict[per_subtask_name]} != '
                f'{per_expect_subtask_annotation_count}')

    for per_subtask_name in sorted(
            EXPECTED_SUBTASK_ANNOTATION_COUNT_DICT.keys()):
        if per_subtask_name not in subtask_annotation_count_dict:
            check_error_message_list.append(
                f'missing subtask {per_subtask_name}')

    # 20个合法subtask原始条数 + 整体丢弃的条数必须刚好等于总行数，一行都不能漏归类
    if sum(EXPECTED_SUBTASK_ANNOTATION_COUNT_DICT.values()
           ) + skip_subtask_count != total_annotation_count:
        check_error_message_list.append(
            f'subtask annotation count not self consistent '
            f'{sum(EXPECTED_SUBTASK_ANNOTATION_COUNT_DICT.values())} + '
            f'{skip_subtask_count} != {total_annotation_count}')

    # 被整体丢弃的unknown_subtask条数硬对账，守住"丢弃范围没被改动过"这条口径
    if skip_subtask_count != EXPECTED_SKIP_SUBTASK_COUNT:
        check_error_message_list.append(
            f'skip subtask count not match '
            f'{skip_subtask_count} != {EXPECTED_SKIP_SUBTASK_COUNT}')

    # 整体丢弃名单与保留映射表不允许有交集
    for per_skip_subtask_name in SKIP_SUBTASK_NAME_LIST:
        if per_skip_subtask_name in GET_SET_NAME_DICT:
            check_error_message_list.append(
                f'skip subtask {per_skip_subtask_name} also in set name dict')

    # 文本层各类不合格指令逐项硬对账
    for per_count_name in sorted(EXPECTED_INVALID_CAPTION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_CAPTION_COUNT_DICT[per_count_name]
        if invalid_caption_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_caption_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    # 其余各项丢弃计数逐项硬对账(实测全为0)
    for per_count_name in sorted(EXPECTED_OTHER_FILTER_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_OTHER_FILTER_COUNT_DICT[per_count_name]
        if other_filter_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{other_filter_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    if different_aspect_ratio_count != EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT:
        check_error_message_list.append(
            f'different aspect ratio count not match '
            f'{different_aspect_ratio_count} != '
            f'{EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT}')

    if valid_annotation_count != EXPECTED_VALID_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'valid annotation count not match '
            f'{valid_annotation_count} != {EXPECTED_VALID_ANNOTATION_COUNT}')

    # 【过滤链路恒等式自校验】总条数减去每一项被丢弃的条数必须正好等于保留条数。
    # 这一条不依赖任何硬编码的期望值，纯粹校验"各项计数之间自洽"，
    # 专门用来拦住"某个EXPECTED_*常量口径被改过/没跟着改"这类问题:
    # 上面那些逐项对账各自都可能因为口径漂移而误报或漏报，
    # 但只要这个恒等式不成立，就一定是计数逻辑或统计口径出了问题
    per_all_filter_count = (skip_subtask_count + different_aspect_ratio_count +
                            sum(invalid_caption_count_dict.values()) +
                            sum(other_filter_count_dict.values()))
    if total_annotation_count - per_all_filter_count != valid_annotation_count:
        check_error_message_list.append(
            f'annotation filter count not self consistent '
            f'{total_annotation_count} - {per_all_filter_count} != '
            f'{valid_annotation_count}')

    # 20个保留子集逐个硬对账
    for per_set_name in sorted(EXPECTED_SET_ANNOTATION_COUNT_DICT.keys()):
        per_expect_set_annotation_count = EXPECTED_SET_ANNOTATION_COUNT_DICT[
            per_set_name]
        if per_set_name not in set_annotation_count_dict:
            check_error_message_list.append(f'missing save set {per_set_name}')
            continue
        if set_annotation_count_dict[
                per_set_name] != per_expect_set_annotation_count:
            check_error_message_list.append(
                f'{per_set_name} set annotation count not match '
                f'{set_annotation_count_dict[per_set_name]} != '
                f'{per_expect_set_annotation_count}')

    # 保留下来的子集集合必须与映射表严格一一对应，不允许多出任何一个子集
    # (mix兜底子集出现就说明上游新增了subtask取值)
    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in EXPECTED_SET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

    # 保存的编辑后图像名必须全局唯一(<parquet名>_<行号>全库0重名时天然满足)，
    # 撞名会让两个样本对在磁盘和json里互相覆盖
    save_edited_image_name_set = set()
    duplicate_save_edited_image_name_list = []
    for per_edit_annotation_pair in edit_annotation_pair_list:
        per_save_edited_image_name = per_edit_annotation_pair[3]
        if per_save_edited_image_name in save_edited_image_name_set:
            duplicate_save_edited_image_name_list.append(
                per_save_edited_image_name)
            continue
        save_edited_image_name_set.add(per_save_edited_image_name)

    if len(duplicate_save_edited_image_name_list) > 0:
        check_error_message_list.append(
            f'duplicate save edited image name num '
            f'{len(duplicate_save_edited_image_name_list)} '
            f'{duplicate_save_edited_image_name_list[:5]}')

    return check_error_message_list


def check_single_image(per_image_path):
    """校验单张图像能否正常解码，并过滤非RGB图和极端分辨率图

    返回图像宽高只用于统计与长宽比判定，
    最终写进json的宽高一定取自实际写盘图像的shape。
    """
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把P图/CMYK图静默转成3通道，
    # 所以必须先用PIL读原始mode才能把灰度图/P图/CMYK图判出来
    try:
        per_image_mode = Image.open(per_image_path).mode
    except Exception as e:
        print('4444', per_image_path, e)
        return None

    if per_image_mode not in VALID_IMAGE_MODE_LIST:
        print('5555', per_image_path, per_image_mode)
        return None

    try:
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
    except Exception as e:
        print('4444', per_image_path, e)
        return None

    if per_image is None or per_image.ndim != 3 or per_image.shape[2] != 3:
        print('4444', per_image_path)
        return None

    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    # 检查图像短边
    if min(per_image_h, per_image_w) < MIN_IMAGE_SHORT_SIDE:
        print('6666', per_image_path, per_image_w, per_image_h)
        return None

    # 检查图像宽高比，取长短边之比，宽高比大于8和小于1/8这两种极端样本一起判掉
    per_image_aspect_ratio = max(per_image_w / per_image_h,
                                 per_image_h / per_image_w)
    if per_image_aspect_ratio > MAX_IMAGE_ASPECT_RATIO:
        print('7777', per_image_path, per_image_w, per_image_h)
        return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_edit_pair_check(edit_annotation_pair):
    """校验单个图像编辑对，并算出每张参考图应该被resize到的目标尺寸

    返回值是[状态字符串, 子集名, 编辑对]:
      'ok'                     -> 第三项是带上每张参考图目标尺寸的编辑对
      'invalid_image'          -> 有图解不开/mode不在白名单/短边或宽高比越界，
                                  第三项为None
      'different_aspect_ratio' -> reference_image[0]与编辑后图长宽比不同(整对丢弃)，
                                  第三项为None。带上子集名是为了在主流程里逐子集统计

    这是长宽比过滤的**第二层**(兜底): 第一层已经在文本解析阶段用标注里的宽高
    把形变样本挡掉了，这里拿真解码出来的shape再判一次，
    防止上游标注记的宽高与磁盘上的实际图像不一致(实测抽样300张100%一致)。

    短边和宽高比的过滤按方案只以编辑后图像为准判定，参考图只要求能正常解码且
    mode命中白名单(实测编辑后图短边最小672、最大宽高比2.33，极端样本预期为0，
    这两个阈值只作兜底)。

    这里之所以能零额外IO地判长宽比: check_single_image本来就已经把编辑后图和
    每张参考图都真解码了一遍并返回了[宽, 高]，接住这些shape，
    既能判长宽比、又能把"每张参考图的目标尺寸"一路带到落盘阶段，
    落盘时不必再重算一次。

    豁免子集(EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST)的目标尺寸一律给None，
    表示编辑后图与全部参考图都完全原样落盘。本数据集没有豁免子集。
    """
    per_set_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_ti2i_caption_dict, per_expect_reference_image_num = edit_annotation_pair

    per_edited_image_shape = check_single_image(per_edited_image_path)
    if per_edited_image_shape is None:
        return ['invalid_image', per_set_name, None]

    per_exempt_aspect_ratio_align_flag = per_set_name in EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST

    per_save_reference_image_shape_list = []
    for per_reference_image_index, per_reference_image_path in enumerate(
            per_reference_image_path_list):
        per_reference_image_shape = check_single_image(
            per_reference_image_path)
        if per_reference_image_shape is None:
            return ['invalid_image', per_set_name, None]

        # 豁免子集完全原样落盘: 不判长宽比、不resize、不丢弃、也不做长边对齐
        if per_exempt_aspect_ratio_align_flag:
            per_save_reference_image_shape_list.append(None)
            continue

        per_save_reference_image_shape = get_save_reference_image_shape(
            per_reference_image_index, per_reference_image_shape,
            per_edited_image_shape)
        # 只有reference_image[0]长宽比不同才会拿到None，此时整对丢弃
        if per_save_reference_image_shape is None:
            return ['different_aspect_ratio', per_set_name, None]

        per_save_reference_image_shape_list.append(
            per_save_reference_image_shape)

    return [
        'ok',
        per_set_name,
        [
            per_set_name,
            per_save_pair_folder_name,
            per_edited_image_path,
            per_save_edited_image_name,
            per_reference_image_path_list,
            per_save_reference_image_name_list,
            per_save_reference_image_shape_list,
            per_ti2i_caption_dict,
            per_expect_reference_image_num,
        ],
    ]


def get_all_edit_pair_save_folder_pair(edit_annotation_pair_list,
                                       save_dataset_path):
    """把过滤后的合格图像编辑对按子集分组，排序后每10000对切成一个文件夹

    切分必须在过滤全部完成之后做，且切分前先按保存的编辑后图像名排序，这样才能保证
    每个文件夹都是满10000对(最后一个文件夹允许不满)。每个图像编辑对在文件夹里再独占
    一个子文件夹，该对的编辑后图像和所有参考图像都存在这个子文件夹里。
    本数据集产出20个子集，其中15个在1万对以上会切出多个满10000对的文件夹、
    另外5个(counting_change / object_extraction / viewpoint_transformation /
    subject_removal / relation_change)各只有1个不满的文件夹，
    实测合计215个文件夹。
    """
    per_set_edit_annotation_pair_dict = {}
    for per_edit_annotation_pair in edit_annotation_pair_list:
        per_set_name = per_edit_annotation_pair[0]
        if per_set_name not in per_set_edit_annotation_pair_dict:
            per_set_edit_annotation_pair_dict[per_set_name] = []
        per_set_edit_annotation_pair_dict[per_set_name].append(
            per_edit_annotation_pair)

    edit_pair_save_folder_pair_list = []
    set_folder_count_dict = {}
    for per_set_name in sorted(per_set_edit_annotation_pair_dict.keys()):
        per_set_edit_annotation_pair_list = sorted(
            per_set_edit_annotation_pair_dict[per_set_name],
            key=lambda x: x[3])

        per_set_folder_count = 0
        for per_folder_start_index in range(
                0, len(per_set_edit_annotation_pair_list),
                PER_FOLDER_EDIT_PAIR_NUM):
            per_folder_edit_annotation_pair_list = per_set_edit_annotation_pair_list[
                per_folder_start_index:per_folder_start_index +
                PER_FOLDER_EDIT_PAIR_NUM]

            per_folder_name = f'{per_set_name}_{per_set_folder_count:05d}'

            for per_edit_annotation_pair in per_folder_edit_annotation_pair_list:
                _, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_save_reference_image_shape_list, per_ti2i_caption_dict, per_expect_reference_image_num = per_edit_annotation_pair

                per_pair_folder_path = os.path.join(save_dataset_path,
                                                    per_set_name,
                                                    per_folder_name,
                                                    per_save_pair_folder_name)
                os.makedirs(per_pair_folder_path, exist_ok=True)

                edit_pair_save_folder_pair_list.append([
                    per_set_name,
                    per_folder_name,
                    per_save_pair_folder_name,
                    per_edited_image_path,
                    per_save_edited_image_name,
                    per_reference_image_path_list,
                    per_save_reference_image_name_list,
                    per_save_reference_image_shape_list,
                    per_ti2i_caption_dict,
                    per_expect_reference_image_num,
                ])

            per_set_folder_count += 1

        set_folder_count_dict[per_set_name] = per_set_folder_count

    return edit_pair_save_folder_pair_list, set_folder_count_dict


def resize_single_image(per_image, per_save_image_shape):
    """把BGR的numpy图像resize到指定的[宽, 高]，返回resize后的BGR numpy图像

    按方案确认用PIL的LANCZOS而不是cv2.resize:
    先BGR->RGB转成PIL、resize、再转回numpy并RGB->BGR，
    中间的颜色通道转换不能省，否则落盘图像的R与B通道会互换。
    """
    per_save_image_w, per_save_image_h = per_save_image_shape

    per_pil_image = Image.fromarray(cv2.cvtColor(per_image, cv2.COLOR_BGR2RGB))
    per_pil_image = per_pil_image.resize((per_save_image_w, per_save_image_h),
                                         SAVE_IMAGE_RESIZE_RESAMPLING)

    return cv2.cvtColor(np.asarray(per_pil_image), cv2.COLOR_RGB2BGR)


def resave_single_image(per_image_path,
                        save_image_path,
                        save_image_shape=None):
    """重新编码保存单张图像，只在显式给了目标尺寸时才resize

    save_image_shape为None(编辑后图与豁免子集的参考图走这条): 图像原分辨率多少
    保存时还是多少，不做任何缩放;
    给了[宽, 高](非豁免子集的参考图走这条): 先LANCZOS resize到这个尺寸再落盘。
    目标尺寸由process_single_edit_pair_check按统一规则算好并一路带下来:
    reference_image[0]一定是编辑后图尺寸(长宽比不同的样本对已经在那一步整对丢弃了)，
    reference_image[k>=1]是编辑后图尺寸或长边对齐后的尺寸。
    本数据集两张图天然同分辨率，所以这里的resize在全部样本上都是恒等操作
    (尺寸已经相等时会走下面那个"只在尺寸真的不一样时才resize"的短路，
     一次多余的重采样都不会做)。

    上游4050000张图全部是PNG无损，这里统一重编码成jpg，只换编码格式不换像素尺寸。
    编码参数显式用SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST(质量97 + 色度4:4:4)，
    而不是cv2的默认值(质量95 + 色度4:2:0): 源图无损，每一次重编码都是从无损到有损的
    第一次损失; 而且本数据集有color_alteration/tone_transformation/texture_editing
    这些直接以颜色/色调/纹理为编辑目标的子集(约57万对、占28%)，
    GT带色度模糊会直接污染监督信号，详见SAVE_IMAGE_JPEG_QUALITY处的注释。
    编辑后图和参考图共用同一套参数，避免两条编码链路引入参考图/GT不对称的伪偏差。
    """
    try:
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
    except Exception as e:
        print('8888', per_image_path, e)
        return None

    if per_image is None or per_image.ndim != 3 or per_image.shape[2] != 3:
        print('8888', per_image_path)
        return None

    # 只有非豁免子集的参考图会带目标尺寸，且只在尺寸真的不一样时才resize
    if save_image_shape is not None:
        per_save_image_w, per_save_image_h = save_image_shape
        if per_save_image_w <= 0 or per_save_image_h <= 0:
            print('8888', per_image_path, save_image_shape)
            return None

        if per_image.shape[1] != per_save_image_w or per_image.shape[
                0] != per_save_image_h:
            try:
                per_image = resize_single_image(per_image, save_image_shape)
            except Exception as e:
                print('8888', per_image_path, save_image_shape, e)
                return None

        # resize之后必须真的等于目标尺寸，不等说明resize没生效
        if per_image.shape[1] != per_save_image_w or per_image.shape[
                0] != per_save_image_h:
            print('8888', per_image_path, per_image.shape, save_image_shape)
            return None

    # 宽高直接取自这个即将被编码写盘的数组的shape，
    # jpg编解码不改变像素尺寸，所以宽高一定和保存图像一致
    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    if not os.path.exists(save_image_path):
        try:
            cv2.imencode(
                '.jpg', per_image,
                SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST)[1].tofile(save_image_path)
        except Exception as e:
            print('8888', save_image_path, e)
            return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_edit_pair(edit_pair_save_folder_pair, save_dataset_path):
    """重新编码保存单个图像编辑对的编辑后图像和所有参考图像，任意一张失败则整对丢弃

    编辑后图**原分辨率落盘、不做任何缩放**，先落盘并拿到它的真实宽高;
    参考图再按check阶段算好的目标尺寸resize后落盘。
    非豁免子集的reference_image[0]的目标尺寸就是编辑后图尺寸，
    所以落盘后还会额外硬对账一次"reference_image[0]的shape == 编辑后图的shape"，
    不等说明resize链路有问题、整对丢弃。
    """
    per_set_name, per_folder_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_save_reference_image_shape_list, per_ti2i_caption_dict, per_expect_reference_image_num = edit_pair_save_folder_pair

    per_pair_folder_path = os.path.join(save_dataset_path, per_set_name,
                                        per_folder_name,
                                        per_save_pair_folder_name)

    save_edited_image_path = os.path.join(per_pair_folder_path,
                                          per_save_edited_image_name)
    per_edited_image_shape = resave_single_image(per_edited_image_path,
                                                 save_edited_image_path)
    if per_edited_image_shape is None:
        return None

    per_edited_image_w, per_edited_image_h = per_edited_image_shape

    for per_reference_image_path, per_save_reference_image_name, per_save_reference_image_shape in zip(
            per_reference_image_path_list, per_save_reference_image_name_list,
            per_save_reference_image_shape_list):
        save_reference_image_path = os.path.join(
            per_pair_folder_path, per_save_reference_image_name)
        # per_save_reference_image_shape为None只出现在豁免子集，表示原样落盘
        per_save_shape = resave_single_image(per_reference_image_path,
                                             save_reference_image_path,
                                             per_save_reference_image_shape)
        if per_save_shape is None:
            return None

        # 落盘后的shape必须与目标尺寸严格相等，不等说明resize链路有问题。
        # 第一张参考图的目标尺寸就是编辑后图尺寸，这一条同时也就是
        # "第一张参考图与编辑后图尺寸必须严格一致"这个核心不变式的落盘期硬校验
        if per_save_reference_image_shape is not None and per_save_shape != list(
                per_save_reference_image_shape):
            print('8888', save_reference_image_path, per_save_shape,
                  per_save_reference_image_shape)
            return None

    return [
        per_folder_name,
        per_save_edited_image_name,
        list(per_save_reference_image_name_list),
        per_edited_image_w,
        per_edited_image_h,
        per_ti2i_caption_dict,
        per_expect_reference_image_num,
    ]


def save_all_folder_annotation_json(save_result_list, save_dataset_path,
                                    set_folder_count_dict):
    """按文件夹汇总标注并写出与文件夹同名的json文件

    每条标注固定只有SAVE_ANNOTATION_KEY_NAME_LIST这七个key，
    其中ti2i_caption与ti2i_caption_length的value都是字典
    (见文件头SAVE_TI2I_CAPTION_KEY_NAME处的说明)。
    上游jsonl里剩下的属性全部丢弃、不另存索引，理由见文件头逐key的核对表:
    instruction与prompt_en逐字相同(纯冗余)、edit_task与edit_subtask是带空格的原值、
    md5_key有5181个重复且不参与落盘命名、
    sample_key/parquet_name/row_index只用于拼保存名与交叉校验(信息已内含在保存名里)、
    reference_image_num恒为1(改由list长度现算)、
    input_image_shape/edited_image_shape只用于长宽比预筛
    (写json的宽高一律取自实际写盘数组的shape)、
    input_image_suffix/edited_image_suffix恒为.png、
    dataset_task_type恒为image_edit、
    task_name是4类粗粒度任务族(已被20个细粒度子集蕴含)。
    """

    folder_annotation_dict = {}
    reference_image_num_mismatch_count = 0
    for per_save_result in save_result_list:
        per_folder_name, per_save_edited_image_name, per_save_reference_image_name_list, per_edited_image_w, per_edited_image_h, per_ti2i_caption_dict, per_expect_reference_image_num = per_save_result
        if per_folder_name not in folder_annotation_dict:
            folder_annotation_dict[per_folder_name] = {}

        # reference_image_num一定由reference_image这个list的长度现算，
        # 保证写进json的数值永远和list长度对得上
        per_reference_image_num = len(per_save_reference_image_name_list)
        # 再和该子集应有的参考图数量(本数据集恒为1)交叉对账，不一致只上报不改数值
        if per_expect_reference_image_num >= 0 and per_reference_image_num != per_expect_reference_image_num:
            reference_image_num_mismatch_count += 1
            print('9999', per_save_edited_image_name, per_reference_image_num,
                  per_expect_reference_image_num)

        per_english_ti2i_caption = per_ti2i_caption_dict[
            SAVE_ENGLISH_CAPTION_KEY_NAME]
        per_chinese_ti2i_caption = per_ti2i_caption_dict[
            SAVE_CHINESE_CAPTION_KEY_NAME]

        # 两个length都直接取即将写进json的那两个字符串的长度，
        # 保证记录的长度和指令永远自洽(两个字符串都已strip并归一化过)
        folder_annotation_dict[per_folder_name][per_save_edited_image_name] = {
            'reference_image': per_save_reference_image_name_list,
            'edited_image': per_save_edited_image_name,
            'reference_image_num': per_reference_image_num,
            'width': per_edited_image_w,
            'height': per_edited_image_h,
            SAVE_TI2I_CAPTION_KEY_NAME: {
                SAVE_ENGLISH_CAPTION_KEY_NAME: per_english_ti2i_caption,
                SAVE_CHINESE_CAPTION_KEY_NAME: per_chinese_ti2i_caption,
            },
            SAVE_TI2I_CAPTION_LENGTH_KEY_NAME: {
                SAVE_ENGLISH_CAPTION_LENGTH_KEY_NAME:
                len(per_english_ti2i_caption),
                SAVE_CHINESE_CAPTION_LENGTH_KEY_NAME:
                len(per_chinese_ti2i_caption),
            },
        }

    folder_edit_pair_count_dict = {}
    for per_set_name in sorted(set_folder_count_dict.keys()):
        for per_folder_index in range(set_folder_count_dict[per_set_name]):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            if per_folder_name not in folder_annotation_dict:
                print('9999', per_folder_name)
                continue

            per_folder_annotation_dict = folder_annotation_dict[
                per_folder_name]
            per_folder_annotation_dict = {
                per_save_edited_image_name:
                per_folder_annotation_dict[per_save_edited_image_name]
                for per_save_edited_image_name in sorted(
                    per_folder_annotation_dict.keys())
            }

            save_json_path = os.path.join(save_dataset_path, per_set_name,
                                          f'{per_folder_name}.json')
            with open(save_json_path, 'w', encoding='UTF-8') as save_json_file:
                json.dump(per_folder_annotation_dict,
                          save_json_file,
                          ensure_ascii=False)

            folder_edit_pair_count_dict[per_folder_name] = len(
                per_folder_annotation_dict)

            print('2222', per_folder_name, len(per_folder_annotation_dict))

    return folder_edit_pair_count_dict, reference_image_num_mismatch_count


def check_single_save_annotation_caption(per_save_edited_image_name,
                                         per_annotation):
    """复检单条标注里的双语指令，返回错误信息列表

    本数据集的ti2i_caption与ti2i_caption_length都是字典，
    所以这里要比单语数据集多校验一层结构:
    1. 两个value都必须是dict，且key集合严格等于约定的两个列表;
    2. 两条指令都必须是字符串、都不允许是null字面量、都必须含有文字字符;
    3. 两条指令的长度都必须在[MIN, MAX]区间内(与过滤阶段的联合判定口径一致);
    4. 两条指令都不允许含[Vn*]占位符(参考图数量恒为1，即N为0);
    5. 两个length都必须等于对应字符串的实际len()。
    """
    check_error_message_list = []

    per_ti2i_caption_dict = per_annotation[SAVE_TI2I_CAPTION_KEY_NAME]
    per_ti2i_caption_length_dict = per_annotation[
        SAVE_TI2I_CAPTION_LENGTH_KEY_NAME]

    if not isinstance(per_ti2i_caption_dict, dict) or not isinstance(
            per_ti2i_caption_length_dict, dict):
        check_error_message_list.append(
            f'{per_save_edited_image_name} ti2i caption not a dict')

        return check_error_message_list

    if sorted(per_ti2i_caption_dict.keys()) != sorted(
            SAVE_TI2I_CAPTION_KEY_NAME_LIST):
        check_error_message_list.append(
            f'{per_save_edited_image_name} ti2i caption key not match')

        return check_error_message_list

    if sorted(per_ti2i_caption_length_dict.keys()) != sorted(
            SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST):
        check_error_message_list.append(
            f'{per_save_edited_image_name} ti2i caption length key not match')

        return check_error_message_list

    # 两条指令走完全相同的一套复检，保证中英口径一致
    for per_caption_key_name, per_caption_length_key_name in zip(
            SAVE_TI2I_CAPTION_KEY_NAME_LIST,
            SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST):
        per_ti2i_caption = per_ti2i_caption_dict[per_caption_key_name]
        per_ti2i_caption_length = per_ti2i_caption_length_dict[
            per_caption_length_key_name]

        if not isinstance(per_ti2i_caption, str):
            check_error_message_list.append(
                f'{per_save_edited_image_name} {per_caption_key_name} not a str'
            )
            continue

        # ti2i_caption的占位符编号集合必须与reference_image这个list的长度自洽:
        # 本数据集恒1张参考图，即不允许出现任何占位符
        if check_invalid_caption(per_ti2i_caption,
                                 len(per_annotation['reference_image'])):
            check_error_message_list.append(
                f'{per_save_edited_image_name} {per_caption_key_name} placeholder index not match reference image num {len(per_annotation["reference_image"])}'
            )
        # 落盘后的指令里不允许再残留null字面量或"不做任何修改"这类无意义指令
        if check_null_like_caption(per_ti2i_caption):
            check_error_message_list.append(
                f'{per_save_edited_image_name} {per_caption_key_name} still a null like caption'
            )
        # 也不允许残留只剩标点、没有任何数字/字母/汉字的指令
        if not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption):
            check_error_message_list.append(
                f'{per_save_edited_image_name} {per_caption_key_name} still a no word char caption'
            )
        # json里存的就是归一化后的指令，长度过滤也是按归一化后判定的，
        # 两者口径一致，这里直接量json里的长度复检
        if len(per_ti2i_caption.strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_edited_image_name} {per_caption_key_name} still an invalid caption'
            )
        if len(per_ti2i_caption.strip()) > MAX_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_edited_image_name} {per_caption_key_name} still a too long caption'
            )
        # 记录的指令长度必须和指令字符串的实际长度对得上
        if per_ti2i_caption_length != len(per_ti2i_caption):
            check_error_message_list.append(
                f'{per_save_edited_image_name} {per_caption_length_key_name} not match'
            )

    return check_error_message_list


def check_save_dataset(save_dataset_path, set_folder_count_dict):
    """全部落盘后的收尾自校验: 文件夹容量、json与磁盘一一对应、双语指令与参考图数量

    每个子集除最后一个文件夹外都必须是满10000对，json里的每个key都必须在磁盘上有
    对应的样本对文件夹且文件恰好等于编辑后图像 + 所有参考图像，磁盘上也不允许有
    json没记录的残留样本对文件夹。另外还要复检双语指令的字典结构与两条指令本身
    (见check_single_save_annotation_caption)。

    CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG为True时还会真解一次落盘后的
    reference_image[0]，硬校验它的真实shape严格等于json里的width/height
    (即等于编辑后图的宽高)。这一条是"第一张参考图与编辑后图尺寸必须一致"
    这个核心不变式的最终验收: 只对账json里的数字是查不出resize有没有真的生效的。
    豁免子集(完全原样落盘)会跳过这一条。
    """

    check_error_message_list = []
    total_edit_pair_count = 0
    for per_set_name in sorted(set_folder_count_dict.keys()):
        per_set_dir_path = os.path.join(save_dataset_path, per_set_name)
        per_set_folder_count = set_folder_count_dict[per_set_name]
        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)
        per_exempt_aspect_ratio_align_flag = per_set_name in EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST
        for per_folder_index in range(per_set_folder_count):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            per_json_path = os.path.join(per_set_dir_path,
                                         f'{per_folder_name}.json')
            if not os.path.isfile(per_json_path):
                check_error_message_list.append(
                    f'{per_folder_name} json not exists')
                continue

            with open(per_json_path, 'r', encoding='UTF-8') as load_json_file:
                per_folder_annotation_dict = json.load(load_json_file)

            total_edit_pair_count += len(per_folder_annotation_dict)

            # 除每个子集最后一个文件夹外都必须是满10000对
            if per_folder_index < per_set_folder_count - 1 and len(
                    per_folder_annotation_dict) != PER_FOLDER_EDIT_PAIR_NUM:
                check_error_message_list.append(
                    f'{per_folder_name} edit pair num not match {len(per_folder_annotation_dict)} != {PER_FOLDER_EDIT_PAIR_NUM}'
                )

            per_folder_path = os.path.join(per_set_dir_path, per_folder_name)
            per_exist_pair_folder_name_list = sorted([
                per_save_pair_folder_name
                for per_save_pair_folder_name in os.listdir(per_folder_path)
                if os.path.isdir(
                    os.path.join(per_folder_path, per_save_pair_folder_name))
            ])
            per_expect_pair_folder_name_list = sorted([
                per_save_edited_image_name.removesuffix('.jpg')
                for per_save_edited_image_name in
                per_folder_annotation_dict.keys()
            ])
            if per_exist_pair_folder_name_list != per_expect_pair_folder_name_list:
                check_error_message_list.append(
                    f'{per_folder_name} pair folder not match {len(per_exist_pair_folder_name_list)} != {len(per_expect_pair_folder_name_list)}'
                )

            for per_save_edited_image_name in sorted(
                    per_folder_annotation_dict.keys()):
                per_annotation = per_folder_annotation_dict[
                    per_save_edited_image_name]

                # 每条标注的字段集合必须和约定的七个key严格一致，不能多也不能少
                if sorted(per_annotation.keys()) != sorted(
                        SAVE_ANNOTATION_KEY_NAME_LIST):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} annotation key not match'
                    )
                    continue

                if per_save_edited_image_name != per_annotation[
                        'edited_image']:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name not match'
                    )
                if not per_save_edited_image_name.endswith(
                        SAVE_EDITED_IMAGE_NAME_SUFFIX):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name suffix not match'
                    )
                if not per_save_edited_image_name.startswith(
                        f'{DATASET_NAME}_{per_set_name}_'):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name prefix not match'
                    )
                if per_save_edited_image_name != per_save_edited_image_name.lower(
                ):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name not all lower case'
                    )
                if not VALID_IMAGE_NAME_PATTERN.match(
                        per_save_edited_image_name):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name not match pattern'
                    )
                if not isinstance(per_annotation['reference_image'], list):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image not a list'
                    )
                    continue
                if per_annotation['reference_image_num'] != len(
                        per_annotation['reference_image']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match'
                    )
                # 本数据集全部20个子集都必须是单参考图
                if per_annotation[
                        'reference_image_num'] != per_expect_reference_image_num:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match set {per_annotation["reference_image_num"]} != {per_expect_reference_image_num}'
                    )
                # 参考图名的前缀必须和编辑后图像名同一个前缀、后缀必须是_reference.jpg
                for per_save_reference_image_name in per_annotation[
                        'reference_image']:
                    if not per_save_reference_image_name.endswith(
                            SAVE_REFERENCE_IMAGE_NAME_SUFFIX):
                        check_error_message_list.append(
                            f'{per_save_reference_image_name} reference image name suffix not match'
                        )
                    if not VALID_IMAGE_NAME_PATTERN.match(
                            per_save_reference_image_name):
                        check_error_message_list.append(
                            f'{per_save_reference_image_name} reference image name not match pattern'
                        )
                    if per_save_reference_image_name.removesuffix(
                            SAVE_REFERENCE_IMAGE_NAME_SUFFIX
                    ) != per_save_edited_image_name.removesuffix(
                            SAVE_EDITED_IMAGE_NAME_SUFFIX):
                        check_error_message_list.append(
                            f'{per_save_reference_image_name} reference image name prefix not match {per_save_edited_image_name}'
                        )
                # 图像宽高必须是正数
                if per_annotation['width'] <= 0 or per_annotation[
                        'height'] <= 0:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image shape not match {per_annotation["width"]} {per_annotation["height"]}'
                    )
                    continue
                # 短边和宽高比必须仍然满足过滤阈值
                if min(per_annotation['width'],
                       per_annotation['height']) < MIN_IMAGE_SHORT_SIDE:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a too small image {per_annotation["width"]} {per_annotation["height"]}'
                    )
                if max(per_annotation['width'] / per_annotation['height'],
                       per_annotation['height'] /
                       per_annotation['width']) > MAX_IMAGE_ASPECT_RATIO:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still an extreme aspect ratio image {per_annotation["width"]} {per_annotation["height"]}'
                    )

                # 双语指令的字典结构与两条指令本身的复检
                check_error_message_list.extend(
                    check_single_save_annotation_caption(
                        per_save_edited_image_name, per_annotation))

                per_save_pair_folder_name = per_save_edited_image_name.removesuffix(
                    '.jpg')
                per_pair_folder_path = os.path.join(per_folder_path,
                                                    per_save_pair_folder_name)
                per_expect_file_name_list = sorted(
                    [per_save_edited_image_name] +
                    list(per_annotation['reference_image']))
                per_exist_file_name_list = sorted(
                    os.listdir(per_pair_folder_path)) if os.path.isdir(
                        per_pair_folder_path) else []
                if per_exist_file_name_list != per_expect_file_name_list:
                    check_error_message_list.append(
                        f'{per_save_pair_folder_name} pair image file not match'
                    )
                    continue

                # 真解一次落盘后的reference_image[0]，硬校验它的shape严格等于
                # json里的width/height(也就是编辑后图的宽高)。
                # 非豁免子集才校验: 豁免子集是完全原样落盘的，尺寸本来就可以不等
                if CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG and not per_exempt_aspect_ratio_align_flag and len(
                        per_annotation['reference_image']) > 0:
                    per_check_save_reference_image_path = os.path.join(
                        per_pair_folder_path,
                        per_annotation['reference_image'][0])
                    try:
                        per_check_save_reference_image_w, per_check_save_reference_image_h = Image.open(
                            per_check_save_reference_image_path).size
                    except Exception as e:
                        per_check_save_reference_image_w, per_check_save_reference_image_h = 0, 0
                        print('4444', per_check_save_reference_image_path, e)

                    if per_check_save_reference_image_w != per_annotation[
                            'width'] or per_check_save_reference_image_h != per_annotation[
                                'height']:
                        check_error_message_list.append(
                            f'{per_save_edited_image_name} first reference image shape not match edited image shape '
                            f'{per_check_save_reference_image_w} {per_check_save_reference_image_h} != '
                            f'{per_annotation["width"]} {per_annotation["height"]}'
                        )

    print('3333', 'check total edit pair:', total_edit_pair_count,
          'check error:', len(check_error_message_list))

    return check_error_message_list, total_edit_pair_count


def check_save_dataset_path_empty(save_dataset_path):
    """落盘前断言输出目录必须为空(或不存在)，非空直接报错

    这一条是resize链路的前提: resave_single_image里有
    "if not os.path.exists(save_image_path)"这个跳过重复写盘的短路，
    如果输出目录里还留着上一轮跑出来的老图，
    这个短路会跳过写盘、但仍然返回内存里resize之后的shape，
    结果json记的宽高与磁盘上的实际文件不一致、收尾自校验也会大面积报错。
    所以本数据集必须落到全新的输出目录(或先手动删掉旧目录)重跑，
    不能在旧产物上增量跑。
    """
    if not os.path.exists(save_dataset_path):
        return

    per_exist_name_list = os.listdir(save_dataset_path)
    if len(per_exist_name_list) > 0:
        raise Exception(
            f'save dataset path not empty {save_dataset_path} '
            f'{len(per_exist_name_list)} {sorted(per_exist_name_list)[:10]}, '
            f'must remove the old resave result first')

    return


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path, SAVE_DATASET_DIR_NAME)
    # 必须落到全新的输出目录: 在旧产物上增量跑会让json宽高与磁盘错位
    check_save_dataset_path_empty(save_dataset_path)
    os.makedirs(save_dataset_path, exist_ok=True)

    edit_annotation_pair_list, total_annotation_file_count, task_annotation_file_count_dict, task_annotation_count_dict, subtask_annotation_count_dict, set_annotation_count_dict, total_annotation_count, illegal_line_count, annotation_dir_name_not_match_count, skip_subtask_count, unknown_subtask_count, empty_caption_count, null_like_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, invalid_placeholder_caption_count, invalid_annotation_image_shape_count, different_aspect_ratio_count, missing_image_count, invalid_save_image_name_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          illegal_line_count, annotation_dir_name_not_match_count,
          skip_subtask_count, unknown_subtask_count, empty_caption_count,
          null_like_caption_count, no_word_char_caption_count,
          too_short_caption_count, too_long_caption_count,
          invalid_placeholder_caption_count,
          invalid_annotation_image_shape_count, different_aspect_ratio_count,
          missing_image_count, invalid_save_image_name_count,
          len(set_annotation_count_dict), len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    invalid_caption_count_dict = {
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
    }

    # 除了subtask丢弃、指令过滤与长宽比过滤之外剩下的几项丢弃计数。
    # 这七项实测都是0，但必须一并算进过滤链路恒等式，
    # 否则上游一旦出现坏行/缺图，恒等式会误报成"计数不自洽"
    other_filter_count_dict = {
        'illegal_line_count': illegal_line_count,
        'annotation_dir_name_not_match_count':
        annotation_dir_name_not_match_count,
        'unknown_subtask_count': unknown_subtask_count,
        'invalid_annotation_image_shape_count':
        invalid_annotation_image_shape_count,
        'different_aspect_ratio_count': 0,
        'missing_image_count': missing_image_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
    }
    # different_aspect_ratio_count在恒等式里由入参单独计入，
    # 这里置0避免重复计算(它在EXPECTED_OTHER_FILTER_COUNT_DICT里的期望值也是0)

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        task_annotation_count_dict, subtask_annotation_count_dict,
        set_annotation_count_dict,
        total_annotation_count, total_annotation_file_count,
        len(edit_annotation_pair_list), skip_subtask_count,
        different_aspect_ratio_count, invalid_caption_count_dict,
        other_filter_count_dict, edit_annotation_pair_list)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        # 上游标注条数对不上说明上游017没跑完、产物被改动过，
        # 或者上游补传了新的parquet分片(见文件头"10M vs 2M"的分析)，
        # 继续往下跑只会得到一个悄悄少样本的新数据集
        raise Exception(
            f'check load annotation count error num {len(load_annotation_check_error_message_list)} {load_annotation_check_error_message_list[:10]}'
        )

    check_edit_annotation_pair_list = []
    invalid_image_count, image_different_aspect_ratio_count = 0, 0
    set_image_different_aspect_ratio_count_dict = {}
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result, per_check_set_name, per_check_edit_annotation_pair in tqdm(
                pool.imap(process_single_edit_pair_check,
                          edit_annotation_pair_list),
                total=len(edit_annotation_pair_list)):
            if per_check_result == 'invalid_image':
                invalid_image_count += 1
                continue
            # reference_image[0]与编辑后图长宽比不同的样本对在这里整对丢弃，
            # 逐子集记一份数字，方便看清是哪些任务类型天生对不齐。
            # 这是长宽比过滤的第二层(兜底)，第一层已经在文本解析阶段做过了，
            # 所以这里实测恒为0; 非0说明上游标注记的宽高与磁盘上的实际图像不一致
            if per_check_result == 'different_aspect_ratio':
                image_different_aspect_ratio_count += 1
                set_image_different_aspect_ratio_count_dict[
                    per_check_set_name] = set_image_different_aspect_ratio_count_dict.get(
                        per_check_set_name, 0) + 1
                continue
            check_edit_annotation_pair_list.append(
                per_check_edit_annotation_pair)

    print('1111', len(check_edit_annotation_pair_list), invalid_image_count,
          image_different_aspect_ratio_count)

    edit_pair_save_folder_pair_list, set_folder_count_dict = get_all_edit_pair_save_folder_pair(
        check_edit_annotation_pair_list, save_dataset_path)

    print('1111', len(edit_pair_save_folder_pair_list),
          len(set_folder_count_dict))
    if len(edit_pair_save_folder_pair_list) > 0:
        print('1111', edit_pair_save_folder_pair_list[0])

    save_result_list = []
    process_func = partial(process_single_edit_pair,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_save_result in tqdm(
                pool.imap(process_func, edit_pair_save_folder_pair_list),
                total=len(edit_pair_save_folder_pair_list)):
            if per_save_result is None:
                continue
            save_result_list.append(per_save_result)

    save_edit_pair_failed_count = len(edit_pair_save_folder_pair_list) - len(
        save_result_list)

    folder_edit_pair_count_dict, reference_image_num_mismatch_count = save_all_folder_annotation_json(
        save_result_list, save_dataset_path, set_folder_count_dict)

    total_save_reference_image_count = sum(
        [len(per_save_result[2]) for per_save_result in save_result_list])

    check_error_message_list, check_total_edit_pair_count = check_save_dataset(
        save_dataset_path, set_folder_count_dict)

    print('3333', 'total annotation file:', total_annotation_file_count,
          'total annotation:', total_annotation_count, 'illegal line:',
          illegal_line_count, 'annotation dir name not match:',
          annotation_dir_name_not_match_count, 'skip subtask:',
          skip_subtask_count, 'unknown subtask:', unknown_subtask_count,
          'empty caption:', empty_caption_count, 'null like caption:',
          null_like_caption_count, 'no word char caption:',
          no_word_char_caption_count, 'too short caption:',
          too_short_caption_count, 'too long caption:', too_long_caption_count,
          'invalid placeholder caption:', invalid_placeholder_caption_count,
          'invalid annotation image shape:',
          invalid_annotation_image_shape_count, 'different aspect ratio:',
          different_aspect_ratio_count, 'missing image:', missing_image_count,
          'invalid save image name:', invalid_save_image_name_count,
          'invalid image:', invalid_image_count,
          'image different aspect ratio:', image_different_aspect_ratio_count,
          'save edit pair failed:',
          save_edit_pair_failed_count, 'total save edit pair:',
          len(save_result_list), 'total save reference image:',
          total_save_reference_image_count, 'reference image num mismatch:',
          reference_image_num_mismatch_count, 'total save set:',
          len(set_folder_count_dict), 'total save folder:',
          len(folder_edit_pair_count_dict), 'check total edit pair:',
          check_total_edit_pair_count, 'check error:',
          len(check_error_message_list))

    save_check_result_path = os.path.join(save_dataset_path,
                                          'resave_check_result.json')
    save_check_result_dict = {
        'total_annotation_file_count': total_annotation_file_count,
        'total_annotation_count': total_annotation_count,
        'illegal_line_count': illegal_line_count,
        'annotation_dir_name_not_match_count':
        annotation_dir_name_not_match_count,
        # 任务类型不可知而被整对丢弃的条数(unknown_subtask那1行)
        'skip_subtask_count': skip_subtask_count,
        'skip_subtask_name_list': SKIP_SUBTASK_NAME_LIST,
        'unknown_subtask_count': unknown_subtask_count,
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
        'invalid_annotation_image_shape_count':
        invalid_annotation_image_shape_count,
        # 长宽比过滤的两层计数: 第一层只读标注不解图(主力)、第二层用真解码的shape兜底
        'different_aspect_ratio_count': different_aspect_ratio_count,
        'image_different_aspect_ratio_count':
        image_different_aspect_ratio_count,
        'set_image_different_aspect_ratio_count_dict':
        set_image_different_aspect_ratio_count_dict,
        'missing_image_count': missing_image_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'invalid_image_count': invalid_image_count,
        # 落盘口径标记，便于下游一眼看出这份产物的规格
        'resize_reference_image_to_edited_image_shape_flag': True,
        'long_side_align_extra_reference_image_flag': True,
        'exempt_aspect_ratio_align_set_name_list':
        EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST,
        'check_save_reference_image_shape_flag':
        CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG,
        # 本数据集特有: ti2i_caption与ti2i_caption_length的value都是字典，
        # 下游读取时必须显式感知这个结构差异
        'bilingual_ti2i_caption_flag': True,
        'ti2i_caption_key_name_list': SAVE_TI2I_CAPTION_KEY_NAME_LIST,
        'ti2i_caption_length_key_name_list':
        SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST,
        # 上游5181个重复md5 key的备查记录: 两行都是完整有效的独立编辑对、
        # 落盘名不用md5 key，所以既不去重也不影响落盘
        'upstream_duplicate_md5_key_count': EXPECTED_DUPLICATE_MD5_KEY_COUNT,
        'save_edit_pair_failed_count': save_edit_pair_failed_count,
        'total_save_edit_pair_count': len(save_result_list),
        'total_save_reference_image_count': total_save_reference_image_count,
        'reference_image_num_mismatch_count':
        reference_image_num_mismatch_count,
        'total_save_set_count': len(set_folder_count_dict),
        'total_save_folder_count': len(folder_edit_pair_count_dict),
        'check_total_edit_pair_count': check_total_edit_pair_count,
        'check_error_count': len(check_error_message_list),
        'save_image_jpeg_quality': SAVE_IMAGE_JPEG_QUALITY,
        'save_image_jpeg_sampling_factor_444_flag': True,
        'task_annotation_file_count_dict': task_annotation_file_count_dict,
        'task_annotation_count_dict': task_annotation_count_dict,
        'subtask_annotation_count_dict': subtask_annotation_count_dict,
        'set_annotation_count_dict': set_annotation_count_dict,
        'set_folder_count_dict': set_folder_count_dict,
        'folder_edit_pair_count_dict': folder_edit_pair_count_dict,
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    if check_total_edit_pair_count != len(save_result_list):
        check_error_message_list.append(
            f'check total edit pair count not match {check_total_edit_pair_count} != {len(save_result_list)}'
        )
    if len(check_error_message_list) > 0:
        # 收尾自校验不通过必须让上层感知，不能静默留下坏样本对或不满的文件夹
        raise Exception(
            f'check save dataset error num {len(check_error_message_list)} {check_error_message_list[:10]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/UnicEdit-10M'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
