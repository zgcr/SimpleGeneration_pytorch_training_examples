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

DATASET_NAME = 'conceptedit'

SAVE_DATASET_DIR_NAME = 'ConceptEdit-12M'

# ==============================================================================
# 【这个数据集只能产出图像编辑数据集，不能产出文生图数据集】
# 上游017解包出来的每一行有33个key，其中的文本字段共8个，逐条核定后:
#   detailed_en_instruction / short_en_instruction /
#   detailed_zh_instruction / short_zh_instruction
#       -> **4条合格且彼此独立的编辑指令**(中英双语 × 粗细两档)，本脚本全部保留;
#   instruction
#       -> **与detailed_en_instruction逐字100%相同**(抽样1182258/1182258行，
#          全库0条不同)，纯冗余，丢弃;
#   recaption_prompt_en / recaption_prompt_zh
#       -> VQA纠偏后的重写指令，全库只有1778742行(14.75%)非空，
#          且实测其中相当比例**根本不是指令而是整图描述**
#          (如"The image shows a woman standing in front of a vintage bus..."、
#           "A low-angle worm's-eye view looking straight up at massive tree
#           trunks...")，另有92条超过512字符。
#          按方案确认整体丢弃、不当主指令也不做兜底;
#   original_simple_caption
#       -> **源图(编辑前原图)的整图描述，不是编辑指令**:
#          抽样1182255条里1088833条(92.1%)是以"A/An/The"起头的陈述句
#          (如"A group of anime ballerinas in white tutus performing on a pink
#           stage.")。它描述的是参考图而不是编辑后图，
#          上游017的注释也明确警告过"**不能**拿来当文生图的caption使用"。
# 也就是说整个数据集**没有任何一列是编辑后图的内容描述**，
# 编辑指令只说"要改什么"、不说"整张图是什么"，拿它当t2i的prompt会得到
# 完全错误的图文对，所以本数据集只走ti2i这一条链路、不另写t2i脚本
# (与013.resave_scaleedit_12m / 015.resave_crispedit_2m /
#  016.resave_unicedit_10m 同样的处置)。
# 上游017自己也把dataset_task_type写死成image_edit(全库单值)，
# HF README的task_categories是image-to-image、
# tags是image-editing/instruction-based-editing。
#
# 【为什么"用源图 + original_simple_caption"这条t2i路线也被否掉】
# 理论上"源图 + 源图描述"本身是一对合法的t2i样本(按<子集>/<batch>/<id前两段>
# 分组后实测有6095133个源图组、组内源图md5完全相同、组内分辨率0冲突)，
# 但上游017的注释指出这些源图"都来自Fine-T2I的同一张原图，只是做了不同预处理"，
# 而t2i_datasets/fine-t2i已经落盘了那批原图(630万张)。
# 实测把ConceptEdit的39万条组caption与fine-t2i全部3087945条prompt做归一化精确
# 匹配**命中0条**(两者文本体系不同: 一个是短caption、一个是长prompt)，
# 所以文本层查不出重复、图像层又没有任何字段能把ConceptEdit的流水号id回连到
# fine-t2i的uuid，无法证实也无法证伪"是否为同一批图的低分辨率副本"。
# 按方案确认: 本数据集**不产出t2i数据集**。
#
# ==============================================================================
# 【上游017.unzip_conceptedit_12m_dataset.py的产物规格(全量实测，非抽样)】
# ConceptEdit-12M/
# ├── unzip_annotations/<4个子集>/batch_{0..N}.jsonl
# │                                    616个文件 / 12057500行 / 26G
# ├── unzip_images/<4个子集>/<batch名>/
# │       <sample_id>.json          上游原样落盘的单样本json(本脚本不读)
# │       <sample_id>_source.jpg    参考图(编辑前原图)，实测恒为JPEG/RGB
# │       <sample_id>_edit.png      编辑后图，实测恒为PNG/RGB
# │                                    36172500个文件(约21T)
# └── unzip_check_missing_images.json  上游自校验: 0缺图 / 0孤儿图 / 0重名成员 /
#                                     0隔离样本 / 0错误，仅4786条非致命warning
# 上游逐子集行数(合计12057500):
#   enhanced_prompt_random_resolution 3182923 /
#   enhanced_prompt_square_resolution 3050142 /
#   original_prompt_random_resolution 3319785 /
#   original_prompt_square_resolution 2504650
#
# 【逐行33个key的全量核对结论(key组合616个分片100%一致、无缺字段)】
#   dataset_task_type        恒为image_edit(整库单值)                -> 丢弃
#   sample_key               "<subset>/<archive>/<sample_id>"        -> 只做交叉校验
#   sample_id                形如0_0_0 / 1114_0_auto                 -> 拼保存图像名
#   subset_name              上游4个子集目录名                        -> 推导子集名
#   archive_name             batch_N                                 -> 只做交叉校验
#   prompt_type              enhanced_prompt / original_prompt       -> **子集名第2段**
#   resolution_type          square_resolution / random_resolution   -> **子集名第3段**
#   instruction              与detailed_en逐字100%相同(纯冗余)        -> 丢弃
#   short_en_instruction     简短英文指令                             -> ti2i_caption
#   short_zh_instruction     简短中文指令                             -> ti2i_caption
#   detailed_en_instruction  详细英文指令                             -> ti2i_caption
#   detailed_zh_instruction  详细中文指令                             -> ti2i_caption
#   original_simple_caption  **源图**的整图描述(不是编辑指令)          -> 丢弃
#   edit_category            192个取值的编辑领域大类                  -> 丢弃(粒度太粗)
#   edit_sub_category        1595个取值的功能模块                     -> 丢弃(粒度太粗)
#   edit_task                7112个取值的**原子编辑操作**             -> **子集名第1段**
#   edit_detail              更细的编辑细节(取值更多)                 -> 丢弃
#   overall_vqa_score        1.0/0.8/0.75/0                          -> 丢弃(不过滤)
#   keep                     **全库12057500行恒为true**              -> 丢弃
#   wrong_count              0(11514348) / 1(543152)                 -> 丢弃(不过滤)
#   recaption_prompt_en/zh   VQA纠偏重写指令，85%为空且含整图描述     -> 丢弃
#   vqa_dimension_num        恒5(另有4486行不等于5)                  -> 丢弃
#   vqa_passed_num           通过的VQA维度数                          -> 丢弃
#   not_passed_vqa_dimension_name_list 未通过的维度名(含4条问句脏值)  -> 丢弃
#   reference_image_path_list  **恒为长度1的list**(全库0例外)         -> 定位参考图
#   reference_image_num      **恒为1**(全库0例外)                    -> 不采信，现算
#   edited_image_path        编辑后图相对路径                         -> 定位编辑后图
#   annotation_path          上游原样落盘的单样本json路径             -> 丢弃
#   reference_image_shape    [宽, 高]                                -> 长宽比预筛
#   edited_image_shape       [宽, 高]                                -> 长宽比预筛
#   reference_image_suffix   **恒为.jpg**                            -> 丢弃
#   edited_image_suffix      **恒为.png**                            -> 丢弃
# 上游的unzip_check_missing_images.json也不读不搬(它的实测值已写死成本脚本的
# EXPECTED_*常量用于硬对账)。
#
# 【图像规格实测】
# 抽样600对(1200张)逐张PIL打开: 参考图100%是JPEG/RGB、编辑后图100%是PNG/RGB、
# 0缺图、0坏图、宽高与标注100%一致。
# 编辑后图分辨率只有9种(全是16的倍数，1024x1024占58.9%)、短边最小768、
# 最大宽高比1.7708; 参考图分辨率26种、短边最小512。
# 即"短边<64"与"宽高比>8"这两条过滤**全库一条都不会命中**(仍照写兜底)。
# 另外专查了8个可能带透明通道的任务(抠图与图层/画面扩展/多图合成与连贯性/
# 参考图驱动/布局与Logo/裁剪与构图/空间与几何变换/视角变换)共320张编辑后图:
# **100%是RGB三通道PNG、0张4通道、0张带alpha**，
# 所以不存在"4通道图被cv2.IMREAD_COLOR静默丢掉alpha"的风险。
# 其中matting_and_layer(抠图与图层)的编辑后图实测**不是黑白二值mask**、
# 而是"主体保留 + 背景替换成纯色(以白底为主)"的常规彩色图
# (binary_like_ratio p50=0.375、saturation_mean p50=48.66)，
# 所以统一按RGB重编码成jpg不会丢任何信息。
# ==============================================================================

# 本脚本只读unzip_annotations这一套标注(616个jsonl / 12057500行)定位样本对，
# **绝不os.walk图像目录**: 上游解出3617万个小文件，扫目录树在NAS上不可接受
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 上游标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如unzip_images/enhanced_prompt_random_resolution/batch_0/0_0_0_source.jpg)，
# 不需要再往前拼任何子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 上游落盘图像名的后缀: 参考图是_source、编辑后图是_edit
# (扩展名实测参考图恒为.jpg、编辑后图恒为.png，但取前缀时仍然先去扩展名
#  再去这个后缀，上游哪天换格式也不会取错)
LOAD_REFERENCE_IMAGE_NAME_SUFFIX = '_source'

LOAD_EDITED_IMAGE_NAME_SUFFIX = '_edit'

# 上游子集目录名(4个)，**子集名的第2、3段就是从它拆出来的**
ANNOTATION_SUBSET_NAME_KEY_NAME = 'subset_name'

# 上游分片名(batch_N)，只用于与标注文件名交叉校验，不参与子集划分也不进保存名
ANNOTATION_ARCHIVE_NAME_KEY_NAME = 'archive_name'

# 子集内唯一的样本id，形如0_0_0 / 1114_0_auto。
# **实测(subset_name, sample_id)全库12057500个组合0重名**，
# 而"prompt类型 × 分辨率类型"与上游4个子集是一一对应的双射，
# 所以"子集名 + sample_id"天然全局唯一，不需要再往保存名里塞batch
ANNOTATION_SAMPLE_ID_KEY_NAME = 'sample_id'

# ==============================================================================
# 【子集划分口径: 任务类型 × prompt类型 × 分辨率类型 = 38 × 2 × 2 = 152个子集】
#
# 【为什么用edit_task这一级而不是edit_category / edit_sub_category】
# 上游有三级编辑分类，与015.resave_crispedit_2m_ti2i_dataset.py的任务类型分级
# (add / remove / replace / color / background_change / style / motion_change
#  这种**原子编辑操作**)做粒度对照后:
#   edit_category     192个取值(7个中文大类覆盖99.1971%)
#                     = "编辑领域大类"，比crispedit**粗两级**:
#                       "通用物体与实体编辑"一类(384万对)就把crispedit的
#                       add/remove/replace/color全部吞掉;
#   edit_sub_category 1595个取值(top22覆盖98.4526%)
#                     = "功能模块"，比crispedit**粗一级**:
#                       "物体管理与操作"(288万对) = add+remove+replace+抠图+
#                       空间变换 五合一;
#                       "环境与风格"(305万对) = style+天气+光照+background 四合一;
#   edit_task         7112个取值(top38覆盖95.7444%)
#                     = **原子编辑操作，与crispedit粒度一致**:
#                       风格迁移<->style / 替换目标<->replace /
#                       颜色与材质更改<->color / 背景管理<->background_change /
#                       姿态与动作驱动<->motion_change，
#                       唯一差异是上游把add与remove合并成了"增删目标"一个值。
# 所以**按edit_task分子集**。
#
# 【为什么还要再乘上prompt类型与分辨率类型】
# 上游017的注释明确要求这两个正交属性"**必须原样保留成属性**(下游按需过滤/配比)":
#   enhanced_prompt / original_prompt     : 编辑指令是否被增强改写过
#   square_resolution / random_resolution : 编辑后图是正方形还是任意长宽比
# 但新标注固定只有7个key、装不下它们，所以按方案确认把它们编进**子集名**里，
# 这样属性不丢、下游依然能按需过滤或配比。
#
# 【阈值为什么取20000】
# 按edit_task条数分档实测: >=100000是26个子集(覆盖89.96%)、>=50000是34个
# (94.81%)、**>=20000是38个(95.7444%)**、>=10000是40个(95.93%)、
# >=5000是50个(96.51%)。取20000之后38个任务每个都在22008对以上、
# 交叉切分后每个子集至少4904对; 再往下放到5000只多覆盖0.77%、
# 却会多出12个连半个文件夹都填不满的子集，所以取20000。
# 被丢弃的是**7112 - 38 = 7074个长尾取值 / 513124条(4.256%)**，其中包含:
#   547个英文写法变体(Style Transfer 10794 / Replace Target 4040 /
#   Exposure & Color Correction 3091 / Environment & Weather Simulation 2373 ...
#   合计69476条);
#   39个含"->"的脏值(上游把两级拼进了一格，如"环境与风格 -> 环境与天气模拟" 4735，
#   合计9863条);
#   其余为真长尾(大量取值只对应1对样本)。
# 按方案确认这些长尾任务**连带其样本整体丢弃**(与003丢弃568个长尾任务、
# 015丢弃motion_change同一处置)，不做任何归一化合并、也不并进mix。
# 长尾判定排在所有指令与图像过滤之前，所以后面每一项过滤计数都只统计38个保留任务。
#
# 【为什么写死映射表而不是每行现lower()】
# 上游edit_task是**中文原值**，而保存图像名里只允许[a-z0-9_\-\.]
# (CJK字符会被VALID_IMAGE_NAME_PATTERN判成非法名)，所以必须有一张中->英映射表;
# 写死之后任何上游新增/改名的取值都会被check_load_annotation_count里的
# "unknown task"硬拦下来，而不是静默把几十万个样本对写进错误子集。
# 实测38个英文子集名互不相同、与edit_task严格一对一，不存在跨组归并
# ==============================================================================
ANNOTATION_TASK_NAME_KEY_NAME = 'edit_task'

ANNOTATION_PROMPT_TYPE_KEY_NAME = 'prompt_type'

ANNOTATION_RESOLUTION_TYPE_KEY_NAME = 'resolution_type'

# 上游edit_task原值 -> 归一化后的任务名(即子集名的第1段)，最终保留38个任务。
# 按过滤前条数降序排列。表里没有的取值就是要整体丢弃的7074个长尾任务
GET_TASK_SET_NAME_DICT = {
    '增删目标': 'add_or_remove_object',
    '风格迁移': 'style_transfer',
    '替换目标': 'replace_object',
    '环境与天气模拟': 'environment_and_weather_simulation',
    '颜色与材质更改': 'color_and_material_change',
    '全局光照控制': 'global_lighting_control',
    '背景管理': 'background_management',
    '发型与毛发编辑': 'hair_editing',
    'ACG与娱乐': 'acg_and_entertainment',
    '曝光与色彩校正': 'exposure_and_color_correction',
    '虚拟试穿': 'virtual_try_on',
    '特征属性编辑': 'facial_attribute_editing',
    '字体与样式': 'font_and_style',
    '文字修改与生成': 'text_modification_and_generation',
    '情绪与表情控制': 'emotion_and_expression_control',
    '电商与营销': 'ecommerce_and_marketing',
    '参考图驱动': 'reference_image_driven',
    '画面扩展': 'image_outpainting',
    '美颜塑形': 'beauty_retouching',
    '文字擦除与去水印': 'text_removal_and_dewatermark',
    '抠图与图层': 'matting_and_layer',
    '姿态与动作驱动': 'pose_and_action_driven',
    '视角变换': 'viewpoint_transformation',
    '布局与Logo': 'layout_and_logo',
    '细节修复': 'detail_restoration',
    '去噪与去模糊': 'denoise_and_deblur',
    '草图与涂抹控制': 'sketch_and_scribble_control',
    '超分辨率/清晰化': 'super_resolution',
    '多图合成与连贯性': 'multi_image_composition',
    '空间与几何变换': 'spatial_and_geometric_transform',
    '神态与视线修复': 'gaze_and_expression_repair',
    '环境与风格': 'environment_and_style',
    '复杂指令遵循': 'complex_instruction_following',
    '合影与多人合成': 'group_photo_composition',
    '逻辑推理生成': 'logical_reasoning_generation',
    '裁剪与构图': 'crop_and_composition',
    '文档与教育': 'document_and_education',
    '身形重塑': 'body_reshaping',
}

# 上游prompt_type -> 子集名的第2段(编辑指令是否被增强改写过)
GET_PROMPT_TYPE_SET_NAME_DICT = {
    'enhanced_prompt': 'enhanced_prompt',
    'original_prompt': 'original_prompt',
}

# 上游resolution_type -> 子集名的第3段(编辑后图是正方形还是任意长宽比)
GET_RESOLUTION_TYPE_SET_NAME_DICT = {
    'square_resolution': 'square_resolution',
    'random_resolution': 'random_resolution',
}

# 上游子集名 -> [prompt_type, resolution_type]，用于与行内那两个字段交叉校验:
# 不一致说明上游产物被搬动过，继续跑会把样本对写进错误子集(实测0条)
GET_SUBSET_PROMPT_RESOLUTION_TYPE_DICT = {
    'enhanced_prompt_random_resolution':
    ['enhanced_prompt', 'random_resolution'],
    'enhanced_prompt_square_resolution':
    ['enhanced_prompt', 'square_resolution'],
    'original_prompt_random_resolution':
    ['original_prompt', 'random_resolution'],
    'original_prompt_square_resolution':
    ['original_prompt', 'square_resolution'],
}

# 找不到任务类型时才用的兜底子集名。
# 本数据集38个保留任务的任务类型全部可知、长尾任务已在解析阶段整体丢弃，
# 一条都不会落进mix，所以最终产出的就是152个交叉子集。
# 保留这个常量只是为了和002/005/013/015/016的口径保持一致，
# 并防止上游之后新增edit_task取值时被静默漏处理
# (新取值会被check_load_annotation_count的"unknown task"硬拦下来)
MIX_SET_NAME = 'mix'

# 上游每个子集的实测行数(合计12057500)，解析阶段逐项硬对账。
# 少一行都说明上游017没跑完或产物被改动过
EXPECTED_SUBSET_ANNOTATION_COUNT_DICT = {
    'enhanced_prompt_random_resolution': 3182923,
    'enhanced_prompt_square_resolution': 3050142,
    'original_prompt_random_resolution': 3319785,
    'original_prompt_square_resolution': 2504650,
}

EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 616

EXPECTED_TOTAL_ANNOTATION_COUNT = 12057500

# 38个保留任务的实测**原始条数**(即"过滤之前"每个任务分到的条数，
# 合计11544376)，解析阶段逐项硬对账。
# 这能拦住"某个edit_task被映射进错误子集"这种总数级对账看不出来的问题
EXPECTED_TASK_ANNOTATION_COUNT_DICT = {
    '增删目标': 1448853,
    '风格迁移': 1424710,
    '替换目标': 1227437,
    '环境与天气模拟': 919891,
    '颜色与材质更改': 819909,
    '全局光照控制': 596284,
    '背景管理': 480804,
    '发型与毛发编辑': 426130,
    'ACG与娱乐': 410353,
    '曝光与色彩校正': 369963,
    '虚拟试穿': 312436,
    '特征属性编辑': 267134,
    '字体与样式': 260978,
    '文字修改与生成': 211446,
    '情绪与表情控制': 198528,
    '电商与营销': 157797,
    '参考图驱动': 155750,
    '画面扩展': 146541,
    '美颜塑形': 143606,
    '文字擦除与去水印': 142042,
    '抠图与图层': 141447,
    '姿态与动作驱动': 134943,
    '视角变换': 130707,
    '布局与Logo': 112464,
    '细节修复': 105250,
    '去噪与去模糊': 101839,
    '草图与涂抹控制': 94256,
    '超分辨率/清晰化': 83650,
    '多图合成与连贯性': 80922,
    '空间与几何变换': 77838,
    '神态与视线修复': 68806,
    '环境与风格': 64227,
    '复杂指令遵循': 60907,
    '合影与多人合成': 54043,
    '逻辑推理生成': 33367,
    '裁剪与构图': 32690,
    '文档与教育': 24420,
    '身形重塑': 22008,
}

# 被长尾任务过滤整体丢弃的实测精确条数(7074个取值 / 513124条 / 4.256%)。
# 这一步发生在任何指令与图像过滤之前，
# 所以下面那些指令过滤计数统计的都只是38个保留任务里的样本
EXPECTED_SKIP_TAIL_TASK_COUNT = 7074

EXPECTED_SKIP_TAIL_TASK_ANNOTATION_COUNT = 513124

# 4条编辑指令的上游key名。
# 落盘顺序按方案确认为: 简短英文 -> 简短中文 -> 详细英文 -> 详细中文,
# 下面SAVE_TI2I_CAPTION_KEY_NAME_LIST与这个列表严格一一对应(同序)
ANNOTATION_CAPTION_KEY_NAME_LIST = [
    'short_en_instruction',
    'short_zh_instruction',
    'detailed_en_instruction',
    'detailed_zh_instruction',
]

# 参考图相对路径(list，本数据集恒为长度1)与编辑后图相对路径(字符串)
ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'reference_image_path_list'

ANNOTATION_EDITED_IMAGE_KEY_NAME = 'edited_image_path'

# 上游记录的样本唯一键，形如
# "enhanced_prompt_random_resolution/batch_0/0_0_0"。
# 只用来和从图像文件名现取的前缀三方交叉比对，不写进新标注
ANNOTATION_SAMPLE_KEY_KEY_NAME = 'sample_key'

# 上游解图像header得到的参考图/编辑后图真实宽高([宽, 高])。
# **只用于长宽比预筛与统计，绝不采信**: 写进json的width/height一律取自
# 实际写盘数组的shape
ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME = 'reference_image_shape'

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
# 【本数据集: ti2i_caption与ti2i_caption_length这两个key的value都是字典】
# 处理方案与016.resave_unicedit_10m_ti2i_dataset.py完全一致(那个数据集是
# 中英双语两条)，只是本数据集有**4条**指令: 中英双语 × 粗细两档。
# 4条都全库非空(空串仅short_zh 118条 / detailed_zh 128条)、
# 且互不相同(short_en与detailed_en相同的只有114条/0.0096%、
# short_zh与detailed_zh相同的只有259条/0.0219%)，
# 短指令长度只有详细指令的28%(p50)，是真正独立的精简指令，4条都有训练价值,
# 丢掉任何一条都是信息损失; 而json的key是编辑后图像名、一个样本对只能有一条记录,
# 没法像单语数据集那样把caption直接写成字符串。
# 所以本数据集把这两个key的value都写成字典:
#   "ti2i_caption": {
#       "short_english_ti2i_caption": "<short_en_instruction原文，已strip>",
#       "short_chinese_ti2i_caption": "<short_zh_instruction原文，已strip>",
#       "detailed_english_ti2i_caption": "<detailed_en_instruction原文，已strip>",
#       "detailed_chinese_ti2i_caption": "<detailed_zh_instruction原文，已strip>"
#   },
#   "ti2i_caption_length": {
#       "short_english_ti2i_caption_length": len(short_english_ti2i_caption),
#       "short_chinese_ti2i_caption_length": len(short_chinese_ti2i_caption),
#       "detailed_english_ti2i_caption_length": len(detailed_english_ti2i_caption),
#       "detailed_chinese_ti2i_caption_length": len(detailed_chinese_ti2i_caption)
#   }
# 下游读取时必须显式感知这个结构差异(其余数据集是字符串/整数)。
# 收尾自校验会专门校验: 两个字典的key集合严格等于下面两个列表、
# 4个length都等于对应字符串的实际len()、4条指令都在[MIN, MAX]区间内、
# 都不含[Vn*]占位符、都不是null字面量、都含有至少一个文字字符。
# ==============================================================================
SAVE_TI2I_CAPTION_KEY_NAME = 'ti2i_caption'

SAVE_TI2I_CAPTION_LENGTH_KEY_NAME = 'ti2i_caption_length'

# 与ANNOTATION_CAPTION_KEY_NAME_LIST严格同序一一对应
SAVE_TI2I_CAPTION_KEY_NAME_LIST = [
    'short_english_ti2i_caption',
    'short_chinese_ti2i_caption',
    'detailed_english_ti2i_caption',
    'detailed_chinese_ti2i_caption',
]

SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST = [
    'short_english_ti2i_caption_length',
    'short_chinese_ti2i_caption_length',
    'detailed_english_ti2i_caption_length',
    'detailed_chinese_ti2i_caption_length',
]

# 本数据集每个编辑对只有编辑前原图这1张参考图
# (上游reference_image_path_list恒为长度1的list、reference_image_num全库恒为1)，
# 所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 保存名形如
# conceptedit_style_transfer_enhanced_prompt_random_resolution_1140000_0_0_edited.jpg
# 实测最长109字符(最长子集名68字符 +
# environment_and_weather_simulation_enhanced_prompt_random_resolution，
# sample_id最长14字符)，远低于文件系统单文件名255字节的上限。
# 38个英文子集名全是ASCII小写、sample_id实测只含 0123456789_aotu 这些字符，
# 所以不会出现CJK字符或其它异常字符
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 上游sample_id的形态: <数字>_<数字>_<数字> 或 <数字>_<数字>_auto。
# 上游017的注释明确警告过"**id不全是纯数字**，绝不能按下划线切分取数字",
# 这里只做整串形态校验、不做任何切分。
# 实测全库12057500行100%满足这个模式(0条例外)，且sample_id里出现过的字符
# 只有 0-9 _ a o t u(即数字与"auto")
VALID_LOAD_SAMPLE_ID_PATTERN = re.compile(r'^\d+_\d+_(?:\d+|auto)$')

# 上游分片名形态: batch_<编号>
VALID_LOAD_ARCHIVE_NAME_PATTERN = re.compile(r'^batch_\d+$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃。
# 实测抽样1520张(参考图 + 编辑后图)100%是RGB:
# 参考图恒为JPEG/RGB、编辑后图恒为PNG/RGB、**0张带alpha通道**
# (专查了抠图与图层等8个可能带透明通道的任务共320张编辑后图，全部是3通道)
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

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
#       长宽比与编辑后图"可对齐" -> LANCZOS resize到编辑后图尺寸
#       长宽比"不可对齐"         -> **整个样本对丢弃**(resize会把画面拉伸变形)
#
#   reference_image[k>=1](第二张视觉条件图/主体图/物体图):
#       长宽比可对齐   -> resize到编辑后图尺寸
#       长宽比不可对齐 -> 按**长边对齐**等比resize(不裁剪、不形变、不丢弃)
#   (本数据集恒为1张参考图，k>=1这条分支不会被走到，只为与003/004/016口径一致)
#
# ==============================================================================
# 【本数据集特有: "可对齐"判定引入1%长宽比容差，这是本项目第一个开容差的脚本】
#
# 003/004/013/015/016用的都是Fraction最简分数比、**零容差**。
# 但本数据集在零容差口径下会丢掉**4044764对(33.5%)**:
#   参考图与编辑后图尺寸完全相同        1503619 (12.5%)
#   长宽比严格相同但尺寸不同            6509117 (54.0%)  -> 等比resize，零形变
#   **长宽比不严格相同                  4044764 (33.5%)** -> 零容差下会全丢
# 而这4044764对的长宽比偏差**只有0.35%~0.46%**(共4种取值，全部集中在两个
# random_resolution子集)，成因是上游把编辑后图resize到了16倍数的bucket:
#   1600x1200 (4:3)  -> 1168x880 (73:55)   偏差0.46%
#   1920x1536 (5:4)  -> 1136x912 (71:57)   偏差0.35%
#   2048x1152 (16:9) -> 1360x768 (85:48)   偏差0.39%
#   1536x2048 (3:4)  -> 880x1168           偏差0.45%
#
# 【为什么可以开容差: 用SIFT + RANSAC做了互斥假设检验，不是拍脑袋放松标准】
# 两个互斥假设:
#   H1 纯各向异性resize: 上游把源图直接非等比缩放到bucket尺寸后再编辑
#      => 参考图resize到(ew,eh)之后与编辑后图**几何完全一致**，逐像素对齐成立;
#   H2 裁剪 + 等比resize: 上游先裁掉约0.4%的一条边再等比缩放
#      => 直接resize会引入0.4%尺度错配、边缘系统性错位，逐像素对齐**不成立**
#         (015的motion_change就是这种情况，实测sx=1.0196/sy=0.9792，只能整体丢弃)。
# 判别方法(每个样本对独立执行): 参考图LANCZOS resize到编辑后图尺寸 ->
# SIFT + 0.75比值检验匹配 -> RANSAC(阈值1.5px)估**完整仿射**(允许独立sx/sy/
# shear/平移) -> 看仿射矩阵是否≈单位矩阵。
# 三组对照(A组长宽比不同 / B组等比缩放 / C组尺寸完全相同(误差下界))实测:
#            指标            A组(待决策)    B组(已接受)   C组(误差下界)
#   sx p50                     1.0001        1.0001        1.0000
#   sy p50                     1.0002        1.0001        1.0001
#   shear p50                  0.0000       -0.0000       -0.0000
#   重投影RMSE p50            0.4749 px     0.5178 px     0.4566 px
#   **恒等位移 p50**          0.3597 px     0.4048 px     0.3351 px
#   恒等位移 p90              0.7769 px     0.9290 px     0.7749 px
# **A组每一项都与C组(误差下界)同量级、且优于当前已被接受的B组**，
# 亚像素残差0.36px就是SIFT定位精度 + LANCZOS重采样 + PNG->JPEG的测量地板,
# 不存在任何系统性错位。
# 竞争假设判别(按7类尺寸对各抽样，89对，两条路各估一次仿射):
#   plan1 各向异性resize到编辑后图: 偏离单位矩阵 p50=0.00025 / p90=0.00474
#   plan2 长边等比对齐(裁剪假设)  : 偏离单位矩阵 p50=0.00445 / p90=0.00681
#   **plan1更接近单位矩阵的有84/89对**; 且plan2在**恰好一个轴**上系统性地出现
#   1.0033~1.0048的尺度，数值**精确等于该尺寸对的理论长宽比偏差**(0.332%~0.484%)，
#   而plan1两个轴都是1.0000x —— 这在统计上排除了H2。
# 结论: 这4044764对是**纯各向异性resize**产生的，参考图resize成编辑后图分辨率
# (同时也就是相同长宽比)之后与编辑后图达到亚像素级逐像素对齐，
# 与"尺寸完全相同"那批的对齐质量无差别，按方案确认**全部保留**。
#
# 【容差取值】实测偏差最大0.46%，这里取1%(2倍安全裕度)。
# 判定顺序: **先判Fraction精确相等，再判容差**，
# 这样"长宽比严格相同"那批(66.5%)走的仍然是与003/004/016完全一致的零容差路径,
# 容差只对那4044764对生效。最大各向异性形变0.46%(肉眼不可辨)。
# 保留Fraction这条精确分支而不是统一用浮点容差，是为了让口径变化可追溯:
# 一旦上游换版出现真正的大偏差样本，它们会被1%这条线挡住而不是被静默拉伸。
# ==============================================================================
MAX_ASPECT_RATIO_TOLERANCE = 0.01

# 参考图resize到目标尺寸时用的重采样方式。
# 按方案确认用PIL的LANCZOS而不是cv2.resize: 与项目既定口径保持一致
SAVE_IMAGE_RESIZE_RESAMPLING = Image.Resampling.LANCZOS

# 豁免尺寸对齐的子集: 这些子集的编辑后图与全部参考图**完全原样落盘**，
# 不判长宽比、不resize、不丢弃、也不做长边对齐。
# 本数据集152个子集全部都要求参考图与编辑后图对齐，没有任何子集需要豁免，
# 所以这里是空列表(保留这个常量只为与003/004/016口径一致)
EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST = []

# ==============================================================================
# 【必须显式感知: 有4个任务天生会改变画面几何，其对齐性质与局部编辑不同】
# 按方案确认这4个任务**全部保留**(它们是合法的编辑任务类型，且已按任务类型
# 独立分子集，下游可以自行决定是否采样/如何配比)，这里只把它们显式列出来打标,
# 并写进resave_check_result.json，让下游能一眼识别、单独控制采样比例。
#
# 实测(参考图resize到编辑后图尺寸之后，与局部编辑类任务做对照):
#   任务                        恒等位移p50   inlier_ratio   匹配失败对数
#   增删目标(对照,局部编辑)        0.258 px       0.963         1/24
#   颜色与材质更改(对照,局部编辑)  0.372 px       0.861         0/24
#   spatial_and_geometric_transform 0.681 px      0.821         6/24
#   image_outpainting              0.694 px       0.819         4/24
#   crop_and_composition           1.853 px       0.748         6/24
#   **viewpoint_transformation     79.219 px      0.338        14/24**
# viewpoint_transformation最极端: sx=0.694/sy=0.725、平移175px，
# 24对里14对根本匹配不上 —— 这是语义上真正的"换机位重画"。
# **这不是预处理缺陷**: 这些任务的编辑语义本身就是改变几何，GT就应该长这样;
# 与015整体丢弃motion_change的理由不同(那个是预处理层面无法还原逐像素对应)。
# 但下游TorchAspectRatioBucketResize会因为"参考图与编辑后图宽高比相同"而走
# **逐像素RoPE对齐分支**，而这4个任务的实际内容并不逐像素对应，
# 所以必须让下游显式知道这批子集的性质
GEOMETRY_CHANGE_TASK_SET_NAME_LIST = [
    'viewpoint_transformation',
    'crop_and_composition',
    'image_outpainting',
    'spatial_and_geometric_transform',
]

# 收尾自校验时是否真解一次reference_image[0]、硬校验它的shape等于json里的
# width/height。按方案置True: "第一张参考图与编辑后图尺寸必须一致"是核心不变式，
# 而只对账json里的数字是查不出resize有没有真的生效的，必须真解一次图。
# 代价是收尾自校验要多解约1154万张参考图，在NAS上会明显变慢
CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG = True

# ==============================================================================
# 【指令长度阈值: 4条指令用同一套[4, 512]，且是"四条联合判定"】
# **联合判定口径**: empty / null_like / no_word_char / too_short / too_long /
# invalid_placeholder 这六项都是**4条指令里只要有任意一条不合格，就整对丢弃**
# (而不是只丢那一条)。理由: 落盘的json里4条指令都要写，
# 留下一条坏的等于把坏数据混进训练集。
#
# 下限4(与003/004/015.0对中文放宽的口径一致，而不是016的10):
#   英文实测 short_en min=8("Add fog." / "Zoom out." 这种完全合法的短指令)、
#            detailed_en min=20;
#   中文实测 short_zh 有**250741条(2.08%)长度<10**，但它们是中文正常的简洁表达
#            (如"将天气改为阴天。"=8字符，对应英文是
#             "Change the weather to an overcast day."=38字符)，
#            用10这条线会砍掉25万条完全正常的样本，属于砍正常类别而不是砍异常值;
#            short_zh长度<4的只有4条。
#   所以统一取4，实测联合判定只砍掉28条。
#
# 上限512(与003/004/013/015/016一致):
#   detailed_en实测 min=20 / p50=230 / p90=300 / p99=380 / max=730,
#   超过512的680条(0.0056%)是模型把整段场景描述写进了指令、语义冗长;
#   short_en max=360、detailed_zh max=640、short_zh max=... 实测
#   **这三条超过512的都是0条**，所以联合判定的680条全部来自detailed_en。
# ==============================================================================
MIN_CAPTION_LENGTH = 4

MAX_CAPTION_LENGTH = 512

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 4条指令都要各自命中，实测0条被判掉，只作防御性拦截
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)，
# 与004/005/013/015/016口径一致。
# 4条指令只要任意一条命中就整对丢弃。实测本数据集0条命中，只作防御性拦截
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
# 这套写法与002/003/004/005/013/015/016完全一致，保证跨数据集口径统一。
# 本数据集恒为1张参考图(N == 0)，所以4条指令里都不允许出现任何占位符
# (实测含[Vn*]形态的0条)，这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2

# jpg重编码质量与色度采样方式，按方案确认用质量97 + 色度4:4:4
# (与005/015/016口径一致)，而不是cv2.imencode的默认值(质量95 + 色度4:2:0)。
#
# 【本数据集必须用这一档的理由】
# 1) 编辑后图**100%是PNG无损**(全库12057500张，edited_image_suffix恒为.png)，
#    不存在"源图本来就是jpg、重编码有量化表幂等性"这种可以省质量的情况:
#    这里每一次重编码都是**从无损到有损的第一次损失**，
#    是编码器损失的真实度量，质量档位越低损失越直接;
# 2) 真正的瓶颈是色度下采样而不是质量值: OpenCV默认的4:2:0会把色度分辨率直接砍半,
#    而本数据集有**大量直接以颜色/色调/纹理为编辑目标**的任务:
#    color_and_material_change(81.9万对) / global_lighting_control(59.6万对) /
#    exposure_and_color_correction(37.0万对) / environment_and_weather_simulation
#    (92.0万对) / style_transfer(142.5万对)，合计约413万对(35.8%)。
#    GT自己带色度模糊，等于要求模型学一个"改颜色但颜色是糊的"的自相矛盾目标，
#    监督信号会被直接污染;
# 3) 分辨率普遍偏大(编辑后图短边最小768、1024x1024占58.9%)，
#    色度下采样的损失在大图上更容易被下游的bucket resize放大;
# 4) 参考图与编辑后图用完全相同的编码参数，避免两条编码链路引入
#    "参考图多一层压缩"这种参考图/GT不对称的伪偏差
#    (注意参考图上游本来就是jpg，这里是二次编码，用高质量档能把二次损失压到最小)。
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

# 文本层各类不合格样本对的实测精确条数，解析阶段逐项硬对账。
# 判定顺序严格按下面process_single_annotation_file里的顺序执行
# (annotation_dir_name_not_match -> unknown_subset ->
#  subset_prompt_resolution_type_not_match -> skip_tail_task ->
#  empty_caption -> null_like_caption -> no_word_char_caption ->
#  too_short_caption -> too_long_caption -> invalid_placeholder_caption ->
#  invalid_annotation_image_shape -> different_aspect_ratio ->
#  missing_image -> invalid_save_image_name)，
# 换顺序会让这些数字互相搬家，所以顺序不能改。
#
# 【四条指令联合判定口径】empty / null_like / no_word_char / too_short /
# too_long / invalid_placeholder 这六项都是**4条指令里只要有任意一条不合格,
# 就整对丢弃**。实测:
#   empty_caption 234条    : short_zh空118条 + detailed_zh空128条,
#                            两者有12条重叠(同一行两条都空)
#   too_short_caption 28条 : 全部是中文短指令(short_zh长度<4的4条 + ...)
#   too_long_caption 680条 : **全部来自detailed_en**(其余三条超过512的都是0条)
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 234,
    'null_like_caption_count': 0,
    'no_word_char_caption_count': 0,
    'too_short_caption_count': 28,
    'too_long_caption_count': 680,
    'invalid_placeholder_caption_count': 0,
}

# 除了长尾任务丢弃与指令过滤之外剩下的几项丢弃计数的实测值，逐项硬对账。
# 这几项实测都是0(上游017自校验已保证0缺图/0孤儿图/0重名/3617万个文件全部落盘、
# 且sample_id与两张图文件名前缀全库100%一致)，
# 但必须一并算进过滤链路恒等式，否则上游一旦出现缺图/坏行，恒等式会误报
EXPECTED_OTHER_FILTER_COUNT_DICT = {
    'illegal_line_count': 0,
    'annotation_dir_name_not_match_count': 0,
    'unknown_subset_count': 0,
    'subset_prompt_resolution_type_not_match_count': 0,
    'invalid_annotation_image_shape_count': 0,
    'missing_image_count': 0,
    'invalid_save_image_name_count': 0,
}

# 被"参考图与编辑后图长宽比不可对齐"这条规则丢弃的实测条数。
# 在1%容差口径下实测为**0**: 全库长宽比偏差只有5种取值
# (0.00% 7670345 / 0.35% 1152133 / 0.39% 1175432 / 0.45% 734573 /
#  0.46% 810951)，最大0.46%，全部落在1%以内。
# 保留这条链路(而不是因为实测0就删掉)是必须的: 上游哪天换版出现真正的大偏差
# 样本时，这条规则会立刻把形变样本挡在几十小时的解码重编码之前
EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT = 0

# 文本层全部过滤之后、图像解码校验之前的样本对数:
#   12057500 - 513124(长尾任务) - 234(空指令) - 28(过短) - 680(过长)
#   == 11543434
# 全量实测(非抽样)的精确数字，解析阶段硬对账
EXPECTED_VALID_ANNOTATION_COUNT = 11543434

# 152个交叉子集过滤后的实测条数(合计11543434)，解析阶段逐个硬对账。
# 子集名 = <38个任务名> + _ + <prompt类型> + _ + <分辨率类型>。
# 其中有16个子集不足10000对(最少4904对)，
# 这些子集各只会切出1个不满10000对的文件夹，这是允许的
# (每个子集的最后一个文件夹允许不满)
EXPECTED_SET_ANNOTATION_COUNT_DICT = {
    'acg_and_entertainment_enhanced_prompt_random_resolution': 111355,
    'acg_and_entertainment_enhanced_prompt_square_resolution': 106721,
    'acg_and_entertainment_original_prompt_random_resolution': 111352,
    'acg_and_entertainment_original_prompt_square_resolution': 80795,
    'add_or_remove_object_enhanced_prompt_random_resolution': 383478,
    'add_or_remove_object_enhanced_prompt_square_resolution': 360178,
    'add_or_remove_object_original_prompt_random_resolution': 404042,
    'add_or_remove_object_original_prompt_square_resolution': 301114,
    'background_management_enhanced_prompt_random_resolution': 123165,
    'background_management_enhanced_prompt_square_resolution': 117265,
    'background_management_original_prompt_random_resolution': 135405,
    'background_management_original_prompt_square_resolution': 104930,
    'beauty_retouching_enhanced_prompt_random_resolution': 36596,
    'beauty_retouching_enhanced_prompt_square_resolution': 34906,
    'beauty_retouching_original_prompt_random_resolution': 40544,
    'beauty_retouching_original_prompt_square_resolution': 31545,
    'body_reshaping_enhanced_prompt_random_resolution': 5531,
    'body_reshaping_enhanced_prompt_square_resolution': 5812,
    'body_reshaping_original_prompt_random_resolution': 5755,
    'body_reshaping_original_prompt_square_resolution': 4904,
    'color_and_material_change_enhanced_prompt_random_resolution': 216895,
    'color_and_material_change_enhanced_prompt_square_resolution': 209357,
    'color_and_material_change_original_prompt_random_resolution': 221787,
    'color_and_material_change_original_prompt_square_resolution': 171842,
    'complex_instruction_following_enhanced_prompt_random_resolution': 16566,
    'complex_instruction_following_enhanced_prompt_square_resolution': 15666,
    'complex_instruction_following_original_prompt_random_resolution': 16550,
    'complex_instruction_following_original_prompt_square_resolution': 12113,
    'crop_and_composition_enhanced_prompt_random_resolution': 8816,
    'crop_and_composition_enhanced_prompt_square_resolution': 8055,
    'crop_and_composition_original_prompt_random_resolution': 9304,
    'crop_and_composition_original_prompt_square_resolution': 6514,
    'denoise_and_deblur_enhanced_prompt_random_resolution': 26529,
    'denoise_and_deblur_enhanced_prompt_square_resolution': 26285,
    'denoise_and_deblur_original_prompt_random_resolution': 27979,
    'denoise_and_deblur_original_prompt_square_resolution': 21043,
    'detail_restoration_enhanced_prompt_random_resolution': 28547,
    'detail_restoration_enhanced_prompt_square_resolution': 27173,
    'detail_restoration_original_prompt_random_resolution': 28144,
    'detail_restoration_original_prompt_square_resolution': 21385,
    'document_and_education_enhanced_prompt_random_resolution': 6518,
    'document_and_education_enhanced_prompt_square_resolution': 6255,
    'document_and_education_original_prompt_random_resolution': 6633,
    'document_and_education_original_prompt_square_resolution': 5009,
    'ecommerce_and_marketing_enhanced_prompt_random_resolution': 39527,
    'ecommerce_and_marketing_enhanced_prompt_square_resolution': 37666,
    'ecommerce_and_marketing_original_prompt_random_resolution': 45606,
    'ecommerce_and_marketing_original_prompt_square_resolution': 34979,
    'emotion_and_expression_control_enhanced_prompt_random_resolution': 51737,
    'emotion_and_expression_control_enhanced_prompt_square_resolution': 49684,
    'emotion_and_expression_control_original_prompt_random_resolution': 55346,
    'emotion_and_expression_control_original_prompt_square_resolution': 41756,
    'environment_and_style_enhanced_prompt_random_resolution': 17258,
    'environment_and_style_enhanced_prompt_square_resolution': 16683,
    'environment_and_style_original_prompt_random_resolution': 17346,
    'environment_and_style_original_prompt_square_resolution': 12936,
    'environment_and_weather_simulation_enhanced_prompt_random_resolution':
    242495,
    'environment_and_weather_simulation_enhanced_prompt_square_resolution':
    233907,
    'environment_and_weather_simulation_original_prompt_random_resolution':
    251060,
    'environment_and_weather_simulation_original_prompt_square_resolution':
    192399,
    'exposure_and_color_correction_enhanced_prompt_random_resolution': 98717,
    'exposure_and_color_correction_enhanced_prompt_square_resolution': 93664,
    'exposure_and_color_correction_original_prompt_random_resolution': 102080,
    'exposure_and_color_correction_original_prompt_square_resolution': 75487,
    'facial_attribute_editing_enhanced_prompt_random_resolution': 69356,
    'facial_attribute_editing_enhanced_prompt_square_resolution': 65286,
    'facial_attribute_editing_original_prompt_random_resolution': 74404,
    'facial_attribute_editing_original_prompt_square_resolution': 58087,
    'font_and_style_enhanced_prompt_random_resolution': 68694,
    'font_and_style_enhanced_prompt_square_resolution': 68833,
    'font_and_style_original_prompt_random_resolution': 71176,
    'font_and_style_original_prompt_square_resolution': 52271,
    'gaze_and_expression_repair_enhanced_prompt_random_resolution': 18441,
    'gaze_and_expression_repair_enhanced_prompt_square_resolution': 17415,
    'gaze_and_expression_repair_original_prompt_random_resolution': 18775,
    'gaze_and_expression_repair_original_prompt_square_resolution': 14175,
    'global_lighting_control_enhanced_prompt_random_resolution': 157571,
    'global_lighting_control_enhanced_prompt_square_resolution': 151888,
    'global_lighting_control_original_prompt_random_resolution': 163915,
    'global_lighting_control_original_prompt_square_resolution': 122885,
    'group_photo_composition_enhanced_prompt_random_resolution': 13936,
    'group_photo_composition_enhanced_prompt_square_resolution': 12970,
    'group_photo_composition_original_prompt_random_resolution': 15616,
    'group_photo_composition_original_prompt_square_resolution': 11514,
    'hair_editing_enhanced_prompt_random_resolution': 110358,
    'hair_editing_enhanced_prompt_square_resolution': 103696,
    'hair_editing_original_prompt_random_resolution': 118150,
    'hair_editing_original_prompt_square_resolution': 93918,
    'image_outpainting_enhanced_prompt_random_resolution': 38441,
    'image_outpainting_enhanced_prompt_square_resolution': 37269,
    'image_outpainting_original_prompt_random_resolution': 40512,
    'image_outpainting_original_prompt_square_resolution': 30316,
    'layout_and_logo_enhanced_prompt_random_resolution': 28823,
    'layout_and_logo_enhanced_prompt_square_resolution': 27806,
    'layout_and_logo_original_prompt_random_resolution': 32521,
    'layout_and_logo_original_prompt_square_resolution': 23313,
    'logical_reasoning_generation_enhanced_prompt_random_resolution': 9115,
    'logical_reasoning_generation_enhanced_prompt_square_resolution': 8612,
    'logical_reasoning_generation_original_prompt_random_resolution': 9141,
    'logical_reasoning_generation_original_prompt_square_resolution': 6488,
    'matting_and_layer_enhanced_prompt_random_resolution': 36882,
    'matting_and_layer_enhanced_prompt_square_resolution': 35039,
    'matting_and_layer_original_prompt_random_resolution': 39631,
    'matting_and_layer_original_prompt_square_resolution': 29894,
    'multi_image_composition_enhanced_prompt_random_resolution': 22185,
    'multi_image_composition_enhanced_prompt_square_resolution': 20721,
    'multi_image_composition_original_prompt_random_resolution': 22432,
    'multi_image_composition_original_prompt_square_resolution': 15575,
    'pose_and_action_driven_enhanced_prompt_random_resolution': 35524,
    'pose_and_action_driven_enhanced_prompt_square_resolution': 34682,
    'pose_and_action_driven_original_prompt_random_resolution': 36117,
    'pose_and_action_driven_original_prompt_square_resolution': 28608,
    'reference_image_driven_enhanced_prompt_random_resolution': 41919,
    'reference_image_driven_enhanced_prompt_square_resolution': 40039,
    'reference_image_driven_original_prompt_random_resolution': 42767,
    'reference_image_driven_original_prompt_square_resolution': 31001,
    'replace_object_enhanced_prompt_random_resolution': 328965,
    'replace_object_enhanced_prompt_square_resolution': 316114,
    'replace_object_original_prompt_random_resolution': 332622,
    'replace_object_original_prompt_square_resolution': 249656,
    'sketch_and_scribble_control_enhanced_prompt_random_resolution': 25440,
    'sketch_and_scribble_control_enhanced_prompt_square_resolution': 24368,
    'sketch_and_scribble_control_original_prompt_random_resolution': 25655,
    'sketch_and_scribble_control_original_prompt_square_resolution': 18790,
    'spatial_and_geometric_transform_enhanced_prompt_random_resolution': 20393,
    'spatial_and_geometric_transform_enhanced_prompt_square_resolution': 20224,
    'spatial_and_geometric_transform_original_prompt_random_resolution': 21179,
    'spatial_and_geometric_transform_original_prompt_square_resolution': 16042,
    'style_transfer_enhanced_prompt_random_resolution': 376425,
    'style_transfer_enhanced_prompt_square_resolution': 360570,
    'style_transfer_original_prompt_random_resolution': 392256,
    'style_transfer_original_prompt_square_resolution': 295081,
    'super_resolution_enhanced_prompt_random_resolution': 22377,
    'super_resolution_enhanced_prompt_square_resolution': 21382,
    'super_resolution_original_prompt_random_resolution': 22980,
    'super_resolution_original_prompt_square_resolution': 16908,
    'text_modification_and_generation_enhanced_prompt_random_resolution':
    55854,
    'text_modification_and_generation_enhanced_prompt_square_resolution':
    54397,
    'text_modification_and_generation_original_prompt_random_resolution':
    58911,
    'text_modification_and_generation_original_prompt_square_resolution':
    42280,
    'text_removal_and_dewatermark_enhanced_prompt_random_resolution': 37089,
    'text_removal_and_dewatermark_enhanced_prompt_square_resolution': 36451,
    'text_removal_and_dewatermark_original_prompt_random_resolution': 40358,
    'text_removal_and_dewatermark_original_prompt_square_resolution': 28141,
    'viewpoint_transformation_enhanced_prompt_random_resolution': 35410,
    'viewpoint_transformation_enhanced_prompt_square_resolution': 33955,
    'viewpoint_transformation_original_prompt_random_resolution': 34905,
    'viewpoint_transformation_original_prompt_square_resolution': 26436,
    'virtual_try_on_enhanced_prompt_random_resolution': 80295,
    'virtual_try_on_enhanced_prompt_square_resolution': 77205,
    'virtual_try_on_original_prompt_random_resolution': 86328,
    'virtual_try_on_original_prompt_square_resolution': 68598,
}

# 最终产出的子集数，必须与"38个任务 × 2种prompt类型 × 2种分辨率类型"严格
# 一一对应(不多也不少)。实测152个组合全部命中、无空子集
EXPECTED_SAVE_SET_COUNT = 152

# 按ceil(条数/10000)逐子集累加得到的预计文件夹数，落盘后硬对账
EXPECTED_SAVE_FOLDER_COUNT = 1230


def get_set_name(per_task_name, per_prompt_type, per_resolution_type):
    """把上游的edit_task/prompt_type/resolution_type三者合成子集名

    子集名 = <任务名> + _ + <prompt类型> + _ + <分辨率类型>，最终产出152个子集。
    任务名优先用写死的映射表(映射规则一改就会静默把几百万个样本对写进错误子集);
    表里没有的取值退回到mix兜底子集，这条路径只在上游新增edit_task时才会走到，
    且会在check_load_annotation_count里被"unknown task"硬拦下来。
    """
    per_task_set_name = GET_TASK_SET_NAME_DICT.get(
        str(per_task_name).strip(), MIX_SET_NAME)
    per_prompt_type_set_name = GET_PROMPT_TYPE_SET_NAME_DICT.get(
        str(per_prompt_type).strip(), MIX_SET_NAME)
    per_resolution_type_set_name = GET_RESOLUTION_TYPE_SET_NAME_DICT.get(
        str(per_resolution_type).strip(), MIX_SET_NAME)

    return (f'{per_task_set_name}_{per_prompt_type_set_name}'
            f'_{per_resolution_type_set_name}')


def check_skip_tail_task(per_task_name):
    """判定这个edit_task是不是要整体丢弃的长尾任务，返回True表示丢弃

    只保留GET_TASK_SET_NAME_DICT里的38个任务(每个都在22008对以上、
    合计覆盖95.7444%)，其余7074个长尾取值(合计513124对)连带其样本整体丢弃。
    这个判定必须排在所有指令与图像过滤之前，
    所以后面每一项过滤计数都只统计38个保留任务里的样本。
    """
    return str(per_task_name).strip() not in GET_TASK_SET_NAME_DICT


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    这个数据集每个编辑对只有编辑前原图这一张参考图(上游reference_image_num
    全库12057500行恒为1、reference_image_path_list恒为长度1的list)，
    所以152个子集全部恒为1。
    保留这个函数是为了和003/004/013/015/016的收尾自校验口径保持一致。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令

    这个数据集恒为1张参考图、不引入任何额外的视觉条件图，
    指令里既没有占位符(实测0条)也没有"the reference image"这类自然语言指代
    (只有1张参考图、指令从不指代它)，所以这里只做strip，不做任何占位符改写。
    4条指令都走这同一个函数，保证4条的归一化口径完全一致。
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
    实测本数据集0条命中(4条指令都是)，只作防御性拦截。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().strip(
        CAPTION_STRIP_CHAR)

    return per_ti2i_caption in NULL_LIKE_CAPTION_LIST


def get_annotation_text_value(per_annotation, per_key_name):
    """从标注里取一个文本字段并strip，取不到或类型不对时返回空串

    上游文本列实测全库无null、几乎无空串(只有short_zh 118条 / detailed_zh 128条)，
    这里的类型兼容只做防御性拦截。
    """
    per_text_value = per_annotation.get(per_key_name, '')
    if isinstance(per_text_value, (list, tuple)):
        per_text_value = per_text_value[0] if len(per_text_value) > 0 else ''
    if not isinstance(per_text_value, str):
        per_text_value = ''

    return per_text_value.strip()


def get_annotation_image_shape(per_image_shape):
    """把标注里的宽高字段规整成[宽, 高]，不合法时返回None

    上游reference_image_shape / edited_image_shape实测12057500行全是2元正int
    list，这里的类型校验只做防御性拦截。
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
    """判定参考图与编辑后图的长宽比是否"可对齐"，返回True表示可以resize对齐

    两个入参都是[宽, 高]。判定分两步(顺序不能换):
    1. 先用Fraction最简分数比做**精确相等**判定(与003/004/013/015/016完全一致
       的零容差口径)。实测全库有8012736对(66.5%)走这条分支;
    2. 精确不相等时再判**相对偏差是否在MAX_ASPECT_RATIO_TOLERANCE(1%)以内**。
       实测全库有4044764对(33.5%)走这条分支，它们的偏差只有0.35%~0.46%,
       成因是上游把编辑后图resize到了16倍数的bucket(不是裁剪)，
       已用SIFT + RANSAC完整仿射估计证实"参考图resize到编辑后图尺寸之后与
       编辑后图达到亚像素级(0.36px)逐像素对齐"、并排除了裁剪假设,
       详见文件头MAX_ASPECT_RATIO_TOLERANCE处的完整论证。
    偏差用"长宽比之比减1"的绝对值度量，对宽高互换是对称的。
    """
    if not per_reference_image_shape or not per_edited_image_shape:
        return False

    # 第一步: 精确相等(零容差)，与其它ti2i脚本完全同口径
    if Fraction(per_reference_image_shape[0],
                per_reference_image_shape[1]) == Fraction(
                    per_edited_image_shape[0], per_edited_image_shape[1]):
        return True

    # 第二步: 1%容差。本数据集特有，实测最大偏差0.46%
    per_reference_image_aspect_ratio = per_reference_image_shape[
        0] / per_reference_image_shape[1]
    per_edited_image_aspect_ratio = per_edited_image_shape[
        0] / per_edited_image_shape[1]
    per_aspect_ratio_deviation = abs(per_reference_image_aspect_ratio /
                                     per_edited_image_aspect_ratio - 1)

    return per_aspect_ratio_deviation <= MAX_ASPECT_RATIO_TOLERANCE


def get_long_side_aligned_shape(per_reference_image_shape,
                                per_edited_image_shape):
    """按长边与编辑后图长边对齐，算出参考图应该被resize到的[宽, 高]

    只有reference_image[k>=1](第二张视觉条件图/主体图/物体图)在长宽比与编辑后图
    不可对齐时才会走到这里: 这类参考图是独立主体/材质样例，本来就不要求与编辑后图
    像素对齐，硬resize到编辑后图尺寸会把画面拉伸变形，所以改成保持它自己的长宽比、
    只把长边缩放到与编辑后图长边相同(等比缩放、零形变、不裁剪)。
    本数据集恒为1张参考图，这个函数不会被走到，只为与003/004/016口径一致而保留。
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
      长宽比与编辑后图可对齐 -> 一律resize到编辑后图尺寸;
      长宽比不可对齐且是reference_image[0](编辑前原图) -> 返回None，整对丢弃;
      长宽比不可对齐且是reference_image[k>=1]          -> 按长边对齐resize。
    """
    if check_same_image_aspect_ratio(per_reference_image_shape,
                                     per_edited_image_shape):
        return list(per_edited_image_shape)

    # 第一张参考图是编辑前原图，它必须与编辑后图像素对齐(这是编辑类样本的根本要求)，
    # 长宽比不可对齐就无法在不形变的前提下对齐，整对丢弃
    if per_reference_image_index == 0:
        return None

    return get_long_side_aligned_shape(per_reference_image_shape,
                                       per_edited_image_shape)


def get_load_image_name_prefix(per_image_relative_path,
                               per_load_image_name_suffix):
    """从上游图像相对路径取出样本主干(去掉扩展名与_source/_edit后缀)

    上游图像名形如0_0_0_source.jpg / 0_0_0_edit.png，
    扩展名实测参考图恒为.jpg、编辑后图恒为.png，但这里仍然先splitext再去后缀，
    上游哪天换格式也不会取错。
    注意上游sample_id里可能带"auto"(如1114_0_auto)，上游017的注释明确警告过
    "**绝不能按下划线切分取数字**"，所以这里只按尾部后缀切、不做任何下划线切分。
    后缀对不上时返回空串，由调用方计入invalid_save_image_name。
    """
    per_image_name_prefix = os.path.splitext(
        os.path.basename(per_image_relative_path))[0].strip().lower()

    if not per_image_name_prefix.endswith(per_load_image_name_suffix):
        return ''

    return per_image_name_prefix[:-len(per_load_image_name_suffix)]


def process_single_annotation_file(annotation_file_pair):
    """解析单个标注文件，组装图像编辑对(参考图+编辑后图+4条编辑指令)的列表

    这一步只做纯文本层面 + 标注里已有宽高的过滤，判定顺序**严格固定**为:
      标注目录名与行内subset_name不一致 -> 未知子集 ->
      行内prompt_type/resolution_type与子集名不自洽 -> 整体丢弃的长尾任务 ->
      4条指令任一为空 -> 任一是null字面量 -> 任一无文字字符 ->
      任一过短 -> 任一过长 -> 任一含坏占位符 ->
      标注宽高非法 -> **参考图与编辑后图长宽比不可对齐** ->
      缺图 -> 保存名非法
    换顺序会让EXPECTED_INVALID_CAPTION_COUNT_DICT、
    EXPECTED_OTHER_FILTER_COUNT_DICT与EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT里
    那些数字互相搬家，所以顺序不能改。

    指令的六项过滤(空/null/无文字/过短/过长/坏占位符)都是**4条联合判定**:
    4条指令只要有任意一条不合格就整对丢弃，因为落盘的json里4条指令都要写，
    留下一条坏的等于把坏数据混进训练集。

    长宽比这条只读标注里的reference_image_shape / edited_image_shape就能判、
    **完全不解图**(按015.0的口径)，所以能把形变样本挡在几十小时的解码重编码之前。
    图像本身的解码校验和分辨率过滤留到后面多进程里做。

    图像是否存在这里用os.path.isfile逐个判，没有按目录缓存os.listdir:
    上游图像按<subset>/<batch名>/分到616个目录里、每个目录约2万×3个文件，
    缓存一个目录的文件名集合就要几MB，32个worker叠起来反而更亏;
    而且每张图后面都要真解码一遍，真缺图在解码阶段一定会被判出来,
    这里的存在性判定只是为了把"缺图"和"图坏"分开统计。
    """

    per_annotation_path, per_dir_subset_name, root_image_path = annotation_file_pair

    per_annotation_file_name_prefix = os.path.splitext(
        os.path.basename(per_annotation_path))[0].strip()

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
    unknown_subset_count = 0
    subset_prompt_resolution_type_not_match_count = 0
    skip_tail_task_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_annotation_image_shape_count = 0
    different_aspect_ratio_count = 0
    missing_image_count = 0
    invalid_save_image_name_count = 0
    subset_annotation_count_dict = {}
    task_annotation_count_dict = {}
    skip_tail_task_name_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    for per_annotation in annotation_list:
        if not isinstance(per_annotation, dict):
            illegal_line_count += 1
            print('2222', per_annotation_path)
            continue

        per_subset_name = get_annotation_text_value(
            per_annotation, ANNOTATION_SUBSET_NAME_KEY_NAME)
        per_archive_name = get_annotation_text_value(
            per_annotation, ANNOTATION_ARCHIVE_NAME_KEY_NAME)
        per_task_name = get_annotation_text_value(
            per_annotation, ANNOTATION_TASK_NAME_KEY_NAME)
        per_prompt_type = get_annotation_text_value(
            per_annotation, ANNOTATION_PROMPT_TYPE_KEY_NAME)
        per_resolution_type = get_annotation_text_value(
            per_annotation, ANNOTATION_RESOLUTION_TYPE_KEY_NAME)

        # 按上游原值统计"过滤之前"的分布，与实测ground truth硬对账。
        # 这里统计的是每一行的原值(含要丢弃的长尾任务)，所以必须在任何过滤之前累加
        subset_annotation_count_dict[
            per_subset_name] = subset_annotation_count_dict.get(
                per_subset_name, 0) + 1
        task_annotation_count_dict[
            per_task_name] = task_annotation_count_dict.get(per_task_name,
                                                            0) + 1

        # 行内子集名与分片名必须和这个标注文件所在的目录/文件名一致:
        # 不一致说明上游产物被搬动过，继续跑会把样本对写进错误子集(实测0条)
        if per_subset_name != per_dir_subset_name or per_archive_name != per_annotation_file_name_prefix:
            annotation_dir_name_not_match_count += 1
            print('3333', per_annotation_path, per_subset_name,
                  per_archive_name)
            continue

        # 上游新增子集目录时会走到这里(实测0条)
        if per_subset_name not in GET_SUBSET_PROMPT_RESOLUTION_TYPE_DICT:
            unknown_subset_count += 1
            print('3333', per_annotation_path, per_subset_name)
            continue

        # 行内的prompt_type/resolution_type必须与子集名拆出来的那两段严格一致:
        # 子集名的第2、3段就取自这两个字段，不自洽会让样本对写进错误子集(实测0条)
        per_expect_prompt_type, per_expect_resolution_type = GET_SUBSET_PROMPT_RESOLUTION_TYPE_DICT[
            per_subset_name]
        if per_prompt_type != per_expect_prompt_type or per_resolution_type != per_expect_resolution_type:
            subset_prompt_resolution_type_not_match_count += 1
            print('3333', per_annotation_path, per_prompt_type,
                  per_resolution_type)
            continue

        # 7074个长尾任务整体丢弃(见check_skip_tail_task)。
        # 必须排在所有指令与图像过滤之前，否则那些计数会被长尾样本污染
        if check_skip_tail_task(per_task_name):
            skip_tail_task_count += 1
            skip_tail_task_name_count_dict[
                per_task_name] = skip_tail_task_name_count_dict.get(
                    per_task_name, 0) + 1
            continue

        per_set_name = get_set_name(per_task_name, per_prompt_type,
                                    per_resolution_type)
        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)

        # 4条指令按ANNOTATION_CAPTION_KEY_NAME_LIST的顺序取出
        # (简短英文 -> 简短中文 -> 详细英文 -> 详细中文)，
        # 这个顺序与SAVE_TI2I_CAPTION_KEY_NAME_LIST严格一一对应
        per_ti2i_caption_list = [
            get_annotation_text_value(per_annotation, per_caption_key_name)
            for per_caption_key_name in ANNOTATION_CAPTION_KEY_NAME_LIST
        ]

        # 【以下六项指令过滤全部是4条联合判定: 任一条不合格就整对丢弃】
        # 空指令、全空格指令视为不合格图像编辑对
        # (实测234条: short_zh空118条 + detailed_zh空128条，有12条重叠)
        if any(not per_ti2i_caption
               for per_ti2i_caption in per_ti2i_caption_list):
            empty_caption_count += 1
            continue

        # null字面量与"不做任何修改"这类无意义指令同样丢弃(实测0条，只作防御)
        if any(
                check_null_like_caption(per_ti2i_caption)
                for per_ti2i_caption in per_ti2i_caption_list):
            null_like_caption_count += 1
            print('3333', per_annotation_path, per_ti2i_caption_list[0][:50])
            continue

        # 只剩标点、没有任何数字/字母/汉字的指令也丢弃(实测0条，只作防御)
        if any(not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption)
               for per_ti2i_caption in per_ti2i_caption_list):
            no_word_char_caption_count += 1
            print('3333', per_annotation_path, per_ti2i_caption_list[0][:50])
            continue

        # 过短指令视为不合格图像编辑对(实测28条，全部是中文短指令)
        if any(
                len(per_ti2i_caption) < MIN_CAPTION_LENGTH
                for per_ti2i_caption in per_ti2i_caption_list):
            too_short_caption_count += 1
            continue

        # 过长指令同样视为不合格图像编辑对
        # (实测680条，**全部来自detailed_en**，其余三条超过512的都是0条)
        if any(
                len(per_ti2i_caption) > MAX_CAPTION_LENGTH
                for per_ti2i_caption in per_ti2i_caption_list):
            too_long_caption_count += 1
            continue

        # 本数据集的指令不需要任何占位符改写，这里只做strip，
        # 写进json的一定是归一化后的指令。
        # 归一化放在长度过滤之后、占位符校验之前: 归一化只做strip，
        # 而上面取值时已经strip过，所以这里不会改变长度、两处口径完全一致
        per_ti2i_caption_list = [
            get_normalized_ti2i_caption(per_ti2i_caption)
            for per_ti2i_caption in per_ti2i_caption_list
        ]

        # 占位符编号与参考图数量不自洽的指令也丢弃。
        # 本数据集恒1张参考图，即4条指令里都不允许出现任何占位符(实测0条)
        if any(
                check_invalid_caption(per_ti2i_caption,
                                      per_expect_reference_image_num)
                for per_ti2i_caption in per_ti2i_caption_list):
            invalid_placeholder_caption_count += 1
            print('3333', per_annotation_path, per_ti2i_caption_list[0][:100])
            continue

        # 【参考图与编辑后图必须能对齐到同一尺寸(第一层，只读标注不解图)】
        # 长宽比可对齐的resize到编辑后图尺寸，不可对齐的整对丢弃。
        # 本数据集在1%容差口径下实测丢0条(最大偏差0.46%)，
        # 但这条链路必须在: 上游换版时它能把形变样本挡在解码重编码之前
        per_annotation_reference_image_shape = get_annotation_image_shape(
            per_annotation.get(ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME,
                               None))
        per_annotation_edited_image_shape = get_annotation_image_shape(
            per_annotation.get(ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME, None))

        # 两个宽高字段实测12057500行全非空全合法，取不到只可能是上游规格变了
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
        # 这里仍然按list遍历，保证与003/004/016的多参考图写法完全同构
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
        # 上游图像名主干就是sample_id(形如0_0_0 / 1114_0_auto)，
        # 而**实测(subset_name, sample_id)全库12057500个组合0重名**、
        # 且"prompt类型 × 分辨率类型"与上游4个子集是一一对应的双射，
        # 所以"子集名 + sample_id"天然全局唯一，不需要再往保存名里塞batch。
        # 这里从两张图的文件名现取主干(而不是直接用sample_id字段)，
        # 再和标注里的sample_id/sample_key三方交叉比对，
        # 任何一处对不上都说明上游成员错位，整对丢弃(实测0条)
        per_sample_id = get_annotation_text_value(
            per_annotation, ANNOTATION_SAMPLE_ID_KEY_NAME).lower()
        per_edited_image_name_prefix = get_load_image_name_prefix(
            per_edited_image_relative_path, LOAD_EDITED_IMAGE_NAME_SUFFIX)
        per_reference_image_name_prefix = get_load_image_name_prefix(
            per_reference_image_relative_path_list[0],
            LOAD_REFERENCE_IMAGE_NAME_SUFFIX)

        per_sample_key = get_annotation_text_value(
            per_annotation, ANNOTATION_SAMPLE_KEY_KEY_NAME)
        per_expect_sample_key = f'{per_subset_name}/{per_archive_name}/{per_sample_id}'

        if not VALID_LOAD_SAMPLE_ID_PATTERN.match(
                per_sample_id
        ) or not VALID_LOAD_ARCHIVE_NAME_PATTERN.match(
                per_archive_name
        ) or per_edited_image_name_prefix != per_sample_id or per_reference_image_name_prefix != per_sample_id or per_sample_key != per_expect_sample_key:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_sample_id,
                  per_edited_image_name_prefix,
                  per_reference_image_name_prefix, per_sample_key)
            continue

        # 保存图像名统一全小写，形如
        # conceptedit_style_transfer_enhanced_prompt_random_resolution_1140000_0_0_edited.jpg
        per_save_image_name_prefix = (f'{DATASET_NAME}_{per_set_name}_'
                                      f'{per_sample_id}')
        per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
        per_save_reference_image_name_list = [
            f'{per_save_image_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}'
        ]
        # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的前缀
        # (即带_edited那一段)，和003/004/016的写法保持一致，
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

        # 4条指令一起带下去，落盘时写成两个字典
        # (见文件头SAVE_TI2I_CAPTION_KEY_NAME的注释)
        per_ti2i_caption_dict = {
            per_save_caption_key_name: per_ti2i_caption
            for per_save_caption_key_name, per_ti2i_caption in zip(
                SAVE_TI2I_CAPTION_KEY_NAME_LIST, per_ti2i_caption_list)
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
        subset_annotation_count_dict,
        task_annotation_count_dict,
        skip_tail_task_name_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        annotation_dir_name_not_match_count,
        unknown_subset_count,
        subset_prompt_resolution_type_not_match_count,
        skip_tail_task_count,
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
    """收集上游全部标注文件，返回[标注文件任务列表, 每个子集的标注文件数]

    上游标注按<subset_name>一级分目录、组内按batch名分文件
    (实测4个子集下共616个jsonl)，这里按"标注文件"这一粒度出任务，
    正好能把多进程铺满。
    子集目录名一并带给worker，用于和行内的subset_name交叉校验。
    """
    root_image_path = os.path.join(root_dataset_path,
                                   *LOAD_IMAGE_DIR_NAME_LIST)

    load_annotation_dir_path = os.path.join(root_dataset_path,
                                            LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    subset_annotation_file_count_dict = {}
    for per_subset_name in sorted(os.listdir(load_annotation_dir_path)):
        per_subset_dir_path = os.path.join(load_annotation_dir_path,
                                           per_subset_name)
        if not os.path.isdir(per_subset_dir_path):
            continue

        per_subset_annotation_file_name_list = sorted([
            per_annotation_file_name
            for per_annotation_file_name in os.listdir(per_subset_dir_path) if
            per_annotation_file_name.endswith(LOAD_ANNOTATION_FILE_NAME_SUFFIX)
        ])

        for per_annotation_file_name in per_subset_annotation_file_name_list:
            annotation_file_pair_list.append([
                os.path.join(per_subset_dir_path, per_annotation_file_name),
                per_subset_name,
                root_image_path,
            ])

        subset_annotation_file_count_dict[per_subset_name] = len(
            per_subset_annotation_file_name_list)

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list, subset_annotation_file_count_dict


def get_all_edit_annotation_pair(root_dataset_path):
    """按标注文件粒度多进程组装全部图像编辑对的列表

    上游有616个标注文件、合计12057500行，逐行还要判2张图像文件是否存在，
    所以这里按标注文件开多进程解析，最后按保存的编辑后图像名统一排序。
    """
    annotation_file_pair_list, subset_annotation_file_count_dict = get_all_annotation_file_pair(
        root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list),
          'annotation subset:', len(subset_annotation_file_count_dict))

    total_annotation_count = 0
    illegal_line_count = 0
    annotation_dir_name_not_match_count = 0
    unknown_subset_count = 0
    subset_prompt_resolution_type_not_match_count = 0
    skip_tail_task_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_annotation_image_shape_count = 0
    different_aspect_ratio_count = 0
    missing_image_count = 0
    invalid_save_image_name_count = 0
    subset_annotation_count_dict = {}
    task_annotation_count_dict = {}
    skip_tail_task_name_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            for per_subset_name, per_subset_count in per_load_result[1].items(
            ):
                subset_annotation_count_dict[
                    per_subset_name] = subset_annotation_count_dict.get(
                        per_subset_name, 0) + per_subset_count
            for per_task_name, per_task_count in per_load_result[2].items():
                task_annotation_count_dict[
                    per_task_name] = task_annotation_count_dict.get(
                        per_task_name, 0) + per_task_count
            for per_task_name, per_task_count in per_load_result[3].items():
                skip_tail_task_name_count_dict[
                    per_task_name] = skip_tail_task_name_count_dict.get(
                        per_task_name, 0) + per_task_count
            for per_set_name, per_set_count in per_load_result[4].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

            total_annotation_count += per_load_result[5]
            illegal_line_count += per_load_result[6]
            annotation_dir_name_not_match_count += per_load_result[7]
            unknown_subset_count += per_load_result[8]
            subset_prompt_resolution_type_not_match_count += per_load_result[9]
            skip_tail_task_count += per_load_result[10]
            empty_caption_count += per_load_result[11]
            null_like_caption_count += per_load_result[12]
            no_word_char_caption_count += per_load_result[13]
            too_short_caption_count += per_load_result[14]
            too_long_caption_count += per_load_result[15]
            invalid_placeholder_caption_count += per_load_result[16]
            invalid_annotation_image_shape_count += per_load_result[17]
            different_aspect_ratio_count += per_load_result[18]
            missing_image_count += per_load_result[19]
            invalid_save_image_name_count += per_load_result[20]

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        subset_annotation_file_count_dict,
        subset_annotation_count_dict,
        task_annotation_count_dict,
        skip_tail_task_name_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        annotation_dir_name_not_match_count,
        unknown_subset_count,
        subset_prompt_resolution_type_not_match_count,
        skip_tail_task_count,
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
        subset_annotation_file_count_dict, subset_annotation_count_dict,
        task_annotation_count_dict, skip_tail_task_name_count_dict,
        set_annotation_count_dict, total_annotation_count,
        total_annotation_file_count, valid_annotation_count,
        skip_tail_task_count, different_aspect_ratio_count,
        invalid_caption_count_dict, other_filter_count_dict,
        edit_annotation_pair_list):
    """解析完标注后按子集/任务/交叉子集三级硬对账，并检查保存图像名是否唯一

    上游标注是017一步跑出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    交叉子集级对账能额外拦住"某个任务被映射进错误子集"这种总数级对账
    看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对。
    """
    check_error_message_list = []

    # 上游标注文件数与逐子集文件数: 对不上说明上游017没跑完或产物被改动过
    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

    for per_subset_name in sorted(subset_annotation_file_count_dict.keys()):
        if per_subset_name not in EXPECTED_SUBSET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown annotation subset dir {per_subset_name}')

    for per_subset_name in sorted(
            EXPECTED_SUBSET_ANNOTATION_COUNT_DICT.keys()):
        if per_subset_name not in subset_annotation_file_count_dict:
            check_error_message_list.append(
                f'missing annotation subset dir {per_subset_name}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    # 上游4个子集的原始行数逐项硬对账
    for per_subset_name in sorted(subset_annotation_count_dict.keys()):
        if per_subset_name not in EXPECTED_SUBSET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown subset {per_subset_name}')
            continue

        per_expect_subset_annotation_count = EXPECTED_SUBSET_ANNOTATION_COUNT_DICT[
            per_subset_name]
        if subset_annotation_count_dict[
                per_subset_name] != per_expect_subset_annotation_count:
            check_error_message_list.append(
                f'{per_subset_name} subset annotation count not match '
                f'{subset_annotation_count_dict[per_subset_name]} != '
                f'{per_expect_subset_annotation_count}')

    for per_subset_name in sorted(
            EXPECTED_SUBSET_ANNOTATION_COUNT_DICT.keys()):
        if per_subset_name not in subset_annotation_count_dict:
            check_error_message_list.append(
                f'missing subset {per_subset_name}')

    # 4个子集原始行数之和必须刚好等于总行数，一行都不能漏归类
    if sum(EXPECTED_SUBSET_ANNOTATION_COUNT_DICT.values()
           ) != total_annotation_count:
        check_error_message_list.append(
            f'subset annotation count not self consistent '
            f'{sum(EXPECTED_SUBSET_ANNOTATION_COUNT_DICT.values())} != '
            f'{total_annotation_count}')

    # 38个保留任务的原始条数逐项硬对账(长尾任务单独对账，见下面)
    for per_task_name in sorted(task_annotation_count_dict.keys()):
        if per_task_name in skip_tail_task_name_count_dict:
            continue

        if per_task_name not in EXPECTED_TASK_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown task {per_task_name[:50]}')
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

    # 38个保留任务原始条数 + 长尾任务丢弃条数必须刚好等于总行数，一行都不能漏归类
    if sum(EXPECTED_TASK_ANNOTATION_COUNT_DICT.values()
           ) + skip_tail_task_count != total_annotation_count:
        check_error_message_list.append(
            f'task annotation count not self consistent '
            f'{sum(EXPECTED_TASK_ANNOTATION_COUNT_DICT.values())} + '
            f'{skip_tail_task_count} != {total_annotation_count}')

    # 被整体丢弃的长尾任务的**取值个数**与**条数**都要硬对账，
    # 守住"丢弃范围没被改动过"这条口径
    if len(skip_tail_task_name_count_dict) != EXPECTED_SKIP_TAIL_TASK_COUNT:
        check_error_message_list.append(
            f'skip tail task name count not match '
            f'{len(skip_tail_task_name_count_dict)} != '
            f'{EXPECTED_SKIP_TAIL_TASK_COUNT}')

    if skip_tail_task_count != EXPECTED_SKIP_TAIL_TASK_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'skip tail task annotation count not match '
            f'{skip_tail_task_count} != '
            f'{EXPECTED_SKIP_TAIL_TASK_ANNOTATION_COUNT}')

    # 丢弃名单与保留映射表不允许有交集
    for per_task_name in sorted(skip_tail_task_name_count_dict.keys()):
        if per_task_name in GET_TASK_SET_NAME_DICT:
            check_error_message_list.append(
                f'skip tail task {per_task_name[:50]} also in task set name dict'
            )

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
    per_all_filter_count = (skip_tail_task_count +
                            different_aspect_ratio_count +
                            sum(invalid_caption_count_dict.values()) +
                            sum(other_filter_count_dict.values()))
    if total_annotation_count - per_all_filter_count != valid_annotation_count:
        check_error_message_list.append(
            f'annotation filter count not self consistent '
            f'{total_annotation_count} - {per_all_filter_count} != '
            f'{valid_annotation_count}')

    # 152个交叉子集逐个硬对账
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

    # 保留下来的子集集合必须与期望表严格一一对应，不允许多出任何一个子集
    # (含mix兜底子集的任何一段出现就说明上游新增了取值)
    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in EXPECTED_SET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown save set {per_set_name}')
        if MIX_SET_NAME in per_set_name.split('_'):
            check_error_message_list.append(
                f'save set fall back to mix {per_set_name}')

    if len(set_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

    # 保存的编辑后图像名必须全局唯一。
    # 实测(subset_name, sample_id)全库0重名、而"prompt类型 × 分辨率类型"与上游
    # 4个子集一一对应，所以"子集名 + sample_id"天然唯一;
    # 但撞名会让两个样本对在磁盘和json里互相覆盖、事后看不出少了多少对，
    # 所以这里必须在落盘前再兜一道
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
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把P图/CMYK图静默转成3通道、
    # 把RGBA图静默丢掉alpha通道，
    # 所以必须先用PIL读原始mode才能把这些图判出来。
    # 实测本数据集参考图恒为JPEG/RGB、编辑后图恒为PNG/RGB、0张带alpha，
    # 这道校验只作兜底
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

    # 检查图像短边(实测编辑后图短边最小768、参考图最小512，全库0条命中)
    if min(per_image_h, per_image_w) < MIN_IMAGE_SHORT_SIDE:
        print('6666', per_image_path, per_image_w, per_image_h)
        return None

    # 检查图像宽高比，取长短边之比，宽高比大于8和小于1/8这两种极端样本一起判掉
    # (实测编辑后图最大宽高比1.7708，全库0条命中)
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
      'different_aspect_ratio' -> reference_image[0]与编辑后图长宽比不可对齐
                                  (整对丢弃)，第三项为None。
                                  带上子集名是为了在主流程里逐子集统计

    这是长宽比过滤的**第二层**(兜底): 第一层已经在文本解析阶段用标注里的宽高
    把不可对齐的样本挡掉了，这里拿真解码出来的shape再判一次，
    防止上游标注记的宽高与磁盘上的实际图像不一致(实测抽样1200张100%一致)。

    短边和宽高比的过滤按方案只以编辑后图像为准判定，参考图只要求能正常解码且
    mode命中白名单(实测编辑后图短边最小768、最大宽高比1.7708，
    极端样本预期为0，这两个阈值只作兜底)。

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
        # 只有reference_image[0]长宽比不可对齐才会拿到None，此时整对丢弃
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
    本数据集产出152个子集，其中136个在1万对以上会切出多个满10000对的文件夹、
    另外16个(body_reshaping / document_and_education /
    logical_reasoning_generation / crop_and_composition 这4个任务
    各自的4个交叉子集)各只有1个不满的文件夹，实测合计1230个文件夹。
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
    reference_image[0]一定是编辑后图尺寸(长宽比不可对齐的样本对已经在那一步
    整对丢弃了)，reference_image[k>=1]是编辑后图尺寸或长边对齐后的尺寸。
    本数据集实测有12.5%的样本对两张图尺寸完全相同(这时resize是恒等操作、
    会走下面那个"只在尺寸真的不一样时才resize"的短路，一次多余的重采样都不会做)，
    其余87.5%会真的做一次resize。

    上游编辑后图全部是PNG无损、参考图是JPEG，这里统一重编码成jpg。
    编码参数显式用SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST(质量97 + 色度4:4:4)，
    而不是cv2的默认值(质量95 + 色度4:2:0): 编辑后图是无损源，
    每一次重编码都是从无损到有损的第一次损失; 而且本数据集有大量直接以
    颜色/色调/纹理为编辑目标的任务(约413万对、占35.8%)，
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
    instruction与detailed_en逐字相同(纯冗余)、
    original_simple_caption是源图的整图描述(不是编辑指令)、
    recaption_prompt_en/zh只有14.75%非空且含整图描述、
    edit_category/edit_sub_category/edit_detail粒度不合适(已被38个任务蕴含)、
    overall_vqa_score/keep/wrong_count/vqa_*/not_passed_*按方案不做质量过滤、
    sample_key/sample_id/subset_name/archive_name/annotation_path
    只用于拼保存名与交叉校验(信息已内含在保存名与子集名里)、
    prompt_type/resolution_type已编进子集名、
    reference_image_num恒为1(改由list长度现算)、
    reference_image_shape/edited_image_shape只用于长宽比预筛
    (写json的宽高一律取自实际写盘数组的shape)、
    reference_image_suffix/edited_image_suffix恒为.jpg/.png、
    dataset_task_type恒为image_edit。
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

        # 4个length都直接取即将写进json的那4个字符串的长度，
        # 保证记录的长度和指令永远自洽(4个字符串都已strip并归一化过)。
        # 两个字典都按SAVE_TI2I_CAPTION_KEY_NAME_LIST的顺序写
        # (简短英文 -> 简短中文 -> 详细英文 -> 详细中文)
        folder_annotation_dict[per_folder_name][per_save_edited_image_name] = {
            'reference_image': per_save_reference_image_name_list,
            'edited_image': per_save_edited_image_name,
            'reference_image_num': per_reference_image_num,
            'width': per_edited_image_w,
            'height': per_edited_image_h,
            SAVE_TI2I_CAPTION_KEY_NAME: {
                per_save_caption_key_name:
                per_ti2i_caption_dict[per_save_caption_key_name]
                for per_save_caption_key_name in
                SAVE_TI2I_CAPTION_KEY_NAME_LIST
            },
            SAVE_TI2I_CAPTION_LENGTH_KEY_NAME: {
                per_save_caption_length_key_name:
                len(per_ti2i_caption_dict[per_save_caption_key_name])
                for per_save_caption_key_name, per_save_caption_length_key_name
                in zip(SAVE_TI2I_CAPTION_KEY_NAME_LIST,
                       SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST)
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
    """复检单条标注里的4条指令，返回错误信息列表

    本数据集的ti2i_caption与ti2i_caption_length都是字典(与016同方案，
    只是本数据集是4条而不是2条)，所以这里要比单语数据集多校验一层结构:
    1. 两个value都必须是dict，且key集合严格等于约定的两个列表;
    2. 4条指令都必须是字符串、都不允许是null字面量、都必须含有文字字符;
    3. 4条指令的长度都必须在[MIN, MAX]区间内(与过滤阶段的联合判定口径一致);
    4. 4条指令都不允许含[Vn*]占位符(参考图数量恒为1，即N为0);
    5. 4个length都必须等于对应字符串的实际len()。
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

    # 4条指令走完全相同的一套复检，保证4条口径一致
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


def check_single_save_folder(folder_check_pair, save_dataset_path):
    """校验单个文件夹: 文件夹容量、json与磁盘一一对应、图像名与4条指令合规

    每个子集除最后一个文件夹外都必须是满10000对，json里的每个key都必须在磁盘上有
    对应的样本对文件夹且文件恰好等于编辑后图像 + 所有参考图像，磁盘上也不允许有
    json没记录的残留样本对文件夹。另外还要复检4条指令的字典结构与4条指令本身
    (见check_single_save_annotation_caption)。

    CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG为True时还会真解一次落盘后的
    reference_image[0]，硬校验它的真实shape严格等于json里的width/height
    (即等于编辑后图的宽高)。这一条是"第一张参考图与编辑后图尺寸必须一致"
    这个核心不变式的最终验收: 只对账json里的数字是查不出resize有没有真的生效的。
    豁免子集(完全原样落盘)会跳过这一条。

    校验内容与003/004/016完全一致，只是把"按子集串行遍历"改成了"按文件夹开多进程"
    (与013同口径): 本数据集有1230个文件夹、合计约1154万个样本对文件夹，
    每个都要listdir一次，串行跑在NAS上要跑非常久。
    """
    per_set_name, per_folder_name, per_is_set_last_folder = folder_check_pair

    check_error_message_list = []

    per_expect_reference_image_num = get_expect_reference_image_num(
        per_set_name)
    per_exempt_aspect_ratio_align_flag = per_set_name in EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST

    per_set_dir_path = os.path.join(save_dataset_path, per_set_name)
    per_json_path = os.path.join(per_set_dir_path, f'{per_folder_name}.json')
    if not os.path.isfile(per_json_path):
        check_error_message_list.append(f'{per_folder_name} json not exists')

        return [per_folder_name, 0, check_error_message_list]

    try:
        with open(per_json_path, 'r', encoding='UTF-8') as load_json_file:
            per_folder_annotation_dict = json.load(load_json_file)
    except Exception as e:
        check_error_message_list.append(
            f'{per_folder_name} load json failed {e}')

        return [per_folder_name, 0, check_error_message_list]

    # 除每个子集最后一个文件夹外都必须是满10000对
    if not per_is_set_last_folder and len(
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
    ]) if os.path.isdir(per_folder_path) else []
    per_expect_pair_folder_name_list = sorted([
        per_save_edited_image_name.removesuffix('.jpg')
        for per_save_edited_image_name in per_folder_annotation_dict.keys()
    ])
    if per_exist_pair_folder_name_list != per_expect_pair_folder_name_list:
        check_error_message_list.append(
            f'{per_folder_name} pair folder not match {len(per_exist_pair_folder_name_list)} != {len(per_expect_pair_folder_name_list)}'
        )

    for per_save_edited_image_name in sorted(
            per_folder_annotation_dict.keys()):
        per_annotation = per_folder_annotation_dict[per_save_edited_image_name]

        # 每条标注的字段集合必须和约定的七个key严格一致，不能多也不能少
        if sorted(per_annotation.keys()) != sorted(
                SAVE_ANNOTATION_KEY_NAME_LIST):
            check_error_message_list.append(
                f'{per_save_edited_image_name} annotation key not match')
            continue

        if per_save_edited_image_name != per_annotation['edited_image']:
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name not match')
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
        if per_save_edited_image_name != per_save_edited_image_name.lower():
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name not all lower case'
            )
        if not VALID_IMAGE_NAME_PATTERN.match(per_save_edited_image_name):
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name has invalid char'
            )
        if not isinstance(per_annotation['reference_image'], list):
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image not a list')
            continue
        if per_annotation['reference_image_num'] != len(
                per_annotation['reference_image']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image num not match')
        # 本数据集全部152个子集都必须是单参考图
        if per_annotation[
                'reference_image_num'] != per_expect_reference_image_num:
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image num not match set {per_annotation["reference_image_num"]} != {per_expect_reference_image_num}'
            )
        # 参考图名的前缀必须和编辑后图像名同一个前缀、后缀必须是_reference.jpg
        for per_save_reference_image_name in per_annotation['reference_image']:
            if not per_save_reference_image_name.endswith(
                    SAVE_REFERENCE_IMAGE_NAME_SUFFIX):
                check_error_message_list.append(
                    f'{per_save_reference_image_name} reference image name suffix not match'
                )
            if not VALID_IMAGE_NAME_PATTERN.match(
                    per_save_reference_image_name):
                check_error_message_list.append(
                    f'{per_save_reference_image_name} reference image name has invalid char'
                )
            if per_save_reference_image_name.removesuffix(
                    SAVE_REFERENCE_IMAGE_NAME_SUFFIX
            ) != per_save_edited_image_name.removesuffix(
                    SAVE_EDITED_IMAGE_NAME_SUFFIX):
                check_error_message_list.append(
                    f'{per_save_reference_image_name} reference image name prefix not match {per_save_edited_image_name}'
                )
        # 图像宽高必须是正数
        if per_annotation['width'] <= 0 or per_annotation['height'] <= 0:
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

        # 4条指令的字典结构与4条指令本身的复检
        check_error_message_list.extend(
            check_single_save_annotation_caption(per_save_edited_image_name,
                                                 per_annotation))

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
                f'{per_save_pair_folder_name} pair image file not match')
            continue

        # 真解一次落盘后的reference_image[0]，硬校验它的shape严格等于
        # json里的width/height(也就是编辑后图的宽高)。
        # 这一条是"第一张参考图与编辑后图尺寸必须一致"这个核心不变式的最终验收,
        # 同时也是1%长宽比容差这条口径的最终验收: 那4044764对经过各向异性resize
        # 之后必须真的与编辑后图同尺寸。
        # 非豁免子集才校验: 豁免子集是完全原样落盘的，尺寸本来就可以不等
        if CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG and not per_exempt_aspect_ratio_align_flag and len(
                per_annotation['reference_image']) > 0:
            per_check_save_reference_image_path = os.path.join(
                per_pair_folder_path, per_annotation['reference_image'][0])
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
                    f'{per_annotation["width"]} {per_annotation["height"]}')

    return [
        per_folder_name,
        len(per_folder_annotation_dict),
        check_error_message_list,
    ]


def check_save_dataset(save_dataset_path, set_folder_count_dict):
    """全部落盘后的收尾自校验: 文件夹容量、json与磁盘一一对应、图像名与4条指令合规

    1230个文件夹每个都要load一份json、再对里面的每个样本对listdir一次、
    还要真解一次参考图，串行跑在NAS上太久，所以这一步按文件夹粒度开多进程
    (与013同口径)。
    """
    check_error_message_list = []

    folder_check_pair_list = []
    for per_set_name in sorted(set_folder_count_dict.keys()):
        per_set_folder_count = set_folder_count_dict[per_set_name]
        for per_folder_index in range(per_set_folder_count):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            # 每个子集只有最后一个文件夹允许不满10000对
            per_is_set_last_folder = per_folder_index == per_set_folder_count - 1
            folder_check_pair_list.append([
                per_set_name,
                per_folder_name,
                per_is_set_last_folder,
            ])

    total_edit_pair_count = 0
    check_func = partial(check_single_save_folder,
                         save_dataset_path=save_dataset_path)
    with Pool(processes=min(PROCESS_NUM, max(len(folder_check_pair_list),
                                             1))) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_func, folder_check_pair_list),
                                     total=len(folder_check_pair_list)):
            _, per_folder_edit_pair_count, per_check_error_message_list = per_check_result
            total_edit_pair_count += per_folder_edit_pair_count
            check_error_message_list.extend(per_check_error_message_list)

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

    edit_annotation_pair_list, total_annotation_file_count, subset_annotation_file_count_dict, subset_annotation_count_dict, task_annotation_count_dict, skip_tail_task_name_count_dict, set_annotation_count_dict, total_annotation_count, illegal_line_count, annotation_dir_name_not_match_count, unknown_subset_count, subset_prompt_resolution_type_not_match_count, skip_tail_task_count, empty_caption_count, null_like_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, invalid_placeholder_caption_count, invalid_annotation_image_shape_count, different_aspect_ratio_count, missing_image_count, invalid_save_image_name_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          illegal_line_count, annotation_dir_name_not_match_count,
          unknown_subset_count,
          subset_prompt_resolution_type_not_match_count, skip_tail_task_count,
          len(skip_tail_task_name_count_dict), empty_caption_count,
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

    # 除了长尾任务丢弃、指令过滤与长宽比过滤之外剩下的几项丢弃计数。
    # 这几项实测都是0，但必须一并算进过滤链路恒等式，
    # 否则上游一旦出现坏行/缺图，恒等式会误报成"计数不自洽"
    other_filter_count_dict = {
        'illegal_line_count': illegal_line_count,
        'annotation_dir_name_not_match_count':
        annotation_dir_name_not_match_count,
        'unknown_subset_count': unknown_subset_count,
        'subset_prompt_resolution_type_not_match_count':
        subset_prompt_resolution_type_not_match_count,
        'invalid_annotation_image_shape_count':
        invalid_annotation_image_shape_count,
        'missing_image_count': missing_image_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        subset_annotation_file_count_dict, subset_annotation_count_dict,
        task_annotation_count_dict, skip_tail_task_name_count_dict,
        set_annotation_count_dict,
        total_annotation_count, total_annotation_file_count,
        len(edit_annotation_pair_list), skip_tail_task_count,
        different_aspect_ratio_count, invalid_caption_count_dict,
        other_filter_count_dict, edit_annotation_pair_list)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        # 上游标注条数对不上说明上游017没跑完、产物被改动过，
        # 或者上游新增了子集/任务取值，
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
            # reference_image[0]与编辑后图长宽比不可对齐的样本对在这里整对丢弃，
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
          annotation_dir_name_not_match_count, 'unknown subset:',
          unknown_subset_count, 'subset prompt resolution type not match:',
          subset_prompt_resolution_type_not_match_count, 'skip tail task:',
          skip_tail_task_count, 'skip tail task name:',
          len(skip_tail_task_name_count_dict), 'empty caption:',
          empty_caption_count, 'null like caption:', null_like_caption_count,
          'no word char caption:', no_word_char_caption_count,
          'too short caption:', too_short_caption_count, 'too long caption:',
          too_long_caption_count, 'invalid placeholder caption:',
          invalid_placeholder_caption_count, 'invalid annotation image shape:',
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
        'unknown_subset_count': unknown_subset_count,
        'subset_prompt_resolution_type_not_match_count':
        subset_prompt_resolution_type_not_match_count,
        # 7074个长尾edit_task取值被整体丢弃的条数(见check_skip_tail_task)
        'skip_tail_task_count': skip_tail_task_count,
        'skip_tail_task_name_count': len(skip_tail_task_name_count_dict),
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
        # 【本数据集特有口径1】长宽比判定带1%容差(本项目第一个开容差的脚本):
        # 33.5%的样本对长宽比偏差只有0.35%~0.46%(上游把编辑后图resize到了16倍数的
        # bucket)，已用SIFT + RANSAC完整仿射估计证实"参考图各向异性resize到编辑后图
        # 尺寸之后达到亚像素级(0.36px)逐像素对齐"、并排除了裁剪假设，
        # 详见脚本头MAX_ASPECT_RATIO_TOLERANCE处的完整论证
        'max_aspect_ratio_tolerance': MAX_ASPECT_RATIO_TOLERANCE,
        # 【本数据集特有口径2】ti2i_caption与ti2i_caption_length的value都是字典，
        # 且是**4条**指令(中英双语 × 粗细两档)，下游读取时必须显式感知这个结构差异
        'multi_caption_ti2i_caption_flag': True,
        'ti2i_caption_key_name_list': SAVE_TI2I_CAPTION_KEY_NAME_LIST,
        'ti2i_caption_length_key_name_list':
        SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST,
        # 【本数据集特有口径3】4个"天生会改变画面几何"的任务(合计约38.8万对)
        # 按方案全部保留，但它们的参考图与编辑后图**在内容上并不逐像素对应**
        # (viewpoint_transformation最极端: 实测恒等位移p50=79px、24对里14对
        #  SIFT根本匹配不上)。这不是预处理缺陷、是编辑语义本身，
        # 但下游按"宽高比相同"走逐像素RoPE对齐分支时必须显式感知，
        # 详见脚本头GEOMETRY_CHANGE_TASK_SET_NAME_LIST处的实测表
        'geometry_change_task_set_name_list':
        GEOMETRY_CHANGE_TASK_SET_NAME_LIST,
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
        'subset_annotation_file_count_dict': subset_annotation_file_count_dict,
        'subset_annotation_count_dict': subset_annotation_count_dict,
        'task_annotation_count_dict': task_annotation_count_dict,
        'set_annotation_count_dict': set_annotation_count_dict,
        'set_folder_count_dict': set_folder_count_dict,
        'folder_edit_pair_count_dict': folder_edit_pair_count_dict,
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    # 落盘后的子集数与文件夹数也要与实测值硬对账
    if len(set_folder_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'total save set count not match '
            f'{len(set_folder_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')
    if len(folder_edit_pair_count_dict) != EXPECTED_SAVE_FOLDER_COUNT:
        check_error_message_list.append(
            f'total save folder count not match '
            f'{len(folder_edit_pair_count_dict)} != '
            f'{EXPECTED_SAVE_FOLDER_COUNT}')
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/ConceptEdit-12M'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
