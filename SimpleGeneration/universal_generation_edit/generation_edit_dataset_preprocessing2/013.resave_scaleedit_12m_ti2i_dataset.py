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

DATASET_NAME = 'scaleedit'

SAVE_DATASET_DIR_NAME = 'ScaleEdit'

# ==============================================================================
# 【这个数据集只能产出图像编辑数据集，不能产出文生图数据集】
# 上游016解包出来的每一行只有15列，其中唯一的文本字段是edit_instruction(英文编辑指令)，
# 形如"Apply a retro comic book style with bold outlines and flat colors."、
# "Replace the text '火火火火' with '玻璃相框'."。
# 整个数据集没有任何一列是图像内容描述(caption): 既没有编辑前原图的描述、
# 也没有编辑后图的描述。编辑指令只说"要改什么"、不说"整张图是什么"，
# 拿它当t2i的prompt会得到完全错误的图文对(比如"Make it anime-style."配一张椅子图)，
# 所以本数据集只走ti2i这一条链路，不另写t2i脚本。
# 上游016自己也把DATASET_TASK_TYPE写死成image_edit，README的task_categories是
# image-to-image。
#
# 【上游016.unzip_scaleedit_12m_dataset.py的产物规格(实测)】
# ScaleEdit-12M/
# ├── unzip_annotations/<23个子集>/<分片>.jsonl            11142945行(完整编辑对，12G)
# ├── unzip_url_source_annotations/<23个子集>/<分片>.jsonl    702078行(只有源图URL，805M)
# ├── unzip_images/<23个子集>/<分片>/<id>_source.jpg|png
# │                                 /<id>_edited.jpg|png   22987968张(约6.7T)
# └── unzip_check_missing_images.json  上游自校验0错误、0隔离样本、0重名
# 逐子集都满足 行数 == 完整对数 + 只有源图URL的对数，
# 全局 11142945 + 702078 == 11845023，所以这些数字可以直接写死当ground truth硬对账。
# ==============================================================================

# 本脚本只读unzip_annotations这一套标注(11142945行)，
# 它里面每一行的参考图与编辑后图都已经落盘、是可直接训练的完整编辑对。
# 绝不os.walk图像目录: 上游解出约2300万个小文件，扫目录树在NAS上不可接受
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

# 【按方案确认: 702078条"只有源图URL、没有源图字节"的样本对整体丢弃】
# 这批行有编辑后图和编辑指令，但**没有参考图**(上游数据集自身的发布规格:
# 网络抓取的源图只给URL不rehost字节)，构不成图像编辑对，
# 落盘的话reference_image只能是空list、reference_image_num只能是0，
# 会破坏"每个编辑对必有参考图"的口径、下游ti2i_dataset也没法训练。
# 所以本脚本完全不读这个目录，只在resave_check_result.json里把条数记下来上报，
# 等后续按URL+sha256把源图回捞回来之后再单独补一版
LOAD_URL_SOURCE_ANNOTATION_DIR_NAME = 'unzip_url_source_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 上游标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如unzip_images/1.1_style_transfer/style_transfer_0000/123_edited.jpg)，
# 不需要再往前拼任何子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 标注行所属的上游子集目录名(形如1.1_style_transfer)，实测11142945行全非空，
# 且与标注文件所在的目录名严格一致。子集名(即图像编辑任务类型)就由它推导
ANNOTATION_SUBSET_NAME_KEY_NAME = 'subset_name'

# 子集内唯一的样本id(int)，上游已硬校验过"片内连号 + 同一子集内分片区间首尾相接、
# 恰好覆盖0..该子集行数-1"，所以<子集名>/<id>是全局唯一key，
# 拼上子集名之后保存图像名天然全局唯一
ANNOTATION_SAMPLE_ID_KEY_NAME = 'id'

# 编辑指令，也是本数据集唯一的文本字段。实测11142945行全非空、无空串
ANNOTATION_CAPTION_KEY_NAME = 'edit_instruction'

# 参考图(编辑前原图)的相对路径列表，本数据集恒为长度1的list
ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'reference_image_path_list'

# 编辑后图的相对路径，实测11142945行全非空
ANNOTATION_EDITED_IMAGE_KEY_NAME = 'edited_image_path'

# 源图状态，unzip_annotations里应该恒为embed_bytes(有源图字节)。
# 上游只有embed_bytes与url_only两种取值，而url_only那批全在
# unzip_url_source_annotations里(本脚本不读那个目录)。
# 这里按白名单判: 只有embed_bytes才继续，url_only以及任何未知取值都判掉。
# 万一上游把两套标注混在一起，url_only的行没有参考图，
# 必须被判掉而不是静默落盘成一个无参考图的假编辑对
ANNOTATION_SOURCE_IMAGE_STATE_KEY_NAME = 'source_image_state'

ANNOTATION_SOURCE_IMAGE_STATE_EMBED_BYTES = 'embed_bytes'

# 上游落盘的两张图共用同一个数字前缀，只靠这两个后缀区分:
#   <id>_source.jpg|png  参考图(编辑前原图)
#   <id>_edited.jpg|png  编辑后图
# 保存图像名的"原始图像名前缀"取的就是这个数字前缀(即parquet的id)
LOAD_EDITED_IMAGE_NAME_SUFFIX = '_edited'

LOAD_REFERENCE_IMAGE_NAME_SUFFIX = '_source'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 新标注固定只存这七个key，多一个少一个都在收尾自校验里报错。
# 上游标注里剩下的属性按方案确认全部丢弃、不另存索引:
#   dataset_task_type        : 恒为image_edit，整个数据集就一个值
#   sample_key / id / parquet_name / row_index : 上游定位用，id已内含在保存图像名里
#   subset_name / edit_task  : 只用来推导子集名(两者等价，edit_task就是子集目录名
#                              去掉<category_id>_前缀的结果)
#   category_id              : 大类编号(1.1/2.3这种)。按方案确认子集名只取纯任务名，
#                              所以这个属性会**永久丢失**，落盘后再也分不出大类
#   source_image_state       : 本脚本只处理embed_bytes那批，落盘后恒定，无需保存
#   source_image_url / source_image_sha256 / source_image_fetch_date :
#                              只有url_only那批非空，而那批已整体丢弃
#   instruction              : 与edit_instruction逐字相同的冗余字段
#   source_image_width/height、edited_image_width/height、
#   source_image_shape、edited_image_shape :
#                              上游记录的宽高。本脚本一律以实际写盘数组的shape为准，
#                              不采信上游数值
#   instruction_following_score / editing_consistency_score /
#   generation_quality_score : 三维质量分。全库实测只有4种组合
#                              (3,3,3: 10775128对 / 3,3,2: 695266 / 3,2,3: 160854 /
#                               3,2,2: 213775，IF恒为3)，
#                              按方案确认**不用于过滤、也不保存**
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# 本数据集每个编辑对固定只有1张参考图(编辑前原图)，没有第二张视觉条件图，
# 所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 上游子集目录名 -> 归一化后的任务名(即子集名)。
# 上游目录名规格是<category_id>_<task_name>(如1.1_style_transfer)，
# 按方案确认子集名去掉category_id前缀、只取纯任务名，最终产出23个子集。
# 这23个任务类型是数据集自带的、明确可知的，所以不需要mix兜底子集。
# 显式写死而不是每行现拆的原因: 拆错一个子集就会静默把几十万个样本对写进错误子集，
# 写死之后任何上游目录名变化都会在check_load_annotation_count里硬失败
GET_SET_NAME_DICT = {
    '1.1_style_transfer': 'style_transfer',
    '1.2_tone_adjustment': 'tone_adjustment',
    '1.3_viewpoint_transformation': 'viewpoint_transformation',
    '1.4_background_replacement': 'background_replacement',
    '2.1_object_addition': 'object_addition',
    '2.2_object_removal': 'object_removal',
    '2.3_object_replacement': 'object_replacement',
    '2.4_action_editing': 'action_editing',
    '2.5_part_extraction': 'part_extraction',
    '3.1_color_change': 'color_change',
    '3.2_material_change': 'material_change',
    '3.3_visual_beautification': 'visual_beautification',
    '3.4_count_change': 'count_change',
    '3.5_size_change': 'size_change',
    '4.1_movie_poster_text_editing': 'movie_poster_text_editing',
    '4.2_gui_interface_text_editing': 'gui_interface_text_editing',
    '4.3_object_surface_text_editing': 'object_surface_text_editing',
    '4.4_building_surface_text_editing': 'building_surface_text_editing',
    '5.1_perceptual_reasoning': 'perceptual_reasoning',
    '5.2_symbolic_reasoning': 'symbolic_reasoning',
    '5.3_social_reasoning': 'social_reasoning',
    '5.4_scientific_reasoning': 'scientific_reasoning',
    '6.1_compositional_editing': 'compositional_editing',
}

# 找不到任务类型时才用的兜底子集名。
# 本数据集23个上游子集的任务类型全部可知(见GET_SET_NAME_DICT)，实测一条都不会落进mix，
# 保留这条路径只是为了和002/006/007的口径保持一致，
# 并防止上游之后新增子集目录时被静默漏处理(新子集会被check_load_annotation_count
# 的"unknown subset"硬拦下来)
MIX_SET_NAME = 'mix'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 保存名形如scaleedit_style_transfer_123456_edited.jpg，
# 23个子集名全是ASCII小写、id全是数字，所以不会出现CJK字符或其它异常字符
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 保存图像名里的原图名前缀(即上游落盘图像名去掉_source/_edited后缀的部分)
# 必须是纯数字，它就是parquet的id。不是纯数字说明上游产物被改动过，
# 这种样本没法保证保存图像名唯一(可能去覆盖别的样本)，整对丢弃并上报
VALID_SAMPLE_ID_PATTERN = re.compile(r'^\d+$')

# 只保留RGB三通道图和灰度图，P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须命中这个白名单，任意一张不合格则整个图像编辑对丢弃。
# 实测抽样4680张里非RGB只有2张(L 1 / RGBA 1，约0.04%)，
# 图像格式JPEG 4679 / PNG 1(全库png只有25808张，占0.11%)
VALID_IMAGE_MODE_LIST = [
    'RGB',
    'L',
]

# 每个上游子集目录下的实测jsonl分片数(合计251，与上游parquet分片数一一对应)，
# 数量不对说明上游016没跑完
EXPECTED_SUBSET_ANNOTATION_FILE_COUNT_DICT = {
    '1.1_style_transfer': 16,
    '1.2_tone_adjustment': 12,
    '1.3_viewpoint_transformation': 1,
    '1.4_background_replacement': 24,
    '2.1_object_addition': 33,
    '2.2_object_removal': 30,
    '2.3_object_replacement': 51,
    '2.4_action_editing': 5,
    '2.5_part_extraction': 11,
    '3.1_color_change': 24,
    '3.2_material_change': 7,
    '3.3_visual_beautification': 2,
    '3.4_count_change': 1,
    '3.5_size_change': 1,
    '4.1_movie_poster_text_editing': 8,
    '4.2_gui_interface_text_editing': 3,
    '4.3_object_surface_text_editing': 8,
    '4.4_building_surface_text_editing': 7,
    '5.1_perceptual_reasoning': 1,
    '5.2_symbolic_reasoning': 1,
    '5.3_social_reasoning': 1,
    '5.4_scientific_reasoning': 1,
    '6.1_compositional_editing': 3,
}

# 每个上游子集目录下unzip_annotations的实测标注行数(合计11142945)，
# 直接取自上游016的对账报告unzip_check_missing_images.json里的
# subset_sample_pair_count_dict。解析阶段按上游子集和新子集两级硬对账，
# 少一行都说明上游016没跑完或产物被改动过
EXPECTED_SUBSET_ANNOTATION_COUNT_DICT = {
    '1.1_style_transfer': 716598,
    '1.2_tone_adjustment': 547660,
    '1.3_viewpoint_transformation': 431,
    '1.4_background_replacement': 1118526,
    '2.1_object_addition': 1519718,
    '2.2_object_removal': 1403222,
    '2.3_object_replacement': 2456122,
    '2.4_action_editing': 171507,
    '2.5_part_extraction': 466155,
    '3.1_color_change': 1082496,
    '3.2_material_change': 285741,
    '3.3_visual_beautification': 49066,
    '3.4_count_change': 458,
    '3.5_size_change': 2019,
    '4.1_movie_poster_text_editing': 384822,
    '4.2_gui_interface_text_editing': 107557,
    '4.3_object_surface_text_editing': 391431,
    '4.4_building_surface_text_editing': 330578,
    '5.1_perceptual_reasoning': 1335,
    '5.2_symbolic_reasoning': 1853,
    '5.3_social_reasoning': 1804,
    '5.4_scientific_reasoning': 6175,
    '6.1_compositional_editing': 97671,
}

# 23个上游子集归一化出的23个新子集(即23种图像编辑任务类型)及其实测标注条数。
# 本数据集上游子集与新子集是一对一的，所以数值与
# EXPECTED_SUBSET_ANNOTATION_COUNT_DICT逐项相同，但仍然分开写死:
# 子集级对账能额外拦住"某个上游子集被映射进错误新子集"这种上游子集级对账
# 看不出来的问题
EXPECTED_SET_ANNOTATION_COUNT_DICT = {
    'style_transfer': 716598,
    'tone_adjustment': 547660,
    'viewpoint_transformation': 431,
    'background_replacement': 1118526,
    'object_addition': 1519718,
    'object_removal': 1403222,
    'object_replacement': 2456122,
    'action_editing': 171507,
    'part_extraction': 466155,
    'color_change': 1082496,
    'material_change': 285741,
    'visual_beautification': 49066,
    'count_change': 458,
    'size_change': 2019,
    'movie_poster_text_editing': 384822,
    'gui_interface_text_editing': 107557,
    'object_surface_text_editing': 391431,
    'building_surface_text_editing': 330578,
    'perceptual_reasoning': 1335,
    'symbolic_reasoning': 1853,
    'social_reasoning': 1804,
    'scientific_reasoning': 6175,
    'compositional_editing': 97671,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = 11142945

EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 251

# 【按方案确认整体丢弃的子集】
# part_extraction(2.5_part_extraction，466155对):
#   这个子集的**编辑后图本身就是从原图里被抠出来的那个局部/部件**
#   (名字就是"部件提取"，不是"编辑整张图")，所以编辑后图与编辑前原图的长宽比
#   天生就不一样。而本次改动要求"reference_image[0]必须与编辑后图长宽比严格相同、
#   否则整对丢弃"，这个子集会被逐条判掉、几乎清零，
#   留着只会在日志里刷满different_aspect_ratio，所以按方案在解析阶段整体跳过。
#   (与003.resave_imgedit_ti2i_dataset.py里丢弃reference_extract是同一个理由)
# 丢弃判定放在"逐子集与逐上游子集条数统计之后"，
# 所以EXPECTED_ANNOTATION_COUNT_DICT / EXPECTED_SET_ANNOTATION_COUNT_DICT /
# EXPECTED_TOTAL_ANNOTATION_COUNT这几个上游侧硬对账口径全部保持原值不变
SKIP_SET_NAME_LIST = [
    'part_extraction',
]

# 被整体丢弃的子集合计条数(part_extraction实测466155条)，解析阶段硬对账。
# 这个数字守护的是"丢弃范围没被改动过"
EXPECTED_SKIP_SET_COUNT = 466155

# 最终产出的子集数: 上游23个子集扣掉按方案整体丢弃的part_extraction后是22个
EXPECTED_SAVE_SET_COUNT = 22

# 被整体丢弃的"只有源图URL、没有源图字节"的样本对数(上游实测)。
# 本脚本不读那个目录，这个数字只写进resave_check_result.json做记录，
# 说明本次落盘相对上游全量少了哪一部分
EXPECTED_SKIP_URL_SOURCE_ONLY_PAIR_COUNT = 702078

# 上游parquet的全量行数(完整编辑对 + 只有源图URL的对)，同样只做记录
EXPECTED_TOTAL_UPSTREAM_ROW_COUNT = 11845023

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
# 长宽比判定用Fraction最简分数比、**不留任何容差**:
# 留容差会让"几乎一样但不严格相等"的样本(如910x512 vs 896x512)被各向异性拉伸落盘。
# 落盘后还会把参考图的真实shape与目标shape硬对账一次，不等则整对丢弃;
# 收尾自校验再真解一次reference_image[0]、硬校验它严格等于json里的width/height。
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
# 按方案确认用PIL的LANCZOS而不是cv2.resize: 与项目既定口径(017.0/017.1)保持一致
SAVE_IMAGE_RESIZE_RESAMPLING = Image.Resampling.LANCZOS

# 豁免尺寸对齐的子集: 这些子集的编辑后图与全部参考图**完全原样落盘**，
# 不判长宽比、不resize、不丢弃、也不做长边对齐。
# 本数据集没有任何子集需要豁免(全部子集都要求参考图与编辑后图像素对齐)，
# 所以这里是空列表(保留这个常量只为与001等脚本口径一致)
EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST = []

# 收尾自校验时是否真解一次reference_image[0]、硬校验它的shape等于json里的
# width/height。按方案确认置True: "第一张参考图与编辑后图尺寸必须一致"是本次改动的
# 核心诉求，而只对账json里的数字是查不出resize有没有真的生效的，必须真解一次图。
# 代价是收尾自校验要多解一遍全部参考图，在NAS上会明显变慢
CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG = True

# 指令长度阈值。实测抽样172900条: min 19 / p50 69 / p90 113 / p99 177 /
# p999 291 / max 512，没有一条短于10。
# 取10/512后正常指令一条都不会被砍，超长的512字符样本全是下面说的那批
# 纯感叹号垃圾指令(它们会先被"无文字字符"这条规则判掉)
MIN_CAPTION_LENGTH = 10

MAX_CAPTION_LENGTH = 512

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 实测抽样172900条里有82条(约0.047%)是恰好512个感叹号的垃圾指令
# (形如'!!!!!...'，集中在1.4/2.1/2.2/2.3/3.1/3.2这几个子集)，
# 是上游生成时崩掉的产物，没有任何可训练的语义，整对丢弃
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 【显式记录一条不采用的过滤规则】
# 不使用"连续重复字符"这类启发式规则来判垃圾指令: 实测
# "Replace the text '火火火火火火火火火火火火' with '玻璃相框'." 这种
# 文字编辑子集(4.3/4.4)的正常指令里，被替换的原文本身就是重复字符，
# 用重复字符规则会大面积误伤正常样本。
# 那批纯感叹号垃圾指令用上面的CAPTION_WORD_CHAR_PATTERN就能精确判掉。

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)，与006口径一致。
# 分两类: null字面量(上游构造流程失败时把空值写成了字符串)、
# 明确表示"不做任何修改"的指令(编辑前后图应该几乎相同，当训练样本是纯噪声)。
# 实测抽样172900条里一条都没命中，这里只作兜底
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

# 需要在归一化时剥掉的整体包裹引号。
# 实测抽样172900条里有1503条(约0.87%，集中在1.3与4.x几个子集)整条指令被一对
# 英文双引号包住，形如 "Draw a side view of the chair and the stool." ，
# 这是上游生成时残留的引号、不是指令内容，落盘前剥掉一层。
# 只在"首尾是同一个引号字符 且 内部不再出现这个引号字符"时才剥，
# 所以 "Replace the text 'A' with 'B'." 会被正确剥成
# Replace the text 'A' with 'B'. (内部的单引号原样保留)，
# 而形如 "A" and "B" 这种首尾恰好是引号、但并不是整体包裹的指令不会被误剥
CAPTION_WRAP_QUOTE_CHAR_LIST = [
    '"',
    "'",
    '“',
    '”',
    '‘',
    '’',
]

# 带编号的视觉参考图占位符，编号从"非原图的第1张参考图"起算:
# [V1*]指代reference_image[1]、[V2*]指代reference_image[2]...[VN*]指代
# reference_image[N]，其中N == reference_image_num - 1。
# 编辑前原图reference_image[0]永远隐式、不写进指令、不占编号。
# 这套写法与002/006/007完全一致，保证跨数据集口径统一。
# 本数据集恒为1张参考图(N == 0)，即指令里不允许出现任何占位符
# (实测抽样172900条命中0条)，这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2


def check_skip_set(per_set_name):
    """判定这个子集是不是要整体丢弃的子集，返回True表示丢弃

    见SKIP_SET_NAME_LIST的注释: part_extraction的编辑后图就是被抠出来的局部部件、
    与编辑前原图长宽比天生不同，按"第一张参考图必须与编辑后图长宽比相同"这条规则
    会被逐条判掉，所以在解析阶段就整体跳过。
    """
    return per_set_name in SKIP_SET_NAME_LIST


def get_set_name(per_subset_name):
    """把上游子集目录名映射成子集名(即图像编辑任务类型)

    上游目录名规格是<category_id>_<task_name>(如1.1_style_transfer)，
    按方案子集名只取纯任务名(style_transfer)。
    优先用写死的映射表(拆错一个子集就会静默把几十万个样本对写进错误子集);
    表里没有的子集才退回到mix，这条路径只在上游新增子集目录时才会走到，
    且会在check_load_annotation_count里被"unknown subset"硬拦下来。
    """
    per_subset_name = str(per_subset_name).strip()

    if per_subset_name in GET_SET_NAME_DICT:
        return GET_SET_NAME_DICT[per_subset_name]

    return MIX_SET_NAME


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    本数据集每个编辑对只有编辑前原图这一张参考图，所有子集恒为1
    (上游标注里的reference_image_num也恒为1)。
    保留这个函数是为了和002/006/007的收尾自校验口径保持一致。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令: strip + 剥掉一层整体包裹的引号

    实测约0.87%的指令整条被一对英文双引号包住，这是上游生成时残留的引号、
    不是指令内容，落盘前剥掉一层。
    只在"首尾是同一个引号字符 且 内部不再出现这个引号字符"时才剥，
    所以文字编辑子集里 "Replace the text 'A' with 'B'." 这种指令
    只会被剥掉外层双引号、内部单引号原样保留。
    本数据集的指令里没有任何视觉参考图占位符，也没有"the reference image"
    这类自然语言指代(只有1张参考图、指令从不指代它)，所以不做任何占位符改写。
    已经带编号的占位符原样保留，不做任何改动(便于后续多参考图数据集复用本函数)。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    for per_quote_char in CAPTION_WRAP_QUOTE_CHAR_LIST:
        if len(per_ti2i_caption) < 2:
            break

        if per_ti2i_caption[0] != per_quote_char or per_ti2i_caption[
                -1] != per_quote_char:
            continue

        per_inner_ti2i_caption = per_ti2i_caption[1:-1]
        # 内部还出现同一个引号字符时说明首尾这两个引号不是一对整体包裹的引号，
        # 剥掉会改变指令语义，这里直接放弃剥离
        if per_quote_char in per_inner_ti2i_caption:
            continue

        per_ti2i_caption = per_inner_ti2i_caption.strip()
        break

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
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().strip(
        CAPTION_STRIP_CHAR)

    return per_ti2i_caption in NULL_LIKE_CAPTION_LIST


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上，逐样本打一次os.path.exists就是一次网络往返，
    1114万个编辑对就是2229万次。实测同一个jsonl里的图像全部落在同一个
    unzip_images/<子集>/<分片名>/目录下(单目录最多50000对、即100000个文件)，
    所以这里按目录缓存一次os.listdir的结果，之后只做集合查表，
    网络往返次数从"图像张数"降到"分片目录数"(251次)。
    listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判，
    保证判定结果和不加缓存时完全一致。
    """
    per_image_dir_path = os.path.dirname(per_image_path)
    per_image_name = os.path.basename(per_image_path)

    if per_image_dir_path not in dir_file_name_cache_dict:
        try:
            dir_file_name_cache_dict[per_image_dir_path] = set(
                os.listdir(per_image_dir_path))
        except Exception:
            dir_file_name_cache_dict[per_image_dir_path] = None

    per_dir_file_name_set = dir_file_name_cache_dict[per_image_dir_path]
    if per_dir_file_name_set is None:
        return os.path.exists(per_image_path)

    return per_image_name in per_dir_file_name_set


def get_image_name_prefix(per_image_relative_path, per_image_name_suffix):
    """从上游落盘图像的相对路径里取出原图名前缀(即parquet的id)

    上游图像名形如123_edited.jpg / 123_source.png，
    这里先去掉扩展名、再去掉_edited或_source后缀，剩下的就是数字前缀。
    后缀对不上时返回空串，由调用方按"保存名非法"整对丢弃并上报。
    """
    per_image_name = os.path.basename(
        str(per_image_relative_path).strip()).strip().lower()
    per_image_name_stem = os.path.splitext(per_image_name)[0]

    if not per_image_name_stem.endswith(per_image_name_suffix):
        return ''

    return per_image_name_stem[:-len(per_image_name_suffix)]


def process_single_annotation_file(annotation_file_pair):
    """解析单个上游jsonl标注，组装图像编辑对(参考图+编辑后图+编辑指令)的列表

    这一步只做纯文本层面的过滤(json坏行、子集名缺失或与目录不自洽、
    源图状态是url_only、缺图、原图名前缀非法、参考图数量不对、保存名非法、
    指令为空、指令是null字面量、指令没有任何文字字符、指令过短、指令过长、
    指令是坏占位符指令)，图像本身的解码校验和分辨率过滤留到后面多进程里做。

    指令的归一化(strip + 剥外层引号)放在所有指令过滤之前，
    这样长度过滤判定的字符串就和最终写进json的字符串完全一致，
    收尾自校验直接量json里的长度就能复检。
    """
    per_annotation_path, per_subset_name, root_dataset_path = annotation_file_pair

    total_annotation_count, illegal_line_count = 0, 0
    missing_subset_name_count = 0
    url_source_only_count = 0
    missing_image_count = 0
    invalid_reference_image_num_count = 0
    invalid_save_image_name_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    wrap_quote_caption_count = 0
    skip_set_count = 0
    subset_annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    # 每个worker只处理一个jsonl，缓存里通常只有一个分片目录，内存开销可忽略
    dir_file_name_cache_dict = {}

    try:
        load_jsonl_file = open(per_annotation_path, 'r', encoding='UTF-8')
    except Exception as e:
        print('2222', per_annotation_path, e)

        return [
            edit_annotation_pair_list,
            per_subset_name,
            total_annotation_count,
            1,
            missing_subset_name_count,
            url_source_only_count,
            missing_image_count,
            invalid_reference_image_num_count,
            invalid_save_image_name_count,
            empty_caption_count,
            null_like_caption_count,
            no_word_char_caption_count,
            too_short_caption_count,
            too_long_caption_count,
            invalid_placeholder_caption_count,
            wrap_quote_caption_count,
            subset_annotation_count_dict,
            set_annotation_count_dict,
            skip_set_count,
        ]

    with load_jsonl_file:
        for per_line in load_jsonl_file:
            per_line = per_line.strip()
            if not per_line:
                continue

            total_annotation_count += 1

            try:
                per_annotation = json.loads(per_line)
            except Exception as e:
                illegal_line_count += 1
                print('2222', per_annotation_path, e)
                continue

            if not isinstance(per_annotation, dict):
                illegal_line_count += 1
                print('2222', per_annotation_path, 'annotation not a dict')
                continue

            per_annotation_subset_name = per_annotation.get(
                ANNOTATION_SUBSET_NAME_KEY_NAME, '')
            if not isinstance(per_annotation_subset_name, str):
                per_annotation_subset_name = ''
            per_annotation_subset_name = per_annotation_subset_name.strip()

            # 行内子集名必须和这个标注文件所在的子集目录一致: 不一致说明上游产物
            # 被搬动过，继续跑会把样本对写进错误子集(实测0条，这里只做防御)
            if not per_annotation_subset_name or per_annotation_subset_name != per_subset_name:
                missing_subset_name_count += 1
                print('3333', per_annotation_path, per_annotation_subset_name,
                      per_subset_name)
                continue

            subset_annotation_count_dict[
                per_annotation_subset_name] = subset_annotation_count_dict.get(
                    per_annotation_subset_name, 0) + 1

            per_set_name = get_set_name(per_annotation_subset_name)

            # 整体丢弃的子集在这里就跳过(见SKIP_SET_NAME_LIST的注释)。
            # 这一步刻意排在subset_annotation_count_dict统计之后、
            # 任何图像与指令过滤之前: 逐子集条数硬对账用的是那个统计口径，
            # 所以上游侧的EXPECTED_*常量全部保持原值不变;
            # 而后面那些过滤计数则只统计保留下来的子集
            if check_skip_set(per_set_name):
                skip_set_count += 1
                continue

            per_source_image_state = per_annotation.get(
                ANNOTATION_SOURCE_IMAGE_STATE_KEY_NAME, '')
            if not isinstance(per_source_image_state, str):
                per_source_image_state = ''
            per_source_image_state = per_source_image_state.strip()

            # unzip_annotations里应该恒为embed_bytes。按白名单判:
            # url_only的行没有参考图、构不成图像编辑对，未知取值也说明上游规格变了，
            # 两者都必须判掉而不是静默落盘成一个无参考图的假编辑对
            if per_source_image_state != ANNOTATION_SOURCE_IMAGE_STATE_EMBED_BYTES:
                url_source_only_count += 1
                print('3333', per_annotation_path, per_source_image_state)
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

            per_edited_image_path = os.path.join(
                root_dataset_path, *LOAD_IMAGE_DIR_NAME_LIST,
                per_edited_image_relative_path)
            if not check_image_file_exists(per_edited_image_path,
                                           dir_file_name_cache_dict):
                missing_image_count += 1
                continue

            # 保存图像名的原图名前缀取自编辑后图像的文件名(去掉扩展名和_edited后缀)，
            # 即parquet的id。这里从真正被读取的那张图现取前缀，
            # 再和标注里的id字段交叉比对一次，保证保存名和图像严格对应
            per_edited_image_name_prefix = get_image_name_prefix(
                per_edited_image_relative_path, LOAD_EDITED_IMAGE_NAME_SUFFIX)

            per_sample_id = per_annotation.get(ANNOTATION_SAMPLE_ID_KEY_NAME,
                                               None)
            per_sample_id_name = f'{per_sample_id}'.strip() if isinstance(
                per_sample_id,
                int) and not isinstance(per_sample_id, bool) else ''

            if not VALID_SAMPLE_ID_PATTERN.match(
                    per_edited_image_name_prefix
            ) or per_edited_image_name_prefix != per_sample_id_name:
                invalid_save_image_name_count += 1
                print('3333', per_edited_image_path,
                      per_edited_image_name_prefix, per_sample_id_name)
                continue

            # 保存名形如scaleedit_style_transfer_123456_edited.jpg。
            # 上游id只在子集内唯一，所以保存名里必须带上子集名才能全局唯一;
            # 加上子集名之后实测11142945个保存名100%唯一(落盘前还会再兜一道)
            per_save_image_name_prefix = (
                f'{DATASET_NAME}_{per_set_name}_{per_edited_image_name_prefix}'
            )
            per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
            # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的前缀
            # (即带_edited那一段)，和002/006/007的写法保持一致，
            # 收尾自校验也是按edited_image去掉.jpg来反推这个文件夹名的
            per_save_pair_folder_name = os.path.splitext(
                per_save_edited_image_name)[0]

            # 保存名里出现路径分隔符或其它异常字符会写坏目录结构，整对丢弃
            if not VALID_IMAGE_NAME_PATTERN.match(per_save_edited_image_name):
                invalid_save_image_name_count += 1
                print('3333', per_edited_image_path,
                      per_save_edited_image_name)
                continue

            per_reference_image_relative_path_list = per_annotation.get(
                ANNOTATION_REFERENCE_IMAGE_KEY_NAME, [])
            if not isinstance(per_reference_image_relative_path_list,
                              (list, tuple)):
                per_reference_image_relative_path_list = []

            # 本数据集每个编辑对固定只有1张参考图，数量不对说明这条不是完整编辑对
            # (url_only那批的这个list是空的，上一步已经judge掉，这里再兜一道)
            if len(per_reference_image_relative_path_list
                   ) != EXPECT_REFERENCE_IMAGE_NUM:
                invalid_reference_image_num_count += 1
                continue

            per_reference_image_path_list = []
            per_save_reference_image_name_list = []
            per_missing_reference_image_count = 0
            per_invalid_save_reference_image_name_count = 0
            for per_reference_image_relative_path in per_reference_image_relative_path_list:
                if not isinstance(per_reference_image_relative_path, str):
                    per_reference_image_relative_path = ''
                per_reference_image_relative_path = per_reference_image_relative_path.replace(
                    '\\', '/').strip().lstrip('/')

                if not per_reference_image_relative_path:
                    per_missing_reference_image_count += 1
                    continue

                per_reference_image_path = os.path.join(
                    root_dataset_path, *LOAD_IMAGE_DIR_NAME_LIST,
                    per_reference_image_relative_path)
                if not check_image_file_exists(per_reference_image_path,
                                               dir_file_name_cache_dict):
                    per_missing_reference_image_count += 1
                    continue

                # 参考图与编辑后图必须是同一个样本(同一个数字前缀)，
                # 不一致说明上游成员错位，这种编辑对的参考图根本不是这张图的原图
                per_reference_image_name_prefix = get_image_name_prefix(
                    per_reference_image_relative_path,
                    LOAD_REFERENCE_IMAGE_NAME_SUFFIX)
                if per_reference_image_name_prefix != per_edited_image_name_prefix:
                    per_invalid_save_reference_image_name_count += 1
                    continue

                per_save_reference_image_name = f'{per_save_image_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}'
                if not VALID_IMAGE_NAME_PATTERN.match(
                        per_save_reference_image_name):
                    per_invalid_save_reference_image_name_count += 1
                    continue

                per_reference_image_path_list.append(per_reference_image_path)
                per_save_reference_image_name_list.append(
                    per_save_reference_image_name)

            if per_invalid_save_reference_image_name_count > 0:
                invalid_save_image_name_count += 1
                print('3333', per_edited_image_path,
                      per_reference_image_relative_path_list)
                continue

            # 同一个样本对里两张参考图撞名会互相覆盖，整对丢弃
            # (本数据集只有1张参考图，这里只做防御性拦截)
            if len(set(per_save_reference_image_name_list)) != len(
                    per_save_reference_image_name_list):
                invalid_save_image_name_count += 1
                print('3333', per_edited_image_path,
                      per_save_reference_image_name_list)
                continue

            per_expect_reference_image_num = get_expect_reference_image_num(
                per_set_name)
            # 参考图缺任意一张都会让这个编辑对的条件信息不完整，整对丢弃
            if per_missing_reference_image_count > 0 or len(
                    per_reference_image_path_list
            ) != per_expect_reference_image_num:
                missing_image_count += 1
                continue

            per_raw_ti2i_caption = per_annotation.get(
                ANNOTATION_CAPTION_KEY_NAME, '')
            if isinstance(per_raw_ti2i_caption, (list, tuple)):
                per_raw_ti2i_caption = per_raw_ti2i_caption[0] if len(
                    per_raw_ti2i_caption) > 0 else ''
            if not isinstance(per_raw_ti2i_caption, str):
                per_raw_ti2i_caption = ''
            per_raw_ti2i_caption = per_raw_ti2i_caption.strip()

            # 归一化(strip + 剥掉一层整体包裹的引号)放在所有指令过滤之前，
            # 保证过滤判定的字符串和写进json的字符串完全一致
            per_ti2i_caption = get_normalized_ti2i_caption(
                per_raw_ti2i_caption)
            if per_ti2i_caption != per_raw_ti2i_caption:
                # 只统计不丢样本: 实测约0.87%的指令被剥掉了一层外层双引号
                wrap_quote_caption_count += 1

            # 空指令、全空格指令视为不合格图像编辑对
            # (上游实测11142945行的指令全非空，这里只作兜底)
            if not per_ti2i_caption:
                empty_caption_count += 1
                continue

            # null字面量与"不做任何修改"这类无意义指令同样丢弃(实测抽样0条命中)
            if check_null_like_caption(per_ti2i_caption):
                null_like_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:50])
                continue

            # 只剩标点、没有任何数字/字母/汉字的指令也丢弃:
            # 实测抽样172900条里有82条(约0.047%)是恰好512个感叹号的垃圾指令
            if not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption):
                no_word_char_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:50])
                continue

            # 过短指令视为不合格图像编辑对(实测抽样最短19字符，一条都不会被砍)
            if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
                too_short_caption_count += 1
                print('3333', per_edited_image_path, len(per_ti2i_caption))
                continue

            # 过长指令同样视为不合格图像编辑对
            # (实测抽样p999只有291、最长512的那批已被"无文字字符"判掉)
            if len(per_ti2i_caption) > MAX_CAPTION_LENGTH:
                too_long_caption_count += 1
                print('3333', per_edited_image_path, len(per_ti2i_caption))
                continue

            # 占位符编号与参考图数量不自洽的指令也丢弃。
            # 本数据集是单参考图，即要求指令里完全没有[Vn*]占位符(实测0条命中)
            if check_invalid_caption(per_ti2i_caption,
                                     per_expect_reference_image_num):
                invalid_placeholder_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:100])
                continue

            set_annotation_count_dict[
                per_set_name] = set_annotation_count_dict.get(per_set_name,
                                                              0) + 1

            edit_annotation_pair_list.append([
                per_set_name,
                per_save_pair_folder_name,
                per_edited_image_path,
                per_save_edited_image_name,
                per_reference_image_path_list,
                per_save_reference_image_name_list,
                per_ti2i_caption,
                per_expect_reference_image_num,
            ])

    return [
        edit_annotation_pair_list,
        per_subset_name,
        total_annotation_count,
        illegal_line_count,
        missing_subset_name_count,
        url_source_only_count,
        missing_image_count,
        invalid_reference_image_num_count,
        invalid_save_image_name_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        wrap_quote_caption_count,
        subset_annotation_count_dict,
        set_annotation_count_dict,
        skip_set_count,
    ]


def get_all_annotation_file_pair(root_dataset_path):
    """收集上游全部标注文件，返回[标注文件任务列表, 每个上游子集的标注文件数]

    上游标注按子集分目录、目录内按parquet分片分文件(实测23个子集共251个jsonl)，
    这里按"标注文件"这一粒度出任务，正好能把多进程铺满。
    只读unzip_annotations，不读unzip_url_source_annotations
    (那702078条没有参考图、按方案整体丢弃)。
    """
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
                root_dataset_path,
            ])

        subset_annotation_file_count_dict[per_subset_name] = len(
            per_subset_annotation_file_name_list)

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list, subset_annotation_file_count_dict


def get_all_edit_annotation_pair(root_dataset_path):
    """按标注文件粒度多进程组装全部图像编辑对的列表

    上游有251个标注文件、合计11142945行，逐行还要判2张图像文件是否存在，
    所以这里按标注文件开多进程解析，最后按保存的编辑后图像名统一排序。
    """
    annotation_file_pair_list, subset_annotation_file_count_dict = get_all_annotation_file_pair(
        root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list),
          'annotation subset:', len(subset_annotation_file_count_dict))

    total_annotation_count, illegal_line_count = 0, 0
    missing_subset_name_count = 0
    url_source_only_count = 0
    missing_image_count = 0
    invalid_reference_image_num_count = 0
    invalid_save_image_name_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    wrap_quote_caption_count = 0
    skip_set_count = 0
    subset_annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            total_annotation_count += per_load_result[2]
            illegal_line_count += per_load_result[3]
            missing_subset_name_count += per_load_result[4]
            url_source_only_count += per_load_result[5]
            missing_image_count += per_load_result[6]
            invalid_reference_image_num_count += per_load_result[7]
            invalid_save_image_name_count += per_load_result[8]
            empty_caption_count += per_load_result[9]
            null_like_caption_count += per_load_result[10]
            no_word_char_caption_count += per_load_result[11]
            too_short_caption_count += per_load_result[12]
            too_long_caption_count += per_load_result[13]
            invalid_placeholder_caption_count += per_load_result[14]
            wrap_quote_caption_count += per_load_result[15]

            for per_subset_name, per_subset_count in per_load_result[16].items(
            ):
                subset_annotation_count_dict[
                    per_subset_name] = subset_annotation_count_dict.get(
                        per_subset_name, 0) + per_subset_count

            for per_set_name, per_set_count in per_load_result[17].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

            skip_set_count += per_load_result[18]

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        subset_annotation_file_count_dict,
        subset_annotation_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        missing_subset_name_count,
        url_source_only_count,
        missing_image_count,
        invalid_reference_image_num_count,
        invalid_save_image_name_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        wrap_quote_caption_count,
        skip_set_count,
    ]


def check_load_annotation_count(subset_annotation_file_count_dict,
                                subset_annotation_count_dict,
                                total_annotation_count,
                                total_annotation_file_count,
                                edit_annotation_pair_list, skip_set_count):
    """解析完标注后按上游子集与新子集两级硬对账，并检查保存图像名是否唯一

    上游标注是016一次性跑出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账能额外拦住"某个上游子集被映射进错误新子集"这种上游子集级对账
    看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对。

    注意这里只硬对账"标注层条数"(逐子集行数 / 总行数 / 标注文件数)，
    不硬对账各类指令过滤条数: 指令过滤的精确条数只做过17.3万条抽样、
    没有全量实测值，写死会误伤。这些计数会全部写进resave_check_result.json上报。
    """
    check_error_message_list = []

    # 按方案整体丢弃的子集(part_extraction)的条数硬对账，
    # 守住"丢弃范围没被改动过"这条口径。
    # 注意上游侧的逐子集/逐上游子集/总行数三个口径都在丢弃之前统计，
    # 所以那几个EXPECTED_*常量仍然包含被丢弃的子集、保持原值
    if skip_set_count != EXPECTED_SKIP_SET_COUNT:
        check_error_message_list.append(
            f'skip set count not match '
            f'{skip_set_count} != {EXPECTED_SKIP_SET_COUNT}')

    # 每个上游子集的标注文件数(即parquet分片数)
    for per_subset_name in sorted(subset_annotation_file_count_dict.keys()):
        if per_subset_name not in EXPECTED_SUBSET_ANNOTATION_FILE_COUNT_DICT:
            check_error_message_list.append(
                f'unknown subset {per_subset_name}')
            continue

        per_expect_annotation_file_count = EXPECTED_SUBSET_ANNOTATION_FILE_COUNT_DICT[
            per_subset_name]
        if subset_annotation_file_count_dict[
                per_subset_name] != per_expect_annotation_file_count:
            check_error_message_list.append(
                f'{per_subset_name} annotation file count not match '
                f'{subset_annotation_file_count_dict[per_subset_name]} != '
                f'{per_expect_annotation_file_count}')

    for per_subset_name in sorted(
            EXPECTED_SUBSET_ANNOTATION_FILE_COUNT_DICT.keys()):
        if per_subset_name not in subset_annotation_file_count_dict:
            check_error_message_list.append(
                f'missing subset dir {per_subset_name}')

    # 每个上游子集的标注行数
    set_annotation_count_dict = {}
    for per_subset_name in sorted(subset_annotation_count_dict.keys()):
        if per_subset_name not in EXPECTED_SUBSET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown subset {per_subset_name}')
            continue

        per_expect_annotation_count = EXPECTED_SUBSET_ANNOTATION_COUNT_DICT[
            per_subset_name]
        if subset_annotation_count_dict[
                per_subset_name] != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_subset_name} annotation count not match '
                f'{subset_annotation_count_dict[per_subset_name]} != '
                f'{per_expect_annotation_count}')

        # 这里必须走和落盘时完全同一条子集名推导路径，否则"上游子集->新子集"映射改了
        # 而对账表没改时，子集级对账会用另一套映射自说自话地对上
        per_set_name = get_set_name(per_subset_name)
        set_annotation_count_dict[per_set_name] = set_annotation_count_dict.get(
            per_set_name, 0) + subset_annotation_count_dict[per_subset_name]

    for per_subset_name in sorted(
            EXPECTED_SUBSET_ANNOTATION_COUNT_DICT.keys()):
        if per_subset_name not in subset_annotation_count_dict:
            check_error_message_list.append(
                f'missing subset {per_subset_name}')

    # 归一化后每个新子集的标注行数
    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in EXPECTED_SET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown set {per_set_name}')
            continue

        per_expect_set_annotation_count = EXPECTED_SET_ANNOTATION_COUNT_DICT[
            per_set_name]
        if set_annotation_count_dict[
                per_set_name] != per_expect_set_annotation_count:
            check_error_message_list.append(
                f'{per_set_name} set annotation count not match '
                f'{set_annotation_count_dict[per_set_name]} != '
                f'{per_expect_set_annotation_count}')

    for per_set_name in sorted(EXPECTED_SET_ANNOTATION_COUNT_DICT.keys()):
        if per_set_name not in set_annotation_count_dict:
            check_error_message_list.append(f'missing set {per_set_name}')

    # 要整体丢弃的子集必须真的在上游标注里存在(不存在说明子集名写错了，
    # 那样丢弃就是空操作、part_extraction会被静默保留下来)
    for per_skip_set_name in SKIP_SET_NAME_LIST:
        if per_skip_set_name not in set_annotation_count_dict:
            check_error_message_list.append(
                f'skip set {per_skip_set_name} not in upstream annotation')

    # 这里的set_annotation_count_dict是"上游标注侧"的子集集合(23个)，
    # 扣掉按方案整体丢弃的子集之后才是最终产出的子集数
    if len(set_annotation_count_dict) - len(
            SKIP_SET_NAME_LIST) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict) - len(SKIP_SET_NAME_LIST)} != '
            f'{EXPECTED_SAVE_SET_COUNT}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

    # 保存的编辑后图像名必须全局唯一(上游id只在子集内唯一，加上子集名后才唯一)，
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


def check_same_image_aspect_ratio(per_reference_image_shape,
                                  per_edited_image_shape):
    """判定参考图与编辑后图的长宽比是否严格相同，返回True表示相同(可以等比resize)

    两个入参都是[宽, 高]。
    用Fraction的最简分数比精确判定、**不留任何容差**，不用浮点相除:
    浮点比较要么因为精度误差把本该相同的判成不同(如1056/1584与832/1248)，
    要么需要引入一个人为的容差阈值，而容差一旦放开就会让"几乎一样但不严格相等"的
    样本(如910x512与896x512)被各向异性拉伸落盘。最简分数比是精确的、可复现的。
    """
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


def check_single_image(per_image_path):
    """校验单张图像能否正常解码，并过滤非RGB图和极端分辨率图

    返回图像宽高只用于统计，最终写进json的宽高一定取自实际写盘图像的shape。
    """
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把P图/CMYK图静默转成3通道、
    # 把RGBA图静默丢掉alpha通道，所以必须先用PIL读原始mode才能把这些图判出来
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

    短边和宽高比的过滤按方案只以编辑后图像为准判定，参考图只要求能正常解码且
    mode命中白名单。

    这里之所以能零额外IO地判长宽比: check_single_image本来就已经把编辑后图和
    每张参考图都真解码了一遍并返回了[宽, 高]，改造前只是把返回值丢掉了。
    现在接住这些shape，既能判长宽比、又能把"每张参考图的目标尺寸"一路带到落盘阶段，
    落盘时不必再重算一次。

    豁免子集(EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST)的目标尺寸一律给None，
    表示编辑后图与全部参考图都完全原样落盘。
    """
    per_set_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_ti2i_caption, per_expect_reference_image_num = edit_annotation_pair

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
            per_ti2i_caption,
            per_expect_reference_image_num,
        ],
    ]


def get_all_edit_pair_save_folder_pair(edit_annotation_pair_list,
                                       save_dataset_path):
    """把过滤后的合格图像编辑对按子集分组，排序后每10000对切成一个文件夹

    切分必须在过滤全部完成之后做，且切分前先按保存的编辑后图像名排序，这样才能保证
    每个文件夹都是满10000对(每个子集只有最后一个文件夹允许不满)。每个图像编辑对在
    文件夹里再独占一个子文件夹，该对的编辑后图像和所有参考图像都存在这个子文件夹里。
    本数据集会产出23个子集，最大的object_replacement约245万对会切出约246个文件夹，
    合计约1120个文件夹。

    和006/007不同的是: 每个图像编辑对自己那层子文件夹的os.makedirs放到
    process_single_edit_pair里由worker并行做，这里只算路径不建目录。
    本数据集有1114万个样本对文件夹，在主进程里串行makedirs一遍，
    光这一步在NAS上就要跑几个小时，而且这几个小时里32个worker全在空等。
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

        # 子集目录先建出来，后面每个文件夹的json直接写在这一层
        os.makedirs(os.path.join(save_dataset_path, per_set_name),
                    exist_ok=True)

        per_set_folder_count = 0
        for per_folder_start_index in range(
                0, len(per_set_edit_annotation_pair_list),
                PER_FOLDER_EDIT_PAIR_NUM):
            per_folder_edit_annotation_pair_list = per_set_edit_annotation_pair_list[
                per_folder_start_index:per_folder_start_index +
                PER_FOLDER_EDIT_PAIR_NUM]

            per_folder_name = f'{per_set_name}_{per_set_folder_count:05d}'

            for per_edit_annotation_pair in per_folder_edit_annotation_pair_list:
                _, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_save_reference_image_shape_list, per_ti2i_caption, per_expect_reference_image_num = per_edit_annotation_pair

                edit_pair_save_folder_pair_list.append([
                    per_set_name,
                    per_folder_name,
                    per_save_pair_folder_name,
                    per_edited_image_path,
                    per_save_edited_image_name,
                    per_reference_image_path_list,
                    per_save_reference_image_name_list,
                    per_save_reference_image_shape_list,
                    per_ti2i_caption,
                    per_expect_reference_image_num,
                ])

            per_set_folder_count += 1

        set_folder_count_dict[per_set_name] = per_set_folder_count

    return edit_pair_save_folder_pair_list, set_folder_count_dict


def resize_single_image(per_image, per_save_image_shape):
    """把BGR的numpy图像resize到指定的[宽, 高]，返回resize后的BGR numpy图像

    按方案确认用PIL的LANCZOS而不是cv2.resize(与017.0/017.1口径一致):
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

    上游图像99.89%本来就是jpg(全库只有25808张png)，这里统一重编码成jpg，
    只换编码格式不换像素尺寸。
    编码参数按方案用cv2.imencode('.jpg', img)的默认值(质量95 + 色度4:2:0)，
    与001~006这几个已产出的数据集口径完全一致:
    本数据集不是图像复原任务、编辑后图也不是无损GT，
    没有必要像007/008那样上q97+4:4:4再多占20%以上的体积。
    编辑后图和参考图共用同一套参数，避免两条编码链路引入不对称的伪偏差。
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
            cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
        except Exception as e:
            print('8888', save_image_path, e)
            return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_edit_pair(edit_pair_save_folder_pair, save_dataset_path):
    """重新编码保存单个图像编辑对的编辑后图像和所有参考图像，任意一张失败则整对丢弃

    这个编辑对独占的子文件夹也在这里建(1114万个目录在主进程里串行建太慢)，
    建目录失败同样算整对失败、不会静默落到上一层目录里。
    """
    per_set_name, per_folder_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_save_reference_image_shape_list, per_ti2i_caption, per_expect_reference_image_num = edit_pair_save_folder_pair

    per_pair_folder_path = os.path.join(save_dataset_path, per_set_name,
                                        per_folder_name,
                                        per_save_pair_folder_name)

    try:
        os.makedirs(per_pair_folder_path, exist_ok=True)
    except Exception as e:
        print('8888', per_pair_folder_path, e)
        return None

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
        per_ti2i_caption,
        per_expect_reference_image_num,
    ]


def save_all_folder_annotation_json(save_result_list, save_dataset_path,
                                    set_folder_count_dict):
    """按文件夹汇总标注并写出与文件夹同名的json文件

    每条标注固定只有SAVE_ANNOTATION_KEY_NAME_LIST这七个key，
    上游标注里剩下的属性(category_id、三维质量分、源图URL三件套、
    上游记录的宽高、定位用的parquet_name/row_index等)全部丢弃，
    理由见文件开头SAVE_ANNOTATION_KEY_NAME_LIST的注释。
    ti2i_caption写的就是上游edit_instruction归一化(strip + 剥一层外层引号)后的原文。
    """

    folder_annotation_dict = {}
    reference_image_num_mismatch_count = 0
    for per_save_result in save_result_list:
        per_folder_name, per_save_edited_image_name, per_save_reference_image_name_list, per_edited_image_w, per_edited_image_h, per_ti2i_caption, per_expect_reference_image_num = per_save_result
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

        # ti2i_caption_length直接取即将写进json的这个字符串的长度，
        # 保证记录的长度和ti2i_caption永远自洽(该字符串已strip并归一化过)
        folder_annotation_dict[per_folder_name][per_save_edited_image_name] = {
            'reference_image': per_save_reference_image_name_list,
            'edited_image': per_save_edited_image_name,
            'reference_image_num': per_reference_image_num,
            'width': per_edited_image_w,
            'height': per_edited_image_h,
            'ti2i_caption': per_ti2i_caption,
            'ti2i_caption_length': len(per_ti2i_caption),
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


def check_single_save_folder(folder_check_pair, save_dataset_path):
    """校验单个文件夹: 文件夹容量、json与磁盘一一对应、图像名与指令合规

    每个子集除最后一个文件夹外都必须是满10000对，json里的每个key都必须在磁盘上有
    对应的样本对文件夹且文件恰好等于编辑后图像 + 所有参考图像，磁盘上也不允许有
    json没记录的残留样本对文件夹。另外还要复检ti2i_caption: 占位符编号集合必须与
    reference_image这个list的长度自洽、长度必须在阈值区间内、不能是null字面量或
    只剩标点的无意义指令、记录的长度必须与字符串实际长度一致。

    校验内容与006/007完全一致，只是把"按子集串行遍历"改成了"按文件夹开多进程":
    本数据集约1120个文件夹、合计1114万个样本对文件夹，每个都要listdir一次，
    串行跑在NAS上要跑非常久。
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
        if per_annotation['reference_image_num'] != len(
                per_annotation['reference_image']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image num not match')
        # 本数据集全部23个子集都必须是单参考图
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
        # ti2i_caption的占位符编号集合必须与reference_image这个list的长度自洽:
        # 单参考图即要求指令里完全没有占位符
        if check_invalid_caption(per_annotation['ti2i_caption'],
                                 len(per_annotation['reference_image'])):
            check_error_message_list.append(
                f'{per_save_edited_image_name} caption placeholder index not match reference image num {len(per_annotation["reference_image"])}'
            )
        # 落盘后的指令里不允许再残留null字面量或"不做任何修改"这类无意义指令
        if check_null_like_caption(per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} still a null like caption')
        # 也不允许残留只剩标点、没有任何数字/字母/汉字的指令
        if not CAPTION_WORD_CHAR_PATTERN.search(
                per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} still a no word char caption')
        # json里存的就是归一化后的指令，长度过滤也是按归一化后判定的，
        # 两者口径一致，这里直接量json里的长度复检
        if len(per_annotation['ti2i_caption'].strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_edited_image_name} still an invalid caption')
        if len(per_annotation['ti2i_caption'].strip()) > MAX_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_edited_image_name} still a too long caption')
        # 记录的指令长度必须和指令字符串的实际长度对得上
        if per_annotation['ti2i_caption_length'] != len(
                per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} ti2i caption length not match')

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

        # 真解一次落盘后的reference_image[0]，硬校验它的shape严格等于
        # json里的width/height(也就是编辑后图的宽高)。
        # 这一条是"第一张参考图与编辑后图尺寸必须一致"这个核心不变式的最终
        # 验收: 只对账json里的数字是查不出resize有没有真的生效的。
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
    """全部落盘后的收尾自校验: 文件夹容量、json与磁盘一一对应、图像名与指令合规

    约1120个文件夹每个都要load一份json、再对里面的每个样本对listdir一次，
    串行跑在NAS上太久，所以这一步按文件夹粒度开多进程。
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
    如果输出目录里还留着上一轮(未resize口径)跑出来的老图，
    这个短路会跳过写盘、但仍然返回内存里resize之后的shape，
    结果json记的宽高与磁盘上的实际文件不一致、收尾自校验也会大面积报错。
    所以本次改动之后这几个数据集必须落到全新的输出目录(或先手动删掉旧目录)重跑，
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
    # 必须落到全新的输出目录: 旧产物是未resize口径的，增量跑会让json宽高与磁盘错位
    check_save_dataset_path_empty(save_dataset_path)
    os.makedirs(save_dataset_path, exist_ok=True)

    edit_annotation_pair_list, total_annotation_file_count, subset_annotation_file_count_dict, subset_annotation_count_dict, set_annotation_count_dict, total_annotation_count, illegal_line_count, missing_subset_name_count, url_source_only_count, missing_image_count, invalid_reference_image_num_count, invalid_save_image_name_count, empty_caption_count, null_like_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, invalid_placeholder_caption_count, wrap_quote_caption_count, skip_set_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          illegal_line_count, missing_subset_name_count, url_source_only_count,
          skip_set_count, missing_image_count,
          invalid_reference_image_num_count, invalid_save_image_name_count,
          empty_caption_count, null_like_caption_count,
          no_word_char_caption_count, too_short_caption_count,
          too_long_caption_count, invalid_placeholder_caption_count,
          wrap_quote_caption_count, len(set_annotation_count_dict),
          len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        subset_annotation_file_count_dict, subset_annotation_count_dict,
        total_annotation_count, total_annotation_file_count,
        edit_annotation_pair_list, skip_set_count)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        raise Exception(
            f'check load annotation count error num {len(load_annotation_check_error_message_list)} {load_annotation_check_error_message_list[:20]}'
        )

    check_edit_annotation_pair_list = []
    invalid_image_count, different_aspect_ratio_count = 0, 0
    set_different_aspect_ratio_count_dict = {}
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result, per_check_set_name, per_check_edit_annotation_pair in tqdm(
                pool.imap(process_single_edit_pair_check,
                          edit_annotation_pair_list),
                total=len(edit_annotation_pair_list)):
            if per_check_result == 'invalid_image':
                invalid_image_count += 1
                continue
            # reference_image[0]与编辑后图长宽比不同的样本对在这里整对丢弃，
            # 逐子集记一份数字，方便看清是哪些任务类型天生对不齐
            if per_check_result == 'different_aspect_ratio':
                different_aspect_ratio_count += 1
                set_different_aspect_ratio_count_dict[
                    per_check_set_name] = set_different_aspect_ratio_count_dict.get(
                        per_check_set_name, 0) + 1
                continue
            check_edit_annotation_pair_list.append(
                per_check_edit_annotation_pair)

    print('1111', len(check_edit_annotation_pair_list), invalid_image_count,
          different_aspect_ratio_count)

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
          illegal_line_count, 'missing subset name:',
          missing_subset_name_count, 'url source only:', url_source_only_count,
          'skip set:', skip_set_count, 'skip url source only pair(upstream):',
          EXPECTED_SKIP_URL_SOURCE_ONLY_PAIR_COUNT, 'missing image:',
          missing_image_count, 'invalid reference image num:',
          invalid_reference_image_num_count, 'invalid save image name:',
          invalid_save_image_name_count, 'empty caption:', empty_caption_count,
          'null like caption:', null_like_caption_count,
          'no word char caption:', no_word_char_caption_count,
          'too short caption:', too_short_caption_count, 'too long caption:',
          too_long_caption_count, 'invalid placeholder caption:',
          invalid_placeholder_caption_count, 'wrap quote caption:',
          wrap_quote_caption_count, 'invalid image:', invalid_image_count,
          'different aspect ratio:', different_aspect_ratio_count,
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
        'total_upstream_row_count': EXPECTED_TOTAL_UPSTREAM_ROW_COUNT,
        'skip_url_source_only_pair_count':
        EXPECTED_SKIP_URL_SOURCE_ONLY_PAIR_COUNT,
        'total_annotation_file_count': total_annotation_file_count,
        'total_annotation_count': total_annotation_count,
        'illegal_line_count': illegal_line_count,
        'missing_subset_name_count': missing_subset_name_count,
        'url_source_only_count': url_source_only_count,
        # 按方案整体丢弃的子集(part_extraction)的条数与子集名
        'skip_set_count': skip_set_count,
        'skip_set_name_list': SKIP_SET_NAME_LIST,
        'missing_image_count': missing_image_count,
        'invalid_reference_image_num_count': invalid_reference_image_num_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
        'wrap_quote_caption_count': wrap_quote_caption_count,
        'invalid_image_count': invalid_image_count,
        # reference_image[0]与编辑后图长宽比不同而被整对丢弃的条数。
        # 第一轮跑完后可以把实测值回填成EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT
        # 再上硬对账，守住"哪些样本被resize对齐、哪些被丢弃"这条口径
        'different_aspect_ratio_count': different_aspect_ratio_count,
        'set_different_aspect_ratio_count_dict':
        set_different_aspect_ratio_count_dict,
        # 落盘口径标记，便于下游一眼看出这份产物是不是"参考图已对齐"的版本
        'resize_reference_image_to_edited_image_shape_flag': True,
        'long_side_align_extra_reference_image_flag': True,
        'exempt_aspect_ratio_align_set_name_list':
        EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST,
        'check_save_reference_image_shape_flag':
        CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG,
        'save_edit_pair_failed_count': save_edit_pair_failed_count,
        'total_save_edit_pair_count': len(save_result_list),
        'total_save_reference_image_count': total_save_reference_image_count,
        'reference_image_num_mismatch_count':
        reference_image_num_mismatch_count,
        'total_save_set_count': len(set_folder_count_dict),
        'total_save_folder_count': len(folder_edit_pair_count_dict),
        'check_total_edit_pair_count': check_total_edit_pair_count,
        'check_error_count': len(check_error_message_list),
        'subset_annotation_file_count_dict': subset_annotation_file_count_dict,
        'subset_annotation_count_dict': subset_annotation_count_dict,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/ScaleEdit-12M'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
