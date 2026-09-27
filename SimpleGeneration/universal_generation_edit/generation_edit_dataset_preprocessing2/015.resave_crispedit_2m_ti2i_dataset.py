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

DATASET_NAME = 'crispedit'

SAVE_DATASET_DIR_NAME = 'CrispEdit-2M'

# ==============================================================================
# 【这个数据集只能产出图像编辑数据集，不能产出文生图数据集】
# 上游016解包出来的每一行只有14个key，其中唯一的文本字段是instruction(英文编辑指令)，
# 形如"Add the lid back onto the jar of white cream"、
# "change the color of skirt to yellow"。
# 整个数据集**没有任何一列是整图内容描述(caption)**: 既没有编辑前原图的描述、
# 也没有编辑后图的描述。编辑指令只说"要改什么"、不说"整张图是什么"，
# 拿它当t2i的prompt会得到完全错误的图文对，所以本数据集只走ti2i这一条链路、
# 不另写t2i脚本(与014.resave_scaleedit_12m_ti2i_dataset.py同样的处置)。
# 上游016自己也把dataset_task_type写死成image_edit，
# HF README的task_categories是image-to-image、tags是image-editing。
# 唯一的例外是全库34条以"caption: "开头的整图描述串(add 15条 / remove 19条)，
# 那是上游构造流程串了字段的脏数据、且无法判断它描述的是编辑前还是编辑后图，
# 34条既不足以也不适合做t2i，本脚本按方案把它们当坏指令整对丢弃
# (见CAPTION_IMAGE_CAPTION_PREFIX_PATTERN)。
#
# 【上游016.unzip_crispedit_2m_dataset.py的产物规格(全量实测，非抽样)】
# CrispEdit-2M/
# ├── unzip_annotations/<7个任务族>/<分片>.jsonl   8971个文件 / 2286540行
# ├── unzip_images/<7个任务族>/<分片>/<分片>_%08d_input.jpg|png   参考图(编辑前原图)
# │                                 /<分片>_%08d_output.jpg|png  编辑后图
# │                                                4573080张(约4.6T)
# └── unzip_check_missing_images.json  上游自校验: 0隔离样本/0警告/0错误、
#                                     total_extract_image_count == 4573080全部落盘
# 上游逐任务族行数(合计2286540，与parquet footer硬对账过):
#   add 309253 / background_change 278475 / color 506730 / motion_change 32314 /
#   remove 353981 / replace 399329 / style 406458
# 图像后缀: jpg 3206880张 + png 1366200张
# (add/background_change/replace/style是JPEG，color/motion_change/remove是PNG)。
# 抽样17942张逐张PIL打开: mode 100%是RGB、0缺图、0坏图、
# 宽高与标注里的input_image_shape/edited_image_shape 100%一致。
# ==============================================================================

# 本脚本只读unzip_annotations这一套标注(8971个jsonl / 2286540行)，
# 每一行的参考图与编辑后图都已经落盘、是可直接训练的完整编辑对。
# **绝不os.walk图像目录**: 上游解出457万个小文件，扫目录树在NAS上不可接受
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 上游标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如unzip_images/add/add_00000/add_00000_00000000_input.jpg)，
# 不需要再往前拼任何子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 上游落盘图像名的后缀: 参考图是_input、编辑后图是_output
# (注意后缀前的扩展名可以是.jpg或.png，所以取前缀时必须先去扩展名再去这个后缀)
LOAD_REFERENCE_IMAGE_NAME_SUFFIX = '_input'

LOAD_EDITED_IMAGE_NAME_SUFFIX = '_output'

# 图像编辑任务类型。上游task_name是把parquet的type列里的空格归一成下划线得到的
# (background change -> background_change)，全库2286540行全非空、
# 且逐片与parquet文件名前缀严格一致，所以直接用它分子集、不需要mix兜底子集。
# 上游还有一个原值字段type(带空格的原始取值)，与task_name同义，本脚本不用
ANNOTATION_TASK_NAME_KEY_NAME = 'task_name'

# 编辑指令，也是本数据集唯一一个真正的文本字段。
# 全库2286540行全非空、无空串、无null字面量、无纯标点串、无[Vn*]占位符，
# 长度9~1882字符(p50=47 / p90=85 / p99=138 / p999=208)。
# 其中79条是中英混写(如"Add the word '爱你' beside the rabbit character.")，
# 按方案原样保留、不做任何语言过滤
ANNOTATION_CAPTION_KEY_NAME = 'instruction'

# 参考图相对路径(list，本数据集恒为长度1)与编辑后图相对路径(字符串)
ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'reference_image_path_list'

ANNOTATION_EDITED_IMAGE_KEY_NAME = 'edited_image_path'

# 上游记录的样本唯一键，形如"add/add_00000_00000000"(<任务族>/<分片名>_<片内行号>)。
# 只用来和从图像文件名现取的前缀交叉比对，不写进新标注
ANNOTATION_SAMPLE_KEY_KEY_NAME = 'sample_key'

# 上游解header得到的参考图/编辑后图真实宽高([宽, 高])。
# **只用于统计与上报，绝不采信**: 写进json的width/height一律取自实际写盘数组的shape
ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME = 'input_image_shape'

ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME = 'edited_image_shape'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 新标注固定只存这七个key，多一个少一个都在收尾自校验里报错。
# 上游jsonl里剩下的属性按方案全部丢弃、不另存索引:
#   dataset_task_type : 恒为image_edit(整库单值、已由本脚本的落盘位置体现)
#   type              : task_name的原始值(带空格，如"background change")，
#                       与task_name同义，只是空格未归一
#   sample_key        : 只用于和图像名前缀交叉校验，信息已内含在保存图像名里
#   parquet_name      : 分片名，已内含在保存图像名里
#   row_index         : 片内行号，已内含在保存图像名里
#   reference_image_num: 恒为1，不采信上游值、由reference_image这个list的长度现算
#   input_image_shape / edited_image_shape:
#                       上游解header得到的宽高(抽样17942张与磁盘100%一致)，
#                       但仍只用于统计与尺寸对齐预筛; 写进json的width/height
#                       一律取自实际写盘数组的shape
#   input_image_suffix / edited_image_suffix:
#                       .jpg或.png，只用于决定jpg重编码口径(见
#                       SAVE_IMAGE_JPEG_QUALITY的注释)，不写进json
# 上游的unzip_check_missing_images.json也不读不搬(它的实测值已写死成本脚本的
# EXPECTED_*常量用于硬对账)
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# 本数据集每个编辑对只有_input这一张参考图(上游parquet只有input_img/output_img
# 两个图像列、没有第二视觉条件图也没有mask)，
# 所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 【按方案确认整体丢弃的子集】motion_change(32314对):
# 这个子集99.4%的样本(32108对)都是1820x1024->1813x1024这一种尺寸变换，
# 而实测这种变换是"裁剪 + x/y独立拉伸"的复合变换
# (SIFT+RANSAC实测sx=1.0196 / sy=0.9792，x方向裁掉2.3%、y方向反而外扩2.1%)，
# 没有任何单一resize或裁剪能把参考图与编辑后图还原成逐像素对应，
# 只能整对丢弃。扣掉这批之后该子集只剩29对(长宽比严格相等的那一小撮)，
# 连一个文件夹都填不满、也不足以支撑任务学习，所以按方案整体丢弃该子集。
# 丢弃判定排在所有指令与图像过滤之前，所以后面每一项过滤计数都只统计保留的6个子集
SKIP_SET_NAME_LIST = [
    'motion_change',
]

# 最终产出的6个子集(即6种图像编辑任务类型)，必须与产出目录严格一一对应
SAVE_SET_NAME_LIST = [
    'add',
    'background_change',
    'color',
    'remove',
    'replace',
    'style',
]

# 找不到任务类型时才用的兜底子集名。
# 本数据集task_name全库非空且只有7个取值，实测一条都不会落进mix，
# 保留这条路径只为与005/014口径一致，并防止上游之后新增任务族时被静默漏处理
MIX_SET_NAME = 'mix'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 本数据集保存名形如crispedit_add_add_00000_00000000_edited.jpg，
# 最长71字符(crispedit_background_change_background_change_00000_00000000_edited.jpg)，
# 远低于文件系统单文件名255字节的上限。
# 原图名前缀是<分片名>_<8位片内行号>，分片名全局唯一 + 片内行号唯一
# => 2286540个保存名100%唯一、0重名(落盘前还会再兜一道)
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 上游图像名前缀必须是<任务族名>_<5位分片号>_<8位片内行号>。
# 任务族名里可能带下划线(background_change / motion_change)，所以前缀用.+
VALID_IMAGE_NAME_PREFIX_PATTERN = re.compile(r'^.+_\d{5}_\d{8}$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃。
# 实测抽样17942张(参考图 + 编辑后图各半)全部是RGB、0张例外
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 上游每个任务族的标注文件数(即parquet分片数)，合计8971。
# 注意add是1213而不是1214: 上游HF仓库自身就缺了add_01211.parquet
# (上游016已用.cache/huggingface/trees仓库文件清单证实并写进白名单放行)
EXPECTED_SET_ANNOTATION_FILE_COUNT_DICT = {
    'add': 1213,
    'background_change': 1091,
    'color': 1984,
    'motion_change': 128,
    'remove': 1388,
    'replace': 1567,
    'style': 1600,
}

EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 8971

# 上游每个任务族的标注行数(全量实测，与上游016的parquet footer硬对账过)，
# 合计2286540。这个口径在"整体丢弃motion_change"之前统计，
# 所以motion_change那一项仍然是32314、保持上游原值
EXPECTED_SET_ANNOTATION_COUNT_DICT = {
    'add': 309253,
    'background_change': 278475,
    'color': 506730,
    'motion_change': 32314,
    'remove': 353981,
    'replace': 399329,
    'style': 406458,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = 2286540

# 被整体丢弃的motion_change的实测行数，解析阶段硬对账，守住"丢弃范围没被改动过"
EXPECTED_SKIP_SET_COUNT = 32314

# 文本层各类不合格指令的实测精确条数(全量实测)，解析阶段逐项硬对账。
# 【注意口径】这些计数**只统计保留的6个子集**: check_skip_set排在指令过滤之前，
# motion_change的样本根本走不到这里。
#   empty / null_like / no_word_char / invalid_placeholder 实测全为0
#   (上游指令全非空、无null字面量、无纯标点串、无[Vn*]占位符)，只作兜底;
#   image_caption_prefix 34条是以"caption: "开头的整图描述串(不是编辑指令);
#   absolute_position    7466条带"Position and size:"或"bbox:"这类绝对像素坐标
#                        (集中在add/remove)，按方案确认整对丢弃;
#   too_short 4条(最短9字符)、too_long 3条(>512字符)
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 0,
    'null_like_caption_count': 0,
    'no_word_char_caption_count': 0,
    'image_caption_prefix_caption_count': 34,
    'absolute_position_caption_count': 7466,
    'too_short_caption_count': 4,
    'too_long_caption_count': 3,
    'invalid_placeholder_caption_count': 0,
}

# 指令归一化(换行/连续空白 -> 单空格)真正改动过的条数，实测5622条，只统计不丢样本。
# 其中13条是含\n的多重编辑指令(形如"change the color of top to green\n
# change the color of skirt to yellow")，按方案确认归一成单空格后保留
EXPECTED_NORMALIZE_WHITESPACE_CAPTION_COUNT = 5622

# 6个保留子集经文本层过滤后的实测条数，合计2246719。
# 2286540 - 32314(motion_change) - 34 - 7466 - 4 - 3 = 2246719
EXPECTED_VALID_ANNOTATION_COUNT = 2246719

# 最终产出的子集数，必须与SAVE_SET_NAME_LIST严格一一对应
EXPECTED_SAVE_SET_COUNT = 6

PROCESS_NUM = 32

PER_FOLDER_EDIT_PAIR_NUM = 10000

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# ==============================================================================
# 【参考图与编辑后图的尺寸对齐规格】
# 编辑后图**原分辨率落盘、不做任何缩放**，它是这个样本对唯一的尺寸基准
# (json里的width/height就是它)。参考图按下面的规则对齐:
#
#   reference_image[0](编辑前原图):
#       长宽比与编辑后图严格相同               -> LANCZOS resize到编辑后图尺寸
#       长宽比不同但命中实测尺寸变换白名单     -> LANCZOS resize到编辑后图尺寸
#       长宽比不同且不在白名单里               -> **整个样本对丢弃**
#
#   reference_image[k>=1](第二张视觉条件图/主体图/物体图):
#       长宽比与编辑后图严格相同 -> resize到编辑后图尺寸
#       长宽比不同               -> 按长边对齐等比resize(不裁剪、不形变、不丢弃)
#   本数据集恒为单参考图，这条分支走不到，只为与004/005/014口径一致而保留。
#
# 【为什么这个数据集必须额外引入白名单(与004/005/014的唯一口径差异)】
# 本数据集有1070933对(46.8%)的参考图与编辑后图长宽比不严格相等，
# 但**绝大多数偏差都在2%以内**(99.98%在2%以内，典型如1365x1024->1359x1024)。
# 按004/005/014的纯严格口径会白扔近一半数据，所以必须先搞清楚这些几像素的差异
# 到底是什么变换。用SIFT+RANSAC估计参考图->编辑后图的仿射矩阵
# [[sx, 0, tx], [0, sy, ty]]逐类实测(该方法能严格区分两种假设:
# 整图各向异性拉伸 => sx == ew/iw 且 sy == eh/ih 且 tx == ty == 0;
# 等比缩放+裁剪 => sx == sy 且 tx/ty 为裁剪偏移)，结论是这1070933对分成两类:
#
#   (1) 真·尺寸规整(可救): 实测sx与ew/iw吻合到小数点后5位、tx == ty ≈ 0，
#       即上游只是把图resize了几像素。典型如1537x1024->1536x1024
#       (实测sx=0.99936 vs 期望0.99935)、1535x1024->1536x1024。
#       这类**直接LANCZOS resize到编辑后图尺寸就是逐像素对齐的**，
#       实测落盘后最大像素错位只有0.1~0.72像素。
#
#   (2) 裁剪+拉伸的复合变换(无解): 实测sx的**方向都是反的**。
#       典型如1365x1024->1359x1024: 期望sx=0.99560，实测sx=1.00925，
#       反推出的几何含义是"编辑后图1359像素宽的画面只对应参考图0~1346.5/1365
#       (98.65%)这一段"，即参考图右边被切掉了约18.5像素再放大回1359。
#       更糟的是1820x1024->1813x1024(motion_change的主体): sx=1.0196 / sy=0.9792，
#       x方向裁掉2.3%而y方向反而外扩2.1%，是x/y独立的复合变换。
#       这类**没有任何单一resize或裁剪能还原**:
#         直接resize      -> 全图水平错位18~43像素;
#         等比缩放+中心裁剪 -> sx == sy根本不成立、还额外引入居中偏移，
#                            实测梯度NCC(0.275)比直接resize(0.284)更差;
#         按实测仿射矩阵warpAffine -> 能对齐(梯度NCC 0.284->0.927)，但需要
#                            对每一对样本单独跑SIFT+RANSAC(约1.5秒/对 x 107万对
#                            ≈ 450 CPU小时)，且对style子集有30%+估不出来。
#       所以这类按方案确认整对丢弃。
#
# 关键是**光看尺寸差的绝对值区分不开这两类**: 1365->1359(-6)、1366->1359(-7)、
# 1820->1813(-7)都是差6~7像素却全是复合变换，而1537->1536(-1)、1533->1536(+3)
# 是真·尺寸规整。唯一可靠的判据是实测出来的sx方向，所以只能走"实测白名单"这条路。
#
# 【白名单是怎么标定出来的】
# 对每一种出现次数>=5的尺寸变换(覆盖1038648/1070933 = 97.0%的不等长宽比样本)
# 各抽6对真图跑SIFT+RANSAC，按"落盘后最大像素错位<=1.0像素"判定可否直接resize，
# 要求>=3个可判定样本且>=80%判为可对齐。共测18719对图，
# 标定出230种可对齐的尺寸变换。
# 再用**与标定不重叠的独立样本**(从每个jsonl尾部取，标定时取的是头部)复验一遍:
# 样本级纯度1674/1750 = 95.7%，其中7种尺寸变换复验不达标(ok/det<0.8)，
# 按方案把这7种从白名单里剔除(合计33148对，宁可少要也不让错位样本落盘)，
# 最终白名单223种、覆盖197428对。
# 白名单外的835732对(含未标定的低频尺寸变换)全部整对丢弃。
#
# 长宽比严格相等的那1213551对实测**输入输出尺寸100%完全相同**
# (全量扫描same_ar_diff_shape == 0)，所以对它们resize是恒等操作、零损失。
#
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
# 注意本数据集那835732对被丢弃的样本比"尺寸不一致"更坏: 它们resize后尺寸是一致的、
# 但画面内容错位18~43像素，连降级分支都触发不了，只会被当成对齐样本硬训，
# 所以必须在这里判掉而不是留给下游。
# ==============================================================================

# 参考图resize到目标尺寸时用的重采样方式。
# 按方案确认用PIL的LANCZOS而不是cv2.resize: 与项目既定口径保持一致
SAVE_IMAGE_RESIZE_RESAMPLING = Image.Resampling.LANCZOS

# 【实测标定出来的"参考图->编辑后图"尺寸变换白名单】
# key的格式是'<参考图宽>x<参考图高>-><编辑后图宽>x<编辑后图高>'。
# 命中这里的尺寸变换才允许把参考图直接resize到编辑后图尺寸(实测逐像素对齐、
# 落盘后最大错位0.72像素)，没命中且长宽比不等的一律整对丢弃。
# 标定与复验过程见上面尺寸对齐规格里的说明: 223种、覆盖197428对。
# 这份名单是**上游数据的客观属性**，上游产物不变则名单不变;
# 上游一旦换代必须重新标定(否则会静默放行错位样本)，
# 所以下面还有EXPECTED_WHITELIST_ALIGNED_COUNT这条硬对账兜底
ALIGNED_IMAGE_SHAPE_PAIR_LIST = [
    '1537x1024->1536x1024',
    '1535x1024->1536x1024',
    '1539x1024->1536x1024',
    '1533x1024->1536x1024',
    '1538x1024->1536x1024',
    '1536x1024->1537x1024',
    '1024x1535->1024x1536',
    '1541x1024->1536x1024',
    '1540x1024->1536x1024',
    '1545x1024->1536x1024',
    '1536x1024->1535x1024',
    '1532x1024->1536x1024',
    '1371x1024->1377x1024',
    '1546x1024->1536x1024',
    '1529x1024->1536x1024',
    '1024x1370->1024x1377',
    '1368x1024->1377x1024',
    '1531x1024->1536x1024',
    '1370x1024->1377x1024',
    '1372x1024->1377x1024',
    '1536x1024->1534x1024',
    '1536x1024->1539x1024',
    '1536x1024->1533x1024',
    '1024x1546->1024x1536',
    '1024x1545->1024x1536',
    '1536x1024->1543x1024',
    '1369x1024->1377x1024',
    '1379x1024->1377x1024',
    '1385x1024->1377x1024',
    '1526x1024->1536x1024',
    '1528x1024->1536x1024',
    '1024x1532->1024x1536',
    '1536x1024->1538x1024',
    '1381x1024->1377x1024',
    '1380x1024->1377x1024',
    '1536x1024->1542x1024',
    '1374x1024->1377x1024',
    '1024x1368->1024x1377',
    '1551x1024->1536x1024',
    '1378x1024->1377x1024',
    '1536x1024->1540x1024',
    '1376x1024->1377x1024',
    '1536x1024->1541x1024',
    '1536x1024->1530x1024',
    '1373x1024->1377x1024',
    '1548x1024->1536x1024',
    '1527x1024->1536x1024',
    '1536x1024->1547x1024',
    '1522x1024->1536x1024',
    '1549x1024->1536x1024',
    '1520x1024->1536x1024',
    '1024x1376->1024x1377',
    '1524x1024->1536x1024',
    '1387x1024->1377x1024',
    '1024x1375->1024x1377',
    '1382x1024->1377x1024',
    '1550x1024->1536x1024',
    '1024x1374->1024x1377',
    '1521x1024->1536x1024',
    '1375x1024->1377x1024',
    '1024x1382->1024x1377',
    '1386x1024->1377x1024',
    '1024x1383->1024x1377',
    '1384x1024->1377x1024',
    '1377x1024->1371x1024',
    '1024x1369->1024x1377',
    '1024x1373->1024x1377',
    '1536x1024->1546x1024',
    '1190x1024->1197x1024',
    '1024x1386->1024x1377',
    '1194x1024->1197x1024',
    '1200x1024->1197x1024',
    '1195x1024->1197x1024',
    '1203x1024->1197x1024',
    '1553x1024->1536x1024',
    '1192x1024->1197x1024',
    '1024x1204->1024x1197',
    '1388x1024->1377x1024',
    '1201x1024->1197x1024',
    '1204x1024->1197x1024',
    '1024x1202->1024x1197',
    '1024x1203->1024x1197',
    '1024x1191->1024x1197',
    '1024x1200->1024x1197',
    '1202x1024->1197x1024',
    '1198x1024->1197x1024',
    '1024x1195->1024x1197',
    '1205x1024->1197x1024',
    '1191x1024->1197x1024',
    '1199x1024->1197x1024',
    '1024x1526->1024x1536',
    '1193x1024->1197x1024',
    '1024x1190->1024x1197',
    '1024x1196->1024x1197',
    '1536x1024->1531x1024',
    '1894x1024->1895x1024',
    '1024x1199->1024x1197',
    '1377x1024->1372x1024',
    '1024x1552->1024x1536',
    '1024x1205->1024x1197',
    '1377x1024->1368x1024',
    '1536x1024->1532x1024',
    '1024x1549->1024x1536',
    '1536x1024->1545x1024',
    '1377x1024->1369x1024',
    '1206x1024->1197x1024',
    '1024x1189->1024x1197',
    '1536x1024->1526x1024',
    '1536x1024->1529x1024',
    '1377x1024->1379x1024',
    '1891x1024->1895x1024',
    '1536x1024->1544x1024',
    '1896x1024->1895x1024',
    '1377x1024->1378x1024',
    '2234x1024->2238x1024',
    '1377x1024->1373x1024',
    '1377x1024->1380x1024',
    '1893x1024->1895x1024',
    '1899x1024->1895x1024',
    '1897x1024->1895x1024',
    '1900x1024->1895x1024',
    '1536x1024->1528x1024',
    '1536x1024->1551x1024',
    '1898x1024->1895x1024',
    '1536x1024->1527x1024',
    '1536x1024->1548x1024',
    '1377x1024->1382x1024',
    '1536x1024->1552x1024',
    '1536x1024->1550x1024',
    '1024x1388->1024x1377',
    '1536x1024->1553x1024',
    '1024x2216->1024x2238',
    '1377x1024->1370x1024',
    '1536x1024->1521x1024',
    '1377x1024->1388x1024',
    '1536x1024->1523x1024',
    '1536x1024->1520x1024',
    '1377x1024->1384x1024',
    '1024x1206->1024x1197',
    '1536x1024->1525x1024',
    '1377x1024->1374x1024',
    '1377x1024->1385x1024',
    '1536x1024->1549x1024',
    '1377x1024->1381x1024',
    '1377x1024->1383x1024',
    '1377x1024->1376x1024',
    '1892x1024->1895x1024',
    '1519x1024->1536x1024',
    '1377x1024->1386x1024',
    '1377x1024->1387x1024',
    '1536x1024->1522x1024',
    '1901x1024->1895x1024',
    '1377x1024->1375x1024',
    '1197x1024->1195x1024',
    '1197x1024->1192x1024',
    '1197x1024->1194x1024',
    '1197x1024->1190x1024',
    '1197x1024->1196x1024',
    '1536x1024->1524x1024',
    '1367x1024->1377x1024',
    '2069x1024->2070x1024',
    '1197x1024->1204x1024',
    '1197x1024->1191x1024',
    '1197x1024->1203x1024',
    '1197x1024->1199x1024',
    '2072x1024->2070x1024',
    '2211x1024->2238x1024',
    '1197x1024->1205x1024',
    '1197x1024->1202x1024',
    '1197x1024->1193x1024',
    '1024x1893->1024x1895',
    '1197x1024->1198x1024',
    '2235x1024->2238x1024',
    '2247x1024->2238x1024',
    '2253x1024->2238x1024',
    '1024x1894->1024x1895',
    '1197x1024->1201x1024',
    '2239x1024->2238x1024',
    '1024x2218->1024x2238',
    '1197x1024->1206x1024',
    '2220x1024->2238x1024',
    '2218x1024->2238x1024',
    '2217x1024->2238x1024',
    '2242x1024->2238x1024',
    '2221x1024->2238x1024',
    '2226x1024->2238x1024',
    '2210x1024->2238x1024',
    '1024x1896->1024x1895',
    '2208x1024->2238x1024',
    '2243x1024->2238x1024',
    '1024x2072->1024x2070',
    '2071x1024->2070x1024',
    '2222x1024->2238x1024',
    '2232x1024->2238x1024',
    '1024x2219->1024x2238',
    '1024x1899->1024x1895',
    '1024x2071->1024x2070',
    '2215x1024->2238x1024',
    '2246x1024->2238x1024',
    '1895x1024->1891x1024',
    '1895x1024->1893x1024',
    '1024x2214->1024x2238',
    '2249x1024->2238x1024',
    '2250x1024->2238x1024',
    '2209x1024->2238x1024',
    '2229x1024->2238x1024',
    '2244x1024->2238x1024',
    '1895x1024->1897x1024',
    '1895x1024->1899x1024',
    '2241x1024->2238x1024',
    '1024x1898->1024x1895',
    '2254x1024->2238x1024',
    '2219x1024->2238x1024',
    '2230x1024->2238x1024',
    '2245x1024->2238x1024',
    '1895x1024->1900x1024',
    '1895x1024->1898x1024',
    '1024x1553->1024x1536',
    '1024x2073->1024x2070',
    '2231x1024->2238x1024',
    '2251x1024->2238x1024',
    '1024x2212->1024x2238',
    '1024x2224->1024x2238',
]

ALIGNED_IMAGE_SHAPE_PAIR_SET = set(ALIGNED_IMAGE_SHAPE_PAIR_LIST)

# 白名单里应有的尺寸变换种数，写死硬对账: 防止有人手改名单却没跟着改期望值
EXPECTED_ALIGNED_IMAGE_SHAPE_PAIR_NUM = 223

# 长宽比严格相等而走resize的实测对数(6个保留子集、文本与图像过滤后)
EXPECTED_SAME_ASPECT_RATIO_COUNT = 1213551

# 长宽比不等但命中白名单而走resize的实测对数
EXPECTED_WHITELIST_ALIGNED_COUNT = 197428

# 长宽比不等且不在白名单里、被整对丢弃的实测对数
EXPECTED_NOT_ALIGNED_COUNT = 835732

# 编辑后图短边<64或宽高比>8的实测对数(按上游标注的shape预筛，实测8对)。
# 实测短边<64的是0对，这8对全是宽高比>8的极端长条图
EXPECTED_BAD_EDITED_IMAGE_COUNT = 8

# 最终预计落盘的图像编辑对数(标注层与尺寸对齐口径):
# 2246719(文本过滤后) - 8(极端分辨率) - 835732(不可对齐) = 1410979。
# 实际落盘数还会因为"真解码图像时发现mode不是RGB或图坏了"再少一点点
# (抽样17942张全是RGB、0坏图，预计invalid_image_count为0)，
# 所以这个数字只写进报告、不上硬对账
EXPECTED_ALIGNED_ANNOTATION_COUNT = 1410979

# 6个保留子集经文本 + 尺寸对齐过滤后的实测对数，合计1410979
EXPECTED_SAVE_SET_EDIT_PAIR_COUNT_DICT = {
    'add': 268725,
    'background_change': 145893,
    'color': 245762,
    'remove': 307857,
    'replace': 201426,
    'style': 241316,
}

# 豁免尺寸对齐的子集: 这些子集的编辑后图与全部参考图**完全原样落盘**，
# 不判长宽比、不resize、不丢弃、也不做长边对齐。
# 本数据集没有任何子集需要豁免(全部子集都要求参考图与编辑后图像素对齐)，
# 所以这里是空列表(保留这个常量只为与001等脚本口径一致)
EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST = []

# 收尾自校验时是否真解一次reference_image[0]、硬校验它的shape等于json里的
# width/height。按方案确认置True: "第一张参考图与编辑后图尺寸必须一致"是核心诉求，
# 而只对账json里的数字是查不出resize有没有真的生效的，必须真解一次图。
# 代价是收尾自校验要多解约141万张参考图，在NAS上会明显变慢
CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG = True

# 指令长度阈值。全量实测2286540条: min 9 / p50 47 / p90 85 / p99 138 /
# p999 208 / max 1882。按方案取10/512(与003/004/014一致):
# 实测只砍掉短于10的4条(如"add a tie")与长于512的3条
MIN_CAPTION_LENGTH = 10

MAX_CAPTION_LENGTH = 512

# jpg重编码质量与色度采样方式。
# 按方案确认用"质量97 + 色度4:4:4(不下采样)"(与005/006/015.0一致)，
# 而不是cv2.imencode('.jpg', img)的默认值(质量95 + 色度4:2:0):
# 本数据集有1366200张源图是**PNG无损**(color/motion_change/remove三个任务族)，
# 占全部457万张的29.9%; 而且color(颜色编辑)与style(风格迁移)这两个子集
# 合计占落盘量的34.5%、对色度保真最敏感——cv2默认的4:2:0会把色度分辨率直接砍半，
# 正是这类任务最不能接受的损失。代价是体积约+50%，磁盘余量200T不构成约束。
# 编辑后图和参考图共用同一套参数，避免两条编码链路引入GT/参考图不对称的伪偏差
SAVE_IMAGE_JPEG_QUALITY = 97

SAVE_IMAGE_JPEG_SAMPLING_FACTOR = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444

SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST = [
    int(cv2.IMWRITE_JPEG_QUALITY),
    int(SAVE_IMAGE_JPEG_QUALITY),
    int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
    int(SAVE_IMAGE_JPEG_SAMPLING_FACTOR),
]

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 全量实测0条命中，只作兜底
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 指令归一化用: 把换行/制表/连续空格统一压成单空格。
# 实测5622条指令被这一步改动过，其中13条是含\n的多重编辑指令
# (形如"change the color of top to green\nchange the color of skirt to yellow")，
# 按方案确认归一成单空格后保留(语义是合法的多重编辑);
# 其余5609条只是指令内部有连续空格
CAPTION_WHITESPACE_PATTERN = re.compile(r'\s+')

# 以"caption:"开头的整图描述串(不是编辑指令)，实测34条(add 15 / remove 19)，
# 形如"caption: The image depicts a serene, wooded environment with lush
# greenery..."。这是上游构造流程串了字段的脏数据，
# 且无法判断它描述的是编辑前还是编辑后图，按方案确认整对丢弃
CAPTION_IMAGE_CAPTION_PREFIX_PATTERN = re.compile(r'^caption\s*[:：]', re.I)

# 带绝对像素坐标的指令，实测7466条(集中在add/remove)，形如
# "Add back the entire village... Approximate position and size: spans nearly
#  the full width... (bbox: [11,588,2626,1565] on 1589x2628)."
# 这类指令语义合法，但把**绝对像素坐标**写进了文本，而且坐标所依据的分辨率
# (bbox里那个"on 1589x2628")与本脚本落盘的分辨率并不一致，
# 模型学到的坐标与实际画面对不上，属于有害监督信号，按方案确认整对丢弃
CAPTION_ABSOLUTE_POSITION_PATTERN = re.compile(
    r'(?:approximate\s+)?position\s+and\s+size\s*[:：]|\bbbox\s*[:：]', re.I)

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)，与004/014口径一致。
# 分两类: null字面量(上游构造流程失败时把空值写成了字符串)、
# 明确表示"不做任何修改"的指令(编辑前后图应该几乎相同，当训练样本是纯噪声)。
# 全量实测0条命中，这里只作兜底
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
# 这套写法与002/004/005/014完全一致，保证跨数据集口径统一。
# 本数据集恒为1张参考图(N == 0)，即指令里不允许出现任何占位符
# (全量实测0条命中)，这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2


def check_skip_set(per_set_name):
    """判定这个子集是不是要整体丢弃的子集，返回True表示丢弃

    见SKIP_SET_NAME_LIST的注释: motion_change有99.4%的样本是
    "裁剪 + x/y独立拉伸"的复合变换、参考图与编辑后图无法逐像素对齐，
    剩下的29对也不足以支撑任务学习，所以在解析阶段就整体跳过。
    这个判定必须排在所有指令与图像过滤之前，保证后面每一项过滤计数
    都只统计保留下来的6个子集。
    """
    return per_set_name in SKIP_SET_NAME_LIST


def get_set_name(per_task_name):
    """把上游task_name映射成子集名(即图像编辑任务类型)

    上游task_name已经是归一化后的小写下划线风格(background change在上游016里
    就被归一成了background_change)，这里只做strip + 小写 + 空格转下划线兜一道。
    实测7个取值归一化后仍是7个、没有任何两个不同的task_name被归并到同一个子集名。
    取不到任务类型时才落进mix(实测0条，只为与005/014口径一致而保留)。
    """
    per_task_name = str(per_task_name).strip().lower().replace(' ', '_')

    if not per_task_name:
        return MIX_SET_NAME

    return per_task_name


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    本数据集上游parquet只有input_img/output_img两个图像列、
    没有第二视觉条件图也没有mask，所以所有子集恒为1。
    保留这个函数是为了和004/005/014的收尾自校验保持同一套交叉对账写法。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令: strip + 把换行/制表/连续空格压成单空格

    实测5622条指令被这一步改动过，其中13条是含\\n的多重编辑指令
    (形如"change the color of top to green\\nchange the color of skirt to
    yellow")，按方案确认归一成单空格后保留(语义是合法的多重编辑);
    其余5609条只是指令内部有连续空格。
    换行留在指令里会让下游文本编码器把一条指令当成多段、
    也会让jsonl的可读性变差，所以必须在落盘前归一。

    归一化放在所有指令过滤之前，这样长度过滤判定的字符串就和最终写进json的
    字符串完全一致，收尾自校验直接量json里的长度就能复检。
    本数据集的指令里没有任何视觉参考图占位符，也没有"the reference image"
    这类自然语言指代(只有1张参考图、指令从不指代它)，所以不做任何占位符改写。
    已经带编号的占位符原样保留，不做任何改动(便于后续多参考图数据集复用本函数)。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    return CAPTION_WHITESPACE_PATTERN.sub(' ', per_ti2i_caption).strip()


def check_invalid_caption(per_ti2i_caption, per_reference_image_num):
    """判定占位符编号与参考图数量不自洽的坏指令，返回True表示这条指令不合格

    参考图里第0张永远是编辑前原图(隐式、不占编号)，所以一条指令应该带的占位符编号
    正好是1...N，其中N = reference_image_num - 1。这里做三条校验:
    1. 同一个编号最多重复2次，超过就是逐字符插占位符的坏指令;
    2. 最大编号必须正好等于N，多了就是指代了不存在的参考图;
    3. 1...N每个编号都必须至少出现一次，不允许跳号，也不允许有图没被指代。
    本数据集N恒为0，即要求指令里完全没有占位符(全量实测0条命中)。
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


def check_image_caption_prefix_caption(per_ti2i_caption):
    """判定指令是不是以"caption:"开头的整图描述串，返回True表示这条指令不合格

    实测34条(add 15 / remove 19)是上游构造流程串了字段的脏数据，
    内容是整图描述而不是编辑指令，且无法判断它描述的是编辑前还是编辑后图，
    按方案确认整对丢弃。
    """
    return CAPTION_IMAGE_CAPTION_PREFIX_PATTERN.match(
        str(per_ti2i_caption).strip()) is not None


def check_absolute_position_caption(per_ti2i_caption):
    """判定指令里有没有绝对像素坐标，返回True表示这条指令不合格

    实测7466条(集中在add/remove)带"Position and size:"或"bbox: [...]"
    这类绝对像素坐标，而坐标所依据的分辨率与本脚本落盘的分辨率并不一致，
    模型学到的坐标与实际画面对不上，按方案确认整对丢弃。
    """
    return CAPTION_ABSOLUTE_POSITION_PATTERN.search(
        str(per_ti2i_caption)) is not None


def get_image_name_prefix(per_image_relative_path, per_image_name_suffix):
    """从上游落盘图像的相对路径里取出原图名前缀

    上游图像名形如add_00000_00000000_input.jpg / add_00000_00000000_output.png，
    这里先去掉扩展名(.jpg或.png，两者都可能)、再去掉_input或_output后缀，
    剩下的就是<分片名>_<8位片内行号>这个前缀。
    后缀对不上时返回空串，由调用方按"保存名非法"整对丢弃并上报。
    """
    per_image_name = os.path.basename(
        str(per_image_relative_path).strip()).strip().lower()
    per_image_name_stem = os.path.splitext(per_image_name)[0]

    if not per_image_name_stem.endswith(per_image_name_suffix):
        return ''

    return per_image_name_stem[:-len(per_image_name_suffix)]


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上，逐样本打一次os.path.exists就是一次网络往返，
    228万个编辑对就是457万次。实测同一个jsonl里的图像全部落在同一个
    unzip_images/<任务族>/<分片名>/目录下(单目录最多256对、即512个文件)，
    所以这里按目录缓存一次os.listdir的结果，之后只做集合查表，
    网络往返次数从"图像张数"降到"分片目录数"(8971次)。
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


def process_single_annotation_file(annotation_file_pair):
    """解析单个上游jsonl标注，组装图像编辑对(参考图+编辑后图+编辑指令)的列表

    这一步只做纯文本层面的过滤(json坏行、任务类型缺失或与目录不自洽、
    整体丢弃的子集、缺图、原图名前缀非法、参考图数量不对、保存名非法、
    指令为空、指令是null字面量、指令没有任何文字字符、指令是整图描述串、
    指令带绝对像素坐标、指令过短、指令过长、指令是坏占位符指令)，
    图像本身的解码校验、分辨率过滤与尺寸对齐判定留到后面多进程里做。

    指令的归一化(strip + 空白压成单空格)放在所有指令过滤之前，
    这样长度过滤判定的字符串就和最终写进json的字符串完全一致，
    收尾自校验直接量json里的长度就能复检。

    整体丢弃的子集(motion_change)在统计完逐子集行数之后、
    任何指令与图像过滤之前就跳过，所以:
      - EXPECTED_SET_ANNOTATION_COUNT_DICT / EXPECTED_TOTAL_ANNOTATION_COUNT
        是上游原始口径(仍然包含motion_change);
      - EXPECTED_INVALID_CAPTION_COUNT_DICT里那些指令过滤计数
        只统计保留的6个子集。
    """
    per_annotation_path, per_dir_set_name, root_dataset_path = annotation_file_pair

    total_annotation_count, illegal_line_count = 0, 0
    missing_task_name_count = 0
    skip_set_count = 0
    missing_image_count = 0
    invalid_reference_image_num_count = 0
    invalid_save_image_name_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count = 0
    image_caption_prefix_caption_count = 0
    absolute_position_caption_count = 0
    too_short_caption_count, too_long_caption_count = 0, 0
    invalid_placeholder_caption_count = 0
    normalize_whitespace_caption_count = 0
    set_annotation_count_dict = {}
    set_valid_annotation_count_dict = {}
    edit_annotation_pair_list = []

    # 每个worker只处理一个jsonl，缓存里通常只有一个分片目录，内存开销可忽略
    dir_file_name_cache_dict = {}

    try:
        load_jsonl_file = open(per_annotation_path, 'r', encoding='UTF-8')
    except Exception as e:
        print('2222', per_annotation_path, e)

        return [
            edit_annotation_pair_list,
            per_dir_set_name,
            total_annotation_count,
            1,
            missing_task_name_count,
            skip_set_count,
            missing_image_count,
            invalid_reference_image_num_count,
            invalid_save_image_name_count,
            empty_caption_count,
            null_like_caption_count,
            no_word_char_caption_count,
            image_caption_prefix_caption_count,
            absolute_position_caption_count,
            too_short_caption_count,
            too_long_caption_count,
            invalid_placeholder_caption_count,
            normalize_whitespace_caption_count,
            set_annotation_count_dict,
            set_valid_annotation_count_dict,
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

            per_task_name = per_annotation.get(ANNOTATION_TASK_NAME_KEY_NAME,
                                               '')
            if not isinstance(per_task_name, str):
                per_task_name = ''
            per_task_name = per_task_name.strip()

            per_set_name = get_set_name(per_task_name)

            # 行内任务类型必须和这个标注文件所在的任务族目录一致: 不一致说明上游
            # 产物被搬动过，继续跑会把样本对写进错误子集(实测0条，这里只做防御)
            if not per_task_name or per_set_name != per_dir_set_name:
                missing_task_name_count += 1
                print('3333', per_annotation_path, per_task_name,
                      per_dir_set_name)
                continue

            set_annotation_count_dict[
                per_set_name] = set_annotation_count_dict.get(per_set_name,
                                                              0) + 1

            # 整体丢弃的子集在这里就跳过(见SKIP_SET_NAME_LIST的注释)。
            # 这一步刻意排在set_annotation_count_dict统计之后、
            # 任何指令与图像过滤之前: 逐子集条数硬对账用的是那个统计口径，
            # 所以上游侧的EXPECTED_*常量全部保持上游原值不变;
            # 而后面那些过滤计数则只统计保留下来的6个子集
            if check_skip_set(per_set_name):
                skip_set_count += 1
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

            # 保存图像名的原图名前缀取自编辑后图像的文件名(去掉扩展名和_output后缀)，
            # 即<分片名>_<8位片内行号>。这里从真正被读取的那张图现取前缀，
            # 再和标注里的sample_key字段交叉比对一次，保证保存名和图像严格对应。
            # 上游sample_key形如"add/add_00000_00000000"，取最后一段与前缀比
            per_edited_image_name_prefix = get_image_name_prefix(
                per_edited_image_relative_path, LOAD_EDITED_IMAGE_NAME_SUFFIX)

            per_sample_key = per_annotation.get(ANNOTATION_SAMPLE_KEY_KEY_NAME,
                                                '')
            if not isinstance(per_sample_key, str):
                per_sample_key = ''
            per_sample_key_name_prefix = per_sample_key.strip().replace(
                '\\', '/').split('/')[-1].strip().lower()

            if not VALID_IMAGE_NAME_PREFIX_PATTERN.match(
                    per_edited_image_name_prefix
            ) or per_edited_image_name_prefix != per_sample_key_name_prefix:
                invalid_save_image_name_count += 1
                print('3333', per_edited_image_path,
                      per_edited_image_name_prefix, per_sample_key_name_prefix)
                continue

            # 保存名形如crispedit_add_add_00000_00000000_edited.jpg。
            # 上游分片名全局唯一 + 片内行号唯一 => 原图名前缀本身就全局唯一，
            # 这里仍然按统一口径拼上数据集名与子集名(便于从文件名一眼看出来源)，
            # 实测2286540个保存名100%唯一、0重名(落盘前还会再兜一道)
            per_save_image_name_prefix = (
                f'{DATASET_NAME}_{per_set_name}_{per_edited_image_name_prefix}'
            )
            per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
            # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的
            # 前缀(即带_edited那一段)，和004/005/014的写法保持一致，
            # 收尾自校验也是按edited_image去掉.jpg来反推这个文件夹名的
            per_save_pair_folder_name = os.path.splitext(
                per_save_edited_image_name)[0]

            # 保存名里出现路径分隔符或其它异常字符会写坏目录结构，整对丢弃(实测0条)
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

                # 参考图与编辑后图必须是同一个样本(同一个原图名前缀)，
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

            # 归一化(strip + 空白压成单空格)放在所有指令过滤之前，
            # 保证过滤判定的字符串和写进json的字符串完全一致
            per_ti2i_caption = get_normalized_ti2i_caption(
                per_raw_ti2i_caption)
            if per_ti2i_caption != per_raw_ti2i_caption:
                # 只统计不丢样本: 实测5622条被压掉了换行或连续空格
                normalize_whitespace_caption_count += 1

            # 空指令、全空格指令视为不合格图像编辑对(全量实测0条，只作兜底)
            if not per_ti2i_caption:
                empty_caption_count += 1
                continue

            # null字面量与"不做任何修改"这类无意义指令同样丢弃(实测0条命中)
            if check_null_like_caption(per_ti2i_caption):
                null_like_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:50])
                continue

            # 只剩标点、没有任何数字/字母/汉字的指令也丢弃(实测0条)
            if not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption):
                no_word_char_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:50])
                continue

            # 以"caption:"开头的整图描述串不是编辑指令，整对丢弃(实测34条)
            if check_image_caption_prefix_caption(per_ti2i_caption):
                image_caption_prefix_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:50])
                continue

            # 带绝对像素坐标的指令整对丢弃(实测7466条)
            if check_absolute_position_caption(per_ti2i_caption):
                absolute_position_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:80])
                continue

            # 过短指令视为不合格图像编辑对(实测4条，最短9字符)
            if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
                too_short_caption_count += 1
                print('3333', per_edited_image_path, len(per_ti2i_caption))
                continue

            # 过长指令同样视为不合格图像编辑对(实测3条超过512字符)
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

            set_valid_annotation_count_dict[
                per_set_name] = set_valid_annotation_count_dict.get(
                    per_set_name, 0) + 1

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
        per_dir_set_name,
        total_annotation_count,
        illegal_line_count,
        missing_task_name_count,
        skip_set_count,
        missing_image_count,
        invalid_reference_image_num_count,
        invalid_save_image_name_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        image_caption_prefix_caption_count,
        absolute_position_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        normalize_whitespace_caption_count,
        set_annotation_count_dict,
        set_valid_annotation_count_dict,
    ]


def get_all_annotation_file_pair(root_dataset_path):
    """收集上游全部标注文件，返回[标注文件任务列表, 每个任务族的标注文件数]

    上游标注按任务族分目录、目录内按parquet分片分文件(实测7个任务族共8971个jsonl)，
    这里按"标注文件"这一粒度出任务，正好能把多进程铺满。
    """
    load_annotation_dir_path = os.path.join(root_dataset_path,
                                            LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    set_annotation_file_count_dict = {}
    for per_set_name in sorted(os.listdir(load_annotation_dir_path)):
        per_set_dir_path = os.path.join(load_annotation_dir_path, per_set_name)
        if not os.path.isdir(per_set_dir_path):
            continue

        per_set_annotation_file_name_list = sorted([
            per_annotation_file_name
            for per_annotation_file_name in os.listdir(per_set_dir_path) if
            per_annotation_file_name.endswith(LOAD_ANNOTATION_FILE_NAME_SUFFIX)
        ])

        for per_annotation_file_name in per_set_annotation_file_name_list:
            annotation_file_pair_list.append([
                os.path.join(per_set_dir_path, per_annotation_file_name),
                per_set_name,
                root_dataset_path,
            ])

        set_annotation_file_count_dict[per_set_name] = len(
            per_set_annotation_file_name_list)

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list, set_annotation_file_count_dict


def get_all_edit_annotation_pair(root_dataset_path):
    """按标注文件粒度多进程组装全部图像编辑对的列表

    上游有8971个标注文件、合计2286540行，逐行还要判2张图像文件是否存在，
    所以这里按标注文件开多进程解析，最后按保存的编辑后图像名统一排序。
    """
    annotation_file_pair_list, set_annotation_file_count_dict = get_all_annotation_file_pair(
        root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list),
          'annotation set:', len(set_annotation_file_count_dict))

    total_annotation_count, illegal_line_count = 0, 0
    missing_task_name_count = 0
    skip_set_count = 0
    missing_image_count = 0
    invalid_reference_image_num_count = 0
    invalid_save_image_name_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count = 0
    image_caption_prefix_caption_count = 0
    absolute_position_caption_count = 0
    too_short_caption_count, too_long_caption_count = 0, 0
    invalid_placeholder_caption_count = 0
    normalize_whitespace_caption_count = 0
    set_annotation_count_dict = {}
    set_valid_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            total_annotation_count += per_load_result[2]
            illegal_line_count += per_load_result[3]
            missing_task_name_count += per_load_result[4]
            skip_set_count += per_load_result[5]
            missing_image_count += per_load_result[6]
            invalid_reference_image_num_count += per_load_result[7]
            invalid_save_image_name_count += per_load_result[8]
            empty_caption_count += per_load_result[9]
            null_like_caption_count += per_load_result[10]
            no_word_char_caption_count += per_load_result[11]
            image_caption_prefix_caption_count += per_load_result[12]
            absolute_position_caption_count += per_load_result[13]
            too_short_caption_count += per_load_result[14]
            too_long_caption_count += per_load_result[15]
            invalid_placeholder_caption_count += per_load_result[16]
            normalize_whitespace_caption_count += per_load_result[17]

            for per_set_name, per_set_count in per_load_result[18].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

            for per_set_name, per_set_count in per_load_result[19].items():
                set_valid_annotation_count_dict[
                    per_set_name] = set_valid_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        set_annotation_file_count_dict,
        set_annotation_count_dict,
        set_valid_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        missing_task_name_count,
        skip_set_count,
        missing_image_count,
        invalid_reference_image_num_count,
        invalid_save_image_name_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        image_caption_prefix_caption_count,
        absolute_position_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        normalize_whitespace_caption_count,
    ]


def check_load_annotation_count(
        set_annotation_file_count_dict, set_annotation_count_dict,
        set_valid_annotation_count_dict, total_annotation_count,
        total_annotation_file_count, valid_annotation_count,
        edit_annotation_pair_list, skip_set_count,
        normalize_whitespace_caption_count, invalid_caption_count_dict,
        other_filter_count_dict):
    """解析完标注后按任务族硬对账，并检查保存图像名是否唯一

    上游标注是016一次性跑出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对。

    本数据集的指令过滤条数是**全量实测**(不是抽样)的，所以逐项都上了硬对账;
    这一点与014不同(那个只做过17.3万条抽样、没法写死)。
    """
    check_error_message_list = []

    # 每个任务族的标注文件数(即parquet分片数)
    for per_set_name in sorted(set_annotation_file_count_dict.keys()):
        if per_set_name not in EXPECTED_SET_ANNOTATION_FILE_COUNT_DICT:
            check_error_message_list.append(f'unknown set dir {per_set_name}')
            continue

        per_expect_annotation_file_count = EXPECTED_SET_ANNOTATION_FILE_COUNT_DICT[
            per_set_name]
        if set_annotation_file_count_dict[
                per_set_name] != per_expect_annotation_file_count:
            check_error_message_list.append(
                f'{per_set_name} annotation file count not match '
                f'{set_annotation_file_count_dict[per_set_name]} != '
                f'{per_expect_annotation_file_count}')

    for per_set_name in sorted(EXPECTED_SET_ANNOTATION_FILE_COUNT_DICT.keys()):
        if per_set_name not in set_annotation_file_count_dict:
            check_error_message_list.append(f'missing set dir {per_set_name}')

    # 每个任务族的标注行数(上游原始口径，仍然包含被整体丢弃的motion_change)
    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in EXPECTED_SET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown set {per_set_name}')
            continue

        per_expect_annotation_count = EXPECTED_SET_ANNOTATION_COUNT_DICT[
            per_set_name]
        if set_annotation_count_dict[
                per_set_name] != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_set_name} annotation count not match '
                f'{set_annotation_count_dict[per_set_name]} != '
                f'{per_expect_annotation_count}')

    for per_set_name in sorted(EXPECTED_SET_ANNOTATION_COUNT_DICT.keys()):
        if per_set_name not in set_annotation_count_dict:
            check_error_message_list.append(f'missing set {per_set_name}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

    # 按方案整体丢弃的motion_change的条数硬对账，守住"丢弃范围没被改动过"
    if skip_set_count != EXPECTED_SKIP_SET_COUNT:
        check_error_message_list.append(
            f'skip set count not match '
            f'{skip_set_count} != {EXPECTED_SKIP_SET_COUNT}')

    # 要整体丢弃的子集必须真的在上游标注里存在(不存在说明子集名写错了，
    # 那样丢弃就是空操作、motion_change会被静默保留下来)
    for per_skip_set_name in SKIP_SET_NAME_LIST:
        if per_skip_set_name not in set_annotation_count_dict:
            check_error_message_list.append(
                f'skip set {per_skip_set_name} not in upstream annotation')
        # 保留白名单与整体丢弃名单不允许有交集
        if per_skip_set_name in SAVE_SET_NAME_LIST:
            check_error_message_list.append(
                f'skip set {per_skip_set_name} also in save set name list')

    # 各类不合格指令的条数逐项硬对账(全量实测值)
    for per_count_name in sorted(EXPECTED_INVALID_CAPTION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_CAPTION_COUNT_DICT[per_count_name]
        if invalid_caption_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_caption_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    # 指令归一化真正改动过的条数(只统计不丢样本)
    if normalize_whitespace_caption_count != EXPECTED_NORMALIZE_WHITESPACE_CAPTION_COUNT:
        check_error_message_list.append(
            f'normalize whitespace caption count not match '
            f'{normalize_whitespace_caption_count} != '
            f'{EXPECTED_NORMALIZE_WHITESPACE_CAPTION_COUNT}')

    if valid_annotation_count != EXPECTED_VALID_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'valid annotation count not match '
            f'{valid_annotation_count} != {EXPECTED_VALID_ANNOTATION_COUNT}')

    # 【过滤链路恒等式自校验】总条数减去每一项被丢弃的条数必须正好等于保留条数。
    # 这一条不依赖任何硬编码的期望值，纯粹校验"各项计数之间自洽"，
    # 专门用来拦住"某个EXPECTED_*常量口径被改过/没跟着改"这类问题:
    # 上面那些逐项对账各自都可能因为口径漂移而误报或漏报，
    # 但只要这个恒等式不成立，就一定是计数逻辑或统计口径出了问题
    per_all_filter_count = (skip_set_count +
                            sum(invalid_caption_count_dict.values()) +
                            sum(other_filter_count_dict.values()))
    if total_annotation_count - per_all_filter_count != valid_annotation_count:
        check_error_message_list.append(
            f'annotation filter count not self consistent '
            f'{total_annotation_count} - {per_all_filter_count} != '
            f'{valid_annotation_count}')

    # 保留下来的子集集合必须与白名单严格一一对应，不允许多出任何一个子集
    for per_set_name in sorted(set_valid_annotation_count_dict.keys()):
        if per_set_name not in SAVE_SET_NAME_LIST:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_valid_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_valid_annotation_count_dict)} != '
            f'{EXPECTED_SAVE_SET_COUNT}')

    # 尺寸变换白名单的种数硬对账: 防止有人手改名单却没跟着改期望值，
    # 名单一旦被改动就会静默放行或误杀一批样本
    if len(ALIGNED_IMAGE_SHAPE_PAIR_LIST
           ) != EXPECTED_ALIGNED_IMAGE_SHAPE_PAIR_NUM:
        check_error_message_list.append(
            f'aligned image shape pair num not match '
            f'{len(ALIGNED_IMAGE_SHAPE_PAIR_LIST)} != '
            f'{EXPECTED_ALIGNED_IMAGE_SHAPE_PAIR_NUM}')
    # 名单里不允许有重复项(重复不影响判定结果但说明名单被手工改乱了)
    if len(ALIGNED_IMAGE_SHAPE_PAIR_SET) != len(ALIGNED_IMAGE_SHAPE_PAIR_LIST):
        check_error_message_list.append(
            f'aligned image shape pair has duplicate '
            f'{len(ALIGNED_IMAGE_SHAPE_PAIR_SET)} != '
            f'{len(ALIGNED_IMAGE_SHAPE_PAIR_LIST)}')
    # 名单里的每一项都必须是"<w>x<h>-><w>x<h>"且长宽比确实不相等
    # (长宽比相等的本来就会走同一条resize分支、不需要进白名单)
    for per_image_shape_pair in ALIGNED_IMAGE_SHAPE_PAIR_LIST:
        per_match_result = re.match(r'^(\d+)x(\d+)->(\d+)x(\d+)$',
                                    per_image_shape_pair)
        if not per_match_result:
            check_error_message_list.append(
                f'invalid aligned image shape pair {per_image_shape_pair}')
            continue
        per_reference_image_w, per_reference_image_h, per_edited_image_w, per_edited_image_h = [
            int(per_value) for per_value in per_match_result.groups()
        ]
        if Fraction(per_reference_image_w,
                    per_reference_image_h) == Fraction(per_edited_image_w,
                                                       per_edited_image_h):
            check_error_message_list.append(
                f'aligned image shape pair has same aspect ratio '
                f'{per_image_shape_pair}')

    # 保存的编辑后图像名必须全局唯一，撞名会让两个样本对在磁盘和json里互相覆盖
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
    样本被各向异性拉伸落盘。最简分数比是精确的、可复现的。

    本数据集里长宽比"几乎一样但不严格相等"的样本恰恰是最危险的那一批
    (实测大多是"裁剪 + 拉伸"的复合变换，画面内容根本对不上)，
    所以这里绝不能放容差，必须走实测白名单(见check_aligned_image_shape_pair)。
    """
    return Fraction(per_reference_image_shape[0],
                    per_reference_image_shape[1]) == Fraction(
                        per_edited_image_shape[0], per_edited_image_shape[1])


def check_aligned_image_shape_pair(per_reference_image_shape,
                                   per_edited_image_shape):
    """判定这个尺寸变换是否命中实测白名单，返回True表示可以直接resize

    两个入参都是[宽, 高]。
    命中白名单说明实测过"把参考图直接resize到编辑后图尺寸之后，
    画面与编辑后图逐像素对齐(最大错位<=0.72像素)"，
    也就是上游只是把图做了几像素的尺寸规整、没有裁剪也没有独立拉伸。
    没命中的尺寸变换一律按"不可对齐"整对丢弃，理由见文件头的尺寸对齐规格。
    """
    per_image_shape_pair = (f'{per_reference_image_shape[0]}x'
                            f'{per_reference_image_shape[1]}->'
                            f'{per_edited_image_shape[0]}x'
                            f'{per_edited_image_shape[1]}')

    return per_image_shape_pair in ALIGNED_IMAGE_SHAPE_PAIR_SET


def get_long_side_aligned_shape(per_reference_image_shape,
                                per_edited_image_shape):
    """按长边与编辑后图长边对齐，算出参考图应该被resize到的[宽, 高]

    只有reference_image[k>=1](第二张视觉条件图/主体图/物体图)在长宽比与编辑后图
    不同时才会走到这里: 这类参考图是独立主体/材质样例，本来就不要求与编辑后图
    像素对齐，硬resize到编辑后图尺寸会把画面拉伸变形，所以改成保持它自己的长宽比、
    只把长边缩放到与编辑后图长边相同(等比缩放、零形变、不裁剪)。
    本数据集恒为单参考图，这条分支走不到，只为与004/005/014口径一致而保留。
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

    本数据集口径(见文件头的尺寸对齐规格):
      长宽比与编辑后图严格相同           -> resize到编辑后图尺寸(等比缩放、零形变);
      长宽比不同但命中实测尺寸变换白名单 -> resize到编辑后图尺寸(实测逐像素对齐);
      长宽比不同且不在白名单里且是reference_image[0] -> 返回None，整对丢弃;
      长宽比不同且不在白名单里且是reference_image[k>=1] -> 按长边对齐resize
      (本数据集恒为单参考图，这条分支走不到)。
    """
    if check_same_image_aspect_ratio(per_reference_image_shape,
                                     per_edited_image_shape):
        return list(per_edited_image_shape)

    # 长宽比不等但实测过"直接resize就逐像素对齐"的尺寸变换，同样resize到编辑后图尺寸
    if check_aligned_image_shape_pair(per_reference_image_shape,
                                      per_edited_image_shape):
        return list(per_edited_image_shape)

    # 第一张参考图是编辑前原图，它必须与编辑后图像素对齐(这是编辑类样本的根本要求)。
    # 走到这里说明这个尺寸变换实测是"裁剪 + 拉伸"的复合变换、
    # 没有任何单一resize或裁剪能还原成逐像素对应，整对丢弃
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
      'not_aligned'            -> reference_image[0]与编辑后图长宽比不同且不在
                                  实测尺寸变换白名单里(整对丢弃)，第三项为None。
                                  带上子集名是为了在主流程里逐子集统计

    短边和宽高比的过滤按方案只以编辑后图像为准判定，参考图只要求能正常解码且
    mode命中白名单。

    这里之所以能零额外IO地判尺寸对齐: check_single_image本来就已经把编辑后图和
    每张参考图都真解码了一遍并返回了[宽, 高]，现在接住这些shape，
    既能判尺寸对齐、又能把"每张参考图的目标尺寸"一路带到落盘阶段，
    落盘时不必再重算一次。
    注意尺寸对齐判定用的是**真解码出来的shape**而不是上游标注里的shape,
    这样即使上游标注的宽高与磁盘不一致(抽样实测100%一致，但不做这个假设)，
    判定也仍然是对的。

    豁免子集(EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST)的目标尺寸一律给None，
    表示编辑后图与全部参考图都完全原样落盘。本数据集没有豁免子集。
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
        # 只有reference_image[0]既不同长宽比又不在白名单里才会拿到None，此时整对丢弃
        if per_save_reference_image_shape is None:
            return ['not_aligned', per_set_name, None]

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
    本数据集会产出6个子集、合计约141万对，预计切出约144个文件夹。

    和004/005不同的是: 每个图像编辑对自己那层子文件夹的os.makedirs放到
    process_single_edit_pair里由worker并行做(与014一致)。
    本数据集有约141万个样本对文件夹，在主进程里串行makedirs一遍，
    光这一步在NAS上就要跑很久，而且这段时间里32个worker全在空等。
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

    按方案确认用PIL的LANCZOS而不是cv2.resize(与项目既定口径一致):
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
    reference_image[0]一定是编辑后图尺寸(既不同长宽比又不在实测白名单里的样本对
    已经在那一步整对丢弃了)。

    上游图像是jpg 3206880张 + png 1366200张混着的，这里统一重编码成jpg，
    只换编码格式不换像素尺寸。
    编码参数显式用SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST(质量97 + 色度4:4:4)，
    而不是cv2的默认值(质量95 + 色度4:2:0): 本数据集有29.9%的源图是PNG无损，
    且color/style两个子集对色度保真最敏感，详见常量处的说明。
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

    这个编辑对独占的子文件夹也在这里建(约141万个目录在主进程里串行建太慢)，
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
    上游jsonl里剩下的属性(dataset_task_type、type、sample_key、parquet_name、
    row_index、reference_image_num、input_image_shape、edited_image_shape、
    input_image_suffix、edited_image_suffix)全部丢弃，
    理由见文件开头SAVE_ANNOTATION_KEY_NAME_LIST的注释。
    ti2i_caption写的就是上游instruction归一化(strip + 空白压成单空格)后的原文，
    中英混写的79条原样保留。
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
    只剩标点的无意义指令、不能是整图描述串或带绝对像素坐标的指令、
    不能残留换行或连续空格、记录的长度必须与字符串实际长度一致。

    校验内容与004/005完全一致，只是把"按子集串行遍历"改成了"按文件夹开多进程"
    (与014一致): 本数据集约144个文件夹、合计约141万个样本对文件夹，
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
        if per_annotation['reference_image_num'] != len(
                per_annotation['reference_image']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image num not match')
        # 本数据集全部6个子集都必须是单参考图
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
        # 不允许残留以caption:开头的整图描述串
        if check_image_caption_prefix_caption(per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} still an image caption prefix caption'
            )
        # 不允许残留带绝对像素坐标的指令
        if check_absolute_position_caption(per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} still an absolute position caption'
            )
        # 落盘后的指令里不允许再残留换行或连续空格(归一化必须真的生效)
        if per_annotation['ti2i_caption'] != get_normalized_ti2i_caption(
                per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} caption not normalized')
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

    约144个文件夹每个都要load一份json、再对里面的每个样本对listdir一次、
    还要真解一次reference_image[0]，串行跑在NAS上太久，
    所以这一步按文件夹粒度开多进程。
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
    所以这个数据集必须落到全新的输出目录(或先手动删掉旧目录)重跑，
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
    # 必须落到全新的输出目录: 增量跑会让json宽高与磁盘错位
    check_save_dataset_path_empty(save_dataset_path)
    os.makedirs(save_dataset_path, exist_ok=True)

    edit_annotation_pair_list, total_annotation_file_count, set_annotation_file_count_dict, set_annotation_count_dict, set_valid_annotation_count_dict, total_annotation_count, illegal_line_count, missing_task_name_count, skip_set_count, missing_image_count, invalid_reference_image_num_count, invalid_save_image_name_count, empty_caption_count, null_like_caption_count, no_word_char_caption_count, image_caption_prefix_caption_count, absolute_position_caption_count, too_short_caption_count, too_long_caption_count, invalid_placeholder_caption_count, normalize_whitespace_caption_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          illegal_line_count, missing_task_name_count, skip_set_count,
          missing_image_count, invalid_reference_image_num_count,
          invalid_save_image_name_count, empty_caption_count,
          null_like_caption_count, no_word_char_caption_count,
          image_caption_prefix_caption_count, absolute_position_caption_count,
          too_short_caption_count, too_long_caption_count,
          invalid_placeholder_caption_count,
          normalize_whitespace_caption_count,
          len(set_valid_annotation_count_dict), len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    invalid_caption_count_dict = {
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'image_caption_prefix_caption_count':
        image_caption_prefix_caption_count,
        'absolute_position_caption_count': absolute_position_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
    }

    # 除了子集丢弃与指令过滤之外剩下的几项丢弃计数，
    # 只给check_load_annotation_count里的过滤链路恒等式自校验用。
    # 这几项实测都是0，但必须一并算进恒等式，否则上游数据一旦出现缺图/缺任务类型，
    # 恒等式会误报成"计数不自洽"
    other_filter_count_dict = {
        'illegal_line_count': illegal_line_count,
        'missing_task_name_count': missing_task_name_count,
        'missing_image_count': missing_image_count,
        'invalid_reference_image_num_count': invalid_reference_image_num_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        set_annotation_file_count_dict, set_annotation_count_dict,
        set_valid_annotation_count_dict,
        total_annotation_count, total_annotation_file_count,
        len(edit_annotation_pair_list), edit_annotation_pair_list,
        skip_set_count, normalize_whitespace_caption_count,
        invalid_caption_count_dict, other_filter_count_dict)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        raise Exception(
            f'check load annotation count error num {len(load_annotation_check_error_message_list)} {load_annotation_check_error_message_list[:20]}'
        )

    check_edit_annotation_pair_list = []
    invalid_image_count, not_aligned_count = 0, 0
    set_not_aligned_count_dict = {}
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result, per_check_set_name, per_check_edit_annotation_pair in tqdm(
                pool.imap(process_single_edit_pair_check,
                          edit_annotation_pair_list),
                total=len(edit_annotation_pair_list)):
            if per_check_result == 'invalid_image':
                invalid_image_count += 1
                continue
            # reference_image[0]与编辑后图长宽比不同且不在实测白名单里的样本对
            # 在这里整对丢弃，逐子集记一份数字，方便看清是哪些任务类型天生对不齐
            if per_check_result == 'not_aligned':
                not_aligned_count += 1
                set_not_aligned_count_dict[
                    per_check_set_name] = set_not_aligned_count_dict.get(
                        per_check_set_name, 0) + 1
                continue
            check_edit_annotation_pair_list.append(
                per_check_edit_annotation_pair)

    print('1111', len(check_edit_annotation_pair_list), invalid_image_count,
          not_aligned_count)

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
          illegal_line_count, 'missing task name:', missing_task_name_count,
          'skip set:', skip_set_count, 'missing image:', missing_image_count,
          'invalid reference image num:', invalid_reference_image_num_count,
          'invalid save image name:', invalid_save_image_name_count,
          'empty caption:', empty_caption_count, 'null like caption:',
          null_like_caption_count, 'no word char caption:',
          no_word_char_caption_count, 'image caption prefix caption:',
          image_caption_prefix_caption_count, 'absolute position caption:',
          absolute_position_caption_count, 'too short caption:',
          too_short_caption_count, 'too long caption:', too_long_caption_count,
          'invalid placeholder caption:', invalid_placeholder_caption_count,
          'normalize whitespace caption:', normalize_whitespace_caption_count,
          'invalid image:', invalid_image_count, 'not aligned:',
          not_aligned_count, 'save edit pair failed:',
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
        'missing_task_name_count': missing_task_name_count,
        # 按方案整体丢弃的子集(motion_change)的条数与子集名
        'skip_set_count': skip_set_count,
        'skip_set_name_list': SKIP_SET_NAME_LIST,
        'missing_image_count': missing_image_count,
        'invalid_reference_image_num_count': invalid_reference_image_num_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'image_caption_prefix_caption_count':
        image_caption_prefix_caption_count,
        'absolute_position_caption_count': absolute_position_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
        'normalize_whitespace_caption_count':
        normalize_whitespace_caption_count,
        'invalid_image_count': invalid_image_count,
        # reference_image[0]与编辑后图长宽比不同且不在实测白名单里
        # 而被整对丢弃的条数(预期835732)
        'not_aligned_count': not_aligned_count,
        'set_not_aligned_count_dict': set_not_aligned_count_dict,
        # 落盘口径标记，便于下游一眼看出这份产物是不是"参考图已对齐"的版本
        'resize_reference_image_to_edited_image_shape_flag': True,
        'long_side_align_extra_reference_image_flag': True,
        # 本数据集特有的实测尺寸变换白名单口径(与004/005/014的唯一差异)
        'aligned_image_shape_pair_num': len(ALIGNED_IMAGE_SHAPE_PAIR_LIST),
        'expected_same_aspect_ratio_count': EXPECTED_SAME_ASPECT_RATIO_COUNT,
        'expected_whitelist_aligned_count': EXPECTED_WHITELIST_ALIGNED_COUNT,
        'expected_not_aligned_count': EXPECTED_NOT_ALIGNED_COUNT,
        'expected_bad_edited_image_count': EXPECTED_BAD_EDITED_IMAGE_COUNT,
        'expected_aligned_annotation_count': EXPECTED_ALIGNED_ANNOTATION_COUNT,
        'expected_save_set_edit_pair_count_dict':
        EXPECTED_SAVE_SET_EDIT_PAIR_COUNT_DICT,
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
        'save_image_jpeg_quality': SAVE_IMAGE_JPEG_QUALITY,
        'save_image_jpeg_sampling_factor_444_flag': True,
        'set_annotation_file_count_dict': set_annotation_file_count_dict,
        'set_annotation_count_dict': set_annotation_count_dict,
        'set_valid_annotation_count_dict': set_valid_annotation_count_dict,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/CrispEdit-2M'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
