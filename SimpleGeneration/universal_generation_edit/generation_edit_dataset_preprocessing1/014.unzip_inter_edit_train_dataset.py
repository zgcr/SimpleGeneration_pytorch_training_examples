import os
import re
import gzip
import json
import shutil
import struct
import tarfile
import collections

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: Inter-Edit-Train(Inter-Edit: First Benchmark for Interactive
#         Instruction-Based Image Editing, CVPR 2026 官方训练集)
#
# 【数据集类型】纯图像编辑(instruction-based image editing)数据集，不是文生图数据集。
# README原文 task_categories: image-to-image，tags: image-editing;
# 每个样本对固定是"1张参考图(编辑前source) + 1条编辑指令 + 1张编辑后图(target)
# + 1张编辑区域mask + 1个bounding_box(不精确空间提示)"，
# 这就是论文里的 I^3E(交互式指令编辑)任务，没有任何"只有caption + 一张图"的纯生成样本。
# 所以下游只能走ti2i_dataset.py那条链路，不能当t2i(文生图)数据用。
#
# 【root_dataset_path实测原始保存规格(共1.98T / 798个文件)】
# Inter-Edit-Train/
# ├── manifest.json      571B，官方规格清单(num_samples/分片数/12个字段名)   -> 有用
# │                      (当预检ground truth用，也随数据一起拷进训练目录)
# ├── metadata/          275个 train-{00000..00274}-of-00275.jsonl.gz (48M)  -> 有用
# ├── source_shards/     245个 source-{00000..00244}-of-00245.tar   (933G)   -> 有用
# ├── asset_shards/      275个 asset-{00000..00274}-of-00275.tar    (1.1T)   -> 有用
# ├── README.md          数据集说明(无用)
# ├── .gitattributes     git lfs配置(无用)
# └── .cache/            huggingface下载缓存，1623个文件，残留24个*.incomplete(无用)
#
# 完整性已用huggingface下载缓存里的仓库文件清单
# .cache/huggingface/trees/*.json 对账过: 清单里共798个文件
# (275个asset tar + 245个source tar + 275个metadata + manifest.json +
# README.md + .gitattributes)，**磁盘上全部存在且字节数100%一致**，
# 且磁盘上没有清单以外的多余文件，
# 所以.cache里那24个*.incomplete只是缓存残渣，数据集本体是完整的。
#
# 【metadata实测规格(1099964行**全量**扫过，不是抽样)】
# - 275片行数 = 274片 * 4000 + 末片3964 = 1099964，与manifest.json声明完全一致;
# - 12个字段**1099964/1099964条全部齐备**，无多余字段、无缺字段;
# - sample_id 恰好覆盖 0..1099963 无重复无缺号;
#   source_id 恰好覆盖 0..610185 无缺号(一张源图最多被3个样本对复用，
#   229986张只被用1次)，所以源图是**全局去重后单独发布**的;
# - 分片自洽性100%成立(这就是"不用打开tar也能知道每个成员该属于谁"的依据):
#     sample_id // 4000        == 所在metadata分片号 == asset分片号
#     source_id // 2500        == source分片号
#     target_file == f'targets/target_{sample_id:07d}.png'
#     mask_file   == f'masks/mask_{sample_id:07d}.png'
#     source_file == f'sources/source_{source_id:07d}.png'
#     asset_archive/source_archive 字段与上面两个分片号 100% 一致;
# - instruction 零空(624298条中文 + 475666条英文);
# - edit_type 只有4种: Local 408044 / Add 312325 / Remove 308050 / Texture 71545;
# - better_data 全部是bool: True 755312 / False 344652;
# - bbox_reference_dimensions 只有7种(README说的16:9到9:16七种常见长宽比)。
#
# 【单条metadata的全部12个字段(全部是有用信息)】
#   sample_id     : 全局唯一样本id(0..1099963)          -> 有用(样本唯一id/图像名)
#   source_id     : 全局唯一源图id(0..610185)           -> 有用(源图去重/复用统计)
#   edit_type     : Local/Add/Remove/Texture            -> 有用(按编辑类型采样/加权)
#   instruction   : 编辑指令(中英混合)                  -> 有用(训练主文本，必需)
#   better_data   : 过滤后判定"更适合训练"的标记         -> 有用(质量加权/子集筛选)
#   bounding_box  : [x1, y1, x2, y2] 不精确空间提示      -> 有用(I^3E任务的空间条件，必需)
#   bbox_reference_dimensions : {width, height}          -> 有用(bbox所在的参考坐标系)
#   source_archive / source_file : 参考图(编辑前图)位置  -> 有用(参考图，必需)
#   asset_archive / target_file  : 编辑后图位置          -> 有用(编辑后图，必需)
#   mask_file                    : 编辑区域mask位置      -> 有用(局部编辑监督/区域约束)
#
# 【tar内部实测规格】
# - asset tar: 没有顶层目录，成员按"target在前、mask在后"交替排列，
#   每片 4000个target + 4000个mask(末片3964对)，成员名恒为
#   targets/target_{sample_id:07d}.png 与 masks/mask_{sample_id:07d}.png;
#   target是8bit RGB PNG，mask是8bit灰度PNG且像素值实测只有{0,255}二值;
# - source tar: 每片 2500张(末片186张)，成员名恒为
#   sources/source_{source_id:07d}.png，8bit RGB PNG;
# - 抽检的4个tar长度全部512字节对齐且结尾1024字节EOF块完整。
#
# 【**必须显式感知的规格坑1: bbox参考尺寸 != 任何一张真实图的像素尺寸**】
# bbox_reference_dimensions 是7种960系分辨率(960x960 / 726x960 / 960x726 /
# 640x960 / 960x535 / 960x640 / 535x960)，
# 但实测真实PNG尺寸完全不是这些值(source常见1328x1328/1056x1584，
# target常见1024x1024/832x1248)，**只有长宽比一致**。
# 也就是说 bounding_box 是画在一个"参考坐标系"上的，
# 下游要用到 bbox 就**必须按比例换算到真实图尺寸**，直接拿去裁真实图一定是错的。
# 另外 mask 的尺寸实测有时等于target、有时等于source(两者本身也不相等)，
# 所以本脚本把 source/target/mask 三张图的**真实PNG宽高**(只解IHDR前26字节、
# 不解像素)全部写进标注，并额外存一份**归一化到[0,1]的bbox**，
# 让下游无论按哪张图训练都能自己乘回去，不需要二次扫盘解图。
#
# 【**必须显式感知的规格坑2: bbox越界与退化**】
# 全量统计: 16637条 x2/y2 超出参考宽高、1条负坐标、75条本身就退化(x1>=x2或y1>=y2)。
# 先把 bbox clip 到参考宽高(记 bounding_box_state='clipped')，
# **clip之后仍然退化的共107条**，这107条虽然三张图和指令都在，
# 但空间提示不可用、不是I^3E任务的完整样本对，
# 按与用户确认的口径**直接丢弃、不落盘、不进标注**，只把sample_id记进校验报告留痕。
# 丢弃这107条的连带影响也算清了: 有30张源图**只被这107条引用**，丢弃后会成为孤儿，
# 所以这30张源图也不落盘(省NAS空间)。
# 最终保存: 样本对 1099964 - 107 = **1099857**，
#           唯一源图 610186 - 30 = **610156**，
#           编辑后图/mask 各 **1099857**。
#
# 【本脚本的处理口径】
# - 阶段0 解压前硬预检(不过就直接抛异常，不白跑几十小时):
#   根目录条目白名单(多出未知条目立即上报)、manifest.json与实测规格逐项对账、
#   3个子目录分片数(275/245/275)与命名格式与0..N-1连号、
#   520个tar的512字节对齐 + 1024字节EOF块(O(1)只读尾部，Pool并行);
# - 阶段1 metadata全量解析(Pool 32 * 275片，48M gz，很快):
#   逐条校验12字段齐备/instruction非空/id与分片自洽/三个文件名格式自洽，
#   bbox先clip再判退化，产出"该落盘成员白名单"
#   (asset分片 -> 有效sample_id集合、source分片 -> 有效source_id集合);
#   全局硬对账 sample_id集合 == 0..1099963、source_id集合 == 0..610185、
#   有效对 + 丢弃对 == 1099964、edit_type/better_data/bbox参考尺寸分布
#   全部与实测ground truth逐项比对;
# - 阶段2 流式解包(Pool 32 * 520个tar，tarfile mode='r|'，**绝不整片进内存**):
#   只落盘白名单内的成员，白名单外(107个target + 107个mask + 30个source)
#   直接跳过并计drop; 每个成员写盘后**立刻校验落盘大小 == tar头里的size**;
#   已存在且大小一致就计skip并跳过，保证脚本可以断点续跑;
#   顺手只解PNG IHDR前26字节拿真实宽高(不解像素，代价可忽略)回填标注;
#   片内硬对账: tar里的成员名集合必须与metadata声明的集合**完全相等**
#   (白名单内一个不少、声明外一个不多)、target数 == mask数、
#   extract+skip+not_save+drop+fail == tar头里的成员总数、且必须读到tar流结束;
#   片内重名成员改写到 unzip_duplicate_members/ 独立目录保留数据，不互相覆盖;
# - 阶段3 标注汇总(Pool 32 * 275片): 重新流式读一遍metadata(只有48M)，
#   与阶段2拿到的真实宽高join后写 unzip_annotations/<metadata分片名>.jsonl，
#   **保留原12个字段的全部属性** + 三张图落盘相对路径 + 三张图真实宽高 +
#   clip后bbox + 归一化bbox + bounding_box_state; 每片写出的条数必须与阶段1
#   算出来的有效对数完全相等，且引用到的每张图都必须在阶段2里被处理过;
# - 无用信息(.cache/.gitattributes/README.md/.DS_Store/CACHEDIR.TAG)全部不拷贝;
# - 任何一环出错都汇总后抛异常，不再静默跑过。
#
# 【跑之前务必确认目标盘扛得住】
# - EXTRACT_IMAGE_FILE_FLAG=True 时输出小文件数约 **330万张png**
#   (1099857张target + 1099857张mask + 610156张source)，约2T，
# - 只想先建索引可把 EXTRACT_IMAGE_FILE_FLAG 置False，
#   图像继续留在原tar里，样本对信息(含真实宽高)一样是完整的。
# ==============================================================================

DATASET_TASK_TYPE = 'image_edit'

# README里没有声明license字段，只给了引用的论文，这里如实记录
DATASET_LICENSE_NAME = 'unknown(readme not declared)'

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(1623个文件，含24个*.incomplete)
# .gitattributes   git lfs配置
# README.md        数据集说明(规格信息已固化进本脚本的ground truth常量)
# .gitignore/.DS_Store/CACHEDIR.TAG  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    '.DS_Store',
    'CACHEDIR.TAG',
]

# 过滤掉无用信息后根目录只应该剩这3个子目录 + manifest.json
METADATA_ROOT_DIR_NAME = 'metadata'

SOURCE_ARCHIVE_ROOT_DIR_NAME = 'source_shards'

ASSET_ARCHIVE_ROOT_DIR_NAME = 'asset_shards'

SUBSET_ROOT_DIR_NAME_LIST = [
    METADATA_ROOT_DIR_NAME,
    SOURCE_ARCHIVE_ROOT_DIR_NAME,
    ASSET_ARCHIVE_ROOT_DIR_NAME,
]

MANIFEST_FILE_NAME = 'manifest.json'

# 分片文件名规格
METADATA_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>train-(?P<index>\d{5})-of-(?P<total>\d{5}))\.jsonl\.gz$')

SOURCE_ARCHIVE_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>source-(?P<index>\d{5})-of-(?P<total>\d{5}))\.tar$')

ASSET_ARCHIVE_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>asset-(?P<index>\d{5})-of-(?P<total>\d{5}))\.tar$')

# tar内成员名规格(全部是匿名化的index命名，没有顶层目录)
TARGET_MEMBER_NAME_PATTERN = re.compile(
    r'^targets/target_(?P<index>\d{7})\.png$')

MASK_MEMBER_NAME_PATTERN = re.compile(r'^masks/mask_(?P<index>\d{7})\.png$')

SOURCE_MEMBER_NAME_PATTERN = re.compile(
    r'^sources/source_(?P<index>\d{7})\.png$')

# 压缩包种类: asset(编辑后图 + mask) / source(参考图)
ARCHIVE_KIND_ASSET = 'asset'

ARCHIVE_KIND_SOURCE = 'source'

# 成员种类
MEMBER_KIND_TARGET = 'target'

MEMBER_KIND_MASK = 'mask'

MEMBER_KIND_SOURCE = 'source'

# 实测(也与manifest.json声明一致)的分片数与切分规格，数量不对说明下载不全
EXPECTED_METADATA_SHARD_NUM = 275

EXPECTED_ASSET_ARCHIVE_NUM = 275

EXPECTED_SOURCE_ARCHIVE_NUM = 245

EXPECTED_SAMPLE_NUM_PER_SHARD = 4000

EXPECTED_SOURCE_NUM_PER_SHARD = 2500

# 实测全量行数与唯一源图数，直接当作完整性ground truth
EXPECTED_TOTAL_SAMPLE_COUNT = 1099964

EXPECTED_TOTAL_SOURCE_IMAGE_COUNT = 610186

# clip后bbox仍退化、按用户确认口径直接丢弃的样本对数
EXPECTED_INVALID_SAMPLE_COUNT = 107

# 丢弃上面107条后最终保存的样本对数
EXPECTED_VALID_SAMPLE_COUNT = 1099857

# 只被那107条丢弃样本引用、丢弃后成为孤儿因而也不落盘的源图数
EXPECTED_ORPHAN_SOURCE_IMAGE_COUNT = 30

# 最终落盘的唯一源图数
EXPECTED_VALID_SOURCE_IMAGE_COUNT = 610156

# manifest.json里应该齐备的字段与实测值，对不上说明上游发布规格变了，必须显式感知
EXPECTED_MANIFEST_VALUE_DICT = {
    'dataset_name': 'Inter-Edit-Train',
    'num_samples': EXPECTED_TOTAL_SAMPLE_COUNT,
    'num_unique_source_images': EXPECTED_TOTAL_SOURCE_IMAGE_COUNT,
    'source_per_shard': EXPECTED_SOURCE_NUM_PER_SHARD,
    'samples_per_shard': EXPECTED_SAMPLE_NUM_PER_SHARD,
    'num_source_shards': EXPECTED_SOURCE_ARCHIVE_NUM,
    'num_sample_shards': EXPECTED_ASSET_ARCHIVE_NUM,
}

# metadata每行必须齐备的全部12个字段(实测1099964/1099964条齐备，多列少列都必须上报)
ANNOTATION_KEY_NAME_LIST = [
    'sample_id',
    'source_id',
    'edit_type',
    'instruction',
    'better_data',
    'bounding_box',
    'bbox_reference_dimensions',
    'source_archive',
    'source_file',
    'asset_archive',
    'target_file',
    'mask_file',
]

ANNOTATION_SAMPLE_ID_KEY_NAME = 'sample_id'

ANNOTATION_SOURCE_ID_KEY_NAME = 'source_id'

# 训练主文本 = instruction(编辑指令)，实测零空，为空则该样本对没有文本条件、不可训练
ANNOTATION_TEXT_KEY_NAME = 'instruction'

ANNOTATION_EDIT_TYPE_KEY_NAME = 'edit_type'

ANNOTATION_BETTER_DATA_KEY_NAME = 'better_data'

ANNOTATION_BOUNDING_BOX_KEY_NAME = 'bounding_box'

ANNOTATION_BBOX_REFERENCE_SHAPE_KEY_NAME = 'bbox_reference_dimensions'

ANNOTATION_SOURCE_ARCHIVE_KEY_NAME = 'source_archive'

ANNOTATION_SOURCE_FILE_KEY_NAME = 'source_file'

ANNOTATION_ASSET_ARCHIVE_KEY_NAME = 'asset_archive'

ANNOTATION_TARGET_FILE_KEY_NAME = 'target_file'

ANNOTATION_MASK_FILE_KEY_NAME = 'mask_file'

# 实测edit_type只有这4种取值，出现新取值必须显式感知
ANNOTATION_EDIT_TYPE_NAME_LIST = [
    'Local',
    'Add',
    'Remove',
    'Texture',
]

# 实测全量edit_type分布(合计1099964)，当完整性ground truth用
EXPECTED_EDIT_TYPE_COUNT_DICT = {
    'Local': 408044,
    'Add': 312325,
    'Remove': 308050,
    'Texture': 71545,
}

# 实测全量better_data分布(合计1099964)
EXPECTED_BETTER_DATA_COUNT_DICT = {
    'True': 755312,
    'False': 344652,
}

# 实测全量bbox参考尺寸分布(7种，合计1099964)。
# 注意这**不是**真实图像尺寸，只是bbox所在的参考坐标系，见文件头规格坑1
EXPECTED_BBOX_REFERENCE_SHAPE_COUNT_DICT = {
    '960x960': 545549,
    '726x960': 93019,
    '960x726': 92810,
    '640x960': 92688,
    '960x535': 92543,
    '960x640': 92179,
    '535x960': 91176,
}

# bbox状态: 原样可用 / clip到参考宽高后可用 / clip后仍退化(丢弃)
BOUNDING_BOX_STATE_NORMAL = 'normal'

BOUNDING_BOX_STATE_CLIPPED = 'clipped'

BOUNDING_BOX_STATE_DEGENERATE = 'degenerate'

# mask真实尺寸与哪张图一致(实测两种都有，只统计不判错，下游按需resize)
MASK_IMAGE_SHAPE_STATE_EQUAL_TARGET = 'equal_target'

MASK_IMAGE_SHAPE_STATE_EQUAL_SOURCE = 'equal_source'

MASK_IMAGE_SHAPE_STATE_OTHER = 'other'

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_SOURCE_IMAGE_DIR_NAME = 'sources'

SAVE_TARGET_IMAGE_DIR_NAME = 'targets'

SAVE_MASK_IMAGE_DIR_NAME = 'masks'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_DUPLICATE_MEMBER_DIR_NAME = 'unzip_duplicate_members'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

PNG_FILE_MAGIC_BYTES = b'\x89PNG\r\n\x1a\n'

PNG_IHDR_CHUNK_TYPE_BYTES = b'IHDR'

# PNG魔数8字节 + 长度4字节 + 'IHDR'4字节 + 宽高各4字节 = 24字节，多读2字节到位深/颜色类型
PNG_HEADER_MIN_SIZE = 26

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.bmp',
    '.gif',
    '.tif',
]

TAR_BLOCK_SIZE = 512

TAR_EOF_BLOCK_SIZE = 1024

# 长宽比一致性判定阈值: bbox参考尺寸与真实图尺寸只有长宽比一致(见规格坑1)，
# 超过这个阈值说明连长宽比都对不上，归一化bbox就不可信了，必须上报warning
IMAGE_ASPECT_RATIO_DIFF_THRESHOLD = 0.02

# 归一化bbox保留的小数位
NORMALIZED_BOUNDING_BOX_ROUND_NDIGITS = 6

# 图像成员是否落盘。
# True : 和其他数据集脚本口径一致，约330万张png(约2T)，
#        NAS上inode和元数据压力大，务必确认目标盘扛得住再跑;
# False: 只流式过一遍tar做成员对账并解PNG头拿真实宽高，
#        生成 unzip_annotations/*.jsonl 索引，图像继续留在原tar里，
#        训练时按tar顺序读，样本对信息一样是完整的。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解包后再os.walk一遍输出目录做二次对账。
# 默认False: 330万个小文件的os.walk在NAS上要跑非常久，而解包时已经做了
# "写盘后立刻校验落盘大小 == tar头里的size" + "成员名集合与metadata声明集合完全相等"
# 两道对账，已经能保证每个该落盘的成员都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 是否解PNG头拿真实宽高。
# 默认True: 只读前26字节不解像素，代价可忽略，
# 而这个数据集的bbox参考尺寸与真实图尺寸不一致(规格坑1)，
# 不存真实宽高下游就没法把bbox换算回去，等于有用信息没保存完整。
PARSE_IMAGE_SHAPE_FLAG = True

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024


def check_skip_file_or_dir(per_file_relative_path):
    """过滤掉.cache、.gitattributes、README.md这几个不需要整理的文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """只把图像后缀的文件计入落盘图像总数，其余一律当未知文件上报"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


def get_stripped_text_value(per_value):
    """文本字段统一转成strip后的字符串，None/非字符串/空白都当成缺失"""
    if not isinstance(per_value, str):
        return ''

    return per_value.strip()


def get_expected_shard_item_count(per_shard_index, per_item_num_per_shard,
                                  per_total_item_count):
    """按"每片固定条数、最后一片是余数"的切分规格算某一片应有的条目数

    实测 metadata: 274片*4000 + 末片3964 == 1099964,
         source  : 244片*2500 + 末片186  == 610186，两者都严格符合这个规则。
    """
    per_start_index = per_shard_index * per_item_num_per_shard
    if per_start_index >= per_total_item_count:
        return 0

    return min(per_item_num_per_shard, per_total_item_count - per_start_index)


def get_png_image_shape(per_image_bytes):
    """只解PNG的IHDR头拿真实宽高(不解像素)，拿不到返回None

    该数据集三类图实测全部是PNG，头部固定是
    8字节魔数 + 4字节长度 + 'IHDR' + 4字节宽 + 4字节高。
    """
    if not isinstance(per_image_bytes,
                      bytes) or len(per_image_bytes) < PNG_HEADER_MIN_SIZE:
        return None

    if not per_image_bytes.startswith(PNG_FILE_MAGIC_BYTES):
        return None

    if per_image_bytes[12:16] != PNG_IHDR_CHUNK_TYPE_BYTES:
        return None

    per_image_width, per_image_height = struct.unpack('>II',
                                                      per_image_bytes[16:24])
    if per_image_width <= 0 or per_image_height <= 0:
        return None

    return [int(per_image_width), int(per_image_height)]


def get_image_aspect_ratio_diff(per_image_shape, per_reference_shape):
    """算两个尺寸的长宽比差，任一尺寸不可用时返回None"""
    if not per_image_shape or not per_reference_shape:
        return None

    if per_image_shape[1] <= 0 or per_reference_shape[1] <= 0:
        return None

    return abs(per_image_shape[0] / per_image_shape[1] -
               per_reference_shape[0] / per_reference_shape[1])


def get_clipped_bounding_box(per_bounding_box, per_reference_shape):
    """把bbox clip到参考宽高内，返回[clip后bbox, bbox状态]

    实测16637条x2/y2越界、1条负坐标，clip后仍退化的107条按用户口径直接丢弃。
    """
    per_x1, per_y1, per_x2, per_y2 = per_bounding_box
    per_reference_width, per_reference_height = per_reference_shape

    per_clipped_x1 = min(max(per_x1, 0), per_reference_width)
    per_clipped_y1 = min(max(per_y1, 0), per_reference_height)
    per_clipped_x2 = min(max(per_x2, 0), per_reference_width)
    per_clipped_y2 = min(max(per_y2, 0), per_reference_height)

    per_clipped_bounding_box = [
        per_clipped_x1,
        per_clipped_y1,
        per_clipped_x2,
        per_clipped_y2,
    ]

    if per_clipped_x1 >= per_clipped_x2 or per_clipped_y1 >= per_clipped_y2:
        return per_clipped_bounding_box, BOUNDING_BOX_STATE_DEGENERATE

    if per_clipped_bounding_box != list(per_bounding_box):
        return per_clipped_bounding_box, BOUNDING_BOX_STATE_CLIPPED

    return per_clipped_bounding_box, BOUNDING_BOX_STATE_NORMAL


def get_normalized_bounding_box(per_clipped_bounding_box, per_reference_shape):
    """把clip后的bbox归一化到[0, 1]

    bbox参考尺寸与真实图尺寸不一致(见规格坑1)，只有长宽比一致，
    所以必须存归一化坐标，下游乘上真实图宽高即可，不需要再回头查参考尺寸。
    """
    per_reference_width, per_reference_height = per_reference_shape

    return [
        round(per_clipped_bounding_box[0] / per_reference_width,
              NORMALIZED_BOUNDING_BOX_ROUND_NDIGITS),
        round(per_clipped_bounding_box[1] / per_reference_height,
              NORMALIZED_BOUNDING_BOX_ROUND_NDIGITS),
        round(per_clipped_bounding_box[2] / per_reference_width,
              NORMALIZED_BOUNDING_BOX_ROUND_NDIGITS),
        round(per_clipped_bounding_box[3] / per_reference_height,
              NORMALIZED_BOUNDING_BOX_ROUND_NDIGITS),
    ]


def get_single_annotation_check_result(per_annotation, per_shard_index):
    """校验单条metadata的有用信息是否完整，并算出clip后bbox

    返回[无效原因列表, clip后bbox, bbox状态, 参考尺寸]，
    无效原因列表非空 == 这条不是包含完整有用信息的编辑样本对，直接丢弃并留痕。
    阶段1(建白名单)与阶段3(写标注)共用这个函数，保证两阶段判定100%一致。
    """
    invalid_reason_list = []

    if not isinstance(per_annotation, dict):
        return ['annotation not a dict'], None, None, None

    per_annotation_key_name_list = sorted(per_annotation.keys())
    if per_annotation_key_name_list != sorted(ANNOTATION_KEY_NAME_LIST):
        # 多字段少字段都说明上游规格变了，必须显式感知
        invalid_reason_list.append(
            f'annotation key not match {per_annotation_key_name_list}')

        return invalid_reason_list, None, None, None

    per_sample_id = per_annotation[ANNOTATION_SAMPLE_ID_KEY_NAME]
    per_source_id = per_annotation[ANNOTATION_SOURCE_ID_KEY_NAME]
    for per_id_key_name, per_id_value, per_id_max_value in [
        [
            ANNOTATION_SAMPLE_ID_KEY_NAME, per_sample_id,
            EXPECTED_TOTAL_SAMPLE_COUNT
        ],
        [
            ANNOTATION_SOURCE_ID_KEY_NAME, per_source_id,
            EXPECTED_TOTAL_SOURCE_IMAGE_COUNT
        ],
    ]:
        if not isinstance(per_id_value, int) or isinstance(per_id_value, bool):
            invalid_reason_list.append(f'{per_id_key_name} not a int')
            continue

        if not 0 <= per_id_value < per_id_max_value:
            invalid_reason_list.append(
                f'{per_id_key_name} out of range {per_id_value}')

    if len(invalid_reason_list) > 0:
        return invalid_reason_list, None, None, None

    # sample_id与所在分片必须自洽，不自洽说明metadata与asset tar已经错位，
    # 再按 sample_id//4000 去找编辑后图就会张冠李戴
    if per_sample_id // EXPECTED_SAMPLE_NUM_PER_SHARD != per_shard_index:
        invalid_reason_list.append(
            f'sample_id not in shard {per_sample_id} shard {per_shard_index}')

    per_edit_type = get_stripped_text_value(
        per_annotation[ANNOTATION_EDIT_TYPE_KEY_NAME])
    if per_edit_type not in ANNOTATION_EDIT_TYPE_NAME_LIST:
        invalid_reason_list.append(f'unknown edit_type {per_edit_type}')

    per_instruction = get_stripped_text_value(
        per_annotation[ANNOTATION_TEXT_KEY_NAME])
    if not per_instruction:
        # 编辑样本对必须有编辑指令，没有指令的图不可训练
        invalid_reason_list.append('empty instruction')

    if not isinstance(per_annotation[ANNOTATION_BETTER_DATA_KEY_NAME], bool):
        invalid_reason_list.append('better_data not a bool')

    # 三个文件名与两个压缩包名都能由 sample_id / source_id 唯一推出来，
    # 实测100%自洽; 一旦不自洽说明成员定位关系变了，绝不能按老规则去tar里取图
    for per_path_key_name, per_expected_path_value in [
        [
            ANNOTATION_SOURCE_ARCHIVE_KEY_NAME,
            f'{SOURCE_ARCHIVE_ROOT_DIR_NAME}/source-{per_source_id // EXPECTED_SOURCE_NUM_PER_SHARD:05d}-of-{EXPECTED_SOURCE_ARCHIVE_NUM:05d}.tar'
        ],
        [
            ANNOTATION_SOURCE_FILE_KEY_NAME,
            f'sources/source_{per_source_id:07d}.png'
        ],
        [
            ANNOTATION_ASSET_ARCHIVE_KEY_NAME,
            f'{ASSET_ARCHIVE_ROOT_DIR_NAME}/asset-{per_shard_index:05d}-of-{EXPECTED_ASSET_ARCHIVE_NUM:05d}.tar'
        ],
        [
            ANNOTATION_TARGET_FILE_KEY_NAME,
            f'targets/target_{per_sample_id:07d}.png'
        ],
        [ANNOTATION_MASK_FILE_KEY_NAME, f'masks/mask_{per_sample_id:07d}.png'],
    ]:
        if per_annotation[per_path_key_name] != per_expected_path_value:
            invalid_reason_list.append(
                f'{per_path_key_name} not match {per_annotation[per_path_key_name]} != {per_expected_path_value}'
            )

    per_reference_shape_dict = per_annotation[
        ANNOTATION_BBOX_REFERENCE_SHAPE_KEY_NAME]
    per_reference_shape = None
    if not isinstance(per_reference_shape_dict, dict) or sorted(
            per_reference_shape_dict.keys()) != ['height', 'width']:
        invalid_reason_list.append(
            'bbox_reference_dimensions not a valid dict')
    else:
        per_reference_width = per_reference_shape_dict['width']
        per_reference_height = per_reference_shape_dict['height']
        if not isinstance(per_reference_width, int) or isinstance(
                per_reference_width, bool) or per_reference_width <= 0:
            invalid_reason_list.append(
                'bbox_reference_dimensions width invalid')
        elif not isinstance(per_reference_height, int) or isinstance(
                per_reference_height, bool) or per_reference_height <= 0:
            invalid_reason_list.append(
                'bbox_reference_dimensions height invalid')
        else:
            per_reference_shape = [per_reference_width, per_reference_height]

    per_bounding_box = per_annotation[ANNOTATION_BOUNDING_BOX_KEY_NAME]
    if not isinstance(per_bounding_box,
                      list) or len(per_bounding_box) != 4 or not all(
                          isinstance(per_coordinate_value, int)
                          and not isinstance(per_coordinate_value, bool)
                          for per_coordinate_value in per_bounding_box):
        invalid_reason_list.append('bounding_box not 4 int')

        return invalid_reason_list, None, None, per_reference_shape

    if per_reference_shape is None:
        return invalid_reason_list, None, None, None

    per_clipped_bounding_box, per_bounding_box_state = get_clipped_bounding_box(
        per_bounding_box, per_reference_shape)
    if per_bounding_box_state == BOUNDING_BOX_STATE_DEGENERATE:
        # clip后仍退化的107条: 空间提示不可用，按用户确认口径直接丢弃
        invalid_reason_list.append(
            f'degenerate bounding_box after clip {per_bounding_box} in {per_reference_shape}'
        )

    return (invalid_reason_list, per_clipped_bounding_box,
            per_bounding_box_state, per_reference_shape)


def check_single_archive_tar_tail(per_archive_path):
    """O(1)预检单个tar是否被截断: 长度必须512字节对齐，且结尾必须有1024字节全0的EOF块

    抽检的4个tar全部满足。如果下载不全，流式解包只会在读到一半时抛异常，
    必须在跑2T解包前先拦住。
    """
    error_message_list = []
    try:
        per_archive_size = os.path.getsize(per_archive_path)
        if per_archive_size <= 0:
            error_message_list.append(f'empty archive {per_archive_path}')

            return error_message_list

        if per_archive_size % TAR_BLOCK_SIZE != 0:
            error_message_list.append(
                f'archive size not 512 aligned {per_archive_path} {per_archive_size}'
            )

        if per_archive_size < TAR_EOF_BLOCK_SIZE:
            error_message_list.append(
                f'archive size too small {per_archive_path} {per_archive_size}'
            )

            return error_message_list

        with open(per_archive_path, 'rb') as load_archive_file:
            load_archive_file.seek(per_archive_size - TAR_EOF_BLOCK_SIZE)
            per_archive_tail_bytes = load_archive_file.read(TAR_EOF_BLOCK_SIZE)

        if per_archive_tail_bytes != b'\x00' * TAR_EOF_BLOCK_SIZE:
            error_message_list.append(
                f'archive tar eof block broken(truncated archive) {per_archive_path}'
            )
    except Exception as e:
        error_message_list.append(
            f'read archive tail failed {per_archive_path} {e}')

    return error_message_list


def check_single_metadata_file_gzip_tail(per_metadata_path):
    """O(1)预检单个jsonl.gz是否被截断: gzip魔数 + 结尾4字节ISIZE必须能读到"""
    error_message_list = []
    try:
        per_metadata_size = os.path.getsize(per_metadata_path)
        if per_metadata_size <= 0:
            error_message_list.append(f'empty metadata {per_metadata_path}')

            return error_message_list

        with open(per_metadata_path, 'rb') as load_metadata_file:
            per_metadata_head_bytes = load_metadata_file.read(2)
            load_metadata_file.seek(per_metadata_size - 4)
            per_metadata_tail_bytes = load_metadata_file.read(4)

        if per_metadata_head_bytes != b'\x1f\x8b':
            error_message_list.append(
                f'metadata gzip magic broken {per_metadata_path}')

        if len(per_metadata_tail_bytes) != 4:
            error_message_list.append(
                f'metadata gzip tail broken {per_metadata_path}')
    except Exception as e:
        error_message_list.append(
            f'read metadata tail failed {per_metadata_path} {e}')

    return error_message_list


def get_all_file_and_shard_group(root_dataset_path):
    """扫描数据集，收集非分片文件列表与三类分片列表

    只走一层目录，不做全盘os.walk: 根目录下就是manifest.json + 3个子目录，
    每个子目录下就是平铺的分片文件，没有更深的层级。
    """
    file_copy_pair_list = []
    metadata_group_list = []
    asset_archive_group_list = []
    source_archive_group_list = []
    error_message_list = []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')

        return (file_copy_pair_list, metadata_group_list,
                asset_archive_group_list, source_archive_group_list,
                error_message_list)

    for per_name in sorted(os.listdir(root_dataset_path)):
        if check_skip_file_or_dir(per_name):
            continue

        per_path = os.path.join(root_dataset_path, per_name)
        if os.path.isdir(per_path):
            if per_name not in SUBSET_ROOT_DIR_NAME_LIST:
                # 根目录下出现新的子目录必须显式上报，否则会被静默漏处理
                error_message_list.append(f'unknown subset dir {per_name}')
            continue

        if per_name != MANIFEST_FILE_NAME:
            error_message_list.append(f'unknown file in root dir {per_name}')
            continue

        # manifest.json是官方规格清单，属于有用信息，原样拷进训练目录
        file_copy_pair_list.append([per_name, per_path])

    for per_subset_name, per_file_name_pattern, per_group_list in [
        [
            METADATA_ROOT_DIR_NAME, METADATA_FILE_NAME_PATTERN,
            metadata_group_list
        ],
        [
            ASSET_ARCHIVE_ROOT_DIR_NAME, ASSET_ARCHIVE_FILE_NAME_PATTERN,
            asset_archive_group_list
        ],
        [
            SOURCE_ARCHIVE_ROOT_DIR_NAME, SOURCE_ARCHIVE_FILE_NAME_PATTERN,
            source_archive_group_list
        ],
    ]:
        per_subset_path = os.path.join(root_dataset_path, per_subset_name)
        if not os.path.exists(per_subset_path):
            error_message_list.append(
                f'subset dir not exist {per_subset_name}')
            continue

        for per_file_name in sorted(os.listdir(per_subset_path)):
            if check_skip_file_or_dir(per_file_name):
                continue

            per_file_path = os.path.join(per_subset_path, per_file_name)
            if not os.path.isfile(per_file_path):
                error_message_list.append(
                    f'unknown dir in subset dir {per_subset_name}/{per_file_name}'
                )
                continue

            per_match_result = per_file_name_pattern.match(per_file_name)
            if not per_match_result:
                # 分片名对不上就没法定位它属于哪个分片，绝不能默默跳过
                error_message_list.append(
                    f'unknown file in subset dir {per_subset_name}/{per_file_name}'
                )
                continue

            per_group_list.append([
                per_match_result.group('prefix'),
                per_subset_name,
                per_file_path,
                int(per_match_result.group('index')),
                int(per_match_result.group('total')),
            ])

    metadata_group_list = sorted(metadata_group_list, key=lambda x: x[3])
    asset_archive_group_list = sorted(asset_archive_group_list,
                                      key=lambda x: x[3])
    source_archive_group_list = sorted(source_archive_group_list,
                                       key=lambda x: x[3])

    return (file_copy_pair_list, metadata_group_list, asset_archive_group_list,
            source_archive_group_list, error_message_list)


def check_manifest_file(root_dataset_path):
    """校验manifest.json与实测规格逐项一致，并核对它声明的12个字段名"""
    error_message_list = []

    per_manifest_path = os.path.join(root_dataset_path, MANIFEST_FILE_NAME)
    if not os.path.exists(per_manifest_path):
        error_message_list.append(
            f'manifest file not exist {MANIFEST_FILE_NAME}')

        return error_message_list

    try:
        with open(per_manifest_path, 'r', encoding='UTF-8') as load_json_file:
            per_manifest_dict = json.load(load_json_file)
    except Exception as e:
        error_message_list.append(f'load manifest failed {e}')

        return error_message_list

    if not isinstance(per_manifest_dict, dict):
        error_message_list.append('manifest not a dict')

        return error_message_list

    for per_key_name, per_expected_value in EXPECTED_MANIFEST_VALUE_DICT.items(
    ):
        per_manifest_value = per_manifest_dict.get(per_key_name, None)
        print('1111', 'manifest', per_key_name, per_manifest_value,
              'expected:', per_expected_value)

        if per_manifest_value != per_expected_value:
            error_message_list.append(
                f'manifest {per_key_name} not match {per_manifest_value} != {per_expected_value}'
            )

    per_manifest_field_name_list = per_manifest_dict.get('fields', None)
    if not isinstance(per_manifest_field_name_list, list) or sorted(
            per_manifest_field_name_list) != sorted(ANNOTATION_KEY_NAME_LIST):
        error_message_list.append(
            f'manifest fields not match {per_manifest_field_name_list}')

    return error_message_list


def check_required_shard_complete(metadata_group_list,
                                  asset_archive_group_list,
                                  source_archive_group_list):
    """解包前预检: 三类分片的数量、-of-总数、0..N-1连号、以及每个文件的尾部完整性

    分片少下几个时脚本照样能跑完并退出0，会静默少掉几万个样本对，必须先拦住。
    """
    error_message_list = []

    tail_check_pair_list = []
    for per_subset_name, per_group_list, per_expected_shard_num in [
        [
            METADATA_ROOT_DIR_NAME, metadata_group_list,
            EXPECTED_METADATA_SHARD_NUM
        ],
        [
            ASSET_ARCHIVE_ROOT_DIR_NAME, asset_archive_group_list,
            EXPECTED_ASSET_ARCHIVE_NUM
        ],
        [
            SOURCE_ARCHIVE_ROOT_DIR_NAME, source_archive_group_list,
            EXPECTED_SOURCE_ARCHIVE_NUM
        ],
    ]:
        print('1111', per_subset_name, 'shard:', len(per_group_list),
              'expected shard:', per_expected_shard_num)

        if len(per_group_list) != per_expected_shard_num:
            error_message_list.append(
                f'{per_subset_name} shard num not match {len(per_group_list)} != {per_expected_shard_num}'
            )

        per_shard_index_dict = {}
        for per_group_name, _, per_file_path, per_shard_index, per_shard_total in per_group_list:
            if per_shard_total != per_expected_shard_num:
                error_message_list.append(
                    f'{per_subset_name}/{per_group_name} shard total not match {per_shard_total} != {per_expected_shard_num}'
                )

            if per_shard_index in per_shard_index_dict:
                error_message_list.append(
                    f'{per_subset_name}/{per_group_name} duplicate shard index {per_shard_index}'
                )
                continue

            per_shard_index_dict[per_shard_index] = per_file_path
            tail_check_pair_list.append([per_subset_name, per_file_path])

        # 分片编号必须是0..N-1连号，缺号说明有分片没下载下来
        per_missing_shard_index_list = sorted(
            set(range(0, per_expected_shard_num)) -
            set(per_shard_index_dict.keys()))
        if len(per_missing_shard_index_list) > 0:
            error_message_list.append(
                f'{per_subset_name} shard index not continuous, missing index {per_missing_shard_index_list[:20]} total missing {len(per_missing_shard_index_list)}'
            )

    print('1111', 'check shard file tail:', len(tail_check_pair_list))
    archive_path_check_list = [
        per_file_path
        for per_subset_name, per_file_path in tail_check_pair_list
        if per_subset_name != METADATA_ROOT_DIR_NAME
    ]
    metadata_path_check_list = [
        per_file_path
        for per_subset_name, per_file_path in tail_check_pair_list
        if per_subset_name == METADATA_ROOT_DIR_NAME
    ]

    with Pool(processes=PROCESS_NUM) as pool:
        for per_tail_error_message_list in tqdm(
                pool.imap_unordered(check_single_archive_tar_tail,
                                    archive_path_check_list),
                total=len(archive_path_check_list)):
            error_message_list.extend(per_tail_error_message_list)

        for per_tail_error_message_list in tqdm(
                pool.imap_unordered(check_single_metadata_file_gzip_tail,
                                    metadata_path_check_list),
                total=len(metadata_path_check_list)):
            error_message_list.extend(per_tail_error_message_list)

    return error_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非分片文件(manifest.json)原样拷贝到目标目录，保持相对路径不变"""
    per_file_relative_path, per_file_path = file_copy_pair

    save_file_path = os.path.join(save_dataset_path, per_file_relative_path)
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if os.path.exists(save_file_path) and os.path.getsize(
            save_file_path) == os.path.getsize(per_file_path):
        return [per_file_relative_path, '']

    try:
        with open(per_file_path, 'rb') as load_file:
            with open(save_file_path, 'wb') as save_file:
                shutil.copyfileobj(load_file, save_file, COPY_FILE_BLOCK_SIZE)
    except Exception as e:
        print('4444', per_file_path, e)

        return [per_file_relative_path, f'copy file failed {e}']

    if not os.path.exists(save_file_path) or os.path.getsize(
            save_file_path) != os.path.getsize(per_file_path):
        print('4444', per_file_path, 'copy file size not match')

        return [per_file_relative_path, 'copy file size not match']

    return [per_file_relative_path, '']


def save_single_member_bytes(save_member_path, per_member_bytes,
                             per_member_size):
    """单个成员独立落盘并立刻校验落盘大小，返回[是否新写, 是否跳过, 错误信息]

    单个成员写盘异常不能让整个tar的循环中断，否则该tar后面几千个成员全都不处理。
    先判存在且大小一致就跳过，保证脚本可以断点续跑。
    """
    if os.path.exists(save_member_path) and os.path.getsize(
            save_member_path) == per_member_size:
        return [False, True, '']

    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_member_path, 'wb') as save_member_file:
            save_member_file.write(per_member_bytes)
    except Exception as e:
        print('6666', save_member_path, e)

        return [False, False, f'write member failed {save_member_path} {e}']

    if not os.path.exists(save_member_path) or os.path.getsize(
            save_member_path) != per_member_size:
        print('6666', save_member_path, 'save member size not match')

        return [False, False, f'save member size not match {save_member_path}']

    return [True, False, '']


def process_single_metadata_file(metadata_group):
    """流式解析单个jsonl.gz，产出该片的有效样本对清单与该落盘成员白名单

    这一步只读48M的gz、完全不碰2T的tar，所以可以先跑完拿到全局白名单，
    再让解包阶段"只解白名单里的成员"，从而不把要丢弃的107个样本对
    和30张孤儿源图写到NAS上。
    """
    per_metadata_group_name, per_metadata_relative_dir, per_metadata_path, per_shard_index, _ = metadata_group

    per_metadata_relative_path = f'{per_metadata_relative_dir}/{per_metadata_group_name}'

    row_count = 0
    valid_sample_pair_list, invalid_sample_list = [], []
    sample_id_list, source_id_list = [], []
    edit_type_count_dict = collections.Counter()
    better_data_count_dict = collections.Counter()
    bbox_reference_shape_count_dict = collections.Counter()
    bounding_box_state_count_dict = collections.Counter()
    error_message_list = []
    reach_metadata_end = False

    sample_id_dict = {}

    try:
        with gzip.open(per_metadata_path, 'rt',
                       encoding='UTF-8') as load_metadata_file:
            for per_line in load_metadata_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                row_count += 1

                try:
                    per_annotation = json.loads(per_line)
                except Exception as e:
                    error_message_list.append(
                        f'load annotation failed row {row_count - 1} {e}')
                    invalid_sample_list.append({
                        'sample_id':
                        None,
                        'source_id':
                        None,
                        'metadata_name':
                        per_metadata_group_name,
                        'reason':
                        f'load annotation failed {e}',
                    })
                    continue

                (per_invalid_reason_list, _, per_bounding_box_state,
                 per_reference_shape) = get_single_annotation_check_result(
                     per_annotation, per_shard_index)

                per_sample_id = per_annotation.get(
                    ANNOTATION_SAMPLE_ID_KEY_NAME, None) if isinstance(
                        per_annotation, dict) else None
                per_source_id = per_annotation.get(
                    ANNOTATION_SOURCE_ID_KEY_NAME, None) if isinstance(
                        per_annotation, dict) else None

                # 分布统计按全量行统计(含被丢弃的行)，才能与实测ground truth对上
                if isinstance(per_annotation, dict):
                    edit_type_count_dict[str(
                        per_annotation.get(ANNOTATION_EDIT_TYPE_KEY_NAME,
                                           None))] += 1
                    better_data_count_dict[str(
                        per_annotation.get(ANNOTATION_BETTER_DATA_KEY_NAME,
                                           None))] += 1
                if per_reference_shape is not None:
                    bbox_reference_shape_count_dict[
                        f'{per_reference_shape[0]}x{per_reference_shape[1]}'] += 1
                if per_bounding_box_state is not None:
                    bounding_box_state_count_dict[per_bounding_box_state] += 1

                if isinstance(per_sample_id,
                              int) and not isinstance(per_sample_id, bool):
                    sample_id_list.append(per_sample_id)
                    if per_sample_id in sample_id_dict:
                        # 片内sample_id重名会导致同一个target/mask被两条标注抢，必须上报
                        error_message_list.append(
                            f'duplicate sample_id {per_sample_id}')
                        per_invalid_reason_list = per_invalid_reason_list + [
                            'duplicate sample_id'
                        ]
                    else:
                        sample_id_dict[per_sample_id] = row_count - 1

                if isinstance(per_source_id,
                              int) and not isinstance(per_source_id, bool):
                    source_id_list.append(per_source_id)

                if len(per_invalid_reason_list) > 0:
                    invalid_sample_list.append({
                        'sample_id':
                        per_sample_id,
                        'source_id':
                        per_source_id,
                        'metadata_name':
                        per_metadata_group_name,
                        'reason':
                        ','.join(per_invalid_reason_list),
                    })
                    continue

                valid_sample_pair_list.append([per_sample_id, per_source_id])

        reach_metadata_end = True
    except Exception as e:
        # gz截断或NAS读失败时必须上报，不能静默少样本对
        print('7777', per_metadata_relative_path, e)
        error_message_list.append(f'read metadata failed {e}')

    if not reach_metadata_end:
        error_message_list.append(
            'not reach metadata stream end, metadata may be truncated')

    per_expected_row_count = get_expected_shard_item_count(
        per_shard_index, EXPECTED_SAMPLE_NUM_PER_SHARD,
        EXPECTED_TOTAL_SAMPLE_COUNT)
    if row_count != per_expected_row_count:
        error_message_list.append(
            f'row count not match {row_count} != {per_expected_row_count}')

    if len(valid_sample_pair_list) + len(invalid_sample_list) != row_count:
        error_message_list.append(
            f'sample count not match {len(valid_sample_pair_list)} + {len(invalid_sample_list)} != {row_count}'
        )

    return {
        'metadata_relative_path': per_metadata_relative_path,
        'metadata_name': per_metadata_group_name,
        'shard_index': per_shard_index,
        'row_count': row_count,
        'expected_row_count': per_expected_row_count,
        'sample_id_list': sample_id_list,
        'source_id_list': source_id_list,
        'valid_sample_pair_list': valid_sample_pair_list,
        'invalid_sample_list': invalid_sample_list,
        'edit_type_count_dict': dict(edit_type_count_dict),
        'better_data_count_dict': dict(better_data_count_dict),
        'bbox_reference_shape_count_dict':
        dict(bbox_reference_shape_count_dict),
        'bounding_box_state_count_dict': dict(bounding_box_state_count_dict),
        'error_message_list': error_message_list,
    }


def get_single_member_kind_and_index(per_member_name, per_archive_kind):
    """按成员名解出成员种类与index，解不出返回[None, None]

    asset tar只应该有 targets/ 与 masks/ 两类成员，source tar只应该有 sources/ 一类，
    出现其他名字说明tar内部规格变了，必须显式上报，不能默认当图像处理。
    """
    if per_archive_kind == ARCHIVE_KIND_ASSET:
        per_match_result = TARGET_MEMBER_NAME_PATTERN.match(per_member_name)
        if per_match_result:
            return MEMBER_KIND_TARGET, int(per_match_result.group('index'))

        per_match_result = MASK_MEMBER_NAME_PATTERN.match(per_member_name)
        if per_match_result:
            return MEMBER_KIND_MASK, int(per_match_result.group('index'))

        return None, None

    per_match_result = SOURCE_MEMBER_NAME_PATTERN.match(per_member_name)
    if per_match_result:
        return MEMBER_KIND_SOURCE, int(per_match_result.group('index'))

    return None, None


def process_single_archive_group(archive_group, save_dataset_path):
    """流式解开单个tar，只落盘白名单内的成员，并顺手解PNG头拿真实宽高

    落盘结构:
      unzip_images/sources/<source分片名>/source_xxxxxxx.png
      unzip_images/targets/<asset分片名>/target_xxxxxxx.png
      unzip_images/masks/<asset分片名>/mask_xxxxxxx.png

    tar按 mode='r|' 顺序流式读，绝不整片进内存(单片最大约4GB);
    单个成员的字节只在处理它的那一瞬间进内存(PNG单张最大几MB)。
    """
    (per_archive_group_name, per_archive_relative_dir, per_archive_path,
     per_archive_kind, per_declare_member_index_list,
     per_valid_member_index_list) = archive_group

    per_archive_relative_path = f'{per_archive_relative_dir}/{per_archive_group_name}'

    per_declare_member_index_dict = {
        per_member_index: 1
        for per_member_index in per_declare_member_index_list
    }
    per_valid_member_index_dict = {
        per_member_index: 1
        for per_member_index in per_valid_member_index_list
    }

    if per_archive_kind == ARCHIVE_KIND_ASSET:
        per_member_kind_name_list = [MEMBER_KIND_TARGET, MEMBER_KIND_MASK]
        per_save_dir_name_dict = {
            MEMBER_KIND_TARGET: SAVE_TARGET_IMAGE_DIR_NAME,
            MEMBER_KIND_MASK: SAVE_MASK_IMAGE_DIR_NAME,
        }
    else:
        per_member_kind_name_list = [MEMBER_KIND_SOURCE]
        per_save_dir_name_dict = {
            MEMBER_KIND_SOURCE: SAVE_SOURCE_IMAGE_DIR_NAME,
        }

    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_MEMBER_DIR_NAME,
                                           per_archive_relative_dir,
                                           per_archive_group_name)

    total_file_member_count = 0
    extract_file_count, skip_file_count = 0, 0
    not_save_file_count, drop_file_count, save_fail_count = 0, 0, 0
    duplicate_member_count = 0
    member_kind_count_dict = collections.Counter()
    image_shape_count_dict = collections.Counter()
    unknown_member_name_list = []
    error_message_list = []
    reach_tar_end = False

    member_index_dict = {
        per_member_kind_name: {}
        for per_member_kind_name in per_member_kind_name_list
    }
    image_shape_dict = {
        per_member_kind_name: {}
        for per_member_kind_name in per_member_kind_name_list
    }

    try:
        with tarfile.open(per_archive_path, mode='r|') as load_tar_file:
            for per_member in load_tar_file:
                per_member_name = per_member.name.replace('\\',
                                                          '/').lstrip('/')
                per_member_name = os.path.normpath(per_member_name).replace(
                    '\\', '/')
                if per_member_name.startswith('..'):
                    print('5555', per_archive_group_name, per_member.name)
                    error_message_list.append(
                        f'illegal member name {per_member.name}')
                    continue

                if per_member.isdir():
                    continue

                if not per_member.isfile():
                    # 实测只有普通文件，出现链接等类型必须显式上报
                    print('5555', per_archive_group_name, per_member.name,
                          'not a regular file')
                    error_message_list.append(
                        f'not a regular file {per_member.name}')
                    continue

                total_file_member_count += 1

                per_member_kind, per_member_index = get_single_member_kind_and_index(
                    per_member_name, per_archive_kind)
                if per_member_kind is None:
                    # 名字对不上就没法知道它属于哪个样本对，绝不能默默跳过
                    unknown_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'unknown member name {per_member_name}')
                    continue

                member_kind_count_dict[per_member_kind] += 1

                if per_member_index not in per_declare_member_index_dict:
                    # tar里有metadata没声明的成员，说明两边规格错位，必须上报
                    unknown_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'member not declared in metadata {per_member_name}')
                    continue

                per_member_is_duplicate = per_member_index in member_index_dict[
                    per_member_kind]
                if per_member_is_duplicate:
                    # 同一个tar里出现重名成员时按名写盘会互相覆盖，
                    # 这里改写到独立目录保留数据并上报，不能静默丢样本
                    duplicate_member_count += 1
                    error_message_list.append(
                        f'duplicate member name {per_member_name}')
                else:
                    member_index_dict[per_member_kind][per_member_index] = 1

                if per_member_index not in per_valid_member_index_dict:
                    # 白名单外 = clip后bbox仍退化被丢弃的样本对，或只被这些样本对
                    # 引用的孤儿源图，按用户确认口径不落盘，只计数留痕
                    drop_file_count += 1
                    continue

                save_member_name = os.path.basename(per_member_name)
                if per_member_is_duplicate:
                    save_member_path = os.path.join(
                        save_duplicate_dir_path, f'{duplicate_member_count}',
                        per_member_name)
                else:
                    save_member_path = os.path.join(
                        save_dataset_path, SAVE_IMAGE_DIR_NAME,
                        per_save_dir_name_dict[per_member_kind],
                        per_archive_group_name, save_member_name)

                if not EXTRACT_IMAGE_FILE_FLAG and not PARSE_IMAGE_SHAPE_FLAG:
                    # 只建索引且不需要真实宽高时，连成员字节都不用读
                    not_save_file_count += 1
                    continue

                if EXTRACT_IMAGE_FILE_FLAG and not PARSE_IMAGE_SHAPE_FLAG and os.path.exists(
                        save_member_path) and os.path.getsize(
                            save_member_path) == per_member.size:
                    # 断点续跑时已经完整落盘的成员直接跳过，不用再读一遍字节
                    skip_file_count += 1
                    continue

                load_member_file = load_tar_file.extractfile(per_member)
                if load_member_file is None:
                    print('6666', per_archive_group_name, per_member.name)
                    error_message_list.append(
                        f'extract member failed {per_member.name}')
                    save_fail_count += 1
                    continue

                per_member_bytes = load_member_file.read()
                if len(per_member_bytes) != per_member.size:
                    error_message_list.append(
                        f'member data truncated {per_member_name} {len(per_member_bytes)} != {per_member.size}'
                    )
                    save_fail_count += 1
                    continue

                if PARSE_IMAGE_SHAPE_FLAG and not per_member_is_duplicate:
                    per_image_shape = get_png_image_shape(per_member_bytes)
                    if per_image_shape is None:
                        error_message_list.append(
                            f'parse png header failed {per_member_name}')
                    else:
                        image_shape_dict[per_member_kind][
                            per_member_index] = per_image_shape
                        image_shape_count_dict[
                            f'{per_member_kind}_{per_image_shape[0]}x{per_image_shape[1]}'] += 1

                if not EXTRACT_IMAGE_FILE_FLAG:
                    not_save_file_count += 1
                    continue

                per_save_new, per_save_skip, per_save_error_message = save_single_member_bytes(
                    save_member_path, per_member_bytes, per_member.size)
                if per_save_error_message:
                    error_message_list.append(per_save_error_message)
                    save_fail_count += 1
                    continue

                if per_save_new:
                    extract_file_count += 1
                if per_save_skip:
                    skip_file_count += 1

        reach_tar_end = True
    except Exception as e:
        # tar截断或NAS读失败时保留已解出的成员，但必须上报，不能静默少样本对
        print('7777', per_archive_relative_path, e)
        error_message_list.append(f'read archive failed {e}')

    if not reach_tar_end:
        error_message_list.append(
            'not reach tar stream end, archive may be truncated')

    if extract_file_count + skip_file_count + not_save_file_count + drop_file_count + save_fail_count != total_file_member_count:
        error_message_list.append(
            f'process file count not match: {extract_file_count} + {skip_file_count} + {not_save_file_count} + {drop_file_count} + {save_fail_count} != {total_file_member_count}'
        )

    # 核心对账: tar里每一类成员的index集合必须与metadata声明的集合**完全相等**。
    # 只比对数量是不够的(多一个少一个刚好抵消就看不出来)，必须比集合;
    # 集合相等才能保证"每个包含完整有用信息的样本对都被处理过"。
    per_declare_member_index_set = set(per_declare_member_index_list)
    for per_member_kind_name in per_member_kind_name_list:
        per_member_index_set = set(
            member_index_dict[per_member_kind_name].keys())

        per_missing_member_index_list = sorted(per_declare_member_index_set -
                                               per_member_index_set)
        if len(per_missing_member_index_list) > 0:
            error_message_list.append(
                f'{per_member_kind_name} member missing in tar, missing index {per_missing_member_index_list[:20]} total missing {len(per_missing_member_index_list)}'
            )

        per_extra_member_index_list = sorted(per_member_index_set -
                                             per_declare_member_index_set)
        if len(per_extra_member_index_list) > 0:
            error_message_list.append(
                f'{per_member_kind_name} member not declared in metadata, extra index {per_extra_member_index_list[:20]} total extra {len(per_extra_member_index_list)}'
            )

    if per_archive_kind == ARCHIVE_KIND_ASSET and member_kind_count_dict[
            MEMBER_KIND_TARGET] != member_kind_count_dict[MEMBER_KIND_MASK]:
        error_message_list.append(
            f'target member count {member_kind_count_dict[MEMBER_KIND_TARGET]} != mask member count {member_kind_count_dict[MEMBER_KIND_MASK]}'
        )

    per_expected_file_member_count = len(per_declare_member_index_list) * len(
        per_member_kind_name_list)
    if total_file_member_count != per_expected_file_member_count:
        error_message_list.append(
            f'tar file member count not match {total_file_member_count} != {per_expected_file_member_count}'
        )

    return {
        'archive_relative_path': per_archive_relative_path,
        'archive_name': per_archive_group_name,
        'archive_kind': per_archive_kind,
        'total_file_member_count': total_file_member_count,
        'extract_file_count': extract_file_count,
        'skip_file_count': skip_file_count,
        'not_save_file_count': not_save_file_count,
        'drop_file_count': drop_file_count,
        'save_fail_count': save_fail_count,
        'duplicate_member_count': duplicate_member_count,
        'member_kind_count_dict': dict(member_kind_count_dict),
        'image_shape_count_dict': dict(image_shape_count_dict),
        'target_image_shape_dict':
        image_shape_dict.get(MEMBER_KIND_TARGET, {}),
        'mask_image_shape_dict': image_shape_dict.get(MEMBER_KIND_MASK, {}),
        'source_image_shape_dict':
        image_shape_dict.get(MEMBER_KIND_SOURCE, {}),
        'unknown_member_name_list': unknown_member_name_list,
        'error_message_list': error_message_list,
    }


def get_single_mask_image_shape_state(per_mask_image_shape,
                                      per_target_image_shape,
                                      per_source_image_shape):
    """判定mask真实尺寸跟哪张图一致

    实测两种情况都有(mask有时等于target、有时等于source，而两者本身也不相等)，
    这是上游发布规格，只统计不判错，但必须让下游知道，否则直接把mask当target
    的alpha用就会尺寸对不上。
    """
    if not per_mask_image_shape:
        return MASK_IMAGE_SHAPE_STATE_OTHER

    if per_target_image_shape and per_mask_image_shape == per_target_image_shape:
        return MASK_IMAGE_SHAPE_STATE_EQUAL_TARGET

    if per_source_image_shape and per_mask_image_shape == per_source_image_shape:
        return MASK_IMAGE_SHAPE_STATE_EQUAL_SOURCE

    return MASK_IMAGE_SHAPE_STATE_OTHER


def get_single_sample_pair_annotation(
        per_annotation, per_metadata_group_name, per_clipped_bounding_box,
        per_bounding_box_state, per_reference_shape,
        per_source_image_relative_path, per_target_image_relative_path,
        per_mask_image_relative_path, per_source_image_shape,
        per_target_image_shape, per_mask_image_shape):
    """拼一条完整样本对标注: 保留原metadata全部12个字段 + 落盘路径/真实宽高/归一化bbox

    下游可以直接按行取样本，不需要为了拿指令和bbox去扫330万个小文件，
    也不需要为了拿真实宽高去解图。
    """
    per_sample_id = per_annotation[ANNOTATION_SAMPLE_ID_KEY_NAME]

    per_save_annotation = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'sample_key':
        f'{per_sample_id:07d}',
        'metadata_name':
        per_metadata_group_name,
        'instruction':
        get_stripped_text_value(per_annotation[ANNOTATION_TEXT_KEY_NAME]),
        'reference_image_path_list': [per_source_image_relative_path],
        'reference_image_num':
        1,
        'edited_image_path':
        per_target_image_relative_path,
        'mask_image_path':
        per_mask_image_relative_path,
        'source_image_shape':
        per_source_image_shape,
        'edited_image_shape':
        per_target_image_shape,
        'mask_image_shape':
        per_mask_image_shape,
        'mask_image_shape_state':
        get_single_mask_image_shape_state(per_mask_image_shape,
                                          per_target_image_shape,
                                          per_source_image_shape),
        'clipped_bounding_box':
        per_clipped_bounding_box,
        'normalized_bounding_box':
        get_normalized_bounding_box(per_clipped_bounding_box,
                                    per_reference_shape),
        'bounding_box_state':
        per_bounding_box_state,
    }

    # 原metadata的12个字段全部原样保留(含bounding_box原值与bbox_reference_dimensions),
    # 上面加的都是新key，不会覆盖
    for per_key_name, per_key_value in per_annotation.items():
        if per_key_name in per_save_annotation:
            continue
        per_save_annotation[per_key_name] = per_key_value

    return per_save_annotation


def process_single_annotation_file(annotation_group, save_dataset_path,
                                   save_annotation_dir_path):
    """重新流式读一遍metadata，与解包阶段拿到的真实宽高join后写该片的jsonl汇总标注

    metadata总共只有48M，再读一遍的代价可忽略，
    比在阶段1把110万条标注塞进进程间管道便宜得多。
    这里的有效性判定与阶段1共用 get_single_annotation_check_result，
    所以两阶段算出来的有效条数必须完全相等(主流程会硬对账)。
    """
    (per_metadata_group_name, per_metadata_relative_dir, per_metadata_path,
     per_shard_index, per_target_image_shape_dict, per_mask_image_shape_dict,
     per_source_image_shape_dict) = annotation_group

    per_metadata_relative_path = f'{per_metadata_relative_dir}/{per_metadata_group_name}'

    save_annotation_path = os.path.join(save_annotation_dir_path,
                                        f'{per_metadata_group_name}.jsonl')

    row_count, valid_sample_pair_count, invalid_sample_count = 0, 0, 0
    mask_image_shape_state_count_dict = collections.Counter()
    bounding_box_state_count_dict = collections.Counter()
    edit_type_count_dict = collections.Counter()
    missing_image_shape_count = 0
    warning_message_list, error_message_list = [], []

    try:
        os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
        with open(save_annotation_path, 'w',
                  encoding='UTF-8') as save_annotation_file:
            with gzip.open(per_metadata_path, 'rt',
                           encoding='UTF-8') as load_metadata_file:
                for per_line in load_metadata_file:
                    per_line = per_line.strip()
                    if not per_line:
                        continue

                    row_count += 1

                    per_annotation = json.loads(per_line)

                    (per_invalid_reason_list, per_clipped_bounding_box,
                     per_bounding_box_state,
                     per_reference_shape) = get_single_annotation_check_result(
                         per_annotation, per_shard_index)
                    if len(per_invalid_reason_list) > 0:
                        invalid_sample_count += 1
                        continue

                    per_sample_id = per_annotation[
                        ANNOTATION_SAMPLE_ID_KEY_NAME]
                    per_source_id = per_annotation[
                        ANNOTATION_SOURCE_ID_KEY_NAME]

                    per_source_image_shape = per_source_image_shape_dict.get(
                        per_source_id, None)
                    per_target_image_shape = per_target_image_shape_dict.get(
                        per_sample_id, None)
                    per_mask_image_shape = per_mask_image_shape_dict.get(
                        per_sample_id, None)

                    if PARSE_IMAGE_SHAPE_FLAG:
                        for per_image_name, per_image_shape in [
                            ['source', per_source_image_shape],
                            ['target', per_target_image_shape],
                            ['mask', per_mask_image_shape],
                        ]:
                            if per_image_shape:
                                continue

                            # 阶段2没处理过这张图 == 这个样本对没被完整解出来，必须上报
                            missing_image_shape_count += 1
                            error_message_list.append(
                                f'sample {per_sample_id} miss {per_image_name} image shape'
                            )

                        # bbox参考尺寸与真实图尺寸只有长宽比一致(规格坑1)，
                        # 连长宽比都对不上就说明归一化bbox不可信，必须上报
                        for per_image_name, per_image_shape in [
                            ['source', per_source_image_shape],
                            ['target', per_target_image_shape],
                        ]:
                            per_aspect_ratio_diff = get_image_aspect_ratio_diff(
                                per_image_shape, per_reference_shape)
                            if per_aspect_ratio_diff is not None and per_aspect_ratio_diff > IMAGE_ASPECT_RATIO_DIFF_THRESHOLD:
                                warning_message_list.append(
                                    f'sample {per_sample_id} {per_image_name} aspect ratio not match bbox reference {per_image_shape} {per_reference_shape}'
                                )

                    per_source_archive_name = os.path.splitext(
                        os.path.basename(
                            per_annotation[ANNOTATION_SOURCE_ARCHIVE_KEY_NAME])
                    )[0]
                    per_asset_archive_name = os.path.splitext(
                        os.path.basename(
                            per_annotation[ANNOTATION_ASSET_ARCHIVE_KEY_NAME])
                    )[0]

                    per_source_image_relative_path = f'{SAVE_IMAGE_DIR_NAME}/{SAVE_SOURCE_IMAGE_DIR_NAME}/{per_source_archive_name}/source_{per_source_id:07d}.png'
                    per_target_image_relative_path = f'{SAVE_IMAGE_DIR_NAME}/{SAVE_TARGET_IMAGE_DIR_NAME}/{per_asset_archive_name}/target_{per_sample_id:07d}.png'
                    per_mask_image_relative_path = f'{SAVE_IMAGE_DIR_NAME}/{SAVE_MASK_IMAGE_DIR_NAME}/{per_asset_archive_name}/mask_{per_sample_id:07d}.png'

                    per_save_annotation = get_single_sample_pair_annotation(
                        per_annotation, per_metadata_group_name,
                        per_clipped_bounding_box, per_bounding_box_state,
                        per_reference_shape, per_source_image_relative_path,
                        per_target_image_relative_path,
                        per_mask_image_relative_path, per_source_image_shape,
                        per_target_image_shape, per_mask_image_shape)

                    save_annotation_file.write(
                        f'{json.dumps(per_save_annotation, ensure_ascii=False)}\n'
                    )

                    valid_sample_pair_count += 1
                    mask_image_shape_state_count_dict[
                        per_save_annotation['mask_image_shape_state']] += 1
                    bounding_box_state_count_dict[per_bounding_box_state] += 1
                    edit_type_count_dict[
                        per_annotation[ANNOTATION_EDIT_TYPE_KEY_NAME]] += 1
    except Exception as e:
        print('9999', per_metadata_relative_path, e)
        error_message_list.append(f'save annotation failed {e}')

    if valid_sample_pair_count + invalid_sample_count != row_count:
        error_message_list.append(
            f'annotation count not match {valid_sample_pair_count} + {invalid_sample_count} != {row_count}'
        )

    return {
        'metadata_relative_path':
        per_metadata_relative_path,
        'metadata_name':
        per_metadata_group_name,
        'shard_index':
        per_shard_index,
        'row_count':
        row_count,
        'valid_sample_pair_count':
        valid_sample_pair_count,
        'invalid_sample_count':
        invalid_sample_count,
        'missing_image_shape_count':
        missing_image_shape_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_metadata_group_name}.jsonl',
        'mask_image_shape_state_count_dict':
        dict(mask_image_shape_state_count_dict),
        'bounding_box_state_count_dict':
        dict(bounding_box_state_count_dict),
        'edit_type_count_dict':
        dict(edit_type_count_dict),
        'warning_message_list':
        warning_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }


def check_single_archive_dir_on_disk(archive_check_pair):
    """可选的二次对账: os.walk单个tar的输出目录，核对落盘文件数与文件名集合"""
    (per_archive_relative_path, per_archive_dir_path_list,
     per_expected_file_name_list) = archive_check_pair

    error_message_list = []
    on_disk_file_name_dict = {}
    unknown_suffix_file_count = 0
    for per_archive_dir_path in per_archive_dir_path_list:
        if not os.path.exists(per_archive_dir_path):
            error_message_list.append(
                f'{per_archive_relative_path} archive dir not exist {per_archive_dir_path}'
            )
            continue

        for per_root_path, _, per_file_name_list in os.walk(
                per_archive_dir_path):
            for per_file_name in per_file_name_list:
                if not check_image_file_suffix(per_file_name):
                    unknown_suffix_file_count += 1
                    continue

                on_disk_file_name_dict[per_file_name] = 1

    per_expected_file_name_set = set(per_expected_file_name_list)
    per_missing_file_name_list = sorted(per_expected_file_name_set -
                                        set(on_disk_file_name_dict.keys()))
    per_orphan_file_name_list = sorted(
        set(on_disk_file_name_dict.keys()) - per_expected_file_name_set)

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_archive_relative_path} unknown suffix file num {unknown_suffix_file_count}'
        )
    if len(per_missing_file_name_list) > 0:
        error_message_list.append(
            f'{per_archive_relative_path} missing image num {len(per_missing_file_name_list)} {per_missing_file_name_list[:10]}'
        )
    if len(per_orphan_file_name_list) > 0:
        error_message_list.append(
            f'{per_archive_relative_path} orphan image num {len(per_orphan_file_name_list)} {per_orphan_file_name_list[:10]}'
        )

    return [
        per_archive_relative_path,
        len(on_disk_file_name_dict),
        len(per_missing_file_name_list),
        len(per_orphan_file_name_list),
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, archive_group_list):
    """可选的二次对账: 遍历输出目录核对每个tar目录里的落盘文件名集合"""
    archive_check_pair_list = []
    for (per_archive_group_name, per_archive_relative_dir, _, per_archive_kind,
         _, per_valid_member_index_list) in archive_group_list:
        if per_archive_kind == ARCHIVE_KIND_ASSET:
            per_archive_dir_path_list = [
                os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                             SAVE_TARGET_IMAGE_DIR_NAME,
                             per_archive_group_name),
                os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                             SAVE_MASK_IMAGE_DIR_NAME, per_archive_group_name),
            ]
            per_expected_file_name_list = [
                f'target_{per_member_index:07d}.png'
                for per_member_index in per_valid_member_index_list
            ] + [
                f'mask_{per_member_index:07d}.png'
                for per_member_index in per_valid_member_index_list
            ]
        else:
            per_archive_dir_path_list = [
                os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                             SAVE_SOURCE_IMAGE_DIR_NAME,
                             per_archive_group_name),
            ]
            per_expected_file_name_list = [
                f'source_{per_member_index:07d}.png'
                for per_member_index in per_valid_member_index_list
            ]

        archive_check_pair_list.append([
            f'{per_archive_relative_dir}/{per_archive_group_name}',
            per_archive_dir_path_list,
            per_expected_file_name_list,
        ])

    error_message_list = []
    total_on_disk_image_count = 0
    total_missing_image_count, total_orphan_image_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_archive_dir_on_disk, archive_check_pair_list),
                                     total=len(archive_check_pair_list)):
            (_, per_on_disk_image_count, per_missing_image_count,
             per_orphan_image_count, per_error_message_list) = per_check_result

            total_on_disk_image_count += per_on_disk_image_count
            total_missing_image_count += per_missing_image_count
            total_orphan_image_count += per_orphan_image_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk image:', total_on_disk_image_count,
          'missing image:', total_missing_image_count, 'orphan image:',
          total_orphan_image_count)

    return error_message_list


def check_metadata_result(metadata_result_list):
    """汇总275片metadata的解析结果，产出全局白名单并做全量硬对账

    返回[asset白名单dict, source白名单dict, 全局统计dict, 错误信息列表]。
    """
    total_row_count = 0
    all_sample_id_dict, all_source_id_dict = {}, {}
    valid_sample_id_dict, valid_source_id_dict = {}, {}
    all_invalid_sample_list = []
    shard_valid_sample_pair_count_dict = {}
    shard_sample_id_dict = {}
    edit_type_count_dict = collections.Counter()
    better_data_count_dict = collections.Counter()
    bbox_reference_shape_count_dict = collections.Counter()
    bounding_box_state_count_dict = collections.Counter()
    error_message_list = []

    for per_metadata_result in metadata_result_list:
        per_metadata_relative_path = per_metadata_result[
            'metadata_relative_path']
        per_shard_index = per_metadata_result['shard_index']

        total_row_count += per_metadata_result['row_count']
        all_invalid_sample_list.extend(
            per_metadata_result['invalid_sample_list'])
        shard_valid_sample_pair_count_dict[per_shard_index] = len(
            per_metadata_result['valid_sample_pair_list'])

        edit_type_count_dict.update(
            per_metadata_result['edit_type_count_dict'])
        better_data_count_dict.update(
            per_metadata_result['better_data_count_dict'])
        bbox_reference_shape_count_dict.update(
            per_metadata_result['bbox_reference_shape_count_dict'])
        bounding_box_state_count_dict.update(
            per_metadata_result['bounding_box_state_count_dict'])

        per_shard_sample_id_list = []
        for per_sample_id in per_metadata_result['sample_id_list']:
            if per_sample_id in all_sample_id_dict:
                # 跨片sample_id重名会让两条标注抢同一张target，必须上报
                error_message_list.append(
                    f'{per_metadata_relative_path} duplicate sample_id across shard {per_sample_id}'
                )
                continue
            all_sample_id_dict[per_sample_id] = per_shard_index
            per_shard_sample_id_list.append(per_sample_id)

        shard_sample_id_dict[per_shard_index] = per_shard_sample_id_list

        for per_source_id in per_metadata_result['source_id_list']:
            all_source_id_dict[per_source_id] = 1

        for per_sample_id, per_source_id in per_metadata_result[
                'valid_sample_pair_list']:
            valid_sample_id_dict[per_sample_id] = per_shard_index
            valid_source_id_dict[per_source_id] = 1

        if len(per_metadata_result['error_message_list']) > 0:
            print('7777', per_metadata_relative_path,
                  per_metadata_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_metadata_relative_path} error num {len(per_metadata_result["error_message_list"])} {per_metadata_result["error_message_list"][:3]}'
            )

    total_valid_sample_count = len(valid_sample_id_dict)
    total_valid_source_image_count = len(valid_source_id_dict)
    total_orphan_source_image_count = len(all_source_id_dict) - len(
        valid_source_id_dict)

    print('3333', 'total row:', total_row_count, 'total sample id:',
          len(all_sample_id_dict), 'total source id:', len(all_source_id_dict),
          'valid sample:', total_valid_sample_count, 'valid source image:',
          total_valid_source_image_count, 'invalid sample:',
          len(all_invalid_sample_list), 'orphan source image:',
          total_orphan_source_image_count)
    print('3333', 'edit type:', dict(edit_type_count_dict))
    print('3333', 'better data:', dict(better_data_count_dict))
    print('3333', 'bbox reference shape:',
          dict(bbox_reference_shape_count_dict))
    print('3333', 'bounding box state:', dict(bounding_box_state_count_dict))

    # 全量硬对账: sample_id/source_id必须严格覆盖0..N-1，
    # 任何缺号都说明有metadata分片没被完整读到，会直接少掉几千个样本对
    per_missing_sample_id_list = sorted(
        set(range(0, EXPECTED_TOTAL_SAMPLE_COUNT)) -
        set(all_sample_id_dict.keys()))
    if len(per_missing_sample_id_list) > 0:
        error_message_list.append(
            f'sample_id not continuous, missing index {per_missing_sample_id_list[:20]} total missing {len(per_missing_sample_id_list)}'
        )

    per_missing_source_id_list = sorted(
        set(range(0, EXPECTED_TOTAL_SOURCE_IMAGE_COUNT)) -
        set(all_source_id_dict.keys()))
    if len(per_missing_source_id_list) > 0:
        error_message_list.append(
            f'source_id not continuous, missing index {per_missing_source_id_list[:20]} total missing {len(per_missing_source_id_list)}'
        )

    if total_row_count != EXPECTED_TOTAL_SAMPLE_COUNT:
        error_message_list.append(
            f'total row count not match {total_row_count} != {EXPECTED_TOTAL_SAMPLE_COUNT}'
        )
    if len(all_sample_id_dict) != EXPECTED_TOTAL_SAMPLE_COUNT:
        error_message_list.append(
            f'total sample id count not match {len(all_sample_id_dict)} != {EXPECTED_TOTAL_SAMPLE_COUNT}'
        )
    if len(all_source_id_dict) != EXPECTED_TOTAL_SOURCE_IMAGE_COUNT:
        error_message_list.append(
            f'total source id count not match {len(all_source_id_dict)} != {EXPECTED_TOTAL_SOURCE_IMAGE_COUNT}'
        )
    if total_valid_sample_count != EXPECTED_VALID_SAMPLE_COUNT:
        error_message_list.append(
            f'total valid sample count not match {total_valid_sample_count} != {EXPECTED_VALID_SAMPLE_COUNT}'
        )
    if len(all_invalid_sample_list) != EXPECTED_INVALID_SAMPLE_COUNT:
        error_message_list.append(
            f'total invalid sample count not match {len(all_invalid_sample_list)} != {EXPECTED_INVALID_SAMPLE_COUNT}'
        )
    if total_valid_sample_count + len(
            all_invalid_sample_list) != total_row_count:
        error_message_list.append(
            f'total sample count not match {total_valid_sample_count} + {len(all_invalid_sample_list)} != {total_row_count}'
        )
    if total_valid_source_image_count != EXPECTED_VALID_SOURCE_IMAGE_COUNT:
        error_message_list.append(
            f'total valid source image count not match {total_valid_source_image_count} != {EXPECTED_VALID_SOURCE_IMAGE_COUNT}'
        )
    if total_orphan_source_image_count != EXPECTED_ORPHAN_SOURCE_IMAGE_COUNT:
        error_message_list.append(
            f'total orphan source image count not match {total_orphan_source_image_count} != {EXPECTED_ORPHAN_SOURCE_IMAGE_COUNT}'
        )

    for per_count_name, per_count_dict, per_expected_count_dict in [
        ['edit type', edit_type_count_dict, EXPECTED_EDIT_TYPE_COUNT_DICT],
        [
            'better data', better_data_count_dict,
            EXPECTED_BETTER_DATA_COUNT_DICT
        ],
        [
            'bbox reference shape', bbox_reference_shape_count_dict,
            EXPECTED_BBOX_REFERENCE_SHAPE_COUNT_DICT
        ],
    ]:
        if dict(per_count_dict) != per_expected_count_dict:
            error_message_list.append(
                f'{per_count_name} count dict not match {dict(per_count_dict)} != {per_expected_count_dict}'
            )

    # 白名单: asset分片 -> [该片声明的全部sample_id, 该片该落盘的sample_id]
    asset_member_index_dict = {}
    for per_shard_index, per_shard_sample_id_list in shard_sample_id_dict.items(
    ):
        asset_member_index_dict[per_shard_index] = [
            sorted(per_shard_sample_id_list),
            sorted([
                per_sample_id for per_sample_id in per_shard_sample_id_list
                if per_sample_id in valid_sample_id_dict
            ]),
        ]

    # 白名单: source分片 -> [该片声明的全部source_id, 该片该落盘的source_id]
    source_declare_index_dict = collections.defaultdict(list)
    for per_source_id in sorted(all_source_id_dict.keys()):
        source_declare_index_dict[per_source_id //
                                  EXPECTED_SOURCE_NUM_PER_SHARD].append(
                                      per_source_id)

    source_member_index_dict = {}
    for per_shard_index, per_shard_source_id_list in source_declare_index_dict.items(
    ):
        source_member_index_dict[per_shard_index] = [
            per_shard_source_id_list,
            [
                per_source_id for per_source_id in per_shard_source_id_list
                if per_source_id in valid_source_id_dict
            ],
        ]

    # 每个分片声明的条目数必须符合"每片固定条数、末片余数"的切分规格
    for per_shard_index in range(0, EXPECTED_ASSET_ARCHIVE_NUM):
        per_expected_sample_count = get_expected_shard_item_count(
            per_shard_index, EXPECTED_SAMPLE_NUM_PER_SHARD,
            EXPECTED_TOTAL_SAMPLE_COUNT)
        per_declare_sample_count = len(
            asset_member_index_dict.get(per_shard_index, [[], []])[0])
        if per_declare_sample_count != per_expected_sample_count:
            error_message_list.append(
                f'asset shard {per_shard_index} declare sample count not match {per_declare_sample_count} != {per_expected_sample_count}'
            )

    for per_shard_index in range(0, EXPECTED_SOURCE_ARCHIVE_NUM):
        per_expected_source_count = get_expected_shard_item_count(
            per_shard_index, EXPECTED_SOURCE_NUM_PER_SHARD,
            EXPECTED_TOTAL_SOURCE_IMAGE_COUNT)
        per_declare_source_count = len(
            source_member_index_dict.get(per_shard_index, [[], []])[0])
        if per_declare_source_count != per_expected_source_count:
            error_message_list.append(
                f'source shard {per_shard_index} declare source count not match {per_declare_source_count} != {per_expected_source_count}'
            )

    metadata_check_result_dict = {
        'total_row_count': total_row_count,
        'total_sample_id_count': len(all_sample_id_dict),
        'total_source_id_count': len(all_source_id_dict),
        'total_valid_sample_count': total_valid_sample_count,
        'total_valid_source_image_count': total_valid_source_image_count,
        'total_orphan_source_image_count': total_orphan_source_image_count,
        'invalid_sample_list': all_invalid_sample_list,
        'shard_valid_sample_pair_count_dict':
        shard_valid_sample_pair_count_dict,
        'edit_type_count_dict': dict(edit_type_count_dict),
        'better_data_count_dict': dict(better_data_count_dict),
        'bbox_reference_shape_count_dict':
        dict(bbox_reference_shape_count_dict),
        'bounding_box_state_count_dict': dict(bounding_box_state_count_dict),
    }

    return (asset_member_index_dict, source_member_index_dict,
            metadata_check_result_dict, error_message_list)


def get_all_archive_group(asset_archive_group_list, source_archive_group_list,
                          asset_member_index_dict, source_member_index_dict):
    """按metadata算出来的白名单，给每个tar配上"该片声明的成员index"与"该落盘的成员index"

    解包worker只认这两个列表，所以要丢弃的107个样本对(107个target + 107个mask)
    和30张孤儿源图从一开始就不会被写到NAS上。
    """
    archive_group_list = []
    error_message_list = []

    for per_archive_group_list, per_archive_kind, per_member_index_dict in [
        [
            asset_archive_group_list, ARCHIVE_KIND_ASSET,
            asset_member_index_dict
        ],
        [
            source_archive_group_list, ARCHIVE_KIND_SOURCE,
            source_member_index_dict
        ],
    ]:
        for (per_archive_group_name, per_archive_relative_dir,
             per_archive_path, per_shard_index, _) in per_archive_group_list:
            if per_shard_index not in per_member_index_dict:
                error_message_list.append(
                    f'{per_archive_relative_dir}/{per_archive_group_name} not declared in metadata'
                )
                continue

            per_declare_member_index_list, per_valid_member_index_list = per_member_index_dict[
                per_shard_index]
            archive_group_list.append([
                per_archive_group_name,
                per_archive_relative_dir,
                per_archive_path,
                per_archive_kind,
                per_declare_member_index_list,
                per_valid_member_index_list,
            ])

    return archive_group_list, error_message_list


def get_all_annotation_group(metadata_group_list, metadata_result_list,
                             target_image_shape_dict, mask_image_shape_dict,
                             source_image_shape_dict):
    """给每个metadata分片配上它自己那批图的真实宽高子集，避免整份宽高表进程间来回拷"""
    metadata_result_dict = {
        per_metadata_result['shard_index']: per_metadata_result
        for per_metadata_result in metadata_result_list
    }

    annotation_group_list = []
    for (per_metadata_group_name, per_metadata_relative_dir, per_metadata_path,
         per_shard_index, _) in metadata_group_list:
        per_metadata_result = metadata_result_dict.get(per_shard_index, None)
        if per_metadata_result is None:
            continue

        per_target_image_shape_dict, per_mask_image_shape_dict = {}, {}
        per_source_image_shape_dict = {}
        for per_sample_id, per_source_id in per_metadata_result[
                'valid_sample_pair_list']:
            if per_sample_id in target_image_shape_dict:
                per_target_image_shape_dict[
                    per_sample_id] = target_image_shape_dict[per_sample_id]
            if per_sample_id in mask_image_shape_dict:
                per_mask_image_shape_dict[
                    per_sample_id] = mask_image_shape_dict[per_sample_id]
            if per_source_id in source_image_shape_dict:
                per_source_image_shape_dict[
                    per_source_id] = source_image_shape_dict[per_source_id]

        annotation_group_list.append([
            per_metadata_group_name,
            per_metadata_relative_dir,
            per_metadata_path,
            per_shard_index,
            per_target_image_shape_dict,
            per_mask_image_shape_dict,
            per_source_image_shape_dict,
        ])

    return annotation_group_list


def save_check_result(save_dataset_path, metadata_check_result_dict,
                      archive_result_list, annotation_result_list):
    """汇总三个阶段的结果，落盘一份校验报告并返回错误信息列表"""
    total_file_member_count = 0
    total_extract_file_count, total_skip_file_count = 0, 0
    total_not_save_file_count, total_drop_file_count = 0, 0
    total_save_fail_count, total_duplicate_member_count = 0, 0
    member_kind_count_dict = collections.Counter()
    image_shape_count_dict = collections.Counter()
    archive_member_count_dict = {}
    unknown_member_name_list = []
    error_message_list = []

    for per_archive_result in archive_result_list:
        per_archive_relative_path = per_archive_result['archive_relative_path']

        total_file_member_count += per_archive_result[
            'total_file_member_count']
        total_extract_file_count += per_archive_result['extract_file_count']
        total_skip_file_count += per_archive_result['skip_file_count']
        total_not_save_file_count += per_archive_result['not_save_file_count']
        total_drop_file_count += per_archive_result['drop_file_count']
        total_save_fail_count += per_archive_result['save_fail_count']
        total_duplicate_member_count += per_archive_result[
            'duplicate_member_count']

        member_kind_count_dict.update(
            per_archive_result['member_kind_count_dict'])
        image_shape_count_dict.update(
            per_archive_result['image_shape_count_dict'])
        archive_member_count_dict[per_archive_relative_path] = [
            per_archive_result['total_file_member_count'],
            per_archive_result['extract_file_count'],
            per_archive_result['skip_file_count'],
            per_archive_result['drop_file_count'],
        ]
        unknown_member_name_list.extend([
            f'{per_archive_relative_path}/{per_member_name}' for
            per_member_name in per_archive_result['unknown_member_name_list']
        ])

        if len(per_archive_result['error_message_list']) > 0:
            print('7777', per_archive_relative_path,
                  per_archive_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_archive_relative_path} error num {len(per_archive_result["error_message_list"])} {per_archive_result["error_message_list"][:3]}'
            )

    total_annotation_row_count, total_valid_sample_pair_count = 0, 0
    total_annotation_invalid_sample_count = 0
    total_missing_image_shape_count = 0
    mask_image_shape_state_count_dict = collections.Counter()
    bounding_box_state_count_dict = collections.Counter()
    annotation_edit_type_count_dict = collections.Counter()
    shard_valid_sample_pair_count_dict = {}
    all_warning_message_list = []

    for per_annotation_result in annotation_result_list:
        per_metadata_relative_path = per_annotation_result[
            'metadata_relative_path']

        total_annotation_row_count += per_annotation_result['row_count']
        total_valid_sample_pair_count += per_annotation_result[
            'valid_sample_pair_count']
        total_annotation_invalid_sample_count += per_annotation_result[
            'invalid_sample_count']
        total_missing_image_shape_count += per_annotation_result[
            'missing_image_shape_count']

        mask_image_shape_state_count_dict.update(
            per_annotation_result['mask_image_shape_state_count_dict'])
        bounding_box_state_count_dict.update(
            per_annotation_result['bounding_box_state_count_dict'])
        annotation_edit_type_count_dict.update(
            per_annotation_result['edit_type_count_dict'])
        shard_valid_sample_pair_count_dict[per_annotation_result[
            'shard_index']] = per_annotation_result['valid_sample_pair_count']

        all_warning_message_list.extend([
            f'{per_metadata_relative_path} {per_warning_message}'
            for per_warning_message in
            per_annotation_result['warning_message_list']
        ])

        if len(per_annotation_result['error_message_list']) > 0:
            print('7777', per_metadata_relative_path,
                  per_annotation_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_metadata_relative_path} error num {len(per_annotation_result["error_message_list"])} {per_annotation_result["error_message_list"][:3]}'
            )

    print('3333', 'total tar file member:', total_file_member_count,
          'extract:', total_extract_file_count, 'skip:', total_skip_file_count,
          'not save:', total_not_save_file_count, 'drop:',
          total_drop_file_count, 'save fail:', total_save_fail_count,
          'duplicate member:', total_duplicate_member_count, 'unknown member:',
          len(unknown_member_name_list))
    print('3333', 'member kind:', dict(member_kind_count_dict))
    print('3333', 'total annotation row:', total_annotation_row_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'annotation invalid sample:', total_annotation_invalid_sample_count,
          'missing image shape:', total_missing_image_shape_count, 'warning:',
          len(all_warning_message_list))
    print('3333', 'mask image shape state:',
          dict(mask_image_shape_state_count_dict))
    print('3333', 'bounding box state(valid sample):',
          dict(bounding_box_state_count_dict))
    print('3333', 'image shape top10:',
          dict(image_shape_count_dict.most_common(10)))

    per_expected_target_member_count = EXPECTED_TOTAL_SAMPLE_COUNT
    per_expected_source_member_count = EXPECTED_TOTAL_SOURCE_IMAGE_COUNT
    per_expected_save_image_count = EXPECTED_VALID_SAMPLE_COUNT * 2 + EXPECTED_VALID_SOURCE_IMAGE_COUNT
    per_expected_drop_image_count = EXPECTED_INVALID_SAMPLE_COUNT * 2 + EXPECTED_ORPHAN_SOURCE_IMAGE_COUNT

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'dataset_license_name':
        DATASET_LICENSE_NAME,
        'extract_image_file_flag':
        EXTRACT_IMAGE_FILE_FLAG,
        'parse_image_shape_flag':
        PARSE_IMAGE_SHAPE_FLAG,
        'total_archive_count':
        len(archive_result_list),
        'total_metadata_count':
        len(annotation_result_list),
        'total_tar_file_member_count':
        total_file_member_count,
        'total_extract_file_count':
        total_extract_file_count,
        'total_skip_file_count':
        total_skip_file_count,
        'total_not_save_file_count':
        total_not_save_file_count,
        'total_drop_file_count':
        total_drop_file_count,
        'expected_drop_file_count':
        per_expected_drop_image_count,
        'total_save_fail_count':
        total_save_fail_count,
        'total_duplicate_member_count':
        total_duplicate_member_count,
        'total_unknown_member_count':
        len(unknown_member_name_list),
        'total_annotation_row_count':
        total_annotation_row_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'expected_valid_sample_pair_count':
        EXPECTED_VALID_SAMPLE_COUNT,
        'total_invalid_sample_count':
        total_annotation_invalid_sample_count,
        'expected_invalid_sample_count':
        EXPECTED_INVALID_SAMPLE_COUNT,
        'total_missing_image_shape_count':
        total_missing_image_shape_count,
        'metadata_check_result_dict': {
            per_key_name: per_key_value
            for per_key_name, per_key_value in
            metadata_check_result_dict.items()
            if per_key_name != 'invalid_sample_list'
        },
        'member_kind_count_dict':
        dict(member_kind_count_dict),
        'mask_image_shape_state_count_dict':
        dict(mask_image_shape_state_count_dict),
        'bounding_box_state_count_dict':
        dict(bounding_box_state_count_dict),
        'annotation_edit_type_count_dict':
        dict(annotation_edit_type_count_dict),
        'image_shape_count_dict':
        dict(image_shape_count_dict),
        'archive_member_count_dict':
        archive_member_count_dict,
        'shard_valid_sample_pair_count_dict':
        shard_valid_sample_pair_count_dict,
        'drop_invalid_sample_list':
        metadata_check_result_dict['invalid_sample_list']
        [:MAX_SAVE_PROBLEM_ITEM_NUM],
        'unknown_member_name_list':
        sorted(unknown_member_name_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        sorted(set(all_warning_message_list))[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'check_error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    # 全量硬对账: 每个tar成员都必须有归属，每条有效样本对都必须写进标注，
    # 少一个都说明有样本对在"读metadata->解tar->写jsonl"这条链路上消失了
    if total_valid_sample_pair_count == 0:
        error_message_list.append('no valid sample pair found')
    if member_kind_count_dict[
            MEMBER_KIND_TARGET] != per_expected_target_member_count:
        error_message_list.append(
            f'total target member count not match {member_kind_count_dict[MEMBER_KIND_TARGET]} != {per_expected_target_member_count}'
        )
    if member_kind_count_dict[
            MEMBER_KIND_MASK] != per_expected_target_member_count:
        error_message_list.append(
            f'total mask member count not match {member_kind_count_dict[MEMBER_KIND_MASK]} != {per_expected_target_member_count}'
        )
    if member_kind_count_dict[
            MEMBER_KIND_SOURCE] != per_expected_source_member_count:
        error_message_list.append(
            f'total source member count not match {member_kind_count_dict[MEMBER_KIND_SOURCE]} != {per_expected_source_member_count}'
        )
    if total_file_member_count != per_expected_target_member_count * 2 + per_expected_source_member_count:
        error_message_list.append(
            f'total tar file member count not match {total_file_member_count} != {per_expected_target_member_count * 2 + per_expected_source_member_count}'
        )
    if total_extract_file_count + total_skip_file_count + total_not_save_file_count + total_drop_file_count + total_save_fail_count != total_file_member_count:
        error_message_list.append(
            f'total process file count not match {total_extract_file_count} + {total_skip_file_count} + {total_not_save_file_count} + {total_drop_file_count} + {total_save_fail_count} != {total_file_member_count}'
        )
    if total_drop_file_count != per_expected_drop_image_count:
        error_message_list.append(
            f'total drop file count not match {total_drop_file_count} != {per_expected_drop_image_count}'
        )
    if EXTRACT_IMAGE_FILE_FLAG and total_extract_file_count + total_skip_file_count != per_expected_save_image_count:
        error_message_list.append(
            f'total save image count not match {total_extract_file_count} + {total_skip_file_count} != {per_expected_save_image_count}'
        )
    if total_save_fail_count > 0:
        error_message_list.append(
            f'total save fail count {total_save_fail_count}')
    if total_duplicate_member_count > 0:
        error_message_list.append(
            f'total duplicate member count {total_duplicate_member_count}')
    if len(unknown_member_name_list) > 0:
        error_message_list.append(
            f'total unknown member count {len(unknown_member_name_list)}')
    if total_annotation_row_count != EXPECTED_TOTAL_SAMPLE_COUNT:
        error_message_list.append(
            f'total annotation row count not match {total_annotation_row_count} != {EXPECTED_TOTAL_SAMPLE_COUNT}'
        )
    if total_valid_sample_pair_count != EXPECTED_VALID_SAMPLE_COUNT:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} != {EXPECTED_VALID_SAMPLE_COUNT}'
        )
    if total_annotation_invalid_sample_count != EXPECTED_INVALID_SAMPLE_COUNT:
        error_message_list.append(
            f'total annotation invalid sample count not match {total_annotation_invalid_sample_count} != {EXPECTED_INVALID_SAMPLE_COUNT}'
        )
    if total_missing_image_shape_count > 0:
        error_message_list.append(
            f'total missing image shape count {total_missing_image_shape_count}'
        )
    if dict(annotation_edit_type_count_dict) == EXPECTED_EDIT_TYPE_COUNT_DICT:
        # 有效样本对少了107条，各edit_type的分布必然与全量分布不同，
        # 完全相等反而说明那107条没被丢掉
        error_message_list.append(
            'annotation edit type count dict same as full dataset, invalid sample not dropped'
        )

    # 阶段1(建白名单)与阶段3(写标注)对每一片算出来的有效条数必须完全相等，
    # 不等说明两阶段判定口径漂移了，白名单与标注会对不上
    for per_shard_index, per_valid_sample_pair_count in metadata_check_result_dict[
            'shard_valid_sample_pair_count_dict'].items():
        per_annotation_valid_sample_pair_count = shard_valid_sample_pair_count_dict.get(
            per_shard_index, 0)
        if per_annotation_valid_sample_pair_count != per_valid_sample_pair_count:
            error_message_list.append(
                f'shard {per_shard_index} valid sample pair count not match {per_annotation_valid_sample_pair_count} != {per_valid_sample_pair_count}'
            )

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    (file_copy_pair_list, metadata_group_list, asset_archive_group_list,
     source_archive_group_list,
     scan_error_message_list) = get_all_file_and_shard_group(root_dataset_path)

    print('1111', len(file_copy_pair_list), len(metadata_group_list),
          len(asset_archive_group_list), len(source_archive_group_list))
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(metadata_group_list) > 0:
        print('1111', metadata_group_list[0][0], metadata_group_list[0][1],
              metadata_group_list[0][3])
    if len(asset_archive_group_list) > 0:
        print('1111', asset_archive_group_list[0][0],
              asset_archive_group_list[0][1], asset_archive_group_list[0][3])
    if len(source_archive_group_list) > 0:
        print('1111', source_archive_group_list[0][0],
              source_archive_group_list[0][1], source_archive_group_list[0][3])

    precheck_error_message_list = scan_error_message_list + check_manifest_file(
        root_dataset_path) + check_required_shard_complete(
            metadata_group_list, asset_archive_group_list,
            source_archive_group_list)
    if len(precheck_error_message_list) > 0:
        # 数据集本身不完整就没必要跑几十小时解包
        raise Exception(
            f'check shard failed error num {len(precheck_error_message_list)} {precheck_error_message_list[:20]}'
        )

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    copy_error_message_list = []
    copy_func = partial(process_single_file_copy,
                        save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_copy_result in tqdm(pool.imap_unordered(
                copy_func, file_copy_pair_list),
                                    total=len(file_copy_pair_list)):
            per_file_relative_path, per_error_message = per_copy_result
            if per_error_message:
                copy_error_message_list.append(
                    f'{per_file_relative_path} {per_error_message}')

    # 阶段1: 只读48M的metadata，先把"该落盘成员白名单"算出来
    metadata_result_list = []
    with Pool(processes=PROCESS_NUM) as pool:
        for per_metadata_result in tqdm(pool.imap_unordered(
                process_single_metadata_file, metadata_group_list),
                                        total=len(metadata_group_list)):
            metadata_result_list.append(per_metadata_result)

            print('2222', per_metadata_result['metadata_relative_path'],
                  'row:', per_metadata_result['row_count'],
                  'valid sample pair:',
                  len(per_metadata_result['valid_sample_pair_list']),
                  'invalid sample:',
                  len(per_metadata_result['invalid_sample_list']))

    (asset_member_index_dict, source_member_index_dict,
     metadata_check_result_dict,
     metadata_error_message_list) = check_metadata_result(metadata_result_list)
    if len(metadata_error_message_list) > 0:
        # metadata都对不上就没必要解2T的tar
        raise Exception(
            f'check metadata failed error num {len(metadata_error_message_list)} {metadata_error_message_list[:20]}'
        )

    archive_group_list, archive_group_error_message_list = get_all_archive_group(
        asset_archive_group_list, source_archive_group_list,
        asset_member_index_dict, source_member_index_dict)
    if len(archive_group_error_message_list) > 0:
        raise Exception(
            f'get archive group failed error num {len(archive_group_error_message_list)} {archive_group_error_message_list[:20]}'
        )

    expected_archive_group_num = EXPECTED_ASSET_ARCHIVE_NUM + EXPECTED_SOURCE_ARCHIVE_NUM
    if len(archive_group_list) != expected_archive_group_num:
        raise Exception(
            f'archive group num not match {len(archive_group_list)} != {expected_archive_group_num}'
        )

    # 阶段2: 流式解2T的tar，只落盘白名单内的成员
    archive_result_list = []
    target_image_shape_dict, mask_image_shape_dict = {}, {}
    source_image_shape_dict = {}
    extract_func = partial(process_single_archive_group,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_archive_result in tqdm(pool.imap_unordered(
                extract_func, archive_group_list),
                                       total=len(archive_group_list)):
            target_image_shape_dict.update(
                per_archive_result.pop('target_image_shape_dict'))
            mask_image_shape_dict.update(
                per_archive_result.pop('mask_image_shape_dict'))
            source_image_shape_dict.update(
                per_archive_result.pop('source_image_shape_dict'))
            archive_result_list.append(per_archive_result)

            print('2222', per_archive_result['archive_relative_path'],
                  'tar file member:',
                  per_archive_result['total_file_member_count'], 'extract:',
                  per_archive_result['extract_file_count'], 'skip:',
                  per_archive_result['skip_file_count'], 'not save:',
                  per_archive_result['not_save_file_count'], 'drop:',
                  per_archive_result['drop_file_count'], 'save fail:',
                  per_archive_result['save_fail_count'], 'duplicate member:',
                  per_archive_result['duplicate_member_count'])

    # 阶段3: 重新读一遍metadata，join真实宽高后写jsonl汇总标注
    annotation_group_list = get_all_annotation_group(metadata_group_list,
                                                     metadata_result_list,
                                                     target_image_shape_dict,
                                                     mask_image_shape_dict,
                                                     source_image_shape_dict)

    annotation_result_list = []
    annotation_func = partial(
        process_single_annotation_file,
        save_dataset_path=save_dataset_path,
        save_annotation_dir_path=save_annotation_dir_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_annotation_result in tqdm(pool.imap_unordered(
                annotation_func, annotation_group_list),
                                          total=len(annotation_group_list)):
            annotation_result_list.append(per_annotation_result)

            print('2222', per_annotation_result['metadata_relative_path'],
                  'row:', per_annotation_result['row_count'],
                  'valid sample pair:',
                  per_annotation_result['valid_sample_pair_count'],
                  'invalid sample:',
                  per_annotation_result['invalid_sample_count'],
                  'missing image shape:',
                  per_annotation_result['missing_image_shape_count'])

    check_error_message_list = save_check_result(save_dataset_path,
                                                 metadata_check_result_dict,
                                                 archive_result_list,
                                                 annotation_result_list)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG and EXTRACT_IMAGE_FILE_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, archive_group_list)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解包/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/Inter-Edit-Train'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
