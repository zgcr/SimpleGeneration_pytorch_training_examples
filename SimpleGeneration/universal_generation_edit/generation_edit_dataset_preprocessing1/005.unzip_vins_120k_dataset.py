import os
import re
import json
import shutil
import tarfile
import collections

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: VINS-120K(Ultra High-Resolution Image Editing, vivo/NJU-PCALab)
#
# 【数据集类型】纯图像编辑(instruction-based image editing)数据集，不是文生图数据集。
# 每个样本对固定是"1张参考图(input) + 1张编辑后图(output) + 1条英文编辑指令(instruction)
# + 1个编辑类型(edit_type)"，所有图像分辨率不低于4K；README声明的task_categories就是
# image-text-to-image，13类编辑覆盖局部编辑/全局编辑/镜头运动/个性化生成。
# 下游只能走ti2i_dataset.py那条链路，不能当t2i(文生图)数据用。
#
# 【root_dataset_path实测原始保存规格】
# VINS-120K/
# ├── nano-consistent.json    22224行标注(有用，唯一的文本提示来源)
# ├── ultravideo.json         25807行标注(有用)
# ├── x2edit.json             83004行标注(有用)
# ├── nano-consistent/        Image.tar.split.{000..012} 13片，合并531.8G，1个tar组
# ├── ultravideo/             clips_short_{1..36}.tar   36个独立完整tar，共780G
# ├── x2edit/                 {0..7}.tar.split.* 共98片，8个tar组，共4.0T
# │                           实测分片数 0->19 1->19 2->11 3->11 4->12 5->12 6->11 7->3
# ├── benchmark.tar           1.9G，VINS-4KEval评测集(100张图 + instructions.json 509行)
# ├── assets/                 vins120k_overview.jpg，README里的示意图(无用)
# ├── README.md               数据集说明(无用)
# ├── .gitattributes          git lfs配置(无用)
# └── .cache/                 huggingface下载缓存，359个文件，里面还残留20多个*.incomplete(无用)
#
# 每个tar内部都带一层"打包时的顶层目录"，而且这个顶层目录名和标注json里的路径前缀
# **大面积不一致**，必须剥掉顶层目录后统一改挂到标注使用的规范子集名下:
#   nano-consistent/Image      tar顶层 ./Nano-consistent/  json写 Image/orignal/...
#   ultravideo/clips_short_N   tar顶层 ./UltraVideo/        json写 clips_short_N/clips_short/<uuid>/0000.png
#   x2edit/{0..7}              tar顶层 ./X2Edit/            json写 3/00026/input/000000001.png
#   benchmark.tar              tar顶层 ./benchmark/         instructions.json写 animals/animals_00001.png
# 剥掉顶层目录后，标注里的相对路径拼上 images/<子集名>/ 前缀就是磁盘真实路径，
# 校验可以按**完整路径精确匹配**。不能按路径尾部模糊匹配: x2edit的
# input/000000001.png 这种尾部在8个tar组之间大面积重名，模糊匹配会把缺图判成存在。
#
# 【解压前预检实测结果】
# 46个tar组(1个nano + 36个ultravideo + 8个x2edit + 1个benchmark)总长度全部512字节对齐、
# 末尾1024字节EOF块全0完好(跨分片从后往前拼尾部)，说明当前数据集是完整的；
# 但.cache里残留20多个*.incomplete，说明下载确实中断过，风险是真实存在的，
# 所以预检必须在跑5.3T解压前先跑完。
#
# 【tar成员与样本对一一对应实测】
# 全量扫描 ultravideo/clips_short_1.tar: 163个目录成员 + 608个png文件成员 + 0个其他类型成员，
# 而 ultravideo.json 里 clips_short_1 引用的唯一图正好608张，**双向零差集**
# (无missing、无orphan)，说明tar内除png外没有任何冗余/垃圾文件，也没有多余的未引用图。
# benchmark.tar: 111个成员 = 9个目录 + 100个png + 1个instructions.json，
# instructions.json 509行的id去重后正好100个，和100张png双向零差集。
#
# 【单条标注的全部字段(实测131035/131035行4个字段全齐备)】
#   edit_type   : 编辑类型(13类)                            -> 有用(可做任务采样/加权)
#   input       : 参考图(编辑前原图)相对路径，一张           -> 有用
#   output      : 编辑后图相对路径，一张                     -> 有用
#   instruction : 英文编辑指令，唯一的训练文本               -> 有用(必需)
#
# 【实测脏数据，全部隔离进invalid_sample_pair_list上报，不静默丢也不混进训练标注】
# 1) nano-consistent.json 有25行 instruction 是空串         -> 没有文本提示不可训练
# 2) ultravideo.json 有1行 edit_type 是null                 -> 属性残缺
# 3) ultravideo.json 有132行 edit_type=NO_CHANGE 且
#    instruction="NO_CHANGE."                               -> "无变化"对照样本，不是真编辑对
# 合计158行被隔离，剩余130877个完整有用信息的样本对，引用212107张唯一图；
# 三个json内部 input==output 的行为0，跨clip/跨input-output目录错位的行为0。
#
# 【本脚本的处理口径】
# - 解压前预检: tar组名必须已知、分片数必须和实测一致、分片编号必须从0连号、
#   整包512对齐 + 尾部1024字节EOF块(O(1)读尾部)、3个标注json的行数与字段齐备性;
# - 解压时剥掉tar自带顶层目录、改挂规范子集名，落盘为 images/<子集名>/<标注相对路径>,
#   标注路径拼前缀后就是磁盘真实路径，后续按完整路径精确匹配校验;
# - 每个成员写盘后立即校验落盘大小 == tar头里的大小，并对账
#   extract + skip == tar里的文件成员总数，且必须正常读到tar流结束;
# - 每个tar组落盘图像数不能少于该组标注引用的唯一图数(少了就是丢样本，硬失败);
#   多了只告警(说明tar里有未被标注引用的多余图，不影响样本对完整性);
# - 3个标注json各解成 annotations/<name>.jsonl，每行一个完整样本对，保留原始全部属性;
# - benchmark.tar是评测集，单独解到 benchmark/ 并把instructions.json解成
#   annotations/benchmark.jsonl，不混进训练样本对统计;
# - 解压+解析完后按子集建落盘图像索引，再逐条回读jsonl做样本对级校验，
#   保证"已写jsonl条数 + 隔离的无效样本对数 == 标注总行数"、缺图数必须为0;
# - 无用信息(.cache/.gitattributes/README.md/assets)全部不拷贝;
# - 任何一环出错都汇总后抛异常并sys.exit(1)，不再静默跑过。
# ==============================================================================

ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.tar\.split\.(?P<part>\d+)$'),
    re.compile(r'^(?P<prefix>.+)\.tar$'),
]

ANNOTATION_FILE_NAME_PATTERN = re.compile(r'^(?P<prefix>.+)\.json$')

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(359个文件，含20多个*.incomplete)
# .gitattributes   git lfs配置
# .gitignore       git配置
# README.md        数据集说明
# assets/          README里的示意图(vins120k_overview.jpg)
# .DS_Store/CACHEDIR.TAG  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    'assets',
    '.DS_Store',
    'CACHEDIR.TAG',
]

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.gif',
    '.webp',
    '.tif',
    '.tiff',
]

# ---------------------------------------------------------------------------
# 压缩包组配置
#
# tar_top_strip_prefix = 压缩包内要剥掉的顶层目录(已归一化，''表示顶层就是./无需剥)
# subset_name          = 落盘用的规范子集名(等于标注json的文件名前缀)
# subset_inner_prefix  = 剥掉顶层目录后，该组所有成员必须共有的第一级目录名
#                        (等于标注json里路径的第一段，用来确认成员没有跨组错位)
ARCHIVE_GROUP_CONFIG_DICT = {
    'nano-consistent/Image': ['Nano-consistent', 'nano-consistent', 'Image'],
    'ultravideo/clips_short_1': ['UltraVideo', 'ultravideo', 'clips_short_1'],
    'ultravideo/clips_short_2': ['UltraVideo', 'ultravideo', 'clips_short_2'],
    'ultravideo/clips_short_3': ['UltraVideo', 'ultravideo', 'clips_short_3'],
    'ultravideo/clips_short_4': ['UltraVideo', 'ultravideo', 'clips_short_4'],
    'ultravideo/clips_short_5': ['UltraVideo', 'ultravideo', 'clips_short_5'],
    'ultravideo/clips_short_6': ['UltraVideo', 'ultravideo', 'clips_short_6'],
    'ultravideo/clips_short_7': ['UltraVideo', 'ultravideo', 'clips_short_7'],
    'ultravideo/clips_short_8': ['UltraVideo', 'ultravideo', 'clips_short_8'],
    'ultravideo/clips_short_9': ['UltraVideo', 'ultravideo', 'clips_short_9'],
    'ultravideo/clips_short_10':
    ['UltraVideo', 'ultravideo', 'clips_short_10'],
    'ultravideo/clips_short_11':
    ['UltraVideo', 'ultravideo', 'clips_short_11'],
    'ultravideo/clips_short_12':
    ['UltraVideo', 'ultravideo', 'clips_short_12'],
    'ultravideo/clips_short_13':
    ['UltraVideo', 'ultravideo', 'clips_short_13'],
    'ultravideo/clips_short_14':
    ['UltraVideo', 'ultravideo', 'clips_short_14'],
    'ultravideo/clips_short_15':
    ['UltraVideo', 'ultravideo', 'clips_short_15'],
    'ultravideo/clips_short_16':
    ['UltraVideo', 'ultravideo', 'clips_short_16'],
    'ultravideo/clips_short_17':
    ['UltraVideo', 'ultravideo', 'clips_short_17'],
    'ultravideo/clips_short_18':
    ['UltraVideo', 'ultravideo', 'clips_short_18'],
    'ultravideo/clips_short_19':
    ['UltraVideo', 'ultravideo', 'clips_short_19'],
    'ultravideo/clips_short_20':
    ['UltraVideo', 'ultravideo', 'clips_short_20'],
    'ultravideo/clips_short_21':
    ['UltraVideo', 'ultravideo', 'clips_short_21'],
    'ultravideo/clips_short_22': [
        'UltraVideo', 'ultravideo', 'clips_short_22'
    ],
    'ultravideo/clips_short_23': [
        'UltraVideo', 'ultravideo', 'clips_short_23'
    ],
    'ultravideo/clips_short_24': [
        'UltraVideo', 'ultravideo', 'clips_short_24'
    ],
    'ultravideo/clips_short_25': [
        'UltraVideo', 'ultravideo', 'clips_short_25'
    ],
    'ultravideo/clips_short_26': [
        'UltraVideo', 'ultravideo', 'clips_short_26'
    ],
    'ultravideo/clips_short_27': [
        'UltraVideo', 'ultravideo', 'clips_short_27'
    ],
    'ultravideo/clips_short_28': [
        'UltraVideo', 'ultravideo', 'clips_short_28'
    ],
    'ultravideo/clips_short_29': [
        'UltraVideo', 'ultravideo', 'clips_short_29'
    ],
    'ultravideo/clips_short_30': [
        'UltraVideo', 'ultravideo', 'clips_short_30'
    ],
    'ultravideo/clips_short_31': [
        'UltraVideo', 'ultravideo', 'clips_short_31'
    ],
    'ultravideo/clips_short_32': [
        'UltraVideo', 'ultravideo', 'clips_short_32'
    ],
    'ultravideo/clips_short_33': [
        'UltraVideo', 'ultravideo', 'clips_short_33'
    ],
    'ultravideo/clips_short_34': [
        'UltraVideo', 'ultravideo', 'clips_short_34'
    ],
    'ultravideo/clips_short_35': [
        'UltraVideo', 'ultravideo', 'clips_short_35'
    ],
    'ultravideo/clips_short_36': [
        'UltraVideo', 'ultravideo', 'clips_short_36'
    ],
    'x2edit/0': ['X2Edit', 'x2edit', '0'],
    'x2edit/1': ['X2Edit', 'x2edit', '1'],
    'x2edit/2': ['X2Edit', 'x2edit', '2'],
    'x2edit/3': ['X2Edit', 'x2edit', '3'],
    'x2edit/4': ['X2Edit', 'x2edit', '4'],
    'x2edit/5': ['X2Edit', 'x2edit', '5'],
    'x2edit/6': ['X2Edit', 'x2edit', '6'],
    'x2edit/7': ['X2Edit', 'x2edit', '7'],
}

# benchmark.tar是VINS-4KEval评测集(100张4K图 + instructions.json 509行)，
# 和三个训练标注json的体系完全无关，单独解压到benchmark/，不参与训练样本对校验。
BENCHMARK_ARCHIVE_GROUP_KEY = './benchmark'

BENCHMARK_TAR_TOP_STRIP_PREFIX = 'benchmark'

BENCHMARK_IMAGE_DIR_NAME = 'img'

BENCHMARK_INSTRUCTION_FILE_NAME = 'instructions.json'

BENCHMARK_SUBSET_NAME = 'benchmark'

# 各压缩包组的分片数(实测)。缺片会让tar流在中途截断、后面的样本全部丢失，
# 而tarfile只会抛一个"unexpected end of data"，所以必须在解压前硬对账。
EXPECTED_ARCHIVE_PART_NUM_DICT = {
    './benchmark': 1,
    'nano-consistent/Image': 13,
    'ultravideo/clips_short_1': 1,
    'ultravideo/clips_short_2': 1,
    'ultravideo/clips_short_3': 1,
    'ultravideo/clips_short_4': 1,
    'ultravideo/clips_short_5': 1,
    'ultravideo/clips_short_6': 1,
    'ultravideo/clips_short_7': 1,
    'ultravideo/clips_short_8': 1,
    'ultravideo/clips_short_9': 1,
    'ultravideo/clips_short_10': 1,
    'ultravideo/clips_short_11': 1,
    'ultravideo/clips_short_12': 1,
    'ultravideo/clips_short_13': 1,
    'ultravideo/clips_short_14': 1,
    'ultravideo/clips_short_15': 1,
    'ultravideo/clips_short_16': 1,
    'ultravideo/clips_short_17': 1,
    'ultravideo/clips_short_18': 1,
    'ultravideo/clips_short_19': 1,
    'ultravideo/clips_short_20': 1,
    'ultravideo/clips_short_21': 1,
    'ultravideo/clips_short_22': 1,
    'ultravideo/clips_short_23': 1,
    'ultravideo/clips_short_24': 1,
    'ultravideo/clips_short_25': 1,
    'ultravideo/clips_short_26': 1,
    'ultravideo/clips_short_27': 1,
    'ultravideo/clips_short_28': 1,
    'ultravideo/clips_short_29': 1,
    'ultravideo/clips_short_30': 1,
    'ultravideo/clips_short_31': 1,
    'ultravideo/clips_short_32': 1,
    'ultravideo/clips_short_33': 1,
    'ultravideo/clips_short_34': 1,
    'ultravideo/clips_short_35': 1,
    'ultravideo/clips_short_36': 1,
    'x2edit/0': 19,
    'x2edit/1': 19,
    'x2edit/2': 11,
    'x2edit/3': 11,
    'x2edit/4': 12,
    'x2edit/5': 12,
    'x2edit/6': 11,
    'x2edit/7': 3,
}

# 各压缩包组应该解出来的图像数(实测: 等于该组标注引用的唯一图数，
# clips_short_1和benchmark做过全量扫描双向零差集验证)。
# 落盘图像数 < 期望值 就是丢样本，硬失败;
# 落盘图像数 > 期望值 只告警(tar里有未被标注引用的多余图，不影响样本对完整性)。
EXPECTED_ARCHIVE_GROUP_IMAGE_NUM_DICT = {
    './benchmark': 100,
    'nano-consistent/Image': 25189,
    'ultravideo/clips_short_1': 608,
    'ultravideo/clips_short_2': 604,
    'ultravideo/clips_short_3': 595,
    'ultravideo/clips_short_4': 625,
    'ultravideo/clips_short_5': 521,
    'ultravideo/clips_short_6': 568,
    'ultravideo/clips_short_7': 575,
    'ultravideo/clips_short_8': 587,
    'ultravideo/clips_short_9': 590,
    'ultravideo/clips_short_10': 523,
    'ultravideo/clips_short_11': 581,
    'ultravideo/clips_short_12': 638,
    'ultravideo/clips_short_13': 514,
    'ultravideo/clips_short_14': 585,
    'ultravideo/clips_short_15': 557,
    'ultravideo/clips_short_16': 621,
    'ultravideo/clips_short_17': 597,
    'ultravideo/clips_short_18': 617,
    'ultravideo/clips_short_19': 526,
    'ultravideo/clips_short_20': 577,
    'ultravideo/clips_short_21': 624,
    'ultravideo/clips_short_22': 646,
    'ultravideo/clips_short_23': 517,
    'ultravideo/clips_short_24': 624,
    'ultravideo/clips_short_25': 575,
    'ultravideo/clips_short_26': 586,
    'ultravideo/clips_short_27': 608,
    'ultravideo/clips_short_28': 531,
    'ultravideo/clips_short_29': 592,
    'ultravideo/clips_short_30': 580,
    'ultravideo/clips_short_31': 557,
    'ultravideo/clips_short_32': 600,
    'ultravideo/clips_short_33': 625,
    'ultravideo/clips_short_34': 598,
    'ultravideo/clips_short_35': 608,
    'ultravideo/clips_short_36': 507,
    'x2edit/0': 32540,
    'x2edit/1': 31214,
    'x2edit/2': 17646,
    'x2edit/3': 18352,
    'x2edit/4': 21990,
    'x2edit/5': 20440,
    'x2edit/6': 19876,
    'x2edit/7': 3950,
}

# ---------------------------------------------------------------------------
# 标注json配置: 文件名前缀 -> [规范子集名, 期望总行数, 期望有效样本对数,
#                             期望无效(隔离)样本对数, 期望有效样本对引用的唯一图数]
# 全部是实测值，是"保证每个完整样本对都被处理且没被静默丢掉"能被验证的关键兜底。
ANNOTATION_FILE_CONFIG_DICT = {
    'nano-consistent': ['nano-consistent', 22224, 22199, 25, 25164],
    'ultravideo': ['ultravideo', 25807, 25674, 133, 20935],
    'x2edit': ['x2edit', 83004, 83004, 0, 166008],
}

ANNOTATION_EDIT_TYPE_KEY_NAME = 'edit_type'

ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'input'

ANNOTATION_TARGET_IMAGE_KEY_NAME = 'output'

ANNOTATION_TEXT_KEY_NAME = 'instruction'

# 每行必须齐备的字段(实测131035/131035行全齐备，一旦缺失说明数据规格变了，必须显式感知)
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    ANNOTATION_EDIT_TYPE_KEY_NAME,
    ANNOTATION_REFERENCE_IMAGE_KEY_NAME,
    ANNOTATION_TARGET_IMAGE_KEY_NAME,
    ANNOTATION_TEXT_KEY_NAME,
]

# edit_type为该值时是"无变化"对照样本(instruction也只有"NO_CHANGE.")，
# 不是真正的编辑对，隔离进invalid_sample_pair_list，不写进训练标注
NO_CHANGE_EDIT_TYPE_NAME = 'NO_CHANGE'

# 全量期望值(实测)
EXPECTED_ANNOTATION_FILE_NUM = 3

EXPECTED_TOTAL_ANNOTATION_ROW_COUNT = 131035

EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT = 130877

EXPECTED_TOTAL_INVALID_SAMPLE_PAIR_COUNT = 158

# 每个样本对固定1张参考图 + 1张编辑后图
EXPECTED_REFERENCE_IMAGE_NUM_PER_SAMPLE_PAIR = 1

EXPECTED_TOTAL_VALID_UNIQUE_IMAGE_COUNT = 212107

# 三个标注json全部行(含被隔离的158行)引用的唯一图数，也等于三个tar子集里的图像总数
EXPECTED_TOTAL_UNZIP_IMAGE_COUNT = 212184

# 只被隔离行引用、不被任何有效样本对引用的图(212184 - 212107)，只告警不判失败
EXPECTED_TOTAL_ORPHAN_IMAGE_COUNT = 77

EXPECTED_BENCHMARK_ROW_COUNT = 509

EXPECTED_BENCHMARK_IMAGE_COUNT = 100

BENCHMARK_IMAGE_KEY_NAME = 'id'

BENCHMARK_EXPECTED_KEY_NAME_LIST = [
    'edit_type',
    'index',
    'id',
    'instruction',
]

SAVE_IMAGE_ROOT_NAME = 'images'

SAVE_BENCHMARK_ROOT_NAME = 'benchmark'

SAVE_ANNOTATION_ROOT_NAME = 'annotations'

SAVE_IMAGE_INDEX_ROOT_NAME = 'image_index'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_result.json'

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

TAR_BLOCK_SIZE = 512

TAR_EOF_BLOCK_SIZE = 1024

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024


class MultiPartArchiveReader:
    """把按字节切分的多个分片压缩包拼接成一个只读的连续字节流

    nano-consistent和x2edit的tar都是按字节切成多片(*.tar.split.NNN)，
    单片本身不是合法tar，必须按编号顺序拼成连续流才能流式解压;
    ultravideo和benchmark是单片完整tar，走同一条路径即可。
    """

    def __init__(self, per_archive_part_path_list):
        self.per_archive_part_path_list = per_archive_part_path_list
        self.current_part_index = 0
        self.current_part_file = open(
            self.per_archive_part_path_list[self.current_part_index], 'rb')

    def read(self, read_size=-1):
        if read_size is None or read_size < 0:
            read_bytes_list = []
            while True:
                per_read_bytes = self.read(EXTRACT_FILE_BLOCK_SIZE)
                if not per_read_bytes:
                    break
                read_bytes_list.append(per_read_bytes)

            return b''.join(read_bytes_list)

        read_bytes_list, remain_read_size = [], read_size
        while remain_read_size > 0:
            if self.current_part_file is None:
                break

            per_read_bytes = self.current_part_file.read(remain_read_size)
            if per_read_bytes:
                read_bytes_list.append(per_read_bytes)
                remain_read_size -= len(per_read_bytes)
                continue

            self.current_part_file.close()
            self.current_part_file = None
            self.current_part_index += 1
            if self.current_part_index < len(self.per_archive_part_path_list):
                self.current_part_file = open(
                    self.per_archive_part_path_list[self.current_part_index],
                    'rb')

        return b''.join(read_bytes_list)

    def close(self):
        if self.current_part_file is not None:
            self.current_part_file.close()
            self.current_part_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


def check_skip_file_or_dir(per_file_relative_path):
    """过滤掉.cache、.gitattributes、README.md、assets这些不需要整理的文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """判断是否是图像文件后缀"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


def get_archive_part_sort_key(per_archive_part_index):
    """分片编号排序key: 纯数字编号按数值排序，字母编号按位数优先再按字典序排序

    返回值统一是[编号类型, 数值编号, 字母编号]三元组，保证不同命名风格之间也能比较。
    不能直接按字符串排序: split.9 会排到 split.10 后面；
    也不能只按[位数, 字符串]排序: split.100 会排到 split.89 前面，
    分片一旦错位整个tar流就从错位处开始全部解不出来。
    """
    per_archive_part_index = per_archive_part_index or ''
    if per_archive_part_index.isdigit():
        return [0, int(per_archive_part_index), '']

    return [1, len(per_archive_part_index), per_archive_part_index]


def get_normalized_member_name(per_member_name):
    """把tar成员名归一化成不带盘符/前导斜杠/./的相对路径

    返回[归一化名, 是否是压缩包根成员]。
    本数据集每个tar的第一个成员都是 ./<顶层目录>，归一化后就是顶层目录名本身；
    如果出现 ./ 这种纯根成员，它是合法的压缩包根目录而不是非法成员，要区别对待。
    越界成员(..开头)返回['', False]。
    """
    per_member_name = per_member_name.replace('\\', '/').lstrip('/')
    if not per_member_name:
        return '', True

    per_member_name = os.path.normpath(per_member_name).replace('\\', '/')
    if per_member_name == '.':
        return '', True

    if per_member_name == '..' or per_member_name.startswith('../'):
        return '', False

    return per_member_name, False


def strip_member_top_prefix(per_member_name, per_tar_top_strip_prefix):
    """剥掉压缩包自带的顶层目录，返回子集内相对路径

    压缩包顶层目录名(Nano-consistent/UltraVideo/X2Edit/benchmark)和标注json里的
    路径前缀完全不一致，剥掉之后统一改挂到规范子集名下，标注路径才能和磁盘路径精确对上。
    成员正好是顶层目录本身时返回空串(调用方跳过)，前缀对不上时返回None(调用方记error)。
    """
    if not per_tar_top_strip_prefix:
        return per_member_name

    if per_member_name == per_tar_top_strip_prefix:
        return ''

    if per_member_name.startswith(f'{per_tar_top_strip_prefix}/'):
        return per_member_name[len(per_tar_top_strip_prefix) + 1:]

    return None


def check_single_archive_tar_tail(per_archive_part_path_list):
    """O(1)预检压缩包是否被截断: 总长度必须512字节对齐，且整包结尾必须有1024字节全0的EOF块

    本数据集的tar都是未压缩tar(单片或按字节切片)，所以只读整包尾部1024字节就能判断
    是否完整，不用把5.3T读一遍。缺尾片、写盘写一半这两种最隐蔽的情况都能在解压前拦住。
    注意EOF块可能跨分片边界(末片本身可能不足1024字节)，所以要从后往前跨片拼尾部字节。
    """
    error_message_list = []

    try:
        per_archive_part_size_list = [
            os.path.getsize(per_archive_part_path)
            for per_archive_part_path in per_archive_part_path_list
        ]
    except Exception as e:
        error_message_list.append(f'read archive part size failed {e}')

        return error_message_list

    total_archive_size = sum(per_archive_part_size_list)

    if total_archive_size == 0:
        error_message_list.append('archive total size is 0')

        return error_message_list

    if total_archive_size % TAR_BLOCK_SIZE != 0:
        error_message_list.append(
            f'archive total size {total_archive_size} not aligned to {TAR_BLOCK_SIZE}'
        )

    if total_archive_size < TAR_EOF_BLOCK_SIZE:
        error_message_list.append(
            f'archive total size {total_archive_size} < {TAR_EOF_BLOCK_SIZE}')

        return error_message_list

    # 从最后一片往前读，直到凑满1024字节尾部
    tail_bytes_list, remain_tail_size = [], TAR_EOF_BLOCK_SIZE
    try:
        for per_archive_part_index in range(
                len(per_archive_part_path_list) - 1, -1, -1):
            if remain_tail_size <= 0:
                break

            per_archive_part_path = per_archive_part_path_list[
                per_archive_part_index]
            per_archive_part_size = per_archive_part_size_list[
                per_archive_part_index]
            if per_archive_part_size == 0:
                continue

            per_read_size = min(remain_tail_size, per_archive_part_size)
            with open(per_archive_part_path, 'rb') as load_archive_file:
                load_archive_file.seek(per_archive_part_size - per_read_size)
                tail_bytes_list.append(load_archive_file.read(per_read_size))

            remain_tail_size -= per_read_size
    except Exception as e:
        error_message_list.append(f'read archive tail failed {e}')

        return error_message_list

    tail_bytes = b''.join(reversed(tail_bytes_list))

    if len(tail_bytes) != TAR_EOF_BLOCK_SIZE:
        error_message_list.append(
            f'read archive tail size {len(tail_bytes)} != {TAR_EOF_BLOCK_SIZE}'
        )

        return error_message_list

    if tail_bytes != b'\x00' * TAR_EOF_BLOCK_SIZE:
        error_message_list.append(
            'archive tail eof block not found(truncated archive)')

    return error_message_list


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表、按分片归组后的压缩包列表和标注json列表"""
    file_copy_pair_list, annotation_group_list = [], []
    archive_part_path_dict = {}
    for per_root_path, per_dir_name_list, per_file_name_list in os.walk(
            root_dataset_path):
        # .cache里有359个文件(含*.incomplete)，直接在遍历时剪掉整棵子树，不要走进去
        per_dir_name_list[:] = [
            per_dir_name for per_dir_name in per_dir_name_list
            if per_dir_name not in SKIP_FILE_OR_DIR_NAME_LIST
        ]

        for per_file_name in sorted(per_file_name_list):
            per_file_path = os.path.join(per_root_path, per_file_name)
            per_file_relative_path = os.path.relpath(per_file_path,
                                                     root_dataset_path)
            per_file_relative_dir = os.path.dirname(per_file_relative_path)

            if check_skip_file_or_dir(per_file_relative_path):
                continue

            per_annotation_match_result = ANNOTATION_FILE_NAME_PATTERN.match(
                per_file_name)
            if per_annotation_match_result:
                annotation_group_list.append([
                    per_annotation_match_result.group('prefix'),
                    per_file_relative_dir,
                    per_file_path,
                ])
                continue

            per_archive_group_name, per_archive_part_index = None, ''
            for per_archive_file_name_pattern in ARCHIVE_FILE_NAME_PATTERN_LIST:
                per_match_result = per_archive_file_name_pattern.match(
                    per_file_name)
                if not per_match_result:
                    continue

                per_archive_group_name = per_match_result.group('prefix')
                per_match_group_dict = per_match_result.groupdict()
                if 'part' in per_match_group_dict and per_match_group_dict[
                        'part'] is not None:
                    per_archive_part_index = per_match_group_dict['part']
                break

            if per_archive_group_name is None:
                file_copy_pair_list.append([
                    per_file_relative_path,
                    per_file_path,
                ])
                continue

            per_archive_relative_dir = per_file_relative_dir if per_file_relative_dir else '.'
            per_archive_group_key = f'{per_archive_relative_dir}/{per_archive_group_name}'
            if per_archive_group_key not in archive_part_path_dict:
                archive_part_path_dict[per_archive_group_key] = [
                    per_archive_group_name,
                    per_archive_relative_dir,
                    [],
                ]
            archive_part_path_dict[per_archive_group_key][2].append([
                per_archive_part_index,
                per_file_path,
            ])

    archive_group_list = []
    for per_archive_group_key in sorted(archive_part_path_dict.keys()):
        per_archive_group_name, per_archive_relative_dir, per_archive_part_list = archive_part_path_dict[
            per_archive_group_key]
        per_archive_part_list = sorted(
            per_archive_part_list,
            key=lambda x: get_archive_part_sort_key(x[0]))
        archive_group_list.append([
            per_archive_group_key,
            per_archive_group_name,
            per_archive_relative_dir,
            [
                per_archive_part_index
                for per_archive_part_index, _ in per_archive_part_list
            ],
            [
                per_archive_part_path
                for _, per_archive_part_path in per_archive_part_list
            ],
        ])

    file_copy_pair_list = sorted(file_copy_pair_list, key=lambda x: x[0])
    annotation_group_list = sorted(annotation_group_list, key=lambda x: x[0])

    return file_copy_pair_list, archive_group_list, annotation_group_list


def check_archive_group_complete(archive_group_list):
    """解压前硬对账压缩包组: 组名必须已知、分片数必须相等、编号必须从0连号、整包不能截断"""
    error_message_list = []

    found_archive_group_key_list = []
    archive_group_tail_check_list = []

    for per_archive_group in archive_group_list:
        per_archive_group_key, _, _, per_archive_part_index_list, per_archive_part_path_list = per_archive_group
        found_archive_group_key_list.append(per_archive_group_key)

        if per_archive_group_key not in EXPECTED_ARCHIVE_PART_NUM_DICT:
            # 出现新的压缩包组必须显式上报，否则会被静默漏处理
            error_message_list.append(
                f'unknown archive group {per_archive_group_key}')
            continue

        per_expected_part_num = EXPECTED_ARCHIVE_PART_NUM_DICT[
            per_archive_group_key]
        if len(per_archive_part_path_list) != per_expected_part_num:
            error_message_list.append(
                f'{per_archive_group_key} part num not match '
                f'{len(per_archive_part_path_list)} != {per_expected_part_num}'
            )

        per_digit_part_index_list = sorted([
            int(per_archive_part_index)
            for per_archive_part_index in per_archive_part_index_list
            if per_archive_part_index.isdigit()
        ])
        if len(per_digit_part_index_list) == len(
                per_archive_part_index_list) and len(
                    per_digit_part_index_list) > 0:
            per_missing_part_index_list = sorted(
                set(range(len(per_digit_part_index_list))) -
                set(per_digit_part_index_list))
            if len(per_missing_part_index_list) > 0:
                error_message_list.append(
                    f'{per_archive_group_key} missing part index '
                    f'{per_missing_part_index_list[:10]}')

        archive_group_tail_check_list.append(
            [per_archive_group_key, per_archive_part_path_list])

    for per_expected_archive_group_key in sorted(
            EXPECTED_ARCHIVE_PART_NUM_DICT.keys()):
        if per_expected_archive_group_key not in found_archive_group_key_list:
            error_message_list.append(
                f'missing archive group {per_expected_archive_group_key}')

    print('1111', 'check archive tar tail:',
          len(archive_group_tail_check_list))
    with Pool(processes=PROCESS_NUM) as pool:
        for per_tail_check_result in tqdm(
                pool.imap_unordered(check_single_archive_group_tar_tail,
                                    archive_group_tail_check_list),
                total=len(archive_group_tail_check_list)):
            error_message_list.extend(per_tail_check_result)

    return error_message_list


def check_single_archive_group_tar_tail(archive_group_tail_check_pair):
    """多进程包装: 对单个压缩包组做尾部EOF块预检"""
    per_archive_group_key, per_archive_part_path_list = archive_group_tail_check_pair

    return [
        f'{per_archive_group_key} {per_tail_error_message}'
        for per_tail_error_message in check_single_archive_tar_tail(
            per_archive_part_path_list)
    ]


def check_annotation_file_complete(annotation_group_list):
    """解析前硬对账标注json: 文件数、文件名、每个文件的行数与字段齐备性都必须和实测一致"""
    error_message_list = []

    if len(annotation_group_list) != EXPECTED_ANNOTATION_FILE_NUM:
        error_message_list.append(
            f'annotation file num not match {len(annotation_group_list)} != '
            f'{EXPECTED_ANNOTATION_FILE_NUM}')

    found_annotation_group_name_list = []
    for per_annotation_group in annotation_group_list:
        per_annotation_group_name, _, per_annotation_path = per_annotation_group
        found_annotation_group_name_list.append(per_annotation_group_name)

        if per_annotation_group_name not in ANNOTATION_FILE_CONFIG_DICT:
            error_message_list.append(
                f'unknown annotation file {per_annotation_group_name}')
            continue

        per_expected_row_count = ANNOTATION_FILE_CONFIG_DICT[
            per_annotation_group_name][1]
        try:
            with open(per_annotation_path, 'r',
                      encoding='UTF-8') as load_json_file:
                per_annotation_list = json.load(load_json_file)
        except Exception as e:
            error_message_list.append(
                f'load annotation {per_annotation_group_name} failed {e}')
            continue

        if not isinstance(per_annotation_list, list):
            error_message_list.append(
                f'{per_annotation_group_name} annotation not a list')
            continue

        if len(per_annotation_list) != per_expected_row_count:
            error_message_list.append(
                f'{per_annotation_group_name} row count not match '
                f'{len(per_annotation_list)} != {per_expected_row_count}')

        per_miss_key_row_count = 0
        for per_annotation in per_annotation_list:
            if not isinstance(per_annotation, dict):
                per_miss_key_row_count += 1
                continue

            for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
                if per_key_name not in per_annotation:
                    per_miss_key_row_count += 1
                    break

        if per_miss_key_row_count > 0:
            # 实测131035/131035行4个字段全齐备，出现缺字段说明数据规格变了
            error_message_list.append(
                f'{per_annotation_group_name} miss key row count '
                f'{per_miss_key_row_count}')

    for per_expected_annotation_group_name in sorted(
            ANNOTATION_FILE_CONFIG_DICT.keys()):
        if per_expected_annotation_group_name not in found_annotation_group_name_list:
            error_message_list.append(
                f'missing annotation file {per_expected_annotation_group_name}'
            )

    return error_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非压缩包非标注json文件原样拷贝到目标目录，保持相对路径不变

    该数据集过滤掉无用信息后这里是空列表，保留这一步只是为了兼容后续新增文件。
    """
    per_file_relative_path, per_file_path = file_copy_pair

    save_file_path = os.path.join(save_dataset_path, per_file_relative_path)
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if os.path.isfile(save_file_path) and os.path.getsize(
            save_file_path) == os.path.getsize(per_file_path):
        return [per_file_relative_path, []]

    error_message_list = []
    try:
        with open(per_file_path, 'rb') as load_file:
            with open(save_file_path, 'wb') as save_file:
                shutil.copyfileobj(load_file, save_file, COPY_FILE_BLOCK_SIZE)

        if os.path.getsize(save_file_path) != os.path.getsize(per_file_path):
            error_message_list.append(
                f'copy size not match {per_file_relative_path}')
    except Exception as e:
        error_message_list.append(f'copy failed {per_file_relative_path} {e}')

    return [per_file_relative_path, error_message_list]


def save_single_member_bytes(save_member_path, per_member_bytes,
                             per_member_size):
    """把内存里的成员字节流落盘并立刻校验落盘大小，返回错误信息(空串表示成功)"""
    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_member_path, 'wb') as save_member_file:
            save_member_file.write(per_member_bytes)
    except Exception as e:
        return f'write member failed {save_member_path} {e}'

    if os.path.getsize(save_member_path) != per_member_size:
        return f'write member size not match {save_member_path}'

    return ''


def save_single_member_file(load_member_file, save_member_path,
                            per_member_size):
    """流式把tar成员写盘并立刻校验落盘大小，返回错误信息(空串表示成功)

    单张4K png实测最大40MB，流式拷贝避免一次性读进内存把32个子进程撑爆。
    """
    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_member_path, 'wb') as save_member_file:
            shutil.copyfileobj(load_member_file, save_member_file,
                               EXTRACT_FILE_BLOCK_SIZE)
    except Exception as e:
        return f'write member failed {save_member_path} {e}'

    if os.path.getsize(save_member_path) != per_member_size:
        return f'write member size not match {save_member_path}'

    return ''


def resolve_annotation_image_path(per_image_name, per_subset_name):
    """把标注里记录的图像路径规约成整理后目录下的规范相对路径

    标注里的路径就是"剥掉tar顶层目录后的子集内相对路径"，
    所以统一规约成 images/<子集名>/<子集内相对路径>，和解压落盘路径完全一致，
    后续可以按完整路径精确匹配(x2edit的input/000000001.png这类尾部在8个组之间大面积重名，
    只能按完整路径匹配，否则缺图会被误判成存在)。
    """
    if not isinstance(per_image_name, str):
        return None, f'illegal image path type {type(per_image_name)}'

    per_image_name = per_image_name.replace('\\', '/').strip()
    if not per_image_name:
        return None, 'empty image path'

    per_path_part_list = [
        per_path_part for per_path_part in per_image_name.split('/')
        if per_path_part and per_path_part != '.'
    ]
    if not per_path_part_list:
        return None, f'illegal image path {per_image_name}'

    if '..' in per_path_part_list:
        return None, f'illegal image path {per_image_name}'

    if not check_image_file_suffix(per_path_part_list[-1]):
        return None, f'image path suffix not image {per_image_name}'

    return f'{SAVE_IMAGE_ROOT_NAME}/{per_subset_name}/' + '/'.join(
        per_path_part_list), ''


def get_single_sample_pair(per_annotation, per_subset_name):
    """把一行标注解析成一个样本对: 1张参考图 + 1张编辑后图 + 1条编辑指令 + 1个编辑类型

    返回[样本对, 不完整原因列表]。原因列表非空的样本对被隔离进invalid_sample_pair_list，
    既不写进训练标注、也不静默丢弃，全部在校验报告里上报。
    """
    reason_list = []

    per_instruction = per_annotation.get(ANNOTATION_TEXT_KEY_NAME, None)
    per_instruction = per_instruction.strip() if isinstance(
        per_instruction, str) else ''
    if not per_instruction:
        # 图像编辑样本对必须有编辑指令，实测nano-consistent有25行是空串
        reason_list.append('empty instruction')

    per_edit_type = per_annotation.get(ANNOTATION_EDIT_TYPE_KEY_NAME, None)
    per_edit_type = per_edit_type.strip() if isinstance(per_edit_type,
                                                        str) else ''
    if not per_edit_type:
        # 实测ultravideo有1行edit_type是null，属性残缺
        reason_list.append('empty edit type')
    elif per_edit_type == NO_CHANGE_EDIT_TYPE_NAME:
        # 实测ultravideo有132行是"无变化"对照样本，不是真正的编辑对
        reason_list.append('no change edit type')

    per_reference_image_path, per_reference_reason = resolve_annotation_image_path(
        per_annotation.get(ANNOTATION_REFERENCE_IMAGE_KEY_NAME, None),
        per_subset_name)
    if per_reference_image_path is None:
        reason_list.append(f'reference image {per_reference_reason}')

    per_target_image_path, per_target_reason = resolve_annotation_image_path(
        per_annotation.get(ANNOTATION_TARGET_IMAGE_KEY_NAME, None),
        per_subset_name)
    if per_target_image_path is None:
        reason_list.append(f'target image {per_target_reason}')

    if per_reference_image_path is not None and per_target_image_path is not None and per_reference_image_path == per_target_image_path:
        # 实测0行，参考图和编辑后图同一张就不是编辑对
        reason_list.append('reference image same as target image')

    per_reference_image_path_list = [
        per_reference_image_path
    ] if per_reference_image_path is not None else []

    per_sample_pair = {
        'edit_type':
        per_edit_type,
        'instruction':
        per_instruction,
        'reference_image_path_list':
        per_reference_image_path_list,
        'reference_image_num':
        len(per_reference_image_path_list),
        'target_image_path':
        per_target_image_path if per_target_image_path is not None else '',
    }

    return per_sample_pair, reason_list


def get_sample_archive_group_name(per_annotation, per_annotation_group_name):
    """从标注里的参考图路径第一段推出该样本对属于哪个压缩包组

    nano-consistent只有一个组(Image)；ultravideo的第一段就是clips_short_N；
    x2edit的第一段就是0..7。带上这个字段，下游可以按压缩包组做分片读取或抽样。
    """
    per_image_name = per_annotation.get(ANNOTATION_REFERENCE_IMAGE_KEY_NAME,
                                        None)
    if not isinstance(per_image_name, str):
        return ''

    per_path_part_list = [
        per_path_part
        for per_path_part in per_image_name.replace('\\', '/').split('/')
        if per_path_part and per_path_part != '.'
    ]
    if not per_path_part_list:
        return ''

    return f'{per_annotation_group_name}/{per_path_part_list[0]}'


def process_single_annotation_file(annotation_group, save_dataset_path):
    """把单个标注json解成annotations/<name>.jsonl，每行是一个完整样本对

    保留原json的全部属性(edit_type/instruction)，再补上落盘后的规范图像路径、
    样本id、所属子集与压缩包组，下游可以直接按行取样本，不用再去关联原json。
    先写.tmp再os.replace，避免中途失败留下半截jsonl被下游当成完整索引用。
    """
    per_annotation_group_name, _, per_annotation_path = annotation_group

    per_subset_name, per_expected_row_count, per_expected_valid_sample_pair_count, per_expected_invalid_sample_pair_count, per_expected_valid_unique_image_count = ANNOTATION_FILE_CONFIG_DICT[
        per_annotation_group_name]

    save_annotation_path = os.path.join(save_dataset_path,
                                        SAVE_ANNOTATION_ROOT_NAME,
                                        f'{per_annotation_group_name}.jsonl')
    os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
    save_temp_annotation_path = f'{save_annotation_path}.tmp'

    row_count, valid_sample_pair_count = 0, 0
    single_reference_sample_pair_count = 0
    valid_unique_image_path_set, all_unique_image_path_set = set(), set()
    edit_type_count_dict = collections.Counter()
    archive_group_sample_pair_count_dict = collections.Counter()
    invalid_sample_pair_list = []
    error_message_list = []

    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_json_file:
            per_annotation_list = json.load(load_json_file)
    except Exception as e:
        return [
            per_annotation_group_name,
            0,
            0,
            0,
            set(),
            set(),
            {},
            {},
            [],
            [f'load annotation failed {e}'],
        ]

    try:
        with open(save_temp_annotation_path, 'w',
                  encoding='UTF-8') as save_annotation_file:
            for per_row_index, per_annotation in enumerate(
                    per_annotation_list):
                row_count += 1

                if not isinstance(per_annotation, dict):
                    error_message_list.append(
                        f'row {per_row_index} annotation not a dict')
                    continue

                per_sample_id = f'{per_annotation_group_name}_{per_row_index:08d}'
                per_archive_group_name = get_sample_archive_group_name(
                    per_annotation, per_annotation_group_name)

                per_sample_pair, per_reason_list = get_single_sample_pair(
                    per_annotation, per_subset_name)

                per_image_path_list = list(
                    per_sample_pair['reference_image_path_list'])
                if per_sample_pair['target_image_path']:
                    per_image_path_list.append(
                        per_sample_pair['target_image_path'])
                all_unique_image_path_set.update(per_image_path_list)

                if len(per_reason_list) > 0:
                    # 不完整样本对全部隔离上报，不写进训练标注也不静默丢
                    if len(invalid_sample_pair_list
                           ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                        invalid_sample_pair_list.append({
                            'annotation_name':
                            per_annotation_group_name,
                            'sample_id':
                            per_sample_id,
                            'row_index':
                            per_row_index,
                            'edit_type':
                            per_annotation.get(ANNOTATION_EDIT_TYPE_KEY_NAME,
                                               None),
                            'reason_list':
                            per_reason_list,
                        })
                    continue

                per_save_annotation = {
                    'sample_id': per_sample_id,
                    'subset_name': per_subset_name,
                    'annotation_name': per_annotation_group_name,
                    'archive_group_name': per_archive_group_name,
                    'task_type': 'image_edit',
                    'row_index': per_row_index,
                }
                per_save_annotation.update(per_sample_pair)

                save_annotation_file.write(
                    f'{json.dumps(per_save_annotation, ensure_ascii=False)}\n')

                valid_sample_pair_count += 1
                valid_unique_image_path_set.update(per_image_path_list)
                edit_type_count_dict[per_sample_pair['edit_type']] += 1
                archive_group_sample_pair_count_dict[
                    per_archive_group_name] += 1

                if per_sample_pair[
                        'reference_image_num'] == EXPECTED_REFERENCE_IMAGE_NUM_PER_SAMPLE_PAIR:
                    single_reference_sample_pair_count += 1

        os.replace(save_temp_annotation_path, save_annotation_path)
    except Exception as e:
        error_message_list.append(f'parse annotation failed {e}')
        if os.path.exists(save_temp_annotation_path):
            os.remove(save_temp_annotation_path)

    # 单文件硬对账: 行数、有效样本对数、隔离样本对数、引用唯一图数全部按实测值卡死
    if row_count != per_expected_row_count:
        error_message_list.append(
            f'row count not match {row_count} != {per_expected_row_count}')

    if valid_sample_pair_count != per_expected_valid_sample_pair_count:
        error_message_list.append(
            f'valid sample pair count not match {valid_sample_pair_count} != '
            f'{per_expected_valid_sample_pair_count}')

    if row_count - valid_sample_pair_count != per_expected_invalid_sample_pair_count:
        error_message_list.append(f'invalid sample pair count not match '
                                  f'{row_count - valid_sample_pair_count} != '
                                  f'{per_expected_invalid_sample_pair_count}')

    if len(valid_unique_image_path_set
           ) != per_expected_valid_unique_image_count:
        error_message_list.append(f'valid unique image count not match '
                                  f'{len(valid_unique_image_path_set)} != '
                                  f'{per_expected_valid_unique_image_count}')

    if single_reference_sample_pair_count != valid_sample_pair_count:
        # 该数据集每个样本对固定只有1张参考图，不等说明解析错位
        error_message_list.append(
            f'single reference sample pair count not match '
            f'{single_reference_sample_pair_count} != {valid_sample_pair_count}'
        )

    return [
        per_annotation_group_name,
        row_count,
        valid_sample_pair_count,
        single_reference_sample_pair_count,
        valid_unique_image_path_set,
        all_unique_image_path_set,
        dict(edit_type_count_dict),
        dict(archive_group_sample_pair_count_dict),
        invalid_sample_pair_list,
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    ]


def process_single_benchmark_instruction(per_instruction_bytes,
                                         save_dataset_path):
    """把benchmark.tar里的instructions.json解成annotations/benchmark.jsonl

    VINS-4KEval评测集一张图配多条不同编辑类型的指令(509行/100张图)，只有参考图没有
    编辑后图(要靠模型生成)，所以单独一份jsonl，不参与训练样本对统计。
    """
    row_count = 0
    unique_image_path_set = set()
    edit_type_count_dict = collections.Counter()
    error_message_list = []

    save_annotation_path = os.path.join(save_dataset_path,
                                        SAVE_ANNOTATION_ROOT_NAME,
                                        f'{BENCHMARK_SUBSET_NAME}.jsonl')
    save_temp_annotation_path = f'{save_annotation_path}.tmp'

    try:
        benchmark_annotation_list = json.loads(
            per_instruction_bytes.decode('UTF-8'))
    except Exception as e:
        return [0, set(), {}, [f'load benchmark instruction failed {e}']]

    if not isinstance(benchmark_annotation_list, list):
        return [0, set(), {}, ['benchmark instruction not a list']]

    try:
        os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
        with open(save_temp_annotation_path, 'w',
                  encoding='UTF-8') as save_annotation_file:
            for per_row_index, per_annotation in enumerate(
                    benchmark_annotation_list):
                row_count += 1

                if not isinstance(per_annotation, dict):
                    error_message_list.append(
                        f'benchmark row {per_row_index} not a dict')
                    continue

                for per_key_name in BENCHMARK_EXPECTED_KEY_NAME_LIST:
                    if per_key_name not in per_annotation:
                        error_message_list.append(
                            f'benchmark row {per_row_index} miss key {per_key_name}'
                        )

                per_instruction = per_annotation.get(ANNOTATION_TEXT_KEY_NAME,
                                                     None)
                per_instruction = per_instruction.strip() if isinstance(
                    per_instruction, str) else ''
                if not per_instruction:
                    error_message_list.append(
                        f'benchmark row {per_row_index} empty instruction')

                per_image_name = per_annotation.get(BENCHMARK_IMAGE_KEY_NAME,
                                                    None)
                if not isinstance(
                        per_image_name,
                        str) or not check_image_file_suffix(per_image_name):
                    error_message_list.append(
                        f'benchmark row {per_row_index} illegal image id {per_image_name}'
                    )
                    continue

                per_image_name = per_image_name.replace('\\', '/').strip()
                per_reference_image_path = (
                    f'{SAVE_BENCHMARK_ROOT_NAME}/{BENCHMARK_IMAGE_DIR_NAME}/'
                    f'{per_image_name}')
                unique_image_path_set.add(per_reference_image_path)

                per_save_annotation = {
                    'sample_id':
                    f'{BENCHMARK_SUBSET_NAME}_{per_row_index:08d}',
                    'subset_name': BENCHMARK_SUBSET_NAME,
                    'annotation_name': BENCHMARK_SUBSET_NAME,
                    'archive_group_name': BENCHMARK_ARCHIVE_GROUP_KEY,
                    'task_type': 'image_edit_benchmark',
                    'row_index': per_row_index,
                    'edit_type': per_annotation.get('edit_type', ''),
                    'edit_type_index': per_annotation.get('index', -1),
                    'instruction': per_instruction,
                    'reference_image_path_list': [per_reference_image_path],
                    'reference_image_num': 1,
                    'target_image_path': '',
                }
                save_annotation_file.write(
                    f'{json.dumps(per_save_annotation, ensure_ascii=False)}\n')

                edit_type_count_dict[str(per_annotation.get('edit_type',
                                                            None))] += 1

        os.replace(save_temp_annotation_path, save_annotation_path)
    except Exception as e:
        error_message_list.append(f'save benchmark annotation failed {e}')
        if os.path.exists(save_temp_annotation_path):
            os.remove(save_temp_annotation_path)

    if row_count != EXPECTED_BENCHMARK_ROW_COUNT:
        error_message_list.append(
            f'benchmark row count not match {row_count} != '
            f'{EXPECTED_BENCHMARK_ROW_COUNT}')

    if len(unique_image_path_set) != EXPECTED_BENCHMARK_IMAGE_COUNT:
        error_message_list.append(
            f'benchmark unique image count not match '
            f'{len(unique_image_path_set)} != {EXPECTED_BENCHMARK_IMAGE_COUNT}'
        )

    return [
        row_count,
        unique_image_path_set,
        dict(edit_type_count_dict),
        error_message_list,
    ]


def process_single_archive_group(archive_group, save_dataset_path):
    """流式解压单个压缩包组，剥掉压缩包自带顶层目录后改挂到规范子集名下

    训练子集落盘为 images/<子集名>/<标注相对路径>，和标注里的路径一一对应；
    benchmark落盘为 benchmark/img/<...>，并把instructions.json顺手解成
    annotations/benchmark.jsonl(它在流式解压时正好在手上，几十KB，代价可忽略)。

    每个成员写盘后立刻校验落盘大小 == tar头里的大小(避免解压完再os.walk 21万个大文件),
    并对账 extract + skip == tar里的文件成员总数，且必须正常读到tar流结束。
    """
    per_archive_group_key, per_archive_group_name, _, _, per_archive_part_path_list = archive_group

    is_benchmark = per_archive_group_key == BENCHMARK_ARCHIVE_GROUP_KEY
    if is_benchmark:
        per_tar_top_strip_prefix = BENCHMARK_TAR_TOP_STRIP_PREFIX
        per_subset_name = BENCHMARK_SUBSET_NAME
        per_subset_inner_prefix = BENCHMARK_IMAGE_DIR_NAME
        save_member_root_path = os.path.join(save_dataset_path,
                                             SAVE_BENCHMARK_ROOT_NAME)
    else:
        per_tar_top_strip_prefix, per_subset_name, per_subset_inner_prefix = ARCHIVE_GROUP_CONFIG_DICT[
            per_archive_group_key]
        save_member_root_path = os.path.join(save_dataset_path,
                                             SAVE_IMAGE_ROOT_NAME,
                                             per_subset_name)

    os.makedirs(save_member_root_path, exist_ok=True)

    total_member_count, total_file_member_count = 0, 0
    extract_file_count, skip_file_count = 0, 0
    image_file_member_count, other_file_member_count = 0, 0
    benchmark_row_count = 0
    benchmark_image_path_set = set()
    benchmark_edit_type_count_dict = {}
    benchmark_instruction_bytes = None
    reach_tar_end = False
    error_message_list = []

    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    try:
        with tarfile.open(fileobj=archive_reader, mode='r|*') as load_tar_file:
            for per_member in load_tar_file:
                total_member_count += 1

                per_member_name, is_archive_root = get_normalized_member_name(
                    per_member.name)
                if not per_member_name:
                    # 压缩包根成员(./)直接跳过，真正越界的成员才记error
                    if not is_archive_root:
                        print('5555', per_archive_group_key, per_member.name)
                        error_message_list.append(
                            f'illegal member name {per_member.name}')
                    continue

                per_relative_path = strip_member_top_prefix(
                    per_member_name, per_tar_top_strip_prefix)
                if per_relative_path is None:
                    # 顶层目录名和实测不一致，说明打包规格变了，必须显式感知
                    print('5555', per_archive_group_key, per_member.name)
                    error_message_list.append(
                        f'member top prefix not match {per_member.name}')
                    continue

                if not per_relative_path:
                    continue

                if check_skip_file_or_dir(per_relative_path):
                    continue

                if per_member.isdir():
                    continue

                if not per_member.isfile():
                    # 实测只有普通文件和目录，出现链接等类型必须显式上报
                    print('5555', per_archive_group_key, per_member.name,
                          'not a regular file')
                    error_message_list.append(
                        f'not a regular file member {per_member.name}')
                    continue

                total_file_member_count += 1

                per_first_path_name = per_relative_path.split('/')[0]
                per_file_name = os.path.basename(per_relative_path)

                if is_benchmark and per_file_name == BENCHMARK_INSTRUCTION_FILE_NAME:
                    # 评测集自带的指令文件: 读进内存生成汇总标注，同时原样落盘保留
                    load_member_file = load_tar_file.extractfile(per_member)
                    if load_member_file is None:
                        print('6666', per_archive_group_key, per_member.name)
                        error_message_list.append(
                            f'extract member failed {per_member.name}')
                        continue

                    per_member_bytes = load_member_file.read()
                    if len(per_member_bytes) != per_member.size:
                        error_message_list.append(
                            f'member data truncated {per_relative_path} '
                            f'{len(per_member_bytes)} != {per_member.size}')
                        continue

                    benchmark_instruction_bytes = per_member_bytes
                    other_file_member_count += 1

                    save_member_path = os.path.join(save_member_root_path,
                                                    per_relative_path)
                    if os.path.isfile(save_member_path) and os.path.getsize(
                            save_member_path) == per_member.size:
                        skip_file_count += 1
                        continue

                    per_save_error_message = save_single_member_bytes(
                        save_member_path, per_member_bytes, per_member.size)
                    if per_save_error_message:
                        print('6666', per_archive_group_key,
                              per_save_error_message)
                        error_message_list.append(per_save_error_message)
                        continue

                    extract_file_count += 1
                    continue

                if per_first_path_name != per_subset_inner_prefix:
                    # 成员跨组错位(比如clips_short_1.tar里出现clips_short_2的图)必须上报
                    error_message_list.append(
                        f'member inner prefix not match {per_member.name}')
                    continue

                if not check_image_file_suffix(per_file_name):
                    # 实测三个训练子集的tar里只有png，出现其他文件必须上报，不能默认当图像
                    other_file_member_count += 1
                    error_message_list.append(
                        f'unknown suffix member {per_relative_path}')
                    continue

                image_file_member_count += 1

                save_member_path = os.path.join(save_member_root_path,
                                                per_relative_path)

                if os.path.isfile(save_member_path) and os.path.getsize(
                        save_member_path) == per_member.size:
                    skip_file_count += 1
                    continue

                load_member_file = load_tar_file.extractfile(per_member)
                if load_member_file is None:
                    print('6666', per_archive_group_key, per_member.name)
                    error_message_list.append(
                        f'extract member failed {per_member.name}')
                    continue

                per_save_error_message = save_single_member_file(
                    load_member_file, save_member_path, per_member.size)
                if per_save_error_message:
                    print('6666', per_archive_group_key,
                          per_save_error_message)
                    error_message_list.append(per_save_error_message)
                    continue

                extract_file_count += 1

        reach_tar_end = True
    except Exception as e:
        # 分片不全或压缩包截断时保留已解压出的文件，但必须上报，不能静默少样本对
        print('7777', per_archive_group_key, len(per_archive_part_path_list),
              e)
        error_message_list.append(f'read archive failed {e}')
    finally:
        archive_reader.close()

    if not reach_tar_end:
        error_message_list.append(
            'not reach tar stream end, archive may be truncated')

    if extract_file_count + skip_file_count != total_file_member_count:
        error_message_list.append(
            f'process file count not match: {extract_file_count} + '
            f'{skip_file_count} != {total_file_member_count}')

    per_expected_image_num = EXPECTED_ARCHIVE_GROUP_IMAGE_NUM_DICT.get(
        per_archive_group_key, None)
    if per_expected_image_num is None:
        error_message_list.append('unknown expected image num')
    elif image_file_member_count < per_expected_image_num:
        # 少于标注引用的唯一图数就是丢样本对，硬失败
        error_message_list.append(
            f'image member count less than expected {image_file_member_count} '
            f'< {per_expected_image_num}')

    if is_benchmark:
        if benchmark_instruction_bytes is None:
            error_message_list.append(
                f'missing benchmark {BENCHMARK_INSTRUCTION_FILE_NAME}')
        else:
            benchmark_row_count, benchmark_image_path_set, benchmark_edit_type_count_dict, per_benchmark_error_message_list = process_single_benchmark_instruction(
                benchmark_instruction_bytes, save_dataset_path)
            error_message_list.extend(per_benchmark_error_message_list)

    return {
        'archive_group_key': per_archive_group_key,
        'archive_group_name': per_archive_group_name,
        'subset_name': per_subset_name,
        'total_member_count': total_member_count,
        'total_file_member_count': total_file_member_count,
        'extract_file_count': extract_file_count,
        'skip_file_count': skip_file_count,
        'image_file_member_count': image_file_member_count,
        'other_file_member_count': other_file_member_count,
        'expected_image_member_count': per_expected_image_num,
        'benchmark_row_count': benchmark_row_count,
        'benchmark_image_count': len(benchmark_image_path_set),
        'benchmark_edit_type_count_dict': benchmark_edit_type_count_dict,
        'error_message_list': error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }


def collect_single_subset_image_index(subset_collect_pair):
    """把一个子集目录下所有已落盘图像的相对路径写成一个索引文件

    21万张图全塞一个set再喂给32个子进程会重复占内存，所以按子集分片建索引，
    校验时每个标注文件只加载它自己那个子集的索引。
    索引里存的是相对save_dataset_path的完整规范路径，和jsonl里的路径同一套口径。
    """
    per_subset_index_name, per_subset_path, per_subset_path_prefix, save_index_path = subset_collect_pair

    image_relative_path_list = []
    error_message_list = []
    try:
        for per_root_path, _, per_file_name_list in os.walk(per_subset_path):
            for per_file_name in per_file_name_list:
                if not check_image_file_suffix(per_file_name):
                    continue

                per_file_path = os.path.join(per_root_path, per_file_name)
                per_file_relative_path = os.path.relpath(
                    per_file_path, per_subset_path).replace('\\', '/')
                image_relative_path_list.append(
                    f'{per_subset_path_prefix}/{per_file_relative_path}')

        os.makedirs(os.path.dirname(save_index_path), exist_ok=True)
        with open(save_index_path, 'w', encoding='UTF-8') as save_index_file:
            for per_image_relative_path in sorted(image_relative_path_list):
                save_index_file.write(f'{per_image_relative_path}\n')
    except Exception as e:
        error_message_list.append(
            f'collect subset {per_subset_index_name} failed {e}')

    return [
        per_subset_index_name,
        len(image_relative_path_list),
        error_message_list,
    ]


def check_single_annotation_file(annotation_check_pair):
    """逐条回读一个annotations/<name>.jsonl，校验每个样本对引用的图像是否真的落盘

    按**完整规范路径**精确匹配。x2edit的 <组>/<样本目录>/input/000000001.png 这种路径，
    尾部两三段在8个组之间大面积重名，按尾部模糊匹配会把缺图判成存在，校验形同虚设。
    """
    per_annotation_group_name, per_annotation_path, per_image_index_path = annotation_check_pair

    exist_image_path_set = set()
    error_message_list = []
    try:
        if os.path.isfile(per_image_index_path):
            with open(per_image_index_path, 'r',
                      encoding='UTF-8') as load_index_file:
                for per_line in load_index_file:
                    per_line = per_line.strip()
                    if per_line:
                        exist_image_path_set.add(per_line)
        else:
            error_message_list.append(
                f'{per_annotation_group_name} image index not exist '
                f'{per_image_index_path}')
    except Exception as e:
        error_message_list.append(
            f'{per_annotation_group_name} load image index failed {e}')

    checked_sample_pair_count, valid_sample_pair_count = 0, 0
    used_image_path_set, missing_image_path_set = set(), set()
    missing_sample_pair_list = []

    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_annotation_file:
            for per_line in load_annotation_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                per_annotation = json.loads(per_line)
                checked_sample_pair_count += 1

                per_image_path_list = list(
                    per_annotation.get('reference_image_path_list', []))
                if per_annotation.get('target_image_path', ''):
                    per_image_path_list.append(
                        per_annotation['target_image_path'])

                used_image_path_set.update(per_image_path_list)

                per_missing_image_path_list = [
                    per_image_path for per_image_path in per_image_path_list
                    if per_image_path not in exist_image_path_set
                ]

                if len(per_missing_image_path_list) > 0:
                    missing_image_path_set.update(per_missing_image_path_list)
                    if len(missing_sample_pair_list
                           ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                        missing_sample_pair_list.append({
                            'annotation_name':
                            per_annotation_group_name,
                            'sample_id':
                            per_annotation.get('sample_id', ''),
                            'missing_image_path_list':
                            per_missing_image_path_list,
                        })
                    continue

                valid_sample_pair_count += 1
    except Exception as e:
        error_message_list.append(
            f'check annotation {per_annotation_group_name} failed {e}')

    return [
        per_annotation_group_name,
        checked_sample_pair_count,
        valid_sample_pair_count,
        used_image_path_set,
        missing_image_path_set,
        missing_sample_pair_list,
        error_message_list,
    ]


def get_subset_collect_pair_list(save_dataset_path):
    """收集需要建落盘图像索引的子集: images/下的每个训练子集 + benchmark/"""
    subset_collect_pair_list = []

    root_image_path = os.path.join(save_dataset_path, SAVE_IMAGE_ROOT_NAME)
    if os.path.isdir(root_image_path):
        for per_subset_name in sorted(os.listdir(root_image_path)):
            per_subset_path = os.path.join(root_image_path, per_subset_name)
            if not os.path.isdir(per_subset_path):
                continue

            subset_collect_pair_list.append([
                per_subset_name,
                per_subset_path,
                f'{SAVE_IMAGE_ROOT_NAME}/{per_subset_name}',
                os.path.join(save_dataset_path, SAVE_IMAGE_INDEX_ROOT_NAME,
                             f'{per_subset_name}.txt'),
            ])

    root_benchmark_path = os.path.join(save_dataset_path,
                                       SAVE_BENCHMARK_ROOT_NAME)
    if os.path.isdir(root_benchmark_path):
        subset_collect_pair_list.append([
            BENCHMARK_SUBSET_NAME,
            root_benchmark_path,
            SAVE_BENCHMARK_ROOT_NAME,
            os.path.join(save_dataset_path, SAVE_IMAGE_INDEX_ROOT_NAME,
                         f'{BENCHMARK_SUBSET_NAME}.txt'),
        ])

    return subset_collect_pair_list


def check_image_annotation_pair(save_dataset_path, process_summary_dict):
    """按样本对逐条校验: annotations里的每个样本对引用的图像是否都真的落盘

    校验口径全部按**唯一规范路径集合**统计，保证
    matched + missing == 标注引用的唯一图像数 且非负。
    """
    check_error_message_list, check_warning_message_list = [], []

    root_annotation_path = os.path.join(save_dataset_path,
                                        SAVE_ANNOTATION_ROOT_NAME)
    if not os.path.isdir(root_annotation_path):
        check_error_message_list.append(
            f'missing annotation dir {root_annotation_path}')

        return check_error_message_list, check_warning_message_list

    # 第一步: 按子集并行建落盘图像索引(一次os.walk，避免几十万次stat)
    subset_collect_pair_list = get_subset_collect_pair_list(save_dataset_path)
    if len(subset_collect_pair_list) == 0:
        check_error_message_list.append('no unzip image subset dir found')

        return check_error_message_list, check_warning_message_list

    total_unzip_image_count = 0
    subset_unzip_image_count_dict = {}
    with Pool(processes=PROCESS_NUM) as pool:
        for per_collect_result in tqdm(pool.imap_unordered(
                collect_single_subset_image_index, subset_collect_pair_list),
                                       total=len(subset_collect_pair_list)):
            per_subset_index_name, per_image_count, per_error_message_list = per_collect_result
            subset_unzip_image_count_dict[
                per_subset_index_name] = per_image_count
            if per_subset_index_name != BENCHMARK_SUBSET_NAME:
                total_unzip_image_count += per_image_count
            check_error_message_list.extend(per_error_message_list)

            print('2222', 'subset', per_subset_index_name, 'unzip image',
                  per_image_count)

    # 第二步: 按标注文件并行逐条校验样本对(benchmark单独统计，不混进训练口径)
    annotation_check_pair_list, benchmark_check_pair_list = [], []
    for per_annotation_file_name in sorted(os.listdir(root_annotation_path)):
        if not per_annotation_file_name.endswith('.jsonl'):
            continue

        per_annotation_group_name = per_annotation_file_name[:-len('.jsonl')]
        if per_annotation_group_name == BENCHMARK_SUBSET_NAME:
            per_subset_index_name = BENCHMARK_SUBSET_NAME
        elif per_annotation_group_name in ANNOTATION_FILE_CONFIG_DICT:
            per_subset_index_name = ANNOTATION_FILE_CONFIG_DICT[
                per_annotation_group_name][0]
        else:
            check_error_message_list.append(
                f'unknown annotation jsonl {per_annotation_file_name}')
            continue

        per_annotation_check_pair = [
            per_annotation_group_name,
            os.path.join(root_annotation_path, per_annotation_file_name),
            os.path.join(save_dataset_path, SAVE_IMAGE_INDEX_ROOT_NAME,
                         f'{per_subset_index_name}.txt'),
        ]
        if per_annotation_group_name == BENCHMARK_SUBSET_NAME:
            benchmark_check_pair_list.append(per_annotation_check_pair)
        else:
            annotation_check_pair_list.append(per_annotation_check_pair)

    if len(annotation_check_pair_list) != EXPECTED_ANNOTATION_FILE_NUM:
        check_error_message_list.append(
            f'annotation jsonl num not match {len(annotation_check_pair_list)} '
            f'!= {EXPECTED_ANNOTATION_FILE_NUM}')

    total_checked_sample_pair_count, total_valid_sample_pair_count = 0, 0
    all_used_image_path_set, all_missing_image_path_set = set(), set()
    all_missing_sample_pair_list = []
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_annotation_file,
                annotation_check_pair_list + benchmark_check_pair_list),
                                     total=len(annotation_check_pair_list) +
                                     len(benchmark_check_pair_list)):
            per_annotation_group_name, per_checked_count, per_valid_count, per_used_image_path_set, per_missing_image_path_set, per_missing_sample_pair_list, per_error_message_list = per_check_result

            all_used_image_path_set.update(per_used_image_path_set)
            all_missing_image_path_set.update(per_missing_image_path_set)
            all_missing_sample_pair_list.extend(
                per_missing_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM])
            check_error_message_list.extend(per_error_message_list)

            if per_annotation_group_name != BENCHMARK_SUBSET_NAME:
                total_checked_sample_pair_count += per_checked_count
                total_valid_sample_pair_count += per_valid_count
            else:
                if per_checked_count != EXPECTED_BENCHMARK_ROW_COUNT:
                    check_error_message_list.append(
                        f'benchmark checked row count not match '
                        f'{per_checked_count} != {EXPECTED_BENCHMARK_ROW_COUNT}'
                    )
                if per_valid_count != per_checked_count:
                    check_error_message_list.append(
                        f'benchmark valid row count not match '
                        f'{per_valid_count} != {per_checked_count}')

            print('2222', per_annotation_group_name, 'checked sample pair',
                  per_checked_count, 'valid sample pair', per_valid_count,
                  'missing image', len(per_missing_image_path_set))

    # 第三步: 统计orphan(解压出来但没被任何有效样本对引用的图)
    total_orphan_image_count = 0
    orphan_image_path_list = []
    for per_subset_collect_pair in subset_collect_pair_list:
        per_subset_index_name, _, _, per_index_path = per_subset_collect_pair
        if per_subset_index_name == BENCHMARK_SUBSET_NAME:
            continue

        if not os.path.isfile(per_index_path):
            continue

        with open(per_index_path, 'r', encoding='UTF-8') as load_index_file:
            for per_line in load_index_file:
                per_line = per_line.strip()
                if not per_line:
                    continue
                if per_line in all_used_image_path_set:
                    continue

                total_orphan_image_count += 1
                if len(orphan_image_path_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                    orphan_image_path_list.append(per_line)

    total_annotation_valid_unique_image_count = len(
        process_summary_dict.get('all_valid_unique_image_path_set', set()))
    total_missing_image_count = len(all_missing_image_path_set)
    total_matched_image_count = len(
        all_used_image_path_set) - total_missing_image_count

    print('3333', 'checked sample pair', total_checked_sample_pair_count,
          'valid sample pair', total_valid_sample_pair_count,
          'annotation valid unique image',
          total_annotation_valid_unique_image_count, 'unzip image',
          total_unzip_image_count, 'matched image', total_matched_image_count,
          'missing image', total_missing_image_count, 'orphan image',
          total_orphan_image_count)

    # 第四步: 全量期望值硬对账
    total_annotation_row_count = process_summary_dict.get(
        'total_annotation_row_count', 0)
    total_sample_pair_count = process_summary_dict.get(
        'total_valid_sample_pair_count', 0)
    total_invalid_sample_pair_count = process_summary_dict.get(
        'total_invalid_sample_pair_count', 0)

    # 写出的样本对 + 被隔离的不完整样本对 必须等于标注json里的总行数，
    # 否则说明有样本对在"解析->写jsonl->回读校验"这条链路上凭空消失了
    if total_checked_sample_pair_count + total_invalid_sample_pair_count != total_annotation_row_count:
        check_error_message_list.append(
            f'checked + invalid sample pair count not match '
            f'{total_checked_sample_pair_count} + '
            f'{total_invalid_sample_pair_count} != {total_annotation_row_count}'
        )

    if total_checked_sample_pair_count != total_sample_pair_count:
        check_error_message_list.append(
            f'checked sample pair count not match '
            f'{total_checked_sample_pair_count} != {total_sample_pair_count}')

    if total_valid_sample_pair_count != total_checked_sample_pair_count:
        check_error_message_list.append(f'valid sample pair count not match '
                                        f'{total_valid_sample_pair_count} != '
                                        f'{total_checked_sample_pair_count}')

    if total_missing_image_count > 0:
        check_error_message_list.append(
            f'missing image count {total_missing_image_count}')

    if total_unzip_image_count < EXPECTED_TOTAL_UNZIP_IMAGE_COUNT:
        # 落盘图像数少于标注引用的全部唯一图数就是丢样本，硬失败
        check_error_message_list.append(
            f'total unzip image count less than expected '
            f'{total_unzip_image_count} < {EXPECTED_TOTAL_UNZIP_IMAGE_COUNT}')
    elif total_unzip_image_count > EXPECTED_TOTAL_UNZIP_IMAGE_COUNT:
        check_warning_message_list.append(
            f'total unzip image count more than expected '
            f'{total_unzip_image_count} > {EXPECTED_TOTAL_UNZIP_IMAGE_COUNT}')

    if subset_unzip_image_count_dict.get(BENCHMARK_SUBSET_NAME,
                                         0) != EXPECTED_BENCHMARK_IMAGE_COUNT:
        check_error_message_list.append(
            f'benchmark unzip image count not match '
            f'{subset_unzip_image_count_dict.get(BENCHMARK_SUBSET_NAME, 0)} != '
            f'{EXPECTED_BENCHMARK_IMAGE_COUNT}')

    if total_orphan_image_count != EXPECTED_TOTAL_ORPHAN_IMAGE_COUNT:
        # orphan只是"只被隔离行引用、没被任何有效样本对用到"的图，不影响样本对完整性，
        # 数量和实测不一致只告警，方便感知数据规格变化
        check_warning_message_list.append(
            f'orphan image count {total_orphan_image_count} != '
            f'{EXPECTED_TOTAL_ORPHAN_IMAGE_COUNT}')

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        'image_edit',
        'total_annotation_row_count':
        total_annotation_row_count,
        'total_valid_sample_pair_count':
        total_sample_pair_count,
        'total_invalid_sample_pair_count':
        total_invalid_sample_pair_count,
        'total_single_reference_sample_pair_count':
        process_summary_dict.get('total_single_reference_sample_pair_count',
                                 0),
        'total_annotation_valid_unique_image_count':
        total_annotation_valid_unique_image_count,
        'total_annotation_all_unique_image_count':
        len(process_summary_dict.get('all_unique_image_path_set', set())),
        'total_unzip_image_count':
        total_unzip_image_count,
        'subset_unzip_image_count_dict':
        subset_unzip_image_count_dict,
        'matched_image_count':
        total_matched_image_count,
        'missing_image_count':
        total_missing_image_count,
        'orphan_image_count':
        total_orphan_image_count,
        'checked_sample_pair_count':
        total_checked_sample_pair_count,
        'valid_sample_pair_count':
        total_valid_sample_pair_count,
        'annotation_row_count_dict':
        process_summary_dict.get('annotation_row_count_dict', {}),
        'annotation_valid_sample_pair_count_dict':
        process_summary_dict.get('annotation_valid_sample_pair_count_dict',
                                 {}),
        'edit_type_count_dict':
        process_summary_dict.get('edit_type_count_dict', {}),
        'archive_group_sample_pair_count_dict':
        process_summary_dict.get('archive_group_sample_pair_count_dict', {}),
        'archive_group_image_member_count_dict':
        process_summary_dict.get('archive_group_image_member_count_dict', {}),
        'benchmark_row_count':
        process_summary_dict.get('benchmark_row_count', 0),
        'benchmark_image_count':
        process_summary_dict.get('benchmark_image_count', 0),
        'benchmark_edit_type_count_dict':
        process_summary_dict.get('benchmark_edit_type_count_dict', {}),
        'invalid_sample_pair_list':
        process_summary_dict.get('invalid_sample_pair_list',
                                 [])[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'missing_sample_pair_list':
        all_missing_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'orphan_image_path_list':
        sorted(orphan_image_path_list),
        'check_warning_message_list':
        check_warning_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'check_error_message_list':
        check_error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }

    try:
        with open(save_check_result_path, 'w',
                  encoding='UTF-8') as save_json_file:
            json.dump(save_check_result_dict,
                      save_json_file,
                      ensure_ascii=False)
    except Exception as e:
        check_error_message_list.append(f'save check result failed {e}')

    return check_error_message_list, check_warning_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    file_copy_pair_list, archive_group_list, annotation_group_list = get_all_file_and_archive_group(
        root_dataset_path)

    print('1111', 'copy file', len(file_copy_pair_list), 'archive group',
          len(archive_group_list), 'annotation file',
          len(annotation_group_list))
    for per_archive_group in archive_group_list:
        print('1111', 'group', per_archive_group[0], 'parts',
              len(per_archive_group[4]))

    # 解压前预检: 分片数/连号/整包截断/标注json行数与字段，任一不过直接中断，
    # 不白跑5.3T解压(实测.cache里残留了20多个*.incomplete，下载确实中断过)
    precheck_error_message_list = []
    precheck_error_message_list.extend(
        check_archive_group_complete(archive_group_list))
    precheck_error_message_list.extend(
        check_annotation_file_complete(annotation_group_list))

    print('1111', 'precheck error', precheck_error_message_list[:20])
    if len(precheck_error_message_list) > 0:
        raise Exception(f'precheck failed {precheck_error_message_list[:20]}')

    copy_error_message_list = []
    copy_func = partial(process_single_file_copy,
                        save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_copy_result in tqdm(pool.imap_unordered(
                copy_func, file_copy_pair_list),
                                    total=len(file_copy_pair_list)):
            copy_error_message_list.extend(per_copy_result[1])

    extract_error_message_list = []
    total_image_member_count = 0
    archive_group_image_member_count_dict = {}
    benchmark_row_count, benchmark_image_count = 0, 0
    benchmark_edit_type_count_dict = {}

    extract_func = partial(process_single_archive_group,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_archive_result in tqdm(pool.imap_unordered(
                extract_func, archive_group_list),
                                       total=len(archive_group_list)):
            per_archive_group_key = per_archive_result['archive_group_key']

            archive_group_image_member_count_dict[
                per_archive_group_key] = per_archive_result[
                    'image_file_member_count']
            if per_archive_group_key != BENCHMARK_ARCHIVE_GROUP_KEY:
                total_image_member_count += per_archive_result[
                    'image_file_member_count']
            else:
                benchmark_row_count = per_archive_result['benchmark_row_count']
                benchmark_image_count = per_archive_result[
                    'benchmark_image_count']
                benchmark_edit_type_count_dict = per_archive_result[
                    'benchmark_edit_type_count_dict']

            print('2222', per_archive_group_key, 'member',
                  per_archive_result['total_member_count'], 'file member',
                  per_archive_result['total_file_member_count'],
                  'image member',
                  per_archive_result['image_file_member_count'],
                  'expected image member',
                  per_archive_result['expected_image_member_count'], 'extract',
                  per_archive_result['extract_file_count'], 'skip',
                  per_archive_result['skip_file_count'], 'error',
                  len(per_archive_result['error_message_list']))

            if len(per_archive_result['error_message_list']) > 0:
                print('7777', per_archive_group_key,
                      per_archive_result['error_message_list'][:5])
                extract_error_message_list.append(
                    f'{per_archive_group_key} extract error num '
                    f'{len(per_archive_result["error_message_list"])} '
                    f'first {per_archive_result["error_message_list"][0]}')

    annotation_error_message_list = []
    total_annotation_row_count, total_valid_sample_pair_count = 0, 0
    total_single_reference_sample_pair_count = 0
    all_valid_unique_image_path_set, all_unique_image_path_set = set(), set()
    annotation_row_count_dict = {}
    annotation_valid_sample_pair_count_dict = {}
    edit_type_count_dict = collections.Counter()
    archive_group_sample_pair_count_dict = collections.Counter()
    all_invalid_sample_pair_list = []

    annotation_func = partial(process_single_annotation_file,
                              save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_annotation_result in tqdm(pool.imap_unordered(
                annotation_func, annotation_group_list),
                                          total=len(annotation_group_list)):
            per_annotation_group_name, per_row_count, per_valid_sample_pair_count, per_single_reference_sample_pair_count, per_valid_unique_image_path_set, per_all_unique_image_path_set, per_edit_type_count_dict, per_archive_group_sample_pair_count_dict, per_invalid_sample_pair_list, per_error_message_list = per_annotation_result

            total_annotation_row_count += per_row_count
            total_valid_sample_pair_count += per_valid_sample_pair_count
            total_single_reference_sample_pair_count += per_single_reference_sample_pair_count
            all_valid_unique_image_path_set.update(
                per_valid_unique_image_path_set)
            all_unique_image_path_set.update(per_all_unique_image_path_set)
            annotation_row_count_dict[
                per_annotation_group_name] = per_row_count
            annotation_valid_sample_pair_count_dict[
                per_annotation_group_name] = per_valid_sample_pair_count
            edit_type_count_dict.update(per_edit_type_count_dict)
            archive_group_sample_pair_count_dict.update(
                per_archive_group_sample_pair_count_dict)
            all_invalid_sample_pair_list.extend(
                per_invalid_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM])

            if len(per_error_message_list) > 0:
                print('7777', per_annotation_group_name,
                      per_error_message_list[:5])
                annotation_error_message_list.extend([
                    f'{per_annotation_group_name} {per_error_message}'
                    for per_error_message in per_error_message_list
                ])

            print('2222', per_annotation_group_name, 'row', per_row_count,
                  'valid sample pair', per_valid_sample_pair_count,
                  'invalid sample pair',
                  per_row_count - per_valid_sample_pair_count,
                  'valid unique image', len(per_valid_unique_image_path_set))

    total_invalid_sample_pair_count = total_annotation_row_count - total_valid_sample_pair_count

    print('3333', 'total annotation row', total_annotation_row_count,
          'total valid sample pair', total_valid_sample_pair_count,
          'total invalid sample pair',
          total_invalid_sample_pair_count, 'total valid unique image',
          len(all_valid_unique_image_path_set), 'total all unique image',
          len(all_unique_image_path_set), 'total image member',
          total_image_member_count, 'benchmark row', benchmark_row_count,
          'benchmark image', benchmark_image_count)
    print('3333', 'edit type:', dict(edit_type_count_dict))

    # 标注侧全量期望值硬对账
    if total_annotation_row_count != EXPECTED_TOTAL_ANNOTATION_ROW_COUNT:
        annotation_error_message_list.append(
            f'total annotation row count not match '
            f'{total_annotation_row_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_ROW_COUNT}')

    if total_valid_sample_pair_count != EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT:
        annotation_error_message_list.append(
            f'total valid sample pair count not match '
            f'{total_valid_sample_pair_count} != '
            f'{EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT}')

    if total_invalid_sample_pair_count != EXPECTED_TOTAL_INVALID_SAMPLE_PAIR_COUNT:
        annotation_error_message_list.append(
            f'total invalid sample pair count not match '
            f'{total_invalid_sample_pair_count} != '
            f'{EXPECTED_TOTAL_INVALID_SAMPLE_PAIR_COUNT}')

    if total_single_reference_sample_pair_count != total_valid_sample_pair_count:
        annotation_error_message_list.append(
            f'total single reference sample pair count not match '
            f'{total_single_reference_sample_pair_count} != '
            f'{total_valid_sample_pair_count}')

    if len(all_valid_unique_image_path_set
           ) != EXPECTED_TOTAL_VALID_UNIQUE_IMAGE_COUNT:
        annotation_error_message_list.append(
            f'total valid unique image count not match '
            f'{len(all_valid_unique_image_path_set)} != '
            f'{EXPECTED_TOTAL_VALID_UNIQUE_IMAGE_COUNT}')

    if benchmark_row_count != EXPECTED_BENCHMARK_ROW_COUNT:
        annotation_error_message_list.append(
            f'benchmark row count not match {benchmark_row_count} != '
            f'{EXPECTED_BENCHMARK_ROW_COUNT}')

    if benchmark_image_count != EXPECTED_BENCHMARK_IMAGE_COUNT:
        annotation_error_message_list.append(
            f'benchmark image count not match {benchmark_image_count} != '
            f'{EXPECTED_BENCHMARK_IMAGE_COUNT}')

    process_summary_dict = {
        'total_annotation_row_count':
        total_annotation_row_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_invalid_sample_pair_count':
        total_invalid_sample_pair_count,
        'total_single_reference_sample_pair_count':
        total_single_reference_sample_pair_count,
        'all_valid_unique_image_path_set':
        all_valid_unique_image_path_set,
        'all_unique_image_path_set':
        all_unique_image_path_set,
        'annotation_row_count_dict':
        annotation_row_count_dict,
        'annotation_valid_sample_pair_count_dict':
        annotation_valid_sample_pair_count_dict,
        'edit_type_count_dict':
        dict(edit_type_count_dict),
        'archive_group_sample_pair_count_dict':
        dict(archive_group_sample_pair_count_dict),
        'archive_group_image_member_count_dict':
        archive_group_image_member_count_dict,
        'benchmark_row_count':
        benchmark_row_count,
        'benchmark_image_count':
        benchmark_image_count,
        'benchmark_edit_type_count_dict':
        benchmark_edit_type_count_dict,
        'invalid_sample_pair_list':
        all_invalid_sample_pair_list,
    }

    check_error_message_list, check_warning_message_list = check_image_annotation_pair(
        save_dataset_path, process_summary_dict)

    for per_warning_message in check_warning_message_list:
        print('2222', per_warning_message)

    all_error_message_list = []
    all_error_message_list.extend(copy_error_message_list)
    all_error_message_list.extend(extract_error_message_list)
    all_error_message_list.extend(annotation_error_message_list)
    all_error_message_list.extend(check_error_message_list)

    print('3333', 'total error', len(all_error_message_list),
          all_error_message_list[:20])

    if len(all_error_message_list) > 0:
        # 拷贝/解压/解析/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} '
            f'{all_error_message_list[:20]}')

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/VINS-120K'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
