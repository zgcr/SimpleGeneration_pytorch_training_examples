import os
import re
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'uno_1m'

SAVE_DATASET_DIR_NAME = 'UNO-1M'

# ==============================================================================
# 【这个数据集为什么只出t2i、不出ti2i】
# UNO-1M原始定位是主体驱动生成(subject-driven generation)数据集: 每条标注是
# "同一个主体(subject)在两个不同场景下各生成一张图 + 两条各自的caption",
# 全库**没有任何编辑指令、没有mask**，两张图也不是"编辑前/编辑后"的同构关系。
# 拿它做图像编辑数据集就必须人工拼一句原始数据里根本不存在的引导语
# (形如"generate an image of the same <subject> as in the reference image: ...")，
# 按方案确认**不做ti2i**，只把每张图 + 它自己的caption当成独立的t2i样本，
# 所以本目录下这个数据集只有这一个脚本、没有对应的resave ti2i脚本。
# 代价是"这两张图是同一主体"这个配对关系在新数据集里彻底丢失(想做主体驱动训练
# 必须回 huggingface_datasets_unzip/UNO-1M 重跑)，上游解压产物会一直保留，可恢复。
#
# 【上游012解压产物实测规格】
# UNO-1M/
# ├── unzip_images/split{1..102}/<原名>.png   2022186张，路径100%唯一
# ├── unzip_annotations/split{1..102}.jsonl   102个分片，合计1011093行样本对
# ├── unzip_source_annotations/split{N}.json  labels原样拷贝(830MB)，
# │                                           是jsonl的真子集，本脚本不读不搬
# └── unzip_check_missing_images.json         上游对账报告(check_error_count=0)
#
# 【单行标注的全部属性与本脚本的取舍】新标注只保留
# width/height/t2i_caption/t2i_caption_length四个key，
# 所以下面标"丢弃"的属性落盘后**永久丢失**，按方案确认全部丢弃、不另存索引:
#   reference_image_relative_path / target_image_relative_path : 两张图的相对路径
#                     -> 有用(定位图像 + 推子集名 + 生成保存图像名)
#   reference_image_caption / target_image_caption : 两张图各自的英文caption
#                     -> 有用(**各自作为独立t2i样本的t2i_caption**)
#   caption         : 与target_image_caption完全相同的冗余字段 -> 不用
#   reference_image_path / target_image_path : = unzip_images/<相对路径>，
#                     上游拼好的落盘路径 -> 不用(本脚本自己拼，只做交叉核对)
#   subject_list    : 主体词1~5个(1个占98.6%、2个13105、3个599、4个43、5个10)
#                     -> 丢弃(主体词丢失后无法再按主体去重/分组)
#   judgment        : VLM判"两图是否同一主体"，实测same 651521 / yes 359571 /
#                     "同一主题" 1(唯一一条中文，上游VLM输出污染)
#                     -> 只用于过滤(见VALID_JUDGMENT_NAME_LIST)，不落盘
#   score_final / score_final_valid / score_final_raw / score_part /
#   score_part_valid : 主体一致性打分(0.0~4.0)及细粒度维度分。README建议主体驱动
#                     训练取>=3.5(528990对)、UNO论文只用4.0(404258对)
#                     -> **按方案确认t2i不按它过滤**: 该分数衡量的是"两张图是不是
#                     同一主体"，与"单张图和它自己的caption是否匹配"无关，
#                     拿它筛t2i会白丢一半数据。代价是落盘后分数永久丢失，
#                     下游无法再复现UNO论文的4.0分子集
#   dataset_task_type / sample_key / archive_name : 上游定位与溯源信息
#                     -> archive_name只用于硬对账，其余丢弃。split{N}这个分片归属
#                     信息落盘后无法恢复，下游不能再"只训某几个split"
#   文件名里的_HxW后缀 : 原始分辨率 -> 只做核对，宽高一律取实际解码shape
# ==============================================================================

# 上游012解压脚本把每个split{N}.tar.gz解成
# unzip_images/split{N}/<原名>.png，同时在
# unzip_annotations/split{N}.jsonl 里按分片另存了一份汇总标注(每行一个样本对)。
# 本脚本只读这102个jsonl定位样本，绝不os.walk图像目录:
# 上游一共解出202万张png，扫一遍目录树在NAS上不可接受
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_SUFFIX = '.jsonl'

# 图像落盘根目录，标注里的*_relative_path形如split1/xxx.png，
# 拼上这一段才是真实图像路径
LOAD_IMAGE_DIR_NAME = 'unzip_images'

# 一个样本对固定拆成2条独立的t2i样本: 每张图配自己那条caption。
# 顺序固定为[参考图, 目标图]，保证输出顺序可复现
ANNOTATION_IMAGE_KEY_NAME_PAIR_LIST = [
    [
        'reference_image_relative_path',
        'reference_image_caption',
    ],
    [
        'target_image_relative_path',
        'target_image_caption',
    ],
]

# 上游拼好的落盘路径字段，只用来和本脚本自己拼出来的路径交叉核对，不直接使用
ANNOTATION_IMAGE_PATH_KEY_NAME_PAIR_LIST = [
    'reference_image_path',
    'target_image_path',
]

ANNOTATION_ARCHIVE_NAME_KEY_NAME = 'archive_name'

ANNOTATION_JUDGMENT_KEY_NAME = 'judgment'

# judgment的合法取值(小写后比较)。实测1011093行里只有same/yes/"同一主题"三种，
# 其中"同一主题"只有1条，是上游VLM把判定结论写成了中文，属于输出污染，
# 按方案确认这一条**整对丢弃**(连带丢掉它的2张图)。
# 这一步排在所有图像与caption过滤之前，所以后面每一项过滤计数都不含这条
VALID_JUDGMENT_NAME_LIST = [
    'same',
    'yes',
]

# 一个样本对里两条caption逐字完全相同的，按方案确认**整对丢弃**。
# 实测71274对(占7.05%)，这类样本对拆成2条t2i样本后会变成"两张不同的图配同一条
# caption"，等价于同一条文本监督重复两次。
# 判定用strip后的原文做全串精确比较，不做任何归一化(大小写/标点都算不同)
SKIP_SAME_CAPTION_PAIR_FLAG = True

# 从原始图像名前缀里切出子集名(即官方的四条造数流水线)。
# 上游标注里没有任何task/edit_type字段，但图像文件名前缀就是造数流水线名，
# 形如 object365_w1024_h2048_split_Shovel_8901_6_7_1_1023x1024，
# 其中_w{W}_h{H}_split之前的那一段就是流水线名。
# 实测同一个样本对的两张图流水线名100%相同(1011093/1011093)，所以可以安全当子集
SET_NAME_PATTERN = re.compile(r'^(?P<source>.+?)_w\d+_h\d+_split')

# 保留下来的子集集合必须与这个白名单严格一一对应，不多也不少。
# 实测全库2022186个图像名归一化后只切出这4个流水线名、没有任何长尾值,
# 所以不需要mix兜底子集; 上游之后新增流水线会在check_load_annotation_count里
# 被"unknown set"硬拦下来
SAVE_SET_NAME_LIST = [
    'class_generation',
    'object365',
    'scene_prompt_object_object_v1',
    'scene_prompt_object_person_v1',
]

SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点，与001~011完全一致
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 原始图像名里的非法字符: 实测有空格503108个(42万张图的名字里带空格，如
# ..._Green Vegetables_Sandwich_...)、'('与')'各392、"'"162、'&'116、'é'48、
# ':'40、','34、';'18。这些字符会让保存名违反上面的白名单，
# 按方案确认把**连续**非法字符压缩成单个下划线(而不是逐字符替换、也不是整条丢弃:
# 整条丢会白丢约42万张含空格的图，占21%)。
# 实测这样归一化之后2022186个名字**仍然100%唯一、0撞名**，
# 归一化后最长前缀142字符(scene_prompt_object_object_v1那条)，
# 加上uno_1m_<子集名>_前缀与.jpg后缀最长约180字节，远低于文件系统255字节上限
ILLEGAL_IMAGE_NAME_CHAR_PATTERN = re.compile(r'[^a-z0-9\-\.]+')

# 原始图像名尾部自带的分辨率后缀。实测它是**高x宽**而不是宽x高
# (抽样150张非正方形图100%命中_{h}x{w}、0张命中_{w}x{h}，另20张是正方形图)。
# 本脚本不拿它做分辨率过滤(万一它和实际图像不一致就会出偏差)，
# 只在解码后顺手比对一次，不一致的张数只上报不丢样本(实测抽样192张0例不符)
IMAGE_NAME_RESOLUTION_PATTERN = re.compile(r'_(?P<height>\d+)x(?P<width>\d+)$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉。
# 实测抽样192张全部是RGB、0张解码失败
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 本机128核，这里和001/003保持一致取32。本数据集要对约188万张图各解码两遍
# (jsonl扫描阶段校验一次、写盘阶段重编码一次)，想跑快可以直接调大这个常量
PROCESS_NUM = 32

PER_FOLDER_IMAGE_NUM = 10000

# 实测上游jsonl分片数(split1..split102连号)，数量不对说明上游012没跑完
EXPECTED_ANNOTATION_FILE_NUM = 102

# 实测每个分片的标注行数(即样本对数)，直接当完整性ground truth，
# 少一行都说明上游012没跑完或产物被改动过。
# 与上游unzip_check_missing_images.json的archive_sample_pair_count_dict完全一致
EXPECTED_ARCHIVE_ANNOTATION_COUNT_DICT = {
    'split1': 10000,
    'split2': 10000,
    'split3': 10000,
    'split4': 10000,
    'split5': 10000,
    'split6': 10000,
    'split7': 10000,
    'split8': 10000,
    'split9': 10000,
    'split10': 10000,
    'split11': 10000,
    'split12': 10000,
    'split13': 10000,
    'split14': 10000,
    'split15': 10000,
    'split16': 10000,
    'split17': 10000,
    'split18': 10000,
    'split19': 10000,
    'split20': 10000,
    'split21': 10000,
    'split22': 10000,
    'split23': 10000,
    'split24': 10000,
    'split25': 10000,
    'split26': 10000,
    'split27': 10000,
    'split28': 10000,
    'split29': 10000,
    'split30': 10000,
    'split31': 10000,
    'split32': 9442,
    'split33': 10000,
    'split34': 10000,
    'split35': 10000,
    'split36': 10000,
    'split37': 10000,
    'split38': 10000,
    'split39': 10000,
    'split40': 10000,
    'split41': 10000,
    'split42': 10000,
    'split43': 10000,
    'split44': 10000,
    'split45': 10000,
    'split46': 10000,
    'split47': 10000,
    'split48': 10000,
    'split49': 10000,
    'split50': 10000,
    'split51': 10000,
    'split52': 10000,
    'split53': 10000,
    'split54': 10000,
    'split55': 10000,
    'split56': 10000,
    'split57': 10000,
    'split58': 10000,
    'split59': 10000,
    'split60': 10000,
    'split61': 10000,
    'split62': 10000,
    'split63': 10000,
    'split64': 10000,
    'split65': 10000,
    'split66': 2080,
    'split67': 10000,
    'split68': 10000,
    'split69': 10000,
    'split70': 10000,
    'split71': 10000,
    'split72': 10000,
    'split73': 10000,
    'split74': 10000,
    'split75': 10000,
    'split76': 10000,
    'split77': 10000,
    'split78': 10000,
    'split79': 10000,
    'split80': 10000,
    'split81': 10000,
    'split82': 10000,
    'split83': 10000,
    'split84': 10000,
    'split85': 10000,
    'split86': 10000,
    'split87': 10000,
    'split88': 10000,
    'split89': 10000,
    'split90': 10000,
    'split91': 10000,
    'split92': 10000,
    'split93': 10000,
    'split94': 10000,
    'split95': 10000,
    'split96': 10000,
    'split97': 10000,
    'split98': 10000,
    'split99': 10000,
    'split100': 10000,
    'split101': 10000,
    'split102': 9571,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = sum(
    EXPECTED_ARCHIVE_ANNOTATION_COUNT_DICT.values())

# 样本对级过滤的实测精确条数(全量扫描102个jsonl，非抽样)，解析阶段逐项硬对账。
# 这两项都排在任何图像与caption过滤之前，实测两者无交集(同时命中的0条)
EXPECTED_SKIP_INVALID_JUDGMENT_PAIR_COUNT = 1

EXPECTED_SKIP_SAME_CAPTION_PAIR_COUNT = 71274

# 被丢弃的非法judgment取值明细(只有中文那一条)
EXPECTED_SKIP_INVALID_JUDGMENT_NAME_COUNT_DICT = {
    '同一主题': 1,
}

# 过完样本对级过滤后剩下的样本对数与图像数(图像数恒为样本对数的2倍，
# 因为一个样本对固定拆成2条t2i样本、且两张图同属一个子集)
EXPECTED_SET_PAIR_COUNT_DICT = {
    'class_generation': 283780,
    'object365': 297966,
    'scene_prompt_object_object_v1': 187539,
    'scene_prompt_object_person_v1': 170533,
}

EXPECTED_TOTAL_PAIR_COUNT = sum(EXPECTED_SET_PAIR_COUNT_DICT.values())

EXPECTED_SET_IMAGE_COUNT_DICT = {
    per_set_name: per_pair_count * 2
    for per_set_name, per_pair_count in EXPECTED_SET_PAIR_COUNT_DICT.items()
}

EXPECTED_TOTAL_IMAGE_COUNT = sum(EXPECTED_SET_IMAGE_COUNT_DICT.values())

# 文本层各类不合格描述的实测精确张数(只统计过完样本对级过滤后的1879636张图)，
# 解析阶段逐项硬对账。
# too_short全在场景类流水线(scene_prompt_object_object_v1 1883 /
# scene_prompt_object_person_v1 29 / object365 2)，形如只有几个字符的主体词;
# too_long只有3张、全在class_generation
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 0,
    'too_short_caption_count': 1914,
    'too_long_caption_count': 3,
}

# 缺图与保存名非法的实测精确张数: 上游解压对账已闭合(missing_image_count=0)，
# 归一化后也0撞名0非法名，所以这三项都必须是0，非0说明上游产物被改动过
EXPECTED_MISSING_IMAGE_COUNT = 0

EXPECTED_INVALID_IMAGE_NAME_COUNT = 0

EXPECTED_DUPLICATE_IMAGE_NAME_COUNT = 0

# 过完文本层全部过滤后每个子集剩下的图像数(= 落盘t2i样本数的上限，
# 再往后只会被图像层过滤扣掉，实测抽样预计为0)，合计1877719张
EXPECTED_SET_TEXT_VALID_IMAGE_COUNT_DICT = {
    'class_generation': 567557,
    'object365': 595930,
    'scene_prompt_object_object_v1': 373195,
    'scene_prompt_object_person_v1': 341037,
}

EXPECTED_TOTAL_TEXT_VALID_IMAGE_COUNT = sum(
    EXPECTED_SET_TEXT_VALID_IMAGE_COUNT_DICT.values())

# 最终产出的子集数，必须与SAVE_SET_NAME_LIST严格一一对应(不多也不少)。
# 4个子集都在34万张以上，每个都会切出多个满10000张的文件夹
# (class_generation 57 / object365 60 / scene_prompt_object_object_v1 38 /
#  scene_prompt_object_person_v1 35，合计190个文件夹)，
# 每个子集的文件夹数都远小于100，所以不需要像003/009/010那样再套一层
# PER_SET_FOLDER_NUM子集目录，子集名直接就是四条流水线名
EXPECTED_SAVE_SET_COUNT = len(SAVE_SET_NAME_LIST)

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 实测1879636条描述(过完样本对级过滤后)长度: 目标图侧min 4/p50 116/p90 170/
# p99 240/max 718，参考图侧min 4/p50 102/p90 155/max 499。
# 阈值按方案确认取10/512: 下限砍掉1914张(0.10%，全是只剩几个字符的主体词)，
# 上限砍掉3张(全在class_generation)
MIN_CAPTION_LENGTH = 10

MAX_CAPTION_LENGTH = 512


def get_set_name(per_image_name_prefix):
    """从原始图像名前缀里切出子集名(即官方的四条造数流水线名)

    图像名形如 object365_w1024_h2048_split_Shovel_8901_6_7_1_1023x1024,
    _w{W}_h{H}_split之前的那一段就是流水线名，统一小写后当子集名。
    上游标注里没有任何task/edit_type字段，这个前缀是全库唯一能拿到的任务类型信息。
    切不出来时返回空串，由调用方按"缺子集名"整对丢弃并上报(实测0条)。
    """
    per_match_result = SET_NAME_PATTERN.match(str(per_image_name_prefix))
    if not per_match_result:
        return ''

    return per_match_result.group('source').strip().lower()


def get_normalized_image_name_prefix(per_image_name_prefix):
    """把原始图像名前缀归一化成只含小写字母/数字/下划线/中划线/点的保存名前缀

    先统一小写，再把**连续**的非法字符压缩成单个下划线
    (原名里有空格/圆括号/单引号/&/é/冒号/逗号/分号，见
    ILLEGAL_IMAGE_NAME_CHAR_PATTERN处的实测明细)。
    实测这样归一化之后全库2022186个名字仍然100%唯一、0撞名。
    """
    per_image_name_prefix = str(per_image_name_prefix).strip().lower()

    return ILLEGAL_IMAGE_NAME_CHAR_PATTERN.sub('_', per_image_name_prefix)


def get_image_name_resolution(per_image_name_prefix):
    """从图像名尾部的分辨率后缀里解析出[宽, 高]，解析不出来时返回[0, 0]

    实测这个后缀是**高x宽**而不是宽x高(抽样150张非正方形图100%命中_{h}x{w})。
    只用于和真实解码结果比对，不参与任何过滤。
    """
    per_match_result = IMAGE_NAME_RESOLUTION_PATTERN.search(
        str(per_image_name_prefix))
    if not per_match_result:
        return [0, 0]

    return [
        int(per_match_result.group('width')),
        int(per_match_result.group('height')),
    ]


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上，逐样本打一次os.path.exists就是一次网络往返，
    188万张图就要打188万次，这一步本身就能占掉整个扫描阶段的大头。
    实测同一个jsonl里的图像全部落在同一个分片目录下(split1.jsonl的图像相对路径
    都是split1/xxx.png)，所以这里按目录缓存一次os.listdir的结果，
    之后只做集合查表，网络往返次数从"图像张数"降到"分片目录数"(102次)。
    listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判，
    保证判定结果和改造前完全一致。
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


def process_single_image_check(per_image_path):
    """校验单张图像能否正常解码，并过滤非RGB图和极端分辨率图

    返回的宽高只用于和图像名里的分辨率后缀比对，
    最终写进json的宽高一定取自实际写盘图像的shape。
    """
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把RGBA图静默丢掉alpha通道、
    # 把CMYK图静默转成3通道，所以必须先用PIL读原始mode才能把这些图判出来
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

    # 检查图像短边(实测抽样192张短边min 666，这个阈值只作兜底)
    if min(per_image_h, per_image_w) < MIN_IMAGE_SHORT_SIDE:
        print('6666', per_image_path, per_image_w, per_image_h)
        return None

    # 检查图像宽高比，取长短边之比，宽高比大于8和小于1/8这两种极端样本一起判掉
    # (实测抽样192张宽高比max 1.54，这个阈值也只作兜底)
    per_image_aspect_ratio = max(per_image_w / per_image_h,
                                 per_image_h / per_image_w)
    if per_image_aspect_ratio > MAX_IMAGE_ASPECT_RATIO:
        print('7777', per_image_path, per_image_w, per_image_h)
        return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_annotation_file(annotation_file_pair):
    """解析单个上游jsonl标注，把每个样本对拆成2条独立的t2i样本

    这个worker把样本对级过滤(json坏行、分片名错位、judgment非法、两条caption
    完全相同、缺子集名、两张图子集名不一致)、文本层过滤(缺图、保存名非法、
    描述为空或过短、描述过长)和图像层过滤(能否解码、是否RGB、短边、宽高比)
    一次做完。
    图像校验没有像001那样单独再开一个Pool，是因为分两个Pool的话主进程要先攒
    188万条记录、再逐条发给check worker、再收回188万条，光进程间序列化就要来回
    搬几十GB，而合并进来之后IPC只传存活样本。
    判定逻辑、过滤口径、日志编号和001/003完全一致，图像也一样是解码两遍
    (这里校验一遍、写盘时重编码再解一遍)，没有为了省时间跳过任何一道校验。

    过滤顺序是固定的(样本对级 -> 文本层 -> 图像层)，各类计数的实测精确值就是按
    这个顺序数出来的，改顺序会让check_load_annotation_count的硬对账全部失效。
    """
    per_jsonl_path, root_dataset_path, per_archive_name = annotation_file_pair

    root_image_path = os.path.join(root_dataset_path, LOAD_IMAGE_DIR_NAME)

    total_annotation_count, load_annotation_failed_count = 0, 0
    archive_name_not_match_count = 0
    invalid_judgment_pair_count, same_caption_pair_count = 0, 0
    invalid_judgment_name_count_dict = {}
    missing_set_name_pair_count, set_name_not_match_pair_count = 0, 0
    unknown_set_name_pair_count = 0
    set_pair_count_dict, set_image_count_dict = {}, {}
    missing_image_count, invalid_image_name_count = 0, 0
    empty_caption_count = 0
    too_short_caption_count, too_long_caption_count = 0, 0
    set_text_valid_image_count_dict = {}
    invalid_image_count = 0
    image_name_resolution_not_match_count = 0
    image_annotation_pair_list = []

    # 每个worker只处理一个jsonl，缓存里通常只有一个分片目录，内存开销可忽略
    dir_file_name_cache_dict = {}

    try:
        load_jsonl_file = open(per_jsonl_path, 'r', encoding='UTF-8')
    except Exception as e:
        print('2222', per_jsonl_path, e)

        return [
            image_annotation_pair_list,
            per_archive_name,
            total_annotation_count,
            1,
            archive_name_not_match_count,
            invalid_judgment_pair_count,
            invalid_judgment_name_count_dict,
            same_caption_pair_count,
            missing_set_name_pair_count,
            set_name_not_match_pair_count,
            unknown_set_name_pair_count,
            set_pair_count_dict,
            set_image_count_dict,
            missing_image_count,
            invalid_image_name_count,
            empty_caption_count,
            too_short_caption_count,
            too_long_caption_count,
            set_text_valid_image_count_dict,
            invalid_image_count,
            image_name_resolution_not_match_count,
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
                load_annotation_failed_count += 1
                print('2222', per_jsonl_path, e)
                continue

            if not isinstance(per_annotation, dict):
                load_annotation_failed_count += 1
                print('2222', per_jsonl_path, 'annotation not a dict')
                continue

            # 行内分片名必须和这个jsonl的文件名前缀一致，不一致说明上游产物被
            # 搬动过、图像相对路径会指向别的分片目录(实测0条，这里只做防御)
            per_annotation_archive_name = per_annotation.get(
                ANNOTATION_ARCHIVE_NAME_KEY_NAME, '')
            if not isinstance(per_annotation_archive_name, str):
                per_annotation_archive_name = ''
            if per_annotation_archive_name.strip() != per_archive_name:
                archive_name_not_match_count += 1
                print('2222', per_jsonl_path, per_annotation_archive_name)
                continue

            # judgment非法的样本对整对丢弃(实测只有中文"同一主题"那1条)。
            # 这一步排在所有图像与caption过滤之前，
            # 保证后面每一项过滤计数都不含这条
            per_judgment_name = per_annotation.get(
                ANNOTATION_JUDGMENT_KEY_NAME, '')
            if not isinstance(per_judgment_name, str):
                per_judgment_name = ''
            per_judgment_name = per_judgment_name.strip().lower()

            if per_judgment_name not in VALID_JUDGMENT_NAME_LIST:
                invalid_judgment_pair_count += 1
                invalid_judgment_name_count_dict[
                    per_judgment_name] = invalid_judgment_name_count_dict.get(
                        per_judgment_name, 0) + 1
                print('3333', per_jsonl_path, per_judgment_name)
                continue

            # 先把两张图的相对路径和各自的caption取出来，
            # 两条caption完全相同的样本对要整对丢弃，所以必须先取完再判
            image_relative_path_list, t2i_caption_list = [], []
            for per_image_key_name, per_caption_key_name in ANNOTATION_IMAGE_KEY_NAME_PAIR_LIST:
                per_image_relative_path = per_annotation.get(
                    per_image_key_name, '')
                if not isinstance(per_image_relative_path, str):
                    per_image_relative_path = ''
                per_image_relative_path = per_image_relative_path.replace(
                    '\\', '/').strip().lstrip('/')
                image_relative_path_list.append(per_image_relative_path)

                per_t2i_caption = per_annotation.get(per_caption_key_name, '')
                # 上游描述固定是str，这里兼容list和str两种形式
                if isinstance(per_t2i_caption, (list, tuple)):
                    per_t2i_caption = per_t2i_caption[0] if len(
                        per_t2i_caption) > 0 else ''
                if not isinstance(per_t2i_caption, str):
                    per_t2i_caption = ''
                t2i_caption_list.append(per_t2i_caption.strip())

            # 两条caption逐字完全相同的样本对整对丢弃(实测71274对)，
            # 否则拆成2条t2i样本后就是"两张不同的图配同一条caption"
            if SKIP_SAME_CAPTION_PAIR_FLAG and t2i_caption_list[
                    0] == t2i_caption_list[1]:
                same_caption_pair_count += 1
                continue

            # 子集名由图像名前缀切出来，切不出来这个样本对就没法安全落盘
            image_name_prefix_list, set_name_list = [], []
            for per_image_relative_path in image_relative_path_list:
                per_image_name_prefix = os.path.splitext(
                    os.path.basename(per_image_relative_path))[0]
                image_name_prefix_list.append(per_image_name_prefix)
                set_name_list.append(get_set_name(per_image_name_prefix))

            if not set_name_list[0] or not set_name_list[1]:
                missing_set_name_pair_count += 1
                print('3333', per_jsonl_path, image_name_prefix_list[0],
                      image_name_prefix_list[1])
                continue

            # 同一个样本对的两张图必须同属一个子集(实测1011093对100%满足)，
            # 不一致说明上游配对错位，整对丢弃
            if set_name_list[0] != set_name_list[1]:
                set_name_not_match_pair_count += 1
                print('3333', per_jsonl_path, set_name_list[0],
                      set_name_list[1])
                continue

            per_set_name = set_name_list[0]

            # 白名单之外的子集名说明上游新增了造数流水线，必须显式感知(实测0条)
            if per_set_name not in SAVE_SET_NAME_LIST:
                unknown_set_name_pair_count += 1
                print('3333', per_jsonl_path, per_set_name)
                continue

            set_pair_count_dict[per_set_name] = set_pair_count_dict.get(
                per_set_name, 0) + 1

            # 到这里样本对级过滤全部结束，下面按图逐张做文本层与图像层过滤:
            # 一个样本对固定拆成2条独立的t2i样本，某一张图不合格只丢这一张,
            # 另一张照常落盘(t2i样本之间本来就没有配对关系)
            for per_image_relative_path, per_image_name_prefix, per_t2i_caption, per_image_path_key_name in zip(
                    image_relative_path_list, image_name_prefix_list,
                    t2i_caption_list,
                    ANNOTATION_IMAGE_PATH_KEY_NAME_PAIR_LIST):
                set_image_count_dict[per_set_name] = set_image_count_dict.get(
                    per_set_name, 0) + 1

                if not per_image_relative_path:
                    missing_image_count += 1
                    continue

                per_image_path = os.path.join(root_image_path,
                                              per_image_relative_path)
                if not check_image_file_exists(per_image_path,
                                               dir_file_name_cache_dict):
                    missing_image_count += 1
                    continue

                # 上游拼好的落盘路径只做交叉核对，不一致只上报不丢样本
                per_annotation_image_path = per_annotation.get(
                    per_image_path_key_name, '')
                if isinstance(per_annotation_image_path,
                              str) and per_annotation_image_path.replace(
                                  '\\', '/').strip().lstrip('/') != (
                                      f'{LOAD_IMAGE_DIR_NAME}/'
                                      f'{per_image_relative_path}'):
                    print('2222', per_image_path, per_annotation_image_path)

                # 保存图像名统一全小写，形如
                # uno_1m_object365_object365_w1024_h2048_split_shovel_8901_6_7_1_1023x1024.jpg
                per_save_image_name_prefix = get_normalized_image_name_prefix(
                    per_image_name_prefix)
                per_save_image_name = (f'{DATASET_NAME}_{per_set_name}_'
                                       f'{per_save_image_name_prefix}'
                                       f'{SAVE_IMAGE_NAME_SUFFIX}')

                # 保存名里出现路径分隔符或其它异常字符会写坏目录结构，
                # 这种样本没法保证保存名唯一(可能去覆盖别的样本)，直接丢弃(实测0条)
                if not VALID_IMAGE_NAME_PATTERN.match(per_save_image_name):
                    invalid_image_name_count += 1
                    print('2222', per_image_path, per_save_image_name)
                    continue

                # 空描述、全空格描述视为不合格样本对(实测0条)
                if not per_t2i_caption:
                    empty_caption_count += 1
                    print('3333', per_image_path, len(per_t2i_caption))
                    continue

                # 过短描述同样视为不合格样本对(实测1914张，基本都是场景类流水线里
                # 只剩几个字符的主体词)
                if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                    too_short_caption_count += 1
                    print('3333', per_image_path, len(per_t2i_caption))
                    continue

                # 过长描述同样视为不合格样本对(实测3张，全在class_generation)
                if len(per_t2i_caption) > MAX_CAPTION_LENGTH:
                    too_long_caption_count += 1
                    print('3333', per_image_path, len(per_t2i_caption))
                    continue

                set_text_valid_image_count_dict[
                    per_set_name] = set_text_valid_image_count_dict.get(
                        per_set_name, 0) + 1

                per_check_result = process_single_image_check(per_image_path)
                if per_check_result is None:
                    invalid_image_count += 1
                    continue

                per_image_w, per_image_h = per_check_result

                # 分辨率过滤一律用上面真实解码出来的shape，这里只是顺手核对一遍
                # 图像名尾部的分辨率后缀(实测是高x宽)，
                # 不一致只计数上报、不丢样本(实测抽样192张0例不符)
                per_name_image_w, per_name_image_h = get_image_name_resolution(
                    per_image_name_prefix)
                if per_name_image_w != per_image_w or per_name_image_h != per_image_h:
                    image_name_resolution_not_match_count += 1
                    print('2222', per_image_path, per_image_w, per_image_h,
                          per_name_image_w, per_name_image_h)

                # 子集名不依赖切分结果，所以保存图像名在这里就能完全定下来,
                # 后面排序与切分直接按保存图像名做
                image_annotation_pair_list.append([
                    per_set_name,
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

    return [
        image_annotation_pair_list,
        per_archive_name,
        total_annotation_count,
        load_annotation_failed_count,
        archive_name_not_match_count,
        invalid_judgment_pair_count,
        invalid_judgment_name_count_dict,
        same_caption_pair_count,
        missing_set_name_pair_count,
        set_name_not_match_pair_count,
        unknown_set_name_pair_count,
        set_pair_count_dict,
        set_image_count_dict,
        missing_image_count,
        invalid_image_name_count,
        empty_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        set_text_valid_image_count_dict,
        invalid_image_count,
        image_name_resolution_not_match_count,
    ]


def get_all_image_annotation_pair(root_dataset_path):
    """扫描上游解压好的jsonl标注，多进程组装图像路径和t2i描述的样本对列表

    这里只listdir unzip_annotations拿到102个jsonl路径，绝不去os.walk图像目录:
    上游解出202万张png，扫目录树在NAS上不可接受。
    每个jsonl里的图像全在同一个分片目录下，worker只需要对那个目录listdir一次。
    最后按保存图像名统一排序，保证输出顺序与串行版本完全一致。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    for per_jsonl_name in sorted(os.listdir(root_annotation_path)):
        if not per_jsonl_name.endswith(LOAD_ANNOTATION_FILE_SUFFIX):
            continue

        # 分片名由主进程按文件名算好后带给worker，
        # worker只认标注文件、数据集根目录、分片名这三个入参
        annotation_file_pair_list.append([
            os.path.join(root_annotation_path, per_jsonl_name),
            root_dataset_path,
            os.path.splitext(per_jsonl_name)[0],
        ])

    total_annotation_count, load_annotation_failed_count = 0, 0
    archive_name_not_match_count = 0
    invalid_judgment_pair_count, same_caption_pair_count = 0, 0
    missing_set_name_pair_count, set_name_not_match_pair_count = 0, 0
    unknown_set_name_pair_count = 0
    missing_image_count, invalid_image_name_count = 0, 0
    empty_caption_count = 0
    too_short_caption_count, too_long_caption_count = 0, 0
    invalid_image_count = 0
    image_name_resolution_not_match_count = 0
    archive_annotation_count_dict = {}
    invalid_judgment_name_count_dict = {}
    set_pair_count_dict, set_image_count_dict = {}, {}
    set_text_valid_image_count_dict = {}
    image_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            image_annotation_pair_list.extend(per_load_result[0])

            per_archive_name = per_load_result[1]
            archive_annotation_count_dict[
                per_archive_name] = archive_annotation_count_dict.get(
                    per_archive_name, 0) + per_load_result[2]

            total_annotation_count += per_load_result[2]
            load_annotation_failed_count += per_load_result[3]
            archive_name_not_match_count += per_load_result[4]
            invalid_judgment_pair_count += per_load_result[5]

            for per_judgment_name, per_judgment_count in per_load_result[
                    6].items():
                invalid_judgment_name_count_dict[
                    per_judgment_name] = invalid_judgment_name_count_dict.get(
                        per_judgment_name, 0) + per_judgment_count

            same_caption_pair_count += per_load_result[7]
            missing_set_name_pair_count += per_load_result[8]
            set_name_not_match_pair_count += per_load_result[9]
            unknown_set_name_pair_count += per_load_result[10]

            for per_set_name, per_set_count in per_load_result[11].items():
                set_pair_count_dict[per_set_name] = set_pair_count_dict.get(
                    per_set_name, 0) + per_set_count
            for per_set_name, per_set_count in per_load_result[12].items():
                set_image_count_dict[per_set_name] = set_image_count_dict.get(
                    per_set_name, 0) + per_set_count

            missing_image_count += per_load_result[13]
            invalid_image_name_count += per_load_result[14]
            empty_caption_count += per_load_result[15]
            too_short_caption_count += per_load_result[16]
            too_long_caption_count += per_load_result[17]

            for per_set_name, per_set_count in per_load_result[18].items():
                set_text_valid_image_count_dict[
                    per_set_name] = set_text_valid_image_count_dict.get(
                        per_set_name, 0) + per_set_count

            invalid_image_count += per_load_result[19]
            image_name_resolution_not_match_count += per_load_result[20]

    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: x[2])

    return [
        image_annotation_pair_list,
        len(annotation_file_pair_list),
        archive_annotation_count_dict,
        total_annotation_count,
        load_annotation_failed_count,
        archive_name_not_match_count,
        invalid_judgment_pair_count,
        invalid_judgment_name_count_dict,
        same_caption_pair_count,
        missing_set_name_pair_count,
        set_name_not_match_pair_count,
        unknown_set_name_pair_count,
        set_pair_count_dict,
        set_image_count_dict,
        missing_image_count,
        invalid_image_name_count,
        empty_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        set_text_valid_image_count_dict,
        invalid_image_count,
        image_name_resolution_not_match_count,
    ]


def check_load_annotation_count(
        annotation_file_count, archive_annotation_count_dict,
        total_annotation_count, invalid_judgment_pair_count,
        invalid_judgment_name_count_dict, same_caption_pair_count,
        set_pair_count_dict, set_image_count_dict,
        set_text_valid_image_count_dict, invalid_count_dict):
    """解析完标注后按分片和子集两级硬对账

    上游012的解压产物是一次性解出来的确定结果(unzip_check_missing_images.json里
    check_error_count=0)，条数对不上说明上游没跑完或产物被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账还能额外拦住"图像名前缀的切分写法被改动"这种分片级对账看不出来的问题。
    这里只对账到"文本层过滤之后"为止: 图像层过滤(解码/mode/短边/宽高比)的实测值
    只来自抽样(192张全部合格)，没有全量跑过，所以不写死期望值，只统计上报。
    """
    check_error_message_list = []

    if annotation_file_count != EXPECTED_ANNOTATION_FILE_NUM:
        check_error_message_list.append(
            f'annotation file num not match '
            f'{annotation_file_count} != {EXPECTED_ANNOTATION_FILE_NUM}')

    for per_archive_name in sorted(archive_annotation_count_dict.keys()):
        if per_archive_name not in EXPECTED_ARCHIVE_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown archive {per_archive_name}')
            continue

        per_expect_annotation_count = EXPECTED_ARCHIVE_ANNOTATION_COUNT_DICT[
            per_archive_name]
        if archive_annotation_count_dict[
                per_archive_name] != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_archive_name} annotation count not match '
                f'{archive_annotation_count_dict[per_archive_name]} != '
                f'{per_expect_annotation_count}')

    for per_archive_name in sorted(
            EXPECTED_ARCHIVE_ANNOTATION_COUNT_DICT.keys()):
        if per_archive_name not in archive_annotation_count_dict:
            check_error_message_list.append(
                f'missing archive {per_archive_name}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    # 样本对级过滤: judgment非法与两条caption完全相同这两项逐项硬对账，
    # 且被丢掉的judgment取值明细也要完全一致(多出新的非法取值必须显式感知)
    if invalid_judgment_pair_count != EXPECTED_SKIP_INVALID_JUDGMENT_PAIR_COUNT:
        check_error_message_list.append(
            f'invalid judgment pair count not match '
            f'{invalid_judgment_pair_count} != '
            f'{EXPECTED_SKIP_INVALID_JUDGMENT_PAIR_COUNT}')

    if invalid_judgment_name_count_dict != EXPECTED_SKIP_INVALID_JUDGMENT_NAME_COUNT_DICT:
        check_error_message_list.append(
            f'invalid judgment name count not match '
            f'{invalid_judgment_name_count_dict} != '
            f'{EXPECTED_SKIP_INVALID_JUDGMENT_NAME_COUNT_DICT}')

    if same_caption_pair_count != EXPECTED_SKIP_SAME_CAPTION_PAIR_COUNT:
        check_error_message_list.append(
            f'same caption pair count not match '
            f'{same_caption_pair_count} != '
            f'{EXPECTED_SKIP_SAME_CAPTION_PAIR_COUNT}')

    # 4个子集的样本对数与图像数逐个硬对账，且图像数必须恰好是样本对数的2倍
    # (一个样本对固定拆成2条t2i样本，两张图同属一个子集)
    for per_set_name in sorted(EXPECTED_SET_PAIR_COUNT_DICT.keys()):
        per_expect_pair_count = EXPECTED_SET_PAIR_COUNT_DICT[per_set_name]
        if per_set_name not in set_pair_count_dict:
            check_error_message_list.append(f'missing save set {per_set_name}')
            continue
        if set_pair_count_dict[per_set_name] != per_expect_pair_count:
            check_error_message_list.append(
                f'{per_set_name} set pair count not match '
                f'{set_pair_count_dict[per_set_name]} != '
                f'{per_expect_pair_count}')

    for per_set_name in sorted(set_pair_count_dict.keys()):
        if per_set_name not in SAVE_SET_NAME_LIST:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_pair_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_pair_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

    if sum(set_pair_count_dict.values()) != EXPECTED_TOTAL_PAIR_COUNT:
        check_error_message_list.append(
            f'total pair count not match '
            f'{sum(set_pair_count_dict.values())} != '
            f'{EXPECTED_TOTAL_PAIR_COUNT}')

    for per_set_name in sorted(EXPECTED_SET_IMAGE_COUNT_DICT.keys()):
        per_expect_image_count = EXPECTED_SET_IMAGE_COUNT_DICT[per_set_name]
        if set_image_count_dict.get(per_set_name, 0) != per_expect_image_count:
            check_error_message_list.append(
                f'{per_set_name} set image count not match '
                f'{set_image_count_dict.get(per_set_name, 0)} != '
                f'{per_expect_image_count}')

        if set_image_count_dict.get(
                per_set_name,
                0) != set_pair_count_dict.get(per_set_name, 0) * 2:
            check_error_message_list.append(
                f'{per_set_name} set image count not match set pair count '
                f'{set_image_count_dict.get(per_set_name, 0)} != '
                f'{set_pair_count_dict.get(per_set_name, 0)} * 2')

    if sum(set_image_count_dict.values()) != EXPECTED_TOTAL_IMAGE_COUNT:
        check_error_message_list.append(
            f'total image count not match '
            f'{sum(set_image_count_dict.values())} != '
            f'{EXPECTED_TOTAL_IMAGE_COUNT}')

    # 缺子集名/两张图子集名不一致/白名单外的子集名都必须是0:
    # 实测全库图像名100%能切出这4个流水线名之一、且同一对的两张图流水线名相同,
    # 非0说明上游新增了造数流水线或配对错位，必须显式感知
    for per_count_name in [
            'missing_set_name_pair_count', 'set_name_not_match_pair_count',
            'unknown_set_name_pair_count', 'archive_name_not_match_count'
    ]:
        if invalid_count_dict[per_count_name] != 0:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_count_dict[per_count_name]} != 0')

    # 缺图与保存名非法也必须是0(上游解压对账已闭合、归一化后0撞名0非法名)
    if invalid_count_dict[
            'missing_image_count'] != EXPECTED_MISSING_IMAGE_COUNT:
        check_error_message_list.append(
            f'missing image count not match '
            f'{invalid_count_dict["missing_image_count"]} != '
            f'{EXPECTED_MISSING_IMAGE_COUNT}')

    if invalid_count_dict[
            'invalid_image_name_count'] != EXPECTED_INVALID_IMAGE_NAME_COUNT:
        check_error_message_list.append(
            f'invalid image name count not match '
            f'{invalid_count_dict["invalid_image_name_count"]} != '
            f'{EXPECTED_INVALID_IMAGE_NAME_COUNT}')

    # 文本层各类不合格描述的张数逐项硬对账
    for per_count_name in sorted(EXPECTED_INVALID_CAPTION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_CAPTION_COUNT_DICT[per_count_name]
        if invalid_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_count_dict[per_count_name]} != {per_expect_count}')

    # 过完文本层全部过滤后每个子集剩下的图像数逐个硬对账
    for per_set_name in sorted(
            EXPECTED_SET_TEXT_VALID_IMAGE_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_SET_TEXT_VALID_IMAGE_COUNT_DICT[
            per_set_name]
        if set_text_valid_image_count_dict.get(per_set_name,
                                               0) != per_expect_count:
            check_error_message_list.append(
                f'{per_set_name} set text valid image count not match '
                f'{set_text_valid_image_count_dict.get(per_set_name, 0)} != '
                f'{per_expect_count}')

    if sum(set_text_valid_image_count_dict.values()
           ) != EXPECTED_TOTAL_TEXT_VALID_IMAGE_COUNT:
        check_error_message_list.append(
            f'total text valid image count not match '
            f'{sum(set_text_valid_image_count_dict.values())} != '
            f'{EXPECTED_TOTAL_TEXT_VALID_IMAGE_COUNT}')

    # 每个子集的图像数必须能被各类过滤计数完全解释:
    # 子集图像数 == 文本层存活数 + 该子集被文本层丢掉的数。
    # 这里只能在总量上闭合(各类过滤计数没有按子集回传)，总量闭合已经足够拦住
    # "某一类过滤被漏统计"这种问题
    per_total_text_invalid_count = (
        invalid_count_dict['missing_image_count'] +
        invalid_count_dict['invalid_image_name_count'] +
        invalid_count_dict['empty_caption_count'] +
        invalid_count_dict['too_short_caption_count'] +
        invalid_count_dict['too_long_caption_count'])
    if sum(set_text_valid_image_count_dict.values(
    )) + per_total_text_invalid_count != sum(set_image_count_dict.values()):
        check_error_message_list.append(
            f'text valid image count not self consistent '
            f'{sum(set_text_valid_image_count_dict.values())} + '
            f'{per_total_text_invalid_count} != '
            f'{sum(set_image_count_dict.values())}')

    return check_error_message_list


def get_deduplicated_image_annotation_pair(image_annotation_pair_list):
    """按保存图像名全局去重，每个保存图像名只保留排序后的第一条

    上游图像相对路径全库100%唯一(2022186个)，归一化成保存名之后实测**仍然
    100%唯一、0撞名**，所以这里正常应该一条都不丢，只是兜一道底:
    撞名会让后写的图像覆盖先写的、静默丢样本。
    排序键取[保存图像名, 图像路径]，保证同一个保存名留下的永远是同一条。
    """
    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[2], x[1]])

    duplicate_key_dict = {}
    duplicate_save_image_name_list = []
    deduplicated_image_annotation_pair_list = []
    for per_image_annotation_pair in image_annotation_pair_list:
        per_set_name, per_image_path, per_save_image_name, per_t2i_caption = per_image_annotation_pair
        if per_save_image_name in duplicate_key_dict:
            duplicate_save_image_name_list.append(per_save_image_name)
            print('2222', per_image_path, per_save_image_name)
            continue

        duplicate_key_dict[per_save_image_name] = 1
        deduplicated_image_annotation_pair_list.append(
            per_image_annotation_pair)

    deduplicated_image_annotation_pair_list = sorted(
        deduplicated_image_annotation_pair_list, key=lambda x: x[2])

    return deduplicated_image_annotation_pair_list, duplicate_save_image_name_list


def get_all_image_save_folder_pair(image_annotation_pair_list,
                                   save_dataset_path):
    """把过滤后的合格样本按子集分组，排序后每10000张切成一个文件夹

    切分必须在过滤全部完成之后做，且切分前先按保存图像名排序，这样才能保证每个
    文件夹都是满10000张(只有每个子集最后一个文件夹允许不满)。
    这里能直接按保存图像名排序、不像003那样要绕道用原始名前缀: 003的子集目录名由
    切分后的位置决定、和保存名互为循环依赖，而本数据集的子集名是图像名里切出来的
    造数流水线名、与切分完全无关，所以保存名在解析阶段就已经定死了。
    4个子集的文件夹数都远小于100(57/60/38/35)，所以不需要再套一层
    PER_SET_FOLDER_NUM子集目录。
    """
    per_set_image_annotation_pair_dict = {}
    for per_image_annotation_pair in image_annotation_pair_list:
        per_set_name = per_image_annotation_pair[0]
        if per_set_name not in per_set_image_annotation_pair_dict:
            per_set_image_annotation_pair_dict[per_set_name] = []
        per_set_image_annotation_pair_dict[per_set_name].append(
            per_image_annotation_pair)

    image_save_folder_pair_list = []
    set_folder_count_dict = {}
    for per_set_name in sorted(per_set_image_annotation_pair_dict.keys()):
        per_set_image_annotation_pair_list = sorted(
            per_set_image_annotation_pair_dict[per_set_name],
            key=lambda x: x[2])

        per_set_folder_count = 0
        for per_folder_start_index in range(
                0, len(per_set_image_annotation_pair_list),
                PER_FOLDER_IMAGE_NUM):
            per_folder_image_annotation_pair_list = per_set_image_annotation_pair_list[
                per_folder_start_index:per_folder_start_index +
                PER_FOLDER_IMAGE_NUM]

            per_folder_name = f'{per_set_name}_{per_set_folder_count:05d}'
            per_folder_image_path = os.path.join(save_dataset_path,
                                                 per_set_name, per_folder_name)
            os.makedirs(per_folder_image_path, exist_ok=True)

            per_folder_save_pair_list = []
            for per_image_annotation_pair in per_folder_image_annotation_pair_list:
                _, per_image_path, per_save_image_name, per_t2i_caption = per_image_annotation_pair
                per_folder_save_pair_list.append([
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

            # 一个文件夹就是一个写盘任务，worker写完这10000张后直接写出该文件夹的
            # json，主进程只收计数，不用把188万条记录再攒一遍
            image_save_folder_pair_list.append([
                per_set_name,
                per_folder_name,
                per_folder_save_pair_list,
            ])

            per_set_folder_count += 1

        set_folder_count_dict[per_set_name] = per_set_folder_count

    return image_save_folder_pair_list, set_folder_count_dict


def process_single_image_folder(image_save_folder_pair, save_dataset_path):
    """重新编码保存一个文件夹的图像，并写出与文件夹同名的json标注

    图像原分辨率多少保存时还是多少，不做任何缩放。
    上游图像全是png，这里统一重编码成jpg，只换编码格式不换像素尺寸;
    jpg编码参数用cv2.imencode('.jpg', img)的默认值(质量95 + 色度4:2:0)，
    与001~006/009~011这些t2i与图像编辑数据集保持完全一致
    (只有007/008那两个图像复原数据集因为GT要求无块效应才改成质量97 + 4:4:4)。
    以文件夹为任务粒度而不是以单张图为粒度: 本数据集约188万张图，逐图收结果的话
    主进程要再攒一份188万条的列表，而且中途挂了只能从头再来;按文件夹收之后
    主进程内存只和文件夹数(190个)相关，且json已经写全的文件夹可以直接跳过、
    支持断点续跑。
    """
    per_set_name, per_folder_name, per_folder_save_pair_list = image_save_folder_pair

    save_folder_path = os.path.join(save_dataset_path, per_set_name,
                                    per_folder_name)
    save_json_path = os.path.join(save_dataset_path, per_set_name,
                                  f'{per_folder_name}.json')

    expect_save_image_name_list = sorted([
        per_save_image_name
        for _, per_save_image_name, _ in per_folder_save_pair_list
    ])

    # 断点续跑: json已经写全且记录的图像名与本次任务完全一致时整个文件夹跳过
    if os.path.isfile(save_json_path):
        try:
            with open(save_json_path, 'r', encoding='UTF-8') as load_json_file:
                per_folder_annotation_dict = json.load(load_json_file)
        except Exception as e:
            print('9999', save_json_path, e)
            per_folder_annotation_dict = {}

        if sorted(per_folder_annotation_dict.keys(
        )) == expect_save_image_name_list and sorted(
                os.listdir(save_folder_path)) == expect_save_image_name_list:
            return [
                per_set_name,
                per_folder_name,
                len(per_folder_annotation_dict),
                0,
            ]

    per_folder_annotation_dict = {}
    save_image_failed_count = 0
    for per_folder_save_pair in per_folder_save_pair_list:
        per_image_path, per_save_image_name, per_t2i_caption = per_folder_save_pair

        try:
            per_image = cv2.imdecode(
                np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            save_image_failed_count += 1
            print('8888', per_image_path, e)
            continue

        if per_image is None or per_image.ndim != 3 or per_image.shape[2] != 3:
            save_image_failed_count += 1
            print('8888', per_image_path)
            continue

        # json里的宽高直接取自这个即将被编码写盘的数组的shape，
        # 中间不做resize，jpg编解码也不改变像素尺寸，所以宽高一定和保存图像一致
        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

        save_image_path = os.path.join(save_folder_path, per_save_image_name)

        if not os.path.exists(save_image_path):
            try:
                cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
            except Exception as e:
                save_image_failed_count += 1
                print('8888', save_image_path, e)
                continue

        # t2i_caption_length直接取即将写进json的这个字符串的长度，
        # 保证记录的长度和t2i_caption永远自洽(该字符串在过滤阶段已经strip过)
        per_folder_annotation_dict[per_save_image_name] = {
            'width': per_image_w,
            'height': per_image_h,
            't2i_caption': per_t2i_caption,
            't2i_caption_length': len(per_t2i_caption),
        }

    per_folder_annotation_dict = {
        per_save_image_name: per_folder_annotation_dict[per_save_image_name]
        for per_save_image_name in sorted(per_folder_annotation_dict.keys())
    }

    try:
        with open(save_json_path, 'w', encoding='UTF-8') as save_json_file:
            json.dump(per_folder_annotation_dict,
                      save_json_file,
                      ensure_ascii=False)
    except Exception as e:
        print('9999', save_json_path, e)

    return [
        per_set_name,
        per_folder_name,
        len(per_folder_annotation_dict),
        save_image_failed_count,
    ]


def check_single_save_folder(folder_check_pair, save_dataset_path):
    """校验单个文件夹: json与磁盘一一对应、图像名和描述合规、文件夹容量

    每个子集只有最后一个文件夹允许不满10000张，其余都必须是满10000张。
    """
    per_set_name, per_folder_name, per_is_set_last_folder = folder_check_pair

    check_error_message_list = []

    per_json_path = os.path.join(save_dataset_path, per_set_name,
                                 f'{per_folder_name}.json')
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

    # 除每个子集最后一个文件夹外都必须是满10000张
    if not per_is_set_last_folder and len(
            per_folder_annotation_dict) != PER_FOLDER_IMAGE_NUM:
        check_error_message_list.append(
            f'{per_folder_name} image num not match {len(per_folder_annotation_dict)} != {PER_FOLDER_IMAGE_NUM}'
        )

    per_folder_path = os.path.join(save_dataset_path, per_set_name,
                                   per_folder_name)
    per_exist_image_name_list = sorted(
        os.listdir(per_folder_path)) if os.path.isdir(per_folder_path) else []
    per_expect_image_name_list = sorted(per_folder_annotation_dict.keys())
    if per_exist_image_name_list != per_expect_image_name_list:
        check_error_message_list.append(
            f'{per_folder_name} image file not match {len(per_exist_image_name_list)} != {len(per_expect_image_name_list)}'
        )

    for per_save_image_name in per_expect_image_name_list:
        per_annotation = per_folder_annotation_dict[per_save_image_name]

        if not per_save_image_name.endswith(SAVE_IMAGE_NAME_SUFFIX):
            check_error_message_list.append(
                f'{per_save_image_name} image name suffix not match')
        if not per_save_image_name.startswith(
                f'{DATASET_NAME}_{per_set_name}_'):
            check_error_message_list.append(
                f'{per_save_image_name} image name prefix not match')
        if per_save_image_name != per_save_image_name.lower():
            check_error_message_list.append(
                f'{per_save_image_name} image name not all lower case')
        if not VALID_IMAGE_NAME_PATTERN.match(per_save_image_name):
            check_error_message_list.append(
                f'{per_save_image_name} image name has invalid char')
        if min(per_annotation['width'],
               per_annotation['height']) < MIN_IMAGE_SHORT_SIDE:
            check_error_message_list.append(
                f'{per_save_image_name} image short side not match')
        if max(
                per_annotation['width'] / per_annotation['height'],
                per_annotation['height'] / per_annotation['width'],
        ) > MAX_IMAGE_ASPECT_RATIO:
            check_error_message_list.append(
                f'{per_save_image_name} image aspect ratio not match')
        if len(per_annotation['t2i_caption'].strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_image_name} still an invalid caption')
        if len(per_annotation['t2i_caption'].strip()) > MAX_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_image_name} still a too long caption')
        # 记录的描述长度必须和描述字符串的实际长度对得上
        if per_annotation['t2i_caption_length'] != len(
                per_annotation['t2i_caption']):
            check_error_message_list.append(
                f'{per_save_image_name} t2i caption length not match')

    return [
        per_folder_name,
        len(per_folder_annotation_dict),
        check_error_message_list,
    ]


def check_save_dataset(save_dataset_path, set_folder_count_dict):
    """全部落盘后的收尾自校验: 文件夹容量、json与磁盘一一对应、图像名和描述合规

    和003的差别在于本数据集不需要PER_SET_FOLDER_NUM那层子集目录(4个子集的文件夹数
    都远小于100)，所以口径回到001的写法: 子集本身就是切分单位，
    每个子集只有最后一个文件夹允许不满10000张。
    190个文件夹每个都要listdir一万个文件再load一份json，串行跑在NAS上太久，
    所以这一步也按文件夹粒度开多进程。
    """
    check_error_message_list = []

    folder_check_pair_list = []
    for per_set_name in sorted(set_folder_count_dict.keys()):
        per_set_folder_count = set_folder_count_dict[per_set_name]
        for per_folder_index in range(per_set_folder_count):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            per_is_set_last_folder = (
                per_folder_index == per_set_folder_count - 1)
            folder_check_pair_list.append([
                per_set_name,
                per_folder_name,
                per_is_set_last_folder,
            ])

    total_image_count = 0
    check_func = partial(check_single_save_folder,
                         save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_func, folder_check_pair_list),
                                     total=len(folder_check_pair_list)):
            _, per_folder_image_count, per_check_error_message_list = per_check_result
            total_image_count += per_folder_image_count
            check_error_message_list.extend(per_check_error_message_list)

    print('3333', 'check total image:', total_image_count, 'check error:',
          len(check_error_message_list))

    return check_error_message_list, total_image_count


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path, SAVE_DATASET_DIR_NAME)
    os.makedirs(save_dataset_path, exist_ok=True)

    image_annotation_pair_list, annotation_file_count, archive_annotation_count_dict, total_annotation_count, load_annotation_failed_count, archive_name_not_match_count, invalid_judgment_pair_count, invalid_judgment_name_count_dict, same_caption_pair_count, missing_set_name_pair_count, set_name_not_match_pair_count, unknown_set_name_pair_count, set_pair_count_dict, set_image_count_dict, missing_image_count, invalid_image_name_count, empty_caption_count, too_short_caption_count, too_long_caption_count, set_text_valid_image_count_dict, invalid_image_count, image_name_resolution_not_match_count = get_all_image_annotation_pair(
        root_dataset_path)

    print('1111', annotation_file_count, total_annotation_count,
          load_annotation_failed_count, archive_name_not_match_count,
          invalid_judgment_pair_count, invalid_judgment_name_count_dict,
          same_caption_pair_count, missing_set_name_pair_count,
          set_name_not_match_pair_count, unknown_set_name_pair_count,
          set_pair_count_dict, set_image_count_dict, missing_image_count,
          invalid_image_name_count, empty_caption_count,
          too_short_caption_count, too_long_caption_count,
          set_text_valid_image_count_dict,
          invalid_image_count, image_name_resolution_not_match_count,
          len(image_annotation_pair_list))

    if len(image_annotation_pair_list) > 0:
        print('1111', image_annotation_pair_list[0])

    if load_annotation_failed_count > 0:
        raise Exception(
            f'load annotation failed count {load_annotation_failed_count}')

    invalid_count_dict = {
        'archive_name_not_match_count': archive_name_not_match_count,
        'missing_set_name_pair_count': missing_set_name_pair_count,
        'set_name_not_match_pair_count': set_name_not_match_pair_count,
        'unknown_set_name_pair_count': unknown_set_name_pair_count,
        'missing_image_count': missing_image_count,
        'invalid_image_name_count': invalid_image_name_count,
        'empty_caption_count': empty_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        annotation_file_count, archive_annotation_count_dict,
        total_annotation_count, invalid_judgment_pair_count,
        invalid_judgment_name_count_dict, same_caption_pair_count,
        set_pair_count_dict, set_image_count_dict,
        set_text_valid_image_count_dict, invalid_count_dict)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        raise Exception(
            f'load annotation check failed {load_annotation_check_error_message_list[:20]}'
        )

    image_annotation_pair_list, duplicate_save_image_name_list = get_deduplicated_image_annotation_pair(
        image_annotation_pair_list)

    print('1111', len(image_annotation_pair_list),
          len(duplicate_save_image_name_list))
    if len(duplicate_save_image_name_list) > 0:
        print('1111', duplicate_save_image_name_list[:10])

    # 归一化后的保存名实测100%唯一，撞名说明归一化写法被改动过，
    # 继续跑会让后写的图像覆盖先写的、静默丢样本
    if len(duplicate_save_image_name_list
           ) != EXPECTED_DUPLICATE_IMAGE_NAME_COUNT:
        raise Exception(
            f'duplicate save image name count not match {len(duplicate_save_image_name_list)} != {EXPECTED_DUPLICATE_IMAGE_NAME_COUNT} {duplicate_save_image_name_list[:10]}'
        )

    image_save_folder_pair_list, set_folder_count_dict = get_all_image_save_folder_pair(
        image_annotation_pair_list, save_dataset_path)

    total_save_task_image_count = sum([
        len(per_folder_save_pair_list)
        for _, _, per_folder_save_pair_list in image_save_folder_pair_list
    ])

    print('1111', len(image_save_folder_pair_list), len(set_folder_count_dict),
          total_save_task_image_count)
    if len(image_save_folder_pair_list) > 0:
        print('1111', image_save_folder_pair_list[0][0],
              image_save_folder_pair_list[0][1],
              image_save_folder_pair_list[0][2][0])

    # 切分之后保存图像名必须全局唯一，撞名会让后写的图像覆盖先写的、静默丢样本，
    # 所以这里再兜一道，撞上就直接中止
    save_image_name_dict = {}
    conflict_save_image_name_list = []
    for _, _, per_folder_save_pair_list in image_save_folder_pair_list:
        for _, per_save_image_name, _ in per_folder_save_pair_list:
            if per_save_image_name in save_image_name_dict:
                conflict_save_image_name_list.append(per_save_image_name)
                continue
            save_image_name_dict[per_save_image_name] = 1

    if len(conflict_save_image_name_list) > 0:
        raise Exception(
            f'conflict save image name num {len(conflict_save_image_name_list)} {conflict_save_image_name_list[:10]}'
        )

    save_image_name_dict = {}

    folder_image_count_dict = {}
    save_image_failed_count = 0
    process_func = partial(process_single_image_folder,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_save_result in tqdm(pool.imap_unordered(
                process_func, image_save_folder_pair_list),
                                    total=len(image_save_folder_pair_list)):
            per_set_name, per_folder_name, per_folder_image_count, per_save_image_failed_count = per_save_result
            folder_image_count_dict[per_folder_name] = per_folder_image_count
            save_image_failed_count += per_save_image_failed_count

            print('2222', per_folder_name, per_folder_image_count,
                  per_save_image_failed_count)

    total_save_image_count = sum(folder_image_count_dict.values())

    check_error_message_list, check_total_image_count = check_save_dataset(
        save_dataset_path, set_folder_count_dict)

    print('3333', 'total annotation file:', annotation_file_count,
          'total annotation:', total_annotation_count,
          'invalid judgment pair:', invalid_judgment_pair_count,
          'same caption pair:',
          same_caption_pair_count, 'total pair after skip:',
          sum(set_pair_count_dict.values()), 'total image after skip:',
          sum(set_image_count_dict.values()), 'missing image:',
          missing_image_count, 'invalid image name:', invalid_image_name_count,
          'empty caption:', empty_caption_count, 'too short caption:',
          too_short_caption_count, 'too long caption:', too_long_caption_count,
          'total text valid image:',
          sum(set_text_valid_image_count_dict.values()), 'invalid image:',
          invalid_image_count, 'image name resolution not match:',
          image_name_resolution_not_match_count, 'duplicate image name:',
          len(duplicate_save_image_name_list), 'save image failed:',
          save_image_failed_count, 'total save image:',
          total_save_image_count, 'total save folder:',
          len(folder_image_count_dict), 'total save set:',
          len(set_folder_count_dict),
          'check total image:', check_total_image_count, 'check error:',
          len(check_error_message_list))

    save_check_result_path = os.path.join(save_dataset_path,
                                          'resave_check_result.json')
    save_check_result_dict = {
        'total_annotation_file_count':
        annotation_file_count,
        'total_annotation_count':
        total_annotation_count,
        'load_annotation_failed_count':
        load_annotation_failed_count,
        'archive_name_not_match_count':
        archive_name_not_match_count,
        'invalid_judgment_pair_count':
        invalid_judgment_pair_count,
        'same_caption_pair_count':
        same_caption_pair_count,
        'missing_set_name_pair_count':
        missing_set_name_pair_count,
        'set_name_not_match_pair_count':
        set_name_not_match_pair_count,
        'unknown_set_name_pair_count':
        unknown_set_name_pair_count,
        'total_pair_count_after_skip':
        sum(set_pair_count_dict.values()),
        'total_image_count_after_skip':
        sum(set_image_count_dict.values()),
        'missing_image_count':
        missing_image_count,
        'invalid_image_name_count':
        invalid_image_name_count,
        'empty_caption_count':
        empty_caption_count,
        'too_short_caption_count':
        too_short_caption_count,
        'too_long_caption_count':
        too_long_caption_count,
        'total_text_valid_image_count':
        sum(set_text_valid_image_count_dict.values()),
        'invalid_image_count':
        invalid_image_count,
        'image_name_resolution_not_match_count':
        image_name_resolution_not_match_count,
        'duplicate_image_name_count':
        len(duplicate_save_image_name_list),
        'total_save_task_image_count':
        total_save_task_image_count,
        'save_image_failed_count':
        save_image_failed_count,
        'total_save_image_count':
        total_save_image_count,
        'total_save_folder_count':
        len(folder_image_count_dict),
        'total_save_set_count':
        len(set_folder_count_dict),
        'check_total_image_count':
        check_total_image_count,
        'check_error_count':
        len(check_error_message_list),
        'archive_annotation_count_dict':
        archive_annotation_count_dict,
        'invalid_judgment_name_count_dict':
        invalid_judgment_name_count_dict,
        'set_pair_count_dict':
        set_pair_count_dict,
        'set_image_count_dict':
        set_image_count_dict,
        'set_text_valid_image_count_dict':
        set_text_valid_image_count_dict,
        'set_folder_count_dict':
        set_folder_count_dict,
        'folder_image_count_dict':
        folder_image_count_dict,
        'duplicate_save_image_name_list':
        duplicate_save_image_name_list[:10000],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    if total_save_image_count != total_save_task_image_count:
        check_error_message_list.append(
            f'total save image count not match {total_save_image_count} != {total_save_task_image_count}'
        )
    if check_total_image_count != total_save_image_count:
        check_error_message_list.append(
            f'check total image count not match {check_total_image_count} != {total_save_image_count}'
        )
    if save_image_failed_count > 0:
        check_error_message_list.append(
            f'save image failed count {save_image_failed_count}')
    if len(set_folder_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'total save set count not match {len(set_folder_count_dict)} != {EXPECTED_SAVE_SET_COUNT}'
        )
    if len(check_error_message_list) > 0:
        # 收尾自校验不通过必须让上层感知，不能静默留下坏样本或不满的文件夹
        raise Exception(
            f'check save dataset error num {len(check_error_message_list)} {check_error_message_list[:10]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/UNO-1M'
    save_dataset_path = r'/root/autodl-tmp/t2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
