import os
import re
import gzip
import json
import glob
import shutil
import struct
import tarfile
import collections

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: UNO-1M(Less-to-More Generalization, bytedance-research)
#
# 【数据集类型】主体驱动生成(subject-driven generation)数据集，即
# "参考图 + 文本 -> 生成图"，下游走ti2i_dataset.py那条链路(参考图作为图像条件输入),
# **不是指令编辑(instruction-based image editing)数据集**: 全库没有任何编辑指令、
# 没有mask、两张图也不是"编辑前/编辑后"的同构关系，而是**同一个主体(subject)在
# 两个不同场景下各生成一张图**(UNO论文的in-context generation造数流程)。
# 每个样本对 = "img_path1 + img_path2 + 两条各自的caption + 主体词 + 一致性打分",
# 两张图地位对称: 训练时任取一张当参考图、另一张当生成目标即可(本脚本默认
# img_path1->参考图 / img_path2->生成目标，同时把两条caption都写进标注，
# 下游想反向用只需交换字段，不需要重新解压)。
# 也可以只取其中一张图 + 它自己的caption退化成纯t2i样本(README的task_categories
# 同时写了text-to-image和image-to-image)。
#
# 【root_dataset_path实测原始保存规格(共2.2T)】
# UNO-1M/
# ├── images/                     102个 split{1..102}.tar.gz (2.2T)
# │                               编号1..102**连号无缺号**，全部带gzip魔数1f 8b 08、
# │                               尾8字节gzip footer(CRC32+ISIZE)可解析;
# │                               每个tar内**没有任何目录成员**，成员全是
# │                               split{N}/<name>.png(实测split102: 19142个成员、
# │                               全部.png、无非普通文件、无重名);
# │                               文件名自带原始分辨率后缀(如..._793x1024.png)
# ├── labels/                     102个 split{1..102}.json (830MB)，与tar一一对应，
# │                               list结构，实测102个文件合计**1011093条样本对**
# │                               (单文件2080~10000条，split32=9442/split66=2080/
# │                                split102=9571，其余99个均为10000)
# ├── uno_1m_total_labels.json     848MB。实测它就是102个split json的**完全并集**:
# │                               条数1011093完全相同、(img_path1,img_path2)集合
# │                               双向差集均为0 => 纯冗余信息(无用，不拷贝，
# │                               否则白占848MB且和labels/互为重复真值)
# ├── assets/uno1m.webp           README里的示意图(无用)
# ├── README.md                   数据集说明(无用)
# ├── .gitattributes              git lfs配置(无用)
# └── .cache/                     huggingface下载缓存(无用，不整理进训练目录)，
#                                 里面残留14个images/*.incomplete，说明下载确实中断过,
#                                 所以解压前的完整性预检不是多余的;
#                                 但 .cache/huggingface/trees/<commit>.json 里记着
#                                 官方仓库全部208个文件的size与lfs_sha256，
#                                 实测本地102个tar.gz + 102个json + 4个根文件的字节数
#                                 与之**100%一致、0缺失**，是现成的完整性ground truth,
#                                 所以本脚本会顺手拿它做一道校验(缓存被删就只告警)
#
# 【单条标注的全部字段(实测抽4个split共39571条，字段100%齐备)】
#   img_path1 / img_path2 : "split{N}/xxx.png"，同一主体的两张图。
#                           实测全库2022186个路径**全部唯一**(=1011093*2，无复用),
#                           且img_path1 != img_path2                    -> 有用(参考图/生成图，必需)
#   caption.img_path1     : img_path1的英文caption，实测0条为空          -> 有用(必需)
#   caption.img_path2     : img_path2的英文caption，实测0条为空          -> 有用(必需)
#   caption.subject       : 主体词列表(1~5个，绝大多数1个)              -> 有用(主体驱动训练/按主体去重)
#   caption.judgment      : "same"/"yes"，VLM对"两图是否同一主体"的判定  -> 有用(配对可信度)
#   vlm_filter_cot.score_part  : 各细粒度维度的一致性分(1~20个维度)      -> 有用(奖励模型/细粒度过滤)
#   vlm_filter_cot.score_final : 0.0~4.0的最终一致性分。README明确建议
#                           主体驱动训练取>=3.5，UNO论文只用满分4.0      -> 有用(**过滤必需，丢了没法复现论文**)
#   生成图本体            : tar里与img_pathN同名的png                    -> 有用(必需)
#
#   【打分字段的实测脏数据(全库扫过1011093条)】上游VLM的CoT输出偶发解析污染:
#     split6 第1295条 : score_part={"Overall Shape and Form":4.0,
#                       "Record Number":107739.0,"ID":285811.0},
#                       score_final=131184.66666666666(就是这三个数的均值)
#     split10第6731条 : score_part里混进了"# Output":8.5(score_final=3.5仍在范围内)
#   也就是把"Record Number"/"ID"/"# Output"这类根本不是打分维度的数字当成了细粒度分。
#   这两条的两张图/两条caption/subject/judgment**全部完好**，是可训练样本对,
#   所以本脚本**不丢样本对**，只把分数标记成不可用(score_final置-1.0、
#   score_final_valid=False、原值留在score_final_raw)，详见
#   ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG。
#   注意: 全库还有约40.8万条score_final != mean(score_part)，那是上游取整/加权口径
#   问题、分数本身在0.0~4.0内可用，本脚本不做这项一致性校验(校了会误判40万条)。
#
# 无用信息: .cache/ + .gitattributes + README.md + assets/ +
#           uno_1m_total_labels.json(与labels/完全重复)
#
# 【样本对对账基线(必须严格闭合，否则说明漏了样本对)】
#   单个split: tar文件成员数 == json引用的图像路径数 == 2 * 有效样本对数
#              (实测split102: 19142 == 2*9571 ✓;
#               split1抽查前3000个成员100%命中json引用集合、0个孤立成员)
#   全库    : 102个tar / 2022186张png / 1011093个样本对，
#             每个split的条数还硬校验到EXPECTED_ARCHIVE_SAMPLE_PAIR_COUNT_DICT。
#   注意本数据集**不做任何分数过滤**: score_final<3.5的样本对信息一样完整,
#   过滤是训练时按score_final现取的事，解压阶段少存一条就是永久丢数据。
#
# 【本脚本的处理口径】
# - 解压前预检(硬失败): 根目录条目白名单(出现新文件必须显式感知)、
#   images/labels各102个且split编号1..102连号、两侧同名一一对应、
#   每个tar.gz的gzip魔数与尾部footer可解析(O(1)拦住下载截断)、
#   用.cache/huggingface/trees的size清单核对每个文件字节数(缓存缺失只告警)、
#   并行解析全部102个labels json逐条校验字段齐备性并核对条数(总数必须==1011093);
# - 解压**故意不用tarfile的mode='r|gz'**: tarfile自带的gz解压走内部_Stream，
#   只调zlib.decompressobj、**不校验gzip尾部的CRC32与ISIZE**，tar被截断时很可能
#   只是"少解出一批图"然后静默正常结束。改成 gzip.GzipFile -> tarfile.open(mode='r|'),
#   tar成员读完后再把GzipFile drain到EOF，CRC32/ISIZE不符会直接抛BadGzipFile;
# - 并行单位 = 102个split(每个tar是独立完整的gzip流，可以按tar并行);
# - 每个成员逐项处理: 路径归一化 + '..'越界拦截、非普通文件上报、
#   非.png后缀上报、tar内重名成员改写到unzip_duplicate_members/并上报(绝不覆盖)、
#   **json没引用到的成员照样落盘**并记入unreferenced清单(不能因为标注没写就丢图)、
#   写盘后立即校验落盘大小 == tar头里的大小、已存在且大小一致则跳过(支持断点续跑);
# - 汇总标注落 unzip_annotations/split{N}.jsonl，每行一个完整样本对，
#   保留原标注全部属性(两条caption/subject/judgment/score_part/score_final),
#   并补上参考图与生成图的落盘路径、sample_key、所属tar;
#   labels/*.json原样拷到 unzip_source_annotations/ 作溯源真值;
# - 每个split严格对账: extract+skip+not_save+fail == tar成员总数、
#   有效样本对数 == json条数 == tar成员数/2 == 该split的实测ground truth,
#   且必须读到tar流末尾与gzip流末尾;
# - **打分被上游CoT污染的样本对不算错误**: 图和caption都在、样本对照常落盘并计入
#   对账，只把score_final置成-1.0(不可用哨兵值) + score_final_valid=False标记,
#   并以告警形式汇总到abnormal_score_*字段。这样既不永久丢数据、也不会让假分数
#   混进score_final>=3.5的高分子集(详见ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG);
# - 任何一环出错(拷贝/解压/校验)都汇总后抛异常，绝不静默少样本对。
# ==============================================================================

DATASET_TASK_TYPE = 'subject_driven_generation'

# 无用信息，不整理进训练目录:
# .cache/                    huggingface下载缓存(含14个残留*.incomplete)
# .gitattributes             git lfs配置
# README.md                  数据集说明
# assets/                    README里的示意图(uno1m.webp)
# uno_1m_total_labels.json   848MB，实测是labels/下102个json的完全并集(条数与
#                            (img_path1,img_path2)集合都完全一致)，纯冗余
# .DS_Store/CACHEDIR.TAG     目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    'assets',
    'uno_1m_total_labels.json',
    '.DS_Store',
    'CACHEDIR.TAG',
]

# 过滤掉无用信息后根目录只应该剩这两个目录
EXPECTED_ROOT_DIR_NAME_LIST = [
    'images',
    'labels',
]

LOAD_ARCHIVE_ROOT_DIR_NAME = 'images'

LOAD_ANNOTATION_ROOT_DIR_NAME = 'labels'

ARCHIVE_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>split(?P<index>\d+))\.tar\.gz$')

ANNOTATION_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>split(?P<index>\d+))\.json$')

# tar内成员名必须是 split{N}/<name>.png，实测无其他后缀
IMAGE_FILE_SUFFIX_LIST = [
    '.png',
]

# 单条标注的顶层必需字段
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    'img_path1',
    'img_path2',
    'caption',
    'vlm_filter_cot',
]

# 单条标注里两张图的路径字段名(同时也是caption字典里的两个caption字段名)
ANNOTATION_IMAGE_KEY_NAME_LIST = [
    'img_path1',
    'img_path2',
]

# caption字典的必需字段(img_path1/img_path2是两张图各自的caption)
ANNOTATION_CAPTION_EXPECTED_KEY_NAME_LIST = [
    'img_path1',
    'img_path2',
    'judgment',
    'subject',
]

# vlm_filter_cot字典的必需字段
ANNOTATION_FILTER_EXPECTED_KEY_NAME_LIST = [
    'score_part',
    'score_final',
]

# 实测score_final取值范围(0.0~4.0)，越界说明上游打分被污染，必须显式感知
ANNOTATION_SCORE_FINAL_RANGE = [0.0, 4.0]

# score_part里每个细粒度维度的取值范围，与score_final同一量纲
ANNOTATION_SCORE_PART_RANGE = [0.0, 4.0]

# 打分异常的样本对是"整条丢掉"还是"保留样本对 + 把分数标记成不可用"。
#
# 默认True(保留 + 标记)。实测全库1011093条里有极少数条目的打分被上游VLM的
# CoT输出解析污染了，例如:
#   split6 第1295条: score_part={"Overall Shape and Form":4.0,
#                    "Record Number":107739.0,"ID":285811.0},
#                    score_final=131184.66666666666(=这三个数的均值)
#   split10第6731条: score_part里混进了"# Output":8.5(score_final=3.5还在范围内)
# 也就是上游把"Record Number"/"ID"/"# Output"这类**根本不是打分维度**的数字
# 当成细粒度分再取了均值。这类条目的两张图/两条caption/subject/judgment
# **全部完好**，是可训练的样本对，所以:
#   - 不能整条丢掉: 丢了就永久少数据，而且会让"有效样本对数 == json条数 ==
#     tar成员数/2"这三条硬对账同时崩掉(报错就是这么来的);
#   - 也绝不能把假分数当真: 131184.67 >= 3.5 会让它混进"推荐训练集",
#     污染按score_final筛出来的高分子集。
# 所以口径是: 样本对照常保留并落盘，但把score_final置成
# SCORE_FINAL_INVALID_VALUE(-1.0，任何>=阈值的筛选都命中不了),
# 原始值留在score_final_raw里溯源，再用score_final_valid=False显式标记,
# 并把这些sample_key统计进校验报告的abnormal_score_*字段里(告警，不是硬失败)。
# 改成False则回到"整条丢弃"的老口径，此时必须同步下调
# EXPECTED_ARCHIVE_SAMPLE_PAIR_COUNT_DICT，否则对账必然失败。
ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG = True

# 打分不可用时写进标注的哨兵值。取负数是为了让
# score_final >= RECOMMEND_SCORE_FINAL_THRESHOLD 这类筛选天然排除它，
# 同时在score_final_count_dict里留下一个显式的"-1.0"桶便于一眼看出有几条
SCORE_FINAL_INVALID_VALUE = -1.0

# README建议的主体驱动训练分数门槛与UNO论文口径，
# 只用来统计可用样本量，**解压阶段不做任何过滤**
RECOMMEND_SCORE_FINAL_THRESHOLD = 3.5

PERFECT_SCORE_FINAL_THRESHOLD = 4.0

# 实测split编号1..102连号，数量不对说明下载不全
EXPECTED_ARCHIVE_NUM = 102

EXPECTED_ARCHIVE_INDEX_LIST = list(range(1, EXPECTED_ARCHIVE_NUM + 1))

# 实测每个split的labels json条数(即样本对数)，直接当完整性ground truth，
# 少一条都说明标注被改过或没下全
EXPECTED_ARCHIVE_SAMPLE_PAIR_COUNT_DICT = {
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

# 实测全库样本对总数与图像总数(每个样本对固定2张图，且全库图像路径无复用)
EXPECTED_TOTAL_SAMPLE_PAIR_COUNT = sum(
    EXPECTED_ARCHIVE_SAMPLE_PAIR_COUNT_DICT.values())

EXPECTED_TOTAL_IMAGE_COUNT = EXPECTED_TOTAL_SAMPLE_PAIR_COUNT * 2

# huggingface下载缓存里的文件清单目录(无用信息，但里面的size是现成的完整性真值)
HUGGINGFACE_TREE_FILE_GLOB_PATTERN = '.cache/huggingface/trees/*.json'

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_SOURCE_ANNOTATION_DIR_NAME = 'unzip_source_annotations'

SAVE_DUPLICATE_MEMBER_DIR_NAME = 'unzip_duplicate_members'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

# 图像成员是否落盘。
# True : 和其他数据集脚本口径一致，输出目录自包含，2022186张png约2.2T、
#        约200万个inode，务必确认目标盘扛得住再跑;
# False: 只解析labels生成 unzip_annotations/*.jsonl 索引(几分钟就能跑完),
#        图像继续留在原tar.gz里按webdataset方式顺序读，样本对信息一样完整。
EXTRACT_IMAGE_FILE_FLAG = True

# labels/*.json(830MB)是否原样拷到unzip_source_annotations/作溯源真值
SAVE_SOURCE_ANNOTATION_FLAG = True

# 是否在解压后再os.walk一遍输出目录做二次对账。
# 默认False: 200万个小文件的os.walk在NAS上要跑很久，而解压时已经做了
# "写盘后立刻校验落盘大小 == tar头大小" + "extract+skip+not_save+fail == tar成员总数"
# + "有效样本对数 == json条数 == tar成员数/2"三道对账，已经能保证不漏样本对。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024

GZIP_MAGIC_BYTES = b'\x1f\x8b\x08'

GZIP_FOOTER_SIZE = 8

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

MAX_PRINT_PROBLEM_ITEM_NUM = 20


class MultiPartArchiveReader:
    """把按字节切分的多个分片压缩包拼接成一个只读的连续字节流

    该数据集每个split{N}.tar.gz都是独立完整的gzip流(不是分卷)，
    这里保留多分片能力只是为了和其他数据集脚本口径一致，
    一旦上游改成按字节切分发布也不用改解压主流程。
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
    """过滤掉.cache、README.md、assets、uno_1m_total_labels.json等无用文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """判断是否是图像文件后缀"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


def get_normalize_member_name(per_member_name):
    """把tar成员名统一成正斜杠相对路径，越界(..)时返回空串"""
    per_member_name = per_member_name.replace('\\', '/').lstrip('/')
    per_member_name = os.path.normpath(per_member_name).replace('\\', '/')
    if per_member_name.startswith('..') or per_member_name in ['.', '']:
        return ''

    return per_member_name


def check_single_archive_gzip_head_and_tail(per_archive_path):
    """O(1)预检单个tar.gz是否被截断: 头部必须有gzip魔数，尾部必须能读出8字节footer

    footer是"CRC32 + ISIZE(解压后字节数 mod 2^32)"，这里只校验"读得到、能解析",
    真正的CRC32/ISIZE比对要等解压时由gzip.GzipFile读到流末尾去做(见
    process_single_archive_group的drain逻辑)。
    实测102个tar.gz全部满足，一旦某个包下载中断就必须在跑几十小时解压前拦住。
    """
    error_message_list = []
    try:
        per_archive_size = os.path.getsize(per_archive_path)
        if per_archive_size <= 0:
            error_message_list.append(f'empty archive {per_archive_path}')

            return error_message_list

        if per_archive_size < len(GZIP_MAGIC_BYTES) + GZIP_FOOTER_SIZE:
            error_message_list.append(
                f'archive size too small {per_archive_path} {per_archive_size}'
            )

            return error_message_list

        with open(per_archive_path, 'rb') as load_archive_file:
            per_archive_head_bytes = load_archive_file.read(
                len(GZIP_MAGIC_BYTES))
            load_archive_file.seek(per_archive_size - GZIP_FOOTER_SIZE)
            per_archive_tail_bytes = load_archive_file.read(GZIP_FOOTER_SIZE)

        if per_archive_head_bytes != GZIP_MAGIC_BYTES:
            error_message_list.append(
                f'archive gzip magic broken {per_archive_path} {per_archive_head_bytes}'
            )

        if len(per_archive_tail_bytes) != GZIP_FOOTER_SIZE:
            error_message_list.append(
                f'read archive gzip footer failed {per_archive_path}')

            return error_message_list

        # 只要能解出来就行，具体数值等解压时用GzipFile逐流校验
        struct.unpack('<II', per_archive_tail_bytes)
    except Exception as e:
        error_message_list.append(
            f'check archive gzip head and tail failed {per_archive_path} {e}')

    return error_message_list


def check_single_annotation_file_magic(per_annotation_path):
    """O(1)预检单个labels json是否被截断: 必须非空，且首尾字符是list的[]"""
    error_message_list = []
    try:
        per_annotation_size = os.path.getsize(per_annotation_path)
        if per_annotation_size <= 0:
            error_message_list.append(
                f'empty annotation file {per_annotation_path}')

            return error_message_list

        with open(per_annotation_path, 'rb') as load_annotation_file:
            per_head_bytes = load_annotation_file.read(1)
            load_annotation_file.seek(per_annotation_size - 1)
            per_tail_bytes = load_annotation_file.read(1)

        if per_head_bytes != b'[' or per_tail_bytes != b']':
            error_message_list.append(
                f'annotation file not a json list(truncated file) {per_annotation_path} {per_head_bytes} {per_tail_bytes}'
            )
    except Exception as e:
        error_message_list.append(
            f'check annotation file magic failed {per_annotation_path} {e}')

    return error_message_list


def check_huggingface_tree_file(root_dataset_path):
    """用huggingface下载缓存里的文件清单核对每个文件的字节数

    .cache/本身是无用信息(不整理进训练目录)，但
    .cache/huggingface/trees/<commit>.json 里记着官方仓库全部208个文件的
    size与lfs_sha256，是现成的完整性ground truth，实测本地文件字节数与之100%一致。
    缓存被删掉时拿不到这份真值，只告警不判失败(文件数/连号/gzip头尾那几道预检还在)。

    返回[错误信息列表, 告警信息列表]。
    """
    error_message_list, warning_message_list = [], []

    tree_file_path_list = sorted(
        glob.glob(
            os.path.join(root_dataset_path,
                         HUGGINGFACE_TREE_FILE_GLOB_PATTERN)))
    if len(tree_file_path_list) == 0:
        warning_message_list.append(
            'huggingface tree file not found, skip file size ground truth check'
        )

        return error_message_list, warning_message_list

    tree_file_size_dict = {}
    for per_tree_file_path in tree_file_path_list:
        try:
            with open(per_tree_file_path, 'r',
                      encoding='UTF-8') as load_json_file:
                per_tree_dict = json.load(load_json_file)
        except Exception as e:
            warning_message_list.append(
                f'load huggingface tree file failed {per_tree_file_path} {e}')
            continue

        for per_file_relative_path, per_file_meta_dict in per_tree_dict.get(
                'files', {}).items():
            per_file_size = per_file_meta_dict.get('size', None)
            if not isinstance(per_file_size, int):
                continue
            tree_file_size_dict[per_file_relative_path.replace(
                '\\', '/')] = per_file_size

    check_file_count = 0
    for per_file_relative_path, per_file_size in sorted(
            tree_file_size_dict.items()):
        # 只核对有用信息(images/与labels/)，无用信息的字节数不关心
        if check_skip_file_or_dir(per_file_relative_path):
            continue

        per_file_path = os.path.join(root_dataset_path, per_file_relative_path)
        if not os.path.exists(per_file_path):
            error_message_list.append(
                f'file in huggingface tree not exist {per_file_relative_path}')
            continue

        check_file_count += 1
        per_local_file_size = os.path.getsize(per_file_path)
        if per_local_file_size != per_file_size:
            error_message_list.append(
                f'file size not match huggingface tree {per_file_relative_path} {per_local_file_size} != {per_file_size}'
            )

    print('1111', 'check huggingface tree file size:', check_file_count,
          'tree file:', len(tree_file_path_list))

    return error_message_list, warning_message_list


def get_single_annotation_error_message_list(per_annotation, per_archive_name,
                                             per_sample_key):
    """校验单条标注的有用信息是否完整，返回[整理后的样本对, 错误信息列表, 打分异常信息列表]

    完整有用信息 = 两张图路径(合法且互不相同) + 两条非空caption + 主体词 +
    judgment + score_part + score_final。
    任何一项不过都返回空样本对并把原因上报，**绝不静默丢样本对**。

    唯一的例外是打分(score_final/score_part)本身被上游CoT污染:
    它不影响样本对能不能训练(图和caption都在)，只影响"按分数筛子集"这件事,
    所以按ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG的说明降级成
    "保留样本对 + 把分数标记成不可用 + 单独回传打分异常清单(告警)"，
    不再把整条样本对连图带文一起丢掉。
    """
    error_message_list, abnormal_score_message_list = [], []

    if not isinstance(per_annotation, dict):
        error_message_list.append(f'{per_sample_key} annotation not a dict')

        return None, error_message_list, abnormal_score_message_list

    for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            error_message_list.append(
                f'{per_sample_key} miss key {per_key_name}')

    per_caption_dict = per_annotation.get('caption', None)
    per_filter_dict = per_annotation.get('vlm_filter_cot', None)
    if not isinstance(per_caption_dict, dict):
        error_message_list.append(f'{per_sample_key} caption not a dict')
        per_caption_dict = {}
    if not isinstance(per_filter_dict, dict):
        error_message_list.append(
            f'{per_sample_key} vlm_filter_cot not a dict')
        per_filter_dict = {}

    for per_key_name in ANNOTATION_CAPTION_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_caption_dict:
            error_message_list.append(
                f'{per_sample_key} caption miss key {per_key_name}')
    for per_key_name in ANNOTATION_FILTER_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_filter_dict:
            error_message_list.append(
                f'{per_sample_key} vlm_filter_cot miss key {per_key_name}')

    # 两张图的相对路径: 必须是 split{N}/xxx.png，且前缀split名要和所属tar一致,
    # 否则说明labels和tar错位(照原路径落盘会串到别的split目录里)
    image_relative_path_list, image_caption_list = [], []
    for per_image_key_name in ANNOTATION_IMAGE_KEY_NAME_LIST:
        per_image_relative_path = per_annotation.get(per_image_key_name, '')
        if not isinstance(per_image_relative_path,
                          str) or not per_image_relative_path:
            error_message_list.append(
                f'{per_sample_key} empty {per_image_key_name}')
            per_image_relative_path = ''
        else:
            per_image_relative_path = per_image_relative_path.replace(
                '\\', '/').lstrip('/')
            if not per_image_relative_path.startswith(f'{per_archive_name}/'):
                error_message_list.append(
                    f'{per_sample_key} {per_image_key_name} archive name not match {per_image_relative_path}'
                )
                per_image_relative_path = ''
            elif not check_image_file_suffix(per_image_relative_path):
                error_message_list.append(
                    f'{per_sample_key} {per_image_key_name} unknown image suffix {per_image_relative_path}'
                )
                per_image_relative_path = ''

        image_relative_path_list.append(per_image_relative_path)

        # caption字典里两条caption的key就是img_path1/img_path2这两个字段名本身
        per_image_caption = per_caption_dict.get(per_image_key_name, '')
        if not isinstance(per_image_caption, str) or len(
                per_image_caption.strip()) == 0:
            error_message_list.append(
                f'{per_sample_key} empty caption {per_image_key_name}')
            per_image_caption = ''

        image_caption_list.append(per_image_caption.strip())

    if image_relative_path_list[0] and image_relative_path_list[
            0] == image_relative_path_list[1]:
        # 实测全库img_path1 != img_path2，相同说明这条标注废了(参考图和目标图是同一张)
        error_message_list.append(
            f'{per_sample_key} img_path1 == img_path2 {image_relative_path_list[0]}'
        )
        image_relative_path_list = ['', '']

    per_subject_list = per_caption_dict.get('subject', None)
    if not isinstance(per_subject_list, list) or len(per_subject_list) == 0:
        error_message_list.append(f'{per_sample_key} empty subject')
        per_subject_list = []

    per_judgment = per_caption_dict.get('judgment', '')
    if not isinstance(per_judgment, str) or len(per_judgment.strip()) == 0:
        error_message_list.append(f'{per_sample_key} empty judgment')
        per_judgment = ''

    # score_part: 空字典说明这条根本没打分(硬失败);
    # 有维度但某个维度的分越界说明上游把非打分字段(如"# Output")当成了细粒度分,
    # 这是打分污染，按开关降级成标记(见ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG)
    per_score_part_valid = True
    per_score_part_dict = per_filter_dict.get('score_part', None)
    if not isinstance(per_score_part_dict,
                      dict) or len(per_score_part_dict) == 0:
        error_message_list.append(f'{per_sample_key} empty score_part')
        per_score_part_dict = {}
        per_score_part_valid = False
    else:
        per_abnormal_score_part_key_list = []
        for per_score_part_key, per_score_part_value in per_score_part_dict.items(
        ):
            if not isinstance(per_score_part_value,
                              (int, float)) or isinstance(
                                  per_score_part_value, bool):
                per_abnormal_score_part_key_list.append(per_score_part_key)
                continue
            if not ANNOTATION_SCORE_PART_RANGE[0] <= float(
                    per_score_part_value) <= ANNOTATION_SCORE_PART_RANGE[1]:
                per_abnormal_score_part_key_list.append(per_score_part_key)

        if len(per_abnormal_score_part_key_list) > 0:
            per_score_part_valid = False
            per_abnormal_message = f'{per_sample_key} score_part out of range {per_abnormal_score_part_key_list[:5]}'
            if ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG:
                abnormal_score_message_list.append(per_abnormal_message)
            else:
                error_message_list.append(per_abnormal_message)

    # score_final: 不是数字或不在0.0~4.0都说明这个分不可用。
    # 它不影响样本对能不能训练(图和caption都在)，只影响"按分数筛子集",
    # 所以默认降级成"保留样本对 + score_final置SCORE_FINAL_INVALID_VALUE + 标记"
    per_score_final_valid = True
    per_raw_score_final = per_filter_dict.get('score_final', None)
    per_score_final = per_raw_score_final
    if not isinstance(per_score_final,
                      (int, float)) or isinstance(per_score_final, bool):
        per_score_final_valid = False
        per_abnormal_message = f'{per_sample_key} score_final not a number {per_raw_score_final}'
        if ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG:
            abnormal_score_message_list.append(per_abnormal_message)
        else:
            error_message_list.append(per_abnormal_message)
    elif not ANNOTATION_SCORE_FINAL_RANGE[0] <= float(
            per_score_final) <= ANNOTATION_SCORE_FINAL_RANGE[1]:
        per_score_final_valid = False
        per_abnormal_message = f'{per_sample_key} score_final out of range {per_raw_score_final}'
        if ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG:
            abnormal_score_message_list.append(per_abnormal_message)
        else:
            error_message_list.append(per_abnormal_message)

    if len(error_message_list) > 0:
        return None, error_message_list, abnormal_score_message_list

    # 分数不可用时写哨兵值，原始值另存到score_final_raw供溯源，信息不丢
    per_save_score_final = float(
        per_score_final
    ) if per_score_final_valid else SCORE_FINAL_INVALID_VALUE

    # 完整有用信息的样本对: 两张图地位对称，默认img_path1当参考图、img_path2当生成目标,
    # 同时把两条caption都写进标注，下游想反向用只需交换字段，不用重新解压
    per_sample_pair = {
        'dataset_task_type': DATASET_TASK_TYPE,
        'sample_key': per_sample_key,
        'archive_name': per_archive_name,
        'reference_image_relative_path': image_relative_path_list[0],
        'target_image_relative_path': image_relative_path_list[1],
        'reference_image_caption': image_caption_list[0],
        'target_image_caption': image_caption_list[1],
        # 训练文本 = 生成目标图的caption(参考图只提供主体外观)
        'caption': image_caption_list[1],
        'subject_list': per_subject_list,
        'judgment': per_judgment.strip(),
        # 分数不可用时这里是SCORE_FINAL_INVALID_VALUE(-1.0)，
        # 下游按score_final >= 3.5/4.0筛选时天然排除，不会被假分数污染
        'score_final': per_save_score_final,
        # 打分是否可信的显式标记 + 原始值，两个字段都留着方便溯源与统计
        'score_final_valid': per_score_final_valid,
        'score_final_raw': per_raw_score_final,
        'score_part': per_score_part_dict,
        'score_part_valid': per_score_part_valid,
    }

    # 上游万一新增字段也一并保留，避免"有用信息没被完整保存下来"
    for per_annotation_key, per_annotation_value in per_annotation.items():
        if per_annotation_key in ANNOTATION_EXPECTED_KEY_NAME_LIST:
            continue
        per_sample_pair[per_annotation_key] = per_annotation_value

    return per_sample_pair, error_message_list, abnormal_score_message_list


def load_single_archive_annotation(annotation_load_pair):
    """加载并逐条校验单个split的labels json

    返回:
      per_archive_name              该split名(=tar名前缀)
      sample_pair_list              完整有用信息的样本对列表(按json原顺序)
      image_relative_path_dict      图像相对路径 -> [[样本下标, 图像角色], ...]
      annotation_sample_count       json里的原始条数(对账基线)
      invalid_annotation_message_list 被隔离的样本对原因清单
      abnormal_score_message_list   打分被上游污染但样本对照常保留的告警清单
      error_message_list            读文件/json结构级错误
    """
    per_archive_name, per_annotation_path = annotation_load_pair

    error_message_list, invalid_annotation_message_list = [], []
    abnormal_score_message_list = []
    sample_pair_list, image_relative_path_dict = [], {}
    annotation_sample_count = 0

    per_annotation_list = None
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_json_file:
            per_annotation_list = json.load(load_json_file)
    except Exception as e:
        error_message_list.append(
            f'{per_archive_name} load annotation failed {e}')

    if per_annotation_list is None:
        return {
            'archive_name': per_archive_name,
            'sample_pair_list': sample_pair_list,
            'image_relative_path_dict': image_relative_path_dict,
            'annotation_sample_count': annotation_sample_count,
            'invalid_annotation_message_list': invalid_annotation_message_list,
            'abnormal_score_message_list': abnormal_score_message_list,
            'error_message_list': error_message_list,
        }

    if not isinstance(per_annotation_list, list):
        error_message_list.append(
            f'{per_archive_name} annotation not a list {type(per_annotation_list)}'
        )

        return {
            'archive_name': per_archive_name,
            'sample_pair_list': sample_pair_list,
            'image_relative_path_dict': image_relative_path_dict,
            'annotation_sample_count': annotation_sample_count,
            'invalid_annotation_message_list': invalid_annotation_message_list,
            'abnormal_score_message_list': abnormal_score_message_list,
            'error_message_list': error_message_list,
        }

    annotation_sample_count = len(per_annotation_list)

    per_expected_sample_pair_count = EXPECTED_ARCHIVE_SAMPLE_PAIR_COUNT_DICT.get(
        per_archive_name, None)
    if per_expected_sample_pair_count is None:
        error_message_list.append(
            f'{per_archive_name} unknown archive name(no expected sample pair count)'
        )
    elif annotation_sample_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_name} annotation sample count not match {annotation_sample_count} != {per_expected_sample_pair_count}'
        )

    for per_sample_index, per_annotation in enumerate(per_annotation_list):
        # sample_key用"split名 + json内下标"生成: 图像文件名太长且没有天然pair id,
        # 而json顺序是固定的，这样每个样本对都有确定、可复现的唯一id
        per_sample_key = f'{per_archive_name}_{per_sample_index:06d}'

        per_sample_pair, per_annotation_error_message_list, per_abnormal_score_message_list = get_single_annotation_error_message_list(
            per_annotation, per_archive_name, per_sample_key)

        # 打分被上游污染的样本对照常保留(分数已被标记成不可用)，只把原因记进告警清单
        abnormal_score_message_list.extend(per_abnormal_score_message_list)

        if per_sample_pair is None:
            invalid_annotation_message_list.extend(
                per_annotation_error_message_list)
            continue

        per_current_sample_index = len(sample_pair_list)
        per_duplicate_image_flag = False
        for per_image_key_name, per_image_relative_path in zip(
                ANNOTATION_IMAGE_KEY_NAME_LIST, [
                    per_sample_pair['reference_image_relative_path'],
                    per_sample_pair['target_image_relative_path'],
                ]):
            if per_image_relative_path in image_relative_path_dict:
                # 实测全库图像路径无复用，一旦复用就说明标注有重复条目，
                # 必须上报(否则同一张图被当成两个样本对的目标图，对账数会对不上)
                per_duplicate_image_flag = True
                invalid_annotation_message_list.append(
                    f'{per_sample_key} duplicate image relative path {per_image_key_name} {per_image_relative_path}'
                )

        if per_duplicate_image_flag:
            continue

        for per_image_key_name, per_image_relative_path in zip(
                ANNOTATION_IMAGE_KEY_NAME_LIST, [
                    per_sample_pair['reference_image_relative_path'],
                    per_sample_pair['target_image_relative_path'],
                ]):
            image_relative_path_dict[per_image_relative_path] = [
                per_current_sample_index,
                per_image_key_name,
            ]

        sample_pair_list.append(per_sample_pair)

    return {
        'archive_name': per_archive_name,
        'sample_pair_list': sample_pair_list,
        'image_relative_path_dict': image_relative_path_dict,
        'annotation_sample_count': annotation_sample_count,
        'invalid_annotation_message_list': invalid_annotation_message_list,
        'abnormal_score_message_list': abnormal_score_message_list,
        'error_message_list': error_message_list,
    }


def check_single_archive_annotation(annotation_load_pair):
    """解压前预检用: 只把单个split的标注校验结论带回主进程(不带回样本对数据本身)

    labels/共830MB，102个文件全解析出来的样本对数据有几GB，
    Pool回传会白白撑爆主进程内存，所以这里只回传计数与问题清单。
    """
    per_annotation_result = load_single_archive_annotation(
        annotation_load_pair)

    return [
        per_annotation_result['archive_name'],
        per_annotation_result['annotation_sample_count'],
        len(per_annotation_result['sample_pair_list']),
        len(per_annotation_result['image_relative_path_dict']),
        per_annotation_result['invalid_annotation_message_list']
        [:MAX_SAVE_PROBLEM_ITEM_NUM],
        per_annotation_result['abnormal_score_message_list']
        [:MAX_SAVE_PROBLEM_ITEM_NUM],
        per_annotation_result['error_message_list']
        [:MAX_SAVE_PROBLEM_ITEM_NUM],
    ]


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集要原样拷贝的文件列表和按split归组后的压缩包列表

    返回:
      file_copy_pair_list  [相对路径, 源文件路径]，这里只有labels/*.json(溯源真值)
      archive_group_list   [split名, split编号, 压缩包分片路径列表, labels json路径]
    """
    file_copy_pair_list = []
    archive_path_dict, annotation_path_dict = {}, {}
    error_message_list = []

    root_archive_path = os.path.join(root_dataset_path,
                                     LOAD_ARCHIVE_ROOT_DIR_NAME)
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_ROOT_DIR_NAME)

    # 这一步在预检之前跑，两个目录缺失时直接返回空，让预检去统一报错，
    # 不要在这里因为os.listdir抛异常而丢掉其他预检信息
    if not os.path.exists(root_archive_path):
        error_message_list.append(
            f'archive dir not exist {LOAD_ARCHIVE_ROOT_DIR_NAME}')
    if not os.path.exists(root_annotation_path):
        error_message_list.append(
            f'annotation dir not exist {LOAD_ANNOTATION_ROOT_DIR_NAME}')
    if len(error_message_list) > 0:
        return [], [], error_message_list

    for per_file_name in sorted(os.listdir(root_archive_path)):
        if check_skip_file_or_dir(per_file_name):
            continue

        per_match_result = ARCHIVE_FILE_NAME_PATTERN.match(per_file_name)
        if not per_match_result:
            error_message_list.append(
                f'unknown file in archive dir {LOAD_ARCHIVE_ROOT_DIR_NAME}/{per_file_name}'
            )
            continue

        per_archive_name = per_match_result.group('prefix')
        per_archive_index = int(per_match_result.group('index'))
        if per_archive_name in archive_path_dict:
            error_message_list.append(
                f'duplicate archive name {per_archive_name}')
            continue

        archive_path_dict[per_archive_name] = [
            per_archive_index,
            os.path.join(root_archive_path, per_file_name),
        ]

    for per_file_name in sorted(os.listdir(root_annotation_path)):
        if check_skip_file_or_dir(per_file_name):
            continue

        per_match_result = ANNOTATION_FILE_NAME_PATTERN.match(per_file_name)
        if not per_match_result:
            error_message_list.append(
                f'unknown file in annotation dir {LOAD_ANNOTATION_ROOT_DIR_NAME}/{per_file_name}'
            )
            continue

        per_archive_name = per_match_result.group('prefix')
        if per_archive_name in annotation_path_dict:
            error_message_list.append(
                f'duplicate annotation name {per_archive_name}')
            continue

        per_annotation_path = os.path.join(root_annotation_path, per_file_name)
        annotation_path_dict[per_archive_name] = per_annotation_path

        if SAVE_SOURCE_ANNOTATION_FLAG:
            file_copy_pair_list.append([
                f'{SAVE_SOURCE_ANNOTATION_DIR_NAME}/{per_file_name}',
                per_annotation_path,
            ])

    # images/与labels/必须一一对应: 少了json拿不到caption，少了tar拿不到图，
    # 两种情况都会整批丢样本对，必须显式上报
    for per_archive_name in sorted(archive_path_dict.keys()):
        if per_archive_name not in annotation_path_dict:
            error_message_list.append(
                f'archive annotation file not exist {per_archive_name}')
    for per_archive_name in sorted(annotation_path_dict.keys()):
        if per_archive_name not in archive_path_dict:
            error_message_list.append(
                f'annotation archive file not exist {per_archive_name}')

    archive_group_list = []
    for per_archive_name in sorted(archive_path_dict.keys()):
        if per_archive_name not in annotation_path_dict:
            continue

        per_archive_index, per_archive_path = archive_path_dict[
            per_archive_name]
        archive_group_list.append([
            per_archive_name,
            per_archive_index,
            [per_archive_path],
            annotation_path_dict[per_archive_name],
        ])

    archive_group_list = sorted(archive_group_list, key=lambda x: x[1])
    file_copy_pair_list = sorted(file_copy_pair_list, key=lambda x: x[0])

    return file_copy_pair_list, archive_group_list, error_message_list


def check_required_dataset_complete(root_dataset_path, archive_group_list):
    """解压前预检: 根目录构成、tar数量与编号连号、gzip头尾、文件字节数真值、标注逐条校验

    数据集本身不完整就没必要跑几十小时解压，也避免"少了几个tar但整体报成功"。
    实测.cache里残留14个images/*.incomplete，说明下载确实中断过，这些预检是必需的。

    返回[错误信息列表, 告警信息列表]。
    """
    error_message_list, warning_message_list = [], []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')

        return error_message_list, warning_message_list

    all_root_dir_name_list = []
    for per_name in sorted(os.listdir(root_dataset_path)):
        if check_skip_file_or_dir(per_name):
            continue

        if os.path.isdir(os.path.join(root_dataset_path, per_name)):
            all_root_dir_name_list.append(per_name)
            continue

        # 根目录下出现新的非跳过文件必须显式上报，否则会被静默漏处理
        error_message_list.append(f'unknown file in root dir {per_name}')

    for per_dir_name in all_root_dir_name_list:
        if per_dir_name not in EXPECTED_ROOT_DIR_NAME_LIST:
            error_message_list.append(f'unknown root dir {per_dir_name}')
    for per_dir_name in EXPECTED_ROOT_DIR_NAME_LIST:
        if per_dir_name not in all_root_dir_name_list:
            error_message_list.append(f'root dir not exist {per_dir_name}')

    print('1111', 'archive group:', len(archive_group_list), 'expected:',
          EXPECTED_ARCHIVE_NUM)

    if len(archive_group_list) != EXPECTED_ARCHIVE_NUM:
        error_message_list.append(
            f'archive num not match {len(archive_group_list)} != {EXPECTED_ARCHIVE_NUM}'
        )

    # split编号必须是1..102连号，缺号说明有tar没下载下来
    archive_index_list = [
        per_archive_index for _, per_archive_index, _, _ in archive_group_list
    ]
    per_missing_index_list = sorted(
        set(EXPECTED_ARCHIVE_INDEX_LIST) - set(archive_index_list))
    if len(per_missing_index_list) > 0:
        error_message_list.append(
            f'archive index not continuous, missing index {per_missing_index_list[:10]}'
        )
    per_unknown_index_list = sorted(
        set(archive_index_list) - set(EXPECTED_ARCHIVE_INDEX_LIST))
    if len(per_unknown_index_list) > 0:
        error_message_list.append(
            f'unknown archive index {per_unknown_index_list[:10]}')

    per_tree_error_message_list, per_tree_warning_message_list = check_huggingface_tree_file(
        root_dataset_path)
    error_message_list.extend(per_tree_error_message_list)
    warning_message_list.extend(per_tree_warning_message_list)

    archive_path_check_list = [
        per_archive_part_path_list[0]
        for _, _, per_archive_part_path_list, _ in archive_group_list
    ]
    print('1111', 'check archive gzip head and tail:',
          len(archive_path_check_list))
    with Pool(processes=PROCESS_NUM) as pool:
        for per_archive_error_message_list in tqdm(
                pool.imap_unordered(check_single_archive_gzip_head_and_tail,
                                    archive_path_check_list),
                total=len(archive_path_check_list)):
            error_message_list.extend(per_archive_error_message_list)

    annotation_path_check_list = [
        per_annotation_path
        for _, _, _, per_annotation_path in archive_group_list
    ]
    print('1111', 'check annotation file magic:',
          len(annotation_path_check_list))
    with Pool(processes=PROCESS_NUM) as pool:
        for per_annotation_error_message_list in tqdm(
                pool.imap_unordered(check_single_annotation_file_magic,
                                    annotation_path_check_list),
                total=len(annotation_path_check_list)):
            error_message_list.extend(per_annotation_error_message_list)

    # 逐条校验全部102个labels json(830MB)。这一步是为了在跑几十小时解压之前
    # 就把"标注缺字段/条数不对"暴露出来，只回传计数与问题清单，不回传样本对数据
    annotation_load_pair_list = [[
        per_archive_name,
        per_annotation_path,
    ] for per_archive_name, _, _, per_annotation_path in archive_group_list]
    print('1111', 'check annotation content:', len(annotation_load_pair_list))

    total_annotation_sample_count, total_valid_sample_pair_count = 0, 0
    total_annotation_image_count, total_abnormal_score_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_archive_annotation, annotation_load_pair_list),
                                     total=len(annotation_load_pair_list)):
            per_archive_name, per_annotation_sample_count, per_valid_sample_pair_count, per_annotation_image_count, per_invalid_annotation_message_list, per_abnormal_score_message_list, per_annotation_error_message_list = per_check_result

            total_annotation_sample_count += per_annotation_sample_count
            total_valid_sample_pair_count += per_valid_sample_pair_count
            total_annotation_image_count += per_annotation_image_count
            total_abnormal_score_count += len(per_abnormal_score_message_list)

            error_message_list.extend(per_annotation_error_message_list)
            if len(per_invalid_annotation_message_list) > 0:
                print('7777', per_archive_name, 'invalid annotation:',
                      len(per_invalid_annotation_message_list),
                      per_invalid_annotation_message_list[:3])
                error_message_list.append(
                    f'{per_archive_name} invalid annotation num {len(per_invalid_annotation_message_list)} {per_invalid_annotation_message_list[:3]}'
                )
            if len(per_abnormal_score_message_list) > 0:
                # 打分被上游CoT污染: 样本对照常保留(分数已标记成不可用)，只告警
                print('2222', per_archive_name, 'abnormal score:',
                      len(per_abnormal_score_message_list),
                      per_abnormal_score_message_list[:3])
                warning_message_list.append(
                    f'{per_archive_name} abnormal score annotation num {len(per_abnormal_score_message_list)} {per_abnormal_score_message_list[:3]}'
                )

    print('1111', 'annotation sample:', total_annotation_sample_count,
          'valid sample pair:', total_valid_sample_pair_count,
          'annotation image:', total_annotation_image_count,
          'abnormal score annotation:', total_abnormal_score_count,
          'expected sample pair:', EXPECTED_TOTAL_SAMPLE_PAIR_COUNT,
          'expected image:', EXPECTED_TOTAL_IMAGE_COUNT)

    if total_annotation_sample_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total annotation sample count not match {total_annotation_sample_count} != {EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}'
        )
    if total_valid_sample_pair_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} != {EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}'
        )
    if total_annotation_image_count != EXPECTED_TOTAL_IMAGE_COUNT:
        error_message_list.append(
            f'total annotation image count not match {total_annotation_image_count} != {EXPECTED_TOTAL_IMAGE_COUNT}'
        )

    return error_message_list, warning_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非压缩包文件原样拷贝到目标目录，保持相对路径不变

    该数据集这里拷的是labels/*.json(830MB)，作为汇总标注的溯源真值。
    """
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

    if os.path.getsize(save_file_path) != os.path.getsize(per_file_path):
        print('4444', per_file_path, 'copy file size not match')

        return [per_file_relative_path, 'copy file size not match']

    return [per_file_relative_path, '']


def save_single_member_file(load_member_file, save_member_path,
                            per_member_size):
    """流式把tar成员写盘并立刻校验落盘大小，返回错误信息(空串表示成功)

    先写盘再校验大小是为了避免"copyfileobj在压缩流截断时静默写出半截文件(甚至0字节),
    后面只看存在性就当成正常样本"。
    """
    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_member_path, 'wb') as save_member_file:
            shutil.copyfileobj(load_member_file, save_member_file,
                               EXTRACT_FILE_BLOCK_SIZE)
    except Exception as e:
        return f'write member failed {save_member_path} {e}'

    try:
        if os.path.getsize(save_member_path) != per_member_size:
            return f'write member size not match {save_member_path}'
    except Exception as e:
        return f'stat member failed {save_member_path} {e}'

    return ''


def process_single_archive_group(archive_group, save_dataset_path,
                                 save_annotation_dir_path):
    """流式解压单个split.tar.gz，并生成该split的jsonl汇总标注

    这里**故意不用tarfile的mode='r|gz'**: tarfile自带的gz解压走内部_Stream，
    只调zlib.decompressobj、**不校验gzip尾部的CRC32与ISIZE**，
    tar被截断时很可能只是"少解出一批图"然后静默正常结束。
    改成 gzip.GzipFile -> tarfile.open(mode='r|') 之后，tar成员读完再把
    GzipFile drain到EOF，GzipFile会校验CRC32与ISIZE，不一致直接抛BadGzipFile。

    tar内成员名自带split{N}/前缀，与labels里的img_pathN完全一致，
    所以直接按原相对路径落到 unzip_images/ 下，标注里的路径就是现成的。
    """
    per_archive_name, per_archive_index, per_archive_part_path_list, per_annotation_path = archive_group

    per_annotation_result = load_single_archive_annotation(
        [per_archive_name, per_annotation_path])

    sample_pair_list = per_annotation_result['sample_pair_list']
    image_relative_path_dict = per_annotation_result[
        'image_relative_path_dict']
    annotation_sample_count = per_annotation_result['annotation_sample_count']
    invalid_annotation_message_list = per_annotation_result[
        'invalid_annotation_message_list']
    abnormal_score_message_list = per_annotation_result[
        'abnormal_score_message_list']
    error_message_list = list(per_annotation_result['error_message_list'])

    save_image_dir_path = os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME)
    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_MEMBER_DIR_NAME,
                                           per_archive_name)

    extract_file_count, skip_file_count = 0, 0
    not_save_file_count, fail_file_count = 0, 0
    total_file_member_count, duplicate_member_count = 0, 0
    useless_member_count = 0
    unknown_suffix_member_name_list = []
    unreferenced_member_relative_path_list = []
    reach_tar_end, reach_gzip_end = False, False

    # 每个成员是否已经确认完整落盘(或按开关有意不落盘)，用于后面和标注对账
    member_save_flag_dict = {}

    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    load_gzip_file = None
    try:
        load_gzip_file = gzip.GzipFile(fileobj=archive_reader, mode='rb')
        with tarfile.open(fileobj=load_gzip_file, mode='r|') as load_tar_file:
            for per_member in load_tar_file:
                per_member_name = get_normalize_member_name(per_member.name)
                if not per_member_name:
                    print('5555', per_archive_name, per_member.name)
                    error_message_list.append(
                        f'illegal member name {per_member.name}')
                    continue

                if per_member.isdir():
                    # 实测该数据集的tar里没有目录成员，出现了也照样建目录
                    os.makedirs(os.path.join(save_image_dir_path,
                                             per_member_name),
                                exist_ok=True)
                    continue

                if not per_member.isfile():
                    # 实测只有普通文件，出现链接等类型必须显式上报
                    print('5555', per_archive_name, per_member.name,
                          'not a regular file')
                    error_message_list.append(
                        f'not a regular file {per_member.name}')
                    continue

                total_file_member_count += 1

                if check_skip_file_or_dir(per_member_name):
                    # tar内出现无用信息(实测没有)，不落盘但要计入对账
                    useless_member_count += 1
                    not_save_file_count += 1
                    continue

                if not check_image_file_suffix(per_member_name):
                    # 既不是png也不是已知图像后缀的成员必须上报，不能默认当图像统计;
                    # 但仍然照原路径落盘，避免丢掉可能有用的信息
                    unknown_suffix_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'unknown suffix member {per_member_name}')

                per_member_is_duplicate = per_member_name in member_save_flag_dict
                if per_member_is_duplicate:
                    # 同一个tar里出现重名成员时按名写盘会互相覆盖，
                    # 这里改写到独立目录保留数据并上报，不能静默丢样本
                    duplicate_member_count += 1
                    error_message_list.append(
                        f'duplicate member name {per_member_name}')
                    save_member_path = os.path.join(
                        save_duplicate_dir_path, f'{duplicate_member_count}',
                        per_member_name)
                else:
                    save_member_path = os.path.join(save_image_dir_path,
                                                    per_member_name)

                    if per_member_name not in image_relative_path_dict:
                        # 标注没引用到的成员也照样落盘(不能因为labels没写就丢图),
                        # 只是它配不出样本对，所以要记入unreferenced清单显式上报
                        unreferenced_member_relative_path_list.append(
                            per_member_name)

                if not EXTRACT_IMAGE_FILE_FLAG:
                    # 只建索引模式: 图像继续留在原tar.gz里，样本对信息一样完整
                    not_save_file_count += 1
                    if not per_member_is_duplicate:
                        member_save_flag_dict[per_member_name] = 1
                    continue

                if os.path.exists(save_member_path) and os.path.getsize(
                        save_member_path) == per_member.size:
                    skip_file_count += 1
                    if not per_member_is_duplicate:
                        member_save_flag_dict[per_member_name] = 1
                    continue

                load_member_file = load_tar_file.extractfile(per_member)
                if load_member_file is None:
                    print('6666', per_archive_name, per_member.name)
                    error_message_list.append(
                        f'extract member failed {per_member.name}')
                    fail_file_count += 1
                    if not per_member_is_duplicate:
                        member_save_flag_dict[per_member_name] = 0
                    continue

                per_save_error_message = save_single_member_file(
                    load_member_file, save_member_path, per_member.size)
                if per_save_error_message:
                    print('6666', per_archive_name, per_save_error_message)
                    error_message_list.append(per_save_error_message)
                    fail_file_count += 1
                    if not per_member_is_duplicate:
                        member_save_flag_dict[per_member_name] = 0
                    continue

                extract_file_count += 1
                if not per_member_is_duplicate:
                    member_save_flag_dict[per_member_name] = 1

        reach_tar_end = True

        # tar成员读完之后必须把GzipFile drain到EOF: 只有读到流末尾，
        # GzipFile才会校验gzip footer里的CRC32与ISIZE。
        # tar的EOF块之后只剩blocking factor对齐的补零，所以这里读的数据量很小
        while True:
            per_remain_bytes = load_gzip_file.read(EXTRACT_FILE_BLOCK_SIZE)
            if not per_remain_bytes:
                break

        reach_gzip_end = True
    except Exception as e:
        # tar截断、gzip CRC不符或NAS读失败时保留已解压出的文件，但必须上报，
        # 不能静默少样本对
        print('7777', per_archive_name, len(per_archive_part_path_list), e)
        error_message_list.append(f'read archive failed {e}')
    finally:
        if load_gzip_file is not None:
            try:
                load_gzip_file.close()
            except Exception as e:
                error_message_list.append(f'close gzip file failed {e}')
        archive_reader.close()

    if not reach_tar_end:
        error_message_list.append(
            'not reach tar stream end, archive may be truncated')
    if not reach_gzip_end:
        error_message_list.append(
            'not reach gzip stream end, gzip crc32/isize not verified')
    if fail_file_count > 0:
        error_message_list.append(f'fail file count {fail_file_count}')

    if extract_file_count + skip_file_count + not_save_file_count + fail_file_count != total_file_member_count:
        error_message_list.append(
            f'process file count not match: {extract_file_count} + {skip_file_count} + {not_save_file_count} + {fail_file_count} != {total_file_member_count}'
        )

    # 生成该split的汇总标注: 只有"两张图都完整落盘 + 两条caption非空 + 打分齐备"
    # 的样本对才算完整有用信息
    valid_annotation_line_list = []
    missing_image_relative_path_list, incomplete_sample_key_list = [], []
    judgment_count_dict = collections.Counter()
    subject_num_count_dict = collections.Counter()
    score_final_count_dict = collections.Counter()
    recommend_score_sample_pair_count, perfect_score_sample_pair_count = 0, 0
    # 打分被上游污染但照常落盘的样本对(分数已置成不可用哨兵值)，只统计不判失败
    abnormal_score_sample_key_list = []

    for per_sample_pair in sample_pair_list:
        per_reference_image_relative_path = per_sample_pair[
            'reference_image_relative_path']
        per_target_image_relative_path = per_sample_pair[
            'target_image_relative_path']

        per_missing_image_relative_path_list = []
        for per_image_relative_path in [
                per_reference_image_relative_path,
                per_target_image_relative_path,
        ]:
            if member_save_flag_dict.get(per_image_relative_path, 0) != 1:
                per_missing_image_relative_path_list.append(
                    per_image_relative_path)

        if len(per_missing_image_relative_path_list) > 0:
            # 图缺一张这个样本对就不完整(参考图或生成图没了都不可训练)，隔离上报
            missing_image_relative_path_list.extend(
                per_missing_image_relative_path_list)
            incomplete_sample_key_list.append(per_sample_pair['sample_key'])
            continue

        per_save_annotation = dict(per_sample_pair)
        if EXTRACT_IMAGE_FILE_FLAG:
            # 图已落盘: image_path是相对save_dataset_path的落盘路径
            per_save_annotation[
                'reference_image_path'] = f'{SAVE_IMAGE_DIR_NAME}/{per_reference_image_relative_path}'
            per_save_annotation[
                'target_image_path'] = f'{SAVE_IMAGE_DIR_NAME}/{per_target_image_relative_path}'
        else:
            # 只建索引模式: 图还在原tar.gz里，image_path写成
            # "相对root_dataset_path的tar路径::tar内成员名"，下游按webdataset方式取
            per_save_annotation[
                'reference_image_path'] = f'{LOAD_ARCHIVE_ROOT_DIR_NAME}/{per_archive_name}.tar.gz::{per_reference_image_relative_path}'
            per_save_annotation[
                'target_image_path'] = f'{LOAD_ARCHIVE_ROOT_DIR_NAME}/{per_archive_name}.tar.gz::{per_target_image_relative_path}'

        valid_annotation_line_list.append(
            json.dumps(per_save_annotation, ensure_ascii=False))

        judgment_count_dict[per_save_annotation['judgment']] += 1
        subject_num_count_dict[str(len(
            per_save_annotation['subject_list']))] += 1

        if not per_save_annotation[
                'score_final_valid'] or not per_save_annotation[
                    'score_part_valid']:
            abnormal_score_sample_key_list.append(
                per_save_annotation['sample_key'])

        per_score_final = per_save_annotation['score_final']
        score_final_count_dict[f'{per_score_final:.1f}'] += 1
        if per_score_final >= RECOMMEND_SCORE_FINAL_THRESHOLD:
            recommend_score_sample_pair_count += 1
        if per_score_final >= PERFECT_SCORE_FINAL_THRESHOLD:
            perfect_score_sample_pair_count += 1

    save_annotation_path = os.path.join(save_annotation_dir_path,
                                        f'{per_archive_name}.jsonl')
    try:
        os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
        with open(save_annotation_path, 'w',
                  encoding='UTF-8') as save_json_file:
            for per_valid_annotation_line in valid_annotation_line_list:
                save_json_file.write(f'{per_valid_annotation_line}\n')
    except Exception as e:
        error_message_list.append(
            f'{per_archive_name} save annotation failed {e}')

    per_valid_sample_pair_count = len(valid_annotation_line_list)
    per_annotation_image_count = len(image_relative_path_dict)
    per_unique_image_member_count = len(member_save_flag_dict)

    # 核心对账: json条数 == 有效样本对数，且 tar成员数 == 标注引用图数 == 样本对数*2。
    # 两张图是"成对一起丢"的，所以只比对落盘文件同名配对永远查不出缺样本对，
    # 必须拿"labels的条数"和"tar头里数出来的成员数"同时当ground truth。
    if per_valid_sample_pair_count != annotation_sample_count:
        error_message_list.append(
            f'{per_archive_name} valid sample pair count not match annotation sample count {per_valid_sample_pair_count} != {annotation_sample_count}'
        )
    if per_annotation_image_count != per_valid_sample_pair_count * 2:
        error_message_list.append(
            f'{per_archive_name} annotation image count not match {per_annotation_image_count} != {per_valid_sample_pair_count} * 2'
        )
    if per_unique_image_member_count != per_annotation_image_count:
        error_message_list.append(
            f'{per_archive_name} tar image member count not match annotation image count {per_unique_image_member_count} != {per_annotation_image_count}'
        )
    if total_file_member_count != per_annotation_image_count + duplicate_member_count + useless_member_count:
        error_message_list.append(
            f'{per_archive_name} tar file member count not match {total_file_member_count} != {per_annotation_image_count} + {duplicate_member_count} + {useless_member_count}'
        )

    per_expected_sample_pair_count = EXPECTED_ARCHIVE_SAMPLE_PAIR_COUNT_DICT.get(
        per_archive_name, -1)
    if per_valid_sample_pair_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_name} valid sample pair count not match expected {per_valid_sample_pair_count} != {per_expected_sample_pair_count}'
        )

    return {
        'archive_name':
        per_archive_name,
        'archive_index':
        per_archive_index,
        'extract_file_count':
        extract_file_count,
        'skip_file_count':
        skip_file_count,
        'not_save_file_count':
        not_save_file_count,
        'fail_file_count':
        fail_file_count,
        'total_file_member_count':
        total_file_member_count,
        'duplicate_member_count':
        duplicate_member_count,
        'useless_member_count':
        useless_member_count,
        'unique_image_member_count':
        per_unique_image_member_count,
        'annotation_sample_count':
        annotation_sample_count,
        'annotation_image_count':
        per_annotation_image_count,
        'valid_sample_pair_count':
        per_valid_sample_pair_count,
        'recommend_score_sample_pair_count':
        recommend_score_sample_pair_count,
        'perfect_score_sample_pair_count':
        perfect_score_sample_pair_count,
        'abnormal_score_sample_pair_count':
        len(abnormal_score_sample_key_list),
        'reach_tar_end':
        reach_tar_end,
        'reach_gzip_end':
        reach_gzip_end,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_archive_name}.jsonl',
        'judgment_count_dict':
        dict(judgment_count_dict),
        'subject_num_count_dict':
        dict(subject_num_count_dict),
        'score_final_count_dict':
        dict(score_final_count_dict),
        'unknown_suffix_member_name_list':
        unknown_suffix_member_name_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'unreferenced_member_relative_path_list':
        unreferenced_member_relative_path_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'missing_image_relative_path_list':
        missing_image_relative_path_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'incomplete_sample_key_list':
        incomplete_sample_key_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'invalid_annotation_message_list':
        invalid_annotation_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'abnormal_score_message_list':
        abnormal_score_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'abnormal_score_sample_key_list':
        abnormal_score_sample_key_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }


def check_single_archive_dir_on_disk(archive_check_pair):
    """可选的二次对账: 遍历单个split的输出图像目录，核对落盘图像数与jsonl引用数"""
    per_archive_name, per_archive_dir_path, per_annotation_path, per_expected_image_count, per_expected_sample_pair_count = archive_check_pair

    error_message_list = []
    if not os.path.exists(per_archive_dir_path):
        error_message_list.append(
            f'{per_archive_name} archive image dir not exist')

        return [per_archive_name, 0, 0, error_message_list]

    image_relative_path_dict = {}
    unknown_suffix_file_count = 0
    for per_root_path, _, per_file_name_list in os.walk(per_archive_dir_path):
        for per_file_name in per_file_name_list:
            per_file_path = os.path.join(per_root_path, per_file_name)
            per_file_relative_path = os.path.relpath(per_file_path,
                                                     per_archive_dir_path)
            per_file_relative_path = per_file_relative_path.replace('\\', '/')

            if not check_image_file_suffix(per_file_name):
                unknown_suffix_file_count += 1
                continue

            image_relative_path_dict[
                f'{per_archive_name}/{per_file_relative_path}'] = 1

    per_sample_pair_count, per_missing_image_count = 0, 0
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_json_file:
            for per_line in load_json_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                per_annotation = json.loads(per_line)
                per_sample_pair_count += 1
                for per_image_relative_path in [
                        per_annotation['reference_image_relative_path'],
                        per_annotation['target_image_relative_path'],
                ]:
                    if per_image_relative_path not in image_relative_path_dict:
                        per_missing_image_count += 1
    except Exception as e:
        error_message_list.append(
            f'{per_archive_name} load save annotation failed {e}')

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_archive_name} unknown suffix file num {unknown_suffix_file_count}'
        )
    if len(image_relative_path_dict) != per_expected_image_count:
        error_message_list.append(
            f'{per_archive_name} on disk image count not match {len(image_relative_path_dict)} != {per_expected_image_count}'
        )
    if per_sample_pair_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_name} save annotation line count not match {per_sample_pair_count} != {per_expected_sample_pair_count}'
        )
    if per_missing_image_count > 0:
        error_message_list.append(
            f'{per_archive_name} on disk missing image count {per_missing_image_count}'
        )

    return [
        per_archive_name,
        len(image_relative_path_dict),
        per_sample_pair_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, archive_result_list):
    """可选的二次对账: 遍历输出目录核对每个split的落盘图像数与jsonl引用数"""
    archive_check_pair_list = []
    for per_archive_result in archive_result_list:
        archive_check_pair_list.append([
            per_archive_result['archive_name'],
            os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                         per_archive_result['archive_name']),
            os.path.join(save_dataset_path,
                         per_archive_result['save_annotation_relative_path']),
            per_archive_result['unique_image_member_count'],
            per_archive_result['valid_sample_pair_count'],
        ])

    error_message_list = []
    total_image_count, total_sample_pair_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_archive_dir_on_disk, archive_check_pair_list),
                                     total=len(archive_check_pair_list)):
            _, per_image_count, per_sample_pair_count, per_error_message_list = per_check_result
            total_image_count += per_image_count
            total_sample_pair_count += per_sample_pair_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk image:', total_image_count, 'on disk sample pair:',
          total_sample_pair_count)

    return error_message_list


def save_check_result(save_dataset_path, archive_result_list,
                      warning_message_list):
    """汇总所有split的解压与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_file_member_count, total_valid_sample_pair_count = 0, 0
    total_extract_file_count, total_skip_file_count = 0, 0
    total_not_save_file_count, total_fail_file_count = 0, 0
    total_duplicate_member_count, total_useless_member_count = 0, 0
    total_unique_image_member_count, total_annotation_sample_count = 0, 0
    total_annotation_image_count = 0
    total_recommend_score_sample_pair_count = 0
    total_perfect_score_sample_pair_count = 0
    total_abnormal_score_sample_pair_count = 0
    archive_sample_pair_count_dict = {}
    judgment_count_dict = collections.Counter()
    subject_num_count_dict = collections.Counter()
    score_final_count_dict = collections.Counter()
    unknown_suffix_member_name_list = []
    unreferenced_member_relative_path_list = []
    missing_image_relative_path_list, incomplete_sample_key_list = [], []
    invalid_annotation_message_list = []
    abnormal_score_message_list, abnormal_score_sample_key_list = [], []
    error_message_list = []

    for per_archive_result in archive_result_list:
        per_archive_name = per_archive_result['archive_name']

        total_file_member_count += per_archive_result[
            'total_file_member_count']
        total_valid_sample_pair_count += per_archive_result[
            'valid_sample_pair_count']
        total_extract_file_count += per_archive_result['extract_file_count']
        total_skip_file_count += per_archive_result['skip_file_count']
        total_not_save_file_count += per_archive_result['not_save_file_count']
        total_fail_file_count += per_archive_result['fail_file_count']
        total_duplicate_member_count += per_archive_result[
            'duplicate_member_count']
        total_useless_member_count += per_archive_result[
            'useless_member_count']
        total_unique_image_member_count += per_archive_result[
            'unique_image_member_count']
        total_annotation_sample_count += per_archive_result[
            'annotation_sample_count']
        total_annotation_image_count += per_archive_result[
            'annotation_image_count']
        total_recommend_score_sample_pair_count += per_archive_result[
            'recommend_score_sample_pair_count']
        total_perfect_score_sample_pair_count += per_archive_result[
            'perfect_score_sample_pair_count']
        total_abnormal_score_sample_pair_count += per_archive_result[
            'abnormal_score_sample_pair_count']

        archive_sample_pair_count_dict[per_archive_name] = per_archive_result[
            'valid_sample_pair_count']
        judgment_count_dict.update(per_archive_result['judgment_count_dict'])
        subject_num_count_dict.update(
            per_archive_result['subject_num_count_dict'])
        score_final_count_dict.update(
            per_archive_result['score_final_count_dict'])

        unknown_suffix_member_name_list.extend(
            per_archive_result['unknown_suffix_member_name_list'])
        unreferenced_member_relative_path_list.extend(
            per_archive_result['unreferenced_member_relative_path_list'])
        missing_image_relative_path_list.extend(
            per_archive_result['missing_image_relative_path_list'])
        incomplete_sample_key_list.extend(
            per_archive_result['incomplete_sample_key_list'])
        invalid_annotation_message_list.extend(
            per_archive_result['invalid_annotation_message_list'])
        abnormal_score_message_list.extend(
            per_archive_result['abnormal_score_message_list'])
        abnormal_score_sample_key_list.extend(
            per_archive_result['abnormal_score_sample_key_list'])

        if not per_archive_result['reach_tar_end']:
            error_message_list.append(f'{per_archive_name} not reach tar end')
        if not per_archive_result['reach_gzip_end']:
            error_message_list.append(f'{per_archive_name} not reach gzip end')

        if len(per_archive_result['error_message_list']) > 0:
            print('7777', per_archive_name,
                  per_archive_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_archive_name} error num {len(per_archive_result["error_message_list"])} {per_archive_result["error_message_list"][:3]}'
            )

    print('3333', 'total archive:', len(archive_result_list),
          'total tar file member:', total_file_member_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'total annotation sample:', total_annotation_sample_count,
          'total annotation image:', total_annotation_image_count,
          'total unique image member:', total_unique_image_member_count,
          'extract:', total_extract_file_count, 'skip:', total_skip_file_count,
          'not save:', total_not_save_file_count, 'fail:',
          total_fail_file_count, 'duplicate member:',
          total_duplicate_member_count, 'useless member:',
          total_useless_member_count)
    print('3333', 'missing image:',
          len(missing_image_relative_path_list), 'incomplete sample pair:',
          len(incomplete_sample_key_list), 'unreferenced member:',
          len(unreferenced_member_relative_path_list),
          'unknown suffix member:', len(unknown_suffix_member_name_list),
          'invalid annotation:', len(invalid_annotation_message_list))
    print('3333', 'score_final >=', RECOMMEND_SCORE_FINAL_THRESHOLD,
          'sample pair:', total_recommend_score_sample_pair_count,
          'score_final >=', PERFECT_SCORE_FINAL_THRESHOLD, 'sample pair:',
          total_perfect_score_sample_pair_count, 'abnormal score sample pair:',
          total_abnormal_score_sample_pair_count)
    print('3333', 'judgment:', dict(judgment_count_dict))
    print('3333', 'subject num:', dict(subject_num_count_dict))
    if total_abnormal_score_sample_pair_count > 0:
        # 打分被上游CoT污染的样本对: 已照常落盘、score_final置成
        # SCORE_FINAL_INVALID_VALUE，只告警不判失败(详见
        # ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG的说明)
        warning_message_list = list(warning_message_list) + [
            f'abnormal score sample pair count {total_abnormal_score_sample_pair_count} {abnormal_score_sample_key_list[:MAX_PRINT_PROBLEM_ITEM_NUM]}'
        ]

    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'extract_image_file_flag':
        EXTRACT_IMAGE_FILE_FLAG,
        'save_source_annotation_flag':
        SAVE_SOURCE_ANNOTATION_FLAG,
        'total_archive_count':
        len(archive_result_list),
        'total_tar_file_member_count':
        total_file_member_count,
        'total_unique_image_member_count':
        total_unique_image_member_count,
        'total_annotation_sample_count':
        total_annotation_sample_count,
        'total_annotation_image_count':
        total_annotation_image_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_extract_file_count':
        total_extract_file_count,
        'total_skip_file_count':
        total_skip_file_count,
        'total_not_save_file_count':
        total_not_save_file_count,
        'total_fail_file_count':
        total_fail_file_count,
        'total_duplicate_member_count':
        total_duplicate_member_count,
        'total_useless_member_count':
        total_useless_member_count,
        'total_recommend_score_sample_pair_count':
        total_recommend_score_sample_pair_count,
        'total_perfect_score_sample_pair_count':
        total_perfect_score_sample_pair_count,
        # 打分被上游污染但照常落盘的样本对数(score_final已置成不可用哨兵值)
        'total_abnormal_score_sample_pair_count':
        total_abnormal_score_sample_pair_count,
        'allow_abnormal_score_sample_pair_flag':
        ALLOW_ABNORMAL_SCORE_SAMPLE_PAIR_FLAG,
        'score_final_invalid_value':
        SCORE_FINAL_INVALID_VALUE,
        'recommend_score_final_threshold':
        RECOMMEND_SCORE_FINAL_THRESHOLD,
        'perfect_score_final_threshold':
        PERFECT_SCORE_FINAL_THRESHOLD,
        'missing_image_count':
        len(missing_image_relative_path_list),
        'incomplete_sample_pair_count':
        len(incomplete_sample_key_list),
        'unreferenced_member_count':
        len(unreferenced_member_relative_path_list),
        'unknown_suffix_member_count':
        len(unknown_suffix_member_name_list),
        'invalid_annotation_count':
        len(invalid_annotation_message_list),
        'archive_sample_pair_count_dict':
        archive_sample_pair_count_dict,
        'judgment_count_dict':
        dict(judgment_count_dict),
        'subject_num_count_dict':
        dict(subject_num_count_dict),
        'score_final_count_dict':
        dict(score_final_count_dict),
        'missing_image_relative_path_list':
        sorted(missing_image_relative_path_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'incomplete_sample_key_list':
        sorted(incomplete_sample_key_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'unreferenced_member_relative_path_list':
        sorted(unreferenced_member_relative_path_list)
        [:MAX_SAVE_PROBLEM_ITEM_NUM],
        'unknown_suffix_member_name_list':
        sorted(unknown_suffix_member_name_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'invalid_annotation_message_list':
        sorted(invalid_annotation_message_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'abnormal_score_message_list':
        sorted(abnormal_score_message_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'abnormal_score_sample_key_list':
        sorted(abnormal_score_sample_key_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        warning_message_list,
        'check_error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    # 全库硬对账: 一条样本对都不能少
    if len(archive_result_list) != EXPECTED_ARCHIVE_NUM:
        error_message_list.append(
            f'total archive count not match {len(archive_result_list)} != {EXPECTED_ARCHIVE_NUM}'
        )
    if total_valid_sample_pair_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} != {EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}'
        )
    if total_annotation_sample_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total annotation sample count not match {total_annotation_sample_count} != {EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}'
        )
    if total_unique_image_member_count != EXPECTED_TOTAL_IMAGE_COUNT:
        error_message_list.append(
            f'total unique image member count not match {total_unique_image_member_count} != {EXPECTED_TOTAL_IMAGE_COUNT}'
        )
    if total_valid_sample_pair_count * 2 != total_unique_image_member_count:
        error_message_list.append(
            f'total valid sample pair count not match image member count {total_valid_sample_pair_count} * 2 != {total_unique_image_member_count}'
        )
    if total_fail_file_count > 0:
        error_message_list.append(
            f'total fail file count {total_fail_file_count}')
    if len(missing_image_relative_path_list) > 0:
        error_message_list.append(
            f'missing image count {len(missing_image_relative_path_list)}')
    if len(incomplete_sample_key_list) > 0:
        error_message_list.append(
            f'incomplete sample pair count {len(incomplete_sample_key_list)}')
    if len(unreferenced_member_relative_path_list) > 0:
        error_message_list.append(
            f'unreferenced member count {len(unreferenced_member_relative_path_list)}'
        )
    if len(unknown_suffix_member_name_list) > 0:
        error_message_list.append(
            f'unknown suffix member count {len(unknown_suffix_member_name_list)}'
        )
    if len(invalid_annotation_message_list) > 0:
        error_message_list.append(
            f'invalid annotation count {len(invalid_annotation_message_list)}')
    if total_duplicate_member_count > 0:
        error_message_list.append(
            f'duplicate member count {total_duplicate_member_count}')

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    file_copy_pair_list, archive_group_list, scan_error_message_list = get_all_file_and_archive_group(
        root_dataset_path)

    preflight_error_message_list, warning_message_list = check_required_dataset_complete(
        root_dataset_path, archive_group_list)
    preflight_error_message_list = scan_error_message_list + preflight_error_message_list
    if len(preflight_error_message_list) > 0:
        # 数据集本身不完整(实测.cache里残留过14个*.incomplete)就没必要跑几十小时解压
        raise Exception(
            f'check dataset failed, error num {len(preflight_error_message_list)} {preflight_error_message_list[:MAX_PRINT_PROBLEM_ITEM_NUM]}'
        )

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    os.makedirs(os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME),
                exist_ok=True)

    print('1111', len(file_copy_pair_list), len(archive_group_list))
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(archive_group_list) > 0:
        print('1111', archive_group_list[0][0], archive_group_list[0][1],
              len(archive_group_list[0][2]), archive_group_list[0][3])

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

    archive_result_list = []
    extract_func = partial(process_single_archive_group,
                           save_dataset_path=save_dataset_path,
                           save_annotation_dir_path=save_annotation_dir_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_archive_result in tqdm(pool.imap_unordered(
                extract_func, archive_group_list),
                                       total=len(archive_group_list)):
            archive_result_list.append(per_archive_result)

            print('2222', per_archive_result['archive_name'], 'extract:',
                  per_archive_result['extract_file_count'], 'skip:',
                  per_archive_result['skip_file_count'], 'not save:',
                  per_archive_result['not_save_file_count'], 'fail:',
                  per_archive_result['fail_file_count'], 'tar file member:',
                  per_archive_result['total_file_member_count'],
                  'valid sample pair:',
                  per_archive_result['valid_sample_pair_count'],
                  'duplicate member:',
                  per_archive_result['duplicate_member_count'])

    check_error_message_list = save_check_result(save_dataset_path,
                                                 archive_result_list,
                                                 warning_message_list)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG and EXTRACT_IMAGE_FILE_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, archive_result_list)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解压/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:MAX_PRINT_PROBLEM_ITEM_NUM]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/UNO-1M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
