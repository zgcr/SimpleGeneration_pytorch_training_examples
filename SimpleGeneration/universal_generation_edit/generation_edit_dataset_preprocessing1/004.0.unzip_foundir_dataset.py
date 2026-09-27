import os
import re
import json
import zlib
import shutil
import struct
import collections

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: FoundIR(Foundation Model for Image Restoration)
#
# 【数据集类型】纯图像编辑(image-to-image edit)数据集，不是文生图数据集。
# 每个样本对 = 1张参考图(LQ, 退化图) + 1张编辑后图(GT, 清晰图)，两张图像素对齐、
# 同分辨率同通道，属于"全能图像复原"任务，下游只能走ti2i_dataset.py那条链路。
# 注意: 该数据集原始文件里一个文本字都没有(无json/txt/parquet/caption)，
# 所以本脚本不合成任何编辑指令，只保留数据集自带的客观信息;
# 下游若要训练，需自行按group_name/degradation_type_list提供指令文本。
#
# 【root_dataset_path实测原始保存规格】
# FoundIR/                       共37个文件、0个子目录，没有.cache/README.md/.gitattributes
# ├── 01Blur.z01 ~ .z04 + 01Blur.zip           5个分卷, 685102984659字节
# ├── 01Blur_total.zip                         685102984659字节(无用，见下)
# ├── 02Blur_Noise.zip                         单卷
# ├── 03Blur_JPEG.zip / 04Blur_Noise_JPEG.zip / 05Noise.zip
# ├── 06JPEG.z01 ~ .z02 + 06JPEG.zip           3个分卷
# ├── 07Noise_JPEG.zip
# ├── 08Haze.z01 ~ .z03 + .zip                 4个分卷
# ├── 09Lowlight_Haze.z01 ~ .z03 + .zip        4个分卷
# ├── 10Rain.z01 ~ .z02 + .zip                 3个分卷
# ├── 11Raindrop.z01 + .zip                    2个分卷
# ├── 12NightRain.zip
# ├── 13Rain_Haze.z01 ~ .z03 + .zip            4个分卷
# ├── 14Lowlight.zip
# ├── 15Lowlight_Blur.z01 + .zip               2个分卷
# └── 16Lowlight_Noise.zip / 17Lowlight_JPEG.zip
#
# 【最关键的一点: 这不是tar、也不是"按字节切片的zip"，而是PKZIP分卷(spanned/split)zip】
# - .zNN是分卷、.zip是最后一卷(中央目录在最后一卷里)，非末卷严格等于150GiB
#   (161061273600字节)，实测17组共20个非末卷全部严格等于该值;
# - 第一卷开头是4字节分卷签名 PK\x07\x08 (单卷组开头是普通的 PK\x03\x04);
# - 中央目录里每个成员记录的是[所在卷号disk_start + 卷内相对偏移rel_offset]，
#   不是整包的全局偏移。
# 所以:
#   a. zipfile.ZipFile直接打开会抛
#      BadZipFile: zipfiles that span multiple disks are not supported;
#   b. 像005/006那样把分卷字节拼成连续流再喂给zipfile也不行: 中央目录能读出来，
#      但取任何成员都会抛 Truncated file header / Bad magic number for file header,
#      因为offset是"卷内相对"的，必须自己加上该卷在整包里的起始偏移。
#      这两条错路都已实测复现过，不要再走。
# 本脚本的做法(已实测跑通): 自己解析 EOCD / Zip64 EOCD / Zip64 EOCD locator /
# 中央目录 + Zip64 extra field，取出每个成员的[disk_start, rel_offset, 压缩大小,
# 原始大小, crc32, 压缩方法]，再按
#   全局偏移 = sum(前面各卷字节数) + rel_offset
# 定位local header，用zlib.decompressobj(-15)流式解deflate。
# 实测首/中/尾成员以及跨卷边界成员(01Blur有4个、11Raindrop有1个跨150GiB边界的成员)
# 全部解压成功且CRC32与中央目录一致。
#
# 【01Blur_total.zip是685GB的重复冗余文件(无用)】
# 它的字节内容 == 01Blur.z01+z02+z03+z04+01Blur.zip 的顺序拼接:
#   总大小完全相等(685102984659)，27处抽样(含每个分卷边界±2048字节)字节完全一致;
#   它自身的中央目录里成员仍然按"5个分卷号"索引，所以当单文件打开必然取不出成员。
# 直接跳过，否则白拷685GB、而且109480个样本对会被重复处理一遍。
#
# 【每个压缩包组内部规格(实测)】
# - 有8个组包内带一层组名顶层目录(01Blur/GT/xxx.png)，另外9个组是GT/、LQ/直接在包顶层，
#   两种规格必须分别处理: 解压时剥掉包自带顶层目录，统一改挂到规范组名下，落盘为
#     images/<group_name>/GT/<文件名>
#     images/<group_name>/LQ/<文件名>
#   这样17个组的磁盘路径规格完全一致，下游不用再判断包内目录风格。
# - 每组只有GT/和LQ/两个目录，成员100%是图像(.png/.jpg/.JPG)，
#   没有任何json/txt/mask/README/demo图，中央目录里多出来的2~3个条目是目录条目。
# - GT与LQ按"文件名主干(7位数字)"严格1:1配对，17组全部满足
#   GT数 == LQ数 == 样本对数，没有孤立成员。
# - GT与LQ的后缀可以不一样(如 GT/0139431.png 对 LQ/0139431.JPG,
#   03/04/06/07/17这5组全部是png对JPG)，14/16组内部还混着.JPG和.jpg，
#   所以只能按主干配对，绝不能按文件名配对。
# - 抽样PIL打开: GT与LQ同分辨率同mode(1920x1080 / 3008x1688 / 5472x3648等)。
# - 样本主干id全局唯一，17组的主干区间首尾相接，恰好覆盖0000001~0920224、
#   无缺号无重叠，合计920224个样本对(GT+LQ共1840448张图，解压后约5.25T)。
#
# 【有用信息 / 无用信息】
# 有用(必须完整保存):
#   LQ/<主干>.<后缀>  参考图(退化图)                 -> 有用
#   GT/<主干>.<后缀>  编辑后图(清晰图)               -> 有用
#   样本主干id、所属退化组名group_name               -> 有用(唯一id/对账/按组采样)
#   degradation_type_list 退化类型(组名机械拆出来的客观标签，不是编造的文本)
#                                                    -> 有用(下游按类型组织指令/加权)
#   两张图的宽高、后缀、文件字节数、尺寸是否一致       -> 有用(分辨率分桶可不解码图像)
# 无用(不整理进训练目录):
#   01Blur_total.zip  685GB整包重复冗余
#   zip里的目录条目(01Blur/ 、GT/ 、LQ/ 共3x17个)
#   .cache/.gitattributes/README.md/.DS_Store/CACHEDIR.TAG/__MACOSX
#   (本数据集实测没有这些，仍保留过滤名单，防止之后补传时被静默漏处理)
#
# 【本脚本如何保证"每个包含完整有用信息的样本对都被处理并完整保存"】
# 1) 解压前O(1)预检: 根目录未知文件上报; 每组分卷数/分卷文件名/非末卷严格150GiB/
#    整包总字节数/首卷分卷签名; 中央目录声明卷数 == 实际分卷数;
#    CD条目数、文件成员数、GT数、LQ数、样本对数、主干区间与连号，全部和上表硬对账;
#    跨组主干区间不许重叠、17组并起来必须恰好覆盖1~920224。任一不过直接抛异常，
#    不白跑几十小时。
# 2) 按样本对分块并行，而不是"一个进程处理一个组": 否则01Blur(685GB)一个进程要跑到天亮，
#    其余31个进程闲着。
# 3) 逐成员三重校验: 解压后 原始大小 == 中央目录usize、CRC32 == 中央目录crc、
#    写盘后落盘大小 == usize。CRC校验比005/006只比对落盘大小强，能查出静默位翻转。
# 4) 只有GT和LQ两张都完整落盘(或已存在且大小一致)才写出这一行标注，
#    否则计入incomplete_sample_pair_list硬上报，绝不静默丢样本对。
# 5) 全量对账: 每组样本对 == 实测期望值、总样本对 == 920224、
#    标注行数 == 样本对数、extract+skip+not_save == 该组文件成员总数、
#    missing/orphan/crc_error/incomplete全部必须为0;
#    任一不满足都汇总后抛异常并sys.exit(1)，不再静默跑过。
#
# 【输出目录规格】
# <save_dataset_path>/FoundIR/
# ├── images/<group_name>/GT/<主干>.<后缀>              编辑后图(清晰)
# ├── images/<group_name>/LQ/<主干>.<后缀>              参考图(退化)
# ├── unzip_annotations/<group_name>/<块号>.jsonl        每行=1个完整样本对
# └── unzip_check_result.json                           全量对账报告
# ==============================================================================

# 非末卷分卷文件名 xxx.z01 / xxx.z02 ...，末卷是 xxx.zip
ARCHIVE_PART_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.z(?P<part>\d+)$'),
    re.compile(r'^(?P<prefix>.+)\.zip$'),
]

# 无用信息，不整理进训练目录:
# 01Blur_total.zip 是01Blur五个分卷的整包字节拼接(685GB重复冗余)，
#                  且它自身中央目录仍按5个分卷号索引，当单文件打开取不出任何成员
# 其余几项本数据集实测没有，保留名单防止之后补传时被静默漏处理
SKIP_FILE_OR_DIR_NAME_LIST = [
    '01Blur_total.zip',
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    '.DS_Store',
    'CACHEDIR.TAG',
    '__MACOSX',
]

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.bmp',
    '.tif',
    '.tiff',
]

# 包内的两个成员目录: GT是编辑后图(清晰)，LQ是参考图(退化)
TARGET_IMAGE_DIR_NAME = 'GT'

REFERENCE_IMAGE_DIR_NAME = 'LQ'

# ---------------------------------------------------------------------------
# 每个压缩包组的实测规格。这是"保证每个完整样本对都被处理"能被验证的关键兜底，
# 任何一项和实测对不上都说明数据集不完整或规格变了，必须显式感知。
#
# [分卷数, 整包总字节数, 中央目录条目数, 文件成员数, 样本对数,
#  最小主干, 最大主干, 包内要剥掉的顶层目录(''表示GT/LQ就在包顶层),
#  退化类型列表(组名机械拆出来的客观标签)]
ARCHIVE_GROUP_CONFIG_DICT = {
    '01Blur':
    [5, 685102984659, 218963, 218960, 109480, 1, 109480, '01Blur', ['blur']],
    '02Blur_Noise': [
        1, 185126075843, 59902, 59900, 29950, 109481, 139430, '',
        ['blur', 'noise']
    ],
    '03Blur_JPEG': [
        1, 91377447804, 59882, 59880, 29940, 139431, 169370, '',
        ['blur', 'jpeg']
    ],
    '04Blur_Noise_JPEG': [
        1, 89745039877, 59902, 59900, 29950, 169371, 199320, '',
        ['blur', 'noise', 'jpeg']
    ],
    '05Noise':
    [1, 139461942253, 116032, 116030, 58015, 199321, 257335, '', ['noise']],
    '06JPEG': [
        3, 461887048221, 119903, 119900, 59950, 257336, 317285, '06JPEG',
        ['jpeg']
    ],
    '07Noise_JPEG': [
        1, 91874845059, 59902, 59900, 29950, 317286, 347235, '',
        ['noise', 'jpeg']
    ],
    '08Haze': [
        4, 514064781570, 159603, 159600, 79800, 347236, 427035, '08Haze',
        ['haze']
    ],
    '09Lowlight_Haze': [
        4, 522395536719, 159603, 159600, 79800, 427036, 506835,
        '09Lowlight_Haze', ['lowlight', 'haze']
    ],
    '10Rain':
    [3, 421449480072, 79803, 79800, 39900, 506836, 546735, '10Rain', ['rain']],
    '11Raindrop': [
        2, 315349064201, 89659, 89656, 44828, 546736, 591563, '11Raindrop',
        ['raindrop']
    ],
    '12NightRain': [
        1, 211930512565, 80224, 80222, 40111, 591564, 631674, '',
        ['night', 'rain']
    ],
    '13Rain_Haze': [
        4, 611681148235, 159503, 159500, 79750, 631675, 711424, '13Rain_Haze',
        ['rain', 'haze']
    ],
    '14Lowlight':
    [1, 96650599759, 79926, 79924, 39962, 711425, 751386, '', ['lowlight']],
    '15Lowlight_Blur': [
        2, 310588662409, 171789, 171786, 85893, 751387, 837279,
        '15Lowlight_Blur', ['lowlight', 'blur']
    ],
    '16Lowlight_Noise': [
        1, 94555396885, 105992, 105990, 52995, 837280, 890274, '',
        ['lowlight', 'noise']
    ],
    '17Lowlight_JPEG': [
        1, 114511056952, 59902, 59900, 29950, 890275, 920224, '',
        ['lowlight', 'jpeg']
    ],
}

# 17组主干区间首尾相接，恰好覆盖1~920224，合计920224个样本对
EXPECTED_TOTAL_SAMPLE_PAIR_COUNT = 920224

EXPECTED_MIN_SAMPLE_KEY_INDEX = 1

EXPECTED_MAX_SAMPLE_KEY_INDEX = 920224

# 分卷zip的非末卷严格等于150GiB，实测17组共20个非末卷全部等于该值。
# 非末卷大小不对说明下载被截断，必须在跑5.2T解压前拦住。
EXPECTED_ARCHIVE_SPLIT_PART_SIZE = 161061273600

# 分卷zip第一卷开头4字节的分卷签名，单卷组开头是普通的local header签名
ARCHIVE_SPANNING_SIGNATURE = b'PK\x07\x08'

ARCHIVE_LOCAL_HEADER_SIGNATURE = b'PK\x03\x04'

ARCHIVE_CENTRAL_HEADER_SIGNATURE = b'PK\x01\x02'

ARCHIVE_END_OF_CENTRAL_DIR_SIGNATURE = b'PK\x05\x06'

ARCHIVE_ZIP64_END_OF_CENTRAL_DIR_SIGNATURE = b'PK\x06\x06'

ARCHIVE_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIGNATURE = b'PK\x06\x07'

ARCHIVE_CENTRAL_HEADER_FIXED_SIZE = 46

ARCHIVE_LOCAL_HEADER_FIXED_SIZE = 30

ARCHIVE_END_OF_CENTRAL_DIR_SEARCH_SIZE = 1024 * 1024

ARCHIVE_ZIP64_EXTRA_FIELD_HEADER_ID = 0x0001

ARCHIVE_STORE_COMPRESS_METHOD = 0

ARCHIVE_DEFLATE_COMPRESS_METHOD = 8

SAVE_IMAGE_DIR_NAME = 'images'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_result.json'

# 一个并行任务处理多少个样本对。
# 必须按样本对分块而不是按压缩包组分块: 01Blur一个组就有685GB/109480对，
# 按组分块的话它会一个进程跑到天亮，其余进程全部闲置。
SAMPLE_PAIR_CHUNK_SIZE = 2000

# 图像成员是否落盘。
# True : GT+LQ共1840448张图、约5.25T，NAS上inode压力可控(不是海量小文件);
# False: 只生成unzip_annotations索引，不解压不落盘，图像继续留在原分卷zip里,
#        此时拿不到宽高(不解压就没法读图像头)，宽高统一记0。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解压后再os.walk一遍输出目录做二次对账。
# 默认False: 解压时已经做了"解压后usize+CRC32校验 + 写盘后落盘大小校验 +
# extract+skip+not_save == 中央目录文件成员总数"三道对账，已能保证每个成员都被
# 处理且完整落盘; 需要彻底放心时再打开。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 解压时最多缓存多少字节的图像头用于解析宽高(不做全图解码，省CPU)。
# JPEG的SOF标记一般在前几十KB，带大EXIF缩略图时可能靠后，1MB足够覆盖。
IMAGE_HEADER_PARSE_SIZE = 1024 * 1024

# JPEG的SOF标记(记录宽高)，0xC4(DHT)/0xC8(JPG)/0xCC(DAC)不是SOF，必须排除
JPEG_START_OF_FRAME_MARKER_LIST = [
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
]

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024


class MultiPartZipReader:
    """把多个分卷zip拼接成一个可按全局偏移随机读取的只读字节流

    005/006里的MultiPartArchiveReader是纯顺序流(tar只需要顺序读)，这里不能照抄:
    分卷zip必须先读末卷尾部的中央目录、再回头按偏移取成员，是随机访问模式，
    所以改成持有全部分卷的文件句柄、按[全局偏移 -> 卷号 + 卷内偏移]换算读取。
    """

    def __init__(self, per_archive_part_path_list):
        self.per_archive_part_path_list = per_archive_part_path_list
        self.per_archive_part_size_list = [
            os.path.getsize(per_archive_part_path)
            for per_archive_part_path in per_archive_part_path_list
        ]

        # 每个分卷在整包里的起始全局偏移
        self.per_archive_part_start_offset_list = []
        total_archive_size = 0
        for per_archive_part_size in self.per_archive_part_size_list:
            self.per_archive_part_start_offset_list.append(total_archive_size)
            total_archive_size += per_archive_part_size
        self.total_archive_size = total_archive_size

        self.archive_part_file_dict = {}

    def get_archive_part_index(self, read_offset):
        """把全局偏移换算成卷号(分卷数最多5个，线性找即可)"""
        per_archive_part_index = 0
        for i in range(len(self.per_archive_part_path_list)):
            if read_offset >= self.per_archive_part_start_offset_list[i]:
                per_archive_part_index = i

        return per_archive_part_index

    def get_archive_part_file(self, per_archive_part_index):
        """分卷文件句柄常驻复用，避免每次读都open/close(NAS上代价很高)"""
        if per_archive_part_index not in self.archive_part_file_dict:
            self.archive_part_file_dict[per_archive_part_index] = open(
                self.per_archive_part_path_list[per_archive_part_index], 'rb')

        return self.archive_part_file_dict[per_archive_part_index]

    def read_at(self, read_offset, read_size):
        """从全局偏移read_offset读read_size字节，自动跨分卷边界拼接

        实测01Blur有4个成员、11Raindrop有1个成员的数据跨150GiB分卷边界，
        这里必须能无缝跨卷读，否则那几个成员永远解压失败。
        """
        read_bytes_list, remain_read_size = [], read_size
        while remain_read_size > 0 and 0 <= read_offset < self.total_archive_size:
            per_archive_part_index = self.get_archive_part_index(read_offset)
            per_archive_part_inner_offset = read_offset - self.per_archive_part_start_offset_list[
                per_archive_part_index]
            per_archive_part_remain_size = self.per_archive_part_size_list[
                per_archive_part_index] - per_archive_part_inner_offset
            if per_archive_part_remain_size <= 0:
                break

            per_archive_part_file = self.get_archive_part_file(
                per_archive_part_index)
            per_archive_part_file.seek(per_archive_part_inner_offset)
            per_read_bytes = per_archive_part_file.read(
                min(remain_read_size, per_archive_part_remain_size))
            if not per_read_bytes:
                break

            read_bytes_list.append(per_read_bytes)
            read_offset += len(per_read_bytes)
            remain_read_size -= len(per_read_bytes)

        return b''.join(read_bytes_list)

    def close(self):
        for per_archive_part_file in self.archive_part_file_dict.values():
            per_archive_part_file.close()
        self.archive_part_file_dict = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


def check_skip_file_or_dir(per_file_relative_path):
    """过滤掉01Blur_total.zip、.cache、README.md等无用文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """判断是否是图像文件后缀(该数据集GT和LQ的后缀可以不一样，必须逐个判断)"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


def get_archive_part_sort_key(per_archive_part_index):
    """分卷排序key: .z01/.z02...按数值升序排在前，末卷.zip(无编号)排最后

    分卷zip的中央目录在末卷，末卷必须是拼接后的最后一段，顺序错了整个偏移体系全废。
    不能直接按文件名字符串排序: 01Blur.zip 会排到 01Blur.z01 前面。
    """
    if per_archive_part_index and per_archive_part_index.isdigit():
        return [0, int(per_archive_part_index)]

    return [1, 0]


def get_normalized_member_name(per_member_name):
    """把zip成员名归一化成不带盘符/前导斜杠/./的相对路径

    返回[归一化名, 是否是压缩包根成员]。越界成员(..开头)返回['', False]。
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


def strip_member_top_prefix(per_member_name, per_member_top_strip_prefix):
    """剥掉压缩包自带的顶层目录，返回GT/xxx 或 LQ/xxx 形式的相对路径

    该数据集17个组里有8个组包内带一层组名顶层目录、9个组的GT/LQ直接在包顶层，
    剥掉之后统一改挂到规范组名下，磁盘路径规格才能17组完全一致。
    剥不掉(前缀不匹配)时返回None，由调用方记error，不能静默跳过。
    """
    if not per_member_top_strip_prefix:
        return per_member_name

    if per_member_name == per_member_top_strip_prefix:
        return ''

    if per_member_name.startswith(f'{per_member_top_strip_prefix}/'):
        return per_member_name[len(per_member_top_strip_prefix) + 1:]

    return None


def get_zip64_extra_field_value_list(per_extra_field_bytes,
                                     need_zip64_value_flag_list):
    """从extra field里取Zip64的[原始大小, 压缩大小, 卷内偏移, 卷号]

    Zip64 extra field(header id 0x0001)里的字段是"按需出现"的:
    只有中央目录里对应字段被写成0xFFFFFFFF(卷号是0xFFFF)时，extra里才有这一项，
    而且出现顺序固定是 原始大小 -> 压缩大小 -> 卷内偏移 -> 卷号。
    所以必须按need_zip64_value_flag_list顺序解析，不能按固定偏移取。
    """
    zip64_value_list = [None, None, None, None]

    per_extra_field_offset = 0
    while per_extra_field_offset + 4 <= len(per_extra_field_bytes):
        per_extra_field_header_id, per_extra_field_data_size = struct.unpack_from(
            '<HH', per_extra_field_bytes, per_extra_field_offset)
        per_extra_field_data_bytes = per_extra_field_bytes[
            per_extra_field_offset + 4:per_extra_field_offset + 4 +
            per_extra_field_data_size]

        if per_extra_field_header_id == ARCHIVE_ZIP64_EXTRA_FIELD_HEADER_ID:
            per_extra_field_data_offset = 0
            for per_zip64_value_index in range(3):
                if not need_zip64_value_flag_list[per_zip64_value_index]:
                    continue
                if per_extra_field_data_offset + 8 > len(
                        per_extra_field_data_bytes):
                    break
                zip64_value_list[per_zip64_value_index] = struct.unpack_from(
                    '<Q', per_extra_field_data_bytes,
                    per_extra_field_data_offset)[0]
                per_extra_field_data_offset += 8

            if need_zip64_value_flag_list[
                    3] and per_extra_field_data_offset + 4 <= len(
                        per_extra_field_data_bytes):
                zip64_value_list[3] = struct.unpack_from(
                    '<I', per_extra_field_data_bytes,
                    per_extra_field_data_offset)[0]

        per_extra_field_offset += 4 + per_extra_field_data_size

    return zip64_value_list


def read_zip_central_directory(archive_reader):
    """读分卷zip的中央目录，返回[中央目录信息字典, 成员条目列表, 错误信息列表]

    每个成员条目是紧凑的list(不用dict，1840448个成员要跨进程传递，dict太占内存):
      [成员名, 卷号disk_start, 卷内偏移rel_offset, 压缩大小, 原始大小, crc32, 压缩方法]
    """
    error_message_list = []
    archive_info_dict = {
        'total_disk_num': 0,
        'central_dir_disk_index': 0,
        'central_dir_entry_num': 0,
        'central_dir_size': 0,
        'central_dir_offset': 0,
    }

    per_search_size = min(archive_reader.total_archive_size,
                          ARCHIVE_END_OF_CENTRAL_DIR_SEARCH_SIZE)
    if per_search_size < 22:
        error_message_list.append(
            'archive too small to hold end of central dir')

        return archive_info_dict, [], error_message_list

    per_tail_offset = archive_reader.total_archive_size - per_search_size
    per_tail_bytes = archive_reader.read_at(per_tail_offset, per_search_size)

    per_end_record_index = per_tail_bytes.rfind(
        ARCHIVE_END_OF_CENTRAL_DIR_SIGNATURE)
    if per_end_record_index < 0:
        error_message_list.append('end of central dir record not found')

        return archive_info_dict, [], error_message_list

    (per_disk_index, per_central_dir_disk_index, _, per_central_dir_entry_num,
     per_central_dir_size,
     per_central_dir_offset) = struct.unpack_from('<HHHHII', per_tail_bytes,
                                                  per_end_record_index + 4)

    # 分卷 + 大包必然走Zip64，32位EOCD里的字段全是0xFFFF/0xFFFFFFFF占位，
    # 真值在Zip64 EOCD记录里，必须优先用Zip64的值
    per_zip64_locator_index = per_tail_bytes.rfind(
        ARCHIVE_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIGNATURE, 0,
        per_end_record_index)

    per_zip64_end_record_global_offset = -1
    if per_zip64_locator_index >= 0:
        (per_zip64_end_record_disk_index, per_zip64_end_record_relative_offset,
         _) = struct.unpack_from('<IQI', per_tail_bytes,
                                 per_zip64_locator_index + 4)
        if per_zip64_end_record_disk_index < len(
                archive_reader.per_archive_part_start_offset_list):
            per_zip64_end_record_global_offset = archive_reader.per_archive_part_start_offset_list[
                per_zip64_end_record_disk_index] + per_zip64_end_record_relative_offset

    per_zip64_end_record_bytes = b''
    if per_zip64_end_record_global_offset >= 0:
        per_zip64_end_record_bytes = archive_reader.read_at(
            per_zip64_end_record_global_offset, 56)
        if per_zip64_end_record_bytes[:
                                      4] != ARCHIVE_ZIP64_END_OF_CENTRAL_DIR_SIGNATURE:
            per_zip64_end_record_bytes = b''

    if not per_zip64_end_record_bytes:
        # locator缺失或指向异常时，退回到在尾部字节里直接找Zip64 EOCD记录
        per_zip64_end_record_index = per_tail_bytes.rfind(
            ARCHIVE_ZIP64_END_OF_CENTRAL_DIR_SIGNATURE, 0,
            per_end_record_index)
        if per_zip64_end_record_index >= 0:
            per_zip64_end_record_bytes = per_tail_bytes[
                per_zip64_end_record_index:per_zip64_end_record_index + 56]

    if per_zip64_end_record_bytes and len(per_zip64_end_record_bytes) >= 56:
        (_, _, _, per_disk_index, per_central_dir_disk_index, _,
         per_central_dir_entry_num, per_central_dir_size,
         per_central_dir_offset) = struct.unpack_from(
             '<QHHIIQQQQ', per_zip64_end_record_bytes, 4)

    if per_central_dir_disk_index >= len(
            archive_reader.per_archive_part_start_offset_list):
        error_message_list.append(
            f'central dir disk index {per_central_dir_disk_index} out of archive part num '
            f'{len(archive_reader.per_archive_part_path_list)}')

        return archive_info_dict, [], error_message_list

    archive_info_dict = {
        'total_disk_num': per_disk_index + 1,
        'central_dir_disk_index': per_central_dir_disk_index,
        'central_dir_entry_num': per_central_dir_entry_num,
        'central_dir_size': per_central_dir_size,
        'central_dir_offset': per_central_dir_offset,
    }

    # 中央目录偏移是"相对所在卷起点"的，必须加上该卷在整包里的起始偏移
    per_central_dir_global_offset = archive_reader.per_archive_part_start_offset_list[
        per_central_dir_disk_index] + per_central_dir_offset
    per_central_dir_bytes = archive_reader.read_at(
        per_central_dir_global_offset, per_central_dir_size)
    if len(per_central_dir_bytes) != per_central_dir_size:
        error_message_list.append(
            f'read central dir size not match {len(per_central_dir_bytes)} != {per_central_dir_size}'
        )

        return archive_info_dict, [], error_message_list

    member_entry_list = []
    per_central_dir_read_offset = 0
    while per_central_dir_read_offset + ARCHIVE_CENTRAL_HEADER_FIXED_SIZE <= len(
            per_central_dir_bytes):
        if per_central_dir_bytes[
                per_central_dir_read_offset:per_central_dir_read_offset +
                4] != ARCHIVE_CENTRAL_HEADER_SIGNATURE:
            error_message_list.append(
                f'central dir header signature broken at {per_central_dir_read_offset}'
            )
            break

        (_, _, _, per_compress_method, _, _, per_member_crc,
         per_member_compress_size, per_member_size, per_member_name_size,
         per_member_extra_size, per_member_comment_size, per_member_disk_index,
         _, _, per_member_relative_offset) = struct.unpack_from(
             '<HHHHHHIIIHHHHHII', per_central_dir_bytes,
             per_central_dir_read_offset + 4)

        per_member_name_offset = per_central_dir_read_offset + ARCHIVE_CENTRAL_HEADER_FIXED_SIZE
        per_member_name = per_central_dir_bytes[
            per_member_name_offset:per_member_name_offset +
            per_member_name_size].decode('UTF-8', 'replace')
        per_member_extra_bytes = per_central_dir_bytes[
            per_member_name_offset +
            per_member_name_size:per_member_name_offset +
            per_member_name_size + per_member_extra_size]

        need_zip64_value_flag_list = [
            per_member_size == 0xFFFFFFFF,
            per_member_compress_size == 0xFFFFFFFF,
            per_member_relative_offset == 0xFFFFFFFF,
            per_member_disk_index == 0xFFFF,
        ]
        if any(need_zip64_value_flag_list):
            zip64_value_list = get_zip64_extra_field_value_list(
                per_member_extra_bytes, need_zip64_value_flag_list)
            if need_zip64_value_flag_list[0] and zip64_value_list[
                    0] is not None:
                per_member_size = zip64_value_list[0]
            if need_zip64_value_flag_list[1] and zip64_value_list[
                    1] is not None:
                per_member_compress_size = zip64_value_list[1]
            if need_zip64_value_flag_list[2] and zip64_value_list[
                    2] is not None:
                per_member_relative_offset = zip64_value_list[2]
            if need_zip64_value_flag_list[3] and zip64_value_list[
                    3] is not None:
                per_member_disk_index = zip64_value_list[3]

        member_entry_list.append([
            per_member_name,
            per_member_disk_index,
            per_member_relative_offset,
            per_member_compress_size,
            per_member_size,
            per_member_crc,
            per_compress_method,
        ])

        per_central_dir_read_offset += (ARCHIVE_CENTRAL_HEADER_FIXED_SIZE +
                                        per_member_name_size +
                                        per_member_extra_size +
                                        per_member_comment_size)

    if len(member_entry_list) != per_central_dir_entry_num:
        error_message_list.append(
            f'central dir entry num not match {len(member_entry_list)} != {per_central_dir_entry_num}'
        )

    return archive_info_dict, member_entry_list, error_message_list


def get_image_size_from_bytes(per_image_head_bytes):
    """只解析图像文件头拿宽高，不做全图解码(1840448张图全解码会白烧几十小时CPU)

    只需要覆盖该数据集实测存在的png/jpg两种格式，拿不到时返回[0, 0]由调用方上报。
    """
    if per_image_head_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        if len(per_image_head_bytes
               ) >= 24 and per_image_head_bytes[12:16] == b'IHDR':
            per_image_width, per_image_height = struct.unpack_from(
                '>II', per_image_head_bytes, 16)

            return [per_image_width, per_image_height]

        return [0, 0]

    if per_image_head_bytes[:2] == b'\xff\xd8':
        per_read_offset = 2
        while per_read_offset + 4 <= len(per_image_head_bytes):
            if per_image_head_bytes[per_read_offset] != 0xFF:
                per_read_offset += 1
                continue

            per_marker = per_image_head_bytes[per_read_offset + 1]
            if per_marker == 0xFF:
                # 标记前可以有任意多个填充的0xFF
                per_read_offset += 1
                continue

            if per_marker in [0xD8, 0x01] or 0xD0 <= per_marker <= 0xD7:
                # 这几个标记没有长度字段
                per_read_offset += 2
                continue

            per_segment_size = struct.unpack_from('>H', per_image_head_bytes,
                                                  per_read_offset + 2)[0]

            if per_marker in JPEG_START_OF_FRAME_MARKER_LIST:
                if per_read_offset + 9 <= len(per_image_head_bytes):
                    per_image_height = struct.unpack_from(
                        '>H', per_image_head_bytes, per_read_offset + 5)[0]
                    per_image_width = struct.unpack_from(
                        '>H', per_image_head_bytes, per_read_offset + 7)[0]

                    return [per_image_width, per_image_height]

                return [0, 0]

            if per_marker == 0xDA:
                # 已经到压缩数据了，后面不会再有SOF
                return [0, 0]

            per_read_offset += 2 + per_segment_size

        return [0, 0]

    return [0, 0]


def get_image_size_from_file(per_image_path):
    """已存在的图像文件只读文件头拿宽高，用于断点重跑时跳过解压但仍要写标注的情况"""
    try:
        with open(per_image_path, 'rb') as load_image_file:
            per_image_head_bytes = load_image_file.read(
                IMAGE_HEADER_PARSE_SIZE)
    except Exception:
        return [0, 0]

    return get_image_size_from_bytes(per_image_head_bytes)


def extract_single_member_to_file(archive_reader, member_entry,
                                  save_member_path):
    """按中央目录记录的偏移解压单个zip成员并落盘

    返回[宽, 高, 错误信息(空串表示成功)]。三重校验:
      1. 解压后原始大小 == 中央目录usize;
      2. 解压后CRC32 == 中央目录crc32(能查出静默位翻转，比只比对落盘大小强);
      3. 写盘后落盘大小 == usize。
    """
    (per_member_name, per_member_disk_index, per_member_relative_offset,
     per_member_compress_size, per_member_size, per_member_crc,
     per_compress_method) = member_entry

    if per_member_disk_index >= len(
            archive_reader.per_archive_part_start_offset_list):
        return [
            0, 0,
            f'member disk index out of range {per_member_name} {per_member_disk_index}'
        ]

    # local header偏移是"相对所在卷起点"的，这一步是分卷zip最容易搞错的地方:
    # 直接把rel_offset当全局偏移用，必然读到错误位置并抛"Truncated file header"
    per_local_header_global_offset = archive_reader.per_archive_part_start_offset_list[
        per_member_disk_index] + per_member_relative_offset
    per_local_header_bytes = archive_reader.read_at(
        per_local_header_global_offset, ARCHIVE_LOCAL_HEADER_FIXED_SIZE)
    if len(per_local_header_bytes) != ARCHIVE_LOCAL_HEADER_FIXED_SIZE:
        return [0, 0, f'read local header failed {per_member_name}']

    if per_local_header_bytes[:4] != ARCHIVE_LOCAL_HEADER_SIGNATURE:
        return [
            0, 0,
            f'local header signature broken {per_member_name} {per_local_header_bytes[:4]!r}'
        ]

    per_local_header_name_size, per_local_header_extra_size = struct.unpack_from(
        '<HH', per_local_header_bytes, 26)
    per_member_data_global_offset = (per_local_header_global_offset +
                                     ARCHIVE_LOCAL_HEADER_FIXED_SIZE +
                                     per_local_header_name_size +
                                     per_local_header_extra_size)

    if per_compress_method not in [
            ARCHIVE_STORE_COMPRESS_METHOD, ARCHIVE_DEFLATE_COMPRESS_METHOD
    ]:
        return [
            0, 0,
            f'unsupported compress method {per_member_name} {per_compress_method}'
        ]

    per_decompress_object = None
    if per_compress_method == ARCHIVE_DEFLATE_COMPRESS_METHOD:
        # -15表示raw deflate(zip成员里没有zlib头)
        per_decompress_object = zlib.decompressobj(-15)

    save_temp_member_path = f'{save_member_path}.tmp'
    per_write_size, per_write_crc = 0, 0
    per_image_head_bytes_list, per_image_head_size = [], 0

    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_temp_member_path, 'wb') as save_member_file:
            per_read_offset, per_remain_read_size = per_member_data_global_offset, per_member_compress_size
            while per_remain_read_size > 0:
                per_read_bytes = archive_reader.read_at(
                    per_read_offset,
                    min(EXTRACT_FILE_BLOCK_SIZE, per_remain_read_size))
                if not per_read_bytes:
                    break

                per_read_offset += len(per_read_bytes)
                per_remain_read_size -= len(per_read_bytes)

                if per_decompress_object is None:
                    per_write_bytes = per_read_bytes
                else:
                    per_write_bytes = per_decompress_object.decompress(
                        per_read_bytes)

                if not per_write_bytes:
                    continue

                save_member_file.write(per_write_bytes)
                per_write_size += len(per_write_bytes)
                per_write_crc = zlib.crc32(per_write_bytes, per_write_crc)

                if per_image_head_size < IMAGE_HEADER_PARSE_SIZE:
                    per_image_head_bytes_list.append(per_write_bytes)
                    per_image_head_size += len(per_write_bytes)

            if per_remain_read_size > 0:
                raise Exception(
                    f'read member data truncated, remain {per_remain_read_size}'
                )

            if per_decompress_object is not None:
                per_write_bytes = per_decompress_object.flush()
                if per_write_bytes:
                    save_member_file.write(per_write_bytes)
                    per_write_size += len(per_write_bytes)
                    per_write_crc = zlib.crc32(per_write_bytes, per_write_crc)
                    if per_image_head_size < IMAGE_HEADER_PARSE_SIZE:
                        per_image_head_bytes_list.append(per_write_bytes)
                        per_image_head_size += len(per_write_bytes)
    except Exception as e:
        if os.path.exists(save_temp_member_path):
            os.remove(save_temp_member_path)

        return [0, 0, f'extract member failed {per_member_name} {e}']

    if per_write_size != per_member_size:
        os.remove(save_temp_member_path)

        return [
            0, 0,
            f'member size not match {per_member_name} {per_write_size} != {per_member_size}'
        ]

    if (per_write_crc & 0xFFFFFFFF) != per_member_crc:
        os.remove(save_temp_member_path)

        return [0, 0, f'member crc32 not match {per_member_name}']

    try:
        os.replace(save_temp_member_path, save_member_path)
    except Exception as e:
        return [0, 0, f'rename member failed {per_member_name} {e}']

    if os.path.getsize(save_member_path) != per_member_size:
        return [0, 0, f'write member size not match {per_member_name}']

    per_image_width, per_image_height = get_image_size_from_bytes(
        b''.join(per_image_head_bytes_list))

    return [per_image_width, per_image_height, '']


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表和按分卷归组后的压缩包组列表

    该数据集根目录只有37个文件、没有子目录，过滤掉01Blur_total.zip后
    file_copy_pair_list实测是空列表，保留这一步只是为了兼容后续新增文件。
    """
    file_copy_pair_list = []
    archive_part_path_dict = {}
    for per_root_path, per_dir_name_list, per_file_name_list in os.walk(
            root_dataset_path):
        per_dir_name_list[:] = [
            per_dir_name for per_dir_name in per_dir_name_list
            if per_dir_name not in SKIP_FILE_OR_DIR_NAME_LIST
        ]

        for per_file_name in sorted(per_file_name_list):
            per_file_path = os.path.join(per_root_path, per_file_name)
            per_file_relative_path = os.path.relpath(per_file_path,
                                                     root_dataset_path)

            if check_skip_file_or_dir(per_file_relative_path):
                continue

            per_archive_group_name, per_archive_part_index = None, ''
            for per_archive_part_file_name_pattern in ARCHIVE_PART_FILE_NAME_PATTERN_LIST:
                per_match_result = per_archive_part_file_name_pattern.match(
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

            if per_archive_group_name not in archive_part_path_dict:
                archive_part_path_dict[per_archive_group_name] = []
            archive_part_path_dict[per_archive_group_name].append([
                per_archive_part_index,
                per_file_path,
            ])

    archive_group_list = []
    for per_archive_group_name in sorted(archive_part_path_dict.keys()):
        per_archive_part_list = sorted(
            archive_part_path_dict[per_archive_group_name],
            key=lambda x: get_archive_part_sort_key(x[0]))
        archive_group_list.append([
            per_archive_group_name,
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

    return file_copy_pair_list, archive_group_list


def check_single_archive_group_complete(archive_group):
    """解压前预检单个压缩包组，并顺手把中央目录解析成样本对列表

    预检内容: 分卷数、分卷文件名、非末卷严格150GiB、整包总字节数、首卷分卷签名、
    中央目录声明卷数、CD条目数、文件成员数、GT数、LQ数、样本对数、主干连号。
    中央目录只有几MB，这里读一次就把样本对列表带出去，后面解压不用二次解析。
    """
    per_archive_group_name, per_archive_part_index_list, per_archive_part_path_list = archive_group

    error_message_list = []

    if per_archive_group_name not in ARCHIVE_GROUP_CONFIG_DICT:
        error_message_list.append(
            f'unknown archive group {per_archive_group_name}')

        return [per_archive_group_name, [], {}, error_message_list]

    (per_expected_part_num, per_expected_total_size,
     per_expected_central_dir_entry_num, per_expected_file_member_num,
     per_expected_sample_pair_num, per_expected_min_sample_key_index,
     per_expected_max_sample_key_index, per_member_top_strip_prefix,
     _) = ARCHIVE_GROUP_CONFIG_DICT[per_archive_group_name]

    if len(per_archive_part_path_list) != per_expected_part_num:
        error_message_list.append(
            f'{per_archive_group_name} archive part num not match '
            f'{len(per_archive_part_path_list)} != {per_expected_part_num}')

    # 分卷编号必须是1..N-1连号，最后一个必须是无编号的末卷.zip
    per_expected_part_index_list = [
        f'{per_part_index:02d}'
        for per_part_index in range(1, per_expected_part_num)
    ] + ['']
    if per_archive_part_index_list != per_expected_part_index_list:
        error_message_list.append(
            f'{per_archive_group_name} archive part index not match '
            f'{per_archive_part_index_list} != {per_expected_part_index_list}')

    per_archive_part_size_list = [
        os.path.getsize(per_archive_part_path)
        for per_archive_part_path in per_archive_part_path_list
    ]
    for per_archive_part_index in range(len(per_archive_part_size_list) - 1):
        # 非末卷严格等于150GiB，不等说明这一卷下载被截断
        if per_archive_part_size_list[
                per_archive_part_index] != EXPECTED_ARCHIVE_SPLIT_PART_SIZE:
            error_message_list.append(
                f'{per_archive_group_name} archive part size not match '
                f'{per_archive_part_path_list[per_archive_part_index]} '
                f'{per_archive_part_size_list[per_archive_part_index]} != '
                f'{EXPECTED_ARCHIVE_SPLIT_PART_SIZE}')

    per_total_archive_size = sum(per_archive_part_size_list)
    if per_total_archive_size != per_expected_total_size:
        error_message_list.append(
            f'{per_archive_group_name} archive total size not match '
            f'{per_total_archive_size} != {per_expected_total_size}')

    try:
        with open(per_archive_part_path_list[0], 'rb') as load_archive_file:
            per_archive_head_bytes = load_archive_file.read(4)
    except Exception as e:
        error_message_list.append(
            f'{per_archive_group_name} read archive head failed {e}')

        return [per_archive_group_name, [], {}, error_message_list]

    per_expected_head_signature = ARCHIVE_SPANNING_SIGNATURE if per_expected_part_num > 1 else ARCHIVE_LOCAL_HEADER_SIGNATURE
    if per_archive_head_bytes != per_expected_head_signature:
        error_message_list.append(
            f'{per_archive_group_name} archive head signature not match '
            f'{per_archive_head_bytes!r} != {per_expected_head_signature!r}')

    archive_reader = MultiPartZipReader(per_archive_part_path_list)
    try:
        per_archive_info_dict, per_member_entry_list, per_central_dir_error_message_list = read_zip_central_directory(
            archive_reader)
    except Exception as e:
        archive_reader.close()
        error_message_list.append(
            f'{per_archive_group_name} read central dir failed {e}')

        return [per_archive_group_name, [], {}, error_message_list]
    finally:
        archive_reader.close()

    error_message_list.extend([
        f'{per_archive_group_name} {per_central_dir_error_message}'
        for per_central_dir_error_message in per_central_dir_error_message_list
    ])

    if per_archive_info_dict.get('total_disk_num',
                                 0) != len(per_archive_part_path_list):
        error_message_list.append(
            f'{per_archive_group_name} central dir disk num not match '
            f'{per_archive_info_dict.get("total_disk_num", 0)} != {len(per_archive_part_path_list)}'
        )

    if per_archive_info_dict.get('central_dir_entry_num',
                                 0) != per_expected_central_dir_entry_num:
        error_message_list.append(
            f'{per_archive_group_name} central dir entry num not match '
            f'{per_archive_info_dict.get("central_dir_entry_num", 0)} != '
            f'{per_expected_central_dir_entry_num}')

    target_member_entry_dict, reference_member_entry_dict = {}, {}
    file_member_count, dir_member_count = 0, 0
    unknown_member_name_list = []

    for per_member_entry in per_member_entry_list:
        per_member_name = per_member_entry[0]

        per_normalized_member_name, per_is_archive_root = get_normalized_member_name(
            per_member_name)
        if not per_normalized_member_name:
            if not per_is_archive_root:
                error_message_list.append(
                    f'{per_archive_group_name} illegal member name {per_member_name}'
                )
            continue

        if per_member_name.endswith('/'):
            dir_member_count += 1
            continue

        if check_skip_file_or_dir(per_normalized_member_name):
            continue

        per_member_relative_path = strip_member_top_prefix(
            per_normalized_member_name, per_member_top_strip_prefix)
        if per_member_relative_path is None:
            error_message_list.append(
                f'{per_archive_group_name} member top prefix not match {per_member_name}'
            )
            continue

        if not per_member_relative_path:
            continue

        file_member_count += 1

        per_member_relative_path_part_list = per_member_relative_path.split(
            '/')
        if len(per_member_relative_path_part_list) != 2:
            # 剥掉顶层目录后必须刚好是 GT/xxx 或 LQ/xxx 两段
            unknown_member_name_list.append(per_member_name)
            error_message_list.append(
                f'{per_archive_group_name} unknown member path {per_member_name}'
            )
            continue

        per_member_dir_name, per_member_file_name = per_member_relative_path_part_list
        if not check_image_file_suffix(per_member_file_name):
            unknown_member_name_list.append(per_member_name)
            error_message_list.append(
                f'{per_archive_group_name} member not an image {per_member_name}'
            )
            continue

        # GT和LQ的后缀可以不一样，只能按文件名主干配对
        per_sample_key = os.path.splitext(per_member_file_name)[0]

        # 跨进程要传1840448个成员，这里只留解压必需的紧凑字段
        per_compact_member_entry = [
            per_member_file_name,
            per_member_entry[1],
            per_member_entry[2],
            per_member_entry[3],
            per_member_entry[4],
            per_member_entry[5],
            per_member_entry[6],
        ]

        if per_member_dir_name == TARGET_IMAGE_DIR_NAME:
            if per_sample_key in target_member_entry_dict:
                error_message_list.append(
                    f'{per_archive_group_name} duplicate target member {per_member_name}'
                )
                continue
            target_member_entry_dict[per_sample_key] = per_compact_member_entry
        elif per_member_dir_name == REFERENCE_IMAGE_DIR_NAME:
            if per_sample_key in reference_member_entry_dict:
                error_message_list.append(
                    f'{per_archive_group_name} duplicate reference member {per_member_name}'
                )
                continue
            reference_member_entry_dict[
                per_sample_key] = per_compact_member_entry
        else:
            unknown_member_name_list.append(per_member_name)
            error_message_list.append(
                f'{per_archive_group_name} unknown member dir {per_member_name}'
            )

    if file_member_count != per_expected_file_member_num:
        error_message_list.append(
            f'{per_archive_group_name} file member num not match '
            f'{file_member_count} != {per_expected_file_member_num}')

    # GT有LQ没有(或反之)的样本对是不完整样本对，必须显式上报，不能静默丢
    missing_reference_sample_key_list = sorted(
        set(target_member_entry_dict.keys()) -
        set(reference_member_entry_dict.keys()))
    missing_target_sample_key_list = sorted(
        set(reference_member_entry_dict.keys()) -
        set(target_member_entry_dict.keys()))
    if len(missing_reference_sample_key_list) > 0:
        error_message_list.append(
            f'{per_archive_group_name} missing reference image sample key num '
            f'{len(missing_reference_sample_key_list)} '
            f'{missing_reference_sample_key_list[:5]}')
    if len(missing_target_sample_key_list) > 0:
        error_message_list.append(
            f'{per_archive_group_name} missing target image sample key num '
            f'{len(missing_target_sample_key_list)} '
            f'{missing_target_sample_key_list[:5]}')

    sample_key_list = sorted(
        set(target_member_entry_dict.keys())
        & set(reference_member_entry_dict.keys()))

    if len(sample_key_list) != per_expected_sample_pair_num:
        error_message_list.append(
            f'{per_archive_group_name} sample pair num not match '
            f'{len(sample_key_list)} != {per_expected_sample_pair_num}')

    non_digit_sample_key_list = [
        per_sample_key for per_sample_key in sample_key_list
        if not per_sample_key.isdigit()
    ]
    if len(non_digit_sample_key_list) > 0:
        error_message_list.append(
            f'{per_archive_group_name} sample key not digit '
            f'{non_digit_sample_key_list[:5]}')

    per_min_sample_key_index, per_max_sample_key_index = 0, 0
    if len(non_digit_sample_key_list) == 0 and len(sample_key_list) > 0:
        sample_key_index_list = [
            int(per_sample_key) for per_sample_key in sample_key_list
        ]
        per_min_sample_key_index = min(sample_key_index_list)
        per_max_sample_key_index = max(sample_key_index_list)

        if per_min_sample_key_index != per_expected_min_sample_key_index or per_max_sample_key_index != per_expected_max_sample_key_index:
            error_message_list.append(
                f'{per_archive_group_name} sample key range not match '
                f'[{per_min_sample_key_index}, {per_max_sample_key_index}] != '
                f'[{per_expected_min_sample_key_index}, {per_expected_max_sample_key_index}]'
            )

        # 主干必须在区间内连号，缺号说明这一组的样本对没下全
        if per_max_sample_key_index - per_min_sample_key_index + 1 != len(
                sample_key_index_list):
            per_missing_sample_key_index_list = sorted(
                set(
                    range(per_min_sample_key_index, per_max_sample_key_index +
                          1)) - set(sample_key_index_list))
            error_message_list.append(
                f'{per_archive_group_name} sample key not continuous, missing '
                f'{len(per_missing_sample_key_index_list)} '
                f'{per_missing_sample_key_index_list[:5]}')

    sample_pair_list = [[
        per_sample_key,
        reference_member_entry_dict[per_sample_key],
        target_member_entry_dict[per_sample_key],
    ] for per_sample_key in sample_key_list]

    group_stat_dict = {
        'archive_part_num':
        len(per_archive_part_path_list),
        'total_archive_size':
        per_total_archive_size,
        'central_dir_entry_num':
        per_archive_info_dict.get('central_dir_entry_num', 0),
        'file_member_count':
        file_member_count,
        'dir_member_count':
        dir_member_count,
        'target_member_count':
        len(target_member_entry_dict),
        'reference_member_count':
        len(reference_member_entry_dict),
        'sample_pair_count':
        len(sample_key_list),
        'min_sample_key_index':
        per_min_sample_key_index,
        'max_sample_key_index':
        per_max_sample_key_index,
        'missing_reference_sample_key_num':
        len(missing_reference_sample_key_list),
        'missing_target_sample_key_num':
        len(missing_target_sample_key_list),
        'unknown_member_name_list':
        unknown_member_name_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }

    return [
        per_archive_group_name,
        sample_pair_list,
        group_stat_dict,
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    ]


def check_all_archive_group_complete(archive_group_list):
    """并行预检全部压缩包组，返回[每组样本对列表, 每组统计, 错误信息列表]

    17个组并行读中央目录只需要几分钟，比先跑几十小时解压再发现少样本便宜得多。
    """
    error_message_list = []

    found_archive_group_name_list = [
        per_archive_group[0] for per_archive_group in archive_group_list
    ]
    for per_expected_archive_group_name in sorted(
            ARCHIVE_GROUP_CONFIG_DICT.keys()):
        if per_expected_archive_group_name not in found_archive_group_name_list:
            error_message_list.append(
                f'missing archive group {per_expected_archive_group_name}')

    group_sample_pair_dict, group_stat_dict = {}, {}
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_archive_group_complete, archive_group_list),
                                     total=len(archive_group_list)):
            per_archive_group_name, per_sample_pair_list, per_group_stat_dict, per_error_message_list = per_check_result

            group_sample_pair_dict[
                per_archive_group_name] = per_sample_pair_list
            group_stat_dict[per_archive_group_name] = per_group_stat_dict
            error_message_list.extend(per_error_message_list)

            print('1111', per_archive_group_name, 'archive part',
                  per_group_stat_dict.get('archive_part_num',
                                          0), 'central dir entry',
                  per_group_stat_dict.get('central_dir_entry_num',
                                          0), 'file member',
                  per_group_stat_dict.get('file_member_count',
                                          0), 'sample pair',
                  per_group_stat_dict.get('sample_pair_count', 0),
                  'sample key range', [
                      per_group_stat_dict.get('min_sample_key_index', 0),
                      per_group_stat_dict.get('max_sample_key_index', 0),
                  ], 'error', len(per_error_message_list))

    # 跨组对账: 17个组的主干区间必须两两不重叠，并起来恰好覆盖1~920224
    sample_key_index_range_list = []
    total_sample_pair_count = 0
    for per_archive_group_name in sorted(group_stat_dict.keys()):
        per_group_stat_dict = group_stat_dict[per_archive_group_name]
        total_sample_pair_count += per_group_stat_dict.get(
            'sample_pair_count', 0)
        sample_key_index_range_list.append([
            per_group_stat_dict.get('min_sample_key_index', 0),
            per_group_stat_dict.get('max_sample_key_index', 0),
            per_archive_group_name,
        ])

    sample_key_index_range_list = sorted(sample_key_index_range_list)
    per_previous_max_sample_key_index = EXPECTED_MIN_SAMPLE_KEY_INDEX - 1
    for per_min_sample_key_index, per_max_sample_key_index, per_archive_group_name in sample_key_index_range_list:
        if per_min_sample_key_index != per_previous_max_sample_key_index + 1:
            error_message_list.append(
                f'{per_archive_group_name} sample key range not continuous with previous group '
                f'{per_min_sample_key_index} != {per_previous_max_sample_key_index} + 1'
            )
        per_previous_max_sample_key_index = per_max_sample_key_index

    if per_previous_max_sample_key_index != EXPECTED_MAX_SAMPLE_KEY_INDEX:
        error_message_list.append(
            f'max sample key index not match {per_previous_max_sample_key_index} != '
            f'{EXPECTED_MAX_SAMPLE_KEY_INDEX}')

    if total_sample_pair_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total sample pair count not match {total_sample_pair_count} != '
            f'{EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}')

    print('1111', 'total sample pair', total_sample_pair_count, 'expected',
          EXPECTED_TOTAL_SAMPLE_PAIR_COUNT)

    return group_sample_pair_dict, group_stat_dict, error_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非压缩包文件原样拷贝到目标目录，保持相对路径不变

    该数据集过滤掉无用信息后这里实测是空列表，保留这一步只是为了兼容后续新增文件。
    """
    per_file_relative_path, per_file_path = file_copy_pair

    save_file_path = os.path.join(save_dataset_path, per_file_relative_path)
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if os.path.isfile(save_file_path) and os.path.getsize(
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


def process_single_sample_pair_member(archive_reader, member_entry,
                                      save_member_path):
    """处理一个样本对里的单张图: 已完整落盘则跳过，否则解压落盘

    返回[宽, 高, 是否跳过, 错误信息]。断点重跑时跳过的图也要读文件头拿宽高,
    否则标注里会缺分辨率(分辨率分桶就得回头解码几百万张图)。
    """
    per_member_size = member_entry[4]

    if not EXTRACT_IMAGE_FILE_FLAG:
        # 只建索引模式: 图像继续留在原分卷zip里，不解压也就拿不到宽高
        return [0, 0, False, '']

    if os.path.isfile(save_member_path) and os.path.getsize(
            save_member_path) == per_member_size:
        per_image_width, per_image_height = get_image_size_from_file(
            save_member_path)

        return [per_image_width, per_image_height, True, '']

    per_image_width, per_image_height, per_error_message = extract_single_member_to_file(
        archive_reader, member_entry, save_member_path)

    return [per_image_width, per_image_height, False, per_error_message]


def process_single_sample_pair_chunk(sample_pair_chunk, save_dataset_path):
    """解压一块样本对，并把这一块的完整样本对写成一个jsonl汇总标注

    按样本对分块而不是按压缩包组分块: 01Blur一个组685GB/109480对，
    按组分块的话它会一个进程跑到天亮，其余31个进程全部闲置。
    """
    (per_archive_group_name, per_chunk_index, per_archive_part_path_list,
     per_sample_pair_list) = sample_pair_chunk

    per_degradation_type_list = ARCHIVE_GROUP_CONFIG_DICT[
        per_archive_group_name][8]

    save_target_image_dir_path = os.path.join(save_dataset_path,
                                              SAVE_IMAGE_DIR_NAME,
                                              per_archive_group_name,
                                              TARGET_IMAGE_DIR_NAME)
    save_reference_image_dir_path = os.path.join(save_dataset_path,
                                                 SAVE_IMAGE_DIR_NAME,
                                                 per_archive_group_name,
                                                 REFERENCE_IMAGE_DIR_NAME)
    os.makedirs(save_target_image_dir_path, exist_ok=True)
    os.makedirs(save_reference_image_dir_path, exist_ok=True)

    extract_file_count, skip_file_count, not_save_file_count = 0, 0, 0
    total_file_member_count = 0
    different_size_sample_pair_count = 0
    unknown_image_size_sample_pair_count = 0
    incomplete_sample_pair_list, error_message_list = [], []
    valid_annotation_line_list = []

    archive_reader = MultiPartZipReader(per_archive_part_path_list)
    try:
        for per_sample_pair in per_sample_pair_list:
            per_sample_key, per_reference_member_entry, per_target_member_entry = per_sample_pair

            per_reference_image_file_name = per_reference_member_entry[0]
            per_target_image_file_name = per_target_member_entry[0]

            total_file_member_count += 2

            per_reference_image_relative_path = f'{SAVE_IMAGE_DIR_NAME}/{per_archive_group_name}/{REFERENCE_IMAGE_DIR_NAME}/{per_reference_image_file_name}'
            per_target_image_relative_path = f'{SAVE_IMAGE_DIR_NAME}/{per_archive_group_name}/{TARGET_IMAGE_DIR_NAME}/{per_target_image_file_name}'

            per_reference_image_width, per_reference_image_height, per_reference_skip_flag, per_reference_error_message = process_single_sample_pair_member(
                archive_reader, per_reference_member_entry,
                os.path.join(save_reference_image_dir_path,
                             per_reference_image_file_name))
            per_target_image_width, per_target_image_height, per_target_skip_flag, per_target_error_message = process_single_sample_pair_member(
                archive_reader, per_target_member_entry,
                os.path.join(save_target_image_dir_path,
                             per_target_image_file_name))

            for per_member_error_message, per_member_skip_flag in [
                [per_reference_error_message, per_reference_skip_flag],
                [per_target_error_message, per_target_skip_flag],
            ]:
                if per_member_error_message:
                    continue
                if not EXTRACT_IMAGE_FILE_FLAG:
                    not_save_file_count += 1
                elif per_member_skip_flag:
                    skip_file_count += 1
                else:
                    extract_file_count += 1

            per_sample_pair_error_message_list = [
                per_member_error_message for per_member_error_message in
                [per_reference_error_message, per_target_error_message]
                if per_member_error_message
            ]
            if len(per_sample_pair_error_message_list) > 0:
                # 参考图或编辑后图任一张没完整落盘，这个样本对就是不完整样本对，
                # 必须硬上报，绝不能静默少写一行标注
                error_message_list.extend([
                    f'{per_archive_group_name}/{per_sample_key} {per_member_error_message}'
                    for per_member_error_message in
                    per_sample_pair_error_message_list
                ])
                if len(incomplete_sample_pair_list
                       ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                    incomplete_sample_pair_list.append({
                        'group_name':
                        per_archive_group_name,
                        'sample_key':
                        per_sample_key,
                        'reason_list':
                        per_sample_pair_error_message_list,
                    })
                continue

            per_same_size_flag = (
                per_reference_image_width == per_target_image_width
                and per_reference_image_height == per_target_image_height)
            if EXTRACT_IMAGE_FILE_FLAG:
                if per_reference_image_width <= 0 or per_target_image_width <= 0:
                    # 拿不到宽高不影响样本对完整性(图已完整落盘)，只统计上报
                    unknown_image_size_sample_pair_count += 1
                elif not per_same_size_flag:
                    different_size_sample_pair_count += 1

            # 完整有用信息的样本对: 该数据集原始文件里没有任何文本，
            # 所以这里不合成指令，只写数据集自带的客观信息 + 从图像头实测的属性
            per_save_annotation = {
                'sample_key': per_sample_key,
                'group_name': per_archive_group_name,
                'degradation_type_list': per_degradation_type_list,
                'task_type': 'image_restoration_edit',
                'reference_image_path': per_reference_image_relative_path,
                'reference_image_path_list': [
                    per_reference_image_relative_path,
                ],
                'reference_image_num': 1,
                'target_image_path': per_target_image_relative_path,
                'reference_image_width': per_reference_image_width,
                'reference_image_height': per_reference_image_height,
                'target_image_width': per_target_image_width,
                'target_image_height': per_target_image_height,
                'same_size_flag': per_same_size_flag,
                'reference_image_file_size': per_reference_member_entry[4],
                'target_image_file_size': per_target_member_entry[4],
            }
            valid_annotation_line_list.append(
                json.dumps(per_save_annotation, ensure_ascii=False))
    except Exception as e:
        print('7777', per_archive_group_name, per_chunk_index, e)
        error_message_list.append(
            f'{per_archive_group_name}/{per_chunk_index} process chunk failed {e}'
        )
    finally:
        archive_reader.close()

    save_annotation_path = os.path.join(save_dataset_path,
                                        SAVE_ANNOTATION_DIR_NAME,
                                        per_archive_group_name,
                                        f'{per_chunk_index:05d}.jsonl')
    save_temp_annotation_path = f'{save_annotation_path}.tmp'
    try:
        os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
        with open(save_temp_annotation_path, 'w',
                  encoding='UTF-8') as save_json_file:
            for per_valid_annotation_line in valid_annotation_line_list:
                save_json_file.write(f'{per_valid_annotation_line}\n')
        os.replace(save_temp_annotation_path, save_annotation_path)
    except Exception as e:
        error_message_list.append(
            f'{per_archive_group_name}/{per_chunk_index} save annotation failed {e}'
        )

    if extract_file_count + skip_file_count + not_save_file_count + len(
            incomplete_sample_pair_list) * 2 < total_file_member_count and len(
                error_message_list) == 0:
        # 处理数对不上但一个错误都没记，说明统计逻辑本身漏了分支，必须显式感知
        error_message_list.append(
            f'{per_archive_group_name}/{per_chunk_index} process file count not match '
            f'{extract_file_count} + {skip_file_count} + {not_save_file_count} '
            f'< {total_file_member_count}')

    return {
        'group_name': per_archive_group_name,
        'chunk_index': per_chunk_index,
        'input_sample_pair_count': len(per_sample_pair_list),
        'valid_sample_pair_count': len(valid_annotation_line_list),
        'total_file_member_count': total_file_member_count,
        'extract_file_count': extract_file_count,
        'skip_file_count': skip_file_count,
        'not_save_file_count': not_save_file_count,
        'different_size_sample_pair_count': different_size_sample_pair_count,
        'unknown_image_size_sample_pair_count':
        unknown_image_size_sample_pair_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_archive_group_name}/{per_chunk_index:05d}.jsonl',
        'incomplete_sample_pair_list': incomplete_sample_pair_list,
        'error_message_list': error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }


def get_all_sample_pair_chunk(archive_group_list, group_sample_pair_dict):
    """把每组的样本对列表切成固定大小的并行任务块"""
    archive_part_path_dict = {
        per_archive_group[0]: per_archive_group[2]
        for per_archive_group in archive_group_list
    }

    sample_pair_chunk_list = []
    for per_archive_group_name in sorted(group_sample_pair_dict.keys()):
        per_sample_pair_list = group_sample_pair_dict[per_archive_group_name]
        per_archive_part_path_list = archive_part_path_dict[
            per_archive_group_name]

        for per_chunk_index, per_chunk_start_index in enumerate(
                range(0, len(per_sample_pair_list), SAMPLE_PAIR_CHUNK_SIZE)):
            sample_pair_chunk_list.append([
                per_archive_group_name,
                per_chunk_index,
                per_archive_part_path_list,
                per_sample_pair_list[
                    per_chunk_start_index:per_chunk_start_index +
                    SAMPLE_PAIR_CHUNK_SIZE],
            ])

    return sample_pair_chunk_list


def check_single_group_dir_on_disk(group_check_pair):
    """可选的二次对账: os.walk单个组的输出目录，核对落盘图像数、同主干配对和标注行数"""
    (per_archive_group_name, per_group_image_dir_path,
     per_group_annotation_dir_path,
     per_expected_sample_pair_count) = group_check_pair

    error_message_list = []

    target_sample_key_dict, reference_sample_key_dict = {}, {}
    unknown_suffix_file_count = 0
    if not os.path.isdir(per_group_image_dir_path):
        error_message_list.append(
            f'{per_archive_group_name} image dir not exist')
    else:
        for per_image_dir_name, per_sample_key_dict in [
            [TARGET_IMAGE_DIR_NAME, target_sample_key_dict],
            [REFERENCE_IMAGE_DIR_NAME, reference_sample_key_dict],
        ]:
            per_image_dir_path = os.path.join(per_group_image_dir_path,
                                              per_image_dir_name)
            if not os.path.isdir(per_image_dir_path):
                error_message_list.append(
                    f'{per_archive_group_name}/{per_image_dir_name} dir not exist'
                )
                continue

            for per_file_name in os.listdir(per_image_dir_path):
                if not check_image_file_suffix(per_file_name):
                    unknown_suffix_file_count += 1
                    continue
                per_sample_key_dict[os.path.splitext(per_file_name)
                                    [0]] = per_file_name

    per_annotation_line_count = 0
    if not os.path.isdir(per_group_annotation_dir_path):
        error_message_list.append(
            f'{per_archive_group_name} annotation dir not exist')
    else:
        for per_annotation_name in sorted(
                os.listdir(per_group_annotation_dir_path)):
            if not per_annotation_name.endswith('.jsonl'):
                error_message_list.append(
                    f'{per_archive_group_name} unknown annotation file {per_annotation_name}'
                )
                continue

            with open(os.path.join(per_group_annotation_dir_path,
                                   per_annotation_name),
                      'r',
                      encoding='UTF-8') as load_json_file:
                for per_line in load_json_file:
                    if per_line.strip():
                        per_annotation_line_count += 1

    per_matched_sample_pair_count = len(
        set(target_sample_key_dict.keys())
        & set(reference_sample_key_dict.keys()))

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_archive_group_name} unknown suffix file num {unknown_suffix_file_count}'
        )
    if per_matched_sample_pair_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_group_name} on disk matched sample pair count not match '
            f'{per_matched_sample_pair_count} != {per_expected_sample_pair_count}'
        )
    if per_annotation_line_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_group_name} on disk annotation line count not match '
            f'{per_annotation_line_count} != {per_expected_sample_pair_count}')

    return [
        per_archive_group_name,
        len(target_sample_key_dict),
        len(reference_sample_key_dict),
        per_annotation_line_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, group_stat_dict):
    """可选的二次对账: 遍历输出目录核对每组落盘图像数、同主干配对和标注行数"""
    group_check_pair_list = []
    for per_archive_group_name in sorted(group_stat_dict.keys()):
        group_check_pair_list.append([
            per_archive_group_name,
            os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                         per_archive_group_name),
            os.path.join(save_dataset_path, SAVE_ANNOTATION_DIR_NAME,
                         per_archive_group_name),
            group_stat_dict[per_archive_group_name].get(
                'sample_pair_count', 0),
        ])

    error_message_list = []
    total_target_image_count, total_reference_image_count = 0, 0
    total_annotation_line_count = 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_group_dir_on_disk, group_check_pair_list),
                                     total=len(group_check_pair_list)):
            (_, per_target_image_count, per_reference_image_count,
             per_annotation_line_count,
             per_error_message_list) = per_check_result

            total_target_image_count += per_target_image_count
            total_reference_image_count += per_reference_image_count
            total_annotation_line_count += per_annotation_line_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk target image:', total_target_image_count,
          'on disk reference image:', total_reference_image_count,
          'on disk annotation line:', total_annotation_line_count)

    return error_message_list


def save_check_result(save_dataset_path, group_stat_dict, chunk_result_list):
    """汇总所有样本对块的解压与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_input_sample_pair_count, total_valid_sample_pair_count = 0, 0
    total_file_member_count = 0
    total_extract_file_count, total_skip_file_count = 0, 0
    total_not_save_file_count = 0
    total_different_size_sample_pair_count = 0
    total_unknown_image_size_sample_pair_count = 0
    group_valid_sample_pair_count_dict = collections.Counter()
    group_file_member_count_dict = collections.Counter()
    degradation_type_count_dict = collections.Counter()
    incomplete_sample_pair_list = []
    error_message_list, warning_message_list = [], []

    for per_chunk_result in chunk_result_list:
        per_archive_group_name = per_chunk_result['group_name']

        total_input_sample_pair_count += per_chunk_result[
            'input_sample_pair_count']
        total_valid_sample_pair_count += per_chunk_result[
            'valid_sample_pair_count']
        total_file_member_count += per_chunk_result['total_file_member_count']
        total_extract_file_count += per_chunk_result['extract_file_count']
        total_skip_file_count += per_chunk_result['skip_file_count']
        total_not_save_file_count += per_chunk_result['not_save_file_count']
        total_different_size_sample_pair_count += per_chunk_result[
            'different_size_sample_pair_count']
        total_unknown_image_size_sample_pair_count += per_chunk_result[
            'unknown_image_size_sample_pair_count']

        group_valid_sample_pair_count_dict[
            per_archive_group_name] += per_chunk_result[
                'valid_sample_pair_count']
        group_file_member_count_dict[
            per_archive_group_name] += per_chunk_result[
                'total_file_member_count']

        for per_degradation_type in ARCHIVE_GROUP_CONFIG_DICT[
                per_archive_group_name][8]:
            degradation_type_count_dict[
                per_degradation_type] += per_chunk_result[
                    'valid_sample_pair_count']

        incomplete_sample_pair_list.extend(
            per_chunk_result['incomplete_sample_pair_list'])

        if len(per_chunk_result['error_message_list']) > 0:
            print('7777', per_archive_group_name,
                  per_chunk_result['chunk_index'],
                  per_chunk_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_archive_group_name}/{per_chunk_result["chunk_index"]} error num '
                f'{len(per_chunk_result["error_message_list"])} '
                f'{per_chunk_result["error_message_list"][:3]}')

    # 每组样本对硬对账: 中央目录里数出来的样本对数是ground truth，
    # 写出的标注行数必须一个不少，少一行就说明有完整样本对被静默丢了
    for per_archive_group_name in sorted(ARCHIVE_GROUP_CONFIG_DICT.keys()):
        per_expected_sample_pair_count = ARCHIVE_GROUP_CONFIG_DICT[
            per_archive_group_name][4]
        per_valid_sample_pair_count = group_valid_sample_pair_count_dict.get(
            per_archive_group_name, 0)
        if per_valid_sample_pair_count != per_expected_sample_pair_count:
            error_message_list.append(
                f'{per_archive_group_name} valid sample pair count not match '
                f'{per_valid_sample_pair_count} != {per_expected_sample_pair_count}'
            )

        per_expected_file_member_count = ARCHIVE_GROUP_CONFIG_DICT[
            per_archive_group_name][3]
        per_file_member_count = group_file_member_count_dict.get(
            per_archive_group_name, 0)
        if per_file_member_count != per_expected_file_member_count:
            error_message_list.append(
                f'{per_archive_group_name} processed file member count not match '
                f'{per_file_member_count} != {per_expected_file_member_count}')

    if total_different_size_sample_pair_count > 0:
        # GT与LQ尺寸不一致不影响样本对完整性(两张图都在)，只打印告警不判失败
        warning_message_list.append(
            f'different size sample pair count {total_different_size_sample_pair_count}'
        )
    if total_unknown_image_size_sample_pair_count > 0:
        warning_message_list.append(
            f'unknown image size sample pair count {total_unknown_image_size_sample_pair_count}'
        )

    print('3333', 'total input sample pair:', total_input_sample_pair_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'total file member:', total_file_member_count, 'extract:',
          total_extract_file_count, 'skip:', total_skip_file_count,
          'not save:', total_not_save_file_count, 'incomplete sample pair:',
          len(incomplete_sample_pair_list), 'different size sample pair:',
          total_different_size_sample_pair_count,
          'unknown image size sample pair:',
          total_unknown_image_size_sample_pair_count)
    print('3333', 'group valid sample pair:',
          dict(group_valid_sample_pair_count_dict))
    print('3333', 'degradation type sample pair:',
          dict(degradation_type_count_dict))
    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        'image_edit',
        'dataset_sub_task_type':
        'image_restoration_edit',
        'has_original_text_prompt':
        False,
        'extract_image_file_flag':
        EXTRACT_IMAGE_FILE_FLAG,
        'total_archive_group_count':
        len(ARCHIVE_GROUP_CONFIG_DICT),
        'total_sample_pair_chunk_count':
        len(chunk_result_list),
        'total_input_sample_pair_count':
        total_input_sample_pair_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_file_member_count':
        total_file_member_count,
        'total_extract_file_count':
        total_extract_file_count,
        'total_skip_file_count':
        total_skip_file_count,
        'total_not_save_file_count':
        total_not_save_file_count,
        'total_incomplete_sample_pair_count':
        len(incomplete_sample_pair_list),
        'total_different_size_sample_pair_count':
        total_different_size_sample_pair_count,
        'total_unknown_image_size_sample_pair_count':
        total_unknown_image_size_sample_pair_count,
        'group_valid_sample_pair_count_dict':
        dict(group_valid_sample_pair_count_dict),
        'degradation_type_count_dict':
        dict(degradation_type_count_dict),
        'group_stat_dict':
        group_stat_dict,
        'incomplete_sample_pair_list':
        incomplete_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        warning_message_list,
        'check_error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    try:
        with open(save_check_result_path, 'w',
                  encoding='UTF-8') as save_json_file:
            json.dump(save_check_result_dict,
                      save_json_file,
                      ensure_ascii=False)
    except Exception as e:
        error_message_list.append(f'save check result failed {e}')

    if total_valid_sample_pair_count == 0:
        error_message_list.append('no valid sample pair found')
    if len(incomplete_sample_pair_list) > 0:
        error_message_list.append(
            f'incomplete sample pair count {len(incomplete_sample_pair_list)}')
    if total_input_sample_pair_count != total_valid_sample_pair_count:
        error_message_list.append(
            f'input sample pair count not match valid sample pair count '
            f'{total_input_sample_pair_count} != {total_valid_sample_pair_count}'
        )
    if total_valid_sample_pair_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} != '
            f'{EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}')
    if total_valid_sample_pair_count * 2 != total_file_member_count:
        error_message_list.append(
            f'total file member count not match {total_valid_sample_pair_count} * 2 != '
            f'{total_file_member_count}')
    if EXTRACT_IMAGE_FILE_FLAG:
        if total_extract_file_count + total_skip_file_count != total_file_member_count:
            error_message_list.append(
                f'extract + skip file count not match {total_extract_file_count} + '
                f'{total_skip_file_count} != {total_file_member_count}')
    elif total_not_save_file_count != total_file_member_count:
        error_message_list.append(
            f'not save file count not match {total_not_save_file_count} != '
            f'{total_file_member_count}')

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    if not os.path.exists(root_dataset_path):
        raise Exception(f'root dataset path not exist {root_dataset_path}')

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    file_copy_pair_list, archive_group_list = get_all_file_and_archive_group(
        root_dataset_path)

    print('1111', 'copy file', len(file_copy_pair_list), 'archive group',
          len(archive_group_list))
    for per_archive_group in archive_group_list:
        print('1111', 'group', per_archive_group[0], 'part',
              len(per_archive_group[2]))

    if len(archive_group_list) != len(ARCHIVE_GROUP_CONFIG_DICT):
        raise Exception(
            f'archive group num not match {len(archive_group_list)} != {len(ARCHIVE_GROUP_CONFIG_DICT)}'
        )

    # 解压前预检: 分卷/字节数/中央目录/GT-LQ配对/主干连号，任一不过直接中断，
    # 不白跑几十小时解压5.2T
    group_sample_pair_dict, group_stat_dict, precheck_error_message_list = check_all_archive_group_complete(
        archive_group_list)

    print('1111', 'precheck error', len(precheck_error_message_list),
          precheck_error_message_list[:20])
    if len(precheck_error_message_list) > 0:
        raise Exception(f'precheck failed {precheck_error_message_list[:20]}')

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

    sample_pair_chunk_list = get_all_sample_pair_chunk(archive_group_list,
                                                       group_sample_pair_dict)
    print('1111', 'sample pair chunk', len(sample_pair_chunk_list),
          'chunk size', SAMPLE_PAIR_CHUNK_SIZE)

    chunk_result_list = []
    extract_func = partial(process_single_sample_pair_chunk,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_chunk_result in tqdm(pool.imap_unordered(
                extract_func, sample_pair_chunk_list),
                                     total=len(sample_pair_chunk_list)):
            chunk_result_list.append(per_chunk_result)

            print('2222', per_chunk_result['group_name'],
                  per_chunk_result['chunk_index'], 'input sample pair',
                  per_chunk_result['input_sample_pair_count'],
                  'valid sample pair',
                  per_chunk_result['valid_sample_pair_count'], 'extract',
                  per_chunk_result['extract_file_count'], 'skip',
                  per_chunk_result['skip_file_count'], 'not save',
                  per_chunk_result['not_save_file_count'], 'incomplete',
                  len(per_chunk_result['incomplete_sample_pair_list']))

    check_error_message_list = save_check_result(save_dataset_path,
                                                 group_stat_dict,
                                                 chunk_result_list)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG and EXTRACT_IMAGE_FILE_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, group_stat_dict)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    print('3333', 'total error', len(all_error_message_list),
          all_error_message_list[:20])

    if len(all_error_message_list) > 0:
        # 拷贝/解压/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/FoundIR'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
