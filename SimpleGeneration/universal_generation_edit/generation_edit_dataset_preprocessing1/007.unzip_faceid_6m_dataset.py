import os
import re
import gzip
import json
import shutil
import tarfile
import collections

from tqdm import tqdm
from queue import Queue
from threading import Thread
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: FaceID-6M(Super-shuhe/FaceID-6M)
#
# 【数据集类型】纯文生图(text-to-image)数据集，不是图像编辑数据集。
# 它是从LAION-5B里按人脸检测 + 人物关键词过滤出来的"FaceID定制化"数据集，
# HF仓库声明的task_categories就是text-to-image。每个样本对是
#   "1条caption + 1张目标图 + 1张同一个人的人脸裁图 + 1个人脸id特征"，
# 没有编辑前后图、没有编辑指令、没有mask，所以:
#   - 可以走t2i链路(caption -> 目标图);
#   - 也可以走ti2i链路(人脸裁图当参考图 + caption -> 目标图，即InstantID那种ID保持生成);
#   - 不能当instruction-based editing(编辑)数据用。
#
# 【root_dataset_path实测原始保存规格】
# FaceID-6M/                          1.4T
# ├── .cache/huggingface/             695个文件(实测残留16个*.incomplete)(无用)
# │   └── trees/17d44283....json      HF落地的仓库文件清单(338条: path/size/lfs_sha256)，
# │                                   本脚本只读它来**逐片**校验字节数，不拷贝。
# │                                   注意这338条 = 336个分片 + .gitattributes +
# │                                   README.md，所以它所有条目size之和
# │                                   (1440472491808)比336个分片之和
# │                                   (1440472482799)多出9009字节，
# │                                   不能拿这个和当分片总大小的期望值
# ├── .gitattributes                  git lfs配置(无用)
# ├── README.md                       数据集说明(无用)
# └── laion_512.tar.gz.000 ~ .335     336片，前335片各4GiB、末片1658438639字节，
#                                     合计1440472482799字节，逐片与仓库清单完全一致
#
# 【实测确认的两个关键规格(决定了整个脚本的实现方式)】
# 1) 这336片是"单个tar.gz按字节切分"，不是336个独立压缩包:
#    只有 .000 带gzip魔数 1f 8b 08 00，.001/.002/.100/.335 开头全是裸数据，
#    所以必须按编号顺序拼成一个连续字节流才能解压(README也写明 cat laion_512.tar.gz.* > x.tar.gz)。
#    末片尾部8字节是合法gzip footer(CRC32=0xa8ba20e2, ISIZE=1294606336)，说明整流是完整的。
#    => 解压只能单流顺序做，不能像其他数据集那样用Pool按压缩包并行。
# 2) tar内只有一个顶层目录 laion_512/，成员只有普通文件和2个目录成员
#    (laion_512/ 和 laion_512/face/)，实测抽样的690万个成员里零链接、零其他类型。
#
# 【单个样本对的构成(README明确: jsonl的第0行对应 0.png / 0.npy / face/0.png)】
#   laion_512/<id>.png        目标图(生成后图)                          -> 有用
#                             实测 5696566.png 解码出来是1519x2048的JPEG，
#                             也就是"后缀叫png、内容其实是JPEG"，落盘保持原名不改后缀，
#                             因为id映射必须靠文件名，改名会破坏与jsonl行号的对应关系
#   laion_512/face/<id>.png   人脸裁图(参考图)，中位数仅7.5KB            -> 有用
#   laion_512/<id>.npy        实测全部恰好2176字节、shape=(512,) float32,
#                             是ArcFace人脸id embedding(README写成"landmarks"是错的,
#                             5个关键点只会是10个float、不可能512维)  -> 对本仓库无用，不落盘
#   *.jsonl                   文本描述，**按行号而不是里面写的路径**对应id -> 有用且必需
#                             (HF讨论区#2有人反馈"解压后所有文件平铺在一个目录里、很难找到
#                              这个jsonl"，说明它确实在包内)
#   实测id取值范围 0 ~ 6231807，成员总数约1870万(约623万样本 x 3个文件)。
#   实测tar成员**不是按id有序、也不是按目录分段**: 前118102个成员是root下png/npy交替，
#   之后是face/的大段，中途又切回root。所以任何"按段处理"的假设都不能用，
#   必须整流扫完才能知道某个id的三个文件是否齐备。
#
# 【无用信息(不整理进训练目录)】
#   .cache/ / .gitattributes / README.md;
#   laion_512/<id>.npy(见下面EXTRACT_FACE_EMBEDDING_FLAG的说明);
#   jsonl里的file_name字段(是作者本地路径，README明确要求忽略它、用行号定位)。
#
# 【.npy为什么不落盘(已核对下游取数代码)】
#   datasets/t2i_dataset.py  的__getitem__只产出 path/image/caption/bucket;
#   datasets/ti2i_dataset.py 的__getitem__只产出 path/reference_path/image/
#                            reference_image/caption/bucket;
#   在这两个dataset和t2i_common.py/ti2i_common.py里grep "npy|embed|arcface|face_id|
#   kps|landmark" 零命中(唯一一条是ti2i_common.py注释里提到模型内部的identity embedding,
#   与这个.npy无关)。也就是说这个512维ArcFace特征是InstantID那套IP-Adapter专用输入,
#   本仓库两条链路都不读它，落盘只会白占约623万个inode。
#   所以这里默认不落盘，但**仍然要解析并计数**(计入not_save_file_count),
#   这样"extract + skip + not_save == tar头里数出来的文件成员总数"这条硬对账依然成立,
#   不会因为跳过它而让对账失效; 每个样本对的标注里也记一个has_face_embedding字段,
#   说明原始数据里有没有这个特征，信息不丢。哪天要训InstantID类模型,
#   把EXTRACT_FACE_EMBEDDING_FLAG改成True重跑即可。
#
# 【必须显式感知的三个规格坑】
# 1) jsonl的行号就是样本id，**不能用jsonl里的file_name去定位图像**(README原文:
#    "Ignore the file paths listed in the .jsonl file and use the line number instead")。
#    由此推出一条硬校验: jsonl行数必须大于实测到的最大id，否则说明jsonl和图像不是同一版本，
#    行号映射会**整体错位**，落出来的标注全是错的图文对。这是本数据集最大的坑，必须硬拦。
# 2) jsonl的文件名和字段名**未能实测**(它在1.4T单流里的位置未知，全量列一遍要几小时,
#    本脚本编写时的后台扫描只走到约690万/1870万个成员，尚未遇到它)。
#    从官方训练代码 utils/dataset.py 可知每行至少有
#      file_name / additional_feature(文本) / bbox / landmarks /
#      penult_id_embed_file / clip_from_seg_file / clip_from_orig_file / seg_map_orig_file
#    其中additional_feature就是caption(HF imagefolder的metadata约定: file_name + 附加列)。
#    所以文本字段用**候选key列表**去取，并统计实际命中的是哪个key;
#    一个key都命中不到就是硬错误(而不是静默把caption当空、把整个数据集判成0个有效样本)。
# 3) 后缀不可信: 目标图后缀是.png但内容是JPEG。本脚本只按后缀分类成员、不解码图像
#    (623万张图解码一遍要很久，真实宽高留给preprocessing2那一步去补)。
#
# 【本脚本的处理口径】
# - 解压前预检(全是O(1)读，几秒钟跑完，避免白跑几小时):
#   根目录只允许已知的无用文件 + 336个分片，出现未知文件/目录一律显式上报;
#   压缩包组名、分片数、分片编号000..335连号、逐片字节数与仓库清单一致、总字节数一致;
#   首片必须有gzip魔数、末片尾8字节的gzip footer必须能解析;
#   任一不过直接抛异常;
# - 解压: MultiPartArchiveReader把336片拼成连续字节流 ->
#   gzip.GzipFile -> tarfile.open(mode='r|')。
#   **这里故意不用tarfile的mode='r|gz'**: tarfile自带的gz解压走内部_Stream,
#   只调zlib.decompressobj、**不校验gzip尾部的CRC32与ISIZE**，分片少一片/被截断时
#   很可能只是"少解出一批文件"然后静默正常结束; 而gzip.GzipFile在读到流末尾时会
#   校验CRC32和ISIZE，不一致直接抛BadGzipFile。所以tar成员读完后还要把GzipFile
#   drain到EOF，用这道校验来证明"336片齐全且一个字节都没坏"，
#   这是本数据集"不漏样本"最硬的保证;
# - 主线程顺序解流 + EXTRACT_THREAD_NUM个写线程并发落盘(单流解压本身不能并行,
#   但NAS上1200多万个小文件的写延迟必须靠多线程藏起来);
# - 每个成员写盘后立刻校验落盘大小 == tar头里的大小(只看存在性会把半截图当正常样本);
# - 落盘按 id // IMAGE_SUB_DIR_SAMPLE_NUM 分桶，避免623万个文件挤在一个目录里
#   把GPFS的目录元数据打爆; 真实路径写进汇总标注，下游只按标注取样本;
# - 原始jsonl原样保留一份到unzip_raw_annotations/(它是"行号 -> 样本id"映射的唯一依据),
#   再按行号合并成 unzip_annotations/laion_512_<shard>.jsonl 汇总标注,
#   保留原行的全部属性，另补 sample_key/image_path/face_image_path/
#   has_face_embedding/caption/caption_key_name/dataset_task_type;
# - "包含完整有用信息的样本对" = caption非空 + 目标图落盘 + 人脸参考图落盘, 三者齐备;
# - 硬对账(任一不过就抛异常、退出码非0，与任何开关无关):
#   extract + skip + not_save == tar头里数出来的文件成员总数;
#   必须读到tar流结束; GzipFile必须drain到EOF且CRC/ISIZE校验通过;
#   写盘失败数必须为0; 重名成员数、未知成员数必须为0;
#   必须且只能有1个原始jsonl(多个的话行号映射就是歧义的);
#   jsonl行数必须 > 最大样本id(否则行号映射整体错位);
#   同一份jsonl里命中的文本字段名必须唯一;
#   落盘目标图数 == 有效样本对数 + orphan数(少了是漏保存、多了是漏整理);
# - 不完整样本对(空caption / 有caption没图 / 有图没caption / jsonl行不合法)
#   按ALLOW_INCOMPLETE_SAMPLE_PAIR_FLAG决定是硬失败还是只告警，默认硬失败,
#   详见该开关处的说明; 无论取什么值，计数与样例都会留在校验报告里;
# - 任何一环出错都汇总后抛异常，不静默跑过。
#
# 【尚未实测、以首次运行为验证的项】
#   jsonl的文件名与字段名; tar成员的精确总数;
#   jsonl行号与图像id是否严格一一对应(即有没有"某id有caption却没图"或反之)。
#   上面的硬对账就是为了让这几项一旦与预期不符时**显式失败**，而不是静默少样本;
#   第一次运行按报出来的真实数字再决定是否放宽ALLOW_INCOMPLETE_SAMPLE_PAIR_FLAG。
# ==============================================================================

# laion_512.tar.gz.000 这种"整包.tar.gz + 3位数字字节分片号"的命名
ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+\.tar\.gz)\.(?P<part>\d+)$'),
    re.compile(r'^(?P<prefix>.+\.tar\.gz)$'),
]

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(695个文件，含16个*.incomplete)
# .gitattributes   git lfs配置
# README.md        数据集说明
# .gitignore/CACHEDIR.TAG/.DS_Store  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    'CACHEDIR.TAG',
    '.DS_Store',
]

# 实测该数据集只有这一个压缩包组
EXPECTED_ARCHIVE_GROUP_NAME = 'laion_512.tar.gz'

# 实测分片数与字节数(与仓库清单完全一致)，数量/字节不对说明下载不全或被截断
EXPECTED_ARCHIVE_PART_NUM = 336

EXPECTED_ARCHIVE_PART_SIZE = 4294967296

EXPECTED_ARCHIVE_LAST_PART_SIZE = 1658438639

# 336个分片的期望总字节数(= 前335片各4GiB + 末片1658438639字节 = 1440472482799)。
# 注意**不能**直接拿.cache/huggingface/trees清单里所有条目的size求和当这个值:
# 那份清单有338条(336个分片 + .gitattributes 2461字节 + README.md 6548字节),
# 把两个非压缩包文件也算进去会让期望值多出9009字节，
# 而下面累加的total_archive_part_size只累加336个分片，
# 两者天然差9009，总大小校验就会必然失败。
# 这里改成由分片规格推导，避免再踩这个坑
EXPECTED_ARCHIVE_TOTAL_SIZE = (
    EXPECTED_ARCHIVE_PART_NUM -
    1) * EXPECTED_ARCHIVE_PART_SIZE + EXPECTED_ARCHIVE_LAST_PART_SIZE

# gzip头魔数与尾部footer。footer是"CRC32 + ISIZE(解压后字节数 mod 2^32)"共8字节，
# 实测末片的这两个值如下; 与实测值不同只告警不判失败(仓库有可能更新过数据),
# 但**解析不出来**就是硬错误(说明末片被截断)
GZIP_MAGIC_BYTES = b'\x1f\x8b'

GZIP_FOOTER_SIZE = 8

EXPECTED_GZIP_FOOTER_CRC32 = 0xa8ba20e2

EXPECTED_GZIP_FOOTER_ISIZE = 1294606336

# tar内唯一的顶层目录，以及人脸裁图所在的子目录
ARCHIVE_TOP_DIR_NAME = 'laion_512'

ARCHIVE_FACE_DIR_NAME = 'face'

# 样本id就是文件名前缀(纯数字)，它同时是jsonl里的行号
SAMPLE_ID_NAME_PATTERN = re.compile(r'^\d+$')

# 目标图/人脸裁图的后缀。实测只有.png(内容其实是JPEG)，
# 这里放宽成常见图像后缀，出现新后缀也能正常归类而不是被当成未知成员
IMAGE_FILE_SUFFIX_LIST = [
    '.png',
    '.jpg',
    '.jpeg',
    '.webp',
    '.bmp',
]

# 人脸id特征后缀与实测字节数(全部恰好2176字节 = 128字节npy头 + 512 * 4字节float32)
FACE_EMBEDDING_FILE_SUFFIX = '.npy'

EXPECTED_FACE_EMBEDDING_FILE_SIZE = 2176

ANNOTATION_FILE_SUFFIX_LIST = [
    '.jsonl',
    '.json',
]

# 每行json里的文本提示候选字段。additional_feature来自官方训练代码
# utils/dataset.py 的 item["additional_feature"]，是最可能的那个;
# 其余三个是常见别名兜底。一个都命中不到就是硬错误(见文件头规格坑2)
ANNOTATION_TEXT_KEY_NAME_LIST = [
    'additional_feature',
    'text',
    'caption',
    'prompt',
]

# 每行除文本外应该齐备的有用属性(来自官方训练代码)，缺失只上报不丢样本:
# bbox是人脸框(训练时用来裁CLIP输入)、landmarks是5点关键点(用来画kps条件图)
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    'bbox',
    'landmarks',
]

# 三类成员是否落盘。
# 目标图与人脸裁图是t2i/ti2i两条链路真正要读的像素数据，必须落盘;
# 人脸id特征(.npy)两条链路都不读，默认不落盘(详见文件头说明)，
# 但仍然会被解析、计数并记进标注的has_face_embedding字段
EXTRACT_TARGET_IMAGE_FLAG = True

EXTRACT_FACE_IMAGE_FLAG = True

EXTRACT_FACE_EMBEDDING_FLAG = False

# 不完整样本对(caption为空 / 有caption没图 / 有图没caption / jsonl行不合法)
# 是当硬错误抛异常，还是只当告警上报。
#
# 默认False(硬失败)。因为"jsonl行数与图像id是否严格一一对应"这条**没能实测**
# (实测到最大id是6231807，而README只说"约6M"，所以有可能存在
#  "jsonl覆盖了某个id但那张图被过滤掉了"的情况)。
# 默认硬失败是为了让第一次运行把真相打出来，而不是静默少样本:
#   - 如果报出来的数是0，说明严格一一对应，理想情况;
#   - 如果报出来一个稳定的非0数，去unzip_check_missing_images.json里看
#     missing_image_sample_key_list / orphan_image_sample_key_list 的样例,
#     确认那些id在原始tar里确实没有对应文件(即数据集原始规格就是这样、不是我们漏处理),
#     再把这个开关改成True重跑; 此时不完整样本对不会写进汇总标注、
#     但会连同计数和样例一起留在校验报告里，信息不丢。
# 无论开关取什么值，写盘失败/流截断/重名/未知成员/成员数对不上这些
# "我们自己处理出问题"的错误都始终是硬失败。
ALLOW_INCOMPLETE_SAMPLE_PAIR_FLAG = False

# 是否在解压后再os.walk一遍输出目录做二次对账。
# 默认False: 1200多万个小文件的os.walk在NAS上要跑非常久，而解压时已经做了
# "写盘后立刻校验落盘大小 == tar头大小" + "extract+skip+not_save == tar成员总数"
# + "GzipFile drain到EOF的CRC/ISIZE校验"三道对账，已经能保证每个成员都被处理且完整落盘
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

SAVE_IMAGE_DIR_NAME = 'images'

SAVE_FACE_EMBEDDING_DIR_NAME = 'face_embeddings'

SAVE_RAW_ANNOTATION_DIR_NAME = 'unzip_raw_annotations'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_DUPLICATE_MEMBER_DIR_NAME = 'unzip_duplicate_members'

SAVE_UNKNOWN_MEMBER_DIR_NAME = 'unzip_unknown_members'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

DATASET_TASK_TYPE = 'text_to_image_faceid_customization'

# 每个落盘子目录放多少个样本(623万个文件平铺在一个目录里会把GPFS目录元数据打爆)
IMAGE_SUB_DIR_SAMPLE_NUM = 10000

# 每个汇总标注jsonl分片放多少行(约623万行 -> 约63个分片)
ANNOTATION_SHARD_SAMPLE_NUM = 100000

# 按样本id下标记录"该id的某类成员是否已落盘"的位图容量。
# 实测最大id是6231807，这里留到32Mi(每个位图32MB)，
# 超出容量的id会被显式上报成错误而不是静默丢掉。
# 用固定容量的bytearray而不是set: 一是省内存(32MB vs 约400MB),
# 二是多写线程共享时"bytearray单字节下标赋值"在GIL下是原子的、不需要加锁，
# 而set在扩容时被并发写会有竞态
MAX_SAMPLE_ID_CAPACITY = 33554432

# README声明约6M个text-image pair，按实测最大id 6231807给一个宽松区间，
# 只做软校验(打印告警)，因为官方没给精确条数，不能拿来当硬性失败条件
EXPECTED_SAMPLE_PAIR_COUNT_RANGE = [5500000, 6300000]

# 最多回传/落盘多少条异常路径样例，避免623万量级下把内存和校验报告打爆
MAX_REPORT_PATH_NUM = 10000

PROCESS_NUM = 32

EXTRACT_THREAD_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024


class MultiPartArchiveReader:
    """把按字节切分的多个分片压缩包拼接成一个只读的连续字节流

    该数据集的336个分片是"单个tar.gz按字节切分"，必须按编号顺序拼接才能解压，
    所以这个类在本脚本里不是兼容用的可选项，而是解压能不能跑起来的前提。
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


class SampleIdFlagArray:
    """按样本id下标记录一个小整数标记的定长位图

    目标图/人脸裁图存的是"落盘后缀在IMAGE_FILE_SUFFIX_LIST里的序号 + 1"(0表示没落盘),
    这样一个位图既当存在性标记、又能还原出落盘时用的真实后缀;
    人脸id特征存的就是0/1。

    多写线程共享同一个实例，只做"单字节下标赋值"和"单字节下标读取",
    在GIL下是原子的，不需要加锁; 容量固定、永不扩容，所以也没有扩容竞态。
    """

    def __init__(self, flag_array_capacity=MAX_SAMPLE_ID_CAPACITY):
        self.flag_array_capacity = flag_array_capacity
        self.flag_bytearray = bytearray(flag_array_capacity)

    def check_sample_id_in_capacity(self, per_sample_id):
        return 0 <= per_sample_id < self.flag_array_capacity

    def set_flag(self, per_sample_id, flag_value=1):
        self.flag_bytearray[per_sample_id] = flag_value

        return

    def get_flag(self, per_sample_id):
        if not self.check_sample_id_in_capacity(per_sample_id):
            return 0

        return self.flag_bytearray[per_sample_id]

    def get_set_flag_count(self):
        return self.flag_array_capacity - self.flag_bytearray.count(0)


def check_skip_file_or_dir(per_file_relative_path):
    """过滤掉.cache、.gitattributes、README.md这几个不需要整理的文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """判断是否是图像文件后缀"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


def match_archive_file_name(per_file_name):
    """匹配压缩包文件名，返回[压缩包组名, 分片编号]，不是压缩包时返回[None, '']"""
    for per_archive_file_name_pattern in ARCHIVE_FILE_NAME_PATTERN_LIST:
        per_match_result = per_archive_file_name_pattern.match(per_file_name)
        if not per_match_result:
            continue

        per_archive_group_name = per_match_result.group('prefix')
        per_match_group_dict = per_match_result.groupdict()
        per_archive_part_index = ''
        if per_match_group_dict.get('part', None) is not None:
            per_archive_part_index = per_match_group_dict['part']

        return [per_archive_group_name, per_archive_part_index]

    return [None, '']


def get_archive_part_sort_key(per_archive_part_index):
    """分片编号排序key: 纯数字编号按数值排序，字母编号按位数优先再按字典序排序

    返回值统一是[编号类型, 数值编号, 字母编号]三元组，保证不同命名风格之间也能比较。
    这里排序错一位整个gzip流就全废了，所以不能直接按字符串排序:
    part.9 会排到 part.10 后面；也不能只按[位数, 字符串]排序:
    part.100 会排到 part.89 前面导致整个流错位。
    """
    per_archive_part_index = per_archive_part_index or ''
    if per_archive_part_index.isdigit():
        return [0, int(per_archive_part_index), '']

    return [1, len(per_archive_part_index), per_archive_part_index]


def get_hf_tree_expect_file_size_dict(root_dataset_path):
    """从.cache/huggingface/trees/*.json(HF落地的仓库文件清单)里读出权威的期望字节数

    返回 {仓库相对路径: 期望字节数}。实测这份清单有338条(336个分片 +
    .gitattributes + README.md)，本地文件与它逐字节数一致，
    所以它是"分片齐不齐、有没有被截断"最直接的ground truth。
    """
    root_tree_path = os.path.join(root_dataset_path, '.cache', 'huggingface',
                                  'trees')

    expect_file_size_dict = {}
    if not os.path.isdir(root_tree_path):
        return expect_file_size_dict

    for per_tree_file_name in sorted(os.listdir(root_tree_path)):
        if os.path.splitext(per_tree_file_name)[1].lower() != '.json':
            continue

        per_tree_file_path = os.path.join(root_tree_path, per_tree_file_name)
        try:
            with open(per_tree_file_path, 'r',
                      encoding='UTF-8') as load_json_file:
                per_tree_dict = json.load(load_json_file)
        except Exception as e:
            print('4444', per_tree_file_path, e)
            continue

        if not isinstance(per_tree_dict, dict):
            continue

        for per_file_relative_path, per_file_info in per_tree_dict.get(
                'files', {}).items():
            if not isinstance(per_file_info, dict):
                continue

            per_file_relative_path = per_file_relative_path.replace('\\', '/')
            per_file_size = per_file_info.get('size', None)
            if per_file_size is not None:
                expect_file_size_dict[per_file_relative_path] = per_file_size

    return expect_file_size_dict


def check_single_archive_part_gzip_head(per_archive_part_path):
    """O(1)预检首片是否带gzip魔数，返回错误信息列表"""
    error_message_list = []
    try:
        with open(per_archive_part_path, 'rb') as load_archive_file:
            per_archive_head_bytes = load_archive_file.read(
                len(GZIP_MAGIC_BYTES))

        if per_archive_head_bytes != GZIP_MAGIC_BYTES:
            error_message_list.append(
                f'first archive part gzip magic broken {per_archive_part_path} {per_archive_head_bytes}'
            )
    except Exception as e:
        error_message_list.append(
            f'read archive head failed {per_archive_part_path} {e}')

    return error_message_list


def check_single_archive_part_gzip_tail(per_archive_part_path):
    """O(1)预检末片尾部8字节的gzip footer，返回[错误信息列表, 告警信息列表]

    footer = CRC32(4字节) + ISIZE(解压后字节数 mod 2^32, 4字节)，都是小端。
    解析不出来(文件太小/读失败)是硬错误，说明末片被截断;
    解析出来但与实测值不同只告警，因为仓库有可能更新过数据。
    """
    error_message_list, warning_message_list = [], []
    try:
        per_archive_part_size = os.path.getsize(per_archive_part_path)
        if per_archive_part_size < GZIP_FOOTER_SIZE:
            error_message_list.append(
                f'last archive part too small {per_archive_part_path} {per_archive_part_size}'
            )

            return error_message_list, warning_message_list

        with open(per_archive_part_path, 'rb') as load_archive_file:
            load_archive_file.seek(per_archive_part_size - GZIP_FOOTER_SIZE)
            per_archive_tail_bytes = load_archive_file.read(GZIP_FOOTER_SIZE)

        if len(per_archive_tail_bytes) != GZIP_FOOTER_SIZE:
            error_message_list.append(
                f'read last archive part gzip footer failed {per_archive_part_path}'
            )

            return error_message_list, warning_message_list

        per_footer_crc32 = int.from_bytes(per_archive_tail_bytes[0:4],
                                          byteorder='little')
        per_footer_isize = int.from_bytes(per_archive_tail_bytes[4:8],
                                          byteorder='little')

        print('1111', 'gzip footer crc32:', hex(per_footer_crc32), 'isize:',
              per_footer_isize)

        if per_footer_crc32 != EXPECTED_GZIP_FOOTER_CRC32:
            warning_message_list.append(
                f'gzip footer crc32 not match expected {hex(per_footer_crc32)} != {hex(EXPECTED_GZIP_FOOTER_CRC32)}'
            )
        if per_footer_isize != EXPECTED_GZIP_FOOTER_ISIZE:
            warning_message_list.append(
                f'gzip footer isize not match expected {per_footer_isize} != {EXPECTED_GZIP_FOOTER_ISIZE}'
            )
    except Exception as e:
        error_message_list.append(
            f'read last archive part gzip footer failed {per_archive_part_path} {e}'
        )

    return error_message_list, warning_message_list


def check_required_archive_complete(root_dataset_path, archive_group_list):
    """解压前预检: 根目录构成、压缩包组名、分片数、分片连号、逐片字节数、gzip头尾

    返回[错误信息列表, 告警信息列表]。
    336片是一个连续gzip流，缺一片或某片被截断都会让后面的样本**全部**解不出来，
    而且tarfile在流式模式下很可能只是"少解出一批文件"然后静默结束，
    所以这些检查必须在跑几小时解压之前先做完。
    """
    error_message_list, warning_message_list = [], []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')

        return error_message_list, warning_message_list

    # 根目录下除了已知的无用项和336个分片之外不应该有别的东西，
    # 出现新文件/新目录必须显式上报，否则会被静默漏处理
    for per_name in sorted(os.listdir(root_dataset_path)):
        if check_skip_file_or_dir(per_name):
            continue

        per_path = os.path.join(root_dataset_path, per_name)
        if os.path.isdir(per_path):
            error_message_list.append(f'unknown dir in root dir {per_name}')
            continue

        per_archive_group_name, _ = match_archive_file_name(per_name)
        if per_archive_group_name is None:
            error_message_list.append(f'unknown file in root dir {per_name}')

    if len(archive_group_list) != 1:
        error_message_list.append(
            f'archive group num not match {len(archive_group_list)} != 1')

        return error_message_list, warning_message_list

    per_archive_group_key, per_archive_group_name, per_archive_relative_dir, per_archive_part_index_list, per_archive_part_path_list = archive_group_list[
        0]

    if per_archive_group_name != EXPECTED_ARCHIVE_GROUP_NAME:
        error_message_list.append(
            f'archive group name not match {per_archive_group_name} != {EXPECTED_ARCHIVE_GROUP_NAME}'
        )
    if per_archive_relative_dir not in ['', '.']:
        error_message_list.append(
            f'archive group not in root dir {per_archive_relative_dir}')

    print('1111', per_archive_group_name, 'archive part:',
          len(per_archive_part_path_list), 'expected archive part:',
          EXPECTED_ARCHIVE_PART_NUM)

    if len(per_archive_part_path_list) != EXPECTED_ARCHIVE_PART_NUM:
        error_message_list.append(
            f'archive part num not match {len(per_archive_part_path_list)} != {EXPECTED_ARCHIVE_PART_NUM}'
        )

    # 分片编号必须是0..N-1连号，缺号说明有分片没下载下来
    exist_part_index_dict = {
        int(per_archive_part_index): 1
        for per_archive_part_index in per_archive_part_index_list
        if per_archive_part_index.isdigit()
    }
    if len(exist_part_index_dict) != len(per_archive_part_index_list):
        error_message_list.append(
            f'archive part index not all digit {per_archive_part_index_list[:10]}'
        )

    missing_part_index_list = sorted(
        set(range(0, EXPECTED_ARCHIVE_PART_NUM)) -
        set(exist_part_index_dict.keys()))
    if len(missing_part_index_list) > 0:
        error_message_list.append(
            f'archive part index not continuous, missing index {missing_part_index_list[:10]}'
        )

    # 逐片字节数: 优先用仓库清单当ground truth，清单缺失时退回
    # "除末片外都是4GiB"的实测规格
    expect_file_size_dict = get_hf_tree_expect_file_size_dict(
        root_dataset_path)
    print('1111', 'hf tree expect file:', len(expect_file_size_dict))

    total_archive_part_size = 0
    for per_archive_part_path in per_archive_part_path_list:
        per_archive_part_relative_path = os.path.relpath(
            per_archive_part_path, root_dataset_path).replace('\\', '/')
        try:
            per_archive_part_size = os.path.getsize(per_archive_part_path)
        except Exception as e:
            error_message_list.append(
                f'read archive part size failed {per_archive_part_relative_path} {e}'
            )
            continue

        total_archive_part_size += per_archive_part_size

        per_expect_part_size = expect_file_size_dict.get(
            per_archive_part_relative_path, None)
        if per_expect_part_size is None:
            # 清单里没有这一片，退回实测规格: 末片1658438639字节、其余4GiB
            per_expect_part_size = EXPECTED_ARCHIVE_LAST_PART_SIZE if per_archive_part_path == per_archive_part_path_list[
                -1] else EXPECTED_ARCHIVE_PART_SIZE
            warning_message_list.append(
                f'archive part not in hf tree {per_archive_part_relative_path}'
            )

        if per_archive_part_size != per_expect_part_size:
            error_message_list.append(
                f'archive part size not match {per_archive_part_relative_path} {per_archive_part_size} != {per_expect_part_size}'
            )

    print('1111', 'total archive part size:', total_archive_part_size,
          'expected total size:', EXPECTED_ARCHIVE_TOTAL_SIZE)

    if total_archive_part_size != EXPECTED_ARCHIVE_TOTAL_SIZE:
        error_message_list.append(
            f'archive total size not match {total_archive_part_size} != {EXPECTED_ARCHIVE_TOTAL_SIZE}'
        )

    if len(per_archive_part_path_list) > 0:
        error_message_list.extend(
            check_single_archive_part_gzip_head(per_archive_part_path_list[0]))

        per_tail_error_message_list, per_tail_warning_message_list = check_single_archive_part_gzip_tail(
            per_archive_part_path_list[-1])
        error_message_list.extend(per_tail_error_message_list)
        warning_message_list.extend(per_tail_warning_message_list)

    return error_message_list, warning_message_list


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表和按分片归组后的压缩包列表"""
    file_copy_pair_list = []
    archive_part_path_dict = {}
    for per_root_path, per_dir_name_list, per_file_name_list in os.walk(
            root_dataset_path):
        # .cache里有695个文件，直接在遍历时剪掉整棵子树，不要走进去
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

            per_archive_group_name, per_archive_part_index = match_archive_file_name(
                per_file_name)

            if per_archive_group_name is None:
                file_copy_pair_list.append([
                    per_file_relative_path,
                    per_file_path,
                ])
                continue

            per_archive_group_key = f'{per_file_relative_dir}/{per_archive_group_name}'
            if per_archive_group_key not in archive_part_path_dict:
                archive_part_path_dict[per_archive_group_key] = [
                    per_archive_group_name,
                    per_file_relative_dir,
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
        per_archive_part_index_list = [
            per_archive_part_index
            for per_archive_part_index, _ in per_archive_part_list
        ]
        per_archive_part_path_list = [
            per_archive_part_path
            for _, per_archive_part_path in per_archive_part_list
        ]
        archive_group_list.append([
            per_archive_group_key,
            per_archive_group_name,
            per_archive_relative_dir,
            per_archive_part_index_list,
            per_archive_part_path_list,
        ])

    file_copy_pair_list = sorted(file_copy_pair_list, key=lambda x: x[0])

    return file_copy_pair_list, archive_group_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非压缩包文件原样拷贝到目标目录，保持相对路径不变

    该数据集过滤掉无用信息后这里是空列表，保留这一步只是为了兼容后续新增文件。
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


def save_single_member_bytes(save_member_path, per_member_bytes,
                             per_member_size):
    """把内存里的成员字节流落盘并立刻校验落盘大小，返回错误信息(空串表示成功)

    必须校验落盘大小: 只看文件存在与否的话，写到一半失败留下的半截图
    会被后面的校验当成正常样本，等于静默把坏样本混进训练集。
    """
    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_member_path, 'wb') as save_member_file:
            save_member_file.write(per_member_bytes)
    except Exception as e:
        return f'write member failed {save_member_path} {e}'

    if os.path.getsize(save_member_path) != per_member_size:
        return f'write member size not match {save_member_path}'

    return ''


def save_single_member_stream_file(load_member_file, save_member_path,
                                   per_member_size):
    """流式把tar成员写盘并立刻校验落盘大小，返回错误信息(空串表示成功)

    只给整份jsonl标注用: 它可能有1GB以上，一次read()进内存不划算。
    落盘大小校验同样不能省: 流被截断时copyfileobj会静默写出一个长度不足的半截文件。
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


def parse_single_member_name(per_member_name):
    """把tar成员名解析成[成员类别, 样本id, 落盘后缀, 错误信息]

    成员类别: target_image / face_image / face_embedding / annotation / unknown。
    annotation的样本id恒为-1(它是整份jsonl，不属于某一个样本)。
    解析不出来的一律归成unknown并带上错误信息，绝不默认当图像处理。
    """
    per_member_name = per_member_name.replace('\\', '/')
    per_name_part_list = [
        per_name_part for per_name_part in per_member_name.split('/')
        if per_name_part
    ]

    if len(per_name_part_list
           ) < 2 or per_name_part_list[0] != ARCHIVE_TOP_DIR_NAME:
        return [
            'unknown', -1, '',
            f'member not under {ARCHIVE_TOP_DIR_NAME}/ {per_member_name}'
        ]

    per_member_dir_name_list = per_name_part_list[1:-1]
    per_member_file_name = per_name_part_list[-1]
    per_member_file_name_prefix, per_member_file_name_suffix = os.path.splitext(
        per_member_file_name)
    per_member_file_name_suffix = per_member_file_name_suffix.lower()

    if len(per_member_dir_name_list) > 1:
        return [
            'unknown', -1, '', f'member dir depth not match {per_member_name}'
        ]

    per_member_dir_name = per_member_dir_name_list[0] if len(
        per_member_dir_name_list) == 1 else ''

    if per_member_dir_name not in ['', ARCHIVE_FACE_DIR_NAME]:
        return ['unknown', -1, '', f'unknown member dir {per_member_name}']

    # 整份jsonl标注只会在顶层目录下
    if per_member_dir_name == '' and per_member_file_name_suffix in ANNOTATION_FILE_SUFFIX_LIST:
        return ['annotation', -1, per_member_file_name_suffix, '']

    if not SAMPLE_ID_NAME_PATTERN.match(per_member_file_name_prefix):
        return [
            'unknown', -1, '',
            f'member name prefix not a sample id {per_member_name}'
        ]

    per_sample_id = int(per_member_file_name_prefix)

    if per_member_dir_name == '':
        if per_member_file_name_suffix in IMAGE_FILE_SUFFIX_LIST:
            return [
                'target_image', per_sample_id, per_member_file_name_suffix, ''
            ]
        if per_member_file_name_suffix == FACE_EMBEDDING_FILE_SUFFIX:
            return [
                'face_embedding', per_sample_id, per_member_file_name_suffix,
                ''
            ]

        return [
            'unknown', -1, '',
            f'unknown member suffix in top dir {per_member_name}'
        ]

    if per_member_file_name_suffix in IMAGE_FILE_SUFFIX_LIST:
        return ['face_image', per_sample_id, per_member_file_name_suffix, '']

    return [
        'unknown', -1, '',
        f'unknown member suffix in face dir {per_member_name}'
    ]


def get_save_member_relative_path(per_member_category, per_sample_id,
                                  per_member_file_name_suffix):
    """按成员类别拼出落盘的相对路径(相对save_dataset_path)

    按 id // IMAGE_SUB_DIR_SAMPLE_NUM 分桶，避免623万个文件平铺在一个目录里
    把GPFS的目录元数据打爆。文件名保持"原始id + 原始后缀"不变:
    id是与jsonl行号对应的唯一凭据，改名就破坏了图文对应关系。
    """
    per_sub_dir_name = f'{per_sample_id // IMAGE_SUB_DIR_SAMPLE_NUM:05d}'
    per_member_file_name = f'{per_sample_id}{per_member_file_name_suffix}'

    if per_member_category == 'target_image':
        return f'{SAVE_IMAGE_DIR_NAME}/{ARCHIVE_TOP_DIR_NAME}/{per_sub_dir_name}/{per_member_file_name}'

    if per_member_category == 'face_image':
        return f'{SAVE_IMAGE_DIR_NAME}/{ARCHIVE_TOP_DIR_NAME}/{ARCHIVE_FACE_DIR_NAME}/{per_sub_dir_name}/{per_member_file_name}'

    return f'{SAVE_FACE_EMBEDDING_DIR_NAME}/{ARCHIVE_TOP_DIR_NAME}/{per_sub_dir_name}/{per_member_file_name}'


def read_single_tar_member_bytes(load_tar_file, per_member,
                                 per_archive_group_name, error_message_list):
    """把单个tar成员完整读进内存并校验字节数，失败返回None

    必须校验读出来的长度 == tar头里的大小: 流被截断时read会静默返回半截数据，
    直接写盘就等于落了一个坏图。
    """
    load_member_file = get_single_tar_member_file(load_tar_file, per_member,
                                                  per_archive_group_name,
                                                  error_message_list)
    if load_member_file is None:
        return None

    try:
        per_member_bytes = load_member_file.read()
    except Exception as e:
        print('6666', per_archive_group_name, per_member.name, e)
        error_message_list.append(f'read member failed {per_member.name} {e}')

        return None

    if len(per_member_bytes) != per_member.size:
        error_message_list.append(
            f'member data truncated {per_member.name} {len(per_member_bytes)} != {per_member.size}'
        )

        return None

    return per_member_bytes


def get_single_tar_member_file(load_tar_file, per_member,
                               per_archive_group_name, error_message_list):
    """取出单个tar成员的只读文件对象，失败返回None"""
    try:
        load_member_file = load_tar_file.extractfile(per_member)
    except Exception as e:
        print('6666', per_archive_group_name, per_member.name, e)
        error_message_list.append(
            f'extract member failed {per_member.name} {e}')

        return None

    if load_member_file is None:
        print('6666', per_archive_group_name, per_member.name)
        error_message_list.append(f'extract member failed {per_member.name}')

        return None

    return load_member_file


def write_single_member_file(write_task_queue, write_result_dict,
                             member_flag_array_dict):
    """写线程: 从队列取出解压好的成员字节流并落盘，落盘成功才置样本id标记

    每个写线程持有**自己的**write_result_dict，主线程join之后再汇总。
    这样计数器不需要加锁(dict[key] += 1 在多线程下不是原子操作，共享会丢计数)。
    member_flag_array_dict里的位图是共享的，但只做单字节下标赋值，GIL下是原子的。
    """
    while True:
        write_task = write_task_queue.get()
        if write_task is None:
            write_task_queue.task_done()
            break

        per_member_category, per_sample_id, per_flag_value, save_member_path, per_member_bytes, per_member_size = write_task

        per_save_error_message = save_single_member_bytes(
            save_member_path, per_member_bytes, per_member_size)
        if per_save_error_message:
            print('6666', per_save_error_message)
            if len(write_result_dict['error_message_list']
                   ) < MAX_REPORT_PATH_NUM:
                write_result_dict['error_message_list'].append(
                    per_save_error_message)
            write_result_dict['write_fail_file_count'] += 1
        else:
            write_result_dict['extract_file_count'] += 1
            # 只有真正完整落盘的成员才置标记，后面的样本对完整性校验才可信
            if per_sample_id >= 0 and per_member_category in member_flag_array_dict:
                member_flag_array_dict[per_member_category].set_flag(
                    per_sample_id, per_flag_value)

        write_task_queue.task_done()

    return


def process_single_archive_group(archive_group, save_dataset_path,
                                 member_flag_array_dict):
    """单流顺序解压整个压缩包组(336个字节分片 = 一个tar.gz)，多写线程并发落盘

    这里故意不用tarfile的mode='r|gz': tarfile自带的gz解压走内部_Stream，
    只调zlib.decompressobj、**不校验gzip尾部的CRC32与ISIZE**，
    分片少一片或被截断时很可能只是"少解出一批文件"然后静默正常结束。
    改成 gzip.GzipFile -> tarfile.open(mode='r|') 之后，
    tar成员读完再把GzipFile drain到EOF，GzipFile会校验CRC32和ISIZE，
    不一致直接抛BadGzipFile。这是"336片齐全且一个字节都没坏"最硬的保证。
    """
    per_archive_group_key, per_archive_group_name, per_archive_relative_dir, per_archive_part_index_list, per_archive_part_path_list = archive_group

    save_raw_annotation_dir_path = os.path.join(save_dataset_path,
                                                SAVE_RAW_ANNOTATION_DIR_NAME)
    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_MEMBER_DIR_NAME)
    save_unknown_dir_path = os.path.join(save_dataset_path,
                                         SAVE_UNKNOWN_MEMBER_DIR_NAME)

    write_task_queue = Queue(maxsize=EXTRACT_THREAD_NUM * 2)
    write_result_dict_list, write_thread_list = [], []
    for _ in range(EXTRACT_THREAD_NUM):
        per_write_result_dict = {
            'extract_file_count': 0,
            'write_fail_file_count': 0,
            'error_message_list': [],
        }
        per_write_thread = Thread(target=write_single_member_file,
                                  args=(
                                      write_task_queue,
                                      per_write_result_dict,
                                      member_flag_array_dict,
                                  ),
                                  daemon=True)
        per_write_thread.start()
        write_result_dict_list.append(per_write_result_dict)
        write_thread_list.append(per_write_thread)

    skip_file_count, not_save_file_count = 0, 0
    total_file_member_count, total_dir_member_count = 0, 0
    skip_name_member_count = 0
    duplicate_member_count, unknown_member_count = 0, 0
    member_category_count_dict = collections.Counter()
    max_sample_id = -1
    unknown_member_name_list, duplicate_member_name_list = [], []
    out_of_capacity_sample_id_list = []
    save_raw_annotation_relative_path_list = []
    reach_tar_end, reach_gzip_end = False, False
    error_message_list = []

    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    load_gzip_file = None
    try:
        load_gzip_file = gzip.GzipFile(fileobj=archive_reader, mode='rb')
        with tarfile.open(fileobj=load_gzip_file, mode='r|') as load_tar_file:
            for per_member in tqdm(load_tar_file):
                per_member_name = per_member.name.replace('\\',
                                                          '/').lstrip('/')
                per_member_name = os.path.normpath(per_member_name)
                if per_member_name.startswith('..'):
                    print('5555', per_archive_group_name, per_member.name)
                    error_message_list.append(
                        f'illegal member name {per_member.name}')
                    continue

                if check_skip_file_or_dir(per_member_name):
                    # 按名跳过的成员(.DS_Store这类垃圾文件)既不落盘也不计入
                    # total_file_member_count，单独计数只为让报告里能看见它们存在
                    skip_name_member_count += 1
                    continue

                if per_member.isdir():
                    # 实测只有 laion_512/ 和 laion_512/face/ 两个目录成员。
                    # 落盘目录是按id重新分桶的，所以这里不需要照搬原目录结构
                    total_dir_member_count += 1
                    continue

                if not per_member.isfile():
                    # 实测抽样只有普通文件，出现链接等类型必须显式上报
                    print('5555', per_archive_group_name, per_member.name,
                          per_member.type)
                    error_message_list.append(
                        f'not a regular file {per_member.name} {per_member.type}'
                    )
                    continue

                total_file_member_count += 1

                per_member_category, per_sample_id, per_member_file_name_suffix, per_parse_error_message = parse_single_member_name(
                    per_member_name)
                member_category_count_dict[per_member_category] += 1

                if per_member_category == 'unknown':
                    # 既不是三类样本文件也不是jsonl的成员: 无法映射到任何样本id，
                    # 但也不能静默丢，改写到独立目录保留并上报
                    unknown_member_count += 1
                    if len(unknown_member_name_list) < MAX_REPORT_PATH_NUM:
                        unknown_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'unknown member {per_member_name} {per_parse_error_message}'
                    )

                    per_member_bytes = read_single_tar_member_bytes(
                        load_tar_file, per_member, per_archive_group_name,
                        error_message_list)
                    if per_member_bytes is None:
                        not_save_file_count += 1
                        continue

                    save_member_path = os.path.join(save_unknown_dir_path,
                                                    per_member_name)
                    per_save_error_message = save_single_member_bytes(
                        save_member_path, per_member_bytes, per_member.size)
                    if per_save_error_message:
                        error_message_list.append(per_save_error_message)

                    # 未知成员无论有没有成功另存都算"已处理过"，
                    # 这样成员数对账(extract + skip + not_save == 总数)始终闭合;
                    # 出错本身已经进了error_message_list，不会被漏掉
                    not_save_file_count += 1
                    continue

                if per_member_category == 'annotation':
                    # 整份jsonl是"行号 -> 样本id"映射的唯一依据，必须原样落盘一份。
                    # 它可能有几百MB到1GB(623万行)，所以在主线程里流式写盘、
                    # 不read()进内存、也不进写线程队列(队列里挤一个1GB的bytes会把内存打爆);
                    # 行解析留到解压结束后再单独扫一遍
                    load_member_file = get_single_tar_member_file(
                        load_tar_file, per_member, per_archive_group_name,
                        error_message_list)
                    if load_member_file is None:
                        not_save_file_count += 1
                        continue

                    per_save_annotation_relative_path = f'{SAVE_RAW_ANNOTATION_DIR_NAME}/{os.path.basename(per_member_name)}'
                    save_member_path = os.path.join(
                        save_raw_annotation_dir_path,
                        os.path.basename(per_member_name))
                    per_save_error_message = save_single_member_stream_file(
                        load_member_file, save_member_path, per_member.size)
                    if per_save_error_message:
                        error_message_list.append(per_save_error_message)
                    else:
                        # 只有完整落盘的jsonl才能当"行号 -> 样本id"的映射依据
                        save_raw_annotation_relative_path_list.append(
                            per_save_annotation_relative_path)
                        print('2222', per_archive_group_name,
                              'raw annotation:', per_member_name,
                              per_member.size)

                    not_save_file_count += 1
                    continue

                # 到这里per_member_category只会是三类样本文件之一
                max_sample_id = max(max_sample_id, per_sample_id)

                per_member_flag_array = member_flag_array_dict[
                    per_member_category]
                if not per_member_flag_array.check_sample_id_in_capacity(
                        per_sample_id):
                    # 位图容量兜不住这个id: 宁可显式失败，也不能静默丢样本
                    if len(out_of_capacity_sample_id_list
                           ) < MAX_REPORT_PATH_NUM:
                        out_of_capacity_sample_id_list.append(per_member_name)
                    error_message_list.append(
                        f'sample id out of flag array capacity {per_member_name}'
                    )
                    not_save_file_count += 1
                    continue

                per_extract_member_flag = {
                    'target_image': EXTRACT_TARGET_IMAGE_FLAG,
                    'face_image': EXTRACT_FACE_IMAGE_FLAG,
                    'face_embedding': EXTRACT_FACE_EMBEDDING_FLAG,
                }[per_member_category]

                per_member_is_duplicate = per_member_flag_array.get_flag(
                    per_sample_id) > 0

                if per_member_category == 'face_embedding' and per_member.size != EXPECTED_FACE_EMBEDDING_FILE_SIZE:
                    # 实测全部恰好2176字节(512维float32)，变了说明特征规格变了，必须显式感知
                    error_message_list.append(
                        f'face embedding size not match {per_member_name} {per_member.size} != {EXPECTED_FACE_EMBEDDING_FILE_SIZE}'
                    )

                if not per_extract_member_flag:
                    # 不落盘的类别(默认是face_embedding): 仍然要置标记和计数，
                    # 这样"extract + skip + not_save == 文件成员总数"的对账依然成立,
                    # 标注里的has_face_embedding也才有依据
                    if per_member_is_duplicate:
                        duplicate_member_count += 1
                        if len(duplicate_member_name_list
                               ) < MAX_REPORT_PATH_NUM:
                            duplicate_member_name_list.append(per_member_name)
                        error_message_list.append(
                            f'duplicate member name {per_member_name}')
                    else:
                        per_member_flag_array.set_flag(per_sample_id, 1)

                    not_save_file_count += 1
                    continue

                # 图像类成员的标记值 = 落盘后缀在IMAGE_FILE_SUFFIX_LIST里的序号 + 1,
                # 这样一张位图既当存在性标记、又能在生成标注时还原出真实后缀;
                # 人脸id特征不是图像，标记值固定为1(只当存在性用)
                per_member_flag_value = 1
                if per_member_file_name_suffix in IMAGE_FILE_SUFFIX_LIST:
                    per_member_flag_value = IMAGE_FILE_SUFFIX_LIST.index(
                        per_member_file_name_suffix) + 1

                if per_member_is_duplicate:
                    # 同一个id的同类成员出现两次时按名写盘会互相覆盖，
                    # 改写到独立目录保留数据并上报，不静默丢样本
                    duplicate_member_count += 1
                    if len(duplicate_member_name_list) < MAX_REPORT_PATH_NUM:
                        duplicate_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'duplicate member name {per_member_name}')

                    per_member_bytes = read_single_tar_member_bytes(
                        load_tar_file, per_member, per_archive_group_name,
                        error_message_list)
                    if per_member_bytes is None:
                        not_save_file_count += 1
                        continue

                    save_member_path = os.path.join(
                        save_duplicate_dir_path, f'{duplicate_member_count}',
                        per_member_name)
                    per_save_error_message = save_single_member_bytes(
                        save_member_path, per_member_bytes, per_member.size)
                    if per_save_error_message:
                        error_message_list.append(per_save_error_message)

                    # 重名成员无论有没有成功另存都算"已处理过"，理由同未知成员
                    not_save_file_count += 1
                    continue

                per_save_member_relative_path = get_save_member_relative_path(
                    per_member_category, per_sample_id,
                    per_member_file_name_suffix)
                save_member_path = os.path.join(save_dataset_path,
                                                per_save_member_relative_path)

                if os.path.exists(save_member_path) and os.path.getsize(
                        save_member_path) == per_member.size:
                    # 断点续跑: 已经完整落盘过的成员直接跳过，标记要在主线程里补上
                    skip_file_count += 1
                    per_member_flag_array.set_flag(per_sample_id,
                                                   per_member_flag_value)
                    continue

                per_member_bytes = read_single_tar_member_bytes(
                    load_tar_file, per_member, per_archive_group_name,
                    error_message_list)
                if per_member_bytes is None:
                    # 读失败的成员已经进了error_message_list(硬失败)，
                    # 这里也要计数让成员数对账闭合，否则会同时报两个错、掩盖真正的原因
                    not_save_file_count += 1
                    continue

                write_task_queue.put([
                    per_member_category,
                    per_sample_id,
                    per_member_flag_value,
                    save_member_path,
                    per_member_bytes,
                    per_member.size,
                ])

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
        # 分片不全、gzip CRC不符或NAS读失败时保留已解压出的文件，但必须上报，
        # 不能静默少样本
        print('7777', per_archive_group_name, len(per_archive_part_path_list),
              e)
        error_message_list.append(f'read archive failed {e}')
    finally:
        for _ in write_thread_list:
            write_task_queue.put(None)
        for per_write_thread in write_thread_list:
            per_write_thread.join()
        if load_gzip_file is not None:
            try:
                load_gzip_file.close()
            except Exception as e:
                error_message_list.append(f'close gzip file failed {e}')
        archive_reader.close()

    extract_file_count, write_fail_file_count = 0, 0
    for per_write_result_dict in write_result_dict_list:
        extract_file_count += per_write_result_dict['extract_file_count']
        write_fail_file_count += per_write_result_dict['write_fail_file_count']
        error_message_list.extend(per_write_result_dict['error_message_list'])

    if not reach_tar_end:
        error_message_list.append(
            'not reach tar stream end, archive may be truncated')
    if not reach_gzip_end:
        error_message_list.append(
            'not reach gzip stream end, gzip crc32/isize not verified')
    if write_fail_file_count > 0:
        error_message_list.append(
            f'write member fail count {write_fail_file_count}')
    if duplicate_member_count > 0:
        # 实测tar内同一个id的同类成员只出现一次，出现重名说明规格变了，
        # 数据已另存到unzip_duplicate_members/但必须人工确认
        error_message_list.append(
            f'duplicate member count {duplicate_member_count}')
    if unknown_member_count > 0:
        # 出现既不是三类样本文件也不是jsonl的成员，说明规格变了，
        # 数据已另存到unzip_unknown_members/但必须人工确认
        error_message_list.append(
            f'unknown member count {unknown_member_count}')

    # 核心对账: tar头里数出来的每个文件成员，都必须落到
    # extract(新落盘) / skip(已存在跳过) / not_save(按配置不落盘或另存) 三者之一。
    # 这是"每个成员都被处理过、一个都没漏"最直接的证据
    if extract_file_count + skip_file_count + not_save_file_count != total_file_member_count:
        error_message_list.append(
            f'process file count not match: {extract_file_count} + {skip_file_count} + {not_save_file_count} != {total_file_member_count}'
        )

    if len(save_raw_annotation_relative_path_list) != 1:
        # 0个: 没有任何文本提示，整个数据集都不可训练;
        # 多个: "行号 -> 样本id"的映射就是歧义的，不能猜，必须人工确认后再改脚本
        error_message_list.append(
            f'raw annotation file num not match {len(save_raw_annotation_relative_path_list)} != 1 {save_raw_annotation_relative_path_list}'
        )

    return {
        'archive_group_name': per_archive_group_name,
        'archive_part_num': len(per_archive_part_path_list),
        'extract_file_count': extract_file_count,
        'skip_file_count': skip_file_count,
        'not_save_file_count': not_save_file_count,
        'write_fail_file_count': write_fail_file_count,
        'total_file_member_count': total_file_member_count,
        'total_dir_member_count': total_dir_member_count,
        'skip_name_member_count': skip_name_member_count,
        'duplicate_member_count': duplicate_member_count,
        'unknown_member_count': unknown_member_count,
        'member_category_count_dict': dict(member_category_count_dict),
        'max_sample_id': max_sample_id,
        'reach_tar_end': reach_tar_end,
        'reach_gzip_end': reach_gzip_end,
        'save_raw_annotation_relative_path_list':
        save_raw_annotation_relative_path_list,
        'unknown_member_name_list': unknown_member_name_list,
        'duplicate_member_name_list': duplicate_member_name_list,
        'out_of_capacity_sample_id_list': out_of_capacity_sample_id_list,
        'error_message_list': error_message_list[:MAX_REPORT_PATH_NUM],
    }


def get_single_annotation_error_message_list(per_annotation, per_sample_key):
    """校验单行标注的有用信息是否完整，返回[文本提示, 命中的文本字段名, 错误信息列表]

    文本提示必须非空(文生图样本对没有caption就不可训练);
    bbox/landmarks这些其他有用属性缺失只上报、不丢样本。
    """
    error_message_list = []

    per_caption, per_caption_key_name = '', ''
    for per_text_key_name in ANNOTATION_TEXT_KEY_NAME_LIST:
        per_text_value = per_annotation.get(per_text_key_name, '')
        if isinstance(per_text_value, str) and len(per_text_value.strip()) > 0:
            per_caption = per_text_value
            per_caption_key_name = per_text_key_name
            break

    if not per_caption:
        error_message_list.append(f'{per_sample_key} empty caption')

    for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            error_message_list.append(
                f'{per_sample_key} miss key {per_key_name}')

    return per_caption, per_caption_key_name, error_message_list


def merge_annotation_and_check_sample_pair(save_dataset_path, archive_result,
                                           member_flag_array_dict):
    """按行号把原始jsonl合并成汇总标注，并逐条核对样本对是否完整

    README原文: "Ignore the file paths listed in the .jsonl file and use the
    line number instead to locate the corresponding image, face, and .npy
    files. For example, the 0th line in the .jsonl file corresponds to 0.png,
    0.npy, and ./face/0.png"。
    所以这里**只用行号当样本id**，jsonl里的file_name等路径字段原样保留供溯源、
    但绝不拿来拼落盘路径。

    "包含完整有用信息的样本对" = caption非空 + 目标图落盘 + 人脸参考图落盘。
    """
    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    target_image_flag_array = member_flag_array_dict['target_image']
    face_image_flag_array = member_flag_array_dict['face_image']
    face_embedding_flag_array = member_flag_array_dict['face_embedding']

    # 用来做orphan(有图没caption)双向差集: 被有效样本对用掉的id在这里置1
    used_sample_id_flag_array = SampleIdFlagArray()

    max_sample_id = archive_result['max_sample_id']
    save_raw_annotation_relative_path_list = archive_result[
        'save_raw_annotation_relative_path_list']

    total_annotation_line_count, valid_sample_pair_count = 0, 0
    illegal_annotation_line_count = 0
    missing_target_image_count, missing_face_image_count = 0, 0
    missing_face_embedding_count, empty_caption_count = 0, 0
    caption_key_name_count_dict = collections.Counter()
    annotation_key_name_count_dict = collections.Counter()
    missing_image_sample_key_list, invalid_annotation_message_list = [], []
    error_message_list = []

    save_annotation_relative_path_list = []
    save_json_file, current_annotation_shard_index = None, -1

    try:
        for per_raw_annotation_relative_path in save_raw_annotation_relative_path_list:
            per_raw_annotation_path = os.path.join(
                save_dataset_path, per_raw_annotation_relative_path)

            with open(per_raw_annotation_path,
                      'r',
                      encoding='UTF-8',
                      errors='replace') as load_json_file:
                for per_line_index, per_line in enumerate(
                        tqdm(load_json_file)):
                    total_annotation_line_count += 1

                    # 行号就是样本id，所以**空行也必须占一个行号**，
                    # 绝不能skip掉，否则后面所有行的id都会整体前移错位
                    per_sample_id = per_line_index
                    per_sample_key = f'{ARCHIVE_TOP_DIR_NAME}/{per_sample_id}'

                    per_annotation_shard_index = per_sample_id // ANNOTATION_SHARD_SAMPLE_NUM
                    if per_annotation_shard_index != current_annotation_shard_index:
                        if save_json_file is not None:
                            save_json_file.close()
                        current_annotation_shard_index = per_annotation_shard_index
                        per_save_annotation_relative_path = f'{SAVE_ANNOTATION_DIR_NAME}/{ARCHIVE_TOP_DIR_NAME}_{per_annotation_shard_index:05d}.jsonl'
                        save_json_file = open(os.path.join(
                            save_dataset_path,
                            per_save_annotation_relative_path),
                                              'w',
                                              encoding='UTF-8')
                        save_annotation_relative_path_list.append(
                            per_save_annotation_relative_path)

                    per_line = per_line.strip()
                    if not per_line:
                        illegal_annotation_line_count += 1
                        if len(invalid_annotation_message_list
                               ) < MAX_REPORT_PATH_NUM:
                            invalid_annotation_message_list.append(
                                f'{per_sample_key} empty annotation line')
                        continue

                    try:
                        per_annotation = json.loads(per_line)
                    except Exception as e:
                        illegal_annotation_line_count += 1
                        if len(invalid_annotation_message_list
                               ) < MAX_REPORT_PATH_NUM:
                            invalid_annotation_message_list.append(
                                f'{per_sample_key} load annotation failed {e}')
                        continue

                    if not isinstance(per_annotation, dict):
                        illegal_annotation_line_count += 1
                        if len(invalid_annotation_message_list
                               ) < MAX_REPORT_PATH_NUM:
                            invalid_annotation_message_list.append(
                                f'{per_sample_key} annotation not a dict')
                        continue

                    for per_annotation_key in per_annotation.keys():
                        annotation_key_name_count_dict[per_annotation_key] += 1

                    per_caption, per_caption_key_name, per_annotation_error_message_list = get_single_annotation_error_message_list(
                        per_annotation, per_sample_key)
                    if len(per_annotation_error_message_list) > 0:
                        if len(invalid_annotation_message_list
                               ) < MAX_REPORT_PATH_NUM:
                            invalid_annotation_message_list.extend(
                                per_annotation_error_message_list)

                    if not per_caption:
                        empty_caption_count += 1
                        continue

                    caption_key_name_count_dict[per_caption_key_name] += 1

                    per_target_image_flag = target_image_flag_array.get_flag(
                        per_sample_id)
                    per_face_image_flag = face_image_flag_array.get_flag(
                        per_sample_id)
                    per_face_embedding_flag = face_embedding_flag_array.get_flag(
                        per_sample_id)

                    per_missing_image_name_list = []
                    if per_target_image_flag <= 0:
                        missing_target_image_count += 1
                        per_missing_image_name_list.append('target_image')
                    if per_face_image_flag <= 0:
                        missing_face_image_count += 1
                        per_missing_image_name_list.append('face_image')
                    if per_face_embedding_flag <= 0:
                        # 人脸id特征缺失不影响t2i/ti2i训练，只计数不算样本对不完整
                        missing_face_embedding_count += 1

                    if len(per_missing_image_name_list) > 0:
                        if len(missing_image_sample_key_list
                               ) < MAX_REPORT_PATH_NUM:
                            missing_image_sample_key_list.append(
                                f'{per_sample_key} miss {per_missing_image_name_list}'
                            )
                        continue

                    per_target_image_relative_path = get_save_member_relative_path(
                        'target_image', per_sample_id,
                        IMAGE_FILE_SUFFIX_LIST[per_target_image_flag - 1])
                    per_face_image_relative_path = get_save_member_relative_path(
                        'face_image', per_sample_id,
                        IMAGE_FILE_SUFFIX_LIST[per_face_image_flag - 1])

                    # 完整有用信息的样本对: 保留原行的全部属性(bbox/landmarks等),
                    # 再补上落盘路径、样本key、命中的文本字段名与任务类型,
                    # 方便下游直接按行取样本，不需要为了拿caption去扫623万个小文件
                    per_save_annotation = {
                        'sample_key': per_sample_id,
                        'dataset_task_type': DATASET_TASK_TYPE,
                        'image_path': per_target_image_relative_path,
                        'face_image_path': per_face_image_relative_path,
                        'has_face_embedding': per_face_embedding_flag > 0,
                        'caption': per_caption,
                        'caption_key_name': per_caption_key_name,
                    }
                    for per_annotation_key, per_annotation_value in per_annotation.items(
                    ):
                        if per_annotation_key == per_caption_key_name:
                            continue
                        per_save_annotation[
                            per_annotation_key] = per_annotation_value

                    save_json_file.write(
                        f'{json.dumps(per_save_annotation, ensure_ascii=False)}\n'
                    )
                    valid_sample_pair_count += 1
                    used_sample_id_flag_array.set_flag(per_sample_id, 1)
    except Exception as e:
        print('9999', 'merge annotation failed', e)
        error_message_list.append(f'merge annotation failed {e}')
    finally:
        if save_json_file is not None:
            save_json_file.close()

    # orphan: 目标图落盘了但没有任何可用caption(caption为空、行不合法、
    # 或者jsonl行数根本没覆盖到这个id)。没有文本提示的图不可训练，只能算orphan
    orphan_image_count, orphan_image_sample_key_list = 0, []
    for per_sample_id in range(0, max(max_sample_id + 1, 0)):
        if target_image_flag_array.get_flag(per_sample_id) <= 0:
            continue
        if used_sample_id_flag_array.get_flag(per_sample_id) > 0:
            continue

        orphan_image_count += 1
        if len(orphan_image_sample_key_list) < MAX_REPORT_PATH_NUM:
            orphan_image_sample_key_list.append(
                f'{ARCHIVE_TOP_DIR_NAME}/{per_sample_id}')

    # 行号映射的硬校验: jsonl行数必须覆盖到最大id，
    # 否则说明jsonl和图像不是同一版本，行号 -> id的映射会**整体错位**,
    # 落出来的标注全是错的图文对，这比少几个样本严重得多
    if total_annotation_line_count <= max_sample_id:
        error_message_list.append(
            f'annotation line count not cover max sample id {total_annotation_line_count} <= {max_sample_id}'
        )

    if len(caption_key_name_count_dict) > 1:
        # 同一份jsonl里文本字段名不统一，说明规格判断有问题，必须人工确认
        error_message_list.append(
            f'annotation text key name not unique {dict(caption_key_name_count_dict)}'
        )

    return {
        'total_annotation_line_count':
        total_annotation_line_count,
        'valid_sample_pair_count':
        valid_sample_pair_count,
        'illegal_annotation_line_count':
        illegal_annotation_line_count,
        'empty_caption_count':
        empty_caption_count,
        'missing_target_image_count':
        missing_target_image_count,
        'missing_face_image_count':
        missing_face_image_count,
        'missing_face_embedding_count':
        missing_face_embedding_count,
        'orphan_image_count':
        orphan_image_count,
        'on_disk_target_image_count':
        target_image_flag_array.get_set_flag_count(),
        'on_disk_face_image_count':
        face_image_flag_array.get_set_flag_count(),
        'on_disk_face_embedding_count':
        face_embedding_flag_array.get_set_flag_count(),
        'caption_key_name_count_dict':
        dict(caption_key_name_count_dict),
        'annotation_key_name_count_dict':
        dict(annotation_key_name_count_dict),
        'save_annotation_relative_path_list':
        save_annotation_relative_path_list,
        'missing_image_sample_key_list':
        missing_image_sample_key_list,
        'orphan_image_sample_key_list':
        orphan_image_sample_key_list,
        'invalid_annotation_message_list':
        invalid_annotation_message_list,
        'error_message_list':
        error_message_list[:MAX_REPORT_PATH_NUM],
    }


def check_single_sub_dir_on_disk(sub_dir_check_pair):
    """可选的二次对账: os.walk单个落盘子目录，核对文件数"""
    per_sub_dir_relative_path, per_sub_dir_path = sub_dir_check_pair

    error_message_list = []
    if not os.path.exists(per_sub_dir_path):
        error_message_list.append(
            f'{per_sub_dir_relative_path} sub dir not exist')

        return [per_sub_dir_relative_path, 0, 0, error_message_list]

    image_file_count, unknown_suffix_file_count = 0, 0
    for per_root_path, _, per_file_name_list in os.walk(per_sub_dir_path):
        for per_file_name in per_file_name_list:
            if check_image_file_suffix(per_file_name):
                image_file_count += 1
            else:
                unknown_suffix_file_count += 1

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_sub_dir_relative_path} unknown suffix file num {unknown_suffix_file_count}'
        )

    return [
        per_sub_dir_relative_path,
        image_file_count,
        unknown_suffix_file_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, archive_result,
                             annotation_result):
    """可选的二次对账: 遍历输出目录核对落盘图像数与位图统计是否一致"""
    root_image_path = os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                                   ARCHIVE_TOP_DIR_NAME)
    root_face_image_path = os.path.join(root_image_path, ARCHIVE_FACE_DIR_NAME)

    sub_dir_check_pair_list = []
    for per_root_image_path, per_image_category in [
        [root_image_path, 'target_image'],
        [root_face_image_path, 'face_image'],
    ]:
        if not os.path.exists(per_root_image_path):
            continue

        for per_sub_dir_name in sorted(os.listdir(per_root_image_path)):
            per_sub_dir_path = os.path.join(per_root_image_path,
                                            per_sub_dir_name)
            if not os.path.isdir(per_sub_dir_path):
                continue

            # face/是target_image目录下的子目录，遍历target_image时要跳过它
            if per_image_category == 'target_image' and per_sub_dir_name == ARCHIVE_FACE_DIR_NAME:
                continue

            sub_dir_check_pair_list.append([
                f'{per_image_category}/{per_sub_dir_name}',
                per_sub_dir_path,
            ])

    error_message_list = []
    on_disk_image_count_dict = collections.Counter()
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_sub_dir_on_disk, sub_dir_check_pair_list),
                                     total=len(sub_dir_check_pair_list)):
            per_sub_dir_relative_path, per_image_file_count, _, per_error_message_list = per_check_result
            on_disk_image_count_dict[per_sub_dir_relative_path.split('/')
                                     [0]] += per_image_file_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk image:', dict(on_disk_image_count_dict))

    for per_image_category, per_expect_image_count in [
        [
            'target_image',
            annotation_result['on_disk_target_image_count'],
        ],
        [
            'face_image',
            annotation_result['on_disk_face_image_count'],
        ],
    ]:
        per_on_disk_image_count = on_disk_image_count_dict.get(
            per_image_category, 0)
        if per_on_disk_image_count != per_expect_image_count:
            error_message_list.append(
                f'{per_image_category} on disk image count not match {per_on_disk_image_count} != {per_expect_image_count}'
            )

    return error_message_list


def save_check_result(save_dataset_path, archive_result, annotation_result,
                      copy_error_message_list, preflight_warning_message_list):
    """汇总解压与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_file_member_count = archive_result['total_file_member_count']
    valid_sample_pair_count = annotation_result['valid_sample_pair_count']

    warning_message_list = list(preflight_warning_message_list)

    if not EXPECTED_SAMPLE_PAIR_COUNT_RANGE[
            0] <= valid_sample_pair_count <= EXPECTED_SAMPLE_PAIR_COUNT_RANGE[
                1]:
        # 官方只声明"约6M"，没给精确条数，所以这里只做软校验、打印告警不判失败
        warning_message_list.append(
            f'valid sample pair count {valid_sample_pair_count} not in {EXPECTED_SAMPLE_PAIR_COUNT_RANGE}'
        )

    if annotation_result['missing_face_embedding_count'] > 0:
        warning_message_list.append(
            f'missing face embedding count {annotation_result["missing_face_embedding_count"]}'
        )

    print('3333', 'total tar file member:', total_file_member_count,
          'total dir member:', archive_result['total_dir_member_count'],
          'skip name member:', archive_result['skip_name_member_count'],
          'extract:', archive_result['extract_file_count'], 'skip:',
          archive_result['skip_file_count'], 'not save:',
          archive_result['not_save_file_count'], 'duplicate member:',
          archive_result['duplicate_member_count'], 'unknown member:',
          archive_result['unknown_member_count'], 'max sample id:',
          archive_result['max_sample_id'])
    print('3333', 'member category:',
          archive_result['member_category_count_dict'])
    print('3333', 'total annotation line:',
          annotation_result['total_annotation_line_count'],
          'total valid sample pair:', valid_sample_pair_count,
          'illegal annotation line:',
          annotation_result['illegal_annotation_line_count'], 'empty caption:',
          annotation_result['empty_caption_count'], 'missing target image:',
          annotation_result['missing_target_image_count'],
          'missing face image:', annotation_result['missing_face_image_count'],
          'missing face embedding:',
          annotation_result['missing_face_embedding_count'], 'orphan image:',
          annotation_result['orphan_image_count'])
    print('3333', 'on disk target image:',
          annotation_result['on_disk_target_image_count'],
          'on disk face image:', annotation_result['on_disk_face_image_count'],
          'on disk face embedding:',
          annotation_result['on_disk_face_embedding_count'])
    print('3333', 'caption key name:',
          annotation_result['caption_key_name_count_dict'])
    print('3333', 'annotation key name:',
          annotation_result['annotation_key_name_count_dict'])
    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'allow_incomplete_sample_pair_flag':
        ALLOW_INCOMPLETE_SAMPLE_PAIR_FLAG,
        'extract_target_image_flag':
        EXTRACT_TARGET_IMAGE_FLAG,
        'extract_face_image_flag':
        EXTRACT_FACE_IMAGE_FLAG,
        'extract_face_embedding_flag':
        EXTRACT_FACE_EMBEDDING_FLAG,
        'archive_group_name':
        archive_result['archive_group_name'],
        'archive_part_num':
        archive_result['archive_part_num'],
        'reach_tar_end':
        archive_result['reach_tar_end'],
        'reach_gzip_end':
        archive_result['reach_gzip_end'],
        'total_tar_file_member_count':
        total_file_member_count,
        'total_tar_dir_member_count':
        archive_result['total_dir_member_count'],
        'total_skip_name_member_count':
        archive_result['skip_name_member_count'],
        'total_extract_file_count':
        archive_result['extract_file_count'],
        'total_skip_file_count':
        archive_result['skip_file_count'],
        'total_not_save_file_count':
        archive_result['not_save_file_count'],
        'total_write_fail_file_count':
        archive_result['write_fail_file_count'],
        'total_duplicate_member_count':
        archive_result['duplicate_member_count'],
        'total_unknown_member_count':
        archive_result['unknown_member_count'],
        'member_category_count_dict':
        archive_result['member_category_count_dict'],
        'max_sample_id':
        archive_result['max_sample_id'],
        'total_annotation_line_count':
        annotation_result['total_annotation_line_count'],
        'total_valid_sample_pair_count':
        valid_sample_pair_count,
        'illegal_annotation_line_count':
        annotation_result['illegal_annotation_line_count'],
        'empty_caption_count':
        annotation_result['empty_caption_count'],
        'missing_target_image_count':
        annotation_result['missing_target_image_count'],
        'missing_face_image_count':
        annotation_result['missing_face_image_count'],
        'missing_face_embedding_count':
        annotation_result['missing_face_embedding_count'],
        'orphan_image_count':
        annotation_result['orphan_image_count'],
        'on_disk_target_image_count':
        annotation_result['on_disk_target_image_count'],
        'on_disk_face_image_count':
        annotation_result['on_disk_face_image_count'],
        'on_disk_face_embedding_count':
        annotation_result['on_disk_face_embedding_count'],
        'caption_key_name_count_dict':
        annotation_result['caption_key_name_count_dict'],
        'annotation_key_name_count_dict':
        annotation_result['annotation_key_name_count_dict'],
        'save_raw_annotation_relative_path_list':
        archive_result['save_raw_annotation_relative_path_list'],
        'save_annotation_relative_path_list':
        annotation_result['save_annotation_relative_path_list'],
        'missing_image_sample_key_list':
        annotation_result['missing_image_sample_key_list'],
        'orphan_image_sample_key_list':
        annotation_result['orphan_image_sample_key_list'],
        'invalid_annotation_message_list':
        annotation_result['invalid_annotation_message_list'],
        'unknown_member_name_list':
        archive_result['unknown_member_name_list'],
        'duplicate_member_name_list':
        archive_result['duplicate_member_name_list'],
        'out_of_capacity_sample_id_list':
        archive_result['out_of_capacity_sample_id_list'],
        'warning_message_list':
        warning_message_list,
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    error_message_list = []
    error_message_list.extend(copy_error_message_list)
    error_message_list.extend(archive_result['error_message_list'])
    error_message_list.extend(annotation_result['error_message_list'])

    if valid_sample_pair_count == 0:
        # 一个有效样本对都没有一定是硬错误，与开关无关
        error_message_list.append('no valid sample pair found')

    # 不完整样本对相关的四项按ALLOW_INCOMPLETE_SAMPLE_PAIR_FLAG决定是硬失败还是只告警,
    # 详见该开关的说明
    incomplete_sample_pair_message_list = []
    if annotation_result['illegal_annotation_line_count'] > 0:
        incomplete_sample_pair_message_list.append(
            f'illegal annotation line count {annotation_result["illegal_annotation_line_count"]}'
        )
    if annotation_result['empty_caption_count'] > 0:
        incomplete_sample_pair_message_list.append(
            f'empty caption count {annotation_result["empty_caption_count"]}')
    if annotation_result['missing_target_image_count'] > 0:
        incomplete_sample_pair_message_list.append(
            f'missing target image count {annotation_result["missing_target_image_count"]}'
        )
    if annotation_result['missing_face_image_count'] > 0:
        incomplete_sample_pair_message_list.append(
            f'missing face image count {annotation_result["missing_face_image_count"]}'
        )
    if annotation_result['orphan_image_count'] > 0:
        incomplete_sample_pair_message_list.append(
            f'orphan image count {annotation_result["orphan_image_count"]}')

    if ALLOW_INCOMPLETE_SAMPLE_PAIR_FLAG:
        warning_message_list.extend(incomplete_sample_pair_message_list)
        for per_warning_message in incomplete_sample_pair_message_list:
            print('2222', per_warning_message)
    else:
        error_message_list.extend(incomplete_sample_pair_message_list)

    # 这两条与开关无关，始终是硬失败: 落盘图像数必须恰好等于有效样本对数。
    # 少了说明有样本对的图没落盘(漏保存)，多了说明有图落了盘却没进汇总标注(漏整理),
    # 两种都属于"我们自己处理出问题"。
    # 注意允许不完整样本对时，多出来的图会先被上面的orphan/missing计数解释掉,
    # 所以这里仍然应该相等; 不相等就说明还有第三种没被解释的差异
    if annotation_result[
            'on_disk_target_image_count'] != valid_sample_pair_count + annotation_result[
                'orphan_image_count']:
        error_message_list.append(
            f'on disk target image count not match {annotation_result["on_disk_target_image_count"]} != {valid_sample_pair_count} + {annotation_result["orphan_image_count"]}'
        )
    if annotation_result['on_disk_face_image_count'] < valid_sample_pair_count:
        error_message_list.append(
            f'on disk face image count less than valid sample pair count {annotation_result["on_disk_face_image_count"]} < {valid_sample_pair_count}'
        )

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    file_copy_pair_list, archive_group_list = get_all_file_and_archive_group(
        root_dataset_path)

    print('1111', len(file_copy_pair_list), len(archive_group_list))
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(archive_group_list) > 0:
        print('1111', archive_group_list[0][1], archive_group_list[0][2],
              len(archive_group_list[0][4]))

    preflight_error_message_list, preflight_warning_message_list = check_required_archive_complete(
        root_dataset_path, archive_group_list)
    for per_warning_message in preflight_warning_message_list:
        print('2222', per_warning_message)
    if len(preflight_error_message_list) > 0:
        # 336片是一个连续gzip流，数据集本身不完整就没必要跑几小时解压
        raise Exception(
            f'check archive failed {preflight_error_message_list[:20]}')

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(save_dataset_path, SAVE_RAW_ANNOTATION_DIR_NAME),
                exist_ok=True)
    os.makedirs(os.path.join(save_dataset_path, SAVE_ANNOTATION_DIR_NAME),
                exist_ok=True)

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

    # 三类成员各一张按样本id下标的位图，记录"该id的这类成员是否已完整落盘"。
    # 解压和标注合并这两步共用它们，所以在这里创建
    member_flag_array_dict = {
        'target_image': SampleIdFlagArray(),
        'face_image': SampleIdFlagArray(),
        'face_embedding': SampleIdFlagArray(),
    }

    archive_result = process_single_archive_group(
        archive_group_list[0],
        save_dataset_path=save_dataset_path,
        member_flag_array_dict=member_flag_array_dict)

    print('2222', archive_result['archive_group_name'], 'extract:',
          archive_result['extract_file_count'], 'skip:',
          archive_result['skip_file_count'], 'not save:',
          archive_result['not_save_file_count'], 'tar file member:',
          archive_result['total_file_member_count'], 'duplicate member:',
          archive_result['duplicate_member_count'], 'unknown member:',
          archive_result['unknown_member_count'])

    annotation_result = merge_annotation_and_check_sample_pair(
        save_dataset_path, archive_result, member_flag_array_dict)

    check_error_message_list = save_check_result(
        save_dataset_path, archive_result, annotation_result,
        copy_error_message_list, preflight_warning_message_list)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, archive_result, annotation_result)

    all_error_message_list = check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解压/标注合并/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/FaceID-6M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
