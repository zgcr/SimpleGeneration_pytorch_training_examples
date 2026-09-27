import io
import os
import re
import json
import struct
import shutil
import tarfile
import collections

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: BM-6M(ByteMorph-6M, ByteMorph/BM-6M)
#
# 【数据集类型】纯图像编辑(instruction-based image editing)数据集，不是文生图数据集。
# 该数据集专攻**非刚性运动编辑**(镜头移动/物体形变/人体articulation/复杂交互):
# 原始视频由Seaweed生成，再抽帧成"编辑前帧 -> 编辑后帧"的图像编辑对，最后由VLM过滤并打标。
# 每个样本对固定是"1张参考图(编辑前帧) + 1条英文编辑指令 + 1张编辑后图(编辑后帧)"，
# 没有第二张视觉条件图、没有mask。下游只能走ti2i_dataset.py那条链路，
# 不能当t2i(文生图)数据用(没有任何"只有caption+一张图"的纯生成样本)。
#
# 【root_dataset_path实测原始保存规格(共2.6T / 2120个tar)】
# BM-6M/
# ├── subset-1 .. subset-9      9个子集，每个子集下固定只有下面这2个kind目录
# │   ├── sampled_frames/       从视频里抽的**首尾两帧**构成的编辑对
# │   │   └── batch_{i}.tar     每个tar约1440个样本对(约650MB)
# │   └── sampled_multi_frames/ 从视频里抽的**多帧相邻两两**构成的编辑对
# │       └── batch_{i}.tar     每个tar约4320个样本对(=1440*3，约2GB)
# ├── README.md                 数据集说明(无用)
# ├── LICENSE.txt               CC0-1.0协议正文(无用，授权信息记进校验报告字段即可)
# ├── .gitattributes            git lfs配置(无用)
# └── .cache/                   huggingface下载缓存，4249个文件，无残留*.incomplete(无用)
#
# 每个kind目录下的tar数(实测，见EXPECTED_SUBSET_KIND_ARCHIVE_NUM_DICT):
#   sampled_frames      : 88 / 88 / 84 / 155 / 117 / 136 / 187 / 75 / 196 = 1126
#   sampled_multi_frames: 88 / 88 / 84 /  23 / 117 / 136 / 187 / 75 / 196 =  994
#   合计2120个tar。
#
# **注意 subset-4/sampled_multi_frames 只有23个tar且编号不连续**
# (0,2,29,40,50,52,56,63,76,80,81,85,86,95,110,111,117,120,129,135,140,153,154)，
# 这是**上游仓库自身的规格，不是本地下载缺失**: 拿huggingface下载缓存里的仓库文件清单
# .cache/huggingface/trees/*.json 对账过，清单里这一组也只有这23个文件，
# 而且**2120个tar全部存在、字节数与清单100%一致、无任何*.incomplete**。
# 所以"编号必须0..N-1连号"这条硬校验只能对其余17组用，
# 这一组必须改成"与写死的稀疏编号集合完全相等"，否则会把正常数据判成缺失。
#
# 【tar内部实测规格】
# - 每个tar有且只有**一层顶层目录，名字与tar同名**(batch_0.tar -> batch_0/);
# - 成员严格"json在前、png在后"成对交替排列，目录深度固定为1，无链接等非普通文件:
#     batch_<i>/<sample_key>.json
#     batch_<i>/<sample_key>.png
# - 成员数恒为偶数，每个prefix恰好出现2次(1个json + 1个png)，无重名、无孤立成员;
# - 2120个tar**全部**满足"长度512字节对齐 + 尾部1024字节全0的EOF块"(已O(1)全量扫过)，
#   说明没有任何一个tar被下载截断。
#
# 【**关键规格: png是左右拼接的"参考图|编辑后图"**】
# 抽样7个tar共2.2万张png，**恒为 1024x512 / 8bit / colortype 2(RGB)**，
# 也就是 左半512x512 = 参考图(编辑前帧)，右半512x512 = 编辑后图(编辑后帧)。
# 已用像素级交叉验证(mean abs diff 全为0.0)确认帧序:
#   sampled_frames/<k>_0.png            = (帧0 | 帧3)
#   sampled_multi_frames/<k>_frame_0_1  = (帧0 | 帧1)
#   sampled_multi_frames/<k>_frame_1_2  = (帧1 | 帧2)
#   sampled_multi_frames/<k>_frame_2_3  = (帧2 | 帧3)
# 即 frames侧的左半 == multi侧frame_0_1的左半，frames侧的右半 == multi侧frame_2_3的右半，
# 且相邻两个frame_*的右半与左半严格相接。
# 两个kind的sample_key集合在同一个batch内**交集为0**(1440 vs 4320)，
# 是两套不同粒度(大位移 vs 小位移)的编辑对，**都是有效样本对，都要保留，不能只取一个kind**。
#
# 【单条json的全部5个字段(抽样2.2万条，keyset 100%一致，0条空值/0条非字符串)】
#   edit_rewrite : **祈使句编辑指令**("Remove the human from the frame and shift the
#                  camera slightly to the left ...")                -> 有用(训练主文本，必需)
#   edit         : 陈述句描述src->tgt发生了什么变化                  -> 有用(备用指令/改写增强)
#   input        : 参考图(编辑前帧)的详细caption                     -> 有用(编辑前图描述)
#   output       : 编辑后图(编辑后帧)的详细caption                   -> 有用(编辑后图描述，
#                                                                      可做t2i辅助监督)
#   caption      : **视频级**运动caption，同一个视频抽出来的所有帧对**共享同一条**
#                  (实测frames的<k>_0与multi的<k>_frame_0_1/1_2/2_3四条完全相同)
#                                                                    -> 有用但粒度粗，
#                                                                      必须标成video级别，
#                                                                      不能当成本帧对的指令
#   同名<key>.png: 1024x512拼接图 -> 拆出参考图 + 编辑后图            -> 有用(必需)
# sample_key本身还编码了溯源信息(实测100%可解析):
#   sampled_frames      : <video_id>_<clip>_<sub_clip>_<global>_<extra>_seed<seed>_<pair>
#   sampled_multi_frames: <video_id>_<clip>_<sub_clip>_<global>_<extra>_seed<seed>_frame_<f0>_<f1>
#   -> video_id/seed/帧号都有用(可做同视频去重、按视频划分train/val、避免同视频泄漏)
#
# 【无用信息(一律不整理进训练目录)】
# .cache/(4249个文件) / .gitattributes / README.md / LICENSE.txt /
# .DS_Store / CACHEDIR.TAG 这类目录元数据垃圾文件。
#
# 【本脚本的处理口径(方案B: unzip阶段就把拼接图拆成两张单图)】
# - 解压前预检(硬失败): 根目录条目白名单(只允许9个subset-N)、每个subset下只允许2个kind目录、
#   tar名必须是batch_<idx>.tar、18组tar数与写死的实测ground truth逐一比对、
#   17组校验0..N-1连号 + subset-4/sampled_multi_frames与写死的23个稀疏编号集合精确相等、
#   2120个tar并行做"512字节对齐 + 尾部1024字节EOF块"O(1)尾检;
# - 并行单位 = 单个tar(2120个任务，Pool(32))，mode='r|*'流式解压，
#   **顺手把json内容读进内存生成汇总标注**(单条json只有约1KB，流式解压时正好在手上，
#   比解压完再去扫590万个小json便宜几个数量级);
# - png成员: 先从PNG的IHDR块头O(1)解析宽高(不解码像素)，硬校验 width == 2 * height，
#   再解码一次并crop成左右两张512x512，**无损存成PNG**分别落盘:
#     unzip_images/<subset>/<kind>/<batch_i>/<sample_key>_reference.png
#     unzip_images/<subset>/<kind>/<batch_i>/<sample_key>_edited.png
#   unzip阶段绝不引入二次有损压缩，resize/转jpg/分辨率分桶留给preprocessing2的resave脚本;
# - 每张图写盘后**立刻反读落盘文件的IHDR校验宽高恰好是512x512且文件非空**
#   (比只看"文件存在"强，能挡住写半截/写0字节)，**两张都成功才算这一对extract成功**;
# - 断点重跑: 两张图都已存在、都非空、IHDR都是512x512 -> 整对计skip并跳过解码;
#   只有一张在就整对重写，绝不留半状态;
# - 每个tar内严格对账(缺一条立刻报错): json成员数 == png成员数 == 有效样本对数、
#   成员总数为偶数且 有效样本对数*2 == 成员总数、
#   extract对数+skip对数+not_save对数+invalid对数 == png成员数、
#   落盘图像文件数 == extract对数*2、顶层目录名 == tar名、必须读到tar流末尾;
# - 绝不静默丢样本对: json解析失败/5个字段缺失/edit_rewrite与edit全空/png缺json/json缺png/
#   宽高不满足w==2h/解码或写盘失败/同tar内重名成员/sample_key basename重复/
#   sample_key不匹配命名规格，全部分门别类记进对应隔离清单并在汇总报告里上报;
#   重名成员改写到 unzip_duplicate_members/ 独立目录保留数据，不互相覆盖;
#   **没有编辑指令的样本对，图仍然照常拆图落盘**(有用信息不丢)，只是不进有效标注，
#   同时记入no_instruction_sample_key_list，等上游补齐指令后即可直接用;
# - 汇总标注落 unzip_annotations/<subset>/<kind>/<batch_i>.jsonl，每行一个完整样本对，
#   保留原json全部5个字段 + 两张图落盘路径 + 宽高 + 视频/seed/帧号等全部有用属性;
# - 拷贝/解压/校验任一环出错都汇总后抛异常，不再静默跑过。
#
# 【方案B的代价(跑之前务必确认目标盘扛得住)】
# - 输出小文件数约 **1180万张png**(约590万对 * 2)，比"原样保留拼接图"翻倍，NAS inode压力大;
# - 需要解码 + 重编码约590万张PNG，是CPU主要开销(Pool(32)下预计数十小时);
# - PNG无损，拆出来的两张图合计字节数与原拼接图同量级(约2.5T上下)，目标盘空间充足。
# ==============================================================================

DATASET_TASK_TYPE = 'image_edit'

DATASET_LICENSE_NAME = 'cc0-1.0'

ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.tar$'),
]

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(4249个文件，含仓库文件清单与*.lock/*.metadata)
# .gitattributes   git lfs配置
# README.md        数据集说明
# LICENSE.txt      CC0-1.0协议正文，授权信息已记进校验报告的dataset_license_name字段
# .DS_Store/CACHEDIR.TAG  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    'LICENSE.txt',
    '.DS_Store',
    'CACHEDIR.TAG',
]

ANNOTATION_FILE_SUFFIX = '.json'

IMAGE_FILE_SUFFIX_LIST = [
    '.png',
    '.jpg',
    '.jpeg',
    '.webp',
    '.bmp',
]

# 过滤掉无用信息后根目录只应该剩这9个子集目录
SUBSET_ROOT_DIR_NAME_LIST = [
    'subset-1',
    'subset-2',
    'subset-3',
    'subset-4',
    'subset-5',
    'subset-6',
    'subset-7',
    'subset-8',
    'subset-9',
]

SAMPLED_FRAMES_DIR_NAME = 'sampled_frames'

SAMPLED_MULTI_FRAMES_DIR_NAME = 'sampled_multi_frames'

# 每个子集目录下有且只有这2个kind目录
FRAME_KIND_DIR_NAME_LIST = [
    SAMPLED_FRAMES_DIR_NAME,
    SAMPLED_MULTI_FRAMES_DIR_NAME,
]

ARCHIVE_SHARD_NAME_PATTERN = re.compile(r'^batch_(?P<index>\d+)$')

# 实测每个<subset>/<kind>下的tar数(与huggingface仓库文件清单100%一致)，数量不对说明下载不全
EXPECTED_SUBSET_KIND_ARCHIVE_NUM_DICT = {
    'subset-1/sampled_frames': 88,
    'subset-1/sampled_multi_frames': 88,
    'subset-2/sampled_frames': 88,
    'subset-2/sampled_multi_frames': 88,
    'subset-3/sampled_frames': 84,
    'subset-3/sampled_multi_frames': 84,
    'subset-4/sampled_frames': 155,
    'subset-4/sampled_multi_frames': 23,
    'subset-5/sampled_frames': 117,
    'subset-5/sampled_multi_frames': 117,
    'subset-6/sampled_frames': 136,
    'subset-6/sampled_multi_frames': 136,
    'subset-7/sampled_frames': 187,
    'subset-7/sampled_multi_frames': 187,
    'subset-8/sampled_frames': 75,
    'subset-8/sampled_multi_frames': 75,
    'subset-9/sampled_frames': 196,
    'subset-9/sampled_multi_frames': 196,
}

# subset-4/sampled_multi_frames在上游仓库里本来就只有这23个稀疏编号(不是本地缺失，
# 已和.cache里的仓库文件清单对账过)，所以这一组不能按0..N-1连号校验，
# 必须与这个写死的集合精确相等: 少了说明下载不全，多了说明上游改了规格，两种都要显式感知
EXPECTED_SPARSE_ARCHIVE_INDEX_DICT = {
    'subset-4/sampled_multi_frames': [
        0,
        2,
        29,
        40,
        50,
        52,
        56,
        63,
        76,
        80,
        81,
        85,
        86,
        95,
        110,
        111,
        117,
        120,
        129,
        135,
        140,
        153,
        154,
    ],
}

# sample_key的命名规格(实测100%可解析)，解析出的video_id/seed/帧号都是有用属性
SAMPLED_FRAMES_SAMPLE_KEY_PATTERN = re.compile(
    r'^(?P<video_id>.+)_(?P<clip_index>\d+)_(?P<sub_clip_index>\d+)_(?P<global_index>\d+)_(?P<extra_index>\d+)_seed(?P<seed_index>\d+)_(?P<frame_pair_index>\d+)$'
)

SAMPLED_MULTI_FRAMES_SAMPLE_KEY_PATTERN = re.compile(
    r'^(?P<video_id>.+)_(?P<clip_index>\d+)_(?P<sub_clip_index>\d+)_(?P<global_index>\d+)_(?P<extra_index>\d+)_seed(?P<seed_index>\d+)_frame_(?P<start_frame_index>\d+)_(?P<end_frame_index>\d+)$'
)

SAMPLE_KEY_PATTERN_DICT = {
    SAMPLED_FRAMES_DIR_NAME: SAMPLED_FRAMES_SAMPLE_KEY_PATTERN,
    SAMPLED_MULTI_FRAMES_DIR_NAME: SAMPLED_MULTI_FRAMES_SAMPLE_KEY_PATTERN,
}

# 训练主文本 = edit_rewrite(祈使句编辑指令)，为空时回退取edit(陈述句变化描述),
# 两个都为空则该样本对没有文本条件、不可训练，隔离上报
ANNOTATION_TEXT_KEY_NAME_LIST = [
    'edit_rewrite',
    'edit',
]

ANNOTATION_EDIT_INSTRUCTION_KEY_NAME = 'edit_rewrite'

ANNOTATION_EDIT_DESCRIPTION_KEY_NAME = 'edit'

ANNOTATION_REFERENCE_IMAGE_CAPTION_KEY_NAME = 'input'

ANNOTATION_EDITED_IMAGE_CAPTION_KEY_NAME = 'output'

ANNOTATION_VIDEO_CAPTION_KEY_NAME = 'caption'

# 每条json里必须齐备的5个有用属性，缺失只上报不丢样本
# (实测2.2万条抽样全部齐备，一旦出现缺失说明数据规格变了，必须显式感知)
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    ANNOTATION_EDIT_INSTRUCTION_KEY_NAME,
    ANNOTATION_EDIT_DESCRIPTION_KEY_NAME,
    ANNOTATION_REFERENCE_IMAGE_CAPTION_KEY_NAME,
    ANNOTATION_EDITED_IMAGE_CAPTION_KEY_NAME,
    ANNOTATION_VIDEO_CAPTION_KEY_NAME,
]

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_DUPLICATE_MEMBER_DIR_NAME = 'unzip_duplicate_members'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.png'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.png'

# unzip阶段只做无损拆图，绝不引入二次有损压缩;
# compress_level=1是"仍然无损、但编码最快"的档位(PNG的compress_level只影响体积和耗时)
SAVE_IMAGE_FILE_FORMAT = 'PNG'

SAVE_IMAGE_PNG_COMPRESS_LEVEL = 1

SAVE_IMAGE_MODE = 'RGB'

# 拼接图必须是"左右两张等宽等高的方图"，即 width == 2 * height
CONCAT_IMAGE_WIDTH_HEIGHT_RATIO = 2

CONCAT_IMAGE_LAYOUT_NAME = 'horizontal_reference_left_edited_right'

# 实测拼接图恒为1024x512(拆出来是两张512x512)，只做软校验(打印告警),
# 因为上游随时可能出更高分辨率的版本，尺寸变了不该判失败，但必须显式感知
EXPECTED_CONCAT_IMAGE_WIDTH = 1024

EXPECTED_CONCAT_IMAGE_HEIGHT = 512

# 实测png全部是colortype 2(truecolor RGB)，出现其它颜色类型说明规格变了，
# 图仍然按convert('RGB')正常拆图落盘，只上报不丢样本
EXPECTED_PNG_COLOR_TYPE_LIST = [
    2,
]

# 按实测每个tar约1440(frames) / 4320(multi)个样本对换算成宽松区间(末尾tar是不满的),
# 只做软校验(打印告警): 官方README只给了"BM-6M"这个量级名和一个780308行的demo子集规格，
# 没有给全量精确条数，不能拿来当硬性失败条件
EXPECTED_FRAME_KIND_SAMPLE_PAIR_COUNT_RANGE_DICT = {
    SAMPLED_FRAMES_DIR_NAME: [1400000, 1630000],
    SAMPLED_MULTI_FRAMES_DIR_NAME: [3700000, 4300000],
}

# 图像成员是否拆图落盘。
# True : 默认口径，约590万对 -> 约1180万张512x512的png，NAS上inode与元数据压力极大，
#        且要解码+重编码590万张png，务必确认目标盘和机器扛得住再跑;
# False: 只解析json生成 unzip_annotations/*.jsonl 索引(几小时即可跑完)，
#        图像继续留在原tar里，标注里的落盘路径先占位(image_file_save_flag=False)，
#        之后再单独跑一遍拆图。
EXTRACT_IMAGE_FILE_FLAG = True

# json成员是否额外原样落盘。
# 默认False: json里的5个字段已被**完整**写进jsonl汇总标注，
# 再落约590万个1KB小文件纯属浪费NAS inode(小文件数会从1180万涨到1770万)。
SAVE_ANNOTATION_MEMBER_FILE_FLAG = False

# 是否在解压后再os.walk一遍输出目录做二次对账。
# 默认False: 1180万个小文件的os.walk在NAS上要跑非常久，而解压时已经做了
# "每张图写盘后立刻反读IHDR校验宽高" + "extract+skip+not_save+invalid == png成员数" +
# "落盘图像文件数 == extract对数*2"三道对账，已经能保证每个成员都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

TAR_BLOCK_SIZE = 512

TAR_EOF_BLOCK_SIZE = 1024

PNG_FILE_MAGIC_BYTES = b'\x89PNG\r\n\x1a\n'

# PNG固定是"8字节magic + 4字节长度 + 4字节'IHDR' + 13字节IHDR数据"，
# 前33字节就能拿到宽高/位深/颜色类型，完全不用解码像素
PNG_IHDR_HEADER_SIZE = 33

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024

MAX_SAVE_MESSAGE_NUM = 10000


class MultiPartArchiveReader:
    """把按字节切分的多个分片压缩包拼接成一个只读的连续字节流

    该数据集每个tar都是独立完整的(不是分卷)，这里保留多分片能力只是为了和其他脚本口径一致。
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
    """过滤掉.cache、.gitattributes、README.md、LICENSE.txt等无用文件或目录"""
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
    不能直接按字符串排序: part-9 会排到 part-10 后面；
    也不能只按[位数, 字符串]排序: part-100 会排到 part-89 前面导致整个tar流错位。
    """
    per_archive_part_index = per_archive_part_index or ''
    if per_archive_part_index.isdigit():
        return [0, int(per_archive_part_index), '']

    return [1, len(per_archive_part_index), per_archive_part_index]


def get_png_image_header_info(per_image_head_bytes):
    """O(1)从PNG的IHDR块头里取[宽, 高, 位深, 颜色类型]，完全不解码像素

    解析失败(不是PNG/头部被截断/第一个块不是IHDR)统一返回None。
    """
    if per_image_head_bytes is None or len(
            per_image_head_bytes) < PNG_IHDR_HEADER_SIZE:
        return None

    if per_image_head_bytes[:len(PNG_FILE_MAGIC_BYTES
                                 )] != PNG_FILE_MAGIC_BYTES:
        return None

    if per_image_head_bytes[12:16] != b'IHDR':
        return None

    try:
        per_image_width, per_image_height = struct.unpack(
            '>II', per_image_head_bytes[16:24])
    except Exception:
        return None

    per_image_bit_depth = per_image_head_bytes[24]
    per_image_color_type = per_image_head_bytes[25]

    if per_image_width <= 0 or per_image_height <= 0:
        return None

    return [
        per_image_width,
        per_image_height,
        per_image_bit_depth,
        per_image_color_type,
    ]


def get_save_image_file_header_info(save_image_path):
    """反读落盘图像文件的PNG头，用于写盘后立刻校验宽高(只读33字节)"""
    try:
        with open(save_image_path, 'rb') as load_image_file:
            per_image_head_bytes = load_image_file.read(PNG_IHDR_HEADER_SIZE)
    except Exception:
        return None

    return get_png_image_header_info(per_image_head_bytes)


def check_single_save_image_file(save_image_path, per_expect_image_width,
                                 per_expect_image_height):
    """校验单张落盘图像: 文件存在且非空，且PNG头里的宽高与期望值一致

    只看"文件存在"挡不住写半截或写0字节，必须反读文件头拿真实宽高做ground truth。
    """
    if not os.path.exists(save_image_path):
        return f'save image not exist {save_image_path}'

    try:
        if os.path.getsize(save_image_path) <= 0:
            return f'save image empty {save_image_path}'
    except Exception as e:
        return f'get save image size failed {save_image_path} {e}'

    per_image_header_info = get_save_image_file_header_info(save_image_path)
    if per_image_header_info is None:
        return f'save image png header broken {save_image_path}'

    per_image_width, per_image_height = per_image_header_info[
        0], per_image_header_info[1]
    if per_image_width != per_expect_image_width or per_image_height != per_expect_image_height:
        return f'save image size not match {save_image_path} {per_image_width}x{per_image_height} != {per_expect_image_width}x{per_expect_image_height}'

    return ''


def check_single_archive_tar_tail(per_archive_path):
    """O(1)预检单个tar是否被截断: 长度必须512字节对齐，且结尾必须有1024字节全0的EOF块

    实测2120个tar全部满足，说明当前数据集是完整的。
    如果下载不全，流式解压只会在读到一半时抛异常，必须在跑2.6T解压前先拦住。
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


def check_single_subset_kind_archive_complete(root_dataset_path,
                                              per_subset_name,
                                              per_frame_kind_name):
    """预检单个<subset>/<kind>目录: tar名规格、tar数量、tar编号集合

    subset-4/sampled_multi_frames在上游仓库里本来就是稀疏编号(实测23个)，
    走EXPECTED_SPARSE_ARCHIVE_INDEX_DICT做精确集合比对；
    其余17组必须是0..N-1连号。
    """
    per_subset_kind_key = f'{per_subset_name}/{per_frame_kind_name}'
    per_subset_kind_path = os.path.join(root_dataset_path, per_subset_name,
                                        per_frame_kind_name)

    error_message_list, archive_path_list = [], []
    if not os.path.exists(per_subset_kind_path):
        error_message_list.append(
            f'subset kind dir not exist {per_subset_kind_key}')

        return error_message_list, archive_path_list

    per_archive_index_list = []
    for per_file_name in sorted(os.listdir(per_subset_kind_path)):
        if check_skip_file_or_dir(per_file_name):
            continue

        per_file_path = os.path.join(per_subset_kind_path, per_file_name)
        if os.path.isdir(per_file_path):
            error_message_list.append(
                f'unknown dir in subset kind dir {per_subset_kind_key}/{per_file_name}'
            )
            continue

        per_file_name_prefix, per_file_name_suffix = os.path.splitext(
            per_file_name)
        if per_file_name_suffix.lower() != '.tar':
            error_message_list.append(
                f'unknown file in subset kind dir {per_subset_kind_key}/{per_file_name}'
            )
            continue

        archive_path_list.append(per_file_path)

        per_match_result = ARCHIVE_SHARD_NAME_PATTERN.match(
            per_file_name_prefix)
        if not per_match_result:
            error_message_list.append(
                f'unknown archive name {per_subset_kind_key}/{per_file_name}')
            continue

        per_archive_index_list.append(int(per_match_result.group('index')))

    per_expected_archive_num = EXPECTED_SUBSET_KIND_ARCHIVE_NUM_DICT[
        per_subset_kind_key]
    print('1111', per_subset_kind_key, 'archive:', len(archive_path_list),
          'expected archive:', per_expected_archive_num)

    if len(archive_path_list) != per_expected_archive_num:
        error_message_list.append(
            f'{per_subset_kind_key} archive num not match {len(archive_path_list)} != {per_expected_archive_num}'
        )

    if per_subset_kind_key in EXPECTED_SPARSE_ARCHIVE_INDEX_DICT:
        # 上游本来就是稀疏编号，只能做精确集合比对: 少了是下载不全，多了是上游改了规格
        per_expected_archive_index_dict = set(
            EXPECTED_SPARSE_ARCHIVE_INDEX_DICT[per_subset_kind_key])
        per_missing_index_list = sorted(per_expected_archive_index_dict -
                                        set(per_archive_index_list))
        per_unexpected_index_list = sorted(
            set(per_archive_index_list) - per_expected_archive_index_dict)
        if len(per_missing_index_list) > 0:
            error_message_list.append(
                f'{per_subset_kind_key} sparse archive index missing {per_missing_index_list[:10]}'
            )
        if len(per_unexpected_index_list) > 0:
            error_message_list.append(
                f'{per_subset_kind_key} sparse archive index unexpected {per_unexpected_index_list[:10]}'
            )
    else:
        # 其余17组必须是0..N-1连号，缺号说明有tar没下载下来
        per_missing_index_list = sorted(
            set(range(0, per_expected_archive_num)) -
            set(per_archive_index_list))
        if len(per_missing_index_list) > 0:
            error_message_list.append(
                f'{per_subset_kind_key} archive index not continuous, missing index {per_missing_index_list[:10]}'
            )

    per_duplicate_index_list = [
        per_archive_index for per_archive_index, per_archive_index_count in
        collections.Counter(per_archive_index_list).items()
        if per_archive_index_count > 1
    ]
    if len(per_duplicate_index_list) > 0:
        error_message_list.append(
            f'{per_subset_kind_key} archive index duplicate {sorted(per_duplicate_index_list)[:10]}'
        )

    return error_message_list, archive_path_list


def check_required_subset_complete(root_dataset_path):
    """解压前预检: 根目录条目白名单、子集与kind目录、tar数量与编号、每个tar的EOF完整性

    数据集本身不完整就没必要跑几十小时解压，也避免"少了几个tar但整体报成功"。
    """
    error_message_list = []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')

        return error_message_list

    all_subset_name_list = []
    for per_name in sorted(os.listdir(root_dataset_path)):
        if check_skip_file_or_dir(per_name):
            continue

        if os.path.isdir(os.path.join(root_dataset_path, per_name)):
            all_subset_name_list.append(per_name)
            continue

        # 根目录下出现新的非跳过文件必须显式上报，否则会被静默漏处理
        error_message_list.append(f'unknown file in root dir {per_name}')

    for per_subset_name in all_subset_name_list:
        if per_subset_name not in SUBSET_ROOT_DIR_NAME_LIST:
            error_message_list.append(f'unknown subset dir {per_subset_name}')

    archive_path_check_list = []
    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        per_subset_path = os.path.join(root_dataset_path, per_subset_name)
        if not os.path.exists(per_subset_path):
            error_message_list.append(
                f'subset dir not exist {per_subset_name}')
            continue

        for per_name in sorted(os.listdir(per_subset_path)):
            if check_skip_file_or_dir(per_name):
                continue

            # 子集目录下有且只有2个kind目录，多出任何条目都必须显式上报
            if per_name not in FRAME_KIND_DIR_NAME_LIST:
                error_message_list.append(
                    f'unknown name in subset dir {per_subset_name}/{per_name}')

        for per_frame_kind_name in FRAME_KIND_DIR_NAME_LIST:
            per_error_message_list, per_archive_path_list = check_single_subset_kind_archive_complete(
                root_dataset_path, per_subset_name, per_frame_kind_name)
            error_message_list.extend(per_error_message_list)
            archive_path_check_list.extend(per_archive_path_list)

    per_expected_archive_num = sum(
        EXPECTED_SUBSET_KIND_ARCHIVE_NUM_DICT.values())
    print('1111', 'check archive tar tail:', len(archive_path_check_list),
          'expected archive:', per_expected_archive_num)

    if len(archive_path_check_list) != per_expected_archive_num:
        error_message_list.append(
            f'total archive num not match {len(archive_path_check_list)} != {per_expected_archive_num}'
        )

    with Pool(processes=PROCESS_NUM) as pool:
        for per_tail_error_message_list in tqdm(
                pool.imap_unordered(check_single_archive_tar_tail,
                                    archive_path_check_list),
                total=len(archive_path_check_list)):
            error_message_list.extend(per_tail_error_message_list)

    return error_message_list


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


def save_single_sample_pair_image(per_image_bytes, save_reference_image_path,
                                  save_edited_image_path,
                                  per_concat_image_width,
                                  per_concat_image_height):
    """把1024x512的拼接图拆成左右两张512x512单图无损落盘，并立刻反读校验宽高

    左半 = 参考图(编辑前帧)，右半 = 编辑后图(编辑后帧)，帧序已用像素级交叉验证确认。
    两张都写成功且宽高都对才算成功，任何一张失败就返回错误信息(空串表示成功)。
    """
    per_single_image_width = per_concat_image_width // CONCAT_IMAGE_WIDTH_HEIGHT_RATIO
    per_single_image_height = per_concat_image_height

    try:
        per_load_image = Image.open(io.BytesIO(per_image_bytes))
        per_load_image = per_load_image.convert(SAVE_IMAGE_MODE)
    except Exception as e:
        return f'load concat image failed {save_reference_image_path} {e}'

    if per_load_image.size != (per_concat_image_width,
                               per_concat_image_height):
        # 解码出来的尺寸必须和PNG头里的一致，不一致说明图像数据本身有问题
        return f'concat image decode size not match {save_reference_image_path} {per_load_image.size} != {(per_concat_image_width, per_concat_image_height)}'

    save_image_pair_list = [
        [
            save_reference_image_path,
            [0, 0, per_single_image_width, per_single_image_height],
        ],
        [
            save_edited_image_path,
            [
                per_single_image_width,
                0,
                per_concat_image_width,
                per_single_image_height,
            ],
        ],
    ]

    for per_save_image_path, per_crop_box in save_image_pair_list:
        try:
            os.makedirs(os.path.dirname(per_save_image_path), exist_ok=True)
            per_save_image = per_load_image.crop(tuple(per_crop_box))
            per_save_image.save(per_save_image_path,
                                format=SAVE_IMAGE_FILE_FORMAT,
                                compress_level=SAVE_IMAGE_PNG_COMPRESS_LEVEL)
        except Exception as e:
            return f'write image failed {per_save_image_path} {e}'

        per_check_error_message = check_single_save_image_file(
            per_save_image_path, per_single_image_width,
            per_single_image_height)
        if per_check_error_message:
            return per_check_error_message

    return ''


def check_single_sample_pair_image_on_disk(save_reference_image_path,
                                           save_edited_image_path,
                                           per_concat_image_width,
                                           per_concat_image_height):
    """断点重跑判断: 两张图都已存在且非空且宽高都对才算这一对已经处理完

    只有一张在就整对重写，绝不留"参考图是新的、编辑后图是上次跑残的"这种半状态。
    """
    per_single_image_width = per_concat_image_width // CONCAT_IMAGE_WIDTH_HEIGHT_RATIO
    per_single_image_height = per_concat_image_height

    for per_save_image_path in [
            save_reference_image_path,
            save_edited_image_path,
    ]:
        if check_single_save_image_file(per_save_image_path,
                                        per_single_image_width,
                                        per_single_image_height):
            return False

    return True


def get_single_member_save_relative_path(per_member_name,
                                         per_archive_group_name):
    """算出单个成员在<subset>/<kind>下的落盘相对路径

    实测每个tar都有且只有一层与tar同名的顶层目录(batch_0.tar -> batch_0/)，
    所以直接用成员原相对路径即可，天然带上batch层、不同tar之间不会撞名。
    万一上游哪天去掉了顶层目录(成员在tar根)，不同tar的成员就会互相覆盖，
    这里显式补一层tar名兜底。
    """
    per_member_name = per_member_name.replace('\\', '/').lstrip('/')
    if '/' not in per_member_name:
        return f'{per_archive_group_name}/{per_member_name}'

    return per_member_name


def get_single_sample_key_parse_result(per_frame_kind_name, per_sample_key):
    """解析sample_key里编码的溯源信息(video_id/seed/帧号)，解析失败返回None

    这些属性用于同视频去重、按视频划分train/val(避免同一个视频的帧对既在训练集又在验证集)。
    """
    per_sample_key_pattern = SAMPLE_KEY_PATTERN_DICT.get(
        per_frame_kind_name, None)
    if per_sample_key_pattern is None:
        return None

    per_match_result = per_sample_key_pattern.match(per_sample_key)
    if not per_match_result:
        return None

    per_match_group_dict = per_match_result.groupdict()

    per_start_frame_index, per_end_frame_index = None, None
    if 'start_frame_index' in per_match_group_dict:
        per_start_frame_index = int(per_match_group_dict['start_frame_index'])
        per_end_frame_index = int(per_match_group_dict['end_frame_index'])

    per_frame_pair_index = None
    if 'frame_pair_index' in per_match_group_dict:
        per_frame_pair_index = int(per_match_group_dict['frame_pair_index'])

    return {
        'video_id': per_match_group_dict['video_id'],
        'clip_index': int(per_match_group_dict['clip_index']),
        'sub_clip_index': int(per_match_group_dict['sub_clip_index']),
        'global_index': int(per_match_group_dict['global_index']),
        'extra_index': int(per_match_group_dict['extra_index']),
        'seed_index': int(per_match_group_dict['seed_index']),
        'seed_name': f'seed{per_match_group_dict["seed_index"]}',
        'start_frame_index': per_start_frame_index,
        'end_frame_index': per_end_frame_index,
        'frame_pair_index': per_frame_pair_index,
    }


def get_single_annotation_error_message_list(per_annotation, per_sample_key):
    """校验单条json的有用信息是否完整: 编辑指令必须非空，其余有用属性缺失只上报

    编辑指令优先取edit_rewrite(祈使句)，为空时回退取edit(陈述句变化描述),
    额外返回实际取到的字段名，便于下游区分"原生祈使句指令"和"回退的陈述句描述"。
    """
    error_message_list = []

    per_edit_instruction, per_edit_instruction_key_name = '', ''
    for per_text_key_name in ANNOTATION_TEXT_KEY_NAME_LIST:
        per_text_value = per_annotation.get(per_text_key_name, '')
        if isinstance(per_text_value, str) and len(per_text_value.strip()) > 0:
            per_edit_instruction = per_text_value
            per_edit_instruction_key_name = per_text_key_name
            break

    if not per_edit_instruction:
        # 图像编辑样本对必须有编辑指令，没有文本条件的编辑对不可训练
        error_message_list.append(f'{per_sample_key} empty edit instruction')

    for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            error_message_list.append(
                f'{per_sample_key} miss key {per_key_name}')
            continue

        per_key_value = per_annotation[per_key_name]
        if not isinstance(per_key_value, str) or len(
                per_key_value.strip()) == 0:
            error_message_list.append(
                f'{per_sample_key} empty key {per_key_name}')

    return per_edit_instruction, per_edit_instruction_key_name, error_message_list


def get_single_frame_index_pair_name(per_frame_kind_name,
                                     per_sample_key_parse_result):
    """把帧号解析结果拼成一个可统计的帧对名

    sampled_multi_frames有start/end帧号(0_1 / 1_2 / 2_3),
    sampled_frames只有一个帧对序号(实测恒为0)，两者不能混用同一套key。
    """
    if per_frame_kind_name == SAMPLED_MULTI_FRAMES_DIR_NAME:
        return f'frame_{per_sample_key_parse_result["start_frame_index"]}_{per_sample_key_parse_result["end_frame_index"]}'

    return f'pair_{per_sample_key_parse_result["frame_pair_index"]}'


def get_single_annotation_text_value(per_annotation, per_key_name):
    """取json里的单个文本字段，非字符串或空值统一返回空串"""
    per_key_value = per_annotation.get(per_key_name, '')
    if not isinstance(per_key_value, str):
        return ''

    return per_key_value


def process_single_archive_group(archive_group, save_dataset_path):
    """流式解压单个tar: 把拼接图拆成两张单图落盘，同时把json解析成该tar的jsonl汇总标注

    json只有约1KB，且流式解压时内容正好在手上，顺手解析出来生成汇总标注，
    比解压完再去扫590万个小json便宜几个数量级。
    """
    per_archive_group_name, per_archive_relative_dir, per_archive_part_path_list = archive_group

    per_archive_relative_path = f'{per_archive_relative_dir}/{per_archive_group_name}'

    per_archive_relative_dir_part_list = per_archive_relative_dir.replace(
        '\\', '/').split('/')
    per_subset_name = per_archive_relative_dir_part_list[0]
    per_frame_kind_name = per_archive_relative_dir_part_list[-1]

    save_kind_image_dir_path = os.path.join(save_dataset_path,
                                            SAVE_IMAGE_DIR_NAME,
                                            per_subset_name,
                                            per_frame_kind_name)

    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_MEMBER_DIR_NAME,
                                           per_subset_name,
                                           per_frame_kind_name,
                                           per_archive_group_name)

    total_file_member_count, duplicate_member_count = 0, 0
    unknown_suffix_member_count = 0
    extract_annotation_file_count, skip_annotation_file_count = 0, 0
    not_save_annotation_file_count = 0
    extract_sample_pair_count, skip_sample_pair_count = 0, 0
    not_save_sample_pair_count, invalid_sample_pair_count = 0, 0
    save_image_file_count = 0
    reach_tar_end = False

    unknown_suffix_member_name_list = []
    unknown_top_dir_member_name_list = []
    invalid_concat_image_message_list = []
    unexpected_concat_image_size_message_list = []
    unexpected_png_color_type_message_list = []
    save_image_error_message_list = []
    duplicate_sample_key_message_list = []
    unknown_sample_key_pattern_list = []
    error_message_list = []

    annotation_dict, image_member_dict = {}, {}
    sample_key_count_dict = collections.Counter()

    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    try:
        with tarfile.open(fileobj=archive_reader, mode='r|*') as load_tar_file:
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

                if check_skip_file_or_dir(per_member_name):
                    continue

                # 实测每个tar有且只有一层与tar同名的顶层目录，不一致必须显式上报
                per_member_top_dir_name = per_member_name.split('/')[0]
                if per_member_top_dir_name != per_archive_group_name:
                    unknown_top_dir_member_name_list.append(per_member_name)

                per_member_save_relative_path = get_single_member_save_relative_path(
                    per_member_name, per_archive_group_name)

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

                per_member_name_prefix, per_member_name_suffix = os.path.splitext(
                    per_member_save_relative_path)
                per_member_name_suffix = per_member_name_suffix.lower()
                per_sample_key = os.path.basename(per_member_name_prefix)

                per_member_is_annotation = per_member_name_suffix == ANNOTATION_FILE_SUFFIX
                per_member_is_image = check_image_file_suffix(
                    per_member_save_relative_path)

                if not per_member_is_annotation and not per_member_is_image:
                    # 既不是json也不是图像的成员必须上报，不能默认当图像统计
                    unknown_suffix_member_count += 1
                    unknown_suffix_member_name_list.append(
                        per_member_save_relative_path)
                    error_message_list.append(
                        f'unknown suffix member {per_member_save_relative_path}'
                    )
                    continue

                per_member_is_duplicate = (
                    per_member_is_annotation
                    and per_member_name_prefix in annotation_dict) or (
                        per_member_is_image
                        and per_member_name_prefix in image_member_dict)

                if per_member_is_duplicate:
                    # 同一个tar里出现重名成员时按名写盘会互相覆盖，
                    # 这里改写到独立目录原样保留数据并上报，不能静默丢样本
                    duplicate_member_count += 1
                    error_message_list.append(
                        f'duplicate member name {per_member_save_relative_path}'
                    )

                    save_member_path = os.path.join(
                        save_duplicate_dir_path, f'{duplicate_member_count}',
                        per_member_save_relative_path)
                    load_member_file = load_tar_file.extractfile(per_member)
                    if load_member_file is None:
                        print('6666', per_archive_group_name, per_member.name)
                        error_message_list.append(
                            f'extract duplicate member failed {per_member.name}'
                        )
                        continue

                    per_member_bytes = load_member_file.read()
                    if len(per_member_bytes) != per_member.size:
                        error_message_list.append(
                            f'duplicate member data truncated {per_member_save_relative_path} {len(per_member_bytes)} != {per_member.size}'
                        )
                        continue

                    per_save_error_message = save_single_member_bytes(
                        save_member_path, per_member_bytes, per_member.size)
                    if per_save_error_message:
                        print('6666', per_archive_group_name,
                              per_save_error_message)
                        error_message_list.append(per_save_error_message)
                    continue

                sample_key_count_dict[per_sample_key] += 1

                if per_member_is_annotation:
                    # json必须读进内存用于生成汇总标注(约1KB，代价可忽略)
                    load_member_file = load_tar_file.extractfile(per_member)
                    if load_member_file is None:
                        print('6666', per_archive_group_name, per_member.name)
                        error_message_list.append(
                            f'extract member failed {per_member.name}')
                        continue

                    per_member_bytes = load_member_file.read()
                    if len(per_member_bytes) != per_member.size:
                        error_message_list.append(
                            f'member data truncated {per_member_save_relative_path} {len(per_member_bytes)} != {per_member.size}'
                        )
                        continue

                    per_annotation = None
                    try:
                        per_annotation = json.loads(
                            per_member_bytes.decode('UTF-8'))
                    except Exception as e:
                        error_message_list.append(
                            f'load annotation failed {per_member_save_relative_path} {e}'
                        )

                    if per_annotation is not None and not isinstance(
                            per_annotation, dict):
                        error_message_list.append(
                            f'annotation not a dict {per_member_save_relative_path}'
                        )
                        per_annotation = None

                    annotation_dict[per_member_name_prefix] = [
                        per_sample_key,
                        per_member_save_relative_path,
                        per_annotation,
                    ]

                    if not SAVE_ANNOTATION_MEMBER_FILE_FLAG:
                        # json的5个字段已被完整写进jsonl汇总标注，不再额外落590万个1KB小文件
                        not_save_annotation_file_count += 1
                        continue

                    save_member_path = os.path.join(
                        save_kind_image_dir_path,
                        per_member_save_relative_path)
                    if os.path.exists(save_member_path) and os.path.getsize(
                            save_member_path) == per_member.size:
                        skip_annotation_file_count += 1
                        continue

                    per_save_error_message = save_single_member_bytes(
                        save_member_path, per_member_bytes, per_member.size)
                    if per_save_error_message:
                        print('6666', per_archive_group_name,
                              per_save_error_message)
                        error_message_list.append(per_save_error_message)
                        continue

                    extract_annotation_file_count += 1
                    continue

                # 图像成员: 1024x512的左右拼接图，拆成两张512x512单图落盘
                per_save_reference_image_relative_path = f'{per_member_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}'
                per_save_edited_image_relative_path = f'{per_member_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
                save_reference_image_path = os.path.join(
                    save_kind_image_dir_path,
                    per_save_reference_image_relative_path)
                save_edited_image_path = os.path.join(
                    save_kind_image_dir_path,
                    per_save_edited_image_relative_path)

                per_image_member_info = {
                    'sample_key': per_sample_key,
                    'member_relative_path': per_member_save_relative_path,
                    'member_size': per_member.size,
                    'concat_image_width': None,
                    'concat_image_height': None,
                    'reference_image_relative_path':
                    per_save_reference_image_relative_path,
                    'edited_image_relative_path':
                    per_save_edited_image_relative_path,
                    'save_state': 'invalid',
                }
                image_member_dict[
                    per_member_name_prefix] = per_image_member_info

                load_member_file = load_tar_file.extractfile(per_member)
                if load_member_file is None:
                    print('6666', per_archive_group_name, per_member.name)
                    error_message_list.append(
                        f'extract member failed {per_member.name}')
                    invalid_sample_pair_count += 1
                    continue

                per_member_bytes = load_member_file.read()
                if len(per_member_bytes) != per_member.size:
                    error_message_list.append(
                        f'member data truncated {per_member_save_relative_path} {len(per_member_bytes)} != {per_member.size}'
                    )
                    invalid_sample_pair_count += 1
                    continue

                # 先O(1)从PNG的IHDR头拿宽高，不解码像素
                per_image_header_info = get_png_image_header_info(
                    per_member_bytes[:PNG_IHDR_HEADER_SIZE])
                if per_image_header_info is None:
                    invalid_concat_image_message_list.append(
                        f'{per_member_save_relative_path} png header broken')
                    invalid_sample_pair_count += 1
                    continue

                per_concat_image_width, per_concat_image_height = per_image_header_info[
                    0], per_image_header_info[1]
                per_concat_image_color_type = per_image_header_info[3]

                if per_concat_image_width != per_concat_image_height * CONCAT_IMAGE_WIDTH_HEIGHT_RATIO:
                    # 不是"左右两张等宽等高方图"的拼接图，拆不出参考图与编辑后图，不可训练
                    invalid_concat_image_message_list.append(
                        f'{per_member_save_relative_path} concat image size not 2:1 {per_concat_image_width}x{per_concat_image_height}'
                    )
                    invalid_sample_pair_count += 1
                    continue

                per_image_member_info[
                    'concat_image_width'] = per_concat_image_width
                per_image_member_info[
                    'concat_image_height'] = per_concat_image_height

                if per_concat_image_width != EXPECTED_CONCAT_IMAGE_WIDTH or per_concat_image_height != EXPECTED_CONCAT_IMAGE_HEIGHT:
                    # 尺寸变了只告警不丢样本，但必须显式感知
                    unexpected_concat_image_size_message_list.append(
                        f'{per_member_save_relative_path} unexpected concat image size {per_concat_image_width}x{per_concat_image_height}'
                    )

                if per_concat_image_color_type not in EXPECTED_PNG_COLOR_TYPE_LIST:
                    # 颜色类型变了只告警不丢样本(convert('RGB')照样能拆图)，但必须显式感知
                    unexpected_png_color_type_message_list.append(
                        f'{per_member_save_relative_path} unexpected png color type {per_concat_image_color_type}'
                    )

                if not EXTRACT_IMAGE_FILE_FLAG:
                    # 只建索引模式: 图像继续留在原tar里，之后再单独跑一遍拆图
                    per_image_member_info['save_state'] = 'not_save'
                    not_save_sample_pair_count += 1
                    continue

                if check_single_sample_pair_image_on_disk(
                        save_reference_image_path, save_edited_image_path,
                        per_concat_image_width, per_concat_image_height):
                    per_image_member_info['save_state'] = 'skip'
                    skip_sample_pair_count += 1
                    continue

                per_save_error_message = save_single_sample_pair_image(
                    per_member_bytes, save_reference_image_path,
                    save_edited_image_path, per_concat_image_width,
                    per_concat_image_height)
                if per_save_error_message:
                    print('6666', per_archive_group_name,
                          per_save_error_message)
                    save_image_error_message_list.append(
                        per_save_error_message)
                    invalid_sample_pair_count += 1
                    continue

                per_image_member_info['save_state'] = 'extract'
                extract_sample_pair_count += 1
                save_image_file_count += 2

        reach_tar_end = True
    except Exception as e:
        # tar截断或NAS读失败时保留已处理出的文件，但必须上报，不能静默少样本
        print('7777', per_archive_group_name, len(per_archive_part_path_list),
              e)
        error_message_list.append(f'read archive failed {e}')
    finally:
        archive_reader.close()

    if not reach_tar_end:
        error_message_list.append(
            'not reach tar stream end, archive may be truncated')

    if len(unknown_top_dir_member_name_list) > 0:
        error_message_list.append(
            f'unknown top dir member num {len(unknown_top_dir_member_name_list)} {unknown_top_dir_member_name_list[:3]}'
        )

    # 生成该tar的汇总标注: 只有"参考图 + 编辑后图 + 非空编辑指令"齐备的样本对才算完整有用信息
    valid_annotation_line_list = []
    missing_image_sample_key_list, orphan_image_relative_path_list = [], []
    no_instruction_sample_key_list = []
    invalid_annotation_message_list = []
    frame_index_pair_count_dict = collections.Counter()
    video_id_count_dict = collections.Counter()

    for per_member_name_prefix in sorted(annotation_dict.keys()):
        per_sample_key, per_annotation_relative_path, per_annotation = annotation_dict[
            per_member_name_prefix]

        if per_annotation is None:
            invalid_annotation_message_list.append(
                f'{per_archive_relative_path}/{per_annotation_relative_path} load annotation failed'
            )
            continue

        if per_member_name_prefix not in image_member_dict:
            # json有图没有: 样本对不完整
            missing_image_sample_key_list.append(
                f'{per_archive_relative_path}/{per_sample_key}')
            continue

        per_image_member_info = image_member_dict[per_member_name_prefix]
        if per_image_member_info['save_state'] == 'invalid':
            # 图像成员本身有问题(头坏/尺寸不对/写盘失败)，已在invalid清单里上报过，
            # 这里不再重复记，但绝不能算成有效样本对
            continue

        per_edit_instruction, per_edit_instruction_key_name, per_annotation_error_message_list = get_single_annotation_error_message_list(
            per_annotation, per_sample_key)

        if len(per_annotation_error_message_list) > 0:
            invalid_annotation_message_list.extend([
                f'{per_archive_relative_path} {per_annotation_error_message}'
                for per_annotation_error_message in
                per_annotation_error_message_list
            ])

        if not per_edit_instruction:
            # 没有编辑指令的样本对不可训练(图已照常拆图落盘，有用信息不丢)，隔离上报
            no_instruction_sample_key_list.append(
                f'{per_archive_relative_path}/{per_sample_key}')
            continue

        per_sample_key_parse_result = get_single_sample_key_parse_result(
            per_frame_kind_name, per_sample_key)
        if per_sample_key_parse_result is None:
            # 命名规格变了必须显式感知，但样本对本身有用信息齐备，照样生成，不能丢
            unknown_sample_key_pattern_list.append(
                f'{per_archive_relative_path}/{per_sample_key}')
            per_sample_key_parse_result = {
                'video_id': None,
                'clip_index': None,
                'sub_clip_index': None,
                'global_index': None,
                'extra_index': None,
                'seed_index': None,
                'seed_name': None,
                'start_frame_index': None,
                'end_frame_index': None,
                'frame_pair_index': None,
            }

        per_concat_image_width = per_image_member_info['concat_image_width']
        per_concat_image_height = per_image_member_info['concat_image_height']
        per_single_image_width = per_concat_image_width // CONCAT_IMAGE_WIDTH_HEIGHT_RATIO
        per_single_image_height = per_concat_image_height

        per_reference_image_path = f'{SAVE_IMAGE_DIR_NAME}/{per_subset_name}/{per_frame_kind_name}/{per_image_member_info["reference_image_relative_path"]}'
        per_edited_image_path = f'{SAVE_IMAGE_DIR_NAME}/{per_subset_name}/{per_frame_kind_name}/{per_image_member_info["edited_image_relative_path"]}'

        # 完整有用信息的样本对: 参考图 + 编辑指令 + 编辑后图三者齐备，
        # 再带上原json的全部5个字段与video/seed/帧号等全部有用属性
        valid_annotation_line_list.append(
            json.dumps(
                {
                    'sample_key':
                    per_sample_key,
                    'dataset_task_type':
                    DATASET_TASK_TYPE,
                    'subset_name':
                    per_subset_name,
                    'frame_kind_name':
                    per_frame_kind_name,
                    'archive_name':
                    per_archive_group_name,
                    'video_id':
                    per_sample_key_parse_result['video_id'],
                    'clip_index':
                    per_sample_key_parse_result['clip_index'],
                    'sub_clip_index':
                    per_sample_key_parse_result['sub_clip_index'],
                    'global_index':
                    per_sample_key_parse_result['global_index'],
                    'extra_index':
                    per_sample_key_parse_result['extra_index'],
                    'seed_index':
                    per_sample_key_parse_result['seed_index'],
                    'seed_name':
                    per_sample_key_parse_result['seed_name'],
                    'start_frame_index':
                    per_sample_key_parse_result['start_frame_index'],
                    'end_frame_index':
                    per_sample_key_parse_result['end_frame_index'],
                    'frame_pair_index':
                    per_sample_key_parse_result['frame_pair_index'],
                    'reference_image_path_list': [per_reference_image_path],
                    'reference_image_num':
                    1,
                    'edited_image_path':
                    per_edited_image_path,
                    'reference_image_width':
                    per_single_image_width,
                    'reference_image_height':
                    per_single_image_height,
                    'edited_image_width':
                    per_single_image_width,
                    'edited_image_height':
                    per_single_image_height,
                    'image_file_save_flag':
                    EXTRACT_IMAGE_FILE_FLAG,
                    # 溯源到原始tar与tar内成员名(不是落盘路径),
                    # 出问题时能直接回原始压缩包定位这一张拼接图
                    'source_archive_relative_path':
                    f'{per_subset_name}/{per_frame_kind_name}/{per_archive_group_name}.tar',
                    'source_concat_image_member_name':
                    per_image_member_info['member_relative_path'],
                    'source_annotation_member_name':
                    per_annotation_relative_path,
                    'source_concat_image_width':
                    per_concat_image_width,
                    'source_concat_image_height':
                    per_concat_image_height,
                    'source_concat_image_layout':
                    CONCAT_IMAGE_LAYOUT_NAME,
                    'edit_instruction':
                    per_edit_instruction,
                    'edit_instruction_key_name':
                    per_edit_instruction_key_name,
                    'edit_description':
                    get_single_annotation_text_value(
                        per_annotation, ANNOTATION_EDIT_DESCRIPTION_KEY_NAME),
                    'reference_image_caption':
                    get_single_annotation_text_value(
                        per_annotation,
                        ANNOTATION_REFERENCE_IMAGE_CAPTION_KEY_NAME),
                    'edited_image_caption':
                    get_single_annotation_text_value(
                        per_annotation,
                        ANNOTATION_EDITED_IMAGE_CAPTION_KEY_NAME),
                    'video_caption':
                    get_single_annotation_text_value(
                        per_annotation, ANNOTATION_VIDEO_CAPTION_KEY_NAME),
                    # json的5个字段已被完整写进本行，默认不再额外落盘;
                    # SAVE_ANNOTATION_MEMBER_FILE_FLAG为True时这个路径才真实存在
                    'annotation_file_path':
                    f'{SAVE_IMAGE_DIR_NAME}/{per_subset_name}/{per_frame_kind_name}/{per_annotation_relative_path}',
                    'annotation_file_save_flag':
                    SAVE_ANNOTATION_MEMBER_FILE_FLAG,
                },
                ensure_ascii=False))

        frame_index_pair_count_dict[get_single_frame_index_pair_name(
            per_frame_kind_name, per_sample_key_parse_result)] += 1
        if per_sample_key_parse_result['video_id'] is not None:

            video_id_count_dict[per_sample_key_parse_result['video_id']] += 1

    for per_member_name_prefix, per_image_member_info in image_member_dict.items(
    ):
        if per_member_name_prefix not in annotation_dict:
            # 图有json没有: 没有编辑指令的图不可训练，只能算orphan
            orphan_image_relative_path_list.append(
                f'{per_archive_relative_path}/{per_image_member_info["member_relative_path"]}'
            )

    for per_sample_key, per_sample_key_count in sample_key_count_dict.items():
        # 实测每个sample_key恰好出现2次(1个json + 1个png)，
        # 出现别的次数说明tar内成员错位或跨子目录basename撞名
        if per_sample_key_count != 2:
            duplicate_sample_key_message_list.append(
                f'{per_archive_relative_path}/{per_sample_key} member count {per_sample_key_count} != 2'
            )

    save_annotation_path = os.path.join(save_dataset_path,
                                        SAVE_ANNOTATION_DIR_NAME,
                                        per_subset_name, per_frame_kind_name,
                                        f'{per_archive_group_name}.jsonl')
    try:
        os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
        with open(save_annotation_path, 'w',
                  encoding='UTF-8') as save_json_file:
            for per_valid_annotation_line in valid_annotation_line_list:
                save_json_file.write(f'{per_valid_annotation_line}\n')
    except Exception as e:
        error_message_list.append(
            f'{per_archive_relative_path} save annotation failed {e}')

    per_annotation_member_count = len(annotation_dict)
    per_image_member_count = len(image_member_dict)
    per_valid_sample_pair_count = len(valid_annotation_line_list)

    # 核心对账: tar内json数 == png数 == 有效样本对数，且成员总数刚好是样本对数的2倍。
    # json和png是"成对一起丢"的，所以只比对落盘文件同名配对的校验永远查不出缺样本，
    # 必须拿tar头里数出来的成员数当ground truth。
    if per_annotation_member_count != per_image_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} annotation member count {per_annotation_member_count} != image member count {per_image_member_count}'
        )
    if total_file_member_count % 2 != 0:
        error_message_list.append(
            f'{per_archive_relative_path} tar file member count not even {total_file_member_count}'
        )
    if per_valid_sample_pair_count * 2 != total_file_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} valid sample pair count not match {per_valid_sample_pair_count} * 2 != {total_file_member_count}'
        )
    if per_annotation_member_count + per_image_member_count + duplicate_member_count + unknown_suffix_member_count != total_file_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} member classify count not match {per_annotation_member_count} + {per_image_member_count} + {duplicate_member_count} + {unknown_suffix_member_count} != {total_file_member_count}'
        )

    # 每个png成员必须有明确归属: 要么拆图落盘成功(extract)、要么已存在(skip)、
    # 要么只建索引(not_save)、要么进invalid隔离清单，一张都不会凭空消失
    if extract_sample_pair_count + skip_sample_pair_count + not_save_sample_pair_count + invalid_sample_pair_count != per_image_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} process sample pair count not match {extract_sample_pair_count} + {skip_sample_pair_count} + {not_save_sample_pair_count} + {invalid_sample_pair_count} != {per_image_member_count}'
        )

    # 拆图口径下每对必须落盘2张图，落盘图像文件数必须是extract对数的2倍
    if save_image_file_count != extract_sample_pair_count * 2:
        error_message_list.append(
            f'{per_archive_relative_path} save image file count not match {save_image_file_count} != {extract_sample_pair_count} * 2'
        )

    if SAVE_ANNOTATION_MEMBER_FILE_FLAG:
        if extract_annotation_file_count + skip_annotation_file_count != per_annotation_member_count:
            error_message_list.append(
                f'{per_archive_relative_path} process annotation file count not match {extract_annotation_file_count} + {skip_annotation_file_count} != {per_annotation_member_count}'
            )
    elif not_save_annotation_file_count != per_annotation_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} not save annotation file count not match {not_save_annotation_file_count} != {per_annotation_member_count}'
        )

    return {
        'archive_relative_path': per_archive_relative_path,
        'subset_name': per_subset_name,
        'frame_kind_name': per_frame_kind_name,
        'archive_name': per_archive_group_name,
        'total_file_member_count': total_file_member_count,
        'annotation_member_count': per_annotation_member_count,
        'image_member_count': per_image_member_count,
        'valid_sample_pair_count': per_valid_sample_pair_count,
        'extract_sample_pair_count': extract_sample_pair_count,
        'skip_sample_pair_count': skip_sample_pair_count,
        'not_save_sample_pair_count': not_save_sample_pair_count,
        'invalid_sample_pair_count': invalid_sample_pair_count,
        'save_image_file_count': save_image_file_count,
        'extract_annotation_file_count': extract_annotation_file_count,
        'skip_annotation_file_count': skip_annotation_file_count,
        'not_save_annotation_file_count': not_save_annotation_file_count,
        'duplicate_member_count': duplicate_member_count,
        'unknown_suffix_member_count': unknown_suffix_member_count,
        'unique_video_id_count': len(video_id_count_dict),
        'frame_index_pair_count_dict': dict(frame_index_pair_count_dict),
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_subset_name}/{per_frame_kind_name}/{per_archive_group_name}.jsonl',
        'unknown_suffix_member_name_list': unknown_suffix_member_name_list,
        'unknown_top_dir_member_name_list': unknown_top_dir_member_name_list,
        'missing_image_sample_key_list': missing_image_sample_key_list,
        'orphan_image_relative_path_list': orphan_image_relative_path_list,
        'no_instruction_sample_key_list': no_instruction_sample_key_list,
        'invalid_annotation_message_list': invalid_annotation_message_list,
        'invalid_concat_image_message_list': invalid_concat_image_message_list,
        'unexpected_concat_image_size_message_list':
        unexpected_concat_image_size_message_list,
        'unexpected_png_color_type_message_list':
        unexpected_png_color_type_message_list,
        'save_image_error_message_list': save_image_error_message_list,
        'duplicate_sample_key_message_list': duplicate_sample_key_message_list,
        'unknown_sample_key_pattern_list': unknown_sample_key_pattern_list,
        'error_message_list': error_message_list,
    }


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表和按分片归组后的压缩包列表"""
    file_copy_pair_list = []
    archive_part_path_dict = {}
    for per_root_path, per_dir_name_list, per_file_name_list in os.walk(
            root_dataset_path):
        # .cache里有4249个文件，直接在遍历时剪掉整棵子树，不要走进去
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
        per_archive_part_path_list = [
            per_archive_part_path
            for _, per_archive_part_path in per_archive_part_list
        ]
        archive_group_list.append([
            per_archive_group_name,
            per_archive_relative_dir,
            per_archive_part_path_list,
        ])

    file_copy_pair_list = sorted(file_copy_pair_list, key=lambda x: x[0])

    return file_copy_pair_list, archive_group_list


def check_single_archive_dir_on_disk(archive_check_pair):
    """可选的二次对账: os.walk单个tar的输出目录，核对落盘图像数和参考图/编辑后图配对"""
    per_archive_relative_path, per_archive_dir_path, per_expected_sample_pair_count = archive_check_pair

    error_message_list = []
    if not os.path.exists(per_archive_dir_path):
        error_message_list.append(
            f'{per_archive_relative_path} archive dir not exist')

        return [per_archive_relative_path, 0, 0, error_message_list]

    reference_image_name_prefix_dict, edited_image_name_prefix_dict = {}, {}
    unknown_suffix_file_count = 0
    for per_root_path, _, per_file_name_list in os.walk(per_archive_dir_path):
        for per_file_name in per_file_name_list:
            per_file_path = os.path.join(per_root_path, per_file_name)
            per_file_relative_path = os.path.relpath(per_file_path,
                                                     per_archive_dir_path)
            per_file_relative_path = per_file_relative_path.replace('\\', '/')

            if per_file_relative_path.endswith(
                    SAVE_REFERENCE_IMAGE_NAME_SUFFIX):
                reference_image_name_prefix_dict[per_file_relative_path[:-len(
                    SAVE_REFERENCE_IMAGE_NAME_SUFFIX)]] = 1
            elif per_file_relative_path.endswith(
                    SAVE_EDITED_IMAGE_NAME_SUFFIX):
                edited_image_name_prefix_dict[per_file_relative_path[:-len(
                    SAVE_EDITED_IMAGE_NAME_SUFFIX)]] = 1
            elif SAVE_ANNOTATION_MEMBER_FILE_FLAG and os.path.splitext(
                    per_file_relative_path)[1].lower(
                    ) == ANNOTATION_FILE_SUFFIX:
                continue
            else:
                # 既不是参考图也不是编辑后图的文件必须上报，不能默认当图像统计
                unknown_suffix_file_count += 1

    per_reference_image_count = len(reference_image_name_prefix_dict)
    per_edited_image_count = len(edited_image_name_prefix_dict)
    per_matched_count = len(
        set(reference_image_name_prefix_dict.keys())
        & set(edited_image_name_prefix_dict.keys()))

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_archive_relative_path} unknown suffix file num {unknown_suffix_file_count}'
        )
    if per_reference_image_count != per_edited_image_count:
        error_message_list.append(
            f'{per_archive_relative_path} reference image count {per_reference_image_count} != edited image count {per_edited_image_count}'
        )
    if per_matched_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_relative_path} matched sample pair count not match {per_matched_count} != {per_expected_sample_pair_count}'
        )

    return [
        per_archive_relative_path,
        per_reference_image_count,
        per_edited_image_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, archive_result_list):
    """可选的二次对账: 遍历输出目录核对每个tar目录里的落盘图像数和参考图/编辑后图配对"""
    archive_check_pair_list = []
    for per_archive_result in archive_result_list:
        archive_check_pair_list.append([
            per_archive_result['archive_relative_path'],
            os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                         per_archive_result['subset_name'],
                         per_archive_result['frame_kind_name'],
                         per_archive_result['archive_name']),
            per_archive_result['valid_sample_pair_count'],
        ])

    error_message_list = []
    total_reference_image_count, total_edited_image_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_archive_dir_on_disk, archive_check_pair_list),
                                     total=len(archive_check_pair_list)):
            _, per_reference_image_count, per_edited_image_count, per_error_message_list = per_check_result
            total_reference_image_count += per_reference_image_count
            total_edited_image_count += per_edited_image_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk reference image:', total_reference_image_count,
          'on disk edited image:', total_edited_image_count)

    return error_message_list


def save_check_result(save_dataset_path, archive_result_list):
    """汇总所有tar的解压与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_file_member_count, total_valid_sample_pair_count = 0, 0
    total_annotation_member_count, total_image_member_count = 0, 0
    total_extract_sample_pair_count, total_skip_sample_pair_count = 0, 0
    total_not_save_sample_pair_count, total_invalid_sample_pair_count = 0, 0
    total_save_image_file_count = 0
    total_extract_annotation_file_count, total_skip_annotation_file_count = 0, 0
    total_not_save_annotation_file_count = 0
    total_duplicate_member_count, total_unknown_suffix_member_count = 0, 0

    subset_sample_pair_count_dict = collections.Counter()
    frame_kind_sample_pair_count_dict = collections.Counter()
    frame_index_pair_count_dict = collections.Counter()
    archive_sample_pair_count_dict = {}
    save_annotation_relative_path_count_dict = {}

    missing_image_sample_key_list, orphan_image_relative_path_list = [], []
    no_instruction_sample_key_list, invalid_annotation_message_list = [], []
    invalid_concat_image_message_list, save_image_error_message_list = [], []
    unexpected_concat_image_size_message_list = []
    unexpected_png_color_type_message_list = []
    duplicate_sample_key_message_list, unknown_sample_key_pattern_list = [], []
    unknown_suffix_member_name_list, unknown_top_dir_member_name_list = [], []
    error_message_list, warning_message_list = [], []

    for per_archive_result in archive_result_list:
        per_archive_relative_path = per_archive_result['archive_relative_path']

        total_file_member_count += per_archive_result[
            'total_file_member_count']
        total_valid_sample_pair_count += per_archive_result[
            'valid_sample_pair_count']
        total_annotation_member_count += per_archive_result[
            'annotation_member_count']
        total_image_member_count += per_archive_result['image_member_count']
        total_extract_sample_pair_count += per_archive_result[
            'extract_sample_pair_count']
        total_skip_sample_pair_count += per_archive_result[
            'skip_sample_pair_count']
        total_not_save_sample_pair_count += per_archive_result[
            'not_save_sample_pair_count']
        total_invalid_sample_pair_count += per_archive_result[
            'invalid_sample_pair_count']
        total_save_image_file_count += per_archive_result[
            'save_image_file_count']
        total_extract_annotation_file_count += per_archive_result[
            'extract_annotation_file_count']
        total_skip_annotation_file_count += per_archive_result[
            'skip_annotation_file_count']
        total_not_save_annotation_file_count += per_archive_result[
            'not_save_annotation_file_count']
        total_duplicate_member_count += per_archive_result[
            'duplicate_member_count']
        total_unknown_suffix_member_count += per_archive_result[
            'unknown_suffix_member_count']

        subset_sample_pair_count_dict[per_archive_result[
            'subset_name']] += per_archive_result['valid_sample_pair_count']
        frame_kind_sample_pair_count_dict[
            per_archive_result['frame_kind_name']] += per_archive_result[
                'valid_sample_pair_count']
        frame_index_pair_count_dict.update(
            per_archive_result['frame_index_pair_count_dict'])
        archive_sample_pair_count_dict[
            per_archive_relative_path] = per_archive_result[
                'valid_sample_pair_count']
        save_annotation_relative_path_count_dict[per_archive_result[
            'save_annotation_relative_path']] = per_archive_result[
                'valid_sample_pair_count']

        missing_image_sample_key_list.extend(
            per_archive_result['missing_image_sample_key_list'])
        orphan_image_relative_path_list.extend(
            per_archive_result['orphan_image_relative_path_list'])
        no_instruction_sample_key_list.extend(
            per_archive_result['no_instruction_sample_key_list'])
        invalid_annotation_message_list.extend(
            per_archive_result['invalid_annotation_message_list'])
        invalid_concat_image_message_list.extend([
            f'{per_archive_relative_path}/{per_message}' for per_message in
            per_archive_result['invalid_concat_image_message_list']
        ])
        unexpected_concat_image_size_message_list.extend([
            f'{per_archive_relative_path}/{per_message}' for per_message in
            per_archive_result['unexpected_concat_image_size_message_list']
        ])
        unexpected_png_color_type_message_list.extend([
            f'{per_archive_relative_path}/{per_message}' for per_message in
            per_archive_result['unexpected_png_color_type_message_list']
        ])
        save_image_error_message_list.extend([
            f'{per_archive_relative_path} {per_message}' for per_message in
            per_archive_result['save_image_error_message_list']
        ])
        duplicate_sample_key_message_list.extend(
            per_archive_result['duplicate_sample_key_message_list'])
        unknown_sample_key_pattern_list.extend(
            per_archive_result['unknown_sample_key_pattern_list'])
        unknown_suffix_member_name_list.extend([
            f'{per_archive_relative_path}/{per_member_name}'
            for per_member_name in
            per_archive_result['unknown_suffix_member_name_list']
        ])
        unknown_top_dir_member_name_list.extend([
            f'{per_archive_relative_path}/{per_member_name}'
            for per_member_name in
            per_archive_result['unknown_top_dir_member_name_list']
        ])

        if len(per_archive_result['error_message_list']) > 0:
            print('7777', per_archive_relative_path,
                  per_archive_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_archive_relative_path} error num {len(per_archive_result["error_message_list"])} {per_archive_result["error_message_list"][:3]}'
            )

    for per_frame_kind_name, per_sample_pair_count_range in EXPECTED_FRAME_KIND_SAMPLE_PAIR_COUNT_RANGE_DICT.items(
    ):
        per_sample_pair_count = frame_kind_sample_pair_count_dict.get(
            per_frame_kind_name, 0)
        if not per_sample_pair_count_range[
                0] <= per_sample_pair_count <= per_sample_pair_count_range[1]:
            # 官方没给全量精确条数，只按实测每tar的样本对数做软校验，打印告警不判失败
            warning_message_list.append(
                f'{per_frame_kind_name} sample pair count {per_sample_pair_count} not in {per_sample_pair_count_range}'
            )

    print('3333', 'total archive:', len(archive_result_list),
          'total tar file member:', total_file_member_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'total annotation member:', total_annotation_member_count,
          'total image member:', total_image_member_count)
    print('3333', 'extract sample pair:', total_extract_sample_pair_count,
          'skip sample pair:', total_skip_sample_pair_count,
          'not save sample pair:', total_not_save_sample_pair_count,
          'invalid sample pair:', total_invalid_sample_pair_count,
          'save image file:', total_save_image_file_count)
    print('3333', 'duplicate member:', total_duplicate_member_count,
          'unknown suffix member:',
          total_unknown_suffix_member_count, 'missing image:',
          len(missing_image_sample_key_list), 'orphan image:',
          len(orphan_image_relative_path_list), 'no instruction:',
          len(no_instruction_sample_key_list), 'invalid annotation:',
          len(invalid_annotation_message_list), 'invalid concat image:',
          len(invalid_concat_image_message_list), 'save image error:',
          len(save_image_error_message_list))
    print('3333', 'subset sample pair:', dict(subset_sample_pair_count_dict))
    print('3333', 'frame kind sample pair:',
          dict(frame_kind_sample_pair_count_dict))
    print('3333', 'frame index pair:', dict(frame_index_pair_count_dict))
    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'dataset_license_name':
        DATASET_LICENSE_NAME,
        'extract_image_file_flag':
        EXTRACT_IMAGE_FILE_FLAG,
        'save_annotation_member_file_flag':
        SAVE_ANNOTATION_MEMBER_FILE_FLAG,
        'concat_image_layout_name':
        CONCAT_IMAGE_LAYOUT_NAME,
        'total_archive_count':
        len(archive_result_list),
        'total_tar_file_member_count':
        total_file_member_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_annotation_member_count':
        total_annotation_member_count,
        'total_image_member_count':
        total_image_member_count,
        'total_extract_sample_pair_count':
        total_extract_sample_pair_count,
        'total_skip_sample_pair_count':
        total_skip_sample_pair_count,
        'total_not_save_sample_pair_count':
        total_not_save_sample_pair_count,
        'total_invalid_sample_pair_count':
        total_invalid_sample_pair_count,
        'total_save_image_file_count':
        total_save_image_file_count,
        'total_extract_annotation_file_count':
        total_extract_annotation_file_count,
        'total_skip_annotation_file_count':
        total_skip_annotation_file_count,
        'total_not_save_annotation_file_count':
        total_not_save_annotation_file_count,
        'total_duplicate_member_count':
        total_duplicate_member_count,
        'total_unknown_suffix_member_count':
        total_unknown_suffix_member_count,
        'missing_image_count':
        len(missing_image_sample_key_list),
        'orphan_image_count':
        len(orphan_image_relative_path_list),
        'no_instruction_count':
        len(no_instruction_sample_key_list),
        'invalid_annotation_count':
        len(invalid_annotation_message_list),
        'invalid_concat_image_count':
        len(invalid_concat_image_message_list),
        'save_image_error_count':
        len(save_image_error_message_list),
        'subset_sample_pair_count_dict':
        dict(subset_sample_pair_count_dict),
        'frame_kind_sample_pair_count_dict':
        dict(frame_kind_sample_pair_count_dict),
        'frame_index_pair_count_dict':
        dict(frame_index_pair_count_dict),
        'archive_sample_pair_count_dict':
        archive_sample_pair_count_dict,
        'save_annotation_relative_path_count_dict':
        save_annotation_relative_path_count_dict,
        'missing_image_sample_key_list':
        sorted(missing_image_sample_key_list)[:MAX_SAVE_MESSAGE_NUM],
        'orphan_image_relative_path_list':
        sorted(orphan_image_relative_path_list)[:MAX_SAVE_MESSAGE_NUM],
        'no_instruction_sample_key_list':
        sorted(no_instruction_sample_key_list)[:MAX_SAVE_MESSAGE_NUM],
        'invalid_annotation_message_list':
        sorted(invalid_annotation_message_list)[:MAX_SAVE_MESSAGE_NUM],
        'invalid_concat_image_message_list':
        sorted(invalid_concat_image_message_list)[:MAX_SAVE_MESSAGE_NUM],
        'unexpected_concat_image_size_message_list':
        sorted(unexpected_concat_image_size_message_list)
        [:MAX_SAVE_MESSAGE_NUM],
        'unexpected_png_color_type_message_list':
        sorted(unexpected_png_color_type_message_list)[:MAX_SAVE_MESSAGE_NUM],
        'save_image_error_message_list':
        sorted(save_image_error_message_list)[:MAX_SAVE_MESSAGE_NUM],
        'duplicate_sample_key_message_list':
        sorted(duplicate_sample_key_message_list)[:MAX_SAVE_MESSAGE_NUM],
        'unknown_sample_key_pattern_list':
        sorted(unknown_sample_key_pattern_list)[:MAX_SAVE_MESSAGE_NUM],
        'unknown_suffix_member_name_list':
        sorted(unknown_suffix_member_name_list)[:MAX_SAVE_MESSAGE_NUM],
        'unknown_top_dir_member_name_list':
        sorted(unknown_top_dir_member_name_list)[:MAX_SAVE_MESSAGE_NUM],
        'warning_message_list':
        warning_message_list,
        'check_error_message_list':
        error_message_list[:MAX_SAVE_MESSAGE_NUM],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    if total_valid_sample_pair_count == 0:
        error_message_list.append('no valid sample pair found')
    if len(missing_image_sample_key_list) > 0:
        error_message_list.append(
            f'missing image count {len(missing_image_sample_key_list)}')
    if len(orphan_image_relative_path_list) > 0:
        error_message_list.append(
            f'orphan image count {len(orphan_image_relative_path_list)}')
    if len(no_instruction_sample_key_list) > 0:
        error_message_list.append(
            f'no instruction count {len(no_instruction_sample_key_list)}')
    if len(invalid_annotation_message_list) > 0:
        error_message_list.append(
            f'invalid annotation count {len(invalid_annotation_message_list)}')
    if len(invalid_concat_image_message_list) > 0:
        error_message_list.append(
            f'invalid concat image count {len(invalid_concat_image_message_list)}'
        )
    if len(unexpected_concat_image_size_message_list) > 0:
        error_message_list.append(
            f'unexpected concat image size count {len(unexpected_concat_image_size_message_list)}'
        )
    if len(unexpected_png_color_type_message_list) > 0:
        error_message_list.append(
            f'unexpected png color type count {len(unexpected_png_color_type_message_list)}'
        )
    if len(save_image_error_message_list) > 0:
        error_message_list.append(
            f'save image error count {len(save_image_error_message_list)}')
    if len(duplicate_sample_key_message_list) > 0:
        error_message_list.append(
            f'duplicate sample key count {len(duplicate_sample_key_message_list)}'
        )
    if len(unknown_sample_key_pattern_list) > 0:
        error_message_list.append(
            f'unknown sample key pattern count {len(unknown_sample_key_pattern_list)}'
        )
    if total_duplicate_member_count > 0:
        error_message_list.append(
            f'duplicate member count {total_duplicate_member_count}')
    if total_unknown_suffix_member_count > 0:
        error_message_list.append(
            f'unknown suffix member count {total_unknown_suffix_member_count}')
    if total_valid_sample_pair_count * 2 != total_file_member_count:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} * 2 != {total_file_member_count}'
        )
    if EXTRACT_IMAGE_FILE_FLAG and total_extract_sample_pair_count + total_skip_sample_pair_count != total_valid_sample_pair_count:
        error_message_list.append(
            f'total extract and skip sample pair count not match {total_extract_sample_pair_count} + {total_skip_sample_pair_count} != {total_valid_sample_pair_count}'
        )

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    subset_error_message_list = check_required_subset_complete(
        root_dataset_path)
    if len(subset_error_message_list) > 0:
        # 数据集本身不完整就没必要跑几十小时解压
        raise Exception(
            f'check subset failed {subset_error_message_list[:20]}')

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    os.makedirs(os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME),
                exist_ok=True)
    os.makedirs(os.path.join(save_dataset_path, SAVE_ANNOTATION_DIR_NAME),
                exist_ok=True)

    file_copy_pair_list, archive_group_list = get_all_file_and_archive_group(
        root_dataset_path)

    print('1111', len(file_copy_pair_list), len(archive_group_list))
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(archive_group_list) > 0:
        print('1111', archive_group_list[0][0], archive_group_list[0][1],
              len(archive_group_list[0][2]))

    expected_archive_group_num = sum(
        EXPECTED_SUBSET_KIND_ARCHIVE_NUM_DICT.values())
    if len(archive_group_list) != expected_archive_group_num:
        raise Exception(
            f'archive group num not match {len(archive_group_list)} != {expected_archive_group_num}'
        )

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
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_archive_result in tqdm(pool.imap_unordered(
                extract_func, archive_group_list),
                                       total=len(archive_group_list)):
            archive_result_list.append(per_archive_result)

            print('2222', per_archive_result['archive_relative_path'],
                  'tar file member:',
                  per_archive_result['total_file_member_count'],
                  'valid sample pair:',
                  per_archive_result['valid_sample_pair_count'], 'extract:',
                  per_archive_result['extract_sample_pair_count'], 'skip:',
                  per_archive_result['skip_sample_pair_count'], 'not save:',
                  per_archive_result['not_save_sample_pair_count'], 'invalid:',
                  per_archive_result['invalid_sample_pair_count'],
                  'save image file:',
                  per_archive_result['save_image_file_count'],
                  'duplicate member:',
                  per_archive_result['duplicate_member_count'])

    check_error_message_list = save_check_result(save_dataset_path,
                                                 archive_result_list)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG and EXTRACT_IMAGE_FILE_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, archive_result_list)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解压/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/BM-6M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
