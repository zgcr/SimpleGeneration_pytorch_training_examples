import os
import re
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'bm_6m'

SAVE_DATASET_DIR_NAME = 'BM-6M'

# ==============================================================================
# 【这个数据集为什么还能出t2i】
# BM-6M(ByteMorph-6M)本身是图像编辑数据集(见同目录013.resave_bm_6m_ti2i_dataset.py),
# 但官方README明确写了它的用途包含文生图: "The primary use of ByteMorph is research
# on text-to-image and instruction-based image editing."
# 上游015解压出来的每一行标注除了编辑指令之外，还带**两条针对单张图的详细整图caption**:
#   reference_image_caption -> 描述参考图(编辑前帧)这一张图
#   edited_image_caption    -> 描述编辑后图(编辑后帧)这一张图
# 也就是天然的"一张图 + 这张图自己的描述"，正好是t2i样本。
# 所以每个编辑对固定拆成**2条独立的t2i样本**(参考图配自己的caption、
# 编辑后图配自己的caption)，按方案确认两条都保留。
#
# 【上游015解压产物实测规格(全量扫过2120个jsonl、5878678行，非抽样)】
# BM-6M/
# ├── unzip_images/<subset-N>/<kind>/<batch_i>/<sample_key>_reference.png
# │                                            <sample_key>_edited.png  1175.7万张
# ├── unzip_annotations/<subset-N>/<kind>/<batch_i>.jsonl  2120个(15G)，共5878678行
# └── unzip_check_missing_images.json  上游对账报告(check_error_count=0)
# 本脚本只读这2120个jsonl定位样本，**绝不os.walk图像目录**:
# 上游解出1175.7万个小文件，扫一遍目录树在NAS上不可接受。
#
# 【实测关键规格】
# - 所有图像**恒为512x512 RGB**，所以短边过滤与宽高比过滤一条都不会命中，只作兜底;
# - reference_image_caption长度0~2484(其中1条为空)、
#   edited_image_caption长度5~3791，p50都在330字符上下;
# - 落盘图像名与sample_key严格对应(basename == <sample_key>_reference.png /
#   <sample_key>_edited.png，全量0例不符);
# - 行内subset_name/frame_kind_name/archive_name与所在目录100%一致(全量0例不符)。
#
# 【关于"同一张帧会被落盘多次"这件事(方案已确认接受)】
# 已用像素md5交叉验证(20个batch/95组，100%命中): 同一个batch内，
# frames的1个对与multi的3个对来自**同一段4帧视频**:
#   sampled_frames      : (帧0 | 帧3)
#   sampled_multi_frames: (帧0 | 帧1) (帧1 | 帧2) (帧2 | 帧3)
# 所以 frames.reference 与 multi frame_0_1.reference 像素完全相同、
# multi frame_0_1.edited 与 multi frame_1_2.reference 像素完全相同...
# 按帧身份去重后587万对里只有6068568个唯一帧，而本脚本按方案(每个编辑对出2条)
# 会产出约1175万条t2i样本 -> 约有568万个帧会以两个不同的保存名各落盘一份。
# 这样做的收益是: 同一张帧在frames侧和multi侧被上游VLM**各自独立打过一次标**，
# 两条caption实测几乎100%不同(抽样200组帧0的两条caption 0组相同、
# 300组帧1的两条caption只有1组相同)，等于同一张图拿到两条不同风格的文本监督。
# 代价是图像存储与inode翻倍(约1175万张jpg)，且"这两条样本其实是同一张图"这个
# 信息在新数据集里不可见。
#
# 【848组跨batch/subset重复的sample_key(必须处理，否则静默覆盖)】
# 全量统计: 848个sample_key各出现2次(frames 212组 + multi 636组)。
# 已核对原始key完全相同(不是大小写差异)，但图像md5与文本都不同，
# 是上游对同一视频片段重复采样出的两个不同样本。
# 按方案确认**涉及重复sample_key的样本全部过滤掉不保留**(1696行 -> 3392条t2i候选),
# 所以本脚本必须先全量扫一遍所有jsonl统计sample_key出现次数、再做第二遍解析。
#
# 【子集划分】
# 按方案确认用官方的9个下载分片subset-1..9作为9个子集(subset_1..subset_9)。
# 注意这不是"任务类型"(t2i没有任务类型概念)，只是官方自带的一级划分，
# 这样每个子集42万~225万条、最多225个文件夹，NAS单目录压力可接受，
# 不需要像003(GPIC)那样再套一层PER_SET_FOLDER_NUM分层。
# ==============================================================================

# 上游015解压脚本按tar另存的汇总标注，每行一个完整编辑对(含两条单图caption)
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如unzip_images/subset-1/sampled_frames/batch_0/xxx_reference.png)，
# 不需要再拼images子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 过滤掉无用信息后unzip_annotations下只应该有这9个子集目录
LOAD_SUBSET_DIR_NAME_LIST = [
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

# 每个子集目录下有且只有这2个kind目录，两个kind的样本全部保留
LOAD_FRAME_KIND_DIR_NAME_LIST = [
    SAMPLED_FRAMES_DIR_NAME,
    SAMPLED_MULTI_FRAMES_DIR_NAME,
]

# 样本id，含大小写(video_id是大小写敏感的youtube id)。
# 全量核对: 转小写后唯一key数5877830 = 5878678 - 848，转小写没有引入任何新撞名
ANNOTATION_SAMPLE_KEY_KEY_NAME = 'sample_key'

# 行内记录的子集名/kind名/tar名，必须与该标注文件所在的目录严格一致，
# 不一致说明上游产物被搬动过(实测全量0例不符)
ANNOTATION_SUBSET_NAME_KEY_NAME = 'subset_name'

ANNOTATION_FRAME_KIND_NAME_KEY_NAME = 'frame_kind_name'

ANNOTATION_ARCHIVE_NAME_KEY_NAME = 'archive_name'

ANNOTATION_REFERENCE_IMAGE_PATH_LIST_KEY_NAME = 'reference_image_path_list'

ANNOTATION_EDITED_IMAGE_PATH_KEY_NAME = 'edited_image_path'

# 一个编辑对固定拆成2条独立的t2i样本: 每张图配自己那条整图caption。
# 顺序固定为[参考图(编辑前帧), 编辑后图(编辑后帧)]，保证输出顺序可复现。
# 每一项是[图像路径字段名, caption字段名, 上游落盘图像名后缀]，
# 最后那个后缀用于反查"这张图是否真的属于这个sample_key"
ANNOTATION_IMAGE_KEY_NAME_PAIR_LIST = [
    [
        ANNOTATION_REFERENCE_IMAGE_PATH_LIST_KEY_NAME,
        'reference_image_caption',
        '_reference.png',
    ],
    [
        ANNOTATION_EDITED_IMAGE_PATH_KEY_NAME,
        'edited_image_caption',
        '_edited.png',
    ],
]

SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 9个子集名(即官方的9个下载分片，把中划线换成下划线)。
# 保留下来的子集集合必须与这个白名单严格一一对应，不多也不少
SAVE_SET_NAME_LIST = [
    'subset_1',
    'subset_2',
    'subset_3',
    'subset_4',
    'subset_5',
    'subset_6',
    'subset_7',
    'subset_8',
    'subset_9',
]

# 新标注固定只存width/height/t2i_caption/t2i_caption_length四个key。
# 上游jsonl里剩下的属性按方案确认全部丢弃、不另存索引(落盘后**永久丢失**,
# 想恢复只能回 huggingface_datasets_unzip/BM-6M 重跑):
#   edit_instruction / edit_instruction_key_name : 编辑指令，由013的ti2i脚本承载
#   edit_description        : 陈述句版变化描述，丢弃
#   video_caption           : **视频级**运动caption，同一段视频的4个帧对共享同一条
#                             (44条为空)，粒度太粗、且不是单图描述，丢弃
#   video_id / clip_index / sub_clip_index / global_index / extra_index /
#   seed_index / seed_name / start_frame_index / end_frame_index /
#   frame_pair_index        : 溯源与帧序元信息。**丢掉后无法再按视频划分train/val
#                             防同视频泄漏、也无法再判断"哪两条t2i样本其实是同一张
#                             帧的两条caption"**，按方案确认丢弃
#   subset_name / frame_kind_name / archive_name / source_archive_relative_path /
#   source_concat_image_member_name / source_annotation_member_name /
#   source_concat_image_width / source_concat_image_height /
#   source_concat_image_layout / image_file_save_flag / annotation_file_path /
#   annotation_file_save_flag / dataset_task_type / reference_image_num /
#   reference_image_width / reference_image_height / edited_image_width /
#   edited_image_height     : 溯源与冗余信息(前三项已内含在保存图像名里;
#                             宽高一律以实际解码shape为准，不采信上游数值)
#   "参考图与编辑后图本来构成一个编辑对"这个配对关系
#                           : 拆成2条独立t2i样本后彻底丢失
#   数据集授权cc0-1.0        : 记录在上游unzip_check_missing_images.json里，新格式不带
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'width',
    'height',
    't2i_caption',
    't2i_caption_length',
]

# 保存图像名里只允许小写字母/数字/下划线/中划线/点，与001~013完全一致。
# 实测1175.4万个保存名100%满足这个模式、0个非法名、0撞名，最长104字符
# (bm_6m_subset_9_sampled_multi_frames_batch_195_<sample_key>_edited.jpg)，
# 远低于文件系统单文件名255字节的上限
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 原始图像名前缀(即<sample_key>_reference / <sample_key>_edited)小写后
# 必须只含小写字母/数字/下划线/中划线(实测全量0例不符)，
# 不满足的样本没法保证保存图像名合法，直接丢弃并上报
VALID_IMAGE_NAME_PREFIX_PATTERN = re.compile(r'^[a-z0-9_\-]+$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉。
# 上游015已硬校验过拼接图colortype恒为2(truecolor RGB)，所以这里一条都不会命中
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 每个<subset>/<kind>下的实测jsonl分片数(与上游015的tar数一一对应)，合计2120个。
# 注意subset-4/sampled_multi_frames只有23个(上游huggingface仓库自身就是稀疏编号，
# 不是本地缺失，上游015已对账过)
EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT = {
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

EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 2120

# 每个<subset>/<kind>下的实测标注行数(全量扫描，非抽样)，合计5878678行,
# 与上游unzip_check_missing_images.json里的total_valid_sample_pair_count一致
EXPECTED_SUBSET_KIND_ANNOTATION_COUNT_DICT = {
    'subset-1/sampled_frames': 126015,
    'subset-1/sampled_multi_frames': 378045,
    'subset-2/sampled_frames': 126094,
    'subset-2/sampled_multi_frames': 378282,
    'subset-3/sampled_frames': 120537,
    'subset-3/sampled_multi_frames': 361611,
    'subset-4/sampled_frames': 222144,
    'subset-4/sampled_multi_frames': 96762,
    'subset-5/sampled_frames': 167204,
    'subset-5/sampled_multi_frames': 501612,
    'subset-6/sampled_frames': 195077,
    'subset-6/sampled_multi_frames': 585231,
    'subset-7/sampled_frames': 266960,
    'subset-7/sampled_multi_frames': 800880,
    'subset-8/sampled_frames': 106954,
    'subset-8/sampled_multi_frames': 320862,
    'subset-9/sampled_frames': 281102,
    'subset-9/sampled_multi_frames': 843306,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = 5878678

# 每行拆2条t2i样本，所以候选样本数恒为标注行数的2倍
EXPECTED_PER_ANNOTATION_IMAGE_NUM = 2

EXPECTED_TOTAL_IMAGE_ANNOTATION_COUNT = 11757356

# 跨batch/subset重复的sample_key: 实测848个key各出现2次(1696行)，整组丢弃，
# 对t2i来说就是丢掉3392条候选样本，解析阶段逐项硬对账
EXPECTED_DUPLICATE_SAMPLE_KEY_COUNT = 848

EXPECTED_DUPLICATE_SAMPLE_KEY_ANNOTATION_COUNT = 1696

EXPECTED_DUPLICATE_SAMPLE_KEY_IMAGE_ANNOTATION_COUNT = 3392

# 文本层各类不合格描述的实测精确条数(已排除上面那3392条)，解析阶段逐项硬对账。
# 判定顺序必须与下面worker里的顺序完全一致(空 -> 只剩标点 -> 过短 -> 过长 -> 占位符),
# 否则这些数字会互相串位:
#   empty_caption_count        : 1条(唯一一条空的reference_image_caption)
#   no_word_char_caption_count : 2条(只剩标点，同一张帧在两个kind里各出现一次)
#   too_short_caption_count    : 2条(strip后长度小于10)
#   too_long_caption_count     : 132条(strip后长度大于1024)
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 1,
    'no_word_char_caption_count': 2,
    'too_short_caption_count': 2,
    'too_long_caption_count': 132,
    'invalid_placeholder_caption_count': 0,
}

# 9个子集过滤后的实测合格样本数，合计11753827条，解析阶段逐项硬对账。
# 子集级对账能额外拦住"某个分片被漏读"或"重复key集合算错"这种
# 总数对账看不出来的问题
EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT = {
    'subset_1': 1006487,
    'subset_2': 1007269,
    'subset_3': 963995,
    'subset_4': 637806,
    'subset_5': 1337603,
    'subset_6': 1560589,
    'subset_7': 2135656,
    'subset_8': 855623,
    'subset_9': 2248799,
}

EXPECTED_VALID_IMAGE_ANNOTATION_COUNT = 11753827

EXPECTED_SAVE_SET_COUNT = 9

PROCESS_NUM = 32

PER_FOLDER_IMAGE_NUM = 10000

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 实测两条整图caption的长度分布: reference侧0~2484、edited侧5~3791，
# p50都在330字符上下、p99约620。
# 下限沿用003(GPIC)的10(实测只砍掉2条)，
# 上限也沿用003的1024(实测只砍掉132条，占0.001%)，
# 与已产出的t2i数据集口径保持一致
MIN_CAPTION_LENGTH = 10

MAX_CAPTION_LENGTH = 1024

# 判定"描述里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 只剩标点的描述没有任何可训练的语义，直接丢弃(实测2条)
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 带编号的视觉参考图占位符。t2i样本没有任何视觉条件图，
# 所以描述里不允许出现任何占位符(实测0条命中)，这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

MAX_SAVE_MESSAGE_NUM = 10000


def get_set_name(per_subset_name):
    """把官方分片目录名归一化成子集名(subset-1 -> subset_1)

    t2i没有"编辑任务类型"这个概念，这里用官方自带的9个下载分片当子集，
    只把中划线换成下划线以满足保存图像名的字符白名单。
    """
    return str(per_subset_name).strip().lower().replace('-', '_')


def check_invalid_caption(per_t2i_caption):
    """判定描述里是不是残留了视觉参考图占位符，返回True表示这条描述不合格

    t2i样本只有文本条件、没有任何视觉条件图，所以描述里出现[Vn*]就说明
    上游文本被污染了(实测0条命中)，这里只做防御性拦截。
    """
    per_t2i_caption = str(per_t2i_caption).strip()

    return len(CAPTION_VISUAL_PLACEHOLDER_PATTERN.findall(per_t2i_caption)) > 0


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上，逐样本打一次os.path.exists就是一次网络往返，
    1175万张图就要打1175万次。
    实测同一个标注文件里的图像全部落在同一个batch目录下，所以这里按目录缓存一次
    os.listdir的结果，之后只做集合查表，网络往返次数从"图像张数"降到"batch目录数"
    (2120次)。listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判，
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


def process_single_image_check(per_image_path):
    """校验单张图像能否正常解码，并过滤非RGB图和极端分辨率图

    返回的宽高只用于统计，最终写进json的宽高一定取自实际写盘图像的shape。
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

    # 检查图像短边(实测恒为512x512，一条都不会命中，只作兜底)
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


def get_all_annotation_file_pair(root_dataset_path):
    """收集上游全部标注文件，返回[标注文件任务列表, 每个<subset>/<kind>的标注文件数]

    上游标注按子集分目录、子集内按kind分目录、kind内按tar分文件
    (实测9个子集 * 2个kind = 18组，合计2120个jsonl)，
    这里按"标注文件"这一粒度出任务，正好能把多进程铺满。
    子集名与kind名由主进程按目录名算好后带给worker，worker会再和行内字段交叉核对。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    subset_kind_annotation_file_count_dict = {}
    for per_subset_name in LOAD_SUBSET_DIR_NAME_LIST:
        for per_frame_kind_name in LOAD_FRAME_KIND_DIR_NAME_LIST:
            per_subset_kind_key = f'{per_subset_name}/{per_frame_kind_name}'
            per_subset_kind_path = os.path.join(root_annotation_path,
                                                per_subset_name,
                                                per_frame_kind_name)
            if not os.path.isdir(per_subset_kind_path):
                print('2222', per_subset_kind_path)
                subset_kind_annotation_file_count_dict[per_subset_kind_key] = 0
                continue

            per_annotation_file_name_list = sorted([
                per_annotation_file_name for per_annotation_file_name in
                os.listdir(per_subset_kind_path)
                if per_annotation_file_name.endswith(
                    LOAD_ANNOTATION_FILE_NAME_SUFFIX)
            ])

            for per_annotation_file_name in per_annotation_file_name_list:
                per_archive_name = per_annotation_file_name[:-len(
                    LOAD_ANNOTATION_FILE_NAME_SUFFIX)]
                annotation_file_pair_list.append([
                    os.path.join(per_subset_kind_path,
                                 per_annotation_file_name),
                    per_subset_name,
                    per_frame_kind_name,
                    per_archive_name,
                ])

            subset_kind_annotation_file_count_dict[per_subset_kind_key] = len(
                per_annotation_file_name_list)

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list, subset_kind_annotation_file_count_dict


def process_single_annotation_file_sample_key(annotation_file_pair):
    """第一遍扫描: 只把单个标注文件里的全部sample_key读出来

    必须先全量扫一遍才能知道哪些sample_key跨batch/subset重复:
    实测848个key各出现2次，它们的图像像素与文本都不同(是上游对同一视频片段
    重复采样出的两个样本)，按方案确认涉及重复key的样本全部丢弃。
    只返回key字符串(约5878678个)，不返回整行，内存与IPC开销都可控。
    """
    per_annotation_path, _, _, _ = annotation_file_pair

    sample_key_list = []
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_jsonl_file:
            for per_line in load_jsonl_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                try:
                    per_annotation = json.loads(per_line)
                except Exception:
                    continue

                if not isinstance(per_annotation, dict):
                    continue

                per_sample_key = per_annotation.get(
                    ANNOTATION_SAMPLE_KEY_KEY_NAME, '')
                if not isinstance(per_sample_key, str):
                    continue

                per_sample_key = per_sample_key.strip()
                if not per_sample_key:
                    continue

                sample_key_list.append(per_sample_key)
    except Exception as e:
        print('2222', per_annotation_path, e)

    return sample_key_list


def get_duplicate_sample_key_dict(annotation_file_pair_list):
    """全量统计sample_key出现次数，返回出现超过1次的key集合

    实测2120个jsonl共5878678行、其中848个key各出现2次。
    这个集合只有848个元素，可以直接随partial带给第二遍解析的worker。
    """
    sample_key_count_dict = {}
    total_sample_key_count = 0
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_sample_key_list in tqdm(pool.imap_unordered(
                process_single_annotation_file_sample_key,
                annotation_file_pair_list),
                                        total=len(annotation_file_pair_list)):
            total_sample_key_count += len(per_sample_key_list)
            for per_sample_key in per_sample_key_list:
                sample_key_count_dict[
                    per_sample_key] = sample_key_count_dict.get(
                        per_sample_key, 0) + 1

    duplicate_sample_key_dict = {
        per_sample_key: per_sample_key_count
        for per_sample_key, per_sample_key_count in
        sample_key_count_dict.items() if per_sample_key_count > 1
    }

    duplicate_sample_key_annotation_count = sum(
        duplicate_sample_key_dict.values())

    return [
        duplicate_sample_key_dict,
        total_sample_key_count,
        len(sample_key_count_dict),
        duplicate_sample_key_annotation_count,
    ]


def get_single_annotation_image_relative_path(per_annotation,
                                              per_image_key_name):
    """取出单条标注里某一张图的相对路径

    参考图字段是长度恒为1的list、编辑后图字段是字符串，这里统一成字符串返回，
    取不到时返回空串。
    """
    per_image_relative_path = per_annotation.get(per_image_key_name, '')

    if isinstance(per_image_relative_path, (list, tuple)):
        per_image_relative_path = per_image_relative_path[0] if len(
            per_image_relative_path) > 0 else ''

    if not isinstance(per_image_relative_path, str):
        return ''

    return per_image_relative_path.replace('\\', '/').strip().lstrip('/')


def process_single_annotation_file(annotation_file_pair,
                                   duplicate_sample_key_dict,
                                   root_dataset_path):
    """解析单个标注文件，把每个编辑对拆成2条t2i样本

    这个worker把文本层过滤(json坏行、缺字段、行内字段与目录不自洽、
    sample_key重复、原图名前缀非法、图像路径与sample_key不对应、缺图、
    描述为空、描述只剩标点、描述过短、描述过长、描述里残留占位符、保存名非法)
    和图像层过滤(能否解码、是否RGB、短边、宽高比)一次做完。
    图像校验没有像002/006那样单独再开一个Pool，是因为本数据集有约1175万条候选:
    分两个Pool的话主进程要先攒1175万条记录、再逐条发给check worker、再收回，
    光进程间序列化就要来回搬几十GB，而合并进来之后IPC只传存活样本。
    判定逻辑、过滤口径、日志编号与003完全一致，图像也一样是解码两遍
    (这里校验一遍、写盘时重编码再解一遍)，没有为了省时间跳过任何一道校验。

    图像是否存在按目录缓存一次os.listdir的结果、之后只做集合查表:
    每个标注文件里的图像全部落在同一个batch目录下，一个worker只需listdir一次。
    """
    per_annotation_path, per_subset_name, per_frame_kind_name, per_archive_name = annotation_file_pair

    per_subset_kind_key = f'{per_subset_name}/{per_frame_kind_name}'
    per_set_name = get_set_name(per_subset_name)

    total_annotation_count, load_annotation_failed_count = 0, 0
    field_not_match_count, duplicate_sample_key_count = 0, 0
    duplicate_sample_key_image_count = 0
    total_image_annotation_count = 0
    missing_image_count, image_name_not_match_count = 0, 0
    invalid_save_image_name_count = 0
    empty_caption_count, no_word_char_caption_count = 0, 0
    too_short_caption_count, too_long_caption_count = 0, 0
    invalid_placeholder_caption_count = 0
    invalid_image_count = 0
    set_annotation_count_dict = {}
    image_annotation_pair_list = []

    # 每个worker只处理一个标注文件，缓存里通常只有一个batch目录，内存开销可忽略
    dir_file_name_cache_dict = {}

    try:
        load_jsonl_file = open(per_annotation_path, 'r', encoding='UTF-8')
    except Exception as e:
        print('2222', per_annotation_path, e)

        return [
            image_annotation_pair_list,
            per_subset_kind_key,
            total_annotation_count,
            1,
            field_not_match_count,
            duplicate_sample_key_count,
            duplicate_sample_key_image_count,
            total_image_annotation_count,
            missing_image_count,
            image_name_not_match_count,
            invalid_save_image_name_count,
            empty_caption_count,
            no_word_char_caption_count,
            too_short_caption_count,
            too_long_caption_count,
            invalid_placeholder_caption_count,
            invalid_image_count,
            set_annotation_count_dict,
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
                print('2222', per_annotation_path, e)
                continue

            if not isinstance(per_annotation, dict):
                load_annotation_failed_count += 1
                print('2222', per_annotation_path, 'annotation not a dict')
                continue

            per_sample_key = per_annotation.get(ANNOTATION_SAMPLE_KEY_KEY_NAME,
                                                '')
            if not isinstance(per_sample_key, str):
                per_sample_key = ''
            per_sample_key = per_sample_key.strip()

            if not per_sample_key:
                # 这一行的2条候选样本都定位不到图，必须显式记进候选总数，
                # 否则"候选条数 + 重复key丢掉的条数 == 行数 * 2"这条对账会串位
                total_image_annotation_count += EXPECTED_PER_ANNOTATION_IMAGE_NUM
                image_name_not_match_count += EXPECTED_PER_ANNOTATION_IMAGE_NUM
                print('2222', per_annotation_path, 'empty sample key')
                continue

            # 行内记录的子集名/kind名/tar名必须和这个标注文件所在的目录一致:
            # 不一致说明上游产物被搬动过，继续跑会把样本写进错误子集(实测0例)
            if per_annotation.get(
                    ANNOTATION_SUBSET_NAME_KEY_NAME,
                    '') != per_subset_name or per_annotation.get(
                        ANNOTATION_FRAME_KIND_NAME_KEY_NAME,
                        '') != per_frame_kind_name or per_annotation.get(
                            ANNOTATION_ARCHIVE_NAME_KEY_NAME,
                            '') != per_archive_name:
                # 同上: 整行丢弃时也要把这2条候选记进候选总数，保证对账不串位
                total_image_annotation_count += EXPECTED_PER_ANNOTATION_IMAGE_NUM
                field_not_match_count += 1
                print('2222', per_annotation_path, per_sample_key)
                continue

            # 跨batch/subset重复的sample_key整行丢弃(实测848组、1696行、3392条候选):
            # 它们的像素与文本都不同，是两个不同的有效样本，
            # 但按方案确认这类样本全部不保留
            if per_sample_key in duplicate_sample_key_dict:
                duplicate_sample_key_count += 1
                duplicate_sample_key_image_count += EXPECTED_PER_ANNOTATION_IMAGE_NUM
                continue

            # 一行拆2条t2i样本: 参考图配reference_image_caption、
            # 编辑后图配edited_image_caption
            for per_image_key_name, per_caption_key_name, per_load_image_name_suffix in ANNOTATION_IMAGE_KEY_NAME_PAIR_LIST:
                total_image_annotation_count += 1

                per_image_relative_path = get_single_annotation_image_relative_path(
                    per_annotation, per_image_key_name)
                if not per_image_relative_path:
                    missing_image_count += 1
                    continue

                # 图像文件名必须与sample_key严格对应，否则说明上游成员错位，
                # 这种样本会让保存名指向另一张图(实测全量0例不符)
                per_image_name = os.path.basename(per_image_relative_path)
                if per_image_name != f'{per_sample_key}{per_load_image_name_suffix}':
                    image_name_not_match_count += 1
                    print('2222', per_annotation_path, per_image_relative_path)
                    continue

                per_image_path = os.path.join(root_dataset_path,
                                              *LOAD_IMAGE_DIR_NAME_LIST,
                                              per_image_relative_path)
                if not check_image_file_exists(per_image_path,
                                               dir_file_name_cache_dict):
                    missing_image_count += 1
                    continue

                # 原始图像名前缀就是<sample_key>_reference / <sample_key>_edited,
                # 全小写后必须只含小写字母/数字/下划线/中划线(实测0例不符)
                per_image_name_prefix = os.path.splitext(
                    per_image_name)[0].lower()
                if not VALID_IMAGE_NAME_PREFIX_PATTERN.match(
                        per_image_name_prefix):
                    invalid_save_image_name_count += 1
                    print('2222', per_image_path, per_image_name_prefix)
                    continue

                # 保存图像名 = 数据集名 + 子集名 + kind名 + tar名 + 原图名前缀。
                # 原图名前缀只在同一个batch内唯一(跨batch/subset有848个重复,
                # 那848组已在上面整行丢弃)，所以必须拼上kind名与tar名才能全局唯一。
                # 实测1175.4万个保存名100%唯一、最长104字符
                per_save_image_name = (
                    f'{DATASET_NAME}_{per_set_name}_{per_frame_kind_name}_'
                    f'{per_archive_name}_{per_image_name_prefix}'
                    f'{SAVE_IMAGE_NAME_SUFFIX}')
                if not VALID_IMAGE_NAME_PATTERN.match(per_save_image_name):
                    invalid_save_image_name_count += 1
                    print('2222', per_image_path, per_save_image_name)
                    continue

                per_t2i_caption = per_annotation.get(per_caption_key_name, '')
                # 上游描述固定是str，这里兼容list和str两种形式
                if isinstance(per_t2i_caption, (list, tuple)):
                    per_t2i_caption = per_t2i_caption[0] if len(
                        per_t2i_caption) > 0 else ''
                if not isinstance(per_t2i_caption, str):
                    per_t2i_caption = ''
                per_t2i_caption = per_t2i_caption.strip()

                # 空描述、全空格描述视为不合格样本对(实测1条)
                if not per_t2i_caption:
                    empty_caption_count += 1
                    continue

                # 只剩标点、没有任何数字/字母/汉字的描述也丢弃(实测2条)
                if not CAPTION_WORD_CHAR_PATTERN.search(per_t2i_caption):
                    no_word_char_caption_count += 1
                    print('3333', per_image_path, per_t2i_caption[:50])
                    continue

                # 过短描述视为不合格样本对(实测2条)
                if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                    too_short_caption_count += 1
                    print('3333', per_image_path, len(per_t2i_caption))
                    continue

                # 过长描述同样视为不合格样本对(实测132条)
                if len(per_t2i_caption) > MAX_CAPTION_LENGTH:
                    too_long_caption_count += 1
                    print('3333', per_image_path, len(per_t2i_caption))
                    continue

                # t2i样本没有任何视觉条件图，描述里不允许残留[Vn*]占位符(实测0条)
                if check_invalid_caption(per_t2i_caption):
                    invalid_placeholder_caption_count += 1
                    print('3333', per_image_path, per_t2i_caption[:100])
                    continue

                if process_single_image_check(per_image_path) is None:
                    invalid_image_count += 1
                    continue

                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + 1

                image_annotation_pair_list.append([
                    per_set_name,
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

    return [
        image_annotation_pair_list,
        per_subset_kind_key,
        total_annotation_count,
        load_annotation_failed_count,
        field_not_match_count,
        duplicate_sample_key_count,
        duplicate_sample_key_image_count,
        total_image_annotation_count,
        missing_image_count,
        image_name_not_match_count,
        invalid_save_image_name_count,
        empty_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_image_count,
        set_annotation_count_dict,
    ]


def get_all_image_annotation_pair(root_dataset_path):
    """两遍扫描上游2120个jsonl，多进程组装全部t2i样本(图像路径 + 单图描述)的列表

    第一遍只读sample_key做全量重复统计(必须先知道848组重复key才能整组丢弃),
    第二遍才真正解析每一行并把每个编辑对拆成2条t2i样本。
    最后按[子集名, 保存图像名]统一排序，保证输出顺序可复现。
    """
    annotation_file_pair_list, subset_kind_annotation_file_count_dict = get_all_annotation_file_pair(
        root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list),
          'subset kind:', len(subset_kind_annotation_file_count_dict))

    duplicate_sample_key_dict, first_pass_annotation_count, unique_sample_key_count, duplicate_sample_key_annotation_count = get_duplicate_sample_key_dict(
        annotation_file_pair_list)

    print('1111', 'first pass annotation:', first_pass_annotation_count,
          'unique sample key:',
          unique_sample_key_count, 'duplicate sample key:',
          len(duplicate_sample_key_dict), 'duplicate sample key annotation:',
          duplicate_sample_key_annotation_count)

    total_annotation_count, load_annotation_failed_count = 0, 0
    field_not_match_count, duplicate_sample_key_count = 0, 0
    duplicate_sample_key_image_count = 0
    total_image_annotation_count = 0
    missing_image_count, image_name_not_match_count = 0, 0
    invalid_save_image_name_count = 0
    empty_caption_count, no_word_char_caption_count = 0, 0
    too_short_caption_count, too_long_caption_count = 0, 0
    invalid_placeholder_caption_count = 0
    invalid_image_count = 0
    subset_kind_annotation_count_dict = {}
    set_annotation_count_dict = {}
    image_annotation_pair_list = []

    process_func = partial(process_single_annotation_file,
                           duplicate_sample_key_dict=duplicate_sample_key_dict,
                           root_dataset_path=root_dataset_path)
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_func, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            image_annotation_pair_list.extend(per_load_result[0])

            per_subset_kind_key = per_load_result[1]
            subset_kind_annotation_count_dict[
                per_subset_kind_key] = subset_kind_annotation_count_dict.get(
                    per_subset_kind_key, 0) + per_load_result[2]

            total_annotation_count += per_load_result[2]
            load_annotation_failed_count += per_load_result[3]
            field_not_match_count += per_load_result[4]
            duplicate_sample_key_count += per_load_result[5]
            duplicate_sample_key_image_count += per_load_result[6]
            total_image_annotation_count += per_load_result[7]
            missing_image_count += per_load_result[8]
            image_name_not_match_count += per_load_result[9]
            invalid_save_image_name_count += per_load_result[10]
            empty_caption_count += per_load_result[11]
            no_word_char_caption_count += per_load_result[12]
            too_short_caption_count += per_load_result[13]
            too_long_caption_count += per_load_result[14]
            invalid_placeholder_caption_count += per_load_result[15]
            invalid_image_count += per_load_result[16]

            for per_set_name, per_set_count in per_load_result[17].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[0], x[2]])

    return [
        image_annotation_pair_list,
        len(annotation_file_pair_list),
        subset_kind_annotation_file_count_dict,
        subset_kind_annotation_count_dict,
        set_annotation_count_dict,
        len(duplicate_sample_key_dict),
        duplicate_sample_key_annotation_count,
        first_pass_annotation_count,
        total_annotation_count,
        load_annotation_failed_count,
        field_not_match_count,
        duplicate_sample_key_count,
        duplicate_sample_key_image_count,
        total_image_annotation_count,
        missing_image_count,
        image_name_not_match_count,
        invalid_save_image_name_count,
        empty_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_image_count,
    ]


def check_load_annotation_count(
        total_annotation_file_count, subset_kind_annotation_file_count_dict,
        subset_kind_annotation_count_dict, set_annotation_count_dict,
        duplicate_sample_key_count, duplicate_sample_key_annotation_count,
        skip_duplicate_sample_key_annotation_count,
        duplicate_sample_key_image_count, first_pass_annotation_count,
        total_annotation_count, total_image_annotation_count,
        valid_image_annotation_count, invalid_caption_count_dict,
        image_annotation_pair_list):
    """解析完标注后硬对账: 分片数、原始条数、候选条数、重复key、描述过滤条数、子集条数

    上游015的解压产物是一次性解出来的确定结果，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    <subset>/<kind>与子集两级对账还能额外拦住"某个分片被漏读"或"重复key集合算错"
    这种总数对账看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名会让后写的图像覆盖先写的、在json里互相顶掉key,
    事后从产物里根本看不出少了多少条。
    """
    check_error_message_list = []

    # 分片数对账: 18组逐一比对，subset-4/sampled_multi_frames本来就只有23个
    for per_subset_kind_key in sorted(
            EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT.keys()):
        per_expect_annotation_file_num = EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT[
            per_subset_kind_key]
        per_annotation_file_num = subset_kind_annotation_file_count_dict.get(
            per_subset_kind_key, 0)
        if per_annotation_file_num != per_expect_annotation_file_num:
            check_error_message_list.append(
                f'{per_subset_kind_key} annotation file num not match '
                f'{per_annotation_file_num} != {per_expect_annotation_file_num}'
            )

    for per_subset_kind_key in sorted(
            subset_kind_annotation_file_count_dict.keys()):
        if per_subset_kind_key not in EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT:
            check_error_message_list.append(
                f'unknown subset kind {per_subset_kind_key}')

    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

    # 原始条数对账: 18组逐一比对
    for per_subset_kind_key in sorted(
            EXPECTED_SUBSET_KIND_ANNOTATION_COUNT_DICT.keys()):
        per_expect_annotation_count = EXPECTED_SUBSET_KIND_ANNOTATION_COUNT_DICT[
            per_subset_kind_key]
        per_annotation_count = subset_kind_annotation_count_dict.get(
            per_subset_kind_key, 0)
        if per_annotation_count != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_subset_kind_key} annotation count not match '
                f'{per_annotation_count} != {per_expect_annotation_count}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    # 两遍扫描读到的行数必须完全一致，不一致说明中途有分片读失败
    if first_pass_annotation_count != total_annotation_count:
        check_error_message_list.append(
            f'first pass annotation count not match '
            f'{first_pass_annotation_count} != {total_annotation_count}')

    # 每行必须恰好拆出2条候选t2i样本(被整行丢弃的重复key那部分单独计数),
    # 所以"候选条数 + 重复key丢掉的条数"必须等于"行数 * 2"
    if total_image_annotation_count + duplicate_sample_key_image_count != total_annotation_count * EXPECTED_PER_ANNOTATION_IMAGE_NUM:
        check_error_message_list.append(
            f'total image annotation count not match '
            f'{total_image_annotation_count} + {duplicate_sample_key_image_count} != '
            f'{total_annotation_count} * {EXPECTED_PER_ANNOTATION_IMAGE_NUM}')

    if total_image_annotation_count + duplicate_sample_key_image_count != EXPECTED_TOTAL_IMAGE_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total image annotation count not match expect '
            f'{total_image_annotation_count} + {duplicate_sample_key_image_count} != '
            f'{EXPECTED_TOTAL_IMAGE_ANNOTATION_COUNT}')

    # 重复sample_key对账: 组数、丢弃行数、丢弃候选条数都必须与实测值一致
    if duplicate_sample_key_count != EXPECTED_DUPLICATE_SAMPLE_KEY_COUNT:
        check_error_message_list.append(
            f'duplicate sample key count not match '
            f'{duplicate_sample_key_count} != '
            f'{EXPECTED_DUPLICATE_SAMPLE_KEY_COUNT}')

    if duplicate_sample_key_annotation_count != EXPECTED_DUPLICATE_SAMPLE_KEY_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'duplicate sample key annotation count not match '
            f'{duplicate_sample_key_annotation_count} != '
            f'{EXPECTED_DUPLICATE_SAMPLE_KEY_ANNOTATION_COUNT}')

    if duplicate_sample_key_image_count != EXPECTED_DUPLICATE_SAMPLE_KEY_IMAGE_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'duplicate sample key image count not match '
            f'{duplicate_sample_key_image_count} != '
            f'{EXPECTED_DUPLICATE_SAMPLE_KEY_IMAGE_ANNOTATION_COUNT}')

    # 第二遍解析真正丢掉的行数必须等于第一遍统计出来的重复行数
    if skip_duplicate_sample_key_annotation_count != duplicate_sample_key_annotation_count:
        check_error_message_list.append(
            f'skip duplicate sample key annotation count not self consistent '
            f'{skip_duplicate_sample_key_annotation_count} != '
            f'{duplicate_sample_key_annotation_count}')

    # 文本层各项过滤条数逐项对账(判定顺序与worker里完全一致)
    for per_count_name in sorted(EXPECTED_INVALID_CAPTION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_CAPTION_COUNT_DICT[per_count_name]
        if invalid_caption_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_caption_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    # 子集级对账: 9个子集逐一比对 + 总数 + 白名单严格一一对应
    for per_set_name in sorted(EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT.keys()):
        per_expect_set_annotation_count = EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT[
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

    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in SAVE_SET_NAME_LIST:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

    if valid_image_annotation_count != EXPECTED_VALID_IMAGE_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'valid image annotation count not match '
            f'{valid_image_annotation_count} != '
            f'{EXPECTED_VALID_IMAGE_ANNOTATION_COUNT}')

    if valid_image_annotation_count != sum(set_annotation_count_dict.values()):
        check_error_message_list.append(
            f'valid image annotation count not self consistent '
            f'{valid_image_annotation_count} != '
            f'{sum(set_annotation_count_dict.values())}')

    # 保存图像名必须全局唯一，撞名会让后写的图像覆盖先写的、静默丢样本
    save_image_name_dict = {}
    duplicate_save_image_name_list = []
    for per_image_annotation_pair in image_annotation_pair_list:
        per_save_image_name = per_image_annotation_pair[2]
        if per_save_image_name in save_image_name_dict:
            duplicate_save_image_name_list.append(per_save_image_name)
            continue
        save_image_name_dict[per_save_image_name] = 1

    if len(duplicate_save_image_name_list) > 0:
        check_error_message_list.append(
            f'duplicate save image name num '
            f'{len(duplicate_save_image_name_list)} '
            f'{duplicate_save_image_name_list[:5]}')

    return check_error_message_list


def get_all_image_save_folder_pair(image_annotation_pair_list,
                                   save_dataset_path):
    """把过滤后的合格样本按子集分组，排序后每10000条切成一个文件夹

    切分必须在过滤全部完成之后做，且切分前先按保存图像名排序，这样才能保证每个
    文件夹都是满10000条(只有每个子集最后一个文件夹允许不满)。
    这里不像003(GPIC)那样再套一层PER_SET_FOLDER_NUM: 本数据集最大的子集
    (subset_9约225万条)也只有225个文件夹，单目录里225个子目录 + 225个json
    对NAS完全没有压力。
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

            # 一个文件夹就是一个写盘任务，worker写完这10000张后直接写出该文件夹的json,
            # 主进程只收计数，不用把1175万条记录再攒一遍
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
    编码参数用cv2.imencode('.jpg', img)的默认值(质量95 + 色度4:2:0)，
    与001~006这几个已产出的数据集口径完全一致。
    以文件夹为任务粒度而不是以单张图为粒度: 本数据集约1175万条样本，逐图收结果的话
    主进程要再攒一份1175万条的列表，而且中途挂了只能从头再来; 按文件夹收之后
    主进程内存只和文件夹数(1179个)相关，且json已经写全的文件夹可以直接跳过、
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

    每个子集只有最后一个文件夹允许不满10000条，其余都必须是满10000条。
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

    # 除每个子集最后一个文件夹外都必须是满10000条
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

        # 每条标注的字段集合必须和约定的四个key严格一致，不能多也不能少
        if sorted(per_annotation.keys()) != sorted(
                SAVE_ANNOTATION_KEY_NAME_LIST):
            check_error_message_list.append(
                f'{per_save_image_name} annotation key not match')
            continue

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
        # 图像宽高必须是正数
        if per_annotation['width'] <= 0 or per_annotation['height'] <= 0:
            check_error_message_list.append(
                f'{per_save_image_name} image shape not match {per_annotation["width"]} {per_annotation["height"]}'
            )
        # 短边和宽高比必须仍然满足过滤阈值
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
        # 落盘后的描述里不允许再残留只剩标点、没有任何数字/字母/汉字的描述
        if not CAPTION_WORD_CHAR_PATTERN.search(per_annotation['t2i_caption']):
            check_error_message_list.append(
                f'{per_save_image_name} still a no word char caption')
        # 也不允许残留视觉参考图占位符
        if check_invalid_caption(per_annotation['t2i_caption']):
            check_error_message_list.append(
                f'{per_save_image_name} still a placeholder caption')
        # json里存的就是过滤时判定的那个字符串，两者口径一致，
        # 这里直接量json里的长度复检
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

    每个子集只有最后一个文件夹允许不满10000条。
    1179个文件夹每个都要listdir一万个文件再load一份json，串行跑在NAS上太久，
    所以这一步也按文件夹粒度开多进程(与003一致)。
    """
    check_error_message_list = []

    folder_check_pair_list = []
    for per_set_name in sorted(set_folder_count_dict.keys()):
        per_set_folder_count = set_folder_count_dict[per_set_name]
        for per_folder_index in range(per_set_folder_count):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            folder_check_pair_list.append([
                per_set_name,
                per_folder_name,
                per_folder_index == per_set_folder_count - 1,
            ])

    total_image_count = 0
    check_func = partial(check_single_save_folder,
                         save_dataset_path=save_dataset_path)
    with Pool(processes=min(PROCESS_NUM, max(len(folder_check_pair_list),
                                             1))) as pool:
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

    image_annotation_pair_list, total_annotation_file_count, subset_kind_annotation_file_count_dict, subset_kind_annotation_count_dict, set_annotation_count_dict, duplicate_sample_key_count, duplicate_sample_key_annotation_count, first_pass_annotation_count, total_annotation_count, load_annotation_failed_count, field_not_match_count, skip_duplicate_sample_key_annotation_count, duplicate_sample_key_image_count, total_image_annotation_count, missing_image_count, image_name_not_match_count, invalid_save_image_name_count, empty_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, invalid_placeholder_caption_count, invalid_image_count = get_all_image_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          load_annotation_failed_count, field_not_match_count,
          duplicate_sample_key_count, duplicate_sample_key_annotation_count,
          skip_duplicate_sample_key_annotation_count,
          duplicate_sample_key_image_count, total_image_annotation_count,
          missing_image_count, image_name_not_match_count,
          invalid_save_image_name_count, empty_caption_count,
          no_word_char_caption_count, too_short_caption_count,
          too_long_caption_count, invalid_placeholder_caption_count,
          invalid_image_count, len(set_annotation_count_dict),
          len(image_annotation_pair_list))

    if len(image_annotation_pair_list) > 0:
        print('1111', image_annotation_pair_list[0])

    invalid_caption_count_dict = {
        'empty_caption_count': empty_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        total_annotation_file_count, subset_kind_annotation_file_count_dict,
        subset_kind_annotation_count_dict, set_annotation_count_dict,
        duplicate_sample_key_count, duplicate_sample_key_annotation_count,
        skip_duplicate_sample_key_annotation_count,
        duplicate_sample_key_image_count, first_pass_annotation_count,
        total_annotation_count, total_image_annotation_count,
        len(image_annotation_pair_list), invalid_caption_count_dict,
        image_annotation_pair_list)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        raise Exception(
            f'load annotation check failed {load_annotation_check_error_message_list[:20]}'
        )

    if load_annotation_failed_count > 0:
        raise Exception(
            f'load annotation failed count {load_annotation_failed_count}')

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

    folder_image_count_dict = {}
    save_image_failed_count = 0
    process_func = partial(process_single_image_folder,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_save_result in tqdm(pool.imap_unordered(
                process_func, image_save_folder_pair_list),
                                    total=len(image_save_folder_pair_list)):
            _, per_folder_name, per_folder_image_count, per_save_image_failed_count = per_save_result
            folder_image_count_dict[per_folder_name] = per_folder_image_count
            save_image_failed_count += per_save_image_failed_count

            print('2222', per_folder_name, per_folder_image_count,
                  per_save_image_failed_count)

    total_save_image_count = sum(folder_image_count_dict.values())

    check_error_message_list, check_total_image_count = check_save_dataset(
        save_dataset_path, set_folder_count_dict)

    print('3333', 'total annotation file:', total_annotation_file_count,
          'total annotation:', total_annotation_count,
          'load annotation failed:', load_annotation_failed_count,
          'field not match:', field_not_match_count, 'duplicate sample key:',
          duplicate_sample_key_count, 'duplicate sample key image:',
          duplicate_sample_key_image_count, 'total image annotation:',
          total_image_annotation_count, 'missing image:', missing_image_count,
          'image name not match:', image_name_not_match_count,
          'invalid save image name:', invalid_save_image_name_count,
          'empty caption:', empty_caption_count, 'no word char caption:',
          no_word_char_caption_count, 'too short caption:',
          too_short_caption_count, 'too long caption:', too_long_caption_count,
          'invalid placeholder caption:', invalid_placeholder_caption_count,
          'invalid image:', invalid_image_count, 'save image failed:',
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
        total_annotation_file_count,
        'total_annotation_count':
        total_annotation_count,
        'first_pass_annotation_count':
        first_pass_annotation_count,
        'load_annotation_failed_count':
        load_annotation_failed_count,
        'field_not_match_count':
        field_not_match_count,
        'duplicate_sample_key_count':
        duplicate_sample_key_count,
        'duplicate_sample_key_annotation_count':
        duplicate_sample_key_annotation_count,
        'skip_duplicate_sample_key_annotation_count':
        skip_duplicate_sample_key_annotation_count,
        'duplicate_sample_key_image_count':
        duplicate_sample_key_image_count,
        'total_image_annotation_count':
        total_image_annotation_count,
        'missing_image_count':
        missing_image_count,
        'image_name_not_match_count':
        image_name_not_match_count,
        'invalid_save_image_name_count':
        invalid_save_image_name_count,
        'empty_caption_count':
        empty_caption_count,
        'no_word_char_caption_count':
        no_word_char_caption_count,
        'too_short_caption_count':
        too_short_caption_count,
        'too_long_caption_count':
        too_long_caption_count,
        'invalid_placeholder_caption_count':
        invalid_placeholder_caption_count,
        'invalid_image_count':
        invalid_image_count,
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
        'min_caption_length':
        MIN_CAPTION_LENGTH,
        'max_caption_length':
        MAX_CAPTION_LENGTH,
        'subset_kind_annotation_file_count_dict':
        subset_kind_annotation_file_count_dict,
        'subset_kind_annotation_count_dict':
        subset_kind_annotation_count_dict,
        'set_annotation_count_dict':
        set_annotation_count_dict,
        'set_folder_count_dict':
        set_folder_count_dict,
        'folder_image_count_dict':
        folder_image_count_dict,
        'check_error_message_list':
        check_error_message_list[:MAX_SAVE_MESSAGE_NUM],
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
    if len(check_error_message_list) > 0:
        # 收尾自校验不通过必须让上层感知，不能静默留下坏样本或不满的文件夹
        raise Exception(
            f'check save dataset error num {len(check_error_message_list)} {check_error_message_list[:10]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/BM-6M'
    save_dataset_path = r'/root/autodl-tmp/t2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
