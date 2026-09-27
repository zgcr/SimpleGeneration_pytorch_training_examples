import os
import re
import io
import csv
import json
import shutil
import pickle
import collections

import pyarrow.parquet as pq

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: MegaStyle(tencent/MegaStyle-1.4M)
#
# 【数据集类型】纯文生图(text-to-image)数据集，不是图像编辑数据集。
# 每个样本对只有"一张生成后图 + 内容提示content + 风格提示style"，
# 没有任何参考图/输入图/编辑指令/mask，README的task_categories也写明是text-to-image，
# 所以下游只能用t2i_dataset.py那条链路，不能当ti2i(编辑)数据用。
#
# 【root_dataset_path实测原始保存规格】
# MegaStyle-1.4M/
# ├── train-{00000-00099}.parquet  100片(2.4T)，编号连续无缺号，
# │                                每片恒为80000行 / 800个row group(每组100行)，
# │                                实测总行数 8,000,000                       -> 有用(唯一的样本数据)
# ├── metadata.csv                 173MB，1000000行(+1行表头)，CRLF换行，9列风格属性 -> 有用
# ├── style_indices.pkl            279MB，dict[风格提示] -> [8个全局行号]，
# │                                1000000个key                              -> 有用
# ├── LICENSE.txt                  腾讯授权条款(无用)
# ├── README.md                    数据集说明(无用)
# ├── .gitattributes               git lfs配置(无用)
# └── .cache/                      huggingface下载缓存，216个文件，
#                                  里面还残留8个*.incomplete(无用)
#
# 【必须显式感知的规格坑1: 目录名是1.4M，实际下载的是8M那一版】
# 目录名叫MegaStyle-1.4M，但100片parquet的footer行数合计是**8,000,000**行，
# = 1,000,000个风格 x 每个风格8张图，与README里v2.0(MegaStyle++-8M)的
# "1M细粒度风格提示 / 8M高质量风格化图像"完全吻合。
# 所以本脚本所有EXPECTED_*常量按实测的8,000,000写，绝不能按目录名的1.4M写，
# 否则预检会把完整数据集判成异常、或者把缺片判成正常。
#
# 【单行parquet的全部4列(100片schema完全一致)】
#   id      : 形如s{风格号}_c{内容号}(如s1_c26226)，实测片内80000/80000唯一、
#             跨片风格号不重叠，所以全局唯一                     -> 有用(样本唯一id/图像名)
#   image   : struct<bytes: binary, path: string>，唯一的生成后图 -> 有用
#   content : 内容提示(画什么)，25~69字符，实测无空值、结尾无标点 -> 有用(文本提示)
#   style   : 风格提示(怎么画)，183~310字符，实测无空值、结尾100%是'.'，
#             且**每连续8行完全相同**(抽样block-of-8违例0次)      -> 有用(文本提示)
#
# 【必须显式感知的规格坑2: image.path写的是.jpg，图像字节却是PNG】
# 实测抽样1716张: image.path恒为'<id>.jpg'，但图像字节100%是PNG魔数，
# 解出来全部是512x512 / RGB / format=PNG。
# 所以推断后缀必须**魔数优先、path后缀兜底**(和009脚本的path优先相反)，
# 否则会把PNG数据写成.jpg文件名，下游按后缀分流时会踩坑。
#
# 【必须显式感知的规格坑3: 风格属性靠"全局行号"隐式挂载，没有显式外键】
# metadata.csv / style_indices.pkl 都不带id列，只能靠位置对齐:
#   第fi片第r行的全局行号 global_row_index = fi * 80000 + r
#   风格序号 style_index = global_row_index // 8
#   style_indices.pkl 的第style_index个key == 该行的style列原文
#   metadata.csv 的第style_index行 == 该风格的9个结构化属性
#   id里的s{N} 满足 N == style_index + 1
# 实测: pkl的1000000个key，第i个key的value严格等于[i*8, i*8+7](0处不连续);
#       抽样1140条上述四方关系100%成立;
#       抽样1005条metadata.csv的overall artistic style都能在对应风格提示里找到。
# 这套隐式映射是把csv/pkl属性挂回样本的唯一依据，所以脚本里**逐行硬校验**，
# 不能默认成立。
#
# 【caption口径: content与style必须拼接，只用其中一个都会显著变差】
# 该数据集是用Qwen-Image拿"内容提示 x 风格提示"合成出来的，也就是说
# **拼接后的完整提示 = 当初生成这张图时真正喂给模型的prompt**，
# 是唯一能完整且无歧义地决定这张图的文本。
# - 只用content: 同一条content会在约8个不同风格下复用(8M图/约1M内容提示)，
#   同一caption对应8张画风截然不同的图，是直接互相矛盾的监督信号，
#   flow matching只能学到"各风格的平均"，风格糊、细节被平均掉;
#   而且这个数据集唯一的价值就是1M个细粒度风格，丢掉style等于把核心信息扔了。
# - 只用style: 风格提示完全不描述画面内容，同一条style对应的8张图是8个
#   毫不相干的场景，同样是矛盾监督，模型学不到图文内容对齐;
#   且1M条caption全部以'In the style of'开头，会严重污染文本分布。
# 所以落盘的 t2i_caption = f'{content}. {style}'
# (content实测结尾无标点，不以.!?结尾时补句点，再空格拼上style)。
# 同时**content / style / 9个结构化风格属性原文都单独保留**，
# 下游还能免费拿到三个能力: 风格dropout做风格CFG、按style_index成组采样做
# 风格一致性正则、按结构化属性做条件或数据均衡采样。
#
# 【无用信息(不整理进训练目录)】
#   .cache/(216个文件，含8个*.incomplete) / .gitattributes / README.md / LICENSE.txt;
#   image.path(内容就是'<id>.jpg'，和id重复且后缀还是错的)，只在魔数认不出时兜底用。
#
# 【本脚本的处理口径】
# - 解包前预检(硬失败): 根目录条目白名单、parquet数量==100与编号连号、
#   每片PAR1头尾魔数(O(1)读，拦下载截断)、只读footer校验每片80000行/总计8000000行/
#   4列schema唯一，任一不过直接抛异常，不白跑几十小时;
# - metadata.csv与style_indices.pkl先做完整性校验(1000000行/1000000个key/
#   每个key的value严格等于[i*8, i*8+7])，再按每片10000个风格切成
#   unzip_style_attributes/<分片名>.jsonl，
#   这样子进程只加载自己那2.5MB的切片，不用把1M条属性复制32份进内存;
#   原始csv/pkl另外原样拷到 unzip_source_annotations/ 作为溯源真值;
# - 解包时流式iter_batches，图像落盘 images/<分片名>/<桶号>/<id>.png
#   (每片80000张按10000一桶分成8个子目录，避免单目录塞8万个文件)，
#   写盘后立刻校验落盘大小 == len(bytes)(避免只看存在性把半截图当正常样本);
# - 同时每行写一条 unzip_annotations/<分片名>.jsonl，含落盘路径、样本id、
#   风格序号/内容序号、全局行号、t2i_caption、content、style原文、
#   9个结构化风格属性、图像宽高与格式，下游直接按行取样本;
# - 片内四重硬对账: 遍历行数 == footer的num_rows、
#   extract+skip+not_save == 行数、有效样本对 + 隔离样本对 == 行数(且实测应全部有效)、
#   片内覆盖的风格数 == 10000且每个风格恰好8个样本;
# - 全量硬对账: 总行数/总有效样本对 == 8000000、覆盖风格数 == 1000000且全部满8张;
# - id片内重名时改写到独立目录保留并上报，不静默覆盖丢样本;
# - 任何一环出错都汇总后抛异常，不再静默跑过。
# ==============================================================================

# 带分片编号的parquet名(train-00000.parquet)，用于分片完整性预检
PARQUET_SHARD_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>train-(?P<index>\d{5}))\.parquet$')

PARQUET_FILE_NAME_PATTERN = re.compile(r'^(?P<prefix>.+)\.parquet$')

# 样本id规格: s{风格号}_c{内容号}，风格号是1起的style_index + 1
SAMPLE_ID_NAME_PATTERN = re.compile(
    r'^s(?P<style_index>\d+)_c(?P<content_index>\d+)$')

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(216个文件，含8个*.incomplete)
# .gitattributes   git lfs配置
# README.md        数据集说明
# LICENSE.txt      腾讯授权条款
# .gitignore/CACHEDIR.TAG/.DS_Store  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    'LICENSE.txt',
    'CACHEDIR.TAG',
    '.DS_Store',
]

# 根目录下除parquet外仅有的两个有用文件(风格属性表与风格->全局行号索引)
LOAD_STYLE_METADATA_FILE_NAME = 'metadata.csv'

LOAD_STYLE_INDICES_FILE_NAME = 'style_indices.pkl'

LOAD_SOURCE_ANNOTATION_FILE_NAME_LIST = [
    LOAD_STYLE_METADATA_FILE_NAME,
    LOAD_STYLE_INDICES_FILE_NAME,
]

# 实测parquet分片数与每片行数，数量不对说明下载不全
EXPECTED_PARQUET_NUM = 100

EXPECTED_PARQUET_ROW_COUNT = 80000

EXPECTED_TOTAL_ROW_COUNT = 8000000

# 实测每个风格恰好8张图(style_indices.pkl的每个value都是8个连续全局行号)
EXPECTED_STYLE_SAMPLE_NUM = 8

EXPECTED_STYLE_NUM = 1000000

# 每片80000行 / 每风格8张 = 每片恰好覆盖10000个风格，且分片间风格不重叠
EXPECTED_PARQUET_STYLE_NUM = EXPECTED_PARQUET_ROW_COUNT // EXPECTED_STYLE_SAMPLE_NUM

# 实测4列schema，100片完全一致
EXPECTED_PARQUET_COLUMN_NAME_LIST = [
    'id',
    'image',
    'content',
    'style',
]

PARQUET_SAMPLE_ID_COLUMN_NAME = 'id'

PARQUET_IMAGE_COLUMN_NAME = 'image'

PARQUET_CONTENT_COLUMN_NAME = 'content'

PARQUET_STYLE_COLUMN_NAME = 'style'

PARQUET_IMAGE_BYTES_KEY_NAME = 'bytes'

PARQUET_IMAGE_PATH_KEY_NAME = 'path'

# metadata.csv的9列风格属性(实测表头完全一致，缺失值写'N/A')。
# 列名带空格，落盘时统一归一化成下划线形式再写进标注
STYLE_METADATA_COLUMN_NAME_LIST = [
    'overall artistic style',
    'dominant colors',
    'supporting colors',
    'light',
    'visual pattern',
    'surface status',
    'medium',
    'brushwork',
    'edge rendering',
]

# metadata.csv里表示"该属性缺失"的占位值，只统计不当异常
STYLE_METADATA_EMPTY_VALUE = 'N/A'

# metadata.csv第i行的overall artistic style 应该能在第i个风格提示里找到，
# 这是csv与pkl位置对齐的软证据(实测抽样1005条0错位)。
# 超过这个比例判为对齐失败，否则只打印告警
STYLE_METADATA_ALIGN_MISMATCH_RATIO_THRESHOLD = 0.01

# content实测结尾无标点，拼接前不以这些字符结尾就补一个句点
CAPTION_SENTENCE_END_CHAR_LIST = [
    '.',
    '!',
    '?',
]

SAVE_IMAGE_DIR_NAME = 'images'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_STYLE_ATTRIBUTE_DIR_NAME = 'unzip_style_attributes'

SAVE_SOURCE_ANNOTATION_DIR_NAME = 'unzip_source_annotations'

SAVE_DUPLICATE_SAMPLE_DIR_NAME = 'unzip_duplicate_samples'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

# 每片80000张图按10000一桶分成8个子目录，避免单目录塞8万个文件拖垮NAS元数据
SAVE_IMAGE_SUB_DIR_SIZE = 10000

IMAGE_BYTES_MAGIC_SUFFIX_LIST = [
    [b'\x89PNG\r\n\x1a\n', '.png'],
    [b'\xff\xd8\xff', '.jpg'],
    [b'GIF87a', '.gif'],
    [b'GIF89a', '.gif'],
    [b'BM', '.bmp'],
    [b'RIFF', '.webp'],
    [b'II*\x00', '.tif'],
    [b'MM\x00*', '.tif'],
]

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.bmp',
    '.gif',
    '.tif',
]

PARQUET_FILE_MAGIC_BYTES = b'PAR1'

# 图像成员是否落盘。
# True : 和其他数据集脚本口径一致，800万张512x512 PNG约2.4T + 100个jsonl，
#        务必确认目标盘的容量和inode扛得住再跑;
# False: 只解析非图像列生成 unzip_annotations/*.jsonl 索引(几小时即可跑完)，
#        图像继续留在原parquet里，训练时按parquet顺序读，样本对信息一样完整。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解包后再os.walk一遍输出目录做二次对账。
# 默认False: 800万个小文件的os.walk在NAS上要跑很久，而解包时已经做了
# "写盘后立刻校验落盘大小 == len(bytes)" + "extract+skip+not_save == 行数"两道对账，
# 已经能保证每行的图都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 是否解码图像header拿真实宽高写进标注。
# 默认True: 实测全部是512x512，PIL的Image.open只读header不解像素，代价可忽略，
# 但下游做分辨率分桶时就不用再打开800万张图。
PARSE_IMAGE_SHAPE_FLAG = True

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

PARQUET_ROW_BATCH_SIZE = 64

# metadata.csv单个字段最长实测几十字符，这里放宽上限只是防御异常长字段直接抛错
CSV_FIELD_SIZE_LIMIT = 1024 * 1024 * 1024


def check_skip_file_or_dir(per_file_relative_path):
    """过滤掉.cache、.gitattributes、README.md、LICENSE.txt这些不需要整理的文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """只把图像后缀的文件计入解出图像总数和二次对账统计"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


def get_image_bytes_suffix(per_image_bytes, per_image_name):
    """魔数优先推断图像后缀，魔数认不出时才退回用parquet里记录的文件名后缀

    该数据集image.path恒为'<id>.jpg'，但图像字节实测100%是PNG，
    所以这里和其他脚本相反，必须魔数优先，否则会把PNG写成.jpg。
    """
    for per_magic_bytes, per_magic_suffix in IMAGE_BYTES_MAGIC_SUFFIX_LIST:
        if per_image_bytes.startswith(per_magic_bytes):
            return per_magic_suffix

    if per_image_name:
        per_image_name_suffix = os.path.splitext(per_image_name)[1].lower()
        if per_image_name_suffix in IMAGE_FILE_SUFFIX_LIST:
            return per_image_name_suffix

    return '.png'


def get_stripped_text_value(per_column_value):
    """文本列统一转成strip后的字符串，None/空白都当成缺失"""
    if not isinstance(per_column_value, str):
        return ''

    return per_column_value.strip()


def get_normalized_style_attribute_key_name(per_column_name):
    """把metadata.csv带空格的列名归一化成下划线形式，方便写进json后直接取用"""
    return per_column_name.strip().lower().replace(' ', '_')


def get_parquet_group_name_by_index(per_parquet_index):
    """按分片编号还原parquet分片名(train-00000)，用于风格属性切片文件对齐"""
    return f'train-{per_parquet_index:05d}'


def get_t2i_caption(per_content, per_style):
    """把内容提示与风格提示拼成训练用的完整文生图提示

    该数据集的图是用"内容提示 x 风格提示"合成出来的，拼接后的提示才是当初
    真正喂给生成模型的prompt，也是唯一能无歧义决定这张图的文本:
    只用content时同一条caption对应8种画风的图、只用style时对应8个不相干场景，
    两种都是互相矛盾的监督信号。
    content实测结尾无标点，所以不以.!?结尾时先补一个句点再拼。
    """
    per_content = per_content.strip()
    per_style = per_style.strip()

    if not per_content:
        return per_style
    if not per_style:
        return per_content

    if per_content[-1] not in CAPTION_SENTENCE_END_CHAR_LIST:
        per_content = f'{per_content}.'

    return f'{per_content} {per_style}'


def get_image_shape(per_image_bytes):
    """只读图像header拿宽高，解码失败时返回[0, 0]并带上错误信息"""
    if not PARSE_IMAGE_SHAPE_FLAG:
        return [0, 0], ''

    try:
        with Image.open(io.BytesIO(per_image_bytes)) as load_image:
            per_image_width, per_image_height = load_image.size
    except Exception as e:
        return [0, 0], f'parse image shape failed {e}'

    return [per_image_width, per_image_height], ''


def check_single_parquet_file_magic(per_parquet_path):
    """O(1)预检单个parquet是否被截断: 文件头尾都必须是PAR1魔数

    实测100片全部满足，说明当前数据集是完整的。但.cache里残留了8个*.incomplete，
    说明下载确实中断过，如果哪次下载不全，流式解包只会在读到一半时抛异常，
    必须在跑几十小时前先拦住。
    """
    error_message_list = []
    try:
        per_parquet_size = os.path.getsize(per_parquet_path)
        if per_parquet_size <= 2 * len(PARQUET_FILE_MAGIC_BYTES):
            error_message_list.append(
                f'parquet size too small {per_parquet_path} {per_parquet_size}'
            )

            return error_message_list

        with open(per_parquet_path, 'rb') as load_parquet_file:
            per_parquet_head_bytes = load_parquet_file.read(
                len(PARQUET_FILE_MAGIC_BYTES))
            load_parquet_file.seek(per_parquet_size -
                                   len(PARQUET_FILE_MAGIC_BYTES))
            per_parquet_tail_bytes = load_parquet_file.read(
                len(PARQUET_FILE_MAGIC_BYTES))

        if per_parquet_head_bytes != PARQUET_FILE_MAGIC_BYTES:
            error_message_list.append(
                f'parquet head magic broken {per_parquet_path}')
        if per_parquet_tail_bytes != PARQUET_FILE_MAGIC_BYTES:
            error_message_list.append(
                f'parquet tail magic broken(truncated file) {per_parquet_path}'
            )
    except Exception as e:
        error_message_list.append(
            f'read parquet magic failed {per_parquet_path} {e}')

    return error_message_list


def check_single_parquet_file_row_count(parquet_group):
    """只读单个parquet的footer拿行数和列名，不碰任何图像字节"""
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path, per_parquet_index = parquet_group

    try:
        load_parquet_file = pq.ParquetFile(per_parquet_path)
        per_row_count = load_parquet_file.metadata.num_rows
        per_column_name_list = list(load_parquet_file.schema_arrow.names)
    except Exception as e:
        print('7777', per_parquet_group_name, e)

        return [
            per_parquet_group_name,
            0,
            [],
            [f'read parquet metadata failed {per_parquet_group_name} {e}'],
        ]

    return [
        per_parquet_group_name,
        per_row_count,
        per_column_name_list,
        [],
    ]


def check_parquet_shard_complete(parquet_group_list):
    """解包前预检: 分片数必须是100、编号必须是00000..00099连号

    100片少下几片时脚本照样能跑完并退出0，会静默少掉几十万个样本对，必须先拦住。
    """
    error_message_list = []

    shard_index_dict = {}
    for per_parquet_group_name, _, per_parquet_path, per_parquet_index in parquet_group_list:
        per_parquet_name = os.path.basename(per_parquet_path)

        per_match_result = PARQUET_SHARD_FILE_NAME_PATTERN.match(
            per_parquet_name)
        if not per_match_result:
            error_message_list.append(
                f'unknown parquet name {per_parquet_name}')
            continue

        if per_parquet_index in shard_index_dict:
            error_message_list.append(
                f'duplicate parquet shard index {per_parquet_index}')
            continue

        shard_index_dict[per_parquet_index] = per_parquet_group_name

    print('1111', 'parquet:', len(shard_index_dict), 'expected parquet:',
          EXPECTED_PARQUET_NUM)

    if len(shard_index_dict) != EXPECTED_PARQUET_NUM:
        error_message_list.append(
            f'parquet num not match {len(shard_index_dict)} != {EXPECTED_PARQUET_NUM}'
        )

    # 分片编号必须是0..99连号，缺号说明有分片没下载下来
    per_missing_shard_index_list = sorted(
        set(range(0, EXPECTED_PARQUET_NUM)) - set(shard_index_dict.keys()))
    if len(per_missing_shard_index_list) > 0:
        error_message_list.append(
            f'parquet shard index not continuous, missing index {per_missing_shard_index_list[:20]} total missing {len(per_missing_shard_index_list)}'
        )

    return error_message_list


def check_parquet_row_count_complete(parquet_group_list):
    """只读footer按每片行数与实测总行数逐项硬对账，同时校验4列schema是否一致"""
    error_message_list = []

    total_row_count = 0
    per_parquet_row_count_dict = {}
    column_name_key_dict = {}

    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_file_row_count, parquet_group_list),
                                     total=len(parquet_group_list)):
            per_parquet_group_name, per_row_count, per_column_name_list, per_error_message_list = per_check_result

            error_message_list.extend(per_error_message_list)
            if len(per_error_message_list) > 0:
                continue

            per_parquet_row_count_dict[per_parquet_group_name] = per_row_count
            total_row_count += per_row_count

            column_name_key_dict.setdefault(','.join(per_column_name_list),
                                            []).append(per_parquet_group_name)

            # 每片行数必须恰好是80000: 风格属性是靠"全局行号 = 片号*80000 + 行号"
            # 隐式挂载的，任何一片行数不对，后面所有片的风格映射都会整体错位
            if per_row_count != EXPECTED_PARQUET_ROW_COUNT:
                error_message_list.append(
                    f'{per_parquet_group_name} row count not match {per_row_count} != {EXPECTED_PARQUET_ROW_COUNT}'
                )

    # 100片的schema实测完全一致(4列)，出现第二种列组合说明数据规格变了
    if len(column_name_key_dict) != 1:
        for per_column_name_key in sorted(column_name_key_dict.keys()):
            error_message_list.append(
                f'parquet column name not unique, parquet num {len(column_name_key_dict[per_column_name_key])} example {column_name_key_dict[per_column_name_key][0]} column {per_column_name_key}'
            )

    for per_column_name_key in sorted(column_name_key_dict.keys()):
        if per_column_name_key.split(',') != EXPECTED_PARQUET_COLUMN_NAME_LIST:
            error_message_list.append(
                f'parquet column name not match {per_column_name_key} != {",".join(EXPECTED_PARQUET_COLUMN_NAME_LIST)}'
            )

    print('1111', 'total row:', total_row_count,
          'expected total row:', EXPECTED_TOTAL_ROW_COUNT, 'parquet:',
          len(per_parquet_row_count_dict))

    if total_row_count != EXPECTED_TOTAL_ROW_COUNT:
        error_message_list.append(
            f'total row count not match {total_row_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )

    return total_row_count, per_parquet_row_count_dict, error_message_list


def check_required_dataset_complete(root_dataset_path, parquet_group_list,
                                    file_copy_pair_list):
    """解包前预检: 根目录条目白名单、分片连号、PAR1魔数、footer行数与schema

    数据集本身不完整就没必要跑几十小时解包，也避免"少了几片但整体报成功"。
    """
    error_message_list = []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')

        return 0, {}, error_message_list

    for per_name in sorted(os.listdir(root_dataset_path)):
        if check_skip_file_or_dir(per_name):
            continue

        # 该数据集根目录下除.cache外没有任何子目录，出现新目录必须显式上报
        if os.path.isdir(os.path.join(root_dataset_path, per_name)):
            error_message_list.append(f'unknown dir in root dir {per_name}')
            continue

        if PARQUET_SHARD_FILE_NAME_PATTERN.match(per_name):
            continue
        if per_name in LOAD_SOURCE_ANNOTATION_FILE_NAME_LIST:
            continue

        # 根目录下出现新的非跳过文件必须显式上报，否则会被静默漏处理
        error_message_list.append(f'unknown file in root dir {per_name}')

    # metadata.csv与style_indices.pkl是把风格属性挂回样本的唯一依据，缺一不可
    all_copy_file_name_dict = {
        os.path.basename(per_file_relative_path): 1
        for per_file_relative_path, _ in file_copy_pair_list
    }
    for per_file_name in LOAD_SOURCE_ANNOTATION_FILE_NAME_LIST:
        if not os.path.exists(os.path.join(root_dataset_path, per_file_name)):
            error_message_list.append(f'source file not exist {per_file_name}')
            continue

        if per_file_name not in all_copy_file_name_dict:
            error_message_list.append(
                f'source file not in copy list {per_file_name}')

    error_message_list.extend(check_parquet_shard_complete(parquet_group_list))

    parquet_path_check_list = [
        per_parquet_path for _, _, per_parquet_path, _ in parquet_group_list
    ]
    print('1111', 'check parquet file magic:', len(parquet_path_check_list))
    with Pool(processes=PROCESS_NUM) as pool:
        for per_magic_error_message_list in tqdm(
                pool.imap_unordered(check_single_parquet_file_magic,
                                    parquet_path_check_list),
                total=len(parquet_path_check_list)):
            error_message_list.extend(per_magic_error_message_list)

    print('1111', 'check parquet row count:', len(parquet_group_list))
    total_row_count, per_parquet_row_count_dict, row_count_error_message_list = check_parquet_row_count_complete(
        parquet_group_list)
    error_message_list.extend(row_count_error_message_list)

    return total_row_count, per_parquet_row_count_dict, error_message_list


def load_style_prompt_list(root_dataset_path):
    """加载style_indices.pkl并校验它就是"风格序号 -> 8个连续全局行号"的位置索引

    实测1000000个key，第i个key的value严格等于[i*8, i*8+7](0处不连续)，
    也就是"第i个风格占用全局行号 i*8 .. i*8+7"，
    这正是把风格属性挂回每一行的依据，必须逐条校验后才敢用。
    """
    error_message_list = []
    style_prompt_list = []

    load_style_indices_path = os.path.join(root_dataset_path,
                                           LOAD_STYLE_INDICES_FILE_NAME)
    try:
        with open(load_style_indices_path, 'rb') as load_pickle_file:
            style_indices_dict = pickle.load(load_pickle_file)
    except Exception as e:
        print('7777', load_style_indices_path, e)
        error_message_list.append(f'load style indices failed {e}')

        return style_prompt_list, error_message_list

    if not isinstance(style_indices_dict, dict):
        error_message_list.append('style indices not a dict')

        return style_prompt_list, error_message_list

    if len(style_indices_dict) != EXPECTED_STYLE_NUM:
        error_message_list.append(
            f'style indices num not match {len(style_indices_dict)} != {EXPECTED_STYLE_NUM}'
        )

    empty_style_prompt_count, not_continuous_index_count = 0, 0
    for per_style_index, (per_style_prompt,
                          per_global_row_index_list) in enumerate(
                              style_indices_dict.items()):
        if not isinstance(per_style_prompt,
                          str) or not per_style_prompt.strip():
            empty_style_prompt_count += 1
            if len(error_message_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                error_message_list.append(
                    f'style {per_style_index} empty style prompt')

        per_expected_global_row_index_list = list(
            range(
                per_style_index * EXPECTED_STYLE_SAMPLE_NUM,
                per_style_index * EXPECTED_STYLE_SAMPLE_NUM +
                EXPECTED_STYLE_SAMPLE_NUM))
        if list(per_global_row_index_list
                ) != per_expected_global_row_index_list:
            not_continuous_index_count += 1
            if len(error_message_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                error_message_list.append(
                    f'style {per_style_index} global row index not continuous {list(per_global_row_index_list)[:4]}'
                )

        style_prompt_list.append(
            per_style_prompt if isinstance(per_style_prompt, str) else '')

    # 提前释放，1M个key加1M个8元素list占内存不小，后面只需要key本身
    del style_indices_dict

    print('1111', 'style prompt:', len(style_prompt_list),
          'empty style prompt:', empty_style_prompt_count,
          'not continuous index:', not_continuous_index_count)

    if len(style_prompt_list
           ) * EXPECTED_STYLE_SAMPLE_NUM != EXPECTED_TOTAL_ROW_COUNT:
        error_message_list.append(
            f'style prompt num not match total row count {len(style_prompt_list)} * {EXPECTED_STYLE_SAMPLE_NUM} != {EXPECTED_TOTAL_ROW_COUNT}'
        )

    # 逐条错误按MAX_SAVE_PROBLEM_ITEM_NUM截断了，这里再补一条总数，
    # 避免异常条数超过上限时上层只看到被截断的样本
    if empty_style_prompt_count > 0:
        error_message_list.append(
            f'empty style prompt count {empty_style_prompt_count}')
    if not_continuous_index_count > 0:
        error_message_list.append(
            f'style global row index not continuous count {not_continuous_index_count}'
        )

    return style_prompt_list, error_message_list


def save_style_attribute_file(root_dataset_path,
                              save_style_attribute_dir_path):
    """把metadata.csv与style_indices.pkl合成按分片切好的风格属性表

    metadata.csv与style_indices.pkl都不带id列，只能靠**位置**对齐:
    csv第i行的9个属性 == pkl第i个key(风格提示) == 全局行号[i*8, i*8+7]那8行的风格。
    这里逐行校验csv的overall artistic style能否在对应风格提示里找到，
    作为两份文件位置对齐的证据(实测抽样1005条0错位)。

    切片落成 unzip_style_attributes/<分片名>.jsonl，每片10000个风格，
    这样32个子进程各自只加载自己那2.5MB的切片，
    不用把1M条风格属性复制32份进内存。
    """
    error_message_list = []

    style_prompt_list, per_error_message_list = load_style_prompt_list(
        root_dataset_path)
    error_message_list.extend(per_error_message_list)
    if len(style_prompt_list) == 0:
        return 0, {}, error_message_list

    os.makedirs(save_style_attribute_dir_path, exist_ok=True)

    load_style_metadata_path = os.path.join(root_dataset_path,
                                            LOAD_STYLE_METADATA_FILE_NAME)

    csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)

    style_metadata_row_count = 0
    style_attribute_align_mismatch_count = 0
    style_attribute_empty_value_count_dict = collections.Counter()
    parquet_style_count_dict = collections.Counter()

    save_style_attribute_file_handle = None
    current_parquet_index = -1
    try:
        with open(load_style_metadata_path, 'r', newline='',
                  encoding='UTF-8') as load_csv_file:
            load_csv_reader = csv.DictReader(load_csv_file)

            if list(load_csv_reader.fieldnames
                    or []) != STYLE_METADATA_COLUMN_NAME_LIST:
                error_message_list.append(
                    f'style metadata column name not match {load_csv_reader.fieldnames}'
                )

            for per_style_index, per_csv_row_dict in enumerate(
                    tqdm(load_csv_reader)):
                style_metadata_row_count += 1

                # csv行数比风格数多说明两份文件对不上，多出来的行没有风格提示可挂
                if per_style_index >= len(style_prompt_list):
                    continue

                per_style_prompt = style_prompt_list[per_style_index]
                per_parquet_index = per_style_index // EXPECTED_PARQUET_STYLE_NUM
                per_parquet_group_name = get_parquet_group_name_by_index(
                    per_parquet_index)

                if per_parquet_index != current_parquet_index:
                    # csv是按风格序号顺序排列的，切片时顺序换文件句柄即可，不用开100个句柄
                    if save_style_attribute_file_handle is not None:
                        save_style_attribute_file_handle.close()

                    save_style_attribute_file_handle = open(os.path.join(
                        save_style_attribute_dir_path,
                        f'{per_parquet_group_name}.jsonl'),
                                                            'w',
                                                            encoding='UTF-8')
                    current_parquet_index = per_parquet_index

                per_style_attribute_dict = {}
                for per_column_name in STYLE_METADATA_COLUMN_NAME_LIST:
                    per_column_value = get_stripped_text_value(
                        per_csv_row_dict.get(per_column_name, ''))
                    if per_column_value == STYLE_METADATA_EMPTY_VALUE or not per_column_value:
                        style_attribute_empty_value_count_dict[
                            per_column_name] += 1

                    per_style_attribute_dict[
                        get_normalized_style_attribute_key_name(
                            per_column_name)] = per_column_value

                # overall artistic style应该原样出现在风格提示里，
                # 这是csv与pkl位置对齐的软证据，不一致只统计不逐条报错
                per_overall_artistic_style = get_stripped_text_value(
                    per_csv_row_dict.get(STYLE_METADATA_COLUMN_NAME_LIST[0],
                                         '')).lower()
                if per_overall_artistic_style and per_overall_artistic_style not in per_style_prompt.lower(
                ):
                    style_attribute_align_mismatch_count += 1

                per_save_style_attribute = {
                    'style_index': per_style_index,
                    'style_prompt': per_style_prompt,
                    'parquet_name': per_parquet_group_name,
                    'global_row_index_start':
                    per_style_index * EXPECTED_STYLE_SAMPLE_NUM,
                    'style_sample_num': EXPECTED_STYLE_SAMPLE_NUM,
                    'style_attribute_dict': per_style_attribute_dict,
                }
                save_style_attribute_file_handle.write(
                    f'{json.dumps(per_save_style_attribute, ensure_ascii=False)}\n'
                )
                parquet_style_count_dict[per_parquet_group_name] += 1
    except Exception as e:
        print('7777', load_style_metadata_path, e)
        error_message_list.append(f'load style metadata failed {e}')
    finally:
        if save_style_attribute_file_handle is not None:
            save_style_attribute_file_handle.close()

    print('3333', 'style metadata row:', style_metadata_row_count,
          'style prompt:', len(style_prompt_list), 'style attribute file:',
          len(parquet_style_count_dict), 'align mismatch:',
          style_attribute_align_mismatch_count)
    print('3333', 'style attribute empty value:',
          dict(style_attribute_empty_value_count_dict))

    if style_metadata_row_count != EXPECTED_STYLE_NUM:
        error_message_list.append(
            f'style metadata row count not match {style_metadata_row_count} != {EXPECTED_STYLE_NUM}'
        )
    if style_metadata_row_count != len(style_prompt_list):
        error_message_list.append(
            f'style metadata row count not match style prompt num {style_metadata_row_count} != {len(style_prompt_list)}'
        )
    if len(parquet_style_count_dict) != EXPECTED_PARQUET_NUM:
        error_message_list.append(
            f'style attribute file num not match {len(parquet_style_count_dict)} != {EXPECTED_PARQUET_NUM}'
        )

    for per_parquet_group_name in sorted(parquet_style_count_dict.keys()):
        if parquet_style_count_dict[
                per_parquet_group_name] != EXPECTED_PARQUET_STYLE_NUM:
            error_message_list.append(
                f'{per_parquet_group_name} style attribute num not match {parquet_style_count_dict[per_parquet_group_name]} != {EXPECTED_PARQUET_STYLE_NUM}'
            )

    # 位置对齐是把9个风格属性挂到样本上的唯一依据，大面积对不上必须硬失败
    if style_metadata_row_count > 0:
        per_align_mismatch_ratio = style_attribute_align_mismatch_count / style_metadata_row_count
        if per_align_mismatch_ratio > STYLE_METADATA_ALIGN_MISMATCH_RATIO_THRESHOLD:
            error_message_list.append(
                f'style metadata align mismatch ratio too large {per_align_mismatch_ratio} > {STYLE_METADATA_ALIGN_MISMATCH_RATIO_THRESHOLD}'
            )
        elif style_attribute_align_mismatch_count > 0:
            print('2222', 'style metadata align mismatch count',
                  style_attribute_align_mismatch_count)

    return style_metadata_row_count, dict(
        style_attribute_empty_value_count_dict), error_message_list


def load_single_style_attribute_file(save_style_attribute_dir_path,
                                     per_parquet_group_name,
                                     per_style_index_start):
    """子进程只加载自己那一片的10000个风格属性(约2.5MB)，并校验风格序号连号"""
    error_message_list = []
    style_attribute_list = []

    load_style_attribute_path = os.path.join(
        save_style_attribute_dir_path, f'{per_parquet_group_name}.jsonl')
    try:
        with open(load_style_attribute_path, 'r',
                  encoding='UTF-8') as load_json_file:
            for per_line in load_json_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                style_attribute_list.append(json.loads(per_line))
    except Exception as e:
        print('7777', load_style_attribute_path, e)
        error_message_list.append(f'load style attribute failed {e}')

        return style_attribute_list, error_message_list

    if len(style_attribute_list) != EXPECTED_PARQUET_STYLE_NUM:
        error_message_list.append(
            f'style attribute num not match {len(style_attribute_list)} != {EXPECTED_PARQUET_STYLE_NUM}'
        )

    for per_style_offset, per_style_attribute in enumerate(
            style_attribute_list):
        if per_style_attribute.get(
                'style_index', -1) != per_style_index_start + per_style_offset:
            error_message_list.append(
                f'style attribute style index not match {per_style_attribute.get("style_index", -1)} != {per_style_index_start + per_style_offset}'
            )
            break

    return style_attribute_list, error_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把metadata.csv与style_indices.pkl原样拷到溯源目录，保持文件名不变

    这两个文件是风格属性的原始真值，虽然已经被切成
    unzip_style_attributes/*.jsonl，但原件必须留一份可回溯。
    """
    per_file_relative_path, per_file_path = file_copy_pair

    save_file_path = os.path.join(save_dataset_path,
                                  SAVE_SOURCE_ANNOTATION_DIR_NAME,
                                  per_file_relative_path)
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


def save_single_image_bytes(save_image_path, per_image_bytes):
    """单张图独立落盘并立刻校验落盘大小，返回[是否新写, 是否跳过, 错误信息]

    单张图写盘异常不能让整片parquet的循环中断，否则该片后面几万行既不解图也不进标注。
    先判存在且大小一致就跳过，保证脚本可以断点续跑。
    """
    if os.path.exists(save_image_path) and os.path.getsize(
            save_image_path) == len(per_image_bytes):
        return [False, True, '']

    try:
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        with open(save_image_path, 'wb') as save_image_file:
            save_image_file.write(per_image_bytes)
    except Exception as e:
        print('6666', save_image_path, e)

        return [False, False, f'write image failed {save_image_path} {e}']

    if not os.path.exists(save_image_path) or os.path.getsize(
            save_image_path) != len(per_image_bytes):
        print('6666', save_image_path, 'save image size not match')

        return [False, False, f'save image size not match {save_image_path}']

    return [True, False, '']


def get_single_sample_id_error_message_list(per_sample_id, per_style_index):
    """校验样本id规格并取出风格号与内容号

    id形如s{风格号}_c{内容号}，实测风格号恒等于style_index + 1(1起)，
    这是"id / style列 / csv / pkl"四方位置对齐里的一环，对不上必须显式感知。
    """
    warning_message_list = []

    per_match_result = SAMPLE_ID_NAME_PATTERN.match(per_sample_id)
    if not per_match_result:
        warning_message_list.append(f'{per_sample_id} unknown sample id name')

        return -1, -1, warning_message_list

    per_id_style_index = int(per_match_result.group('style_index'))
    per_content_index = int(per_match_result.group('content_index'))

    if per_id_style_index != per_style_index + 1:
        warning_message_list.append(
            f'{per_sample_id} style index not match {per_id_style_index} != {per_style_index + 1}'
        )

    return per_id_style_index, per_content_index, warning_message_list


def process_single_parquet_file(parquet_group, save_dataset_path,
                                save_annotation_dir_path,
                                save_style_attribute_dir_path):
    """流式解开单个parquet，把内嵌图像字节写成图像文件、其余列写成jsonl汇总标注

    落盘结构:
      images/<分片名>/<桶号>/<id>.png
      unzip_annotations/<分片名>.jsonl
    每行jsonl就是一个完整有用信息的文生图样本对，
    保留 t2i_caption(content与style拼接) + content/style原文 + 9个结构化风格属性。

    parquet按iter_batches流式读，绝不整片进内存(单片约24G图像字节)。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path, per_parquet_index = parquet_group

    per_style_index_start = per_parquet_index * EXPECTED_PARQUET_STYLE_NUM
    per_global_row_index_start = per_parquet_index * EXPECTED_PARQUET_ROW_COUNT

    save_image_dir_path = os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                                       per_parquet_group_name)
    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_SAMPLE_DIR_NAME,
                                           per_parquet_group_name)
    save_annotation_path = os.path.join(save_annotation_dir_path,
                                        f'{per_parquet_group_name}.jsonl')

    if EXTRACT_IMAGE_FILE_FLAG:
        os.makedirs(save_image_dir_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)

    row_count, valid_sample_pair_count = 0, 0
    extract_image_count, skip_image_count, not_save_image_count = 0, 0, 0
    duplicate_sample_count, style_prompt_mismatch_count = 0, 0
    invalid_sample_pair_count, warning_message_count = 0, 0
    image_shape_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    style_sample_count_dict = collections.Counter()
    invalid_sample_pair_list, warning_message_list = [], []
    error_message_list = []
    reach_parquet_end = False

    sample_id_dict = {}

    style_attribute_list, per_error_message_list = load_single_style_attribute_file(
        save_style_attribute_dir_path, per_parquet_group_name,
        per_style_index_start)
    error_message_list.extend([
        f'{per_parquet_group_name} {per_error_message}'
        for per_error_message in per_error_message_list
    ])

    try:
        load_parquet_file = pq.ParquetFile(per_parquet_path)
        expected_row_count = load_parquet_file.metadata.num_rows

        with open(save_annotation_path, 'w',
                  encoding='UTF-8') as save_annotation_file:
            for per_record_batch in load_parquet_file.iter_batches(
                    batch_size=PARQUET_ROW_BATCH_SIZE):
                for per_row_dict in per_record_batch.to_pylist():
                    per_row_index = row_count
                    row_count += 1

                    # 风格属性没有外键，只能靠全局行号隐式挂载:
                    # 全局行号 = 片号*80000 + 片内行号，风格序号 = 全局行号//8
                    per_global_row_index = per_global_row_index_start + per_row_index
                    per_style_index = per_global_row_index // EXPECTED_STYLE_SAMPLE_NUM
                    per_style_offset = per_style_index - per_style_index_start

                    per_sample_id = get_stripped_text_value(
                        per_row_dict.get(PARQUET_SAMPLE_ID_COLUMN_NAME, None))
                    per_content = get_stripped_text_value(
                        per_row_dict.get(PARQUET_CONTENT_COLUMN_NAME, None))
                    per_style = get_stripped_text_value(
                        per_row_dict.get(PARQUET_STYLE_COLUMN_NAME, None))

                    _, per_content_index, per_sample_id_warning_message_list = get_single_sample_id_error_message_list(
                        per_sample_id, per_style_index)
                    warning_message_count += len(
                        per_sample_id_warning_message_list)
                    if len(warning_message_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                        # 每片8万行，告警全留会撑爆内存也撑爆子进程回传的pickle，
                        # 所以只留前若干条样本，总数单独用计数器统计
                        warning_message_list.extend([
                            f'row {per_row_index} {per_warning_message}'
                            for per_warning_message in
                            per_sample_id_warning_message_list
                        ])

                    # 取出该行对应的风格属性，并用style列原文反查位置映射是否成立
                    per_style_attribute_dict = {}
                    if 0 <= per_style_offset < len(style_attribute_list):
                        per_style_attribute = style_attribute_list[
                            per_style_offset]
                        per_style_attribute_dict = per_style_attribute.get(
                            'style_attribute_dict', {})

                        if get_stripped_text_value(
                                per_style_attribute.get('style_prompt',
                                                        '')) != per_style:
                            style_prompt_mismatch_count += 1
                            per_style_attribute_dict = {}
                            if len(error_message_list
                                   ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                                error_message_list.append(
                                    f'row {per_row_index} style prompt not match style index {per_style_index}'
                                )
                    else:
                        if len(error_message_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                            error_message_list.append(
                                f'row {per_row_index} style offset out of range {per_style_offset}'
                            )

                    # 同一片里出现重名id时按名写盘会互相覆盖，
                    # 这里改写到独立目录保留数据并上报，不能静默丢样本
                    per_sample_is_duplicate = bool(
                        per_sample_id) and per_sample_id in sample_id_dict
                    if per_sample_is_duplicate:
                        duplicate_sample_count += 1
                        if len(error_message_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                            error_message_list.append(
                                f'duplicate sample id {per_sample_id} row {per_row_index}'
                            )
                    elif per_sample_id:
                        sample_id_dict[per_sample_id] = per_row_index

                    per_image_dict = per_row_dict.get(
                        PARQUET_IMAGE_COLUMN_NAME, None)
                    per_image_bytes, per_image_name = None, ''
                    if isinstance(per_image_dict, dict):
                        per_image_bytes = per_image_dict.get(
                            PARQUET_IMAGE_BYTES_KEY_NAME, None)
                        per_image_name = get_stripped_text_value(
                            per_image_dict.get(PARQUET_IMAGE_PATH_KEY_NAME,
                                               None))

                    per_saved_image_flag = False
                    per_image_relative_path, per_image_suffix = '', ''
                    per_image_shape = [0, 0]
                    if isinstance(per_image_bytes,
                                  bytes) and len(per_image_bytes) > 0:
                        # image.path写的是.jpg但字节实测是PNG，所以魔数优先
                        per_image_suffix = get_image_bytes_suffix(
                            per_image_bytes, per_image_name)
                        image_suffix_count_dict[per_image_suffix] += 1

                        per_image_shape, per_image_shape_error_message = get_image_shape(
                            per_image_bytes)
                        if per_image_shape_error_message:
                            warning_message_count += 1
                            if len(warning_message_list
                                   ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                                warning_message_list.append(
                                    f'row {per_row_index} {per_image_shape_error_message}'
                                )
                        else:

                            image_shape_count_dict[
                                f'{per_image_shape[0]}x{per_image_shape[1]}'] += 1

                        # id实测全局唯一，直接当图像名;
                        # 万一id为空或重名，退化用行号保证不覆盖
                        per_image_name_prefix = per_sample_id if per_sample_id else f'{per_parquet_group_name}_{per_row_index:08d}'
                        if per_sample_is_duplicate:
                            per_image_name_prefix = f'{per_image_name_prefix}_{per_row_index:08d}'

                        per_save_image_name = f'{per_image_name_prefix}{per_image_suffix}'
                        per_image_sub_dir_name = f'{per_row_index // SAVE_IMAGE_SUB_DIR_SIZE:04d}'
                        per_image_relative_path = f'{per_parquet_group_name}/{per_image_sub_dir_name}/{per_save_image_name}'

                        if not EXTRACT_IMAGE_FILE_FLAG:
                            # 只建索引模式: 图像继续留在原parquet里，样本对信息一样完整
                            not_save_image_count += 1
                            per_saved_image_flag = True
                        else:
                            if per_sample_is_duplicate:
                                save_image_path = os.path.join(
                                    save_duplicate_dir_path,
                                    per_image_sub_dir_name,
                                    per_save_image_name)
                            else:
                                save_image_path = os.path.join(
                                    save_image_dir_path,
                                    per_image_sub_dir_name,
                                    per_save_image_name)

                            per_write_flag, per_skip_flag, per_save_error_message = save_single_image_bytes(
                                save_image_path, per_image_bytes)
                            if per_save_error_message:
                                if len(error_message_list
                                       ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                                    error_message_list.append(
                                        per_save_error_message)
                            else:
                                per_saved_image_flag = True
                                if per_write_flag:
                                    extract_image_count += 1
                                elif per_skip_flag:
                                    skip_image_count += 1
                    else:
                        if len(error_message_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                            error_message_list.append(
                                f'row {per_row_index} empty image bytes')

                    # 完整有用信息的文生图样本对必需条件(缺任一就不写进有效标注):
                    # 生成后图落盘成功 + 样本id非空 + 内容提示非空 + 风格提示非空
                    per_invalid_reason_list = []
                    if not per_saved_image_flag:
                        per_invalid_reason_list.append(
                            'missing generated image')
                    if not per_sample_id:
                        per_invalid_reason_list.append('empty sample id')
                    if not per_content:
                        per_invalid_reason_list.append('empty content')
                    if not per_style:
                        per_invalid_reason_list.append('empty style')

                    if len(per_invalid_reason_list) > 0:
                        # 信息不完整的样本对不写进有效标注，但必须留痕，不能静默消失。
                        # 明细只留前若干条(避免极端情况下8万条明细撑爆回传的pickle)，
                        # 总数用计数器统计，后面的对账一律以计数器为准
                        invalid_sample_pair_count += 1
                        if len(invalid_sample_pair_list
                               ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                            invalid_sample_pair_list.append({
                                'parquet_name':
                                per_parquet_group_name,
                                'sample_id':
                                per_sample_id,
                                'row_index':
                                per_row_index,
                                'global_row_index':
                                per_global_row_index,
                                'invalid_reason':
                                ','.join(per_invalid_reason_list),
                            })
                        continue

                    per_save_annotation = {
                        'image_path': per_image_relative_path,
                        'sample_id': per_sample_id,
                        'parquet_name': per_parquet_group_name,
                        'row_index': per_row_index,
                        'global_row_index': per_global_row_index,
                        'style_index': per_style_index,
                        'content_index': per_content_index,
                        'task_type': 'text_to_image',
                        't2i_caption': get_t2i_caption(per_content, per_style),
                        'content': per_content,
                        'style': per_style,
                        'style_attribute_dict': per_style_attribute_dict,
                        'image_width': per_image_shape[0],
                        'image_height': per_image_shape[1],
                        'image_suffix': per_image_suffix,
                    }

                    save_annotation_file.write(
                        f'{json.dumps(per_save_annotation, ensure_ascii=False)}\n'
                    )
                    valid_sample_pair_count += 1
                    style_sample_count_dict[per_style_index] += 1

        reach_parquet_end = True

        # 核心对账之一: 实际遍历到的行数必须等于footer里数出来的行数，
        # 否则说明流式读的时候有batch被静默吞掉了
        if row_count != expected_row_count:
            error_message_list.append(
                f'{per_parquet_group_name} row count not match {row_count} != {expected_row_count}'
            )
    except Exception as e:
        # parquet损坏或NAS读失败时保留已解出的图像和标注，但必须上报，不能静默少样本
        print('7777', per_parquet_group_name, e)
        error_message_list.append(f'read parquet failed {e}')

    if not reach_parquet_end:

        error_message_list.append(
            'not reach parquet row batch end, parquet may be truncated')

    # 核心对账之二: 每行只有一张图，所以 extract+skip+not_save 必须等于行数
    if extract_image_count + skip_image_count + not_save_image_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} process image count not match: {extract_image_count} + {skip_image_count} + {not_save_image_count} != {row_count}'
        )

    # 核心对账之三: 每一行要么是有效样本对、要么进隔离清单，绝不静默消失;
    # 实测每行都是完整样本对，所以有效样本对数还必须等于行数
    if valid_sample_pair_count + invalid_sample_pair_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} sample pair count not match {valid_sample_pair_count} + {invalid_sample_pair_count} != {row_count}'
        )

    if valid_sample_pair_count != EXPECTED_PARQUET_ROW_COUNT:
        error_message_list.append(
            f'{per_parquet_group_name} valid sample pair count not match {valid_sample_pair_count} != {EXPECTED_PARQUET_ROW_COUNT}'
        )

    # 核心对账之四: 每片必须恰好覆盖10000个风格，且每个风格恰好8个样本对，
    # 少一个就说明这一片的风格分布被破坏了(样本丢失或行号错位)
    per_style_count = len(style_sample_count_dict)
    per_full_style_count = sum([
        1 for per_sample_count in style_sample_count_dict.values()
        if per_sample_count == EXPECTED_STYLE_SAMPLE_NUM
    ])
    if per_style_count != EXPECTED_PARQUET_STYLE_NUM:
        error_message_list.append(
            f'{per_parquet_group_name} style count not match {per_style_count} != {EXPECTED_PARQUET_STYLE_NUM}'
        )
    if per_full_style_count != EXPECTED_PARQUET_STYLE_NUM:
        error_message_list.append(
            f'{per_parquet_group_name} full style count not match {per_full_style_count} != {EXPECTED_PARQUET_STYLE_NUM}'
        )
    if style_prompt_mismatch_count > 0:
        error_message_list.append(
            f'{per_parquet_group_name} style prompt mismatch count {style_prompt_mismatch_count}'
        )

    return {
        'parquet_relative_path':
        f'{per_parquet_relative_dir}/{per_parquet_group_name}'
        if per_parquet_relative_dir else per_parquet_group_name,
        'parquet_name':
        per_parquet_group_name,
        'parquet_index':
        per_parquet_index,
        'row_count':
        row_count,
        'valid_sample_pair_count':
        valid_sample_pair_count,
        'extract_image_count':
        extract_image_count,
        'skip_image_count':
        skip_image_count,
        'not_save_image_count':
        not_save_image_count,
        'duplicate_sample_count':
        duplicate_sample_count,
        'style_count':
        per_style_count,
        'full_style_count':
        per_full_style_count,
        'style_prompt_mismatch_count':
        style_prompt_mismatch_count,
        'invalid_sample_pair_count':
        invalid_sample_pair_count,
        'warning_message_count':
        warning_message_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_parquet_group_name}.jsonl',
        'image_shape_count_dict':
        dict(image_shape_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'invalid_sample_pair_list':
        invalid_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        warning_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }


def get_all_file_and_parquet_group(root_dataset_path):
    """扫描数据集，收集非parquet文件列表和parquet文件列表，每个parquet视为一个分片"""
    file_copy_pair_list, parquet_group_list = [], []
    for per_root_path, per_dir_name_list, per_file_name_list in os.walk(
            root_dataset_path):
        # .cache里有216个下载缓存文件，直接在遍历时剪掉整棵子树，不要走进去
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

            per_match_result = PARQUET_FILE_NAME_PATTERN.match(per_file_name)
            if not per_match_result:
                file_copy_pair_list.append([
                    per_file_relative_path,
                    per_file_path,
                ])
                continue

            # 分片编号是全局行号的来源，取不到编号就没法把风格属性挂回样本
            per_shard_match_result = PARQUET_SHARD_FILE_NAME_PATTERN.match(
                per_file_name)
            per_parquet_index = int(per_shard_match_result.group(
                'index')) if per_shard_match_result else -1

            parquet_group_list.append([
                per_match_result.group('prefix'),
                per_file_relative_dir,
                per_file_path,
                per_parquet_index,
            ])

    file_copy_pair_list = sorted(file_copy_pair_list, key=lambda x: x[0])
    parquet_group_list = sorted(parquet_group_list, key=lambda x: x[2])

    return file_copy_pair_list, parquet_group_list


def check_single_parquet_dir_on_disk(parquet_check_pair):
    """可选的二次对账: os.walk单个分片的图像目录，核对落盘图像数与标注条数"""
    per_parquet_group_name, per_image_dir_path, per_annotation_path, per_expected_valid_sample_pair_count = parquet_check_pair

    error_message_list = []

    if not os.path.exists(per_image_dir_path):
        error_message_list.append(
            f'{per_parquet_group_name} image dir not exist')

        return [per_parquet_group_name, 0, 0, error_message_list]

    image_name_dict = {}
    unknown_suffix_file_count = 0
    for per_root_path, _, per_file_name_list in os.walk(per_image_dir_path):
        for per_file_name in per_file_name_list:
            if check_image_file_suffix(per_file_name):
                image_name_dict[per_file_name] = 1
            else:
                # 图像目录里不该出现非图像文件，必须上报，不能默认当图像统计
                unknown_suffix_file_count += 1

    annotation_count, missing_image_count = 0, 0
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_annotation_file:
            for per_annotation_line in load_annotation_file:
                per_annotation_line = per_annotation_line.strip()
                if not per_annotation_line:
                    continue

                per_annotation = json.loads(per_annotation_line)
                annotation_count += 1

                per_image_name = os.path.basename(
                    per_annotation.get('image_path', ''))
                if per_image_name not in image_name_dict:
                    missing_image_count += 1
    except Exception as e:
        error_message_list.append(
            f'{per_parquet_group_name} load annotation failed {e}')

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_parquet_group_name} unknown suffix file num {unknown_suffix_file_count}'
        )
    if annotation_count != per_expected_valid_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_group_name} on disk annotation count not match {annotation_count} != {per_expected_valid_sample_pair_count}'
        )
    if len(image_name_dict) != per_expected_valid_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_group_name} on disk image count not match {len(image_name_dict)} != {per_expected_valid_sample_pair_count}'
        )
    if missing_image_count > 0:
        error_message_list.append(
            f'{per_parquet_group_name} on disk missing image count {missing_image_count}'
        )

    return [
        per_parquet_group_name,
        len(image_name_dict),
        annotation_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, save_annotation_dir_path,
                             parquet_result_list):
    """可选的二次对账: 遍历输出目录核对每个分片的落盘图像数与标注条数"""
    parquet_check_pair_list = []
    for per_parquet_result in parquet_result_list:
        parquet_check_pair_list.append([
            per_parquet_result['parquet_name'],
            os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                         per_parquet_result['parquet_name']),
            os.path.join(save_annotation_dir_path,
                         f'{per_parquet_result["parquet_name"]}.jsonl'),
            per_parquet_result['valid_sample_pair_count'],
        ])

    error_message_list = []
    total_image_count, total_annotation_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_dir_on_disk, parquet_check_pair_list),
                                     total=len(parquet_check_pair_list)):
            _, per_image_count, per_annotation_count, per_error_message_list = per_check_result
            total_image_count += per_image_count
            total_annotation_count += per_annotation_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk image:', total_image_count, 'on disk annotation:',
          total_annotation_count)

    return error_message_list


def save_check_result(save_dataset_path, parquet_result_list,
                      expected_total_row_count,
                      style_attribute_empty_value_count_dict):
    """汇总所有parquet的解包与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_row_count, total_valid_sample_pair_count = 0, 0
    total_extract_image_count, total_skip_image_count = 0, 0
    total_not_save_image_count, total_duplicate_sample_count = 0, 0
    total_style_count, total_full_style_count = 0, 0
    total_style_prompt_mismatch_count = 0
    total_invalid_sample_pair_count, total_warning_message_count = 0, 0
    image_shape_count_dict = collections.Counter()

    image_suffix_count_dict = collections.Counter()
    parquet_sample_pair_count_dict = {}
    all_invalid_sample_pair_list, all_warning_message_list = [], []
    error_message_list = []

    for per_parquet_result in parquet_result_list:
        per_parquet_group_name = per_parquet_result['parquet_name']

        total_row_count += per_parquet_result['row_count']
        total_valid_sample_pair_count += per_parquet_result[
            'valid_sample_pair_count']
        total_extract_image_count += per_parquet_result['extract_image_count']
        total_skip_image_count += per_parquet_result['skip_image_count']
        total_not_save_image_count += per_parquet_result[
            'not_save_image_count']
        total_duplicate_sample_count += per_parquet_result[
            'duplicate_sample_count']
        total_style_count += per_parquet_result['style_count']
        total_full_style_count += per_parquet_result['full_style_count']
        total_style_prompt_mismatch_count += per_parquet_result[
            'style_prompt_mismatch_count']
        total_invalid_sample_pair_count += per_parquet_result[
            'invalid_sample_pair_count']
        total_warning_message_count += per_parquet_result[
            'warning_message_count']

        image_shape_count_dict.update(
            per_parquet_result['image_shape_count_dict'])
        image_suffix_count_dict.update(
            per_parquet_result['image_suffix_count_dict'])
        parquet_sample_pair_count_dict[
            per_parquet_group_name] = per_parquet_result[
                'valid_sample_pair_count']

        all_invalid_sample_pair_list.extend(
            per_parquet_result['invalid_sample_pair_list'])
        all_warning_message_list.extend([
            f'{per_parquet_group_name} {per_warning_message}' for
            per_warning_message in per_parquet_result['warning_message_list']
        ])

        if len(per_parquet_result['error_message_list']) > 0:
            print('7777', per_parquet_group_name,
                  per_parquet_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_parquet_group_name} error num {len(per_parquet_result["error_message_list"])} {per_parquet_result["error_message_list"][:3]}'
            )

    print('3333', 'total row:', total_row_count, 'total valid sample pair:',
          total_valid_sample_pair_count, 'extract image:',
          total_extract_image_count, 'skip image:', total_skip_image_count,
          'not save image:', total_not_save_image_count, 'duplicate sample:',
          total_duplicate_sample_count, 'invalid sample pair:',
          total_invalid_sample_pair_count, 'warning:',
          total_warning_message_count)

    print('3333', 'total style:', total_style_count, 'full style:',
          total_full_style_count, 'style prompt mismatch:',
          total_style_prompt_mismatch_count)
    print('3333', 'image suffix:', dict(image_suffix_count_dict))
    print('3333', 'image shape top10:',
          dict(image_shape_count_dict.most_common(10)))

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        'text_to_image',
        'extract_image_file_flag':
        EXTRACT_IMAGE_FILE_FLAG,
        'total_parquet_count':
        len(parquet_result_list),
        'total_row_count':
        total_row_count,
        'expected_total_row_count':
        expected_total_row_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_extract_image_count':
        total_extract_image_count,
        'total_skip_image_count':
        total_skip_image_count,
        'total_not_save_image_count':
        total_not_save_image_count,
        'total_duplicate_sample_count':
        total_duplicate_sample_count,
        'total_style_count':
        total_style_count,
        'total_full_style_count':
        total_full_style_count,
        'total_style_prompt_mismatch_count':
        total_style_prompt_mismatch_count,
        'invalid_sample_pair_count':
        total_invalid_sample_pair_count,
        'warning_message_count':
        total_warning_message_count,
        'style_attribute_empty_value_count_dict':
        style_attribute_empty_value_count_dict,
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'image_shape_count_dict':
        dict(image_shape_count_dict),
        'parquet_sample_pair_count_dict':
        parquet_sample_pair_count_dict,
        'invalid_sample_pair_list':
        all_invalid_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        sorted(set(all_warning_message_list))[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'check_error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    # 全量硬对账: 解析行数与有效样本对数都必须等于预检时从footer数出来的总行数，
    # 有效样本对数少一个都说明有样本对在"读parquet->写图->写jsonl"这条链路上消失了
    if total_valid_sample_pair_count == 0:
        error_message_list.append('no valid sample pair found')
    if total_row_count != expected_total_row_count:
        error_message_list.append(
            f'total row count not match {total_row_count} != {expected_total_row_count}'
        )
    if total_valid_sample_pair_count != expected_total_row_count:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} != {expected_total_row_count}'
        )
    if total_extract_image_count + total_skip_image_count + total_not_save_image_count != expected_total_row_count:
        error_message_list.append(
            f'total process image count not match {total_extract_image_count} + {total_skip_image_count} + {total_not_save_image_count} != {expected_total_row_count}'
        )
    if total_invalid_sample_pair_count > 0:
        error_message_list.append(
            f'invalid sample pair count {total_invalid_sample_pair_count}')
    if total_duplicate_sample_count > 0:

        error_message_list.append(
            f'duplicate sample count {total_duplicate_sample_count}')
    if total_style_prompt_mismatch_count > 0:
        error_message_list.append(
            f'style prompt mismatch count {total_style_prompt_mismatch_count}')

    # 100片风格互不重叠，所以覆盖风格数累加起来必须刚好是1000000且全部满8张
    if total_style_count != EXPECTED_STYLE_NUM:
        error_message_list.append(
            f'total style count not match {total_style_count} != {EXPECTED_STYLE_NUM}'
        )
    if total_full_style_count != EXPECTED_STYLE_NUM:
        error_message_list.append(
            f'total full style count not match {total_full_style_count} != {EXPECTED_STYLE_NUM}'
        )

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    file_copy_pair_list, parquet_group_list = get_all_file_and_parquet_group(
        root_dataset_path)

    print('1111', len(file_copy_pair_list), len(parquet_group_list))
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(parquet_group_list) > 0:
        print('1111', parquet_group_list[0][0], parquet_group_list[0][1],
              parquet_group_list[0][2], parquet_group_list[0][3])

    if len(parquet_group_list) == 0:
        raise Exception('no parquet file found')

    expected_total_row_count, per_parquet_row_count_dict, precheck_error_message_list = check_required_dataset_complete(
        root_dataset_path, parquet_group_list, file_copy_pair_list)
    if len(precheck_error_message_list) > 0:
        # 数据集本身不完整就没必要跑几十小时解包
        raise Exception(
            f'check dataset failed error num {len(precheck_error_message_list)} {precheck_error_message_list[:20]}'
        )

    if len(parquet_group_list) != EXPECTED_PARQUET_NUM:
        raise Exception(
            f'parquet group num not match {len(parquet_group_list)} != {EXPECTED_PARQUET_NUM}'
        )

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    save_style_attribute_dir_path = os.path.join(
        save_dataset_path, SAVE_STYLE_ATTRIBUTE_DIR_NAME)
    os.makedirs(save_style_attribute_dir_path, exist_ok=True)

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

    # 先把风格属性切成每片一个jsonl，子进程解包时直接按分片名加载自己那10000条
    print('1111', 'save style attribute file')
    style_metadata_row_count, style_attribute_empty_value_count_dict, style_attribute_error_message_list = save_style_attribute_file(
        root_dataset_path, save_style_attribute_dir_path)
    if len(style_attribute_error_message_list) > 0:
        # 风格属性对不上就没必要往下跑，否则解出来的样本对会缺风格信息
        raise Exception(
            f'save style attribute failed error num {len(style_attribute_error_message_list)} {style_attribute_error_message_list[:20]}'
        )

    print('1111', 'style attribute row:', style_metadata_row_count)

    parquet_result_list = []
    extract_func = partial(
        process_single_parquet_file,
        save_dataset_path=save_dataset_path,
        save_annotation_dir_path=save_annotation_dir_path,
        save_style_attribute_dir_path=save_style_attribute_dir_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_parquet_result in tqdm(pool.imap_unordered(
                extract_func, parquet_group_list),
                                       total=len(parquet_group_list)):
            parquet_result_list.append(per_parquet_result)

            print('2222', per_parquet_result['parquet_name'], 'row:',
                  per_parquet_result['row_count'], 'valid sample pair:',
                  per_parquet_result['valid_sample_pair_count'],
                  'extract image:', per_parquet_result['extract_image_count'],
                  'skip image:', per_parquet_result['skip_image_count'],
                  'not save image:',
                  per_parquet_result['not_save_image_count'], 'style:',
                  per_parquet_result['style_count'], 'full style:',
                  per_parquet_result['full_style_count'], 'duplicate sample:',
                  per_parquet_result['duplicate_sample_count'],
                  'invalid sample pair:',
                  len(per_parquet_result['invalid_sample_pair_list']))

    check_error_message_list = save_check_result(
        save_dataset_path, parquet_result_list, expected_total_row_count,
        style_attribute_empty_value_count_dict)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG and EXTRACT_IMAGE_FILE_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, save_annotation_dir_path, parquet_result_list)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解包/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/MegaStyle-1.4M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
