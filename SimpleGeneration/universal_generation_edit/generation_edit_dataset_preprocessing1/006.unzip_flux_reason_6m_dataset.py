import os
import re
import io
import json
import shutil
import collections

import pyarrow.parquet as pq

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: FLUX-Reason-6M(LucasFang/FLUX-Reason-6M)
#
# 【数据集类型】纯文生图(text-to-image)数据集，不是图像编辑数据集。
# 每个样本对只有"一张目标图 + 多条不同粒度的caption"，没有任何参考图/输入图/编辑指令/mask，
# README原文也写明是"6-million-scale text-to-image dataset"，
# 所以下游只能用t2i那条链路，不能当ti2i(编辑)数据用。
#
# 【root_dataset_path实测原始保存规格】
# FLUX-Reason-6M/
# ├── Aesthetics-Part01/  415个 fluxdb-aesthetics-part01-{00000-00414}-of-00415.parquet
# │                       (228G, 2071075行, 前414片各5000行 + 最后一片1075行)
# ├── Aesthetics-Part02/  454个 fluxdb-aesthetics-part02-{00000-00453}-of-00454.parquet
# │                       (264G, 2269710行, 前453片各5000行 + 最后一片4710行)
# ├── Imaginative/        168个 fluxdb-imaginative-{00000-00167}-of-00168.parquet
# │                       (269G,  838157行, 前167片各5000行 + 最后一片3157行)
# ├── Text/               143个 fluxdb-text-{00000-00142}-of-00143.parquet
# │                       ( 62G,  711337行, 前142片各5000行 + 最后一片1337行)
# ├── README.md           数据集说明(无用)
# ├── .gitattributes      git lfs配置(无用)
# └── .cache/             huggingface下载缓存(1.3M，实测0个*.incomplete)(无用)
#
# 实测共1180个parquet、5890279行(README宣称6M，实测5.89M)；
# 4个子集目录下0个非parquet文件；每片50个row group、PAR1头尾魔数完整；
# 4个子集的schema完全一致，都是36列。
#
# 【单行parquet的全部36列(4个子集完全一致)】
#   id            : 形如aesthetics-part01-00000000，自带子集前缀所以全局唯一         -> 有用(样本唯一id/图像名)
#                   实测每片内编号连续且起点 = 分片号*5000，跨片无重叠
#   image         : struct<bytes: binary, path: string>                             -> 有用(生成后图，唯一的图)
#                   实测path恒为None、bytes全部是JPEG魔数\xff\xd8\xff、全部1024x1024、0个空图
#   caption_composition / caption_entity / caption_text / caption_imaginative /
#   caption_style / caption_abstract / caption_original / caption_detail            -> 有用(8种粒度英文提示)
#                   caption_detail就是README里说的GCoT(Generation Chain-of-Thought)
#   caption_*_cn ×8 : 上面8种的中文版，实测与英文版同步为null                        -> 有用(中英双语训练)
#   bool_caption_* ×8 : 该caption是否可用的质量门控标志                              -> 有用(caption采样门控)
#   score_composition / score_entity / score_text / score_imaginative /
#   score_style / score_abstract / score_original                                   -> 有用(6个特性打分, [0,10])
#   score_image_clarity / score_image_structure                                     -> 有用(图像质量打分, [0,10])
#
# 【无用信息(不整理进训练目录)】
#   .cache/ / .gitattributes / README.md;
#   image.path恒为None，只在推断图像后缀时兜底用一下，不单独落盘。
#
# 【两个必须显式感知的规格坑】
# 1) bool_caption_X 不能当"caption是否存在"用。
#    实测composition/entity/text/imaginative/style/abstract/detail这7类，
#    bool_caption_X 与 "caption_X文本非空" 100%一致(可当presence用);
#    但 bool_caption_original 大面积不一致: Aesthetics实测1287/5000行是
#    "bool=False但文本非空"，Text子集更是5000/5000全部"bool=False但文本非空"。
#    说明它是质量门控标志(实测bool=True的score_original>=8、bool=False的<=7)，
#    不是presence标志。所以判断caption有没有一律看文本本身，
#    original这一类的bool与presence不一致进白名单不报错。
# 2) caption_detail是唯一恒存在的caption(实测抽样15000行全部非空)，
#    其余7类大量为null(属数据集原始规格，不算信息不完整)。
#    所以"样本对必须有文本"这条硬性条件只能压在caption_detail上，
#    其余7类按非空情况记进caption_type_list，供下游按粒度采样。
#
# 【本脚本的处理口径】
# - 解包前预检: 子集目录、每子集parquet数量、分片编号连号与-of-总数自洽、
#   每片PAR1头尾魔数(O(1)读，拦下载截断)、只读footer拿num_rows并按子集硬对账，
#   任一不过直接抛异常，不白跑几十小时;
# - 解包时流式iter_batches，图像落盘 images/<subset>/<分片名>/<id>.jpg，
#   写盘后立刻校验落盘大小 == len(bytes)(避免只看存在性把半截图当正常样本);
# - 同时把每行的非图像列汇总成 unzip_annotations/<subset>/<分片名>.jsonl，
#   原36列的有用属性全部保留，再补上落盘路径/样本id/子集名/分片名/行号/图像宽高，
#   下游直接按行取样本，不需要为了拿caption去扫589万个小文件;
# - 片内三方硬对账: 解析行数 == footer num_rows、
#   extract+skip+not_save == 行数、有效样本对数 == 行数;
# - id片内重名时改写到独立目录保留并上报，不静默覆盖丢样本;
# - 任何一环出错都汇总后抛异常，不再静默跑过。
# ==============================================================================

# 带分片编号的parquet名(fluxdb-aesthetics-part01-00000-of-00415.parquet)，
# 用于分片完整性预检
PARQUET_SHARD_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>(?P<split>.+)-(?P<index>\d+)-of-(?P<total>\d+))\.parquet$')

PARQUET_FILE_NAME_PATTERN = re.compile(r'^(?P<prefix>.+)\.parquet$')

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(1.3M，实测0个*.incomplete)
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

# 该数据集只有这四个子集目录
SUBSET_ROOT_DIR_NAME_LIST = [
    'Aesthetics-Part01',
    'Aesthetics-Part02',
    'Imaginative',
    'Text',
]

# 实测每子集parquet分片数(与文件名里的-of-总数一致)，数量不对说明下载不全
EXPECTED_SUBSET_PARQUET_NUM_DICT = {
    'Aesthetics-Part01': 415,
    'Aesthetics-Part02': 454,
    'Imaginative': 168,
    'Text': 143,
}

# 实测每子集parquet footer行数总和，直接当作完整性ground truth，
# 缺子集/缺分片/少行都能拦住(合计5890279行)
EXPECTED_SUBSET_ROW_COUNT_DICT = {
    'Aesthetics-Part01': 2071075,
    'Aesthetics-Part02': 2269710,
    'Imaginative': 838157,
    'Text': 711337,
}

EXPECTED_TOTAL_ROW_COUNT = 5890279

# 实测每片行数(除每子集最后一片是余数外都是5000)，用来软校验分片切分规格
EXPECTED_PARQUET_ROW_COUNT = 5000

# 每行的样本唯一id列，实测自带子集前缀所以全局唯一，直接当图像文件名前缀
PARQUET_SAMPLE_ID_COLUMN_NAME = 'id'

# 每行唯一的图像列(生成后图)。文生图数据集没有参考图，所以只有这一列
PARQUET_IMAGE_COLUMN_NAME = 'image'

PARQUET_IMAGE_BYTES_KEY_NAME = 'bytes'

PARQUET_IMAGE_PATH_KEY_NAME = 'path'

# 8种caption粒度后缀。caption_<X>是英文、caption_<X>_cn是中文、
# bool_caption_<X>是质量门控标志
ANNOTATION_CAPTION_TYPE_NAME_LIST = [
    'composition',
    'entity',
    'text',
    'imaginative',
    'style',
    'abstract',
    'original',
    'detail',
]

# 样本对必须有的文本提示。实测caption_detail(即GCoT)是唯一恒非空的caption，
# 其余7类大量为null属数据集原始规格，不能拿来当必需字段
ANNOTATION_REQUIRED_CAPTION_TYPE_NAME = 'detail'

# bool_caption_X 与 "caption_X文本非空" 已知不一致的类型白名单。
# bool_caption_original是质量门控(score_original>=8)而非presence标志，
# 实测Text子集5000/5000行都是"bool=False但文本非空"，不能当异常上报
BOOL_CAPTION_PRESENCE_INCONSISTENT_TYPE_NAME_LIST = [
    'original',
]

# 每行除caption外必须齐备的打分属性(实测抽样全部齐备，缺失只上报不丢样本)
ANNOTATION_EXPECTED_SCORE_KEY_NAME_LIST = [
    'score_composition',
    'score_entity',
    'score_text',
    'score_imaginative',
    'score_style',
    'score_abstract',
    'score_original',
    'score_image_clarity',
    'score_image_structure',
]

# 实测所有打分都在[0, 10]闭区间内，越界只上报不丢样本
ANNOTATION_SCORE_VALUE_RANGE = [0, 10]

SAVE_IMAGE_DIR_NAME = 'images'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_DUPLICATE_SAMPLE_DIR_NAME = 'unzip_duplicate_samples'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

IMAGE_BYTES_MAGIC_SUFFIX_LIST = [
    [b'\xff\xd8\xff', '.jpg'],
    [b'\x89PNG\r\n\x1a\n', '.png'],
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
# True : 和其他数据集脚本口径一致，589万张1024x1024 JPEG约820G + 1180个jsonl，
#        务必确认目标盘扛得住再跑;
# False: 只解析非图像列生成 unzip_annotations/*.jsonl 索引(一两小时即可跑完)，
#        图像继续留在原parquet里，训练时按parquet顺序读，样本对信息一样完整。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解包后再os.walk一遍输出目录做二次对账。
# 默认False: 589万个小文件的os.walk在NAS上要跑很久，而解包时已经做了
# "写盘后立刻校验落盘大小 == len(bytes)" + "extract+skip+not_save == 行数"两道对账，
# 已经能保证每行的图都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 是否解码图像拿真实宽高写进标注。
# 默认True: 实测全部是1024x1024，只解header不解像素(PIL的Image.open是惰性的)，
# 代价可忽略，但下游做分辨率分桶时就不用再打开589万张图。
PARSE_IMAGE_SHAPE_FLAG = True

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

PARQUET_ROW_BATCH_SIZE = 64


def check_skip_file_or_dir(per_file_relative_path):
    """过滤掉.cache、.gitattributes、README.md这几个不需要整理的文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """只把图像后缀的文件计入解出图像总数和orphan统计"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


def get_image_bytes_suffix(per_image_bytes, per_image_name):
    """优先用parquet中记录的图像文件名后缀，取不到时再用图像字节的魔数推断后缀

    实测该数据集image.path恒为None、图像字节全部是JPEG魔数，所以基本都走魔数分支。
    """
    if per_image_name:
        per_image_name_suffix = os.path.splitext(per_image_name)[1].lower()
        if per_image_name_suffix in IMAGE_FILE_SUFFIX_LIST:
            return per_image_name_suffix

    for per_magic_bytes, per_magic_suffix in IMAGE_BYTES_MAGIC_SUFFIX_LIST:
        if per_image_bytes.startswith(per_magic_bytes):
            return per_magic_suffix

    return '.jpg'


def get_json_serializable_value(per_column_value):
    """把非图像列里可能出现的裸字节转成占位字符串，避免json.dump整片抛错丢样本"""
    if isinstance(per_column_value, bytes):
        return f'<bytes len={len(per_column_value)}>'

    if isinstance(per_column_value, dict):
        return {
            per_key: get_json_serializable_value(per_value)
            for per_key, per_value in per_column_value.items()
        }

    if isinstance(per_column_value, (list, tuple)):
        return [
            get_json_serializable_value(per_value)
            for per_value in per_column_value
        ]

    return per_column_value


def get_stripped_text_value(per_column_value):
    """文本列统一转成strip后的字符串，None/空白都当成缺失"""
    if not isinstance(per_column_value, str):
        return ''

    return per_column_value.strip()


def get_parquet_subset_name(per_parquet_relative_dir):
    """parquet所在的相对目录名就是子集名(Aesthetics-Part01/Imaginative/Text等)"""
    per_parquet_relative_dir = per_parquet_relative_dir.replace('\\', '/')

    return per_parquet_relative_dir.split('/')[0]


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

    实测1180个parquet全部满足，说明当前数据集是完整的。
    如果下载不全，流式解包只会在读到一半时抛异常，必须在跑几十小时前先拦住。
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
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    try:
        load_parquet_file = pq.ParquetFile(per_parquet_path)
        per_row_count = load_parquet_file.metadata.num_rows
        per_column_name_list = list(load_parquet_file.schema_arrow.names)
    except Exception as e:
        print('7777', per_parquet_group_name, e)

        return [
            per_parquet_group_name,
            per_parquet_relative_dir,
            0,
            [],
            [f'read parquet metadata failed {per_parquet_group_name} {e}'],
        ]

    return [
        per_parquet_group_name,
        per_parquet_relative_dir,
        per_row_count,
        per_column_name_list,
        [],
    ]


def check_parquet_shard_complete(parquet_group_list):
    """解包前预检: 每子集的分片数、分片编号连号、-of-总数自洽，缺片直接中止

    1180片少下几片时脚本照样能跑完并退出0，会静默少掉几万个样本对，必须先拦住。
    """
    error_message_list = []

    subset_shard_index_dict = {}
    for per_parquet_group_name, per_parquet_relative_dir, per_parquet_path in parquet_group_list:
        per_subset_name = get_parquet_subset_name(per_parquet_relative_dir)
        per_parquet_name = os.path.basename(per_parquet_path)

        per_match_result = PARQUET_SHARD_FILE_NAME_PATTERN.match(
            per_parquet_name)
        if not per_match_result:
            error_message_list.append(
                f'unknown parquet name {per_subset_name}/{per_parquet_name}')
            continue

        per_shard_index = int(per_match_result.group('index'))
        per_shard_total = int(per_match_result.group('total'))
        subset_shard_index_dict.setdefault(per_subset_name,
                                           [{}, {}])[0][per_shard_index] = 1
        subset_shard_index_dict[per_subset_name][1][per_shard_total] = 1

    for per_subset_name in sorted(subset_shard_index_dict.keys()):
        per_shard_index_dict, per_shard_total_dict = subset_shard_index_dict[
            per_subset_name]

        # 同一个子集里所有分片文件名的-of-总数必须一致，否则说明混进了别的版本
        if len(per_shard_total_dict) != 1:
            error_message_list.append(
                f'{per_subset_name} shard total not unique {sorted(per_shard_total_dict.keys())}'
            )
            continue

        per_shard_total = list(per_shard_total_dict.keys())[0]
        per_expected_parquet_num = EXPECTED_SUBSET_PARQUET_NUM_DICT[
            per_subset_name]

        print('1111', per_subset_name, 'parquet:', len(per_shard_index_dict),
              '/', per_shard_total, 'expected parquet:',
              per_expected_parquet_num)

        if per_shard_total != per_expected_parquet_num:
            error_message_list.append(
                f'{per_subset_name} shard total not match {per_shard_total} != {per_expected_parquet_num}'
            )
        if len(per_shard_index_dict) != per_expected_parquet_num:
            error_message_list.append(
                f'{per_subset_name} parquet num not match {len(per_shard_index_dict)} != {per_expected_parquet_num}'
            )

        # 分片编号必须是0..N-1连号，缺号说明有分片没下载下来
        per_missing_shard_index_list = sorted(
            set(range(0, per_expected_parquet_num)) -
            set(per_shard_index_dict.keys()))
        if len(per_missing_shard_index_list) > 0:
            error_message_list.append(
                f'{per_subset_name} shard index not continuous, missing index {per_missing_shard_index_list[:20]} total missing {len(per_missing_shard_index_list)}'
            )

    # 少下整个子集时，上面按-of-连号的校验完全发现不了
    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        if per_subset_name not in subset_shard_index_dict:
            error_message_list.append(f'subset not exists {per_subset_name}')

    for per_subset_name in sorted(subset_shard_index_dict.keys()):
        if per_subset_name not in SUBSET_ROOT_DIR_NAME_LIST:
            error_message_list.append(f'unexpected subset {per_subset_name}')

    return error_message_list


def check_parquet_row_count_complete(parquet_group_list):
    """只读footer按子集与实测总行数逐项硬对账，同时校验36列schema是否一致"""
    error_message_list = []

    total_row_count = 0
    per_parquet_row_count_dict = {}
    subset_row_count_dict = collections.Counter()
    subset_parquet_num_dict = collections.Counter()
    column_name_key_dict = {}

    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_file_row_count, parquet_group_list),
                                     total=len(parquet_group_list)):
            per_parquet_group_name, per_parquet_relative_dir, per_row_count, per_column_name_list, per_error_message_list = per_check_result

            error_message_list.extend(per_error_message_list)
            if len(per_error_message_list) > 0:
                continue

            per_subset_name = get_parquet_subset_name(per_parquet_relative_dir)
            per_parquet_row_count_dict[per_parquet_group_name] = per_row_count
            subset_row_count_dict[per_subset_name] += per_row_count
            subset_parquet_num_dict[per_subset_name] += 1
            total_row_count += per_row_count

            column_name_key_dict.setdefault(','.join(per_column_name_list),
                                            []).append(per_parquet_group_name)

    # 4个子集的schema实测完全一致(36列)，出现第二种列组合说明数据规格变了
    if len(column_name_key_dict) != 1:
        for per_column_name_key in sorted(column_name_key_dict.keys()):
            error_message_list.append(
                f'parquet column name not unique, parquet num {len(column_name_key_dict[per_column_name_key])} example {column_name_key_dict[per_column_name_key][0]}'
            )

    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        per_expected_row_count = EXPECTED_SUBSET_ROW_COUNT_DICT[
            per_subset_name]
        per_subset_row_count = subset_row_count_dict.get(per_subset_name, 0)

        print('1111', per_subset_name, 'parquet:',
              subset_parquet_num_dict.get(per_subset_name, 0), 'row:',
              per_subset_row_count, 'expected row:', per_expected_row_count)

        if per_subset_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_subset_name} row count not match {per_subset_row_count} != {per_expected_row_count}'
            )

    print('1111', 'total row:', total_row_count,
          'expected total row:', EXPECTED_TOTAL_ROW_COUNT, 'parquet:',
          len(per_parquet_row_count_dict))

    if total_row_count != EXPECTED_TOTAL_ROW_COUNT:
        error_message_list.append(
            f'total row count not match {total_row_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )

    return total_row_count, per_parquet_row_count_dict, error_message_list


def check_required_subset_complete(root_dataset_path, parquet_group_list):
    """解包前预检: 子集目录、parquet数量、分片连号、PAR1魔数、footer行数

    数据集本身不完整就没必要跑几十小时解包，也避免"少了几片但整体报成功"。
    """
    error_message_list = []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')

        return 0, {}, error_message_list

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

    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        if not os.path.exists(os.path.join(root_dataset_path,
                                           per_subset_name)):
            error_message_list.append(
                f'subset dir not exist {per_subset_name}')

    error_message_list.extend(check_parquet_shard_complete(parquet_group_list))

    parquet_path_check_list = [
        per_parquet_path for _, _, per_parquet_path in parquet_group_list
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


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非parquet文件原样拷贝到目标目录，保持相对路径不变

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

    if not os.path.exists(save_file_path) or os.path.getsize(
            save_file_path) != os.path.getsize(per_file_path):
        print('4444', per_file_path, 'copy file size not match')

        return [per_file_relative_path, 'copy file size not match']

    return [per_file_relative_path, '']


def save_single_image_bytes(save_image_path, per_image_bytes):
    """单张图独立落盘并立刻校验落盘大小，返回[是否新写, 是否跳过, 错误信息]

    单张图写盘异常不能让整片parquet的循环中断，否则该片后面几千行既不解图也不进标注。
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


def get_single_annotation_caption_dict(per_row_dict):
    """取出一行里8种粒度的中英文caption、门控标志与非空类型，并上报规格异常

    返回:
      per_caption_dict         : caption_<X> / caption_<X>_cn / bool_caption_<X> 全量原值
      per_caption_type_name_list : 英文caption非空的粒度类型列表(下游按粒度采样用)
      error_message_list       : bool与presence不一致等规格异常(只上报，不丢样本)
    """
    per_caption_dict = {}
    per_caption_type_name_list = []
    error_message_list = []

    for per_caption_type_name in ANNOTATION_CAPTION_TYPE_NAME_LIST:
        per_caption_key_name = f'caption_{per_caption_type_name}'
        per_caption_cn_key_name = f'caption_{per_caption_type_name}_cn'
        per_bool_caption_key_name = f'bool_caption_{per_caption_type_name}'

        for per_key_name in [
                per_caption_key_name,
                per_caption_cn_key_name,
                per_bool_caption_key_name,
        ]:
            if per_key_name not in per_row_dict:
                error_message_list.append(f'miss key {per_key_name}')

        per_caption = get_stripped_text_value(
            per_row_dict.get(per_caption_key_name, None))
        per_caption_cn = get_stripped_text_value(
            per_row_dict.get(per_caption_cn_key_name, None))
        per_bool_caption = per_row_dict.get(per_bool_caption_key_name, None)

        per_caption_dict[per_caption_key_name] = per_caption
        per_caption_dict[per_caption_cn_key_name] = per_caption_cn
        per_caption_dict[per_bool_caption_key_name] = bool(per_bool_caption)

        if per_caption:
            per_caption_type_name_list.append(per_caption_type_name)

        # bool_caption_original是质量门控而非presence标志(实测Text子集5000/5000不一致)，
        # 进白名单不报;其余7类实测100%一致，一旦不一致说明数据规格变了，必须显式感知
        if per_caption_type_name in BOOL_CAPTION_PRESENCE_INCONSISTENT_TYPE_NAME_LIST:
            continue

        if bool(per_bool_caption) != bool(per_caption):
            error_message_list.append(
                f'{per_bool_caption_key_name} not match caption presence {bool(per_bool_caption)} != {bool(per_caption)}'
            )

    return per_caption_dict, per_caption_type_name_list, error_message_list


def get_single_annotation_error_message_list(per_row_dict, per_sample_id,
                                             per_caption_type_name_list,
                                             per_saved_image_flag):
    """判定一行是否是完整有用信息的文生图样本对

    必需条件(缺任一该样本对就不完整，不写进有效标注):
      1. 生成后图落盘成功(或只建索引模式下图像字节非空);
      2. 样本唯一id非空;
      3. caption_detail(GCoT)非空——实测它是唯一恒非空的caption。
    只上报不丢样本的软条件: 打分属性缺失或越界。
    """
    invalid_reason_list, warning_message_list = [], []

    if not per_saved_image_flag:
        invalid_reason_list.append('missing generated image')

    if not per_sample_id:
        invalid_reason_list.append('empty sample id')

    if ANNOTATION_REQUIRED_CAPTION_TYPE_NAME not in per_caption_type_name_list:
        # 文生图样本对必须有文本提示，没有caption_detail的图不可训练
        invalid_reason_list.append(
            f'empty caption_{ANNOTATION_REQUIRED_CAPTION_TYPE_NAME}')

    for per_score_key_name in ANNOTATION_EXPECTED_SCORE_KEY_NAME_LIST:
        if per_score_key_name not in per_row_dict:
            warning_message_list.append(f'miss key {per_score_key_name}')
            continue

        per_score_value = per_row_dict[per_score_key_name]
        if not isinstance(per_score_value, int) or isinstance(
                per_score_value, bool):
            warning_message_list.append(
                f'{per_score_key_name} not a int {per_score_value}')
            continue

        if not ANNOTATION_SCORE_VALUE_RANGE[
                0] <= per_score_value <= ANNOTATION_SCORE_VALUE_RANGE[1]:
            warning_message_list.append(
                f'{per_score_key_name} out of range {per_score_value}')

    return invalid_reason_list, warning_message_list


def process_single_parquet_file(parquet_group, save_dataset_path,
                                save_annotation_dir_path):
    """流式解开单个parquet，把内嵌图像字节写成图像文件、非图像列写成jsonl汇总标注

    落盘结构:
      images/<subset>/<分片名>/<id>.jpg
      unzip_annotations/<subset>/<分片名>.jsonl
    每行jsonl就是一个完整有用信息的文生图样本对，保留原parquet的全部有用属性。

    parquet按iter_batches流式读，绝不整片进内存(单片最大约1.6G图像字节)。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    per_subset_name = get_parquet_subset_name(per_parquet_relative_dir)
    per_parquet_relative_path = f'{per_subset_name}/{per_parquet_group_name}'

    save_image_dir_path = os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                                       per_subset_name, per_parquet_group_name)
    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_SAMPLE_DIR_NAME,
                                           per_subset_name,
                                           per_parquet_group_name)
    save_annotation_path = os.path.join(save_annotation_dir_path,
                                        per_subset_name,
                                        f'{per_parquet_group_name}.jsonl')

    if EXTRACT_IMAGE_FILE_FLAG:
        os.makedirs(save_image_dir_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)

    row_count, valid_sample_pair_count = 0, 0
    extract_image_count, skip_image_count, not_save_image_count = 0, 0, 0
    duplicate_sample_count = 0
    caption_type_count_dict = collections.Counter()
    caption_cn_type_count_dict = collections.Counter()
    image_shape_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    invalid_sample_pair_list, warning_message_list = [], []
    error_message_list = []
    reach_parquet_end = False

    sample_id_dict = {}

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

                    per_sample_id = get_stripped_text_value(
                        per_row_dict.get(PARQUET_SAMPLE_ID_COLUMN_NAME, None))

                    # 同一片里出现重名id时按名写盘会互相覆盖，
                    # 这里改写到独立目录保留数据并上报，不能静默丢样本
                    per_sample_is_duplicate = bool(
                        per_sample_id) and per_sample_id in sample_id_dict
                    if per_sample_is_duplicate:
                        duplicate_sample_count += 1
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
                    per_image_relative_path = ''
                    per_image_shape = [0, 0]
                    if isinstance(per_image_bytes,
                                  bytes) and len(per_image_bytes) > 0:
                        per_image_suffix = get_image_bytes_suffix(
                            per_image_bytes, per_image_name)
                        image_suffix_count_dict[per_image_suffix] += 1

                        per_image_shape, per_image_shape_error_message = get_image_shape(
                            per_image_bytes)
                        if per_image_shape_error_message:
                            warning_message_list.append(
                                f'row {per_row_index} {per_image_shape_error_message}'
                            )
                        else:
                            image_shape_count_dict[
                                f'{per_image_shape[0]}x{per_image_shape[1]}'] += 1

                        # id实测全局唯一(自带子集前缀)，直接当图像名;
                        # 万一id为空或重名，退化用行号保证不覆盖
                        per_image_name_prefix = per_sample_id if per_sample_id else f'{per_parquet_group_name}_{per_row_index:08d}'
                        if per_sample_is_duplicate:
                            per_image_name_prefix = f'{per_image_name_prefix}_{per_row_index:08d}'

                        per_save_image_name = f'{per_image_name_prefix}{per_image_suffix}'
                        per_image_relative_path = f'{per_subset_name}/{per_parquet_group_name}/{per_save_image_name}'

                        if not EXTRACT_IMAGE_FILE_FLAG:
                            # 只建索引模式: 图像继续留在原parquet里，样本对信息一样完整
                            not_save_image_count += 1
                            per_saved_image_flag = True
                        else:
                            if per_sample_is_duplicate:
                                save_image_path = os.path.join(
                                    save_duplicate_dir_path,
                                    per_save_image_name)
                            else:
                                save_image_path = os.path.join(
                                    save_image_dir_path, per_save_image_name)

                            per_write_flag, per_skip_flag, per_save_error_message = save_single_image_bytes(
                                save_image_path, per_image_bytes)
                            if per_save_error_message:
                                error_message_list.append(
                                    per_save_error_message)
                            else:
                                per_saved_image_flag = True
                                if per_write_flag:
                                    extract_image_count += 1
                                elif per_skip_flag:
                                    skip_image_count += 1
                    else:
                        error_message_list.append(
                            f'row {per_row_index} empty image bytes')

                    per_caption_dict, per_caption_type_name_list, per_caption_error_message_list = get_single_annotation_caption_dict(
                        per_row_dict)
                    warning_message_list.extend([
                        f'row {per_row_index} {per_caption_error_message}'
                        for per_caption_error_message in
                        per_caption_error_message_list
                    ])

                    per_invalid_reason_list, per_warning_message_list = get_single_annotation_error_message_list(
                        per_row_dict, per_sample_id,
                        per_caption_type_name_list, per_saved_image_flag)
                    warning_message_list.extend([
                        f'row {per_row_index} {per_warning_message}'
                        for per_warning_message in per_warning_message_list
                    ])

                    if len(per_invalid_reason_list) > 0:
                        # 信息不完整的样本对不写进有效标注，但必须留痕，不能静默消失
                        invalid_sample_pair_list.append({
                            'subset_name':
                            per_subset_name,
                            'parquet_name':
                            per_parquet_group_name,
                            'sample_id':
                            per_sample_id,
                            'row_index':
                            per_row_index,
                            'invalid_reason':
                            ','.join(per_invalid_reason_list),
                        })
                        continue

                    # 完整有用信息的样本对: 保留原parquet的全部有用属性
                    # (8种中英文caption、8个门控标志、9个打分)，
                    # 再补上落盘路径、样本id、子集名、分片名、行号与图像宽高，
                    # 下游可以直接按行取样本，不需要为了拿caption去扫589万个小文件
                    per_save_annotation = {
                        'image_path':
                        per_image_relative_path,
                        'sample_id':
                        per_sample_id,
                        'subset_name':
                        per_subset_name,
                        'parquet_name':
                        per_parquet_group_name,
                        'row_index':
                        per_row_index,
                        'task_type':
                        'text_to_image',
                        'caption':
                        per_caption_dict[
                            f'caption_{ANNOTATION_REQUIRED_CAPTION_TYPE_NAME}'],
                        'caption_type_list':
                        per_caption_type_name_list,
                        'image_width':
                        per_image_shape[0],
                        'image_height':
                        per_image_shape[1],
                    }
                    for per_column_name, per_column_value in per_row_dict.items(
                    ):
                        if per_column_name == PARQUET_IMAGE_COLUMN_NAME:
                            continue
                        if per_column_name in per_caption_dict:
                            continue
                        per_save_annotation[
                            per_column_name] = get_json_serializable_value(
                                per_column_value)
                    for per_caption_key_name, per_caption_value in per_caption_dict.items(
                    ):
                        per_save_annotation[
                            per_caption_key_name] = per_caption_value

                    save_annotation_file.write(
                        f'{json.dumps(per_save_annotation, ensure_ascii=False)}\n'
                    )
                    valid_sample_pair_count += 1

                    for per_caption_type_name in per_caption_type_name_list:
                        caption_type_count_dict[per_caption_type_name] += 1
                    for per_caption_type_name in ANNOTATION_CAPTION_TYPE_NAME_LIST:
                        if per_caption_dict[
                                f'caption_{per_caption_type_name}_cn']:
                            caption_cn_type_count_dict[
                                per_caption_type_name] += 1

        reach_parquet_end = True

        # 核心对账之一: 实际遍历到的行数必须等于footer里数出来的行数，
        # 否则说明流式读的时候有batch被静默吞掉了
        if row_count != expected_row_count:
            error_message_list.append(
                f'{per_parquet_relative_path} row count not match {row_count} != {expected_row_count}'
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
            f'{per_parquet_relative_path} process image count not match: {extract_image_count} + {skip_image_count} + {not_save_image_count} != {row_count}'
        )

    # 核心对账之三: 实测每行都是完整样本对，所以有效样本对数必须等于行数;
    # 一旦少了就说明有行的图或caption_detail缺失，必须显式感知
    if valid_sample_pair_count + len(invalid_sample_pair_list) != row_count:
        error_message_list.append(
            f'{per_parquet_relative_path} sample pair count not match {valid_sample_pair_count} + {len(invalid_sample_pair_list)} != {row_count}'
        )
    if valid_sample_pair_count != row_count:
        error_message_list.append(
            f'{per_parquet_relative_path} valid sample pair count not match {valid_sample_pair_count} != {row_count}'
        )

    return {
        'parquet_relative_path':
        per_parquet_relative_path,
        'subset_name':
        per_subset_name,
        'parquet_name':
        per_parquet_group_name,
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
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_subset_name}/{per_parquet_group_name}.jsonl',
        'caption_type_count_dict':
        dict(caption_type_count_dict),
        'caption_cn_type_count_dict':
        dict(caption_cn_type_count_dict),
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
        # .cache里有下载缓存，直接在遍历时剪掉整棵子树，不要走进去
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

            parquet_group_list.append([
                per_match_result.group('prefix'),
                per_file_relative_dir,
                per_file_path,
            ])

    file_copy_pair_list = sorted(file_copy_pair_list, key=lambda x: x[0])
    parquet_group_list = sorted(parquet_group_list, key=lambda x: x[2])

    return file_copy_pair_list, parquet_group_list


def check_single_parquet_dir_on_disk(parquet_check_pair):
    """可选的二次对账: os.walk单个分片的图像目录，核对落盘图像数与标注条数"""
    per_parquet_relative_path, per_image_dir_path, per_annotation_path, per_expected_valid_sample_pair_count = parquet_check_pair

    error_message_list = []

    if not os.path.exists(per_image_dir_path):
        error_message_list.append(
            f'{per_parquet_relative_path} image dir not exist')

        return [per_parquet_relative_path, 0, 0, error_message_list]

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
            f'{per_parquet_relative_path} load annotation failed {e}')

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_parquet_relative_path} unknown suffix file num {unknown_suffix_file_count}'
        )
    if annotation_count != per_expected_valid_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_relative_path} on disk annotation count not match {annotation_count} != {per_expected_valid_sample_pair_count}'
        )
    if len(image_name_dict) != per_expected_valid_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_relative_path} on disk image count not match {len(image_name_dict)} != {per_expected_valid_sample_pair_count}'
        )
    if missing_image_count > 0:
        error_message_list.append(
            f'{per_parquet_relative_path} on disk missing image count {missing_image_count}'
        )

    return [
        per_parquet_relative_path,
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
            per_parquet_result['parquet_relative_path'],
            os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                         per_parquet_result['subset_name'],
                         per_parquet_result['parquet_name']),
            os.path.join(save_annotation_dir_path,
                         per_parquet_result['subset_name'],
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
                      expected_total_row_count):
    """汇总所有parquet的解包与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_row_count, total_valid_sample_pair_count = 0, 0
    total_extract_image_count, total_skip_image_count = 0, 0
    total_not_save_image_count, total_duplicate_sample_count = 0, 0
    subset_sample_pair_count_dict = collections.Counter()
    caption_type_count_dict = collections.Counter()
    caption_cn_type_count_dict = collections.Counter()
    image_shape_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    parquet_sample_pair_count_dict = {}
    all_invalid_sample_pair_list, all_warning_message_list = [], []
    error_message_list = []

    for per_parquet_result in parquet_result_list:
        per_parquet_relative_path = per_parquet_result['parquet_relative_path']

        total_row_count += per_parquet_result['row_count']
        total_valid_sample_pair_count += per_parquet_result[
            'valid_sample_pair_count']
        total_extract_image_count += per_parquet_result['extract_image_count']
        total_skip_image_count += per_parquet_result['skip_image_count']
        total_not_save_image_count += per_parquet_result[
            'not_save_image_count']
        total_duplicate_sample_count += per_parquet_result[
            'duplicate_sample_count']

        subset_sample_pair_count_dict[per_parquet_result[
            'subset_name']] += per_parquet_result['valid_sample_pair_count']
        caption_type_count_dict.update(
            per_parquet_result['caption_type_count_dict'])
        caption_cn_type_count_dict.update(
            per_parquet_result['caption_cn_type_count_dict'])
        image_shape_count_dict.update(
            per_parquet_result['image_shape_count_dict'])
        image_suffix_count_dict.update(
            per_parquet_result['image_suffix_count_dict'])
        parquet_sample_pair_count_dict[
            per_parquet_relative_path] = per_parquet_result[
                'valid_sample_pair_count']

        all_invalid_sample_pair_list.extend(
            per_parquet_result['invalid_sample_pair_list'])
        all_warning_message_list.extend([
            f'{per_parquet_relative_path} {per_warning_message}' for
            per_warning_message in per_parquet_result['warning_message_list']
        ])

        if len(per_parquet_result['error_message_list']) > 0:
            print('7777', per_parquet_relative_path,
                  per_parquet_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_parquet_relative_path} error num {len(per_parquet_result["error_message_list"])} {per_parquet_result["error_message_list"][:3]}'
            )

    print('3333', 'total row:', total_row_count, 'total valid sample pair:',
          total_valid_sample_pair_count, 'extract image:',
          total_extract_image_count, 'skip image:', total_skip_image_count,
          'not save image:', total_not_save_image_count, 'duplicate sample:',
          total_duplicate_sample_count, 'invalid sample pair:',
          len(all_invalid_sample_pair_list), 'warning:',
          len(all_warning_message_list))
    print('3333', 'subset sample pair:', dict(subset_sample_pair_count_dict))
    print('3333', 'caption type:', dict(caption_type_count_dict))
    print('3333', 'caption cn type:', dict(caption_cn_type_count_dict))
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
        'invalid_sample_pair_count':
        len(all_invalid_sample_pair_list),
        'warning_message_count':
        len(all_warning_message_list),
        'subset_sample_pair_count_dict':
        dict(subset_sample_pair_count_dict),
        'caption_type_count_dict':
        dict(caption_type_count_dict),
        'caption_cn_type_count_dict':
        dict(caption_cn_type_count_dict),
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
    if len(all_invalid_sample_pair_list) > 0:
        error_message_list.append(
            f'invalid sample pair count {len(all_invalid_sample_pair_list)}')
    if total_duplicate_sample_count > 0:
        error_message_list.append(
            f'duplicate sample count {total_duplicate_sample_count}')

    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        per_expected_row_count = EXPECTED_SUBSET_ROW_COUNT_DICT[
            per_subset_name]
        per_sample_pair_count = subset_sample_pair_count_dict.get(
            per_subset_name, 0)
        if per_sample_pair_count != per_expected_row_count:
            error_message_list.append(
                f'{per_subset_name} valid sample pair count not match {per_sample_pair_count} != {per_expected_row_count}'
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
              parquet_group_list[0][2])

    if len(parquet_group_list) == 0:
        raise Exception('no parquet file found')

    expected_total_row_count, per_parquet_row_count_dict, precheck_error_message_list = check_required_subset_complete(
        root_dataset_path, parquet_group_list)
    if len(precheck_error_message_list) > 0:
        # 数据集本身不完整就没必要跑几十小时解包
        raise Exception(
            f'check subset failed error num {len(precheck_error_message_list)} {precheck_error_message_list[:20]}'
        )

    expected_parquet_group_num = sum(EXPECTED_SUBSET_PARQUET_NUM_DICT.values())
    if len(parquet_group_list) != expected_parquet_group_num:
        raise Exception(
            f'parquet group num not match {len(parquet_group_list)} != {expected_parquet_group_num}'
        )

    # 每片行数实测除每子集最后一片是余数外都是5000，只做软校验(打印告警)，
    # 因为官方没承诺分片切分规格，不能拿来当硬性失败条件
    for per_parquet_group_name in sorted(per_parquet_row_count_dict.keys()):
        if per_parquet_row_count_dict[
                per_parquet_group_name] > EXPECTED_PARQUET_ROW_COUNT:
            print('2222', per_parquet_group_name, 'row count larger than',
                  EXPECTED_PARQUET_ROW_COUNT,
                  per_parquet_row_count_dict[per_parquet_group_name])

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

    parquet_result_list = []
    extract_func = partial(process_single_parquet_file,
                           save_dataset_path=save_dataset_path,
                           save_annotation_dir_path=save_annotation_dir_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_parquet_result in tqdm(pool.imap_unordered(
                extract_func, parquet_group_list),
                                       total=len(parquet_group_list)):
            parquet_result_list.append(per_parquet_result)

            print('2222', per_parquet_result['parquet_relative_path'], 'row:',
                  per_parquet_result['row_count'], 'valid sample pair:',
                  per_parquet_result['valid_sample_pair_count'],
                  'extract image:', per_parquet_result['extract_image_count'],
                  'skip image:', per_parquet_result['skip_image_count'],
                  'not save image:',
                  per_parquet_result['not_save_image_count'],
                  'duplicate sample:',
                  per_parquet_result['duplicate_sample_count'],
                  'invalid sample pair:',
                  len(per_parquet_result['invalid_sample_pair_list']))

    check_error_message_list = save_check_result(save_dataset_path,
                                                 parquet_result_list,
                                                 expected_total_row_count)

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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/FLUX-Reason-6M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
