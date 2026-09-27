import io
import os
import re
import json
import shutil
import collections

import pyarrow.parquet as pq

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: CrispEdit-2M(WeiChow/CrispEdit-2M, 论文EditMGT arXiv:2512.11715)
#
# 【数据集类型】纯图像编辑(instruction-based image editing)数据集，不是文生图数据集。
# README原文 task_categories: image-to-image，tags: image-editing/instruction-guided，
# 每行固定是"1张参考图(编辑前图input_img) + 1条英文编辑指令(instruction) +
# 1张编辑后图(output_img)"，没有第二张视觉条件图、没有mask、
# 也没有任何"只有caption+一张图"的纯生成样本，
# 所以下游只能走ti2i_dataset.py那条链路，不能当t2i(文生图)数据用。
# 抽样1825行逐行解码验证: input_img与output_img的字节md5**没有一对相同**，
# 969/1825对的输入输出分辨率还不一样，确实是真实编辑对而不是同图复制。
#
# 【root_dataset_path实测原始保存规格(共4.6T / 8971个parquet / 2286540行)】
# CrispEdit-2M/
# ├── data/             8971个 <type>_%05d.parquet  (4.6T)  -> 唯一有用数据
# ├── README.md         数据集说明(无用)
# ├── .gitattributes    git lfs配置(无用)
# └── .cache/           huggingface下载缓存，9047个文件，**残留66个*.incomplete**(无用)
#
# data/下按7个编辑任务族前缀分片(**文件名前缀 == 行内type列，8971/8971片完全一致**):
#   前缀                parquet数   实测总行数   README声称
#   add                   1213       309253       303K
#   background change     1091       278475       272K
#   color                 1984       506730       496K
#   motion change          128        32314        32K
#   remove                1388       353981       347K
#   replace               1567       399329       391K
#   style                 1600       406458       400K
#   合计                  8971      2286540      2241K
#
# 【必须显式感知的三个规格坑】
# 1) **分片编号不是全部连号: add缺 add_01211.parquet**(编号0..1213共1214个位置、实际1213个文件),
#    其余6个前缀都是0..N-1严格连号。
#    已用huggingface下载缓存里的仓库文件清单 .cache/huggingface/trees/*.json 对账:
#    清单里共8973个文件(8971个parquet + README.md + .gitattributes)，
#    **磁盘上全部存在且字节数100%一致，且清单里本来就没有add_01211**,
#    所以这是上游仓库自身的缺号、不是本地下载缺失，
#    写进 ALLOW_MISSING_PARQUET_INDEX_DICT 白名单放行;
#    **除这一个白名单外的任何缺号都必须硬失败中止**，否则就是静默少几千个样本对。
#    .cache里那66个*.incomplete只是缓存残渣，数据集本体是完整的。
# 2) **两个任务族名里带空格**(background change / motion change)。
#    空格进目录名与文件名会让后续shell/训练脚本各种踩坑，
#    所以落盘时统一把空格归一成下划线(background_change / motion_change),
#    **原始type原值与原始parquet文件名照样写进jsonl的type/parquet_name字段，溯源不丢**。
# 3) **不是每片都恰好256行**。README声称"256 items per file"，
#    但实测有175片不足256(最少的 motion change_00108 只有3行、remove_00157 只有5行),
#    所以绝不能把"每片256行"当硬性条件，只能逐片读footer的num_rows当ground truth。
#    另外每片row group数在1~15之间不等(单片最大约1.2GB)，必须iter_batches流式读。
#
# 【单行parquet的全部4列(8971片schema完全一致, SNAPPY, parquet-cpp-arrow 21.0.0)】
#   input_img    : struct<bytes: binary, path: string>
#                  .bytes = **参考图(编辑前图)原始字节，全库null=0**       -> 有用(参考图，必需)
#                  .path  = **全库2286540行全部为null**，恒空占位列        -> 无用(不落盘、不进标注)
#   instruction  : **英文编辑指令，全库null=0**，抽样51084行无空串,
#                  长度11~853                                              -> 有用(训练主文本，必需)
#   output_img   : struct<bytes: binary, path: string>
#                  .bytes = **编辑后图原始字节，全库null=0**               -> 有用(编辑后图，必需)
#                  .path  = 同上，恒空                                     -> 无用
#   type         : 7类编辑任务名，**全库null=0且逐片min==max==文件名前缀**  -> 有用(任务族采样/加权)
# 除这4列外该数据集**没有任何其他单样本属性**(无id/无宽高/无打分/无授权列),
# 所以样本唯一id只能由 <任务族>/<分片名>_<片内行号> 合成
# (分片名全局唯一 + 片内行号唯一 => 合成key全局唯一，天然不会重名，不需要重名隔离目录)。
#
# 【图像格式实测(逐行解码1825行)】
# add / background change / replace / style -> JPEG;
# color / motion change / remove            -> PNG;
# 全部RGB三通道，主流分辨率短边1024(1024x1024最多)，
# 所以落盘后缀必须按字节魔数逐张判定，不能统一写死.jpg。
#
# 【无用信息(一律不整理进训练目录)】
# .cache/(9047个文件，含66个*.incomplete) / .gitattributes / README.md /
# input_img.path与output_img.path(全null的占位列) /
# .DS_Store / CACHEDIR.TAG 这类目录元数据垃圾文件。
#
# 【本脚本的处理口径】
# - 解包前预检(硬失败，不过就不白跑几十小时):
#   根目录条目白名单(只允许data/，多出未知文件或未知目录立即上报)、
#   data/下必须全部是 <type>_%05d.parquet 且type属于7个白名单、
#   每前缀分片数与实测ground truth逐一比对、
#   分片编号连号校验(**只放行add缺1211这一个白名单缺号**)、
#   每片PAR1头尾魔数(O(1)读，拦下载截断)、
#   只读footer校验4列schema一致 + type统计量min==max==文件名前缀 +
#   input_img.bytes/output_img.bytes/instruction/type的null数必须为0 +
#   逐前缀行数与全局总行数2286540全部硬对账;
# - 并行单位 = 单个parquet(8971个任务，Pool(32)),
#   pq.iter_batches(batch_size=32)流式读，**绝不整片进内存**;
# - 图像落盘 unzip_images/<task>/<parquet>/<parquet>_%08d_input.jpg|png 与 _output.jpg|png,
#   **直接写原始字节，unzip阶段绝不引入二次编解码**，
#   resize/转格式/分辨率分桶留给preprocessing2的resave脚本;
# - 每张图写盘后**立刻校验落盘大小 == len(bytes)**(比只看存在性强，能挡住写半截/写0字节);
#   已存在且大小一致就计skip并跳过，保证脚本可以断点续跑;
# - 汇总标注落 unzip_annotations/<task>/<parquet>.jsonl，每行一个完整编辑样本对,
#   保留instruction/type全部有用属性，并补上落盘路径、样本key、行号、真实宽高;
# - 片内四方硬对账: 遍历行数 == footer num_rows、
#   参考图成员数 == 编辑后图成员数 == 行数、
#   extract+skip+not_save+fail == 两类图像成员数之和、
#   有效样本对 + 隔离样本对 == 行数;
# - 绝不静默丢样本对: 指令为空/参考图字节为空/编辑后图字节为空/写盘失败,
#   全部分门别类记进隔离清单并在汇总报告里上报(实测应恒为空，一旦非空必须显式感知);
# - 全局硬对账: 总行数 == 2286540、逐任务族行数 == 实测值、
#   有效样本对 == 2286540、图像成员总数 == 4573080，少一个立即抛异常;
# - 拷贝/解包/校验任一环出错都汇总后抛异常，不再静默跑过。
#
# 【跑之前务必确认目标盘扛得住】
# - EXTRACT_IMAGE_FILE_FLAG=True 时输出小文件数 **4573080张图**(约4.6T),
# - 只想先建索引可把 EXTRACT_IMAGE_FILE_FLAG 置False，
#   图像继续留在原parquet里，样本对信息一样完整。
# ==============================================================================

DATASET_TASK_TYPE = 'image_edit'

DATASET_LICENSE_NAME = 'cc-by-4.0'

PARQUET_FILE_NAME_PATTERN = re.compile(r'^(?P<prefix>.+)\.parquet$')

# 带分片编号的parquet名(add_00000.parquet / background change_00000.parquet),
# 用于分片完整性预检; task_name里允许有空格
PARQUET_SHARD_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>(?P<task_name>.+)_(?P<index>\d{5}))\.parquet$')

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(9047个文件，含66个*.incomplete)
# .gitattributes   git lfs配置
# README.md        数据集说明(授权信息已记进校验报告的dataset_license_name字段)
# .gitignore/.DS_Store/CACHEDIR.TAG  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    '.DS_Store',
    'CACHEDIR.TAG',
]

# 过滤掉无用信息后根目录只应该剩这一个数据目录
ROOT_DATA_DIR_NAME = 'data'

ROOT_DIR_NAME_LIST = [
    ROOT_DATA_DIR_NAME,
]

# 7个编辑任务族(== parquet文件名前缀 == 行内type列取值)
TASK_NAME_LIST = [
    'add',
    'background change',
    'color',
    'motion change',
    'remove',
    'replace',
    'style',
]

# 实测每任务族parquet分片数(合计8971)，数量不对说明下载不全
EXPECTED_TASK_PARQUET_NUM_DICT = {
    'add': 1213,
    'background change': 1091,
    'color': 1984,
    'motion change': 128,
    'remove': 1388,
    'replace': 1567,
    'style': 1600,
}

# 实测每任务族的最大分片编号。
# add比别的多一档(0..1213共1214个位置)是因为上游仓库自身缺了add_01211
EXPECTED_TASK_PARQUET_MAX_INDEX_DICT = {
    'add': 1213,
    'background change': 1090,
    'color': 1983,
    'motion change': 127,
    'remove': 1387,
    'replace': 1566,
    'style': 1599,
}

# **上游仓库自身就没有的分片编号白名单**(已用.cache/huggingface/trees仓库文件清单证实:
# 清单里8971个parquet磁盘上全部存在且字节数100%一致，且清单里本来就没有add_01211)。
# 只有这里列出来的编号允许缺，其余任何缺号都必须硬失败中止
ALLOW_MISSING_PARQUET_INDEX_DICT = {
    'add': [1211],
}

# 实测每任务族parquet footer行数总和(合计2286540)，直接当作完整性ground truth,
# 缺任务族/缺分片/少行都能拦住
EXPECTED_TASK_ROW_COUNT_DICT = {
    'add': 309253,
    'background change': 278475,
    'color': 506730,
    'motion change': 32314,
    'remove': 353981,
    'replace': 399329,
    'style': 406458,
}

EXPECTED_TOTAL_PARQUET_NUM = 8971

EXPECTED_TOTAL_ROW_COUNT = 2286540

# 实测每行的参考图/编辑后图/指令/任务名全部齐备(footer里四列null数全为0),
# 所以完整编辑样本对数 == 总行数，少一个都说明链路上丢了样本
EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT = 2286540

# 每个完整编辑对贡献2张图(参考图 + 编辑后图)
EXPECTED_TOTAL_IMAGE_COUNT = 4573080

# README声称每片256行，但实测有175片不足256(最少3行),
# 所以只软校验"不能超过256"(打印告警)，不足256是正常的
EXPECTED_PARQUET_ROW_COUNT = 256

# parquet里应该齐备的全部4个顶层列名，少列/多列说明上游数据规格变了，必须显式感知
PARQUET_COLUMN_NAME_LIST = [
    'input_img',
    'instruction',
    'output_img',
    'type',
]

# footer统计量是按叶子列组织的(struct会被展平成 <列名>.<字段名>)
PARQUET_LEAF_COLUMN_NAME_LIST = [
    'input_img.bytes',
    'input_img.path',
    'instruction',
    'output_img.bytes',
    'output_img.path',
    'type',
]

# 这几个叶子列的null数实测全库为0，非0就说明存在不完整样本对，必须硬失败
PARQUET_NOT_NULL_LEAF_COLUMN_NAME_LIST = [
    'input_img.bytes',
    'output_img.bytes',
    'instruction',
    'type',
]

# 这两个叶子列实测全库恒为null(恒空占位列)，不落盘也不进标注;
# 万一上游哪天填了值，只打印告警提醒，不判失败
PARQUET_ALL_NULL_LEAF_COLUMN_NAME_LIST = [
    'input_img.path',
    'output_img.path',
]

PARQUET_LEAF_TASK_NAME_COLUMN_NAME = 'type'

# 参考图(编辑前图)与编辑后图的struct列，这两列是唯一需要落盘成图像文件的列
PARQUET_INPUT_IMAGE_COLUMN_NAME = 'input_img'

PARQUET_OUTPUT_IMAGE_COLUMN_NAME = 'output_img'

PARQUET_IMAGE_STRUCT_BYTES_KEY_NAME = 'bytes'

# 训练主文本 = instruction(英文编辑指令)，为空则该样本对没有文本条件、不可训练，隔离上报
PARQUET_INSTRUCTION_COLUMN_NAME = 'instruction'

PARQUET_TASK_NAME_COLUMN_NAME = 'type'

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_INPUT_IMAGE_NAME_SUFFIX = '_input'

SAVE_OUTPUT_IMAGE_NAME_SUFFIX = '_output'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

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
# True : 和其他数据集脚本口径一致，4573080张图(约4.6T),
#        NAS上inode和元数据压力大，务必确认目标盘扛得住再跑;
# False: 只解析非图像列生成 unzip_annotations/*.jsonl 索引(一两小时即可跑完),
#        图像继续留在原parquet里，训练时按parquet顺序读，样本对信息一样是完整的。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解包后再os.walk一遍输出目录做二次对账。
# 默认False: 457万个小文件的os.walk在NAS上要跑非常久，而解包时已经做了
# "写盘后立刻校验落盘大小 == len(bytes)" + "两类图像成员数与行数四方对账"两道对账，
# 已经能保证每一行的图都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 是否解码图像header拿真实宽高写进标注。
# 默认True: 只解header不解像素(PIL的Image.open是惰性的)，代价可忽略;
# 该数据集parquet里**没有任何宽高列**，不解header下游就只能在训练时逐张打开图才能分桶,
# 所以这里顺手把宽高存进jsonl，省掉后续一次457万张图的全量扫盘。
PARSE_IMAGE_SHAPE_FLAG = True

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

# 单片最大约1.2GB且row group数1~15不等，必须小batch流式读，绝不整片进内存
PARQUET_ROW_BATCH_SIZE = 32


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


def get_image_bytes_suffix(per_image_bytes):
    """用图像字节的魔数推断后缀

    该数据集parquet里只存裸图像字节、path列恒为null，
    且实测按任务族分成JPEG(add/background change/replace/style)与
    PNG(color/motion change/remove)两种，所以必须逐张按魔数判定，不能写死.jpg。
    """
    for per_magic_bytes, per_magic_suffix in IMAGE_BYTES_MAGIC_SUFFIX_LIST:
        if per_image_bytes.startswith(per_magic_bytes):
            return per_magic_suffix

    return '.jpg'


def get_stripped_text_value(per_column_value):
    """文本列统一转成strip后的字符串，None/空白都当成缺失"""
    if not isinstance(per_column_value, str):
        return ''

    return per_column_value.strip()


def get_normalized_name(per_name):
    """把任务族名/分片名里的空格归一成下划线，用作落盘目录名与文件名前缀

    background change -> background_change, motion change -> motion_change。
    原始取值照样写进jsonl的type/parquet_name字段，溯源不丢。
    """
    per_name = per_name.replace('\\', '/').replace('/', '_')

    return '_'.join(per_name.split())


def get_parquet_task_name(per_parquet_group_name):
    """从分片名(<task_name>_%05d)里取任务族名，取不到时返回空串由调用方上报"""
    per_match_result = PARQUET_SHARD_FILE_NAME_PATTERN.match(
        f'{per_parquet_group_name}.parquet')
    if not per_match_result:
        return ''

    return per_match_result.group('task_name')


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

    实测8971个parquet的字节数与仓库清单100%一致，说明当前数据集是完整的。
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


def check_single_parquet_file_metadata(parquet_group):
    """只读单个parquet的footer拿行数/列名/type取值/各叶子列null数，不碰任何图像字节

    每片row group数1~15不等，这里按全部row group累加，规格变了也不会算错。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    per_task_name = get_parquet_task_name(per_parquet_group_name)

    per_metadata_dict = {
        'parquet_group_name': per_parquet_group_name,
        'parquet_relative_dir': per_parquet_relative_dir,
        'task_name': per_task_name,
        'row_count': 0,
        'row_group_num': 0,
        'column_name_list': [],
        'leaf_column_name_list': [],
        'task_name_value_list': [],
        'leaf_column_null_count_dict': {},
    }

    try:
        load_parquet_file = pq.ParquetFile(per_parquet_path)
        per_parquet_metadata = load_parquet_file.metadata
        per_column_name_list = list(load_parquet_file.schema_arrow.names)
        per_leaf_column_name_list = [
            per_parquet_metadata.schema.column(per_column_index).path
            for per_column_index in range(per_parquet_metadata.num_columns)
        ]
    except Exception as e:
        print('7777', per_parquet_group_name, e)

        return per_metadata_dict, [
            f'read parquet metadata failed {per_parquet_group_name} {e}',
        ]

    error_message_list = []

    per_metadata_dict['row_count'] = per_parquet_metadata.num_rows
    per_metadata_dict['row_group_num'] = per_parquet_metadata.num_row_groups
    per_metadata_dict['column_name_list'] = per_column_name_list
    per_metadata_dict['leaf_column_name_list'] = per_leaf_column_name_list

    if per_column_name_list != PARQUET_COLUMN_NAME_LIST:
        error_message_list.append(
            f'{per_parquet_group_name} column name not match {per_column_name_list}'
        )

        return per_metadata_dict, error_message_list

    if per_leaf_column_name_list != PARQUET_LEAF_COLUMN_NAME_LIST:
        error_message_list.append(
            f'{per_parquet_group_name} leaf column name not match {per_leaf_column_name_list}'
        )

        return per_metadata_dict, error_message_list

    per_task_name_value_dict = {}
    per_leaf_column_null_count_dict = collections.Counter()
    for per_row_group_index in range(per_parquet_metadata.num_row_groups):
        per_row_group_metadata = per_parquet_metadata.row_group(
            per_row_group_index)

        for per_column_index, per_leaf_column_name in enumerate(
                per_leaf_column_name_list):
            per_column_statistics = per_row_group_metadata.column(
                per_column_index).statistics
            if per_column_statistics is None:
                # footer里没有统计量就没法做O(1)预检，必须显式感知
                error_message_list.append(
                    f'{per_parquet_group_name} row group {per_row_group_index} column {per_leaf_column_name} statistics not exist'
                )
                continue

            per_leaf_column_null_count_dict[
                per_leaf_column_name] += per_column_statistics.null_count

            if per_leaf_column_name == PARQUET_LEAF_TASK_NAME_COLUMN_NAME and per_column_statistics.has_min_max:
                per_task_name_value_dict[per_column_statistics.min] = 1
                per_task_name_value_dict[per_column_statistics.max] = 1

    per_metadata_dict['task_name_value_list'] = sorted(
        per_task_name_value_dict.keys())
    per_metadata_dict['leaf_column_null_count_dict'] = dict(
        per_leaf_column_null_count_dict)

    return per_metadata_dict, error_message_list


def check_parquet_shard_complete(parquet_group_list):
    """解包前预检: 每任务族的分片数、分片名规格、分片编号连号，缺片直接中止

    只放行 ALLOW_MISSING_PARQUET_INDEX_DICT 里那几个"上游仓库自身就没有"的编号,
    其余任何缺号都是本地下载缺失，必须硬失败。
    """
    error_message_list = []

    task_shard_index_dict = {}
    for per_parquet_group_name, per_parquet_relative_dir, per_parquet_path in parquet_group_list:
        per_parquet_name = os.path.basename(per_parquet_path)

        if per_parquet_relative_dir.replace('\\', '/') != ROOT_DATA_DIR_NAME:
            # data/下不应该再有下一层目录
            error_message_list.append(
                f'unknown parquet relative dir {per_parquet_relative_dir}')
            continue

        per_match_result = PARQUET_SHARD_FILE_NAME_PATTERN.match(
            per_parquet_name)
        if not per_match_result:
            error_message_list.append(
                f'unknown parquet name {per_parquet_name}')
            continue

        per_task_name = per_match_result.group('task_name')
        if per_task_name not in TASK_NAME_LIST:
            error_message_list.append(
                f'unknown task name in parquet name {per_parquet_name}')
            continue

        per_shard_index = int(per_match_result.group('index'))
        per_shard_index_dict = task_shard_index_dict.setdefault(
            per_task_name, {})
        if per_shard_index in per_shard_index_dict:
            # 同一个编号出现两次说明文件名规格变了，按名解析会漏统计
            error_message_list.append(
                f'duplicate parquet index {per_parquet_name}')
            continue

        per_shard_index_dict[per_shard_index] = per_parquet_group_name

    for per_task_name in TASK_NAME_LIST:
        per_shard_index_dict = task_shard_index_dict.get(per_task_name, {})
        per_expected_parquet_num = EXPECTED_TASK_PARQUET_NUM_DICT[
            per_task_name]
        per_expected_max_index = EXPECTED_TASK_PARQUET_MAX_INDEX_DICT[
            per_task_name]
        per_allow_missing_index_list = ALLOW_MISSING_PARQUET_INDEX_DICT.get(
            per_task_name, [])

        print('1111', per_task_name, 'parquet:', len(per_shard_index_dict),
              'expected parquet:', per_expected_parquet_num,
              'expected max index:', per_expected_max_index,
              'allow missing index:', per_allow_missing_index_list)

        if len(per_shard_index_dict) != per_expected_parquet_num:
            error_message_list.append(
                f'{per_task_name} parquet num not match {len(per_shard_index_dict)} != {per_expected_parquet_num}'
            )

        # 分片编号必须覆盖 0..max_index 里除白名单缺号外的全部编号,
        # 多一个少一个都说明上游规格变了或本地没下全
        per_expected_index_set = set(range(
            0, per_expected_max_index + 1)) - set(per_allow_missing_index_list)
        per_missing_index_list = sorted(per_expected_index_set -
                                        set(per_shard_index_dict.keys()))
        per_unexpected_index_list = sorted(
            set(per_shard_index_dict.keys()) - per_expected_index_set)
        if len(per_missing_index_list) > 0:
            error_message_list.append(
                f'{per_task_name} parquet index not continuous, missing index {per_missing_index_list[:10]}'
            )
        if len(per_unexpected_index_list) > 0:
            error_message_list.append(
                f'{per_task_name} parquet index unexpected {per_unexpected_index_list[:10]}'
            )

    for per_task_name in sorted(task_shard_index_dict.keys()):
        if per_task_name not in EXPECTED_TASK_PARQUET_NUM_DICT:
            error_message_list.append(
                f'unknown task name in parquet group {per_task_name}')

    return error_message_list


def check_parquet_metadata_complete(parquet_group_list):
    """只读footer按任务族与实测ground truth逐项硬对账

    对账项: 4列schema与6个叶子列一致、type取值 == 文件名前缀、
    参考图/编辑后图/指令/任务名四个叶子列的null数必须为0、
    逐任务族行数与全局总行数全部与写死的实测值相等。
    """
    error_message_list = []

    total_row_count = 0
    per_parquet_metadata_dict = {}
    task_row_count_dict = collections.Counter()
    task_parquet_num_dict = collections.Counter()
    leaf_column_null_count_dict = collections.Counter()

    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_file_metadata, parquet_group_list),
                                     total=len(parquet_group_list)):
            per_metadata_dict, per_error_message_list = per_check_result
            error_message_list.extend(per_error_message_list)

            per_parquet_group_name = per_metadata_dict['parquet_group_name']
            per_task_name = per_metadata_dict['task_name']
            per_row_count = per_metadata_dict['row_count']

            per_parquet_metadata_dict[
                per_parquet_group_name] = per_metadata_dict
            task_row_count_dict[per_task_name] += per_row_count
            task_parquet_num_dict[per_task_name] += 1
            total_row_count += per_row_count
            leaf_column_null_count_dict.update(
                per_metadata_dict['leaf_column_null_count_dict'])

            if per_metadata_dict['row_group_num'] <= 0:
                error_message_list.append(
                    f'{per_parquet_group_name} row group num {per_metadata_dict["row_group_num"]}'
                )

            if per_row_count > EXPECTED_PARQUET_ROW_COUNT:
                # 实测每片最多256行(不足256是正常的)，超了只打印告警
                print('2222', per_parquet_group_name, 'row count larger than',
                      EXPECTED_PARQUET_ROW_COUNT, per_row_count)

            if per_metadata_dict['task_name_value_list'] != [per_task_name]:
                # type列必须与文件名前缀完全一致，不一致说明分片内容放错了
                error_message_list.append(
                    f'{per_parquet_group_name} task name value not match {per_metadata_dict["task_name_value_list"]} != [{per_task_name}]'
                )

            for per_leaf_column_name in PARQUET_NOT_NULL_LEAF_COLUMN_NAME_LIST:
                per_null_count = per_metadata_dict[
                    'leaf_column_null_count_dict'].get(per_leaf_column_name, 0)
                if per_null_count != 0:
                    # 参考图/编辑后图/指令/任务名任一为null都是不完整样本对
                    error_message_list.append(
                        f'{per_parquet_group_name} {per_leaf_column_name} null count {per_null_count}'
                    )

    for per_task_name in TASK_NAME_LIST:
        per_expected_row_count = EXPECTED_TASK_ROW_COUNT_DICT[per_task_name]
        per_row_count = task_row_count_dict.get(per_task_name, 0)

        print('1111', per_task_name, 'parquet:',
              task_parquet_num_dict.get(per_task_name, 0), 'row:',
              per_row_count, 'expected row:', per_expected_row_count)

        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_task_name} row count not match {per_row_count} != {per_expected_row_count}'
            )

    for per_leaf_column_name in PARQUET_ALL_NULL_LEAF_COLUMN_NAME_LIST:
        per_null_count = leaf_column_null_count_dict.get(
            per_leaf_column_name, 0)
        if per_null_count != total_row_count:
            # 实测path列全库恒为null，万一上游填了值只打印告警(多出来的信息不影响样本对完整性)
            print('2222', per_leaf_column_name, 'null count', per_null_count,
                  '!= total row count', total_row_count)

    print('1111', 'total row:', total_row_count, 'expected total row:',
          EXPECTED_TOTAL_ROW_COUNT, 'parquet:', len(per_parquet_metadata_dict))
    print('1111', 'leaf column null count:', dict(leaf_column_null_count_dict))

    if total_row_count != EXPECTED_TOTAL_ROW_COUNT:
        error_message_list.append(
            f'total row count not match {total_row_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )

    return total_row_count, per_parquet_metadata_dict, error_message_list


def check_required_subset_complete(root_dataset_path, parquet_group_list):
    """解包前预检: 根目录条目白名单、data目录、parquet数量与连号、PAR1魔数、footer对账

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

        if os.path.isdir(os.path.join(root_dataset_path, per_name)):
            if per_name not in ROOT_DIR_NAME_LIST:
                error_message_list.append(
                    f'unknown dir in root dir {per_name}')
            continue

        # 根目录下出现新的非跳过文件必须显式上报，否则会被静默漏处理
        error_message_list.append(f'unknown file in root dir {per_name}')

    if not os.path.exists(os.path.join(root_dataset_path, ROOT_DATA_DIR_NAME)):
        error_message_list.append(f'data dir not exist {ROOT_DATA_DIR_NAME}')

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

    print('1111', 'check parquet metadata:', len(parquet_group_list))
    total_row_count, per_parquet_metadata_dict, metadata_error_message_list = check_parquet_metadata_complete(
        parquet_group_list)
    error_message_list.extend(metadata_error_message_list)

    return total_row_count, per_parquet_metadata_dict, error_message_list


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

    单张图写盘异常不能让整片parquet的循环中断，否则该片后面几百行既不解图也不进标注。
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


def get_single_row_image_bytes(per_row_dict, per_image_column_name):
    """取出一行里某个图像struct列的原始字节，None/空字节都当成缺失

    该列是 struct<bytes: binary, path: string>，to_pylist后是dict;
    path实测恒为null(无用)，只取bytes。
    """
    per_image_struct = per_row_dict.get(per_image_column_name, None)
    if not isinstance(per_image_struct, dict):
        return None

    per_image_bytes = per_image_struct.get(PARQUET_IMAGE_STRUCT_BYTES_KEY_NAME,
                                           None)
    if not isinstance(per_image_bytes, bytes) or len(per_image_bytes) == 0:
        return None

    return per_image_bytes


def get_single_sample_pair_annotation(
        per_sample_key, per_task_name, per_normalized_task_name,
        per_parquet_group_name, per_row_index, per_instruction,
        per_input_image_relative_path, per_output_image_relative_path,
        per_input_image_shape, per_output_image_shape, per_input_image_suffix,
        per_output_image_suffix):
    """拼一条完整编辑样本对标注

    该数据集parquet只有4列且两列是图像字节、两列path恒空，
    所以有用属性就是instruction与type，其余全是本脚本补的落盘路径与定位信息,
    下游可以直接按行取样本，不需要为了拿指令去扫457万个小文件。
    """
    per_save_annotation = {
        'dataset_task_type': DATASET_TASK_TYPE,
        'sample_key': per_sample_key,
        'task_name': per_normalized_task_name,
        'type': per_task_name,
        'parquet_name': per_parquet_group_name,
        'row_index': per_row_index,
        'instruction': per_instruction,
        'reference_image_path_list': [per_input_image_relative_path],
        'reference_image_num': 1,
        'edited_image_path': per_output_image_relative_path,
        'input_image_shape': per_input_image_shape,
        'edited_image_shape': per_output_image_shape,
        'input_image_suffix': per_input_image_suffix,
        'edited_image_suffix': per_output_image_suffix,
    }

    return per_save_annotation


def process_single_parquet_file(parquet_group, save_dataset_path,
                                save_annotation_dir_path):
    """流式解开单个parquet，把内嵌图像字节写成图像文件、非图像列写成jsonl汇总标注

    落盘结构(任务族名与分片名里的空格统一归一成下划线):
      unzip_images/<task>/<parquet>/<parquet>_%08d_input.jpg|png
      unzip_images/<task>/<parquet>/<parquet>_%08d_output.jpg|png
      unzip_annotations/<task>/<parquet>.jsonl

    该数据集没有id列，样本key由 <task>/<parquet>_<片内行号> 合成:
    分片名全局唯一 + 片内行号唯一 => 合成key全局唯一，天然不会重名，
    所以不需要重名隔离目录。

    parquet按iter_batches流式读，绝不整片进内存(单片最大约1.2GB)。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    per_task_name = get_parquet_task_name(per_parquet_group_name)
    per_normalized_task_name = get_normalized_name(per_task_name)
    per_normalized_parquet_name = get_normalized_name(per_parquet_group_name)

    save_image_dir_path = os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                                       per_normalized_task_name,
                                       per_normalized_parquet_name)
    save_image_relative_dir = f'{SAVE_IMAGE_DIR_NAME}/{per_normalized_task_name}/{per_normalized_parquet_name}'
    save_annotation_path = os.path.join(
        save_annotation_dir_path, per_normalized_task_name,
        f'{per_normalized_parquet_name}.jsonl')

    if EXTRACT_IMAGE_FILE_FLAG:
        os.makedirs(save_image_dir_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)

    row_count, valid_sample_pair_count = 0, 0
    input_image_member_count, output_image_member_count = 0, 0
    extract_image_count, skip_image_count = 0, 0
    not_save_image_count, save_image_fail_count = 0, 0
    task_name_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    output_image_shape_count_dict = collections.Counter()
    invalid_sample_pair_list, warning_message_list = [], []
    error_message_list = []
    reach_parquet_end = False

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

                    per_image_name_prefix = f'{per_normalized_parquet_name}_{per_row_index:08d}'
                    per_sample_key = f'{per_normalized_task_name}/{per_image_name_prefix}'

                    per_invalid_reason_list = []

                    per_instruction = get_stripped_text_value(
                        per_row_dict.get(PARQUET_INSTRUCTION_COLUMN_NAME,
                                         None))
                    if not per_instruction:
                        # 图像编辑样本对必须有编辑指令，没有指令的图不可训练
                        per_invalid_reason_list.append('empty instruction')

                    per_row_task_name = get_stripped_text_value(
                        per_row_dict.get(PARQUET_TASK_NAME_COLUMN_NAME, None))
                    if per_row_task_name != per_task_name:
                        # 行内type与文件名前缀不一致说明分片内容放错了，必须显式感知
                        warning_message_list.append(
                            f'row {per_row_index} task name not match {per_row_task_name} != {per_task_name}'
                        )

                    per_input_image_bytes = get_single_row_image_bytes(
                        per_row_dict, PARQUET_INPUT_IMAGE_COLUMN_NAME)
                    per_output_image_bytes = get_single_row_image_bytes(
                        per_row_dict, PARQUET_OUTPUT_IMAGE_COLUMN_NAME)

                    if per_input_image_bytes is None:
                        # 参考图实测全库null=0，恒存在，为空就是不完整样本对
                        per_invalid_reason_list.append(
                            'empty input image bytes')
                    if per_output_image_bytes is None:
                        per_invalid_reason_list.append(
                            'empty output image bytes')

                    per_input_image_relative_path = ''
                    per_output_image_relative_path = ''
                    per_input_image_shape = [0, 0]
                    per_output_image_shape = [0, 0]
                    per_input_image_suffix = ''
                    per_output_image_suffix = ''

                    for per_image_bytes, per_image_name_suffix in [
                        [
                            per_input_image_bytes,
                            SAVE_INPUT_IMAGE_NAME_SUFFIX,
                        ],
                        [
                            per_output_image_bytes,
                            SAVE_OUTPUT_IMAGE_NAME_SUFFIX,
                        ],
                    ]:
                        if per_image_bytes is None:
                            continue

                        per_is_input_image = per_image_name_suffix == SAVE_INPUT_IMAGE_NAME_SUFFIX
                        if per_is_input_image:
                            input_image_member_count += 1
                        else:
                            output_image_member_count += 1

                        per_image_suffix = get_image_bytes_suffix(
                            per_image_bytes)
                        image_suffix_count_dict[per_image_suffix] += 1

                        per_save_image_name = f'{per_image_name_prefix}{per_image_name_suffix}{per_image_suffix}'

                        per_image_relative_path = f'{save_image_relative_dir}/{per_save_image_name}'

                        per_image_shape, per_image_shape_error_message = get_image_shape(
                            per_image_bytes)
                        if per_image_shape_error_message:
                            warning_message_list.append(
                                f'row {per_row_index} {per_save_image_name} {per_image_shape_error_message}'
                            )

                        if not EXTRACT_IMAGE_FILE_FLAG:
                            # 只建索引模式: 图像继续留在原parquet里，样本对信息一样完整
                            not_save_image_count += 1
                        else:
                            per_write_flag, per_skip_flag, per_save_error_message = save_single_image_bytes(
                                os.path.join(save_image_dir_path,
                                             per_save_image_name),
                                per_image_bytes)
                            if per_save_error_message:
                                save_image_fail_count += 1
                                error_message_list.append(
                                    f'row {per_row_index} {per_save_error_message}'
                                )
                                per_invalid_reason_list.append(
                                    f'save {per_image_name_suffix.strip("_")} image failed'
                                )
                                continue

                            if per_write_flag:
                                extract_image_count += 1
                            elif per_skip_flag:
                                skip_image_count += 1

                        if per_is_input_image:
                            per_input_image_relative_path = per_image_relative_path
                            per_input_image_shape = per_image_shape
                            per_input_image_suffix = per_image_suffix
                        else:
                            per_output_image_relative_path = per_image_relative_path
                            per_output_image_shape = per_image_shape
                            per_output_image_suffix = per_image_suffix

                    if not per_input_image_relative_path:
                        per_invalid_reason_list.append('missing input image')
                    if not per_output_image_relative_path:
                        per_invalid_reason_list.append('missing output image')

                    if len(per_invalid_reason_list) > 0:
                        # 信息不完整的样本对不写进有效标注，但必须留痕，不能静默消失
                        invalid_sample_pair_list.append({
                            'sample_key':
                            per_sample_key,
                            'task_name':
                            per_normalized_task_name,
                            'parquet_name':
                            per_parquet_group_name,
                            'row_index':
                            per_row_index,
                            'invalid_reason':
                            ','.join(per_invalid_reason_list),
                        })
                        continue

                    per_save_annotation = get_single_sample_pair_annotation(
                        per_sample_key, per_task_name,
                        per_normalized_task_name, per_parquet_group_name,
                        per_row_index, per_instruction,
                        per_input_image_relative_path,
                        per_output_image_relative_path, per_input_image_shape,
                        per_output_image_shape, per_input_image_suffix,
                        per_output_image_suffix)

                    save_annotation_file.write(
                        f'{json.dumps(per_save_annotation, ensure_ascii=False)}\n'
                    )
                    valid_sample_pair_count += 1

                    task_name_count_dict[per_row_task_name] += 1
                    if per_output_image_shape[0] > 0:
                        output_image_shape_count_dict[
                            f'{per_output_image_shape[0]}x{per_output_image_shape[1]}'] += 1

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

    # 核心对账之二: 每一行都恰好有一张参考图和一张编辑后图
    if input_image_member_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} input image member count not match {input_image_member_count} != {row_count}'
        )
    if output_image_member_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} output image member count not match {output_image_member_count} != {row_count}'
        )

    # 核心对账之三: 每张图像成员都必须有明确归属(新写/跳过/不落盘/写失败)
    if extract_image_count + skip_image_count + not_save_image_count + save_image_fail_count != input_image_member_count + output_image_member_count:
        error_message_list.append(
            f'{per_parquet_group_name} process image count not match: {extract_image_count} + {skip_image_count} + {not_save_image_count} + {save_image_fail_count} != {input_image_member_count} + {output_image_member_count}'
        )
    if save_image_fail_count > 0:
        error_message_list.append(
            f'{per_parquet_group_name} save image fail count {save_image_fail_count}'
        )

    # 核心对账之四: 每一行都必须有归属，要么是完整编辑对、要么进隔离清单，
    # 一条都不会凭空消失
    if valid_sample_pair_count + len(invalid_sample_pair_list) != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} sample pair count not match {valid_sample_pair_count} + {len(invalid_sample_pair_list)} != {row_count}'
        )
    # 实测每行的指令与两张图都齐备，所以隔离清单应该恒为空，一旦非空必须显式感知
    if len(invalid_sample_pair_list) > 0:
        error_message_list.append(
            f'{per_parquet_group_name} invalid sample pair count {len(invalid_sample_pair_list)}'
        )
    if valid_sample_pair_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} valid sample pair count not match {valid_sample_pair_count} != {row_count}'
        )

    return {
        'parquet_group_name':
        per_parquet_group_name,
        'task_name':
        per_task_name,
        'normalized_task_name':
        per_normalized_task_name,
        'normalized_parquet_name':
        per_normalized_parquet_name,
        'row_count':
        row_count,
        'valid_sample_pair_count':
        valid_sample_pair_count,
        'input_image_member_count':
        input_image_member_count,
        'output_image_member_count':
        output_image_member_count,
        'extract_image_count':
        extract_image_count,
        'skip_image_count':
        skip_image_count,
        'not_save_image_count':
        not_save_image_count,
        'save_image_fail_count':
        save_image_fail_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_normalized_task_name}/{per_normalized_parquet_name}.jsonl',
        'task_name_count_dict':
        dict(task_name_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'output_image_shape_count_dict':
        dict(output_image_shape_count_dict),
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
        # .cache里有9047个下载缓存文件(含66个*.incomplete)，
        # 直接在遍历时剪掉整棵子树，不要走进去
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

                per_image_path_list = list(
                    per_annotation.get('reference_image_path_list', []))
                if per_annotation.get('edited_image_path', ''):
                    per_image_path_list.append(
                        per_annotation['edited_image_path'])

                for per_image_path in per_image_path_list:
                    if os.path.basename(per_image_path) not in image_name_dict:
                        missing_image_count += 1
    except Exception as e:
        error_message_list.append(
            f'{per_parquet_group_name} load annotation failed {per_annotation_path} {e}'
        )

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_parquet_group_name} unknown suffix file num {unknown_suffix_file_count}'
        )
    if annotation_count != per_expected_valid_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_group_name} on disk annotation count not match {annotation_count} != {per_expected_valid_sample_pair_count}'
        )
    # 每个完整编辑对贡献2张图(参考图 + 编辑后图)
    per_expected_image_count = per_expected_valid_sample_pair_count * 2
    if len(image_name_dict) != per_expected_image_count:
        error_message_list.append(
            f'{per_parquet_group_name} on disk image count not match {len(image_name_dict)} != {per_expected_image_count}'
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
            per_parquet_result['parquet_group_name'],
            os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                         per_parquet_result['normalized_task_name'],
                         per_parquet_result['normalized_parquet_name']),
            os.path.join(
                save_annotation_dir_path,
                per_parquet_result['normalized_task_name'],
                f'{per_parquet_result["normalized_parquet_name"]}.jsonl'),
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
    total_input_image_member_count, total_output_image_member_count = 0, 0
    total_extract_image_count, total_skip_image_count = 0, 0
    total_not_save_image_count, total_save_image_fail_count = 0, 0
    task_row_count_dict = collections.Counter()
    task_sample_pair_count_dict = collections.Counter()
    task_name_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    output_image_shape_count_dict = collections.Counter()
    parquet_sample_pair_count_dict = {}
    all_invalid_sample_pair_list, all_warning_message_list = [], []
    error_message_list = []

    for per_parquet_result in parquet_result_list:
        per_parquet_group_name = per_parquet_result['parquet_group_name']
        per_task_name = per_parquet_result['task_name']

        total_row_count += per_parquet_result['row_count']
        total_valid_sample_pair_count += per_parquet_result[
            'valid_sample_pair_count']
        total_input_image_member_count += per_parquet_result[
            'input_image_member_count']
        total_output_image_member_count += per_parquet_result[
            'output_image_member_count']
        total_extract_image_count += per_parquet_result['extract_image_count']
        total_skip_image_count += per_parquet_result['skip_image_count']
        total_not_save_image_count += per_parquet_result[
            'not_save_image_count']
        total_save_image_fail_count += per_parquet_result[
            'save_image_fail_count']

        task_row_count_dict[per_task_name] += per_parquet_result['row_count']
        task_sample_pair_count_dict[per_task_name] += per_parquet_result[
            'valid_sample_pair_count']
        task_name_count_dict.update(per_parquet_result['task_name_count_dict'])
        image_suffix_count_dict.update(
            per_parquet_result['image_suffix_count_dict'])
        output_image_shape_count_dict.update(
            per_parquet_result['output_image_shape_count_dict'])
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
          total_valid_sample_pair_count, 'input image member:',
          total_input_image_member_count, 'output image member:',
          total_output_image_member_count, 'extract image:',
          total_extract_image_count, 'skip image:', total_skip_image_count,
          'not save image:', total_not_save_image_count, 'save image fail:',
          total_save_image_fail_count, 'invalid sample pair:',
          len(all_invalid_sample_pair_list), 'warning:',
          len(all_warning_message_list))
    print('3333', 'task row:', dict(task_row_count_dict))
    print('3333', 'task sample pair:', dict(task_sample_pair_count_dict))
    print('3333', 'row type value:', dict(task_name_count_dict))
    print('3333', 'image suffix:', dict(image_suffix_count_dict))
    print('3333', 'output image shape top10:',
          dict(output_image_shape_count_dict.most_common(10)))

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'dataset_license_name':
        DATASET_LICENSE_NAME,
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
        'total_input_image_member_count':
        total_input_image_member_count,
        'total_output_image_member_count':
        total_output_image_member_count,
        'total_extract_image_count':
        total_extract_image_count,
        'total_skip_image_count':
        total_skip_image_count,
        'total_not_save_image_count':
        total_not_save_image_count,
        'total_save_image_fail_count':
        total_save_image_fail_count,
        'invalid_sample_pair_count':
        len(all_invalid_sample_pair_list),
        'warning_message_count':
        len(all_warning_message_list),
        'allow_missing_parquet_index_dict':
        ALLOW_MISSING_PARQUET_INDEX_DICT,
        'task_row_count_dict':
        dict(task_row_count_dict),
        'task_sample_pair_count_dict':
        dict(task_sample_pair_count_dict),
        'row_type_value_count_dict':
        dict(task_name_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'output_image_shape_count_dict':
        dict(output_image_shape_count_dict),
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

    # 全量硬对账: 每一行都必须有归属，完整编辑对数必须等于预检时从footer数出来的
    # 实测ground truth，少一个都说明有样本对在"读parquet->写图->写jsonl"这条链路上消失了
    if total_valid_sample_pair_count == 0:
        error_message_list.append('no valid sample pair found')
    if total_row_count != expected_total_row_count:
        error_message_list.append(
            f'total row count not match {total_row_count} != {expected_total_row_count}'
        )
    if total_row_count != EXPECTED_TOTAL_ROW_COUNT:
        error_message_list.append(
            f'total row count not match expected {total_row_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )
    if total_valid_sample_pair_count != EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} != {EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT}'
        )
    if total_valid_sample_pair_count + len(
            all_invalid_sample_pair_list) != total_row_count:
        error_message_list.append(
            f'total sample pair count not match {total_valid_sample_pair_count} + {len(all_invalid_sample_pair_list)} != {total_row_count}'
        )
    if total_input_image_member_count != total_row_count:
        error_message_list.append(
            f'total input image member count not match {total_input_image_member_count} != {total_row_count}'
        )
    if total_output_image_member_count != total_row_count:
        error_message_list.append(
            f'total output image member count not match {total_output_image_member_count} != {total_row_count}'
        )
    if total_input_image_member_count + total_output_image_member_count != EXPECTED_TOTAL_IMAGE_COUNT:
        error_message_list.append(
            f'total image member count not match {total_input_image_member_count} + {total_output_image_member_count} != {EXPECTED_TOTAL_IMAGE_COUNT}'
        )
    if total_extract_image_count + total_skip_image_count + total_not_save_image_count + total_save_image_fail_count != total_input_image_member_count + total_output_image_member_count:
        error_message_list.append(
            f'total process image count not match {total_extract_image_count} + {total_skip_image_count} + {total_not_save_image_count} + {total_save_image_fail_count} != {total_input_image_member_count} + {total_output_image_member_count}'
        )
    if total_save_image_fail_count > 0:
        error_message_list.append(
            f'total save image fail count {total_save_image_fail_count}')
    if len(all_invalid_sample_pair_list) > 0:
        error_message_list.append(
            f'invalid sample pair count {len(all_invalid_sample_pair_list)}')

    for per_task_name in TASK_NAME_LIST:
        per_expected_row_count = EXPECTED_TASK_ROW_COUNT_DICT[per_task_name]
        per_row_count = task_row_count_dict.get(per_task_name, 0)
        per_valid_sample_pair_count = task_sample_pair_count_dict.get(
            per_task_name, 0)

        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_task_name} row count not match {per_row_count} != {per_expected_row_count}'
            )
        if per_valid_sample_pair_count != per_expected_row_count:
            error_message_list.append(
                f'{per_task_name} valid sample pair count not match {per_valid_sample_pair_count} != {per_expected_row_count}'
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

    expected_total_row_count, _, precheck_error_message_list = check_required_subset_complete(
        root_dataset_path, parquet_group_list)
    if len(precheck_error_message_list) > 0:
        # 数据集本身不完整就没必要跑几十小时解包
        raise Exception(
            f'check subset failed error num {len(precheck_error_message_list)} {precheck_error_message_list[:20]}'
        )

    if len(parquet_group_list) != EXPECTED_TOTAL_PARQUET_NUM:
        raise Exception(
            f'parquet group num not match {len(parquet_group_list)} != {EXPECTED_TOTAL_PARQUET_NUM}'
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

    parquet_result_list = []
    extract_func = partial(process_single_parquet_file,
                           save_dataset_path=save_dataset_path,
                           save_annotation_dir_path=save_annotation_dir_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_parquet_result in tqdm(pool.imap_unordered(
                extract_func, parquet_group_list),
                                       total=len(parquet_group_list)):
            parquet_result_list.append(per_parquet_result)

            print('2222', per_parquet_result['parquet_group_name'], 'row:',
                  per_parquet_result['row_count'], 'valid sample pair:',
                  per_parquet_result['valid_sample_pair_count'],
                  'extract image:', per_parquet_result['extract_image_count'],
                  'skip image:', per_parquet_result['skip_image_count'],
                  'not save image:',
                  per_parquet_result['not_save_image_count'],
                  'save image fail:',
                  per_parquet_result['save_image_fail_count'],
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/CrispEdit-2M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
