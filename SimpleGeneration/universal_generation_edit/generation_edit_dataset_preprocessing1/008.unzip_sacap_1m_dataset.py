import os
import re
import json
import shutil

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

import pyarrow.parquet as pq

# ==============================================================================
# 数据集: SACap-1M(Seg2Any, 0xLDF)
#
# 【数据集类型】纯文生图(text-to-image)数据集，不是图像编辑数据集。
# 每个样本对只有"一张目标图 + 一条全图caption"，没有任何参考图/输入图/编辑指令，
# 所以下游只能用t2i_dataset.py那条链路，不能当ti2i(编辑)数据用。
# 注意: 原始数据集本身是为"分割mask到图像生成(segmentation-mask-to-image)"准备的，
# 除全图caption外还给了实例级mask(anno_id)和区域caption，但本脚本按纯文生图口径整理，
# **实例mask与区域caption整块丢弃**(见下面【无用信息】)。
#
# 【root_dataset_path实测原始保存规格】
# SACap-1M/                                                          (共415M)
# ├── annotations/
# │   └── anno_train.parquet   395MB，1个row group，1029250行         -> 唯一有用文件
# ├── cache/train/
# │   └── 512H_512W-group_bucket.parquet  1029250行，只有cond_seq_len/txt_seq_len两列，
# │                            是Seg2Any在512x512分辨率+它自己那套文本编码器下预算的
# │                            序列长度缓存(实测cond 120~848 / txt 51~731)，
# │                            和本仓库的分辨率分桶与文本编码器都不匹配，可现算(无用)
# ├── data_samples.png         16.7MB，README里的示意图(无用)
# ├── mask_distribution.png    每图mask数分布图，README里的示意图(无用)
# ├── README.md                数据集说明(无用)
# ├── .gitattributes           git lfs配置(无用)
# └── .cache/                  huggingface下载缓存，21个文件，无残留*.incomplete(无用)
#
# 【anno_train.parquet的全部4列(实测1029250行全部齐备)】
#   imagename     : 形如sa_3865365.jpg，**1029250条全部唯一**            -> 有用(样本唯一id/定位图像)
#   image_group   : 形如sa_000345，共1000组，每组1005~1079条            -> 有用(定位图像所在目录/并行分片)
#   caption       : 全图caption(平均58.6词)，**0条为空**                -> 有用(唯一的训练文本，必需)
#   segments_info : list<struct<anno_id:int64, caption:string>>,
#                   实测合计5882281个实例mask，每图1~20个，区域caption 0条为空
#                                                                       -> 无用(按纯文生图口径丢弃)
#
# 【depend_dataset_path: 本数据集不含任何图像，图像全部来自SA-1B】
# SACap-1M只发布dense标注，目标图必须从已解压好的SA-1B里取。
# sa_1b/                                                            (共约11T)
# └── sa_000000 .. sa_000999   1000个目录，编号连续无缺号，
#                             每个目录**只有jpg和json、没有任何其他文件**，
#                             合计 11185362 jpg + 11185362 json，同名配对11185362、无孤立文件
#   sa_<image_id>.jpg   目标图(平均约1.0MB)                            -> 有用(生成后图，必需)
#   sa_<image_id>.json  {"image":{image_id,width,height,file_name},
#                        "annotations":[{id,bbox,area,segmentation(RLE),
#                                        predicted_iou,point_coords,crop_box,
#                                        stability_score}]}
#                       -> 只有开头的image字段有用(宽高可免解码分桶);
#                          annotations里的RLE mask按纯文生图口径全部丢弃(无用)
#
# 【SACap-1M与SA-1B的实测对账(全量扫完1000个目录，不是抽样推测)】
# - SACap-1M需要的1029250张图，在sa_1b里**缺jpg 0张、缺json 0张**;
# - 抽样2000图共11350个anno_id在sa_1b的annotations[].id里**100%命中**;
# - sa_1b json里的image.file_name与parquet的imagename**100%一致**(抽样3000条);
# 也就是说 **1029250行 == 1029250个包含完整有用信息的样本对，一条都不该丢**。
#
# 【无用信息(一律不整理进训练目录)】
# .cache/ / .gitattributes / README.md / data_samples.png / mask_distribution.png /
# cache/(512H_512W-group_bucket.parquet) /
# parquet的segments_info列(5882281个实例mask的anno_id与区域caption) /
# sa_1b json里的全部annotations(RLE mask等)
#
# 【本脚本的处理口径】
# - 因为丢弃了mask，sa_1b那个平均63KB的json**只需读前512字节**就能拿到
#   image.{image_id,width,height,file_name}(实测抽样3000条全部解析成功)，
#   不用解析完整json、更不用碰RLE，扫盘代价降一个数量级，同时还免解码拿到分辨率;
# - 整理前预检(硬失败): 根目录条目白名单、parquet的PAR1头尾魔数(O(1)拦下载截断)、
#   只读footer校验num_rows与3个有用列齐备、依赖数据集的1000个sa_%06d目录连号、
#   每个目录scandir一次核对jpg==json且无未知后缀文件、jpg总数与实测一致;
# - 并行分片单位 = 1000个image_group(天然分片，且和sa_1b目录一一对应);
# - 每条样本逐项校验: caption非空、jpg存在且非空、sa_1b json存在且能取到宽高、
#   json里的file_name与imagename一致、sample_key在组内不重复，
#   任何一项不过都记进对应的隔离清单显式上报，**绝不静默丢样本对**;
# - 每个组内严格对账: 有效样本对 + 各隔离清单条数 == 该组的parquet行数;
# - 每张图写盘后立即校验落盘大小 == 源文件大小(不做二次os.walk 100万个文件);
# - 汇总标注落 unzip_annotations/<image_group>.jsonl，每行一个完整样本对;
#   原始parquet原样拷到 unzip_source_annotations/ 作为溯源真值;
# - EXPECTED_*常量全部写实测ground truth(1029250 / 1000 / 11185362)，少一条立即抛异常;
# - 拷贝/整理/校验任一环出错都汇总后抛异常，不再静默跑过。
# ==============================================================================

DATASET_TASK_TYPE = 'text_to_image'

# 无用信息，不整理进训练目录:
# .cache/                huggingface下载缓存(21个文件)
# .gitattributes         git lfs配置
# README.md              数据集说明
# data_samples.png       README里的示意图(16.7MB)
# mask_distribution.png  README里的每图mask数分布图
# cache/                 Seg2Any在512x512下预算的cond/txt序列长度缓存，
#                        和本仓库的分辨率分桶与文本编码器都不匹配，可现算
# .DS_Store/CACHEDIR.TAG 目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'README.md',
    'data_samples.png',
    'mask_distribution.png',
    'cache',
    '.DS_Store',
    'CACHEDIR.TAG',
]

# 过滤掉无用信息后根目录只应该剩这一个目录
EXPECTED_ROOT_DIR_NAME_LIST = [
    'annotations',
]

LOAD_ANNOTATION_RELATIVE_PATH = 'annotations/anno_train.parquet'

# 只读这3列，segments_info(实例mask与区域caption)按纯文生图口径不读、不保存,
# 少读一列能把常驻内存从约400MB降到约200MB
LOAD_ANNOTATION_COLUMN_NAME_LIST = [
    'imagename',
    'image_group',
    'caption',
]

# parquet里应该齐备的全部列名(含被丢弃的segments_info),
# 少列说明上游数据规格变了，必须显式感知
EXPECTED_ANNOTATION_ALL_COLUMN_NAME_LIST = [
    'imagename',
    'image_group',
    'caption',
    'segments_info',
]

# 实测parquet footer行数，直接当作完整性ground truth，少一行都说明标注被改过或没下全
EXPECTED_ANNOTATION_ROW_COUNT = 1029250

# 实测image_group数(和依赖数据集sa_1b的目录数一致)
EXPECTED_IMAGE_GROUP_NUM = 1000

# 实测依赖数据集sa_1b里的jpg总数(json数与之相同)，数量不对说明依赖数据集不全
EXPECTED_DEPEND_IMAGE_FILE_COUNT = 11185362

# 实测每个image_group的样本条数区间(1005~1079)，官方没给精确条数，
# 只做软校验(打印告警)，硬校验靠总行数1029250
EXPECTED_IMAGE_GROUP_SAMPLE_PAIR_COUNT_RANGE = [1000, 1100]

IMAGE_GROUP_DIR_NAME_PATTERN = re.compile(r'^sa_(?P<index>\d{6})$')

IMAGE_FILE_NAME_PATTERN = re.compile(r'^sa_(?P<image_id>\d+)\.jpg$')

DEPEND_IMAGE_FILE_SUFFIX = '.jpg'

DEPEND_ANNOTATION_FILE_SUFFIX = '.json'

# sa_1b的json是{"image":{...},"annotations":[...]}，image字段一定在最前面，
# 读前512字节就能取到，不需要解析平均63KB的完整json、更不需要碰RLE
DEPEND_ANNOTATION_HEAD_READ_SIZE = 512

DEPEND_IMAGE_META_PATTERN = re.compile(r'"image"\s*:\s*\{(?P<body>[^{}]*)\}')

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_SOURCE_ANNOTATION_DIR_NAME = 'unzip_source_annotations'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

# 目标图是否拷贝到输出目录。
# True : 和其他数据集脚本口径一致，输出目录自包含，1029250张图约1.07T、100万个inode,
#        务必确认目标盘扛得住再跑;
# False: 图像继续留在依赖数据集sa_1b里(同一块NAS)，标注里的image_path是
#        相对depend_dataset_path的路径，样本对信息一样完整，省1.07T和100万个inode。
SAVE_IMAGE_FILE_FLAG = True

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

PARQUET_FILE_MAGIC_BYTES = b'PAR1'

MAX_SAVE_PROBLEM_ITEM_NUM = 10000


def check_skip_file_or_dir(per_file_relative_path):
    """过滤掉.cache、README.md、cache、data_samples.png等无用文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_single_parquet_file_magic(per_parquet_path):
    """O(1)预检parquet是否被截断: 文件头尾都必须是PAR1魔数

    如果下载不全，读footer只会在最后一步抛异常，必须在跑几小时整理前先拦住。
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


def check_single_depend_image_group_dir(per_depend_image_group_dir_pair):
    """scandir单个sa_1b组目录: 核对jpg与json同名配对，且没有未知后缀文件

    实测1000个目录里只有jpg和json、jpg数==json数==同名配对数，
    出现任何第三种文件都必须显式上报，不能被静默漏处理。
    """
    per_image_group, per_image_group_dir_path = per_depend_image_group_dir_pair

    error_message_list = []
    image_name_prefix_dict, annotation_name_prefix_dict = {}, {}
    unknown_suffix_file_name_list = []
    try:
        for per_dir_entry in os.scandir(per_image_group_dir_path):
            if not per_dir_entry.is_file():
                error_message_list.append(
                    f'{per_image_group} not a regular file {per_dir_entry.name}'
                )
                continue

            per_file_name_prefix, per_file_name_suffix = os.path.splitext(
                per_dir_entry.name)
            per_file_name_suffix = per_file_name_suffix.lower()

            if per_file_name_suffix == DEPEND_IMAGE_FILE_SUFFIX:
                image_name_prefix_dict[per_file_name_prefix] = 1
            elif per_file_name_suffix == DEPEND_ANNOTATION_FILE_SUFFIX:
                annotation_name_prefix_dict[per_file_name_prefix] = 1
            else:
                unknown_suffix_file_name_list.append(per_dir_entry.name)
    except Exception as e:
        error_message_list.append(
            f'{per_image_group} scan depend image group dir failed {e}')

        return [per_image_group, 0, 0, 0, error_message_list]

    per_image_file_count = len(image_name_prefix_dict)
    per_annotation_file_count = len(annotation_name_prefix_dict)
    per_matched_file_count = len(
        set(image_name_prefix_dict.keys())
        & set(annotation_name_prefix_dict.keys()))

    if len(unknown_suffix_file_name_list) > 0:
        error_message_list.append(
            f'{per_image_group} unknown suffix file num '
            f'{len(unknown_suffix_file_name_list)} '
            f'{sorted(unknown_suffix_file_name_list)[:3]}')
    if per_image_file_count != per_annotation_file_count:
        error_message_list.append(
            f'{per_image_group} image file count {per_image_file_count} != '
            f'annotation file count {per_annotation_file_count}')
    if per_matched_file_count != per_image_file_count:
        error_message_list.append(
            f'{per_image_group} matched file count {per_matched_file_count} != '
            f'image file count {per_image_file_count}')

    return [
        per_image_group,
        per_image_file_count,
        per_annotation_file_count,
        per_matched_file_count,
        error_message_list,
    ]


def check_required_dataset_complete(root_dataset_path, depend_dataset_path):
    """整理前预检: 根目录条目白名单、parquet魔数与footer、依赖数据集的1000个组目录

    数据集或依赖数据集本身不完整就没必要跑几小时整理，
    也避免"少了几十万张图但整体报成功"。
    """
    error_message_list = []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')
    if not os.path.exists(depend_dataset_path):
        error_message_list.append(
            f'depend dataset path not exist {depend_dataset_path}')

    if len(error_message_list) > 0:
        return error_message_list

    for per_name in sorted(os.listdir(root_dataset_path)):
        if check_skip_file_or_dir(per_name):
            continue

        if not os.path.isdir(os.path.join(root_dataset_path, per_name)):
            # 根目录下出现新的非跳过文件必须显式上报，否则会被静默漏处理
            error_message_list.append(f'unknown file in root dir {per_name}')
            continue

        if per_name not in EXPECTED_ROOT_DIR_NAME_LIST:
            error_message_list.append(f'unknown dir in root dir {per_name}')

    load_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_RELATIVE_PATH)
    if not os.path.exists(load_annotation_path):
        error_message_list.append(
            f'annotation parquet not exist {load_annotation_path}')

        return error_message_list

    error_message_list.extend(
        check_single_parquet_file_magic(load_annotation_path))

    try:
        load_parquet_file = pq.ParquetFile(load_annotation_path)
        per_annotation_row_count = load_parquet_file.metadata.num_rows
        per_annotation_column_name_list = list(
            load_parquet_file.schema_arrow.names)
    except Exception as e:
        print('7777', load_annotation_path, e)
        error_message_list.append(f'read parquet metadata failed {e}')

        return error_message_list

    print('1111', 'annotation row:', per_annotation_row_count,
          'expected annotation row:', EXPECTED_ANNOTATION_ROW_COUNT, 'column:',
          per_annotation_column_name_list)

    if per_annotation_row_count != EXPECTED_ANNOTATION_ROW_COUNT:
        error_message_list.append(
            f'annotation row count not match {per_annotation_row_count} != '
            f'{EXPECTED_ANNOTATION_ROW_COUNT}')

    for per_column_name in EXPECTED_ANNOTATION_ALL_COLUMN_NAME_LIST:
        if per_column_name not in per_annotation_column_name_list:
            error_message_list.append(
                f'annotation miss column {per_column_name}')

    depend_image_group_dir_pair_list, depend_image_group_index_list = [], []
    for per_name in sorted(os.listdir(depend_dataset_path)):
        per_image_group_dir_path = os.path.join(depend_dataset_path, per_name)
        if not os.path.isdir(per_image_group_dir_path):
            error_message_list.append(
                f'unknown file in depend dataset root dir {per_name}')
            continue

        per_match_result = IMAGE_GROUP_DIR_NAME_PATTERN.match(per_name)
        if not per_match_result:
            error_message_list.append(f'unknown depend image group dir '
                                      f'{per_name}')
            continue

        depend_image_group_index_list.append(
            int(per_match_result.group('index')))
        depend_image_group_dir_pair_list.append([
            per_name,
            per_image_group_dir_path,
        ])

    print('1111', 'depend image group dir:',
          len(depend_image_group_dir_pair_list), 'expected image group dir:',
          EXPECTED_IMAGE_GROUP_NUM)

    if len(depend_image_group_dir_pair_list) != EXPECTED_IMAGE_GROUP_NUM:
        error_message_list.append(
            f'depend image group dir num not match '
            f'{len(depend_image_group_dir_pair_list)} != '
            f'{EXPECTED_IMAGE_GROUP_NUM}')

    # 组目录编号必须是0..N-1连号，缺号说明依赖数据集有目录没下载下来
    per_missing_index_list = sorted(
        set(range(0, EXPECTED_IMAGE_GROUP_NUM)) -
        set(depend_image_group_index_list))
    if len(per_missing_index_list) > 0:
        error_message_list.append(
            f'depend image group index not continuous, missing index '
            f'{per_missing_index_list[:10]}')

    print('1111', 'check depend image group dir:',
          len(depend_image_group_dir_pair_list))
    total_depend_image_file_count, total_depend_annotation_file_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(
                pool.imap_unordered(check_single_depend_image_group_dir,
                                    depend_image_group_dir_pair_list),
                total=len(depend_image_group_dir_pair_list)):
            _, per_image_file_count, per_annotation_file_count, _, per_error_message_list = per_check_result
            total_depend_image_file_count += per_image_file_count
            total_depend_annotation_file_count += per_annotation_file_count
            error_message_list.extend(per_error_message_list)

    print('1111', 'depend image file:', total_depend_image_file_count,
          'depend annotation file:', total_depend_annotation_file_count,
          'expected depend image file:', EXPECTED_DEPEND_IMAGE_FILE_COUNT)

    if total_depend_image_file_count != EXPECTED_DEPEND_IMAGE_FILE_COUNT:
        error_message_list.append(f'depend image file count not match '
                                  f'{total_depend_image_file_count} != '
                                  f'{EXPECTED_DEPEND_IMAGE_FILE_COUNT}')
    if total_depend_annotation_file_count != EXPECTED_DEPEND_IMAGE_FILE_COUNT:
        error_message_list.append(f'depend annotation file count not match '
                                  f'{total_depend_annotation_file_count} != '
                                  f'{EXPECTED_DEPEND_IMAGE_FILE_COUNT}')

    return error_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把原始parquet标注原样拷贝到目标目录作为溯源真值，保持相对路径不变"""
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


def save_single_image_file(per_image_path, save_image_path, per_image_size):
    """流式把目标图拷到输出目录并立刻校验落盘大小，返回错误信息(空串表示成功)"""
    try:
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        with open(per_image_path, 'rb') as load_image_file:
            with open(save_image_path, 'wb') as save_image_file:
                shutil.copyfileobj(load_image_file, save_image_file,
                                   COPY_FILE_BLOCK_SIZE)
    except Exception as e:
        return f'write image failed {save_image_path} {e}'

    if os.path.getsize(save_image_path) != per_image_size:
        return f'write image size not match {save_image_path}'

    return ''


def get_single_depend_image_meta(per_depend_annotation_path):
    """只读sa_1b json的前512字节，取出image.{image_id,width,height,file_name}

    sa_1b的json平均63KB，绝大部分是实例mask的RLE，而本脚本按纯文生图口径把mask全丢了，
    所以完全没必要解析完整json: image字段一定在最前面，读一个512字节的头块就够，
    实测抽样3000条全部解析成功、file_name与parquet的imagename 100%一致。
    """
    try:
        with open(per_depend_annotation_path, 'rb') as load_annotation_file:
            per_head_bytes = load_annotation_file.read(
                DEPEND_ANNOTATION_HEAD_READ_SIZE)
    except Exception as e:
        return None, f'read depend annotation head failed {e}'

    per_match_result = DEPEND_IMAGE_META_PATTERN.search(
        per_head_bytes.decode('UTF-8', 'ignore'))
    if not per_match_result:
        return None, 'depend annotation image meta not found'

    try:
        per_image_meta = json.loads('{' + per_match_result.group('body') + '}')
    except Exception as e:
        return None, f'load depend annotation image meta failed {e}'

    if not isinstance(per_image_meta, dict):
        return None, 'depend annotation image meta not a dict'

    return per_image_meta, ''


def process_single_image_group(image_group_task, depend_dataset_path,
                               save_dataset_path, save_annotation_dir_path):
    """整理单个image_group: 逐条样本校验目标图与caption，落盘图像并生成jsonl汇总标注

    以"该组的parquet行数"为ground truth逐行处理，保证每一行要么进有效样本对、
    要么进某个显式上报的隔离清单，一条都不会凭空消失。
    """
    per_image_group, per_annotation_pair_list = image_group_task

    load_image_group_dir_path = os.path.join(depend_dataset_path,
                                             per_image_group)
    save_image_group_dir_path = os.path.join(save_dataset_path,
                                             SAVE_IMAGE_DIR_NAME,
                                             per_image_group)

    copy_image_count, skip_image_count, not_save_image_count = 0, 0, 0
    caption_char_length_sum, caption_word_count_sum = 0, 0
    empty_caption_sample_key_list, duplicate_sample_key_list = [], []
    missing_image_sample_key_list = []
    missing_depend_annotation_sample_key_list = []
    invalid_image_meta_message_list = []
    invalid_image_name_message_list = []
    error_message_list = []

    sample_key_dict = {}
    valid_annotation_line_list = []

    for per_image_name, per_caption in per_annotation_pair_list:
        per_sample_key = os.path.splitext(per_image_name)[0]

        if per_sample_key in sample_key_dict:
            # 同一组里出现重名样本时按名写盘会互相覆盖，只能保留第一条并显式上报
            duplicate_sample_key_list.append(
                f'{per_image_group}/{per_image_name}')
            continue
        sample_key_dict[per_sample_key] = 1

        if not isinstance(per_caption, str) or len(per_caption.strip()) == 0:
            # 文生图样本对必须有文本提示，没有caption的图不可训练
            empty_caption_sample_key_list.append(
                f'{per_image_group}/{per_image_name}')
            continue
        per_caption = per_caption.strip()

        per_match_result = IMAGE_FILE_NAME_PATTERN.match(per_image_name)
        if not per_match_result:
            # 图像名规格变了必须显式感知，但不因此丢样本对
            invalid_image_name_message_list.append(
                f'{per_image_group}/{per_image_name} unknown image name')

        per_image_path = os.path.join(load_image_group_dir_path,
                                      per_image_name)
        per_depend_annotation_path = os.path.join(
            load_image_group_dir_path,
            f'{per_sample_key}{DEPEND_ANNOTATION_FILE_SUFFIX}')

        per_image_size = -1
        try:
            per_image_size = os.path.getsize(per_image_path)
        except Exception:
            per_image_size = -1

        if per_image_size <= 0:
            # 目标图缺失或0字节，样本对不完整
            missing_image_sample_key_list.append(
                f'{per_image_group}/{per_image_name}')
            continue

        if not os.path.exists(per_depend_annotation_path):
            # 宽高只能从sa_1b的json里拿，拿不到就当样本对不完整显式上报
            missing_depend_annotation_sample_key_list.append(
                f'{per_image_group}/{per_sample_key}'
                f'{DEPEND_ANNOTATION_FILE_SUFFIX}')
            continue

        per_image_meta, per_image_meta_error_message = get_single_depend_image_meta(
            per_depend_annotation_path)
        if per_image_meta_error_message:
            invalid_image_meta_message_list.append(
                f'{per_image_group}/{per_sample_key} '
                f'{per_image_meta_error_message}')
            per_image_meta = {}

        per_image_id = per_image_meta.get('image_id', None)
        per_image_width = per_image_meta.get('width', 0)
        per_image_height = per_image_meta.get('height', 0)
        per_depend_image_name = per_image_meta.get('file_name', '')

        if not isinstance(per_image_width, int) or not isinstance(
                per_image_height,
                int) or per_image_width <= 0 or per_image_height <= 0:
            # 宽高不可用只上报，不丢样本对(下游可以退回解码图像拿宽高)
            invalid_image_meta_message_list.append(
                f'{per_image_group}/{per_sample_key} invalid image size '
                f'{per_image_width} {per_image_height}')
            per_image_width, per_image_height = 0, 0

        if per_depend_image_name and per_depend_image_name != per_image_name:
            invalid_image_name_message_list.append(
                f'{per_image_group}/{per_image_name} depend image name not '
                f'match {per_depend_image_name}')

        if SAVE_IMAGE_FILE_FLAG:
            save_image_path = os.path.join(save_image_group_dir_path,
                                           per_image_name)
            if os.path.exists(save_image_path) and os.path.getsize(
                    save_image_path) == per_image_size:
                skip_image_count += 1
            else:
                per_save_error_message = save_single_image_file(
                    per_image_path, save_image_path, per_image_size)
                if per_save_error_message:
                    print('6666', per_image_group, per_save_error_message)
                    error_message_list.append(per_save_error_message)
                    continue

                copy_image_count += 1

            per_save_image_relative_path = (
                f'{SAVE_IMAGE_DIR_NAME}/{per_image_group}/{per_image_name}')
            per_save_image_path_root = 'save_dataset_path'
        else:
            # 只建索引模式: 图像继续留在依赖数据集sa_1b里，样本对信息一样完整
            not_save_image_count += 1
            per_save_image_relative_path = (
                f'{per_image_group}/{per_image_name}')
            per_save_image_path_root = 'depend_dataset_path'

        # 完整有用信息的样本对: 一张目标图 + 一条非空全图caption,
        # 再补上样本key、所属组、宽高(免解码分桶)与任务类型，方便下游直接按行取样本
        per_save_annotation = {
            'image_path': per_save_image_relative_path,
            'image_path_root': per_save_image_path_root,
            'sample_key': per_sample_key,
            'image_id': per_image_id,
            'image_group': per_image_group,
            'caption': per_caption,
            'img_width': per_image_width,
            'img_height': per_image_height,
            'dataset_task_type': DATASET_TASK_TYPE,
        }
        valid_annotation_line_list.append(
            json.dumps(per_save_annotation, ensure_ascii=False))

        caption_char_length_sum += len(per_caption)
        caption_word_count_sum += len(per_caption.split())

    save_annotation_path = os.path.join(save_annotation_dir_path,
                                        f'{per_image_group}.jsonl')
    try:
        os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
        with open(save_annotation_path, 'w',
                  encoding='UTF-8') as save_json_file:
            for per_valid_annotation_line in valid_annotation_line_list:
                save_json_file.write(f'{per_valid_annotation_line}\n')
    except Exception as e:
        error_message_list.append(
            f'{per_image_group} save annotation failed {e}')

    per_annotation_row_count = len(per_annotation_pair_list)
    per_valid_sample_pair_count = len(valid_annotation_line_list)

    # 核心对账: 该组parquet的每一行都必须有归属,
    # 要么是有效样本对，要么落进某个显式上报的隔离清单，绝不静默消失
    per_process_row_count = (per_valid_sample_pair_count +
                             len(empty_caption_sample_key_list) +
                             len(duplicate_sample_key_list) +
                             len(missing_image_sample_key_list) +
                             len(missing_depend_annotation_sample_key_list))
    if per_process_row_count != per_annotation_row_count:
        error_message_list.append(
            f'{per_image_group} process row count not match '
            f'{per_process_row_count} != {per_annotation_row_count}')

    if SAVE_IMAGE_FILE_FLAG and copy_image_count + skip_image_count != per_valid_sample_pair_count:
        error_message_list.append(
            f'{per_image_group} save image count not match '
            f'{copy_image_count} + {skip_image_count} != '
            f'{per_valid_sample_pair_count}')

    return {
        'image_group': per_image_group,
        'annotation_row_count': per_annotation_row_count,
        'valid_sample_pair_count': per_valid_sample_pair_count,
        'copy_image_count': copy_image_count,
        'skip_image_count': skip_image_count,
        'not_save_image_count': not_save_image_count,
        'caption_char_length_sum': caption_char_length_sum,
        'caption_word_count_sum': caption_word_count_sum,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_image_group}.jsonl',
        'empty_caption_sample_key_list': empty_caption_sample_key_list,
        'duplicate_sample_key_list': duplicate_sample_key_list,
        'missing_image_sample_key_list': missing_image_sample_key_list,
        'missing_depend_annotation_sample_key_list':
        missing_depend_annotation_sample_key_list,
        'invalid_image_meta_message_list': invalid_image_meta_message_list,
        'invalid_image_name_message_list': invalid_image_name_message_list,
        'error_message_list': error_message_list,
    }


def get_all_file_and_image_group_task(root_dataset_path):
    """读parquet的3个有用列，按image_group归组成并行任务分片，并收集待拷贝的原始标注

    segments_info(实例mask与区域caption)按纯文生图口径不读,
    1029250行只留imagename/image_group/caption，常驻内存约200MB。
    """
    file_copy_pair_list = []
    load_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_RELATIVE_PATH)
    file_copy_pair_list.append([
        f'{SAVE_SOURCE_ANNOTATION_DIR_NAME}/'
        f'{os.path.basename(load_annotation_path)}',
        load_annotation_path,
    ])

    error_message_list = []
    load_annotation_dict = pq.read_table(
        load_annotation_path,
        columns=LOAD_ANNOTATION_COLUMN_NAME_LIST).to_pydict()

    image_name_list = load_annotation_dict['imagename']
    image_group_list = load_annotation_dict['image_group']
    caption_list = load_annotation_dict['caption']

    total_annotation_row_count = len(image_name_list)
    if len(image_group_list) != total_annotation_row_count or len(
            caption_list) != total_annotation_row_count:
        error_message_list.append(
            f'annotation column length not match {total_annotation_row_count} '
            f'{len(image_group_list)} {len(caption_list)}')

    image_group_annotation_pair_dict = {}
    all_image_name_dict, duplicate_image_name_list = {}, []
    for per_row_index in range(total_annotation_row_count):
        per_image_name = image_name_list[per_row_index]
        per_image_group = image_group_list[per_row_index]
        per_caption = caption_list[per_row_index]

        if not isinstance(per_image_name, str) or not per_image_name:
            error_message_list.append(f'row {per_row_index} empty image name')
            continue

        if not isinstance(per_image_group, str) or not per_image_group:
            error_message_list.append(f'row {per_row_index} empty image group')
            continue

        # imagename实测1029250条全局唯一，一旦重名说明标注被改过，
        # 后面按名写盘会互相覆盖，必须先在这里显式上报
        if per_image_name in all_image_name_dict:
            duplicate_image_name_list.append(
                f'{per_image_group}/{per_image_name}')
        all_image_name_dict[per_image_name] = 1

        if per_image_group not in image_group_annotation_pair_dict:
            image_group_annotation_pair_dict[per_image_group] = []
        image_group_annotation_pair_dict[per_image_group].append([
            per_image_name,
            per_caption,
        ])

    if len(duplicate_image_name_list) > 0:
        error_message_list.append(
            f'duplicate image name num {len(duplicate_image_name_list)} '
            f'{sorted(duplicate_image_name_list)[:3]}')

    image_group_task_list = []
    for per_image_group in sorted(image_group_annotation_pair_dict.keys()):
        image_group_task_list.append([
            per_image_group,
            image_group_annotation_pair_dict[per_image_group],
        ])

    file_copy_pair_list = sorted(file_copy_pair_list, key=lambda x: x[0])

    return [
        file_copy_pair_list,
        image_group_task_list,
        total_annotation_row_count,
        error_message_list,
    ]


def check_image_group_task_complete(image_group_task_list,
                                    total_annotation_row_count,
                                    depend_dataset_path):
    """整理前预检: 组数、组名、每组条数、组目录是否都在依赖数据集里"""
    error_message_list, warning_message_list = [], []

    if len(image_group_task_list) != EXPECTED_IMAGE_GROUP_NUM:
        error_message_list.append(
            f'image group num not match {len(image_group_task_list)} != '
            f'{EXPECTED_IMAGE_GROUP_NUM}')

    total_task_row_count = 0
    for per_image_group, per_annotation_pair_list in image_group_task_list:
        total_task_row_count += len(per_annotation_pair_list)

        if not IMAGE_GROUP_DIR_NAME_PATTERN.match(per_image_group):
            error_message_list.append(f'unknown image group {per_image_group}')
            continue

        if not os.path.exists(
                os.path.join(depend_dataset_path, per_image_group)):
            # 标注引用的组目录在依赖数据集里不存在，那一整组样本对全都拿不到图
            error_message_list.append(
                f'depend image group dir not exist {per_image_group}')
            continue

        if not EXPECTED_IMAGE_GROUP_SAMPLE_PAIR_COUNT_RANGE[0] <= len(
                per_annotation_pair_list
        ) <= EXPECTED_IMAGE_GROUP_SAMPLE_PAIR_COUNT_RANGE[1]:
            # 官方没给每组精确条数，只按实测区间做软校验，打印告警不判失败
            warning_message_list.append(
                f'{per_image_group} sample pair count '
                f'{len(per_annotation_pair_list)} not in '
                f'{EXPECTED_IMAGE_GROUP_SAMPLE_PAIR_COUNT_RANGE}')

    if total_task_row_count != total_annotation_row_count:
        error_message_list.append(
            f'image group task row count not match {total_task_row_count} != '
            f'{total_annotation_row_count}')
    if total_annotation_row_count != EXPECTED_ANNOTATION_ROW_COUNT:
        error_message_list.append(
            f'annotation row count not match {total_annotation_row_count} != '
            f'{EXPECTED_ANNOTATION_ROW_COUNT}')

    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    return error_message_list, warning_message_list


def save_check_result(save_dataset_path, depend_dataset_path,
                      image_group_result_list, warning_message_list):
    """汇总所有组的整理与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_annotation_row_count, total_valid_sample_pair_count = 0, 0
    total_copy_image_count, total_skip_image_count = 0, 0
    total_not_save_image_count = 0
    total_caption_char_length_sum, total_caption_word_count_sum = 0, 0
    image_group_sample_pair_count_dict = {}
    empty_caption_sample_key_list, duplicate_sample_key_list = [], []
    missing_image_sample_key_list = []
    missing_depend_annotation_sample_key_list = []
    invalid_image_meta_message_list, invalid_image_name_message_list = [], []
    error_message_list = []

    for per_image_group_result in image_group_result_list:
        per_image_group = per_image_group_result['image_group']

        total_annotation_row_count += per_image_group_result[
            'annotation_row_count']
        total_valid_sample_pair_count += per_image_group_result[
            'valid_sample_pair_count']
        total_copy_image_count += per_image_group_result['copy_image_count']
        total_skip_image_count += per_image_group_result['skip_image_count']
        total_not_save_image_count += per_image_group_result[
            'not_save_image_count']
        total_caption_char_length_sum += per_image_group_result[
            'caption_char_length_sum']
        total_caption_word_count_sum += per_image_group_result[
            'caption_word_count_sum']

        image_group_sample_pair_count_dict[
            per_image_group] = per_image_group_result[
                'valid_sample_pair_count']

        empty_caption_sample_key_list.extend(
            per_image_group_result['empty_caption_sample_key_list'])
        duplicate_sample_key_list.extend(
            per_image_group_result['duplicate_sample_key_list'])
        missing_image_sample_key_list.extend(
            per_image_group_result['missing_image_sample_key_list'])
        missing_depend_annotation_sample_key_list.extend(
            per_image_group_result['missing_depend_annotation_sample_key_list']
        )
        invalid_image_meta_message_list.extend(
            per_image_group_result['invalid_image_meta_message_list'])
        invalid_image_name_message_list.extend(
            per_image_group_result['invalid_image_name_message_list'])

        if len(per_image_group_result['error_message_list']) > 0:
            print('7777', per_image_group,
                  per_image_group_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_image_group} error num '
                f'{len(per_image_group_result["error_message_list"])} '
                f'{per_image_group_result["error_message_list"][:3]}')

    average_caption_char_length, average_caption_word_count = 0, 0
    if total_valid_sample_pair_count > 0:
        average_caption_char_length = round(
            total_caption_char_length_sum / total_valid_sample_pair_count, 2)
        average_caption_word_count = round(
            total_caption_word_count_sum / total_valid_sample_pair_count, 2)

    print('3333', 'total annotation row:', total_annotation_row_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'copy image:', total_copy_image_count, 'skip image:',
          total_skip_image_count, 'not save image:',
          total_not_save_image_count, 'empty caption:',
          len(empty_caption_sample_key_list), 'duplicate sample key:',
          len(duplicate_sample_key_list), 'missing image:',
          len(missing_image_sample_key_list), 'missing depend annotation:',
          len(missing_depend_annotation_sample_key_list),
          'invalid image meta:', len(invalid_image_meta_message_list),
          'invalid image name:', len(invalid_image_name_message_list))
    print('3333', 'average caption char length:', average_caption_char_length,
          'average caption word count:', average_caption_word_count)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'depend_dataset_path':
        depend_dataset_path,
        'save_image_file_flag':
        SAVE_IMAGE_FILE_FLAG,
        'total_image_group_count':
        len(image_group_result_list),
        'total_annotation_row_count':
        total_annotation_row_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_copy_image_count':
        total_copy_image_count,
        'total_skip_image_count':
        total_skip_image_count,
        'total_not_save_image_count':
        total_not_save_image_count,
        'average_caption_char_length':
        average_caption_char_length,
        'average_caption_word_count':
        average_caption_word_count,
        'empty_caption_count':
        len(empty_caption_sample_key_list),
        'duplicate_sample_key_count':
        len(duplicate_sample_key_list),
        'missing_image_count':
        len(missing_image_sample_key_list),
        'missing_depend_annotation_count':
        len(missing_depend_annotation_sample_key_list),
        'invalid_image_meta_count':
        len(invalid_image_meta_message_list),
        'invalid_image_name_count':
        len(invalid_image_name_message_list),
        'image_group_sample_pair_count_dict':
        image_group_sample_pair_count_dict,
        'empty_caption_sample_key_list':
        sorted(empty_caption_sample_key_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'duplicate_sample_key_list':
        sorted(duplicate_sample_key_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'missing_image_sample_key_list':
        sorted(missing_image_sample_key_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'missing_depend_annotation_sample_key_list':
        sorted(missing_depend_annotation_sample_key_list)
        [:MAX_SAVE_PROBLEM_ITEM_NUM],
        'invalid_image_meta_message_list':
        sorted(invalid_image_meta_message_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'invalid_image_name_message_list':
        sorted(invalid_image_name_message_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        warning_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'check_error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    if len(image_group_result_list) != EXPECTED_IMAGE_GROUP_NUM:
        error_message_list.append(
            f'image group count not match {len(image_group_result_list)} != '
            f'{EXPECTED_IMAGE_GROUP_NUM}')
    if total_annotation_row_count != EXPECTED_ANNOTATION_ROW_COUNT:
        error_message_list.append(
            f'total annotation row count not match '
            f'{total_annotation_row_count} != {EXPECTED_ANNOTATION_ROW_COUNT}')
    # 实测1029250行全部是包含完整有用信息的样本对，少一条就说明漏了样本对
    if total_valid_sample_pair_count != EXPECTED_ANNOTATION_ROW_COUNT:
        error_message_list.append(f'total valid sample pair count not match '
                                  f'{total_valid_sample_pair_count} != '
                                  f'{EXPECTED_ANNOTATION_ROW_COUNT}')
    if len(empty_caption_sample_key_list) > 0:
        error_message_list.append(
            f'empty caption count {len(empty_caption_sample_key_list)}')
    if len(duplicate_sample_key_list) > 0:
        error_message_list.append(
            f'duplicate sample key count {len(duplicate_sample_key_list)}')
    if len(missing_image_sample_key_list) > 0:
        error_message_list.append(
            f'missing image count {len(missing_image_sample_key_list)}')
    if len(missing_depend_annotation_sample_key_list) > 0:
        error_message_list.append(
            f'missing depend annotation count '
            f'{len(missing_depend_annotation_sample_key_list)}')
    if len(invalid_image_meta_message_list) > 0:
        error_message_list.append(
            f'invalid image meta count {len(invalid_image_meta_message_list)}')
    if len(invalid_image_name_message_list) > 0:
        error_message_list.append(
            f'invalid image name count {len(invalid_image_name_message_list)}')
    if SAVE_IMAGE_FILE_FLAG and total_copy_image_count + total_skip_image_count != total_valid_sample_pair_count:
        error_message_list.append(
            f'total save image count not match {total_copy_image_count} + '
            f'{total_skip_image_count} != {total_valid_sample_pair_count}')

    return error_message_list


def preprocess_dataset(root_dataset_path, depend_dataset_path,
                       save_dataset_path):
    dataset_error_message_list = check_required_dataset_complete(
        root_dataset_path, depend_dataset_path)
    if len(dataset_error_message_list) > 0:
        # 数据集或依赖数据集本身不完整就没必要跑几小时整理
        raise Exception(
            f'check dataset failed {dataset_error_message_list[:20]}')

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    file_copy_pair_list, image_group_task_list, total_annotation_row_count, load_error_message_list = get_all_file_and_image_group_task(
        root_dataset_path)
    if len(load_error_message_list) > 0:
        raise Exception(
            f'load annotation failed {load_error_message_list[:20]}')

    print('1111', len(file_copy_pair_list), len(image_group_task_list),
          total_annotation_row_count)
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(image_group_task_list) > 0:
        print('1111', image_group_task_list[0][0],
              len(image_group_task_list[0][1]),
              image_group_task_list[0][1][0][0])

    task_error_message_list, warning_message_list = check_image_group_task_complete(
        image_group_task_list, total_annotation_row_count, depend_dataset_path)
    if len(task_error_message_list) > 0:
        raise Exception(
            f'check image group task failed {task_error_message_list[:20]}')

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

    image_group_result_list = []
    extract_func = partial(process_single_image_group,
                           depend_dataset_path=depend_dataset_path,
                           save_dataset_path=save_dataset_path,
                           save_annotation_dir_path=save_annotation_dir_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_image_group_result in tqdm(pool.imap_unordered(
                extract_func, image_group_task_list),
                                           total=len(image_group_task_list)):
            image_group_result_list.append(per_image_group_result)

            print('2222', per_image_group_result['image_group'],
                  'annotation row:',
                  per_image_group_result['annotation_row_count'],
                  'valid sample pair:',
                  per_image_group_result['valid_sample_pair_count'],
                  'copy image:', per_image_group_result['copy_image_count'],
                  'skip image:', per_image_group_result['skip_image_count'],
                  'not save image:',
                  per_image_group_result['not_save_image_count'])

    check_error_message_list = save_check_result(save_dataset_path,
                                                 depend_dataset_path,
                                                 image_group_result_list,
                                                 warning_message_list)

    all_error_message_list = copy_error_message_list + check_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/整理/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} '
            f'{all_error_message_list[:20]}')

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/SACap-1M'
    depend_dataset_path = r'/root/autodl-tmp/public_datasets/sa_1b'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, depend_dataset_path,
                       save_dataset_path)
