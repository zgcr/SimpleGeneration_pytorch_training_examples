import os
import re
import json
import shutil
import tarfile

import pyarrow.parquet as pq

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ImgEdit是纯图像编辑数据集(1.2M编辑对)，没有任何文生图子集。
# 每个样本对 = 1~2张参考图 + 1张编辑后图 + 1条编辑指令(prompt)，
# 另外每个样本目录里还自带result.json(原图路径/分辨率/edit_type/bbox/clip_score/aes_score)
# 和judge.json/judge_2scores.json(打分)，根目录all_dataset_gpt_score.json是同一套打分的汇总。

ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.tar\.split\.(?P<part>\d+)$'),
    re.compile(r'^(?P<prefix>.+)\.tar$'),
]

PARQUET_FILE_NAME_PATTERN = re.compile(r'^(?P<prefix>.+)\.parquet$')

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.gif',
    '.webp',
    '.tif',
    '.tiff',
]

SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    '.gitignore',
    'CACHEDIR.TAG',
    '.DS_Store',
    'README.md',
    # ImgEdit_Judge是打分用的判别模型权重(4个safetensors共十几GB)，不是训练数据
    'ImgEdit_Judge',
]

# ---------------------------------------------------------------------------
# 压缩包组配置
#
# 这是本脚本的核心修复点。parquet里记录的图像路径前缀、压缩包名、压缩包内顶层目录
# 三者大面积不一致(实测35个图像压缩包组里有6组对不上)，例如:
#   hybrid_part0.parquet写results_compose_part0/、压缩包却叫results_hybrid_part0.tar；
#   replace_part0.parquet写replace/、压缩包内顶层目录却是results/；
#   action_part1.parquet直接写裸文件名、压缩包内顶层目录却是part1/；
#   results_adjust_canny_laion_part2.tar的顶层目录是./；
#   results_content_understanding_part2.tar的顶层目录是mnt/data/lzj/codes/...绝对路径残留。
#
# 修复思路: 不再"额外建一层压缩包名目录然后模糊匹配路径尾部"，而是解压时
# 剥掉压缩包自带的顶层目录、统一改挂到parquet使用的规范子集名下，即落盘为
#   images/<subset_name>/<样本目录>/<文件名>
# 这样parquet里的路径拼上images/前缀后就是磁盘真实路径，校验可以按**完整路径精确匹配**。
#
# tar_top_strip_prefix = 压缩包内要剥掉的顶层目录(已归一化，''表示压缩包顶层是./无需剥)
# subset_name          = 落盘用的规范子集名(等于parquet里的路径前缀)
ARCHIVE_GROUP_CONFIG_DICT = {
    'Singleturn/action_part1': ['part1', 'part1'],
    'Singleturn/action_part2': ['part2', 'part2'],
    'Singleturn/action_part3': ['part3', 'part3'],
    'Singleturn/action_part4': ['part4', 'part4'],
    'Singleturn/results_add_laion_part0': [
        'results_add_laion_part0',
        'results_add_laion_part0',
    ],
    'Singleturn/results_add_laion_part1': [
        'results_add_laion_part1',
        'results_add_laion_part1',
    ],
    'Singleturn/results_add_laion_part4': [
        'results_add_laion_part4',
        'results_add_laion_part4',
    ],
    'Singleturn/results_add_laion_part5': [
        'results_add_laion_part5',
        'results_add_laion_part5',
    ],
    'Singleturn/results_adjust_canny_laion_part0': [
        'results_adjust_canny_laion_part0',
        'results_adjust_canny_laion_part0',
    ],
    # 顶层目录是./，没有可剥的目录名
    'Singleturn/results_adjust_canny_laion_part2': [
        '',
        'results_adjust_canny_laion_part2',
    ],
    'Singleturn/results_adjust_canny_laion_part3': [
        'results_adjust_canny_laion_part3',
        'results_adjust_canny_laion_part3',
    ],
    'Singleturn/results_adjust_canny_laion_part4': [
        'results_adjust_canny_laion_part4',
        'results_adjust_canny_laion_part4',
    ],
    'Singleturn/results_background_laion_part0': [
        'results_background_laion_part0',
        'results_background_laion_part0',
    ],
    'Singleturn/results_background_laion_part2': [
        'results_background_laion_part2',
        'results_background_laion_part2',
    ],
    'Singleturn/results_background_laion_part3': [
        'results_background_laion_part3',
        'results_background_laion_part3',
    ],
    'Singleturn/results_background_laion_part5': [
        'results_background_laion_part5',
        'results_background_laion_part5',
    ],
    'Singleturn/results_background_laion_part7': [
        'results_background_laion_part7',
        'results_background_laion_part7',
    ],
    'Singleturn/results_extract_and_visualedit_part1': [
        'results_extract_ref_part1',
        'results_extract_ref_part1',
    ],
    'Singleturn/results_extract_and_visualedit_part7': [
        'results_extract_ref_part7',
        'results_extract_ref_part7',
    ],
    'Singleturn/results_hybrid_part0': [
        'results_compose_part0',
        'results_compose_part0',
    ],
    'Singleturn/results_hybrid_part2': [
        'results_compose_part2',
        'results_compose_part2',
    ],
    'Singleturn/results_hybrid_part6': [
        'results_compose_part6_fix',
        'results_compose_part6_fix',
    ],
    'Singleturn/results_remove_laion_part1': [
        'results_remove_laion_part1',
        'results_remove_laion_part1',
    ],
    'Singleturn/results_remove_laion_part4': [
        'results_remove_laion_part4',
        'results_remove_laion_part4',
    ],
    'Singleturn/results_remove_laion_part5': [
        'results_remove_laion_part5',
        'results_remove_laion_part5',
    ],
    'Singleturn/results_remove_part0': ['results_remove', 'results_remove'],
    'Singleturn/results_replace_laion_part1': [
        'results_replace_laion_part1',
        'results_replace_laion_part1',
    ],
    'Singleturn/results_replace_laion_part4': [
        'results_replace_laion_part4',
        'results_replace_laion_part4',
    ],
    'Singleturn/results_replace_laion_part5': [
        'results_replace_laion_part5',
        'results_replace_laion_part5',
    ],
    # 压缩包内顶层目录是results/，但parquet写的是replace/
    'Singleturn/results_replace_part0': ['results', 'replace'],
    'Singleturn/results_style_transfer': [
        'results_style_transfer',
        'results_style_transfer',
    ],
    'Singleturn/results_style_transfer_part0': [
        'results_style_transfer_part0_cap36472',
        'results_style_transfer_part0_cap36472',
    ],
    'Multiturn/results_content_memory_part2': [
        'results_content_memory_part2',
        'results_content_memory_part2',
    ],
    # 顶层目录是打包时的绝对路径残留
    'Multiturn/results_content_understanding_part2': [
        'mnt/data/lzj/codes/shitedit_comfyui/results_content_understanding_part2',
        'results_content_understanding_part2',
    ],
    'Multiturn/results_version_backtracking_part0': [
        'results_version_backtracking_part0',
        'results_version_backtracking_part0',
    ],
}

# Benchmark是评测集(287个成员)，自带annotation.json/annotation.jsonl/singleturn.json，
# 和Parquet标注体系完全无关，单独解压到benchmark/，不参与训练样本对校验。
# 原脚本把Benchmark塞进图像子集列表一起统计，导致它的图全被算成orphan。
BENCHMARK_ARCHIVE_GROUP_KEY = './Benchmark'

BENCHMARK_TAR_TOP_STRIP_PREFIX = 'Benchmark'

SAVE_BENCHMARK_ROOT_NAME = 'benchmark'

# 各压缩包组的分片数(实测)。缺片会导致tar流在中途截断、后面的样本全部丢失，
# 而且tarfile只会抛一个"unexpected end of data"，所以必须在解压前硬对账。
EXPECTED_ARCHIVE_PART_NUM_DICT = {
    './Benchmark': 1,
    'Multiturn/results_content_memory_part2': 27,
    'Multiturn/results_content_understanding_part2': 56,
    'Multiturn/results_version_backtracking_part0': 56,
    'Singleturn/action_part1': 2,
    'Singleturn/action_part2': 2,
    'Singleturn/action_part3': 2,
    'Singleturn/action_part4': 2,
    'Singleturn/results_add_laion_part0': 4,
    'Singleturn/results_add_laion_part1': 9,
    'Singleturn/results_add_laion_part4': 4,
    'Singleturn/results_add_laion_part5': 4,
    'Singleturn/results_adjust_canny_laion_part0': 5,
    'Singleturn/results_adjust_canny_laion_part2': 1,
    'Singleturn/results_adjust_canny_laion_part3': 5,
    'Singleturn/results_adjust_canny_laion_part4': 5,
    'Singleturn/results_background_laion_part0': 2,
    'Singleturn/results_background_laion_part2': 2,
    'Singleturn/results_background_laion_part3': 2,
    'Singleturn/results_background_laion_part5': 2,
    'Singleturn/results_background_laion_part7': 1,
    'Singleturn/results_extract_and_visualedit_part1': 11,
    'Singleturn/results_extract_and_visualedit_part7': 3,
    'Singleturn/results_hybrid_part0': 3,
    'Singleturn/results_hybrid_part2': 3,
    'Singleturn/results_hybrid_part6': 3,
    'Singleturn/results_remove_laion_part1': 9,
    'Singleturn/results_remove_laion_part4': 1,
    'Singleturn/results_remove_laion_part5': 5,
    'Singleturn/results_remove_part0': 11,
    'Singleturn/results_replace_laion_part1': 9,
    'Singleturn/results_replace_laion_part4': 5,
    'Singleturn/results_replace_laion_part5': 2,
    'Singleturn/results_replace_part0': 14,
    'Singleturn/results_style_transfer': 4,
    'Singleturn/results_style_transfer_part0': 17,
}

# ---------------------------------------------------------------------------
# parquet配置: 每个parquet属于哪个规范子集、是单轮还是多轮、应该有多少行(实测)
#
# 注意reference_extract_part1和reference_replace_part1共用同一个子集
# (results_extract_ref_part1)，是一个压缩包配两份标注，属正常情况。
PARQUET_CONFIG_DICT = {
    'action_part1': ['part1', False, 37224],
    'action_part2': ['part2', False, 37416],
    'action_part3': ['part3', False, 40294],
    'action_part4': ['part4', False, 44074],
    'add_part0': ['results_add_laion_part0', False, 32233],
    'add_part1': ['results_add_laion_part1', False, 77039],
    'add_part4': ['results_add_laion_part4', False, 35127],
    'add_part5': ['results_add_laion_part5', False, 31068],
    'adjust_canny_part0': ['results_adjust_canny_laion_part0', False, 42030],
    'adjust_canny_part2': ['results_adjust_canny_laion_part2', False, 7756],
    'adjust_canny_part3': ['results_adjust_canny_laion_part3', False, 42904],
    'adjust_canny_part4': ['results_adjust_canny_laion_part4', False, 42819],
    'background_part0': ['results_background_laion_part0', False, 14135],
    'background_part2': ['results_background_laion_part2', False, 13991],
    'background_part3': ['results_background_laion_part3', False, 13493],
    'background_part5': ['results_background_laion_part5', False, 14095],
    'background_part7': ['results_background_laion_part7', False, 2376],
    'content_memory_part2': ['results_content_memory_part2', True, 30861],
    'content_understanding_part2': [
        'results_content_understanding_part2',
        True,
        42139,
    ],
    'hybrid_part0': ['results_compose_part0', False, 10267],
    'hybrid_part2': ['results_compose_part2', False, 8656],
    'hybrid_part6': ['results_compose_part6_fix', False, 9467],
    'reference_extract_part1': ['results_extract_ref_part1', False, 76907],
    'reference_extract_part7': ['results_extract_ref_part7', False, 41993],
    'reference_replace_part1': ['results_extract_ref_part1', False, 76907],
    'reference_replace_part7': ['results_extract_ref_part7', False, 41993],
    'remove_part0': ['results_remove', False, 32222],
    'remove_part1': ['results_remove_laion_part1', False, 76981],
    'remove_part4': ['results_remove_laion_part4', False, 7698],
    'remove_part5': ['results_remove_laion_part5', False, 42745],
    'replace_part0': ['replace', False, 32209],
    'replace_part1': ['results_replace_laion_part1', False, 75626],
    'replace_part4': ['results_replace_laion_part4', False, 39912],
    'replace_part5': ['results_replace_laion_part5', False, 11648],
    'style_transfer': ['results_style_transfer', False, 28374],
    'style_transfer_part0': [
        'results_style_transfer_part0_cap36472',
        False,
        36472,
    ],
    'version_backtracking_part0': [
        'results_version_backtracking_part0',
        True,
        42023,
    ],
}

PARQUET_INPUT_IMAGE_COLUMN_NAME = 'input_images'

PARQUET_OUTPUT_IMAGE_COLUMN_NAME = 'output_images'

PARQUET_PROMPT_COLUMN_NAME = 'prompt'

# Multiturn的三个parquet只有data这一列，里面是每一轮编辑的字典列表，
# 必须递归进去才能取到图像路径。原脚本只看顶层input_images/output_images，
# 导致这三个子集(11.5万行/31.4万个编辑对)引用的图一张都没进校验集合。
PARQUET_TURN_LIST_COLUMN_NAME = 'data'

# 全量期望值(实测)。这是"保证每个完整样本对都被处理"能被验证的关键兜底。
EXPECTED_PARQUET_FILE_NUM = 37

EXPECTED_TOTAL_ANNOTATION_ROW_COUNT = 1293174

EXPECTED_TOTAL_SAMPLE_PAIR_COUNT = 1492359

# content_memory的每行第0轮只有一句全局约束、没有图，是"全局指令"不是编辑对
EXPECTED_TOTAL_GLOBAL_INSTRUCTION_TURN_COUNT = 30861

EXPECTED_TOTAL_ANNOTATION_IMAGE_REFERENCE_COUNT = 3103618

EXPECTED_TOTAL_ANNOTATION_UNIQUE_IMAGE_COUNT = 2865662

EXPECTED_SINGLE_REFERENCE_SAMPLE_PAIR_COUNT = 1373459

EXPECTED_DOUBLE_REFERENCE_SAMPLE_PAIR_COUNT = 118900

EXPECTED_MULTITURN_TURN_NUM = 3

# ---------------------------------------------------------------------------
# 样本目录里自带的小标注文件。这些是"单个样本的其他所有属性"，属于有用信息，
# 但原样解压会在盘上留下几百万个小文件，所以边解压边解析、每个压缩包汇总成一个jsonl。
SAMPLE_METADATA_FILE_NAME_LIST = [
    'result.json',
    'judge.json',
    'judge_2scores.json',
]

# 根目录的208MB打分汇总，按<前缀>/<样本目录名>索引。它的前缀又和parquet前缀有3处不一致。
GPT_SCORE_FILE_NAME = 'all_dataset_gpt_score.json'

GPT_SCORE_PREFIX_ALIAS_DICT = {
    'results': 'replace',
    'results_extract_laion_part1': 'results_extract_ref_part1',
    'results_compose_part6': 'results_compose_part6_fix',
}

SAVE_IMAGE_ROOT_NAME = 'images'

SAVE_ANNOTATION_ROOT_NAME = 'annotations'

SAVE_METADATA_ROOT_NAME = 'metadata'

SAVE_IMAGE_INDEX_ROOT_NAME = 'image_index'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_result.json'

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

TAR_BLOCK_SIZE = 512

TAR_EOF_BLOCK_SIZE = 1024

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024

PARQUET_ROW_BATCH_SIZE = 4096


class MultiPartArchiveReader:
    """把按字节切分的多个分片压缩包拼接成一个只读的连续字节流"""

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
    """过滤掉.cache、.gitattributes、README.md、ImgEdit_Judge这些不需要整理的文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def get_archive_part_sort_key(per_archive_part_index):
    """分片编号排序key: 纯数字编号按数值排序，字母编号按位数优先再按字典序排序

    返回值统一是[编号类型, 数值编号, 字母编号]三元组，保证不同命名风格之间也能比较。
    不能直接按字符串排序: part-9 会排到 part-10 后面；
    也不能只按[位数, 字符串]排序: part-100 会排到 part-89 前面导致整个tar流错位。
    """
    per_archive_part_index = per_archive_part_index or ''
    if per_archive_part_index.isdigit():
        return [0, int(per_archive_part_index), '']

    return [1, 0, per_archive_part_index]


def get_normalized_member_name(per_member_name):
    """把tar成员名归一化成不带盘符/前导斜杠/./的相对路径

    返回[归一化名, 是否是压缩包根成员]。
    results_adjust_canny_laion_part2.tar的顶层成员就是'./'，归一化后是'.'，
    它是合法的压缩包根目录而不是非法成员，要区别对待、不能记成error。
    越界成员(..开头)返回['', False]。
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


def strip_member_top_prefix(per_member_name, per_tar_top_strip_prefix):
    """剥掉压缩包自带的顶层目录，返回样本内相对路径

    压缩包顶层目录和parquet前缀大面积不一致，剥掉之后统一改挂到规范子集名下，
    parquet路径才能和磁盘路径精确对上。剥不掉(前缀不匹配)时返回空串，由调用方记error。
    """
    if not per_tar_top_strip_prefix:
        return per_member_name

    if per_member_name == per_tar_top_strip_prefix:
        return ''

    if per_member_name.startswith(f'{per_tar_top_strip_prefix}/'):
        return per_member_name[len(per_tar_top_strip_prefix) + 1:]

    return None


def check_single_archive_tar_tail(per_archive_part_path_list):
    """O(1)预检压缩包是否被截断: 总长度必须512字节对齐，且末片结尾必须有1024字节全0的EOF块

    ImgEdit的压缩包是未压缩tar按字节切片，所以可以只读整包尾部1024字节就判断是否完整，
    不用把5.4TB读一遍。缺尾片/写盘写一半这两种最隐蔽的情况都能在解压前被拦住。

    注意EOF块可能跨分片边界(末片本身可能不足1024字节)，所以要从后往前跨片拼尾部字节，
    不能只读末片。
    """
    error_message_list = []

    per_archive_part_size_list = [
        os.path.getsize(per_archive_part_path)
        for per_archive_part_path in per_archive_part_path_list
    ]
    total_archive_size = sum(per_archive_part_size_list)

    if total_archive_size == 0:
        error_message_list.append('archive total size is 0')
        return error_message_list

    if total_archive_size % TAR_BLOCK_SIZE != 0:
        error_message_list.append(
            f'archive total size {total_archive_size} not aligned to {TAR_BLOCK_SIZE}'
        )

    if total_archive_size < TAR_EOF_BLOCK_SIZE:
        error_message_list.append(
            f'archive total size {total_archive_size} < {TAR_EOF_BLOCK_SIZE}')
        return error_message_list

    # 从最后一片往前读，直到凑满1024字节尾部
    tail_bytes_list, remain_tail_size = [], TAR_EOF_BLOCK_SIZE
    for per_archive_part_index in range(
            len(per_archive_part_path_list) - 1, -1, -1):
        if remain_tail_size <= 0:
            break

        per_archive_part_path = per_archive_part_path_list[
            per_archive_part_index]
        per_archive_part_size = per_archive_part_size_list[
            per_archive_part_index]
        if per_archive_part_size == 0:
            continue

        per_read_size = min(remain_tail_size, per_archive_part_size)
        with open(per_archive_part_path, 'rb') as load_file:
            load_file.seek(per_archive_part_size - per_read_size)
            tail_bytes_list.append(load_file.read(per_read_size))

        remain_tail_size -= per_read_size

    tail_bytes = b''.join(reversed(tail_bytes_list))

    if len(tail_bytes) != TAR_EOF_BLOCK_SIZE:
        error_message_list.append(
            f'read archive tail size {len(tail_bytes)} != {TAR_EOF_BLOCK_SIZE}'
        )
        return error_message_list

    if tail_bytes != b'\x00' * TAR_EOF_BLOCK_SIZE:
        error_message_list.append('archive tail eof block not found')

    return error_message_list


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表、按分片归组后的压缩包列表和parquet文件列表"""
    file_copy_pair_list, parquet_group_list = [], []
    archive_part_path_dict = {}
    for per_root_path, _, per_file_name_list in os.walk(root_dataset_path):
        for per_file_name in sorted(per_file_name_list):
            per_file_path = os.path.join(per_root_path, per_file_name)
            per_file_relative_path = os.path.relpath(per_file_path,
                                                     root_dataset_path)
            per_file_relative_dir = os.path.dirname(per_file_relative_path)

            if check_skip_file_or_dir(per_file_relative_path):
                continue

            per_parquet_match_result = PARQUET_FILE_NAME_PATTERN.match(
                per_file_name)
            if per_parquet_match_result:
                parquet_group_list.append([
                    per_parquet_match_result.group('prefix'),
                    per_file_relative_dir,
                    per_file_path,
                ])
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

            per_archive_relative_dir = per_file_relative_dir if per_file_relative_dir else '.'
            per_archive_group_key = f'{per_archive_relative_dir}/{per_archive_group_name}'
            if per_archive_group_key not in archive_part_path_dict:
                archive_part_path_dict[per_archive_group_key] = [
                    per_archive_group_name,
                    per_archive_relative_dir,
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
        archive_group_list.append([
            per_archive_group_key,
            per_archive_group_name,
            per_archive_relative_dir,
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
    parquet_group_list = sorted(parquet_group_list, key=lambda x: x[0])

    return file_copy_pair_list, archive_group_list, parquet_group_list


def check_archive_group_complete(archive_group_list):
    """解压前硬对账压缩包组: 组名必须已知、分片数必须相等、编号必须从0连号、整包不能截断"""
    error_message_list = []

    all_expected_group_key_list = sorted(EXPECTED_ARCHIVE_PART_NUM_DICT.keys())
    found_group_key_list = []

    for per_archive_group in archive_group_list:
        per_archive_group_key, per_archive_group_name, _, per_archive_part_index_list, per_archive_part_path_list = per_archive_group
        found_group_key_list.append(per_archive_group_key)

        if per_archive_group_key not in EXPECTED_ARCHIVE_PART_NUM_DICT:
            error_message_list.append(
                f'unknown archive group {per_archive_group_key}')
            continue

        expected_part_num = EXPECTED_ARCHIVE_PART_NUM_DICT[
            per_archive_group_key]
        if len(per_archive_part_path_list) != expected_part_num:
            error_message_list.append(
                f'{per_archive_group_key} part num not match '
                f'{len(per_archive_part_path_list)} != {expected_part_num}')

        digit_part_index_list = sorted([
            int(per_archive_part_index)
            for per_archive_part_index in per_archive_part_index_list
            if per_archive_part_index.isdigit()
        ])
        if len(digit_part_index_list) == len(
                per_archive_part_index_list) and len(
                    digit_part_index_list) > 0:
            missing_part_index_list = sorted(
                set(range(len(digit_part_index_list))) -
                set(digit_part_index_list))
            if missing_part_index_list:
                error_message_list.append(
                    f'{per_archive_group_key} missing part index '
                    f'{missing_part_index_list}')

        for per_tail_error_message in check_single_archive_tar_tail(
                per_archive_part_path_list):
            error_message_list.append(
                f'{per_archive_group_key} {per_tail_error_message}')

    for per_expected_group_key in all_expected_group_key_list:
        if per_expected_group_key not in found_group_key_list:
            error_message_list.append(
                f'missing archive group {per_expected_group_key}')

    return error_message_list


def check_parquet_group_complete(parquet_group_list):
    """解开parquet前硬对账: 文件数、文件名、每个文件的行数都必须和实测期望一致"""
    error_message_list = []

    if len(parquet_group_list) != EXPECTED_PARQUET_FILE_NUM:
        error_message_list.append(
            f'parquet file num not match {len(parquet_group_list)} != '
            f'{EXPECTED_PARQUET_FILE_NUM}')

    found_parquet_name_list = []
    for per_parquet_group in parquet_group_list:
        per_parquet_group_name, _, per_parquet_path = per_parquet_group
        found_parquet_name_list.append(per_parquet_group_name)

        if per_parquet_group_name not in PARQUET_CONFIG_DICT:
            error_message_list.append(
                f'unknown parquet file {per_parquet_group_name}')
            continue

        expected_row_count = PARQUET_CONFIG_DICT[per_parquet_group_name][2]
        try:
            per_row_count = pq.ParquetFile(per_parquet_path).metadata.num_rows
        except Exception as e:
            error_message_list.append(
                f'read parquet {per_parquet_group_name} failed {e}')
            continue

        if per_row_count != expected_row_count:
            error_message_list.append(
                f'{per_parquet_group_name} row count not match '
                f'{per_row_count} != {expected_row_count}')

    for per_expected_parquet_name in sorted(PARQUET_CONFIG_DICT.keys()):
        if per_expected_parquet_name not in found_parquet_name_list:
            error_message_list.append(
                f'missing parquet file {per_expected_parquet_name}')

    return error_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非压缩包非parquet文件原样拷贝到目标目录，保持相对路径不变"""
    per_file_relative_path, per_file_path = file_copy_pair

    save_file_path = os.path.join(save_dataset_path, per_file_relative_path)
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if os.path.isfile(save_file_path) and os.path.getsize(
            save_file_path) == os.path.getsize(per_file_path):
        return [per_file_relative_path, []]

    error_message_list = []
    try:
        with open(per_file_path, 'rb') as load_file:
            with open(save_file_path, 'wb') as save_file:
                shutil.copyfileobj(load_file, save_file, COPY_FILE_BLOCK_SIZE)

        if os.path.getsize(save_file_path) != os.path.getsize(per_file_path):
            error_message_list.append(
                f'copy size not match {per_file_relative_path}')
    except Exception as e:
        error_message_list.append(f'copy failed {per_file_relative_path} {e}')

    return [per_file_relative_path, error_message_list]


def process_single_archive_group(archive_group, save_dataset_path):
    """流式解压单个压缩包组，剥掉压缩包自带顶层目录后改挂到规范子集名下

    落盘结构: images/<subset_name>/<样本目录>/<文件名>，和parquet里的路径一一对应。
    样本目录里的result.json/judge.json不落盘成几百万个小文件，而是边解析边汇总成
    metadata/<压缩包组名>.jsonl，每行是一个样本的全部附加属性。
    """
    per_archive_group_key, per_archive_group_name, _, _, per_archive_part_path_list = archive_group

    is_benchmark = per_archive_group_key == BENCHMARK_ARCHIVE_GROUP_KEY
    if is_benchmark:
        per_tar_top_strip_prefix = BENCHMARK_TAR_TOP_STRIP_PREFIX
        save_image_dir_path = os.path.join(save_dataset_path,
                                           SAVE_BENCHMARK_ROOT_NAME)
    else:
        per_tar_top_strip_prefix, per_subset_name = ARCHIVE_GROUP_CONFIG_DICT[
            per_archive_group_key]
        save_image_dir_path = os.path.join(save_dataset_path,
                                           SAVE_IMAGE_ROOT_NAME,
                                           per_subset_name)

    os.makedirs(save_image_dir_path, exist_ok=True)

    save_metadata_path = os.path.join(save_dataset_path,
                                      SAVE_METADATA_ROOT_NAME,
                                      f'{per_archive_group_name}.jsonl')
    os.makedirs(os.path.dirname(save_metadata_path), exist_ok=True)

    total_member_count, total_file_member_count = 0, 0
    extract_file_count, skip_file_count, metadata_count = 0, 0, 0
    error_message_list = []
    sample_metadata_dict = {}

    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    try:
        with tarfile.open(fileobj=archive_reader, mode='r|*') as load_tar_file:
            for per_member in load_tar_file:
                total_member_count += 1

                per_member_name, is_archive_root = get_normalized_member_name(
                    per_member.name)
                if not per_member_name:
                    # 压缩包根成员(./)直接跳过，真正越界的成员才记error
                    if not is_archive_root:
                        error_message_list.append(
                            f'illegal member name {per_member.name}')
                    continue

                per_relative_path = strip_member_top_prefix(
                    per_member_name, per_tar_top_strip_prefix)
                if per_relative_path is None:
                    error_message_list.append(
                        f'member top prefix not match {per_member.name}')
                    continue

                if not per_relative_path:
                    continue

                if check_skip_file_or_dir(per_relative_path):
                    continue

                if per_member.isdir():
                    continue

                if not per_member.isfile():
                    error_message_list.append(
                        f'not regular file member {per_member.name}')
                    continue

                total_file_member_count += 1

                per_file_name = os.path.basename(per_relative_path)
                per_file_name_suffix = os.path.splitext(
                    per_file_name)[1].lower()

                # 样本目录自带的小标注文件: 边解析边汇总，不落盘成海量小文件
                if (not is_benchmark
                    ) and per_file_name in SAMPLE_METADATA_FILE_NAME_LIST:
                    per_sample_dir = os.path.dirname(per_relative_path)
                    load_member_file = load_tar_file.extractfile(per_member)
                    if load_member_file is None:
                        error_message_list.append(
                            f'extractfile none {per_member.name}')
                        continue

                    try:
                        per_metadata = json.loads(
                            load_member_file.read().decode('UTF-8'))
                    except Exception as e:
                        error_message_list.append(
                            f'parse metadata failed {per_member.name} {e}')
                        continue

                    sample_metadata_dict.setdefault(
                        per_sample_dir,
                        {})[os.path.splitext(per_file_name)[0]] = per_metadata
                    metadata_count += 1
                    continue

                # Benchmark自带的annotation.json/jsonl/singleturn.json要原样保留
                if (not is_benchmark
                    ) and per_file_name_suffix not in IMAGE_FILE_SUFFIX_LIST:
                    continue

                save_member_path = os.path.join(save_image_dir_path,
                                                per_relative_path)

                if os.path.isfile(save_member_path) and os.path.getsize(
                        save_member_path) == per_member.size:
                    skip_file_count += 1
                    continue

                os.makedirs(os.path.dirname(save_member_path), exist_ok=True)

                load_member_file = load_tar_file.extractfile(per_member)
                if load_member_file is None:
                    error_message_list.append(
                        f'extractfile none {per_member.name}')
                    continue

                try:
                    with open(save_member_path, 'wb') as save_member_file:
                        shutil.copyfileobj(load_member_file, save_member_file,
                                           EXTRACT_FILE_BLOCK_SIZE)

                    if os.path.getsize(save_member_path) != per_member.size:
                        error_message_list.append(
                            f'extract size not match {per_member.name}')
                        continue
                except Exception as e:
                    error_message_list.append(
                        f'extract failed {per_member.name} {e}')
                    continue

                extract_file_count += 1
    except Exception as e:
        # 分片不全或压缩包截断时保留已解压出的文件，但必须把异常记成error硬上报
        error_message_list.append(f'read archive failed {e}')
    finally:
        archive_reader.close()

    try:
        with open(save_metadata_path, 'w',
                  encoding='UTF-8') as save_metadata_file:
            for per_sample_dir in sorted(sample_metadata_dict.keys()):
                per_save_metadata = {
                    'sample_dir': per_sample_dir,
                }
                per_save_metadata.update(sample_metadata_dict[per_sample_dir])
                save_metadata_file.write(
                    f'{json.dumps(per_save_metadata, ensure_ascii=False)}\n')
    except Exception as e:
        error_message_list.append(f'save metadata failed {e}')

    return [
        per_archive_group_key,
        total_member_count,
        total_file_member_count,
        extract_file_count,
        skip_file_count,
        metadata_count,
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    ]


def resolve_annotation_image_path(per_image_name, per_subset_name):
    """把parquet里记录的图像路径规约成整理后目录下的规范相对路径

    parquet里有两种写法:
      1. <子集前缀>/<样本目录>/<文件名>  (绝大多数子集)
      2. <文件名>                        (action_part1~4，只有裸文件名)
    统一规约成 <subset_name>/<...>，和解压落盘路径完全一致，可以精确匹配。
    """
    per_image_name = per_image_name.replace('\\', '/').strip()
    if not per_image_name:
        return None, 'empty image path'

    per_path_part_list = [
        per_path_part for per_path_part in per_image_name.split('/')
        if per_path_part and per_path_part != '.'
    ]
    if not per_path_part_list:
        return None, f'illegal image path {per_image_name}'

    if len(per_path_part_list) == 1:
        return f'{per_subset_name}/{per_path_part_list[0]}', ''

    if per_path_part_list[0] != per_subset_name:
        return None, (f'image path prefix not match {per_image_name} '
                      f'expect {per_subset_name}')

    return '/'.join([per_subset_name] + per_path_part_list[1:]), ''


def get_single_turn_sample_pair(per_turn, per_subset_name):
    """把一轮编辑解析成一个样本对: 1~2张参考图 + 1张编辑后图 + 1条编辑指令"""
    per_prompt = per_turn.get(PARQUET_PROMPT_COLUMN_NAME, None)
    per_prompt = per_prompt.strip() if isinstance(per_prompt, str) else ''

    per_input_image_name_list = per_turn.get(PARQUET_INPUT_IMAGE_COLUMN_NAME,
                                             None) or []
    per_output_image_name_list = per_turn.get(PARQUET_OUTPUT_IMAGE_COLUMN_NAME,
                                              None) or []
    if not isinstance(per_input_image_name_list, list):
        per_input_image_name_list = [per_input_image_name_list]
    if not isinstance(per_output_image_name_list, list):
        per_output_image_name_list = [per_output_image_name_list]

    # content_memory每行第0轮只有一句全局约束、没有任何图，是全局指令不是编辑对
    if not per_input_image_name_list and not per_output_image_name_list:
        return None, per_prompt, []

    reason_list = []
    if not per_prompt:
        reason_list.append('empty prompt')

    reference_image_path_list = []
    for per_image_name in per_input_image_name_list:
        per_image_path, per_reason = resolve_annotation_image_path(
            per_image_name, per_subset_name)
        if per_image_path is None:
            reason_list.append(per_reason)
            continue
        reference_image_path_list.append(per_image_path)

    target_image_path_list = []
    for per_image_name in per_output_image_name_list:
        per_image_path, per_reason = resolve_annotation_image_path(
            per_image_name, per_subset_name)
        if per_image_path is None:
            reason_list.append(per_reason)
            continue
        target_image_path_list.append(per_image_path)

    if not reference_image_path_list:
        reason_list.append('no reference image')
    if len(target_image_path_list) != 1:
        reason_list.append(
            f'target image num {len(target_image_path_list)} != 1')

    per_sample_pair = {
        'prompt':
        per_prompt,
        'reference_image_path_list':
        reference_image_path_list,
        'reference_image_num':
        len(reference_image_path_list),
        'target_image_path':
        target_image_path_list[0] if len(target_image_path_list) == 1 else '',
    }

    return per_sample_pair, per_prompt, reason_list


def get_sample_dir_name(per_image_path):
    """从规范图像路径里取出<子集名>/<样本目录名>，用来关联样本自带属性和gpt打分"""
    per_path_part_list = per_image_path.split('/')
    if len(per_path_part_list) >= 3:
        return '/'.join(per_path_part_list[:2])

    return per_path_part_list[0]


def process_single_parquet_file(parquet_group, save_dataset_path):
    """把单个parquet解开成annotations/<parquet名>.jsonl，每行是一个完整样本对

    单轮子集一行就是一个样本对；多轮子集一行是一段3轮对话，展开成多个样本对，
    每行都自带sample_id/turn_index/total_turn_num/global_prompt/history_prompt_list，
    下游既能当独立样本对用、也能按sample_id重新组回多轮对话。
    """
    per_parquet_group_name, _, per_parquet_path = parquet_group

    per_subset_name, is_multiturn, expected_row_count = PARQUET_CONFIG_DICT[
        per_parquet_group_name]

    save_annotation_path = os.path.join(save_dataset_path,
                                        SAVE_ANNOTATION_ROOT_NAME,
                                        f'{per_parquet_group_name}.jsonl')
    os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
    save_temp_annotation_path = f'{save_annotation_path}.tmp'

    row_count, sample_pair_count = 0, 0
    global_instruction_turn_count = 0
    image_reference_count = 0
    single_reference_count, double_reference_count = 0, 0
    unique_image_path_set = set()
    invalid_sample_pair_list = []
    error_message_list = []

    try:
        with open(save_temp_annotation_path, 'w',
                  encoding='UTF-8') as save_annotation_file:
            load_parquet_file = pq.ParquetFile(per_parquet_path)
            for per_record_batch in load_parquet_file.iter_batches(
                    batch_size=PARQUET_ROW_BATCH_SIZE):
                for per_row in per_record_batch.to_pylist():
                    per_row_index = row_count
                    row_count += 1

                    if is_multiturn:
                        per_turn_list = per_row.get(
                            PARQUET_TURN_LIST_COLUMN_NAME, None) or []
                        if not isinstance(per_turn_list, list):
                            per_turn_list = [per_turn_list]
                    else:
                        per_turn_list = [per_row]

                    per_global_prompt = ''
                    per_history_prompt_list = []
                    per_row_sample_pair_list = []

                    for per_turn_index, per_turn in enumerate(per_turn_list):
                        if not isinstance(per_turn, dict):
                            error_message_list.append(
                                f'row {per_row_index} turn {per_turn_index} not dict'
                            )
                            continue

                        per_sample_pair, per_prompt, per_reason_list = get_single_turn_sample_pair(
                            per_turn, per_subset_name)

                        if per_sample_pair is None:
                            # 无图轮次 = 全局约束指令，挂到后续所有轮次上
                            global_instruction_turn_count += 1
                            if per_prompt:
                                per_global_prompt = per_prompt
                            continue

                        per_sample_pair['global_prompt'] = per_global_prompt
                        per_sample_pair['history_prompt_list'] = list(
                            per_history_prompt_list)
                        per_sample_pair['turn_index'] = per_turn_index
                        per_history_prompt_list.append(per_prompt)

                        per_sample_pair['reason_list'] = per_reason_list
                        per_row_sample_pair_list.append(per_sample_pair)

                    per_sample_dir = ''
                    for per_sample_pair in per_row_sample_pair_list:
                        for per_image_path in per_sample_pair[
                                'reference_image_path_list']:
                            per_sample_dir = get_sample_dir_name(
                                per_image_path)
                            break
                        if per_sample_dir:
                            break

                    per_sample_id = f'{per_parquet_group_name}_{per_row_index:08d}'

                    for per_sample_pair in per_row_sample_pair_list:
                        per_reason_list = per_sample_pair.pop('reason_list')

                        per_annotation = {
                            'sample_id': per_sample_id,
                            'subset_name': per_subset_name,
                            'parquet_name': per_parquet_group_name,
                            'task_type': 'image_edit',
                            'row_index': per_row_index,
                            'total_turn_num': len(per_row_sample_pair_list),
                            'sample_dir': per_sample_dir,
                        }
                        per_annotation.update(per_sample_pair)

                        sample_pair_count += 1

                        per_image_path_list = per_sample_pair[
                            'reference_image_path_list'] + ([
                                per_sample_pair['target_image_path']
                            ] if per_sample_pair['target_image_path'] else [])
                        image_reference_count += len(per_image_path_list)
                        unique_image_path_set.update(per_image_path_list)

                        if per_sample_pair['reference_image_num'] == 1:
                            single_reference_count += 1
                        elif per_sample_pair['reference_image_num'] == 2:
                            double_reference_count += 1

                        if per_reason_list:
                            if len(invalid_sample_pair_list
                                   ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                                invalid_sample_pair_list.append({
                                    'parquet_name':
                                    per_parquet_group_name,
                                    'sample_id':
                                    per_sample_id,
                                    'row_index':
                                    per_row_index,
                                    'turn_index':
                                    per_sample_pair['turn_index'],
                                    'reason_list':
                                    per_reason_list,
                                })
                            continue

                        save_annotation_file.write(
                            f'{json.dumps(per_annotation, ensure_ascii=False)}\n'
                        )

        if row_count != expected_row_count:
            error_message_list.append(
                f'{per_parquet_group_name} row count not match '
                f'{row_count} != {expected_row_count}')

        os.replace(save_temp_annotation_path, save_annotation_path)
    except Exception as e:
        error_message_list.append(f'parse parquet failed {e}')
        if os.path.exists(save_temp_annotation_path):
            os.remove(save_temp_annotation_path)

    return [
        per_parquet_group_name,
        row_count,
        sample_pair_count,
        global_instruction_turn_count,
        image_reference_count,
        len(unique_image_path_set),
        single_reference_count,
        double_reference_count,
        invalid_sample_pair_list,
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    ]


def collect_single_subset_image_index(subset_collect_pair):
    """把一个子集目录下所有已落盘图像的相对路径写成一个索引文件

    2.8M张图全塞一个set再喂给32个子进程会占十几GB内存，所以按子集分片建索引，
    校验时每个parquet只加载它自己那个子集的索引。
    """
    per_subset_name, per_subset_path, save_index_path = subset_collect_pair

    image_relative_path_list = []
    error_message_list = []
    try:
        for per_root_path, _, per_file_name_list in os.walk(per_subset_path):
            for per_file_name in per_file_name_list:
                if os.path.splitext(per_file_name)[1].lower(
                ) not in IMAGE_FILE_SUFFIX_LIST:
                    continue

                per_file_path = os.path.join(per_root_path, per_file_name)
                per_file_relative_path = os.path.relpath(
                    per_file_path, per_subset_path).replace('\\', '/')
                image_relative_path_list.append(
                    f'{per_subset_name}/{per_file_relative_path}')

        with open(save_index_path, 'w', encoding='UTF-8') as save_index_file:
            for per_image_relative_path in image_relative_path_list:
                save_index_file.write(f'{per_image_relative_path}\n')
    except Exception as e:
        error_message_list.append(
            f'collect subset {per_subset_name} failed {e}')

    return [
        per_subset_name,
        len(image_relative_path_list),
        error_message_list,
    ]


def check_single_annotation_file(annotation_check_pair):
    """逐条校验一个annotations/<parquet名>.jsonl里的样本对，图像必须真的落盘

    这里按**完整规范路径**精确匹配，不再按路径尾部模糊匹配。
    原脚本按<样本目录>/<文件名>两段key匹配，实测161万个两段key里有58万个
    在多个子集之间重名(最多8个子集共用同一个key，例如00049_00040_000401278/original.png
    在add/remove/replace/extract等子集里都存在)，一张图存在就会把最多8个不同子集的
    标注图像同时标记成"已存在"，missing数被大幅低估，校验形同虚设。
    """
    per_parquet_group_name, per_annotation_path, per_image_index_path = annotation_check_pair

    exist_image_path_set = set()
    error_message_list = []
    try:
        if os.path.isfile(per_image_index_path):
            with open(per_image_index_path, 'r',
                      encoding='UTF-8') as load_index_file:
                for per_line in load_index_file:
                    per_line = per_line.strip()
                    if per_line:
                        exist_image_path_set.add(per_line)
    except Exception as e:
        error_message_list.append(f'load image index failed {e}')

    checked_sample_pair_count, valid_sample_pair_count = 0, 0
    missing_image_path_set = set()
    used_image_path_set = set()
    missing_sample_pair_list = []

    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_annotation_file:
            for per_line in load_annotation_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                per_annotation = json.loads(per_line)
                checked_sample_pair_count += 1

                per_image_path_list = list(
                    per_annotation.get('reference_image_path_list', []))
                if per_annotation.get('target_image_path', ''):
                    per_image_path_list.append(
                        per_annotation['target_image_path'])

                used_image_path_set.update(per_image_path_list)

                per_missing_image_path_list = [
                    per_image_path for per_image_path in per_image_path_list
                    if per_image_path not in exist_image_path_set
                ]

                if per_missing_image_path_list:
                    missing_image_path_set.update(per_missing_image_path_list)
                    if len(missing_sample_pair_list
                           ) < MAX_SAVE_PROBLEM_ITEM_NUM:
                        missing_sample_pair_list.append({
                            'parquet_name':
                            per_parquet_group_name,
                            'sample_id':
                            per_annotation.get('sample_id', ''),
                            'missing_image_path_list':
                            per_missing_image_path_list,
                        })
                    continue

                valid_sample_pair_count += 1
    except Exception as e:
        error_message_list.append(
            f'check annotation {per_parquet_group_name} failed {e}')

    return [
        per_parquet_group_name,
        checked_sample_pair_count,
        valid_sample_pair_count,
        used_image_path_set,
        missing_image_path_set,
        missing_sample_pair_list,
        error_message_list,
    ]


def attach_gpt_score(root_dataset_path, save_dataset_path):
    """把根目录208MB的gpt打分汇总按样本目录并进metadata，不再只是原样拷一份

    打分的key前缀又和parquet前缀有3处不一致(results->replace、
    results_extract_laion_part1->results_extract_ref_part1、
    results_compose_part6->results_compose_part6_fix)，必须走别名表。
    """
    load_gpt_score_path = os.path.join(root_dataset_path, GPT_SCORE_FILE_NAME)
    if not os.path.isfile(load_gpt_score_path):
        return 0, [f'missing {GPT_SCORE_FILE_NAME}']

    error_message_list = []
    try:
        with open(load_gpt_score_path, 'r',
                  encoding='UTF-8') as load_json_file:
            gpt_score_dict = json.load(load_json_file)
    except Exception as e:
        return 0, [f'load {GPT_SCORE_FILE_NAME} failed {e}']

    save_gpt_score_path = os.path.join(save_dataset_path,
                                       SAVE_METADATA_ROOT_NAME,
                                       'all_dataset_gpt_score.jsonl')
    os.makedirs(os.path.dirname(save_gpt_score_path), exist_ok=True)

    save_count = 0
    try:
        with open(save_gpt_score_path, 'w',
                  encoding='UTF-8') as save_json_file:
            for per_key in sorted(gpt_score_dict.keys()):
                per_path_part_list = per_key.split('/')
                if len(per_path_part_list) < 2:
                    continue

                per_prefix = per_path_part_list[0]
                per_subset_name = GPT_SCORE_PREFIX_ALIAS_DICT.get(
                    per_prefix, per_prefix)

                per_gpt_score = {
                    'sample_dir':
                    f'{per_subset_name}/{per_path_part_list[-1]}',
                    'original_key': per_key,
                    'gpt_score': gpt_score_dict[per_key],
                }
                save_json_file.write(
                    f'{json.dumps(per_gpt_score, ensure_ascii=False)}\n')
                save_count += 1
    except Exception as e:
        error_message_list.append(f'save gpt score failed {e}')

    return save_count, error_message_list


def check_image_annotation_pair(save_dataset_path, process_summary_dict):
    """按样本对逐条校验: annotations里的每个样本对引用的图像是否都真的落盘

    校验口径全部按**唯一规范路径集合**统计，保证
    matched + missing == 标注引用的唯一图像数 且非负。
    原脚本matched按引用次数累加、missing用"唯一数-累加值"算，本数据集有
    310万次引用但只有287万张唯一图(reference_extract和reference_replace共用同一批图)，
    这个口径必然算出负数把真实缺图掩盖掉。
    """
    check_error_message_list = []

    root_annotation_path = os.path.join(save_dataset_path,
                                        SAVE_ANNOTATION_ROOT_NAME)
    root_image_path = os.path.join(save_dataset_path, SAVE_IMAGE_ROOT_NAME)
    root_image_index_path = os.path.join(save_dataset_path,
                                         SAVE_IMAGE_INDEX_ROOT_NAME)

    if not os.path.isdir(root_annotation_path):
        check_error_message_list.append(
            f'missing annotation dir {root_annotation_path}')
        return check_error_message_list

    if not os.path.isdir(root_image_path):
        check_error_message_list.append(f'missing image dir {root_image_path}')
        return check_error_message_list

    os.makedirs(root_image_index_path, exist_ok=True)

    # 第一步: 按子集并行建图像索引(一次os.walk，避免几百万次stat)
    subset_collect_pair_list = []
    for per_subset_name in sorted(os.listdir(root_image_path)):
        per_subset_path = os.path.join(root_image_path, per_subset_name)
        if not os.path.isdir(per_subset_path):
            continue

        subset_collect_pair_list.append([
            per_subset_name,
            per_subset_path,
            os.path.join(root_image_index_path, f'{per_subset_name}.txt'),
        ])

    total_unzip_image_count = 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_collect_result in tqdm(pool.imap_unordered(
                collect_single_subset_image_index, subset_collect_pair_list),
                                       total=len(subset_collect_pair_list)):
            per_subset_name, per_image_count, per_error_message_list = per_collect_result
            total_unzip_image_count += per_image_count
            check_error_message_list.extend(per_error_message_list)
            print('2222', 'subset', per_subset_name, 'unzip image',
                  per_image_count)

    # 第二步: 按parquet并行逐条校验样本对
    annotation_check_pair_list = []
    for per_annotation_name in sorted(os.listdir(root_annotation_path)):
        if not per_annotation_name.endswith('.jsonl'):
            continue

        per_parquet_group_name = per_annotation_name[:-len('.jsonl')]
        per_subset_name = PARQUET_CONFIG_DICT.get(per_parquet_group_name,
                                                  [''])[0]
        annotation_check_pair_list.append([
            per_parquet_group_name,
            os.path.join(root_annotation_path, per_annotation_name),
            os.path.join(root_image_index_path, f'{per_subset_name}.txt'),
        ])

    total_checked_sample_pair_count, total_valid_sample_pair_count = 0, 0
    all_used_image_path_set, all_missing_image_path_set = set(), set()
    all_missing_sample_pair_list = []
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_annotation_file, annotation_check_pair_list),
                                     total=len(annotation_check_pair_list)):
            per_parquet_group_name, per_checked_count, per_valid_count, per_used_image_path_set, per_missing_image_path_set, per_missing_sample_pair_list, per_error_message_list = per_check_result

            total_checked_sample_pair_count += per_checked_count
            total_valid_sample_pair_count += per_valid_count
            all_used_image_path_set.update(per_used_image_path_set)
            all_missing_image_path_set.update(per_missing_image_path_set)
            all_missing_sample_pair_list.extend(
                per_missing_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM])
            check_error_message_list.extend(per_error_message_list)

            print('2222', per_parquet_group_name, 'checked sample pair',
                  per_checked_count, 'valid sample pair', per_valid_count,
                  'missing image', len(per_missing_image_path_set))

    # 第三步: 统计orphan(解压出来但没被任何标注引用的图)
    total_orphan_image_count = 0
    orphan_image_path_list = []
    for per_subset_collect_pair in subset_collect_pair_list:
        _, _, per_index_path = per_subset_collect_pair
        if not os.path.isfile(per_index_path):
            continue

        with open(per_index_path, 'r', encoding='UTF-8') as load_index_file:
            for per_line in load_index_file:
                per_line = per_line.strip()
                if not per_line:
                    continue
                if per_line in all_used_image_path_set:
                    continue

                total_orphan_image_count += 1
                if len(orphan_image_path_list) < MAX_SAVE_PROBLEM_ITEM_NUM:
                    orphan_image_path_list.append(per_line)

    total_annotation_unique_image_count = len(all_used_image_path_set)
    total_missing_image_count = len(all_missing_image_path_set)
    total_matched_image_count = total_annotation_unique_image_count - total_missing_image_count

    print('3333', 'checked sample pair', total_checked_sample_pair_count,
          'valid sample pair', total_valid_sample_pair_count,
          'annotation unique image', total_annotation_unique_image_count,
          'unzip image', total_unzip_image_count, 'matched image',
          total_matched_image_count, 'missing image',
          total_missing_image_count, 'orphan image', total_orphan_image_count)

    # 第四步: 全量期望值硬对账
    total_sample_pair_count = process_summary_dict.get(
        'total_sample_pair_count', 0)
    total_invalid_sample_pair_count = len(
        process_summary_dict.get('invalid_sample_pair_list', []))

    # 写出的样本对 + 被隔离的不完整样本对 必须等于parquet里解析出的样本对总数，
    # 否则说明有样本对在"解析->写jsonl->回读校验"这条链路上凭空消失了
    if total_checked_sample_pair_count + total_invalid_sample_pair_count != total_sample_pair_count:
        check_error_message_list.append(
            f'checked + invalid sample pair count not match '
            f'{total_checked_sample_pair_count} + '
            f'{total_invalid_sample_pair_count} != {total_sample_pair_count}')

    if total_sample_pair_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        check_error_message_list.append(
            f'total sample pair count not match {total_sample_pair_count} != '
            f'{EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}')

    if total_missing_image_count > 0:
        check_error_message_list.append(
            f'missing image count {total_missing_image_count}')

    if total_valid_sample_pair_count != total_checked_sample_pair_count:
        check_error_message_list.append(
            f'valid sample pair count not match '
            f'{total_valid_sample_pair_count} != {total_checked_sample_pair_count}'
        )

    if total_annotation_unique_image_count != EXPECTED_TOTAL_ANNOTATION_UNIQUE_IMAGE_COUNT:
        check_error_message_list.append(
            f'total annotation unique image count not match '
            f'{total_annotation_unique_image_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_UNIQUE_IMAGE_COUNT}')

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'total_annotation_row_count':
        process_summary_dict.get('total_annotation_row_count', 0),
        'total_sample_pair_count':
        total_sample_pair_count,
        'total_global_instruction_turn_count':
        process_summary_dict.get('total_global_instruction_turn_count', 0),
        'total_annotation_image_reference_count':
        process_summary_dict.get('total_annotation_image_reference_count', 0),
        'total_annotation_unique_image_count':
        total_annotation_unique_image_count,
        'total_unzip_image_count':
        total_unzip_image_count,
        'matched_image_count':
        total_matched_image_count,
        'missing_image_count':
        total_missing_image_count,
        'orphan_image_count':
        total_orphan_image_count,
        'checked_sample_pair_count':
        total_checked_sample_pair_count,
        'valid_sample_pair_count':
        total_valid_sample_pair_count,
        'single_reference_sample_pair_count':
        process_summary_dict.get('total_single_reference_count', 0),
        'double_reference_sample_pair_count':
        process_summary_dict.get('total_double_reference_count', 0),
        'gpt_score_count':
        process_summary_dict.get('gpt_score_count', 0),
        'invalid_sample_pair_list':
        process_summary_dict.get('invalid_sample_pair_list',
                                 [])[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'missing_sample_pair_list':
        all_missing_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'orphan_image_path_list':
        sorted(orphan_image_path_list),
        'check_error_message_list':
        check_error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }

    try:
        with open(save_check_result_path, 'w',
                  encoding='UTF-8') as save_json_file:
            json.dump(save_check_result_dict,
                      save_json_file,
                      ensure_ascii=False)
    except Exception as e:
        check_error_message_list.append(f'save check result failed {e}')

    return check_error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    file_copy_pair_list, archive_group_list, parquet_group_list = get_all_file_and_archive_group(
        root_dataset_path)

    print('1111', 'copy file', len(file_copy_pair_list), 'archive group',
          len(archive_group_list), 'parquet file', len(parquet_group_list))
    for per_archive_group in archive_group_list:
        print('1111', 'group', per_archive_group[0], 'parts',
              len(per_archive_group[4]))

    # 解压前预检: 分片数/连号/整包截断/parquet行数，任一不过直接中断，不白跑几十小时
    precheck_error_message_list = []
    precheck_error_message_list.extend(
        check_archive_group_complete(archive_group_list))
    precheck_error_message_list.extend(
        check_parquet_group_complete(parquet_group_list))

    print('1111', 'precheck error', precheck_error_message_list[:20])
    if precheck_error_message_list:
        raise Exception(f'precheck failed {precheck_error_message_list[:20]}')

    # 根目录的all_dataset_gpt_score.json由attach_gpt_score单独并进metadata，不重复拷
    file_copy_pair_list = [
        per_file_copy_pair for per_file_copy_pair in file_copy_pair_list
        if os.path.basename(per_file_copy_pair[0]) != GPT_SCORE_FILE_NAME
    ]

    copy_error_message_list = []
    copy_func = partial(process_single_file_copy,
                        save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_copy_result in tqdm(pool.imap_unordered(
                copy_func, file_copy_pair_list),
                                    total=len(file_copy_pair_list)):
            copy_error_message_list.extend(per_copy_result[1])

    extract_error_message_list = []
    extract_func = partial(process_single_archive_group,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_extract_result in tqdm(pool.imap_unordered(
                extract_func, archive_group_list),
                                       total=len(archive_group_list)):
            per_archive_group_key, per_member_count, per_file_member_count, per_extract_count, per_skip_count, per_metadata_count, per_error_message_list = per_extract_result

            print('2222', per_archive_group_key, 'member', per_member_count,
                  'file member', per_file_member_count, 'extract',
                  per_extract_count, 'skip', per_skip_count, 'metadata',
                  per_metadata_count, 'error', len(per_error_message_list))

            if per_error_message_list:
                extract_error_message_list.append(
                    f'{per_archive_group_key} extract error num '
                    f'{len(per_error_message_list)} '
                    f'first {per_error_message_list[0]}')

    parquet_error_message_list = []
    total_annotation_row_count, total_sample_pair_count = 0, 0
    total_global_instruction_turn_count = 0
    total_annotation_image_reference_count = 0
    total_single_reference_count, total_double_reference_count = 0, 0
    all_invalid_sample_pair_list = []

    parquet_func = partial(process_single_parquet_file,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_parquet_result in tqdm(pool.imap_unordered(
                parquet_func, parquet_group_list),
                                       total=len(parquet_group_list)):
            per_parquet_group_name, per_row_count, per_sample_pair_count, per_global_instruction_turn_count, per_image_reference_count, per_unique_image_count, per_single_reference_count, per_double_reference_count, per_invalid_sample_pair_list, per_error_message_list = per_parquet_result

            total_annotation_row_count += per_row_count
            total_sample_pair_count += per_sample_pair_count
            total_global_instruction_turn_count += per_global_instruction_turn_count
            total_annotation_image_reference_count += per_image_reference_count
            total_single_reference_count += per_single_reference_count
            total_double_reference_count += per_double_reference_count
            all_invalid_sample_pair_list.extend(
                per_invalid_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM])
            parquet_error_message_list.extend(per_error_message_list)

            print('2222', per_parquet_group_name, 'row', per_row_count,
                  'sample pair', per_sample_pair_count, 'global instruction',
                  per_global_instruction_turn_count, 'image reference',
                  per_image_reference_count, 'unique image',
                  per_unique_image_count, 'invalid',
                  len(per_invalid_sample_pair_list))

    gpt_score_count, gpt_score_error_message_list = attach_gpt_score(
        root_dataset_path, save_dataset_path)
    print('2222', 'gpt score', gpt_score_count)

    # parquet侧全量期望值硬对账
    if total_annotation_row_count != EXPECTED_TOTAL_ANNOTATION_ROW_COUNT:
        parquet_error_message_list.append(
            f'total annotation row count not match '
            f'{total_annotation_row_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_ROW_COUNT}')

    if total_sample_pair_count != EXPECTED_TOTAL_SAMPLE_PAIR_COUNT:
        parquet_error_message_list.append(
            f'total sample pair count not match {total_sample_pair_count} != '
            f'{EXPECTED_TOTAL_SAMPLE_PAIR_COUNT}')

    if total_global_instruction_turn_count != EXPECTED_TOTAL_GLOBAL_INSTRUCTION_TURN_COUNT:
        parquet_error_message_list.append(
            f'total global instruction turn count not match '
            f'{total_global_instruction_turn_count} != '
            f'{EXPECTED_TOTAL_GLOBAL_INSTRUCTION_TURN_COUNT}')

    if total_annotation_image_reference_count != EXPECTED_TOTAL_ANNOTATION_IMAGE_REFERENCE_COUNT:
        parquet_error_message_list.append(
            f'total annotation image reference count not match '
            f'{total_annotation_image_reference_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_IMAGE_REFERENCE_COUNT}')

    if total_single_reference_count != EXPECTED_SINGLE_REFERENCE_SAMPLE_PAIR_COUNT:
        parquet_error_message_list.append(
            f'single reference sample pair count not match '
            f'{total_single_reference_count} != '
            f'{EXPECTED_SINGLE_REFERENCE_SAMPLE_PAIR_COUNT}')

    if total_double_reference_count != EXPECTED_DOUBLE_REFERENCE_SAMPLE_PAIR_COUNT:
        parquet_error_message_list.append(
            f'double reference sample pair count not match '
            f'{total_double_reference_count} != '
            f'{EXPECTED_DOUBLE_REFERENCE_SAMPLE_PAIR_COUNT}')

    if all_invalid_sample_pair_list:
        parquet_error_message_list.append(
            f'invalid sample pair count {len(all_invalid_sample_pair_list)}')

    process_summary_dict = {
        'total_annotation_row_count': total_annotation_row_count,
        'total_sample_pair_count': total_sample_pair_count,
        'total_global_instruction_turn_count':
        total_global_instruction_turn_count,
        'total_annotation_image_reference_count':
        total_annotation_image_reference_count,
        'total_single_reference_count': total_single_reference_count,
        'total_double_reference_count': total_double_reference_count,
        'gpt_score_count': gpt_score_count,
        'invalid_sample_pair_list': all_invalid_sample_pair_list,
    }

    check_error_message_list = check_image_annotation_pair(
        save_dataset_path, process_summary_dict)

    all_error_message_list = []
    all_error_message_list.extend(copy_error_message_list)
    all_error_message_list.extend(extract_error_message_list)
    all_error_message_list.extend(parquet_error_message_list)
    all_error_message_list.extend(gpt_score_error_message_list)
    all_error_message_list.extend(check_error_message_list)

    print('3333', 'total error', len(all_error_message_list),
          all_error_message_list[:20])

    if all_error_message_list:
        raise Exception(f'preprocess failed {all_error_message_list[:20]}')

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/ImgEdit'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
