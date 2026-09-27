import os
import re
import json
import shutil
import tarfile

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.tar$'),
]

SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    'README.md',
]

ANNOTATION_FILE_SUFFIX_LIST = [
    '.json',
    '.txt',
]

# X2Edit-Dataset是纯图像编辑数据集(实测6026214个编辑样本对)，没有任何文生图子集。
# 根目录只有.cache/.gitattributes/README.md这三个无用项加一个X2Edit_data数据目录，
# X2Edit_data/<构造模型名>/<分片编号>/<压缩包编号>.tar共1396个tar，tar内没有任何顶层目录，
# 一个样本对固定是下面这几个同前缀文件:
#   000000000.1.0.jpg  参考图(编辑前原图)
#   000000000.1.1.jpg  第二参考图(只有textflux子集有，是文字前景mask)
#   000000000.2.jpg    编辑后图
#   000000000.json     该样本的全部属性(caption/instruction/task/model/各类质量分等)
#   000000000.txt      编辑指令，实测和json里的instruction字段完全一致，属冗余信息
# 实测10个子集只有两种文件组合: 4文件(无1.1.jpg)和5文件(textflux)，没有任何残缺组合。
# 一个完整样本对必须同时具备的文件后缀，1.1.jpg是textflux独有的第二参考图，可有可无
SAMPLE_REQUIRED_FILE_SUFFIX_LIST = [
    '1.0.jpg',
    '2.jpg',
    'json',
    'txt',
]

SAMPLE_OPTIONAL_FILE_SUFFIX_LIST = [
    '1.1.jpg',
]

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024


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
    """过滤掉.cache、.gitattributes、README.md这几个不需要整理的文件或目录"""
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


def get_sample_name_prefix(per_file_relative_path):
    """从图像或标注文件名中截取样本编号前缀

    该数据集一个样本对应000000000.1.0.jpg、000000000.2.jpg、000000000.json、000000000.txt这几个文件，
    textflux这类子集还会多出000000000.1.1.jpg这张第二参考图，样本编号是文件名中第一个小数点之前的部分。
    """
    per_file_relative_dir = os.path.dirname(per_file_relative_path)
    per_file_name = os.path.basename(per_file_relative_path)
    per_sample_name_prefix = per_file_name.split('.')[0]

    if per_file_relative_dir:
        return f'{per_file_relative_dir}/{per_sample_name_prefix}'

    return per_sample_name_prefix


def get_dedup_member_name(per_member_name, member_name_count_dict):
    """给压缩包内重复出现的成员名加上重复序号后缀，避免后写的文件把先写的覆盖掉

    该数据集kontext_subject/gpt4o/textflux等子集的tar是由多个tar直接拼接而成的，
    同一个tar里会出现两条同名但内容不同的成员(实测kontext_subject单个tar 19976个成员只有10516个唯一名)，
    如果都按原名落盘会丢掉将近一半样本。
    """
    per_member_repeat_index = member_name_count_dict.get(per_member_name, 0)
    member_name_count_dict[per_member_name] = per_member_repeat_index + 1
    if per_member_repeat_index == 0:
        return per_member_name

    # 000000000.1.0.jpg重复第1次时改名成000000000_dup1.1.0.jpg，
    # 保证同一个样本的图像和标注仍然共享同一个样本编号前缀
    per_member_dir_name = os.path.dirname(per_member_name)
    per_member_base_name = os.path.basename(per_member_name)
    per_sample_name_prefix = per_member_base_name.split('.')[0]
    per_member_remain_name = per_member_base_name[len(per_sample_name_prefix):]
    per_dedup_base_name = f'{per_sample_name_prefix}_dup{per_member_repeat_index}{per_member_remain_name}'

    if per_member_dir_name:
        return f'{per_member_dir_name}/{per_dedup_base_name}'

    return per_dedup_base_name


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非压缩包文件原样拷贝到目标目录，保持相对路径不变"""
    per_file_relative_path, per_file_path = file_copy_pair

    save_file_path = os.path.join(save_dataset_path, per_file_relative_path)
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    if os.path.exists(save_file_path) and os.path.getsize(
            save_file_path) == os.path.getsize(per_file_path):
        return

    try:
        with open(per_file_path, 'rb') as load_file:
            with open(save_file_path, 'wb') as save_file:
                shutil.copyfileobj(load_file, save_file, COPY_FILE_BLOCK_SIZE)
    except Exception as e:
        print('4444', per_file_path, e)

    return


def process_single_archive_group(archive_group, save_dataset_path):
    """流式解压单个压缩包组(可能由多个按字节切分的分片组成)

    该数据集的压缩包内没有顶层目录，为了保证各子集之间不互相覆盖，额外建一层以压缩包名命名的子目录。
    """
    per_archive_group_name, per_archive_relative_dir, per_archive_part_path_list = archive_group

    save_archive_dir_path = os.path.join(save_dataset_path,
                                         per_archive_relative_dir,
                                         per_archive_group_name)
    os.makedirs(save_archive_dir_path, exist_ok=True)

    extract_file_count, skip_file_count = 0, 0
    member_name_count_dict, dedup_member_count = {}, 0
    empty_member_count = 0
    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    try:
        # 该数据集部分tar是多个完整tar直接拼接而成，拼接处保留了前一个tar的全零结束块。
        # tarfile默认遇到全零块就认为文件结束，会直接丢掉后面所有成员，
        # 所以必须开ignore_zeros=True跳过中间的结束块继续读完整个字节流。
        with tarfile.open(fileobj=archive_reader,
                          mode='r|*',
                          ignore_zeros=True) as load_tar_file:

            for per_member in load_tar_file:
                per_member_name = per_member.name.replace('\\',
                                                          '/').lstrip('/')
                per_member_name = os.path.normpath(per_member_name)
                if per_member_name.startswith('..'):
                    print('5555', per_archive_group_name, per_member.name)
                    continue

                if check_skip_file_or_dir(per_member_name):
                    continue

                if per_member.isdir():
                    os.makedirs(os.path.join(save_archive_dir_path,
                                             per_member_name),
                                exist_ok=True)
                    continue

                if not per_member.isfile():
                    continue

                # 该数据集部分tar是多个tar拼接而成，同名成员要改名保存，否则后写的会覆盖先写的
                per_dedup_member_name = get_dedup_member_name(
                    per_member_name.replace('\\', '/'), member_name_count_dict)
                if per_dedup_member_name != per_member_name:
                    dedup_member_count += 1
                per_member_name = per_dedup_member_name

                save_member_path = os.path.join(save_archive_dir_path,
                                                per_member_name)

                if os.path.exists(save_member_path) and os.path.getsize(
                        save_member_path) == per_member.size:
                    skip_file_count += 1
                    continue

                os.makedirs(os.path.dirname(save_member_path), exist_ok=True)

                load_member_file = load_tar_file.extractfile(per_member)
                if load_member_file is None:
                    print('6666', per_archive_group_name, per_member.name)
                    continue

                with open(save_member_path, 'wb') as save_member_file:
                    shutil.copyfileobj(load_member_file, save_member_file,
                                       EXTRACT_FILE_BLOCK_SIZE)

                # bagel子集有大量0字节的.txt(对应json里instruction也是空串)，
                # 这类样本没有编辑指令，属于原始数据本身的缺陷，这里只统计不丢弃
                if per_member.size == 0:
                    empty_member_count += 1

                extract_file_count += 1

    except Exception as e:
        # 分片不全或压缩包截断时保留已解压出的文件，不中断整体流程
        print('7777', per_archive_group_name, len(per_archive_part_path_list),
              e)
    finally:
        archive_reader.close()

    if dedup_member_count > 0:
        print('3333', per_archive_group_name, 'duplicate member name:',
              dedup_member_count)

    return [
        per_archive_group_name,
        extract_file_count,
        skip_file_count,
        dedup_member_count,
        empty_member_count,
    ]


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表和按分片归组后的压缩包列表"""
    file_copy_pair_list = []
    archive_part_path_dict = {}
    for per_root_path, _, per_file_name_list in os.walk(root_dataset_path):
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


def get_sample_file_suffix(per_file_name):
    """取样本编号之后的完整后缀，例如000000000.1.0.jpg返回1.0.jpg、000000000.json返回json

    不能用os.path.splitext，因为.1.0.jpg和.1.1.jpg只差中间一段，
    只看最后一个扩展名会把参考图和第二参考图混成同一种文件。
    """
    return '.'.join(os.path.basename(per_file_name).split('.')[1:])


def check_single_subset_dir(subset_check_pair):
    """按样本对逐个校验解压结果是否完整

    一个完整样本对必须同时有.1.0.jpg参考图、.2.jpg编辑结果图、.json属性和.txt指令，
    textflux子集还会多一张.1.1.jpg第二参考图(可选)。
    这里必须逐个后缀比对，不能只统计"标注前缀数"和"图像文件总数"，
    否则一个样本缺了.2.jpg但还留着.1.0.jpg时会被当成完整样本漏过去。
    """
    per_subset_relative_path, per_subset_path = subset_check_pair

    sample_file_dict = {}
    total_file_count, empty_file_count = 0, 0
    empty_file_relative_path_list = []
    for per_root_path, _, per_file_name_list in os.walk(per_subset_path):
        for per_file_name in per_file_name_list:
            per_file_path = os.path.join(per_root_path, per_file_name)
            per_file_relative_path = os.path.relpath(per_file_path,
                                                     per_subset_path)
            per_file_relative_path = per_file_relative_path.replace('\\', '/')

            total_file_count += 1
            if os.path.getsize(per_file_path) == 0:
                empty_file_count += 1
                if len(empty_file_relative_path_list) < 100:
                    empty_file_relative_path_list.append(
                        f'{per_subset_relative_path}/{per_file_relative_path}')

            per_sample_name_prefix = get_sample_name_prefix(
                per_file_relative_path)
            per_file_suffix = get_sample_file_suffix(per_file_name)
            sample_file_dict.setdefault(per_sample_name_prefix,
                                        set()).add(per_file_suffix)

    complete_sample_count = 0
    incomplete_sample_info_list, unknown_file_suffix_list = [], []
    for per_sample_name_prefix, per_file_suffix_set in sample_file_dict.items(
    ):
        per_missing_file_suffix_list = [
            per_file_suffix
            for per_file_suffix in SAMPLE_REQUIRED_FILE_SUFFIX_LIST
            if per_file_suffix not in per_file_suffix_set
        ]
        if len(per_missing_file_suffix_list) == 0:
            complete_sample_count += 1
        elif len(incomplete_sample_info_list) < 1000:
            incomplete_sample_info_list.append([
                f'{per_subset_relative_path}/{per_sample_name_prefix}',
                sorted(per_missing_file_suffix_list),
                sorted(per_file_suffix_set),
            ])

        for per_file_suffix in per_file_suffix_set:
            if per_file_suffix in SAMPLE_REQUIRED_FILE_SUFFIX_LIST:
                continue
            if per_file_suffix in SAMPLE_OPTIONAL_FILE_SUFFIX_LIST:
                continue
            if len(unknown_file_suffix_list) < 100:
                unknown_file_suffix_list.append([
                    f'{per_subset_relative_path}/{per_sample_name_prefix}',
                    per_file_suffix,
                ])

    return [
        per_subset_relative_path,
        len(sample_file_dict),
        complete_sample_count,
        total_file_count,
        empty_file_count,
        incomplete_sample_info_list,
        unknown_file_suffix_list,
        empty_file_relative_path_list,
    ]


def check_image_annotation_pair(save_dataset_path):
    """按样本对汇总校验解压结果

    该数据集是X2Edit_data/<构造模型名>/<分片编号>/<压缩包名>这种四层结构，
    每个压缩包解开后的目录里一个样本对应两张(textflux是三张)图和一个json加一个txt。
    """

    root_data_path = os.path.join(save_dataset_path, 'X2Edit_data')

    if not os.path.exists(root_data_path):
        print('8888', root_data_path)
        return

    subset_check_pair_list = []
    for per_edit_type_name in sorted(os.listdir(root_data_path)):
        per_edit_type_path = os.path.join(root_data_path, per_edit_type_name)
        if not os.path.isdir(per_edit_type_path):
            continue

        for per_shard_name in sorted(os.listdir(per_edit_type_path)):
            per_shard_path = os.path.join(per_edit_type_path, per_shard_name)
            if not os.path.isdir(per_shard_path):
                continue

            for per_subset_name in sorted(os.listdir(per_shard_path)):
                per_subset_path = os.path.join(per_shard_path, per_subset_name)
                if not os.path.isdir(per_subset_path):
                    continue

                subset_check_pair_list.append([
                    f'{per_edit_type_name}/{per_shard_name}/{per_subset_name}',
                    per_subset_path,
                ])

    total_sample_count, total_complete_sample_count = 0, 0
    total_file_count, total_empty_file_count = 0, 0
    incomplete_sample_info_dict = {}
    unknown_file_suffix_dict, empty_file_relative_path_dict = {}, {}
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_subset_dir, subset_check_pair_list),
                                     total=len(subset_check_pair_list)):
            per_subset_relative_path, per_sample_count, per_complete_sample_count, per_file_count, per_empty_file_count, per_incomplete_sample_info_list, per_unknown_file_suffix_list, per_empty_file_relative_path_list = per_check_result

            total_sample_count += per_sample_count
            total_complete_sample_count += per_complete_sample_count
            total_file_count += per_file_count
            total_empty_file_count += per_empty_file_count

            if len(per_incomplete_sample_info_list) > 0:
                print('2222', per_subset_relative_path, 'sample:',
                      per_sample_count, 'incomplete sample:',
                      per_sample_count - per_complete_sample_count)
                incomplete_sample_info_dict[
                    per_subset_relative_path] = per_incomplete_sample_info_list

            if len(per_unknown_file_suffix_list) > 0:
                unknown_file_suffix_dict[
                    per_subset_relative_path] = per_unknown_file_suffix_list

            if len(per_empty_file_relative_path_list) > 0:
                empty_file_relative_path_dict[
                    per_subset_relative_path] = per_empty_file_relative_path_list

    print('3333', 'total sample:', total_sample_count, 'complete sample:',
          total_complete_sample_count, 'incomplete sample:',
          total_sample_count - total_complete_sample_count, 'total file:',
          total_file_count, 'empty file:', total_empty_file_count)

    save_check_result_path = os.path.join(save_dataset_path,
                                          'unzip_check_sample_result.json')
    save_check_result_dict = {
        'total_sample_count': total_sample_count,
        'complete_sample_count': total_complete_sample_count,
        'incomplete_sample_count':
        total_sample_count - total_complete_sample_count,
        'total_file_count': total_file_count,
        'empty_file_count': total_empty_file_count,
        'incomplete_sample_info_dict': incomplete_sample_info_dict,
        'unknown_file_suffix_dict': unknown_file_suffix_dict,
        'empty_file_relative_path_dict': empty_file_relative_path_dict,
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    return


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    file_copy_pair_list, archive_group_list = get_all_file_and_archive_group(
        root_dataset_path)

    print('1111', len(file_copy_pair_list), len(archive_group_list))
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(archive_group_list) > 0:
        print('1111', archive_group_list[0][0], archive_group_list[0][1],
              len(archive_group_list[0][2]))

    copy_func = partial(process_single_file_copy,
                        save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        list(
            tqdm(pool.imap_unordered(copy_func, file_copy_pair_list),
                 total=len(file_copy_pair_list)))

    extract_func = partial(process_single_archive_group,
                           save_dataset_path=save_dataset_path)
    total_extract_file_count, total_dedup_member_count = 0, 0
    total_empty_member_count = 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_extract_result in tqdm(pool.imap_unordered(
                extract_func, archive_group_list),
                                       total=len(archive_group_list)):
            total_extract_file_count += per_extract_result[1]
            total_dedup_member_count += per_extract_result[3]
            total_empty_member_count += per_extract_result[4]
            if per_extract_result[1] == 0 and per_extract_result[2] == 0:
                # 一个tar一个文件都没解出来说明这个tar根本没读进去，必须显式报出来
                print('2222', per_extract_result)

    print('1111', 'total extract file:', total_extract_file_count,
          'total dedup member:', total_dedup_member_count,
          'total empty member:', total_empty_member_count)

    check_image_annotation_pair(save_dataset_path)


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/X2Edit-Dataset'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
