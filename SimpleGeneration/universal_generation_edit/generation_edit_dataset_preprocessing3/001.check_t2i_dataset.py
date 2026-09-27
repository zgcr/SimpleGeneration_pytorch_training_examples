import os
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

# 通用文生图数据集数据完整性复核脚本
# t2i_datasets下每个一级子目录都是一个文生图数据集,且文件组织结构完全一致:
# t2i_datasets/<dataset_dir_name>/<set_name>/<folder_name>/*.jpg
# t2i_datasets/<dataset_dir_name>/<set_name>/<folder_name>.json
# 每个<folder_name>.json与同级同名的<folder_name>文件夹一一对应,json内容形如:
# {"<save_image_name>.jpg": {"width": int, "height": int,
#                            "t2i_caption": str, "t2i_caption_length": int}}
# 本脚本只做只读复核,不对数据集做任何修改(全程只有os.listdir/os.path/只读open/
# json.load,复核报告一律写到数据集目录之外的save_check_result_path下)
# 每次运行只复核一个数据集,运行前修改__main__里的root_dataset_path即可

# 落盘标注固定只有这4个key,多一个少一个都算标注key-value对不完整
ANNOTATION_KEY_NAME_LIST = [
    'width',
    'height',
    't2i_caption',
    't2i_caption_length',
]

SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 图像完整性校验级别:
# exists: 只查图像文件存在且字节数大于0,最快,查不出内容损坏
# header: 只读文件头拿到真实宽高并与标注比对,再读文件尾2字节校验jpeg结束标记
#         (FFD9),每张图只有几次小IO,能查出丢失/零字节/文件头损坏/写盘截断,
#         千万级图像量也能全量跑完,默认用这一档
# decode: 完整cv2解码并比对shape,最彻底但最慢(千万级图像量要跑几十小时)
CHECK_IMAGE_LEVEL = 'header'

VALID_CHECK_IMAGE_LEVEL_LIST = [
    'exists',
    'header',
    'decode',
]

# jpeg文件结束标记,写盘被截断的图像尾部一定不是这两个字节
JPEG_END_MARKER_BYTES = b'\xff\xd9'

PROCESS_NUM = 32

# 上游resave时每个文件夹放满10000张(每个子集最后一个文件夹允许不满),
# 文件夹不满不算数据损坏,只作为warning上报
PER_FOLDER_IMAGE_NUM = 10000

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 各数据集上游的caption长度上限并不一致(FaceCaption-1M取200,GPIC取1024),
# 所以通用复核脚本只校验caption非空以及记录长度与实际长度自洽,不校验长度上限
MIN_CAPTION_LENGTH = 1

# 落盘图像名形如<dataset_name>_<set_name>_<image_name_prefix>.jpg,
# 所以图像名里一定包含_<set_name>_这一段,用这个做通用的图像名规范校验
CHECK_IMAGE_NAME_CONTAIN_SET_NAME = True

# 只复核指定子集(留空表示复核该数据集下的全部子集)
CHECK_SET_NAME_LIST = []

# 每一类问题最多在报告里保留多少条样例信息,避免坏样本极多时报告体积爆炸
MAX_ERROR_MESSAGE_NUM = 100

# 复核结果按问题类别分类统计,每一类既统计条数也保留样例信息
CHECK_ERROR_TYPE_NAME_LIST = [
    # 标注json自身无法解析(整个分片的标注都不可用)
    'broken_annotation_json',
    # <folder_name>文件夹与<folder_name>.json没有一一对应
    'folder_json_not_paired',
    # 标注里记录的样本对在磁盘上找不到图像文件
    # (既是"每个样本对图像文件是否存在"的失败项,也就是孤立标注)
    'orphan_annotation',
    # 磁盘上的图像文件在标注json里找不到对应标注
    'orphan_image',
    # 图像文件存在但内容不完整(零字节/文件头损坏/写盘截断/无法解码)
    'broken_image',
    # 图像真实宽高与标注里记录的width/height不一致
    'image_size_not_match',
    # 标注的key-value对不完整(缺key/多key/类型非法/取值非法/长度不自洽)
    'invalid_annotation',
    # 图像文件名不符合落盘命名规范
    'invalid_image_name',
    # 分片文件夹里混入了非.jpg的残留文件
    'unexpected_file',
]


def check_single_image_file(per_image_path, per_annotation_image_w,
                            per_annotation_image_h):
    """按CHECK_IMAGE_LEVEL校验单个图像文件是否存在且完整,并比对真实宽高

    只返回第一个命中的问题,返回[错误类别, 错误信息],完全正常时返回[None, None]。
    上游resave时图像都是cv2重编码写盘的,不带EXIF方向信息,所以PIL读文件头拿到的
    宽高与cv2解码后的shape一定一致,两种校验级别得到的宽高口径相同。
    """
    try:
        per_image_file_size = os.path.getsize(per_image_path)
    except Exception as e:
        return ['broken_image', f'{per_image_path} get file size error {e}']

    if per_image_file_size <= 0:
        return ['broken_image', f'{per_image_path} file size is 0']

    if CHECK_IMAGE_LEVEL == 'exists':
        return [None, None]

    per_image_w, per_image_h = 0, 0
    if CHECK_IMAGE_LEVEL == 'header':
        # PIL的Image.open是惰性的,只读文件头就能拿到宽高,不解码整张图像
        try:
            with Image.open(per_image_path) as per_image:
                per_image_w, per_image_h = per_image.size
        except Exception as e:
            return [
                'broken_image', f'{per_image_path} read image header error {e}'
            ]

        # 文件头正常但写盘被截断的图像,尾部不会是jpeg的结束标记
        if per_image_path.endswith('.jpg') or per_image_path.endswith('.jpeg'):
            try:
                with open(per_image_path, 'rb') as load_image_file:
                    load_image_file.seek(-len(JPEG_END_MARKER_BYTES),
                                         os.SEEK_END)
                    per_image_end_bytes = load_image_file.read(
                        len(JPEG_END_MARKER_BYTES))
            except Exception as e:
                return [
                    'broken_image',
                    f'{per_image_path} read image end bytes error {e}'
                ]

            if per_image_end_bytes != JPEG_END_MARKER_BYTES:
                return [
                    'broken_image',
                    f'{per_image_path} jpeg end marker not match'
                ]
    else:
        try:
            per_image = cv2.imdecode(
                np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return ['broken_image', f'{per_image_path} decode image error {e}']

        if per_image is None or per_image.ndim != 3 or per_image.shape[2] != 3:
            return ['broken_image', f'{per_image_path} decode image failed']

        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    if per_image_w <= 0 or per_image_h <= 0:
        return [
            'broken_image',
            f'{per_image_path} image shape illegal {per_image_w} {per_image_h}'
        ]

    # 标注里的width/height缺失或类型非法时已经在标注校验里报过,这里不再重复比对
    # (bool是int的子类,必须显式排除,否则True会被当成1参与比对)
    if not isinstance(per_annotation_image_w, int) or isinstance(
            per_annotation_image_w,
            bool) or not isinstance(per_annotation_image_h, int) or isinstance(
                per_annotation_image_h, bool):
        return [None, None]

    if per_image_w != per_annotation_image_w or per_image_h != per_annotation_image_h:
        return [
            'image_size_not_match',
            f'{per_image_path} image size not match real {per_image_w} {per_image_h} annotation {per_annotation_image_w} {per_annotation_image_h}'
        ]

    return [None, None]


def check_single_annotation(per_set_name, per_image_name, per_annotation):
    """校验单个样本对标注的key-value对是否完整合法,返回错误信息列表

    只做标注自身的完整性和自洽性校验,不涉及磁盘上的图像文件。
    """
    check_error_message_list = []

    if not isinstance(per_annotation, dict):
        check_error_message_list.append(
            f'{per_set_name}/{per_image_name} annotation is not a dict')

        return check_error_message_list

    # 缺key和多key都算标注key-value对不完整
    for per_key_name in ANNOTATION_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            check_error_message_list.append(
                f'{per_set_name}/{per_image_name} annotation key {per_key_name} not exists'
            )
    for per_key_name in per_annotation.keys():
        if per_key_name not in ANNOTATION_KEY_NAME_LIST:
            check_error_message_list.append(
                f'{per_set_name}/{per_image_name} annotation has unexpected key {per_key_name}'
            )

    if len(check_error_message_list) > 0:
        return check_error_message_list

    per_image_w = per_annotation['width']
    per_image_h = per_annotation['height']
    per_t2i_caption = per_annotation['t2i_caption']
    per_t2i_caption_length = per_annotation['t2i_caption_length']

    # bool是int的子类,这里必须显式把bool判为非法类型
    if not isinstance(per_image_w, int) or isinstance(
            per_image_w,
            bool) or not isinstance(per_image_h, int) or isinstance(
                per_image_h, bool):
        check_error_message_list.append(
            f'{per_set_name}/{per_image_name} annotation width height type illegal {type(per_image_w)} {type(per_image_h)}'
        )
    elif per_image_w <= 0 or per_image_h <= 0:
        check_error_message_list.append(
            f'{per_set_name}/{per_image_name} annotation width height illegal {per_image_w} {per_image_h}'
        )
    else:
        if min(per_image_w, per_image_h) < MIN_IMAGE_SHORT_SIDE:
            check_error_message_list.append(
                f'{per_set_name}/{per_image_name} annotation image short side not match {per_image_w} {per_image_h}'
            )
        if max(per_image_w / per_image_h,
               per_image_h / per_image_w) > MAX_IMAGE_ASPECT_RATIO:
            check_error_message_list.append(
                f'{per_set_name}/{per_image_name} annotation image aspect ratio not match {per_image_w} {per_image_h}'
            )

    if not isinstance(per_t2i_caption, str):
        check_error_message_list.append(
            f'{per_set_name}/{per_image_name} annotation t2i_caption type illegal {type(per_t2i_caption)}'
        )
    else:
        if len(per_t2i_caption.strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_set_name}/{per_image_name} annotation t2i_caption is empty'
            )
        # 记录的描述长度必须和描述字符串的实际长度对得上
        if not isinstance(per_t2i_caption_length, int) or isinstance(
                per_t2i_caption_length, bool):
            check_error_message_list.append(
                f'{per_set_name}/{per_image_name} annotation t2i_caption_length type illegal {type(per_t2i_caption_length)}'
            )
        elif per_t2i_caption_length != len(per_t2i_caption):
            check_error_message_list.append(
                f'{per_set_name}/{per_image_name} annotation t2i_caption_length not match {per_t2i_caption_length} != {len(per_t2i_caption)}'
            )

    return check_error_message_list


def check_single_image_name(per_set_name, per_image_name):
    """校验单个落盘图像名是否符合命名规范,返回错误信息列表"""
    check_error_message_list = []

    if not per_image_name.endswith(SAVE_IMAGE_NAME_SUFFIX):
        check_error_message_list.append(
            f'{per_set_name}/{per_image_name} image name suffix not match')
    if per_image_name != per_image_name.lower():
        check_error_message_list.append(
            f'{per_set_name}/{per_image_name} image name not all lower case')
    if CHECK_IMAGE_NAME_CONTAIN_SET_NAME and f'_{per_set_name.lower()}_' not in per_image_name:
        check_error_message_list.append(
            f'{per_set_name}/{per_image_name} image name not contain set name')

    return check_error_message_list


def get_empty_check_result_dict():
    """初始化单次复核的计数与样例信息容器"""
    check_result_dict = {
        'total_annotation_count': 0,
        'total_disk_image_count': 0,
        'checked_image_count': 0,
        'valid_image_annotation_pair_count': 0,
    }
    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        check_result_dict[f'{per_error_type_name}_count'] = 0
        check_result_dict[f'{per_error_type_name}_message_list'] = []

    return check_result_dict


def add_check_error_message(check_result_dict, per_error_type_name,
                            per_error_message):
    """按问题类别累计条数,样例信息只保留前MAX_ERROR_MESSAGE_NUM条"""
    check_result_dict[f'{per_error_type_name}_count'] += 1
    per_error_message_list = check_result_dict[
        f'{per_error_type_name}_message_list']
    if len(per_error_message_list) < MAX_ERROR_MESSAGE_NUM:
        per_error_message_list.append(per_error_message)

    return


def merge_check_result_dict(check_result_dict, per_folder_check_result_dict):
    """把单个分片的复核结果合并进数据集级别的复核结果"""
    for per_key_name in [
            'total_annotation_count',
            'total_disk_image_count',
            'checked_image_count',
            'valid_image_annotation_pair_count',
    ]:
        check_result_dict[per_key_name] += per_folder_check_result_dict[
            per_key_name]

    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        check_result_dict[f'{per_error_type_name}_count'] += (
            per_folder_check_result_dict[f'{per_error_type_name}_count'])
        per_error_message_list = check_result_dict[
            f'{per_error_type_name}_message_list']
        for per_error_message in per_folder_check_result_dict[
                f'{per_error_type_name}_message_list']:
            if len(per_error_message_list) >= MAX_ERROR_MESSAGE_NUM:
                break
            per_error_message_list.append(per_error_message)

    return check_result_dict


def get_check_error_count(check_result_dict):
    """统计所有问题类别的问题总条数"""
    check_error_count = 0
    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        check_error_count += check_result_dict[f'{per_error_type_name}_count']

    return check_error_count


def get_all_check_folder_pair(root_dataset_path):
    """扫描数据集目录结构,配出所有<folder_name>文件夹与<folder_name>.json的任务对

    这一步只做目录层面的结构校验(文件夹与json是否一一对应、有没有杂项文件),
    单个分片内部的逐样本校验留到后面多进程里做。
    """
    structure_check_result_dict = get_empty_check_result_dict()
    check_warning_message_list = []

    folder_check_pair_list = []
    set_folder_count_dict = {}

    try:
        per_dataset_dir_name_list = sorted(os.listdir(root_dataset_path))
    except Exception as e:
        raise Exception(f'{root_dataset_path} listdir error {e}')

    for per_set_name in per_dataset_dir_name_list:
        per_set_dir_path = os.path.join(root_dataset_path, per_set_name)

        # 数据集根目录下除子集目录外还可能有resave时留下的统计json等杂项文件,
        # 这些文件不影响样本对完整性,只作为warning上报
        if not os.path.isdir(per_set_dir_path):
            check_warning_message_list.append(
                f'{per_set_name} is not a set dir')
            continue

        if len(CHECK_SET_NAME_LIST
               ) > 0 and per_set_name not in CHECK_SET_NAME_LIST:
            continue

        try:
            per_set_sub_name_list = sorted(os.listdir(per_set_dir_path))
        except Exception as e:
            add_check_error_message(structure_check_result_dict,
                                    'folder_json_not_paired',
                                    f'{per_set_name} listdir error {e}')
            continue

        per_set_folder_name_list = []
        per_set_json_name_prefix_list = []
        for per_sub_name in per_set_sub_name_list:
            per_sub_path = os.path.join(per_set_dir_path, per_sub_name)
            if os.path.isdir(per_sub_path):
                per_set_folder_name_list.append(per_sub_name)
            elif per_sub_name.endswith('.json'):
                per_set_json_name_prefix_list.append(
                    os.path.splitext(per_sub_name)[0])
            else:
                add_check_error_message(
                    structure_check_result_dict, 'unexpected_file',
                    f'{per_set_name}/{per_sub_name} is an unexpected file in set dir'
                )

        if len(per_set_folder_name_list) == 0 and len(
                per_set_json_name_prefix_list) == 0:
            check_warning_message_list.append(f'{per_set_name} set dir empty')
            continue

        per_set_folder_name_set = set(per_set_folder_name_list)
        per_set_json_name_prefix_set = set(per_set_json_name_prefix_list)

        # 有文件夹没json: 这个分片的全部图像都是孤立图像,无法逐样本复核
        for per_folder_name in per_set_folder_name_list:
            if per_folder_name not in per_set_json_name_prefix_set:
                add_check_error_message(
                    structure_check_result_dict, 'folder_json_not_paired',
                    f'{per_set_name}/{per_folder_name} annotation json not exists'
                )

        # 有json没文件夹: 这个分片的全部标注都是孤立标注,无法逐样本复核
        for per_json_name_prefix in per_set_json_name_prefix_list:
            if per_json_name_prefix not in per_set_folder_name_set:
                add_check_error_message(
                    structure_check_result_dict, 'folder_json_not_paired',
                    f'{per_set_name}/{per_json_name_prefix} image folder not exists'
                )

        # 只有文件夹和json都在的分片才进入逐样本复核
        per_set_paired_folder_name_list = sorted(
            per_set_folder_name_set
            & per_set_json_name_prefix_set)
        set_folder_count_dict[per_set_name] = len(
            per_set_paired_folder_name_list)

        for per_folder_index, per_folder_name in enumerate(
                per_set_paired_folder_name_list):
            # 每个子集最后一个分片允许不满PER_FOLDER_IMAGE_NUM张
            per_is_last_folder = (
                per_folder_index == len(per_set_paired_folder_name_list) - 1)
            folder_check_pair_list.append([
                per_set_name,
                per_folder_name,
                os.path.join(per_set_dir_path, per_folder_name),
                os.path.join(per_set_dir_path, f'{per_folder_name}.json'),
                per_is_last_folder,
            ])

    return [
        folder_check_pair_list,
        set_folder_count_dict,
        structure_check_result_dict,
        check_warning_message_list,
    ]


def check_single_folder(folder_check_pair):
    """复核单个分片: 标注与磁盘图像双向比对,并逐样本校验图像完整性与标注完整性

    上游图像都放在NAS上,逐样本打一次os.path.exists就是一次网络往返,千万级样本
    就要打上千万次。实测一个分片的全部图像都落在同一个文件夹下,所以这里每个分片
    只做一次os.listdir,之后所有存在性判定都退化成集合查表,网络往返次数从
    "样本条数"降到"分片数"。
    """
    per_set_name, per_folder_name, per_folder_dir_path, per_folder_json_path, per_is_last_folder = folder_check_pair

    per_folder_check_result_dict = get_empty_check_result_dict()
    per_folder_check_warning_message_list = []

    # 标注json无法解析时整个分片的标注都不可用,只上报一条问题后直接返回,
    # 此时磁盘上的图像既无法判定是否孤立,也无法逐样本复核
    try:
        with open(per_folder_json_path, 'r',
                  encoding='UTF-8') as load_json_file:
            per_folder_annotation_dict = json.load(load_json_file)
    except Exception as e:
        add_check_error_message(
            per_folder_check_result_dict, 'broken_annotation_json',
            f'{per_set_name}/{per_folder_name}.json load error {e}')

        return [
            per_set_name,
            per_folder_name,
            per_folder_check_result_dict,
            per_folder_check_warning_message_list,
        ]

    if not isinstance(per_folder_annotation_dict, dict):
        add_check_error_message(
            per_folder_check_result_dict, 'broken_annotation_json',
            f'{per_set_name}/{per_folder_name}.json is not a dict')

        return [
            per_set_name,
            per_folder_name,
            per_folder_check_result_dict,
            per_folder_check_warning_message_list,
        ]

    try:
        per_folder_file_name_list = sorted(os.listdir(per_folder_dir_path))
    except Exception as e:
        add_check_error_message(
            per_folder_check_result_dict, 'folder_json_not_paired',
            f'{per_set_name}/{per_folder_name} listdir error {e}')

        return [
            per_set_name,
            per_folder_name,
            per_folder_check_result_dict,
            per_folder_check_warning_message_list,
        ]

    per_folder_image_name_list = []
    for per_file_name in per_folder_file_name_list:
        if per_file_name.endswith(SAVE_IMAGE_NAME_SUFFIX):
            per_folder_image_name_list.append(per_file_name)
        else:
            # 分片文件夹里只应该有.jpg图像,其他文件/子目录都是残留
            add_check_error_message(
                per_folder_check_result_dict, 'unexpected_file',
                f'{per_set_name}/{per_folder_name}/{per_file_name} is an unexpected file in image folder'
            )

    # 图像文件存在性一律按磁盘上的全部文件名判定,而不是只按.jpg文件名判定:
    # 标注key万一是个非.jpg的名字,这种情况该报的是图像名不规范和残留文件,
    # 不能再顺带误报成孤立标注(否则同一个问题会被重复计成好几类)
    per_folder_file_name_set = set(per_folder_file_name_list)
    per_folder_image_name_set = set(per_folder_image_name_list)
    per_folder_annotation_image_name_set = set(
        per_folder_annotation_dict.keys())

    per_folder_check_result_dict['total_annotation_count'] = len(
        per_folder_annotation_image_name_set)
    per_folder_check_result_dict['total_disk_image_count'] = len(
        per_folder_image_name_set)

    # 除每个子集最后一个分片外都应该是满PER_FOLDER_IMAGE_NUM张,
    # 不满不代表样本对损坏,只作为warning上报
    if not per_is_last_folder and len(
            per_folder_annotation_image_name_set) != PER_FOLDER_IMAGE_NUM:
        per_folder_check_warning_message_list.append(
            f'{per_set_name}/{per_folder_name} annotation num not full {len(per_folder_annotation_image_name_set)} != {PER_FOLDER_IMAGE_NUM}'
        )

    # 磁盘上有图像但标注json里没有对应标注,即孤立图像
    for per_image_name in sorted(per_folder_image_name_set -
                                 per_folder_annotation_image_name_set):
        add_check_error_message(
            per_folder_check_result_dict, 'orphan_image',
            f'{per_set_name}/{per_folder_name}/{per_image_name} image has no annotation'
        )

    for per_image_name in sorted(per_folder_annotation_image_name_set):
        per_annotation = per_folder_annotation_dict[per_image_name]

        per_annotation_error_message_list = check_single_annotation(
            per_set_name, per_image_name, per_annotation)
        for per_error_message in per_annotation_error_message_list:
            add_check_error_message(per_folder_check_result_dict,
                                    'invalid_annotation', per_error_message)

        per_image_name_error_message_list = check_single_image_name(
            per_set_name, per_image_name)
        for per_error_message in per_image_name_error_message_list:
            add_check_error_message(per_folder_check_result_dict,
                                    'invalid_image_name', per_error_message)

        # 标注json里有标注但磁盘上没有对应图像文件,即孤立标注
        if per_image_name not in per_folder_file_name_set:
            add_check_error_message(
                per_folder_check_result_dict, 'orphan_annotation',
                f'{per_set_name}/{per_folder_name}/{per_image_name} annotation has no image file'
            )
            continue

        per_annotation_image_w, per_annotation_image_h = None, None
        if isinstance(per_annotation, dict):
            per_annotation_image_w = per_annotation.get('width', None)
            per_annotation_image_h = per_annotation.get('height', None)

        per_image_path = os.path.join(per_folder_dir_path, per_image_name)
        per_image_error_type_name, per_image_error_message = check_single_image_file(
            per_image_path, per_annotation_image_w, per_annotation_image_h)
        per_folder_check_result_dict['checked_image_count'] += 1

        if per_image_error_type_name is not None:
            add_check_error_message(per_folder_check_result_dict,
                                    per_image_error_type_name,
                                    per_image_error_message)
            continue

        # 图像文件存在且完整、标注key-value对完整、图像名规范,才算一个完好的样本对
        if len(per_annotation_error_message_list) == 0 and len(
                per_image_name_error_message_list) == 0:
            per_folder_check_result_dict[
                'valid_image_annotation_pair_count'] += 1

    return [
        per_set_name,
        per_folder_name,
        per_folder_check_result_dict,
        per_folder_check_warning_message_list,
    ]


def check_dataset(root_dataset_path, save_check_result_path):
    dataset_dir_name = os.path.basename(root_dataset_path.rstrip('/'))

    # 校验级别写错时必须直接失败,不能静默退化成某一档校验后给出"复核通过"的结论
    if CHECK_IMAGE_LEVEL not in VALID_CHECK_IMAGE_LEVEL_LIST:
        raise Exception(f'{CHECK_IMAGE_LEVEL} is an illegal check image level')

    if not os.path.isdir(root_dataset_path):
        raise Exception(f'{root_dataset_path} is not a dir')

    # 复核报告一律写到数据集目录之外,保证本脚本不对数据集做任何修改
    if os.path.abspath(save_check_result_path).startswith(
            os.path.abspath(root_dataset_path) + os.sep):
        raise Exception(
            f'{save_check_result_path} must be outside {root_dataset_path}')

    folder_check_pair_list, set_folder_count_dict, check_result_dict, check_warning_message_list = get_all_check_folder_pair(
        root_dataset_path)

    print('1111', dataset_dir_name, 'check set:', len(set_folder_count_dict),
          'check folder:', len(folder_check_pair_list), 'structure error:',
          get_check_error_count(check_result_dict), 'structure warning:',
          len(check_warning_message_list))

    for per_check_warning_message in check_warning_message_list:
        print('2222', per_check_warning_message)

    per_set_check_result_dict = {}
    for per_set_name in set_folder_count_dict.keys():
        per_set_check_result_dict[per_set_name] = get_empty_check_result_dict()

    with Pool(processes=min(PROCESS_NUM, max(len(folder_check_pair_list),
                                             1))) as pool:
        for per_folder_check_result in tqdm(pool.imap_unordered(
                check_single_folder, folder_check_pair_list),
                                            total=len(folder_check_pair_list)):
            per_set_name, per_folder_name, per_folder_check_result_dict, per_folder_check_warning_message_list = per_folder_check_result

            merge_check_result_dict(check_result_dict,
                                    per_folder_check_result_dict)
            merge_check_result_dict(per_set_check_result_dict[per_set_name],
                                    per_folder_check_result_dict)

            for per_check_warning_message in per_folder_check_warning_message_list:
                check_warning_message_list.append(per_check_warning_message)
                print('2222', per_check_warning_message)

            per_folder_check_error_count = get_check_error_count(
                per_folder_check_result_dict)
            if per_folder_check_error_count > 0:
                print('3333', f'{per_set_name}/{per_folder_name}',
                      'check error:', per_folder_check_error_count)

    check_error_count = get_check_error_count(check_result_dict)

    print('4444', dataset_dir_name, 'total annotation:',
          check_result_dict['total_annotation_count'], 'total disk image:',
          check_result_dict['total_disk_image_count'], 'checked image:',
          check_result_dict['checked_image_count'],
          'valid image annotation pair:',
          check_result_dict['valid_image_annotation_pair_count'],
          'check error:', check_error_count, 'check warning:',
          len(check_warning_message_list))

    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        print('5555', per_error_type_name,
              check_result_dict[f'{per_error_type_name}_count'])
        for per_error_message in check_result_dict[
                f'{per_error_type_name}_message_list'][:10]:
            print('6666', per_error_message)

    # 复核报告一律写到数据集目录之外,保证本脚本不对数据集做任何修改
    os.makedirs(save_check_result_path, exist_ok=True)
    save_check_result_json_path = os.path.join(
        save_check_result_path,
        f'{dataset_dir_name}_integrity_check_result.json')
    save_check_result_dict = {
        'dataset_dir_name':
        dataset_dir_name,
        'root_dataset_path':
        root_dataset_path,
        'check_image_level':
        CHECK_IMAGE_LEVEL,
        'total_set_count':
        len(set_folder_count_dict),
        'total_folder_count':
        len(folder_check_pair_list),
        'check_error_count':
        check_error_count,
        'check_warning_count':
        len(check_warning_message_list),
        'check_warning_message_list':
        check_warning_message_list[:MAX_ERROR_MESSAGE_NUM],
        'set_folder_count_dict':
        set_folder_count_dict,
        'check_result_dict':
        check_result_dict,
        'per_set_check_result_dict':
        per_set_check_result_dict,
    }
    with open(save_check_result_json_path, 'w',
              encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    print('7777', save_check_result_json_path)

    if check_error_count > 0:
        # 复核不通过必须让上层感知,不能静默留下坏样本对或孤立文件
        raise Exception(
            f'check dataset {dataset_dir_name} error num {check_error_count}')

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/BM-6M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/FaceID-6M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/fine-t2i'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/FLUX-Reason-6M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/GPIC'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/MegaStyle-8M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/SACap-1M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/t2i_datasets/UNO-1M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result1'
    check_dataset(root_dataset_path, save_check_result_path)
