import os
import re
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'sacap_1m'

SAVE_DATASET_DIR_NAME = 'SACap-1M'

# ==============================================================================
# 【数据集类型判定】SACap-1M 是纯文生图(text-to-image)数据集,
# 不能处理成图像编辑数据集。
# 上游011解压脚本把parquet的每一行解成"一张SA-1B原图 + 一条全图caption",
# 全程没有任何参考图/输入图/编辑指令/前后图对,标注里的dataset_task_type
# 全量1029250行恒为text_to_image(实测Counter只有这一个取值),
# 所以本目录下只出这一个t2i脚本,不出ti2i(图像编辑)脚本。
# 原始数据集本来是给"分割mask到图像生成"用的,唯一的额外结构信息是
# segments_info(实例mask的anno_id + 区域caption),但上游011按纯文生图口径
# 已经整块丢弃、没有写进jsonl,而且mask本身也构不成"编辑前图/编辑后图"样本对。
#
# 【上游011解出的目录规格(实测全量扫完1000个jsonl,不是抽样推测)】
# SACap-1M/
# ├── unzip_annotations/sa_000000.jsonl .. sa_000999.jsonl  1000个,合计1029250行
# ├── unzip_images/sa_000000 .. sa_000999/<sample_key>.jpg  1000个目录,1029250张(970G)
# ├── unzip_source_annotations/anno_train.parquet           414MB原始标注(含segments_info)
# └── unzip_check_missing_images.json                       上游对账报告
#                                                           (0缺图/0空caption/0重名/0非法)
#
# 本脚本只读这1000个jsonl定位样本对,绝不os.walk图像目录:
# 上游一共解出103万个图像文件,扫一遍目录树在NAS上不可接受。
#
# 【jsonl每行固定9个字段(实测1029250行齐备、无缺字段)】
#   image_path / image_path_root / sample_key / image_id / image_group /
#   caption / img_width / img_height / dataset_task_type
#
# 【全量实测结论】
# - 1029250行,sample_key(形如sa_9973)100%唯一、0重复、100%匹配^sa_\d+$;
# - image_path与unzip_images/<image_group>/<sample_key>.jpg 100%一致,
#   image_group与jsonl文件名100%一致,image_path_root恒为save_dataset_path;
# - caption长度(strip后) min 100/p1 231/p50 350/p90 416/p99 474/p999 517/max 654,
#   >512只有1339条、>700为0条,没有一条小于100;
# - 标注宽高: 短边最小1080、最大宽高比1.8,即"短边<64"和"宽高比>8"这两条过滤
#   全量一条都不会命中(仍照写兜底);
# - 图像抽样300张: 全部是3通道baseline JPEG(非渐进、无灰度/RGBA/CMYK),
#   平均1.01MB,解出的宽高与标注0例不符;
# - 每个image_group实测932~1135条(上游注释里写的1005~1079与实测不符,
#   所以本脚本不复用那个区间,只用总行数1029250硬对账)。
# ==============================================================================

# 上游011解压脚本按image_group另存的汇总标注目录
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_SUFFIX = '.jsonl'

# 上游把图像按image_group分目录落在这里,jsonl里的image_path就是相对数据集根目录的路径
LOAD_IMAGE_DIR_NAME = 'unzip_images'

# 本数据集没有官方的train/val/test划分(原始标注只有anno_train.parquet一个文件),
# 所以整个数据集只有一个系列train,1029250条过滤后按sample_key全局排序统一重切
LOAD_SERIES_NAME = 'train'

# 实测上游jsonl分片数(1000个image_group各一个),数量不对说明上游没跑完
EXPECTED_ANNOTATION_FILE_NUM = 1000

# 实测上游有效样本对数,与上游对账报告里的total_valid_sample_pair_count完全一致,
# 直接当完整性ground truth: 缺分片/少行都能拦住
EXPECTED_TOTAL_ROW_COUNT = 1029250

ANNOTATION_IMAGE_KEY_NAME = 'image_path'

# 上游用它标记image_path是相对数据集根目录(save_dataset_path)还是相对SA-1B
# (depend_dataset_path)。实测1029250行恒为save_dataset_path(上游
# SAVE_IMAGE_FILE_FLAG=True,图像已经拷进unzip_images),
# 一旦出现depend_dataset_path说明上游是按"只建索引"模式跑的,
# 那时图像根本不在本数据集目录下,本脚本的相对路径拼接会全部落空,必须显式拦住
ANNOTATION_IMAGE_PATH_ROOT_KEY_NAME = 'image_path_root'

EXPECTED_ANNOTATION_IMAGE_PATH_ROOT = 'save_dataset_path'

# sample_key形如sa_9973,等于落盘图像的文件名前缀,实测1029250条100%唯一
# (上游报告duplicate_sample_key_count也是0),所以它可以直接当去重键,
# 也直接当保存图像名的后半段
ANNOTATION_IMAGE_NAME_KEY_NAME = 'sample_key'

# image_group形如sa_000345,就是SA-1B的分片目录名,也是jsonl文件名前缀。
# 只用来校验image_path是否落在本jsonl对应的目录下,不进新标注
ANNOTATION_IMAGE_GROUP_KEY_NAME = 'image_group'

# 全图caption,是本数据集唯一的训练文本,实测0条为空
ANNOTATION_CAPTION_KEY_NAME = 'caption'

# 上游从SA-1B的json头部取到的原图宽高。本脚本不拿它做分辨率过滤(万一它和实际图像
# 不一致就会出偏差),只在解码后顺手比对一次,不一致的条数只上报不丢样本
# (实测抽样300张0例不符)
ANNOTATION_IMAGE_WIDTH_KEY_NAME = 'img_width'

ANNOTATION_IMAGE_HEIGHT_KEY_NAME = 'img_height'

# 上游给每行都标了dataset_task_type,实测恒为text_to_image。
# 一旦出现别的取值说明上游规格变了(比如混进了编辑样本),这种样本不能进t2i数据集,
# 直接丢弃并计数上报
ANNOTATION_TASK_TYPE_KEY_NAME = 'dataset_task_type'

EXPECTED_ANNOTATION_TASK_TYPE = 'text_to_image'

# 上游标注里另外这些属性,按方案确认全部丢弃,新标注只留
# width/height/t2i_caption/t2i_caption_length:
# image_group  : sa_000000~sa_000999,SA-1B的分片号。过滤后按sample_key全局重切成
#                每10000张一个文件夹,分片归属信息落盘后彻底丢失,
#                下游不能再"只训某几个SA-1B分片"
# image_id     : SA-1B的整数图像id(如9973),与sample_key的数字部分相同,
#                落盘后只剩图像名里的sa_9973
# sample_key   : 以保存图像名后半段的形式保留,不再单独存字段
# image_path / image_path_root : 上游定位用
# img_width / img_height       : 由真实解码后的shape得到的width/height承载
# dataset_task_type            : 恒为text_to_image,只用于校验,不落盘
#
# 【上游011就已经丢弃、jsonl里根本没有的信息(只在unzip_source_annotations的
#   414MB parquet和SA-1B原始json里还留着)】
# segments_info : parquet第4列,list<struct<anno_id,caption>>,合计5882281个实例
#                 mask的id + 区域caption(每图1~20个,0条为空)。这是本数据集相对
#                 普通caption数据集最有价值的独有标注,按方案确认整块丢弃,
#                 落盘后无法从新数据集恢复(只能回头再读那份parquet)
# SA-1B json的annotations[] : RLE segmentation/bbox/area/predicted_iou/
#                 point_coords/crop_box/stability_score,上游只读了前512字节的
#                 image字段,这些从未被解析
SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 原始图像名前缀必须是sample_key那种sa_<数字>的组合(形如sa_9973),
# 不符的样本没法保证保存图像名全局唯一(可能去覆盖别的样本),直接丢弃并上报
VALID_IMAGE_NAME_PREFIX_PATTERN = re.compile(r'^sa_\d+$')

# jsonl文件名(去掉后缀)就是image_group,形如sa_000345
VALID_IMAGE_GROUP_PATTERN = re.compile(r'^sa_\d{6}$')

# 只保留RGB三通道图,灰度图/P图/RGBA图/CMYK图等一律过滤掉。
# 实测抽样300张全部是3通道baseline JPEG
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 本机128核,这里和001/003/009保持一致取32。本数据集要对103万张图各解码两遍
# (jsonl扫描阶段校验一次、写盘阶段重编码一次),想跑快可以直接调大这个常量
PROCESS_NUM = 32

PER_FOLDER_IMAGE_NUM = 10000

# 每个子集目录最多放多少个文件夹。1029250张切出103个文件夹(前102个满10000、
# 最后一个9250张),按方案确认只留一个子集目录train_000,所以这里取1000
# (远大于103),103个文件夹会全部归进train_000
PER_SET_FOLDER_NUM = 1000

# train系列按PER_SET_FOLDER_NUM切成带编号的子集目录。实测只会切出train_000一个,
# 但仍按编号命名,保证和003/009的子集目录命名规格一致
SPLIT_SET_DIR_SERIES_NAME_LIST = [
    'train',
]

# 实测上游标注里103万张图短边最小1080、最大宽高比1.8,所以这两条过滤实际一条都不会
# 命中,但仍然照写: 只要有一张图不符合就必须被判掉,不能靠"实测都合格"这个假设去
# 省掉校验(而且过滤一律以真实解码出来的shape为准,不信标注里的宽高)
MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 实测1029250条caption没有一条strip后长度小于100,所以这条实际也不会命中,同样照写兜底
MIN_CAPTION_LENGTH = 10

# 实测1029250条caption(strip后)长度 min 100/p50 350/p90 416/p99 474/p999 517/
# max 654,超过512的只有1339条(0.13%)、超过700的0条。
# 所以阈值取1024时一条都不会丢,既和003保持同一个量级口径,又不会误砍正常样本
MAX_CAPTION_LENGTH = 1024


def get_set_name(per_series_name, per_set_index):
    """子集目录名: train按每PER_SET_FOLDER_NUM个文件夹切成train_000/train_001/...

    子集目录名同时是保存图像名的中间段,所以它必须在切分完成后才能确定,
    这也是后面排序键只能用sample_key而不能用保存图像名的原因。
    本数据集103个文件夹全部落在train_000里,实际只会出这一个子集目录。
    """
    if per_series_name not in SPLIT_SET_DIR_SERIES_NAME_LIST:
        return per_series_name

    return f'{per_series_name}_{per_set_index:03d}'


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上,逐样本打一次os.path.exists就是一次网络往返,
    103万条标注就要打103万次,这一步本身就能占掉整个扫描阶段的大头。
    实测同一个jsonl里的图像全部落在同一个image_group目录下
    (sa_000000.jsonl的image_path都是unzip_images/sa_000000/xxx.jpg),
    所以这里按目录缓存一次os.listdir的结果,之后只做集合查表,
    网络往返次数从"标注条数"降到"image_group目录数"(1000次)。
    listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判,
    保证判定结果和改造前完全一致。
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
    """校验单张图像能否正常解码,并过滤非RGB图和极端分辨率图

    返回的宽高只用于和上游标注比对,最终写进json的宽高一定取自实际写盘图像的shape。
    """
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把RGBA图静默丢掉alpha通道、
    # 把CMYK图静默转成3通道,所以必须先用PIL读原始mode才能把这些图判出来
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

    # 检查图像短边
    if min(per_image_h, per_image_w) < MIN_IMAGE_SHORT_SIDE:
        print('6666', per_image_path, per_image_w, per_image_h)
        return None

    # 检查图像宽高比,取长短边之比,宽高比大于8和小于1/8这两种极端样本一起判掉
    per_image_aspect_ratio = max(per_image_w / per_image_h,
                                 per_image_h / per_image_w)
    if per_image_aspect_ratio > MAX_IMAGE_ASPECT_RATIO:
        print('7777', per_image_path, per_image_w, per_image_h)
        return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_annotation_file(annotation_file_pair):
    """解析单个上游jsonl标注,组装图像路径和t2i描述的样本对列表

    这个worker把文本层过滤(image_path_root不是save_dataset_path、缺图、
    image_group与jsonl文件名不一致、图像名与sample_key不一致、图像名非法、
    dataset_task_type不是text_to_image、描述为空或过短、描述过长)和图像层过滤
    (能否解码、是否RGB、短边、宽高比)一次做完。
    图像校验没有像001那样单独再开一个Pool,是因为本数据集有103万条标注:
    分两个Pool的话主进程要先攒103万条记录、再逐条发给check worker、再收回103万条,
    光进程间序列化就要来回搬十几GB,而合并进来之后IPC只传存活样本。
    判定逻辑、过滤口径、日志编号和001/003/009完全一致,图像也一样是解码两遍
    (这里校验一遍、写盘时重编码再解一遍),没有为了省时间跳过任何一道校验。
    """
    per_jsonl_path, root_dataset_path, per_series_name, per_image_group = annotation_file_pair

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    image_annotation_pair_list = []

    # 每个worker只处理一个jsonl,缓存里通常只有一个image_group目录,内存开销可忽略
    dir_file_name_cache_dict = {}

    try:
        load_jsonl_file = open(per_jsonl_path, 'r', encoding='UTF-8')
    except Exception as e:
        print('2222', per_jsonl_path, e)

        return [
            image_annotation_pair_list,
            per_series_name,
            total_annotation_count,
            1,
            missing_image_count,
            invalid_image_name_count,
            invalid_task_type_count,
            invalid_caption_count,
            too_long_caption_count,
            invalid_image_count,
            annotation_image_size_not_match_count,
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
                print('2222', per_jsonl_path, e)
                continue

            if not isinstance(per_annotation, dict):
                load_annotation_failed_count += 1
                print('2222', per_jsonl_path, 'annotation not a dict')
                continue

            # 上游是按"只建索引"模式跑的话图像根本不在本数据集目录下,
            # 相对路径拼接会全部落空,这种样本不能进t2i数据集
            per_image_path_root = per_annotation.get(
                ANNOTATION_IMAGE_PATH_ROOT_KEY_NAME, '')
            if not isinstance(per_image_path_root, str):
                per_image_path_root = ''
            if per_image_path_root.strip(
            ) != EXPECTED_ANNOTATION_IMAGE_PATH_ROOT:
                missing_image_count += 1
                print('2222', per_jsonl_path, per_image_path_root)
                continue

            per_image_relative_path = per_annotation.get(
                ANNOTATION_IMAGE_KEY_NAME, '')
            if not isinstance(per_image_relative_path, str):
                per_image_relative_path = ''
            if not per_image_relative_path:
                missing_image_count += 1
                continue

            # 上游image_path形如unzip_images/sa_000000/sa_9973.jpg,
            # 前两段固定是unzip_images和本jsonl对应的image_group,
            # 不一致说明上游成员错位,这种样本定位到的图像不属于本组,直接丢弃
            per_image_relative_path = per_image_relative_path.replace(
                '\\', '/').lstrip('/')
            per_image_relative_path_name_list = per_image_relative_path.split(
                '/')
            if len(
                    per_image_relative_path_name_list
            ) != 3 or per_image_relative_path_name_list[
                    0] != LOAD_IMAGE_DIR_NAME or per_image_relative_path_name_list[
                        1] != per_image_group:
                invalid_image_name_count += 1
                print('2222', per_jsonl_path, per_image_relative_path)
                continue

            # jsonl里另存的image_group也必须和jsonl文件名一致,兜一道底
            per_annotation_image_group = per_annotation.get(
                ANNOTATION_IMAGE_GROUP_KEY_NAME, '')
            if not isinstance(per_annotation_image_group, str):
                per_annotation_image_group = ''
            if per_annotation_image_group.strip() != per_image_group:
                invalid_image_name_count += 1
                print('2222', per_jsonl_path, per_annotation_image_group)
                continue

            per_image_path = os.path.join(root_dataset_path,
                                          per_image_relative_path)
            if not check_image_file_exists(per_image_path,
                                           dir_file_name_cache_dict):
                missing_image_count += 1
                continue

            # 落盘图像的文件名前缀就是sample_key,和标注里的sample_key必须完全一致,
            # 不一致说明上游成员错位,这种样本没法保证保存图像名唯一,直接丢弃
            per_image_name_prefix = os.path.splitext(
                os.path.basename(per_image_relative_path))[0].lower()
            per_sample_key = per_annotation.get(ANNOTATION_IMAGE_NAME_KEY_NAME,
                                                '')
            if not isinstance(per_sample_key, str):
                per_sample_key = ''
            per_sample_key = per_sample_key.strip().lower()

            if not per_sample_key or per_sample_key != per_image_name_prefix:
                invalid_image_name_count += 1
                print('2222', per_image_path, per_sample_key)
                continue

            if not VALID_IMAGE_NAME_PREFIX_PATTERN.match(
                    per_image_name_prefix):
                invalid_image_name_count += 1
                print('2222', per_image_path, per_image_name_prefix)
                continue

            per_task_type = per_annotation.get(ANNOTATION_TASK_TYPE_KEY_NAME,
                                               '')
            if not isinstance(per_task_type, str):
                per_task_type = ''
            per_task_type = per_task_type.strip().lower()

            # 实测上游dataset_task_type恒为text_to_image,一旦出现别的取值说明上游
            # 规格变了,这种样本不能进t2i数据集
            if per_task_type != EXPECTED_ANNOTATION_TASK_TYPE:
                invalid_task_type_count += 1
                print('2222', per_image_path, per_task_type)
                continue

            per_t2i_caption = per_annotation.get(ANNOTATION_CAPTION_KEY_NAME,
                                                 '')
            # 上游描述固定是str,这里兼容list和str两种形式
            if isinstance(per_t2i_caption, (list, tuple)):
                per_t2i_caption = per_t2i_caption[0] if len(
                    per_t2i_caption) > 0 else ''
            if not isinstance(per_t2i_caption, str):
                per_t2i_caption = ''
            per_t2i_caption = per_t2i_caption.strip()

            # 空描述、全空格描述、过短描述都视为不合格样本对
            # (上游已校验caption全部非空,实测103万条最短100)
            if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                invalid_caption_count += 1
                print('3333', per_image_path, len(per_t2i_caption))
                continue

            # 过长描述同样视为不合格样本对(实测最长654,这条一条都不会命中)
            if len(per_t2i_caption) > MAX_CAPTION_LENGTH:
                too_long_caption_count += 1
                print('3333', per_image_path, len(per_t2i_caption))
                continue

            per_check_result = process_single_image_check(per_image_path)
            if per_check_result is None:
                invalid_image_count += 1
                continue

            per_image_w, per_image_h = per_check_result

            # 分辨率过滤一律用上面真实解码出来的shape,这里只是顺手核对一遍上游标注里
            # 记录的宽高,不一致只计数上报、不丢样本(实测抽样300张0例不符)
            if per_annotation.get(
                    ANNOTATION_IMAGE_WIDTH_KEY_NAME,
                    per_image_w) != per_image_w or per_annotation.get(
                        ANNOTATION_IMAGE_HEIGHT_KEY_NAME,
                        per_image_h) != per_image_h:
                annotation_image_size_not_match_count += 1
                print(
                    '2222', per_image_path, per_image_w, per_image_h,
                    per_annotation.get(ANNOTATION_IMAGE_WIDTH_KEY_NAME, None),
                    per_annotation.get(ANNOTATION_IMAGE_HEIGHT_KEY_NAME, None))

            # 保存图像名要等切完子集目录才能拼出来,这里只带上sample_key前缀
            image_annotation_pair_list.append([
                per_series_name,
                per_image_path,
                per_image_name_prefix,
                per_t2i_caption,
            ])

    return [
        image_annotation_pair_list,
        per_series_name,
        total_annotation_count,
        load_annotation_failed_count,
        missing_image_count,
        invalid_image_name_count,
        invalid_task_type_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_image_count,
        annotation_image_size_not_match_count,
    ]


def get_all_image_annotation_pair(root_dataset_path):
    """扫描上游解压好的jsonl标注,多进程组装图像路径和t2i描述的样本对列表

    这里只listdir unzip_annotations一层拿到1000个jsonl路径,
    绝不去os.walk图像目录: 上游解出103万个图像文件,扫目录树在NAS上不可接受。
    每个jsonl里的图像全在同一个image_group目录下,worker只需要对那个目录listdir一次。
    最后按[系列名, sample_key]统一排序,保证输出顺序与串行版本完全一致。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    invalid_annotation_file_name_list = []
    if not os.path.isdir(root_annotation_path):
        print('2222', root_annotation_path)
    else:
        for per_jsonl_name in sorted(os.listdir(root_annotation_path)):
            if not per_jsonl_name.endswith(LOAD_ANNOTATION_FILE_SUFFIX):
                continue

            # jsonl文件名去掉后缀就是image_group,由主进程算好后带给worker,
            # worker只认标注文件、数据集根目录、系列名、组名这四个入参
            per_image_group = per_jsonl_name[:-len(LOAD_ANNOTATION_FILE_SUFFIX
                                                   )]
            if not VALID_IMAGE_GROUP_PATTERN.match(per_image_group):
                # 组名规格变了说明上游产物不是本脚本认识的那一份,必须显式感知
                invalid_annotation_file_name_list.append(per_jsonl_name)
                print('2222', root_annotation_path, per_jsonl_name)
                continue

            annotation_file_pair_list.append([
                os.path.join(root_annotation_path, per_jsonl_name),
                root_dataset_path,
                LOAD_SERIES_NAME,
                per_image_group,
            ])

    annotation_file_count = len(annotation_file_pair_list)

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    image_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            image_annotation_pair_list.extend(per_load_result[0])
            total_annotation_count += per_load_result[2]
            load_annotation_failed_count += per_load_result[3]
            missing_image_count += per_load_result[4]
            invalid_image_name_count += per_load_result[5]
            invalid_task_type_count += per_load_result[6]
            invalid_caption_count += per_load_result[7]
            too_long_caption_count += per_load_result[8]
            invalid_image_count += per_load_result[9]
            annotation_image_size_not_match_count += per_load_result[10]

    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[0], x[2]])

    return [
        image_annotation_pair_list,
        annotation_file_count,
        invalid_annotation_file_name_list,
        total_annotation_count,
        load_annotation_failed_count,
        missing_image_count,
        invalid_image_name_count,
        invalid_task_type_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_image_count,
        annotation_image_size_not_match_count,
    ]


def get_deduplicated_image_annotation_pair(image_annotation_pair_list):
    """按sample_key全局去重,每个sample_key只保留排序后的第一条

    sample_key就是SA-1B的图像名前缀(sa_9973),实测1029250条100%唯一、
    上游的对账报告也是0重复,所以这里正常应该一条都不丢,只是兜一道底:
    撞名会让后写的图像覆盖先写的、静默丢样本。
    排序键取[sample_key, 系列名],保证同一个sample_key留下的永远是同一条。
    """
    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[2], x[0]])

    duplicate_key_dict = {}
    duplicate_image_name_prefix_list = []
    deduplicated_image_annotation_pair_list = []
    for per_image_annotation_pair in image_annotation_pair_list:
        per_series_name, per_image_path, per_image_name_prefix, per_t2i_caption = per_image_annotation_pair
        if per_image_name_prefix in duplicate_key_dict:
            duplicate_image_name_prefix_list.append(per_image_name_prefix)
            print('2222', per_image_path, per_image_name_prefix)
            continue

        duplicate_key_dict[per_image_name_prefix] = 1
        deduplicated_image_annotation_pair_list.append(
            per_image_annotation_pair)

    deduplicated_image_annotation_pair_list = sorted(
        deduplicated_image_annotation_pair_list, key=lambda x: [x[0], x[2]])

    return deduplicated_image_annotation_pair_list, duplicate_image_name_prefix_list


def get_all_image_save_folder_pair(image_annotation_pair_list,
                                   save_dataset_path):
    """把过滤后的合格样本按系列分组,排序后每10000张切成一个文件夹、每PER_SET_FOLDER_NUM个文件夹归一个子集目录

    切分必须在过滤全部完成之后做,且切分前先按sample_key排序,这样才能保证每个文件夹都是
    满10000张(只有每个系列全局最后一个文件夹允许不满)。
    排序键用sample_key而不是保存图像名: 保存图像名里含子集目录名,而子集目录名恰恰由排序
    后的位置决定,存在循环依赖;同一个子集目录内所有图像名前缀完全相同,
    所以按sample_key排序与按保存图像名排序结果完全等价。
    上游的1000个image_group只是SA-1B的分片目录、不是语义子集,每组只有约1000条,
    按组切永远凑不满10000张,所以这里把它们并成同一个train系列统一重切。
    """
    per_series_image_annotation_pair_dict = {}
    for per_image_annotation_pair in image_annotation_pair_list:
        per_series_name = per_image_annotation_pair[0]
        if per_series_name not in per_series_image_annotation_pair_dict:
            per_series_image_annotation_pair_dict[per_series_name] = []
        per_series_image_annotation_pair_dict[per_series_name].append(
            per_image_annotation_pair)

    image_save_folder_pair_list = []
    set_folder_count_dict, series_set_name_list_dict = {}, {}
    for per_series_name in sorted(
            per_series_image_annotation_pair_dict.keys()):
        per_series_image_annotation_pair_list = sorted(
            per_series_image_annotation_pair_dict[per_series_name],
            key=lambda x: x[2])

        per_series_folder_count = 0
        per_series_set_name_list = []
        for per_folder_start_index in range(
                0, len(per_series_image_annotation_pair_list),
                PER_FOLDER_IMAGE_NUM):
            per_folder_image_annotation_pair_list = per_series_image_annotation_pair_list[
                per_folder_start_index:per_folder_start_index +
                PER_FOLDER_IMAGE_NUM]

            # 全局文件夹序号先定子集目录,再定子集目录内的文件夹序号
            per_set_index = per_series_folder_count // PER_SET_FOLDER_NUM
            per_set_folder_index = per_series_folder_count % PER_SET_FOLDER_NUM

            per_set_name = get_set_name(per_series_name, per_set_index)
            per_folder_name = f'{per_set_name}_{per_set_folder_index:05d}'
            per_folder_image_path = os.path.join(save_dataset_path,
                                                 per_set_name, per_folder_name)
            os.makedirs(per_folder_image_path, exist_ok=True)

            per_folder_save_pair_list = []
            for per_image_annotation_pair in per_folder_image_annotation_pair_list:
                _, per_image_path, per_image_name_prefix, per_t2i_caption = per_image_annotation_pair
                # 保存图像名统一全小写,形如sacap_1m_train_000_sa_9973.jpg
                per_save_image_name = f'{DATASET_NAME}_{per_set_name}_{per_image_name_prefix}{SAVE_IMAGE_NAME_SUFFIX}'
                per_folder_save_pair_list.append([
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

            # 一个文件夹就是一个写盘任务,worker写完这10000张后直接写出该文件夹的json,
            # 主进程只收计数,不用把103万条记录再攒一遍
            image_save_folder_pair_list.append([
                per_set_name,
                per_folder_name,
                per_folder_save_pair_list,
            ])

            if per_set_name not in set_folder_count_dict:
                set_folder_count_dict[per_set_name] = 0
                per_series_set_name_list.append(per_set_name)
            set_folder_count_dict[per_set_name] += 1

            per_series_folder_count += 1

        series_set_name_list_dict[per_series_name] = per_series_set_name_list

    return image_save_folder_pair_list, set_folder_count_dict, series_set_name_list_dict


def process_single_image_folder(image_save_folder_pair, save_dataset_path):
    """重新编码保存一个文件夹的图像,并写出与文件夹同名的json标注

    图像原分辨率多少保存时还是多少,不做任何缩放。
    以文件夹为任务粒度而不是以单张图为粒度: 本数据集约103万张图,逐图收结果的话
    主进程要再攒一份103万条的列表,而且中途挂了只能从头再来;按文件夹收之后
    主进程内存只和文件夹数(103)相关,且json已经写全的文件夹可以直接跳过、支持断点续跑。
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

        # json里的宽高直接取自这个即将被编码写盘的数组的shape,
        # 中间不做resize,jpg编解码也不改变像素尺寸,所以宽高一定和保存图像一致
        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

        save_image_path = os.path.join(save_folder_path, per_save_image_name)

        if not os.path.exists(save_image_path):
            try:
                cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
            except Exception as e:
                save_image_failed_count += 1
                print('8888', save_image_path, e)
                continue

        # t2i_caption_length直接取即将写进json的这个字符串的长度,
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

    每个系列只有全局最后一个文件夹允许不满10000张,其余都必须是满10000张。
    """
    per_set_name, per_folder_name, per_is_series_last_folder = folder_check_pair

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

    # 除每个系列全局最后一个文件夹外都必须是满10000张
    if not per_is_series_last_folder and len(
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


def check_save_dataset(save_dataset_path, set_folder_count_dict,
                       series_set_name_list_dict):
    """全部落盘后的收尾自校验: 子集目录容量、文件夹容量、json与磁盘一一对应、图像名和描述合规

    和001的差别只在"满10000张"的口径: 001里子集本身就是切分单位,所以每个子集的最后一个
    文件夹都允许不满;这里子集目录只是PER_SET_FOLDER_NUM个文件夹的容器,
    本数据集103个文件夹全部归进train_000,所以只有整个train系列全局最后一个文件夹
    允许不满,其余每个文件夹都必须是满10000张。
    同理每个系列只有最后一个子集目录允许不满PER_SET_FOLDER_NUM个文件夹。
    103个文件夹每个都要listdir一万个文件再load一份json,串行跑在NAS上太久,
    所以这一步也按文件夹粒度开多进程。
    """
    check_error_message_list = []

    folder_check_pair_list = []
    for per_series_name in sorted(series_set_name_list_dict.keys()):
        per_series_set_name_list = series_set_name_list_dict[per_series_name]
        for per_set_index, per_set_name in enumerate(per_series_set_name_list):
            per_set_folder_count = set_folder_count_dict[per_set_name]

            # 每个系列只有最后一个子集目录允许不满PER_SET_FOLDER_NUM个文件夹
            if per_set_index < len(
                    per_series_set_name_list
            ) - 1 and per_set_folder_count != PER_SET_FOLDER_NUM:
                check_error_message_list.append(
                    f'{per_set_name} folder num not match {per_set_folder_count} != {PER_SET_FOLDER_NUM}'
                )

            for per_folder_index in range(per_set_folder_count):
                per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
                per_is_series_last_folder = (
                    per_set_index == len(per_series_set_name_list) - 1
                    and per_folder_index == per_set_folder_count - 1)
                folder_check_pair_list.append([
                    per_set_name,
                    per_folder_name,
                    per_is_series_last_folder,
                ])

    total_image_count = 0
    check_func = partial(check_single_save_folder,
                         save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
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

    image_annotation_pair_list, annotation_file_count, invalid_annotation_file_name_list, total_annotation_count, load_annotation_failed_count, missing_image_count, invalid_image_name_count, invalid_task_type_count, invalid_caption_count, too_long_caption_count, invalid_image_count, annotation_image_size_not_match_count = get_all_image_annotation_pair(
        root_dataset_path)

    print('1111', annotation_file_count,
          len(invalid_annotation_file_name_list), total_annotation_count,
          load_annotation_failed_count, missing_image_count,
          invalid_image_name_count, invalid_task_type_count,
          invalid_caption_count, too_long_caption_count,
          invalid_image_count, annotation_image_size_not_match_count,
          len(image_annotation_pair_list))

    if len(image_annotation_pair_list) > 0:
        print('1111', image_annotation_pair_list[0])

    # 上游jsonl分片数量或标注条数不对说明上游没跑完,继续跑只会静默少样本对
    annotation_file_error_message_list = []
    if annotation_file_count != EXPECTED_ANNOTATION_FILE_NUM:
        annotation_file_error_message_list.append(
            f'annotation file num not match {annotation_file_count} != {EXPECTED_ANNOTATION_FILE_NUM}'
        )

    if len(invalid_annotation_file_name_list) > 0:
        annotation_file_error_message_list.append(
            f'invalid annotation file num {len(invalid_annotation_file_name_list)} {invalid_annotation_file_name_list[:10]}'
        )

    if total_annotation_count != EXPECTED_TOTAL_ROW_COUNT:
        annotation_file_error_message_list.append(
            f'total annotation count not match {total_annotation_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )

    if len(annotation_file_error_message_list) > 0:
        raise Exception(
            f'check annotation file failed {annotation_file_error_message_list}'
        )

    if load_annotation_failed_count > 0:
        raise Exception(
            f'load annotation failed count {load_annotation_failed_count}')

    image_annotation_pair_list, duplicate_image_name_prefix_list = get_deduplicated_image_annotation_pair(
        image_annotation_pair_list)

    print('1111', len(image_annotation_pair_list),
          len(duplicate_image_name_prefix_list))
    if len(duplicate_image_name_prefix_list) > 0:
        print('1111', duplicate_image_name_prefix_list[:10])

    image_save_folder_pair_list, set_folder_count_dict, series_set_name_list_dict = get_all_image_save_folder_pair(
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

    # 切分之后保存图像名必须全局唯一,撞名会让后写的图像覆盖先写的、静默丢样本,
    # 所以这里再兜一道,撞上就直接中止
    save_image_name_dict = {}
    conflict_save_image_name_list = []
    for _, _, per_folder_save_pair_list in image_save_folder_pair_list:
        for _, per_save_image_name, _ in per_folder_save_pair_list:
            if per_save_image_name in save_image_name_dict:
                conflict_save_image_name_list.append(per_save_image_name)
                continue
            save_image_name_dict[per_save_image_name] = 1

    if len(conflict_save_image_name_list) > 0:
        raise Exception(
            f'conflict save image name num {len(conflict_save_image_name_list)} {conflict_save_image_name_list[:10]}'
        )

    save_image_name_dict = {}

    folder_image_count_dict = {}
    save_image_failed_count = 0
    process_func = partial(process_single_image_folder,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_save_result in tqdm(pool.imap_unordered(
                process_func, image_save_folder_pair_list),
                                    total=len(image_save_folder_pair_list)):
            per_set_name, per_folder_name, per_folder_image_count, per_save_image_failed_count = per_save_result
            folder_image_count_dict[per_folder_name] = per_folder_image_count
            save_image_failed_count += per_save_image_failed_count

            print('2222', per_folder_name, per_folder_image_count,
                  per_save_image_failed_count)

    total_save_image_count = sum(folder_image_count_dict.values())

    check_error_message_list, check_total_image_count = check_save_dataset(
        save_dataset_path, set_folder_count_dict, series_set_name_list_dict)

    print('3333', 'total annotation:', total_annotation_count,
          'missing image:', missing_image_count, 'invalid image name:',
          invalid_image_name_count, 'invalid task type:',
          invalid_task_type_count, 'invalid caption:', invalid_caption_count,
          'too long caption:', too_long_caption_count, 'invalid image:',
          invalid_image_count, 'annotation image size not match:',
          annotation_image_size_not_match_count, 'duplicate image name:',
          len(duplicate_image_name_prefix_list), 'save image failed:',
          save_image_failed_count, 'total save image:',
          total_save_image_count, 'total save folder:',
          len(folder_image_count_dict), 'total save set:',
          len(set_folder_count_dict),
          'check total image:', check_total_image_count, 'check error:',
          len(check_error_message_list))

    save_check_result_path = os.path.join(save_dataset_path,
                                          'resave_check_result.json')
    save_check_result_dict = {
        'annotation_file_count':
        annotation_file_count,
        'invalid_annotation_file_count':
        len(invalid_annotation_file_name_list),
        'total_annotation_count':
        total_annotation_count,
        'load_annotation_failed_count':
        load_annotation_failed_count,
        'missing_image_count':
        missing_image_count,
        'invalid_image_name_count':
        invalid_image_name_count,
        'invalid_task_type_count':
        invalid_task_type_count,
        'invalid_caption_count':
        invalid_caption_count,
        'too_long_caption_count':
        too_long_caption_count,
        'invalid_image_count':
        invalid_image_count,
        'annotation_image_size_not_match_count':
        annotation_image_size_not_match_count,
        'duplicate_image_name_count':
        len(duplicate_image_name_prefix_list),
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
        'series_set_name_list_dict':
        series_set_name_list_dict,
        'set_folder_count_dict':
        set_folder_count_dict,
        'folder_image_count_dict':
        folder_image_count_dict,
        'invalid_annotation_file_name_list':
        invalid_annotation_file_name_list[:10000],
        'duplicate_image_name_prefix_list':
        duplicate_image_name_prefix_list[:10000],
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
        # 收尾自校验不通过必须让上层感知,不能静默留下坏样本或不满的文件夹
        raise Exception(
            f'check save dataset error num {len(check_error_message_list)} {check_error_message_list[:10]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/SACap-1M'
    save_dataset_path = r'/root/autodl-tmp/t2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
