import os
import re
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'flux_reason_6m'

SAVE_DATASET_DIR_NAME = 'FLUX-Reason-6M'

# ==============================================================================
# 【数据集类型判定】FLUX-Reason-6M 是纯文生图(text-to-image)数据集,
# 不能处理成图像编辑数据集。
# 上游009解包脚本把每行parquet解成"一张生成后图 + 8种粒度的中英文caption",
# 全程没有任何参考图/输入图/编辑指令/mask,标注里的task_type恒为text_to_image,
# 官方README也写明是"6-million-scale text-to-image dataset",
# 所以本目录下只出这一个t2i脚本,不出ti2i(图像编辑)脚本。
#
# 【上游009解出的目录规格(实测)】
# FLUX-Reason-6M/
# ├── images/<子集>/<parquet分片名>/<sample_id>.jpg   4个子集、1180个分片目录、5890279张
# ├── unzip_annotations/<子集>/<parquet分片名>.jsonl  1180个jsonl,每行一个样本对
# └── unzip_check_missing_images.json                 上游对账报告
#                                                     (0 invalid/0 warning/0 duplicate)
#
# 本脚本只读这1180个jsonl定位样本对,绝不os.walk图像目录:
# 上游一共解出589万个小文件,扫一遍目录树在NAS上不可接受。
# ==============================================================================

# 上游009解压脚本按tar(parquet分片)另存的汇总标注目录
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_SUFFIX = '.jsonl'

# 上游009把图像落在 images/<子集>/<parquet分片名>/<sample_id>.jpg,
# 而jsonl里的image_path记的是**不带images这一层前缀**的相对路径
# (形如Text/fluxdb-text-00000-of-00143/text-00000000.jpg),
# 所以拼图像绝对路径时必须把这一层补回去,
# 否则每一条标注都会指向一个不存在的路径、589万条全部被判成缺图
LOAD_IMAGE_DIR_NAME = 'images'

# 上游解出的四个子集目录,unzip_annotations下也是这四个同名目录。
# 这四个目录既是上游的子集名,也是本脚本的系列名(切分后再变成带编号的子集目录名)
LOAD_SERIES_DIR_NAME_LIST = [
    'Aesthetics-Part01',
    'Aesthetics-Part02',
    'Imaginative',
    'Text',
]

# 实测上游jsonl分片数(415 + 454 + 168 + 143 = 1180),数量不对说明上游没跑完
EXPECTED_SERIES_ANNOTATION_FILE_NUM_DICT = {
    'Aesthetics-Part01': 415,
    'Aesthetics-Part02': 454,
    'Imaginative': 168,
    'Text': 143,
}

# 实测上游每子集的有效样本对数(合计5890279),与上游对账报告里的
# subset_sample_pair_count_dict完全一致,直接当完整性ground truth:
# 缺子集/缺分片/少行都能拦住
EXPECTED_SERIES_ROW_COUNT_DICT = {
    'Aesthetics-Part01': 2071075,
    'Aesthetics-Part02': 2269710,
    'Imaginative': 838157,
    'Text': 711337,
}

EXPECTED_TOTAL_ROW_COUNT = 5890279

# 上游jsonl每行固定44个字段(实测抽样16万行key组合100%一致、无缺字段)
ANNOTATION_IMAGE_KEY_NAME = 'image_path'

# sample_id形如aesthetics-part01-00985000,等于落盘图像的文件名前缀,
# 自带子集前缀所以全局唯一(实测抽样16万条100%唯一、上游报告0重复),
# 所以它可以直接当去重键,也直接当保存图像名的后半段
ANNOTATION_IMAGE_NAME_KEY_NAME = 'sample_id'

# t2i描述取caption_detail(即README里说的GCoT)这一列,原因:
# 1) 它是8种粒度里唯一100%非空的caption(其余7类非空率17%~87%),
#    换成别的粒度会大面积丢样本;
# 2) 实测抽样8万条caption_detail: 0条含<think>/<answer>之类思维链标签、
#    0条含换行、0条含"Step 1:/First,/Therefore/Let me"这类推理句式,
#    落盘文本就是一段纯稠密英文图像描述,虽然官方叫GCoT但不含思维链,
#    可以直接当t2i描述用;
# 3) 优先英文,所以取caption_detail而不是caption_detail_cn。
ANNOTATION_CAPTION_KEY_NAME = 'caption_detail'

# 上游标注里记录的生成图宽高。本脚本不拿它做分辨率过滤(万一它和实际图像不一致
# 就会出偏差),只在解码后顺手比对一次,不一致的条数只上报不丢样本
# (实测上游报告image_shape_count_dict全部是1024x1024)
ANNOTATION_IMAGE_WIDTH_KEY_NAME = 'image_width'

ANNOTATION_IMAGE_HEIGHT_KEY_NAME = 'image_height'

# 上游给每行都标了task_type,实测恒为text_to_image。
# 一旦出现别的取值说明上游规格变了(比如混进了编辑样本),这种样本不能进t2i数据集,
# 直接丢弃并计数上报
ANNOTATION_TASK_TYPE_KEY_NAME = 'task_type'

EXPECTED_ANNOTATION_TASK_TYPE = 'text_to_image'

# 上游标注里另外这些属性,按方案确认全部丢弃,新标注只留
# width/height/t2i_caption/t2i_caption_length:
# caption                : 实测与caption_detail 100%相同的冗余字段
# caption_detail_cn      : caption_detail的中文版(按方案优先英文,丢弃)
# caption_composition / caption_entity / caption_text / caption_imaginative /
# caption_style / caption_abstract / caption_original 及它们的_cn版本 :
#                          另外7种粒度的中英文caption,一张图只能对应一个
#                          t2i_caption,所以这些粒度落盘后永久丢失
# bool_caption_* ×8      : caption质量门控标志(上游已确认
#                          bool_caption_original是质量门控而非presence标志),
#                          按方案不做门控过滤,直接丢弃
# score_composition / score_entity / score_text / score_imaginative /
# score_style / score_abstract / score_original / score_image_clarity /
# score_image_structure  : 9个[0,10]打分,按方案不做打分过滤,直接丢弃
# caption_type_list      : 该行哪些粒度非空,落盘后无意义
# id                     : 与sample_id完全相同
# subset_name / parquet_name / row_index / image_path / task_type : 上游定位用
# image_width / image_height : 由真实解码后的shape得到的width/height承载
SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 原始图像名前缀必须是sample_id那种"小写字母/数字/中划线"的组合
# (形如aesthetics-part01-00985000),不符的样本没法保证保存图像名全局唯一
# (可能去覆盖别的样本),直接丢弃并上报
VALID_IMAGE_NAME_PREFIX_PATTERN = re.compile(r'^[a-z0-9\-]+$')

# 只保留RGB三通道图。实测抽样600张全部是3通道baseline JPEG、无灰度/RGBA/CMYK图
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 本机128核,这里和001/003保持一致取32。本数据集要对589万张图各解码两遍
# (jsonl扫描阶段校验一次、写盘阶段重编码一次),想跑快可以直接调大这个常量
PROCESS_NUM = 32

PER_FOLDER_IMAGE_NUM = 10000

# 每个子集目录放100个文件夹(即100万张图)。Aesthetics-Part02约227万张会切出
# 约227个文件夹,全塞进一个目录下NAS元数据压力太大,所以每100个文件夹再归入
# 一个子集目录,子集目录形如aesthetics-part02_000/aesthetics-part02_001/...
PER_SET_FOLDER_NUM = 100

# 四个系列都按PER_SET_FOLDER_NUM切成带编号的子集目录:
# Aesthetics-Part01约207万张(约3个子集目录)、Aesthetics-Part02约227万张
# (约3个)、Imaginative约84万张(1个)、Text约71万张(1个)。
# 即使只出一个子集目录也带编号,保证四个系列的子集目录命名规格一致
SPLIT_SET_DIR_SERIES_NAME_LIST = [
    'Aesthetics-Part01',
    'Aesthetics-Part02',
    'Imaginative',
    'Text',
]

# 实测上游报告里589万张图全部是1024x1024,所以短边和宽高比这两条过滤实际
# 一条都不会命中,但仍然照写: 只要有一张图不符合就必须被判掉,不能靠"实测都合格"
# 这个假设去省掉校验
MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 实测抽样8万条caption_detail没有一条strip后长度小于252,所以这条实际也不会命中,
# 同样照写兜底
MIN_CAPTION_LENGTH = 10

# 实测抽样8万条caption_detail(strip后)长度 min 252/p50 1073/p90 1259/
# p99 1440/max 6907,超过1024的占67.8%、超过1280的占8.5%、超过1536的占0.27%、
# 超过2048的占0.008%。
# 所以阈值不能沿用003的1024(会砍掉三分之二的正常样本),
# 按方案取1536,只丢约0.27%的长尾异常样本
MAX_CAPTION_LENGTH = 1536


def get_set_name(per_series_name, per_set_index):
    """子集目录名: 每个系列按每100个文件夹切成<系列名小写>_000/_001/...

    子集目录名同时是保存图像名的中间段,所以它必须在切分完成后才能确定,
    这也是后面排序键只能用sample_id而不能用保存图像名的原因。
    上游系列名带大写(Aesthetics-Part01),而保存图像名要求全小写,所以这里统一转小写。
    """
    per_set_name = per_series_name.lower()

    if per_series_name not in SPLIT_SET_DIR_SERIES_NAME_LIST:
        return per_set_name

    return f'{per_set_name}_{per_set_index:03d}'


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上,逐样本打一次os.path.exists就是一次网络往返,
    589万条标注就要打589万次,这一步本身就能占掉整个扫描阶段的大头。
    实测同一个jsonl里的图像全部落在同一个分片目录下
    (fluxdb-text-00000-of-00143.jsonl的image_path都是
    Text/fluxdb-text-00000-of-00143/xxx.jpg),所以这里按目录缓存一次
    os.listdir的结果,之后只做集合查表,网络往返次数从"标注条数"降到"分片目录数"
    (1180次)。listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判,
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

    这个worker把文本层过滤(缺图、图像名与sample_id不一致、图像名非法、
    task_type不是text_to_image、描述为空或过短、描述过长)和图像层过滤
    (能否解码、是否RGB、短边、宽高比)一次做完。
    图像校验没有像001那样单独再开一个Pool,是因为本数据集有589万条标注:
    分两个Pool的话主进程要先攒589万条记录、再逐条发给check worker、再收回589万条,
    光进程间序列化就要来回搬几十GB,而合并进来之后IPC只传存活样本。
    判定逻辑、过滤口径、日志编号和001/003完全一致,图像也一样是解码两遍
    (这里校验一遍、写盘时重编码再解一遍),没有为了省时间跳过任何一道校验。
    """
    per_jsonl_path, root_dataset_path, per_series_name = annotation_file_pair

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    image_annotation_pair_list = []

    # 每个worker只处理一个jsonl,缓存里通常只有一个分片目录,内存开销可忽略
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

            per_image_relative_path = per_annotation.get(
                ANNOTATION_IMAGE_KEY_NAME, '')
            if not isinstance(per_image_relative_path, str):
                per_image_relative_path = ''
            if not per_image_relative_path:
                missing_image_count += 1
                continue

            # 上游image_path形如Text/fluxdb-text-00000-of-00143/text-00000000.jpg,
            # 第一段就是子集名,必须和当前jsonl所在的系列目录一致,
            # 不一致说明上游成员错位,这种样本会被切进错误的子集目录,直接丢弃
            per_image_relative_path = per_image_relative_path.replace(
                '\\', '/').lstrip('/')
            if per_image_relative_path.split('/')[0] != per_series_name:
                invalid_image_name_count += 1
                print('2222', per_jsonl_path, per_image_relative_path)
                continue

            # image_path不带images这一层前缀,所以这里要补上LOAD_IMAGE_DIR_NAME,
            # 拼出来才是上游真正的落盘路径
            # <root>/images/<子集>/<parquet分片名>/<sample_id>.jpg
            per_image_path = os.path.join(root_dataset_path,
                                          LOAD_IMAGE_DIR_NAME,
                                          per_image_relative_path)
            if not check_image_file_exists(per_image_path,
                                           dir_file_name_cache_dict):
                missing_image_count += 1
                continue

            # 落盘图像的文件名前缀就是sample_id,和标注里的sample_id必须完全一致,
            # 不一致说明上游成员错位,这种样本没法保证保存图像名唯一,直接丢弃
            per_image_name_prefix = os.path.splitext(
                os.path.basename(per_image_relative_path))[0].lower()
            per_sample_id = per_annotation.get(ANNOTATION_IMAGE_NAME_KEY_NAME,
                                               '')
            if not isinstance(per_sample_id, str):
                per_sample_id = ''
            per_sample_id = per_sample_id.strip().lower()

            if not per_sample_id or per_sample_id != per_image_name_prefix:
                invalid_image_name_count += 1
                print('2222', per_image_path, per_sample_id)
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

            # 实测上游task_type恒为text_to_image,一旦出现别的取值说明上游规格变了,
            # 这种样本不能进t2i数据集
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
            # (上游已校验caption_detail全部非空,实测抽样8万条最短252)
            if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                invalid_caption_count += 1
                print('3333', per_image_path, len(per_t2i_caption))
                continue

            # 过长描述同样视为不合格样本对(实测抽样里超过1536的占0.27%)
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
            # 记录的宽高,不一致只计数上报、不丢样本(上游报告全部是1024x1024)
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

            # 保存图像名要等切完子集目录才能拼出来,这里只带上sample_id前缀
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

    这里只listdir unzip_annotations下的四个系列目录拿到1180个jsonl路径,
    绝不去os.walk图像目录: 上游解出589万个小文件,扫目录树在NAS上不可接受。
    每个jsonl里的图像全在同一个分片目录下,worker只需要对那个目录listdir一次。
    最后按[系列名, sample_id]统一排序,保证输出顺序与串行版本完全一致。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    series_annotation_file_count_dict = {}
    for per_series_name in LOAD_SERIES_DIR_NAME_LIST:
        per_series_annotation_path = os.path.join(root_annotation_path,
                                                  per_series_name)
        if not os.path.isdir(per_series_annotation_path):
            print('2222', per_series_annotation_path)
            series_annotation_file_count_dict[per_series_name] = 0
            continue

        per_series_annotation_file_count = 0
        for per_jsonl_name in sorted(os.listdir(per_series_annotation_path)):
            if not per_jsonl_name.endswith(LOAD_ANNOTATION_FILE_SUFFIX):
                continue

            # 系列名由主进程按目录名算好后带给worker,
            # worker只认标注文件、数据集根目录、系列名这三个入参
            annotation_file_pair_list.append([
                os.path.join(per_series_annotation_path, per_jsonl_name),
                root_dataset_path,
                per_series_name,
            ])
            per_series_annotation_file_count += 1

        series_annotation_file_count_dict[
            per_series_name] = per_series_annotation_file_count

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    series_annotation_count_dict = {}
    image_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            image_annotation_pair_list.extend(per_load_result[0])
            # 按系列累计上游标注条数,用来和上游实测行数逐子集硬对账
            series_annotation_count_dict[
                per_load_result[1]] = series_annotation_count_dict.get(
                    per_load_result[1], 0) + per_load_result[2]
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
        series_annotation_file_count_dict,
        series_annotation_count_dict,
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
    """按sample_id全局去重,每个sample_id只保留排序后的第一条

    sample_id自带子集前缀(aesthetics-part01-00985000),实测抽样16万条100%唯一、
    上游的对账报告也是0重复,所以这里正常应该一条都不丢,只是兜一道底:
    撞名会让后写的图像覆盖先写的、静默丢样本。
    排序键取[sample_id, 系列名],保证同一个sample_id留下的永远是同一条。
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
    """把过滤后的合格样本按系列分组,排序后每10000张切成一个文件夹、每100个文件夹归一个子集目录

    切分必须在过滤全部完成之后做,且切分前先按sample_id排序,这样才能保证每个文件夹都是
    满10000张(只有每个系列全局最后一个文件夹允许不满)。
    排序键用sample_id而不是保存图像名: 保存图像名里含子集目录名,而子集目录名恰恰由排序后
    的位置决定,存在循环依赖;同一个子集目录内所有图像名前缀完全相同,
    所以按sample_id排序与按保存图像名排序结果完全等价。
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
                # 保存图像名统一全小写,形如
                # flux_reason_6m_aesthetics-part01_000_aesthetics-part01-00985000.jpg
                # (sample_id本身自带子集前缀,所以子集名会在图像名里出现两次,
                #  这是"数据集名+子集名+原始图像名前缀"这个命名规则的必然结果)
                per_save_image_name = f'{DATASET_NAME}_{per_set_name}_{per_image_name_prefix}{SAVE_IMAGE_NAME_SUFFIX}'
                per_folder_save_pair_list.append([
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

            # 一个文件夹就是一个写盘任务,worker写完这10000张后直接写出该文件夹的json,
            # 主进程只收计数,不用把589万条记录再攒一遍
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
    以文件夹为任务粒度而不是以单张图为粒度: 本数据集约589万张图,逐图收结果的话
    主进程要再攒一份589万条的列表,而且中途挂了只能从头再来;按文件夹收之后
    主进程内存只和文件夹数(约589)相关,且json已经写全的文件夹可以直接跳过、支持断点续跑。
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
    文件夹都允许不满;这里子集目录只是100个文件夹的容器,Aesthetics-Part02被切成约3个
    子集目录,所以只有整个系列全局最后一个文件夹允许不满,
    aesthetics-part02_000/_001下的每个文件夹都必须是满10000张。
    同理每个系列只有最后一个子集目录允许不满100个文件夹。
    约589个文件夹每个都要listdir一万个文件再load一份json,串行跑在NAS上太久,
    所以这一步也按文件夹粒度开多进程。
    """
    check_error_message_list = []

    folder_check_pair_list = []
    for per_series_name in sorted(series_set_name_list_dict.keys()):
        per_series_set_name_list = series_set_name_list_dict[per_series_name]
        for per_set_index, per_set_name in enumerate(per_series_set_name_list):
            per_set_folder_count = set_folder_count_dict[per_set_name]

            # 每个系列只有最后一个子集目录允许不满100个文件夹
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

    image_annotation_pair_list, series_annotation_file_count_dict, series_annotation_count_dict, total_annotation_count, load_annotation_failed_count, missing_image_count, invalid_image_name_count, invalid_task_type_count, invalid_caption_count, too_long_caption_count, invalid_image_count, annotation_image_size_not_match_count = get_all_image_annotation_pair(
        root_dataset_path)

    print('1111', series_annotation_file_count_dict,
          series_annotation_count_dict, total_annotation_count,
          load_annotation_failed_count, missing_image_count,
          invalid_image_name_count, invalid_task_type_count,
          invalid_caption_count, too_long_caption_count,
          invalid_image_count, annotation_image_size_not_match_count,
          len(image_annotation_pair_list))

    if len(image_annotation_pair_list) > 0:
        print('1111', image_annotation_pair_list[0])

    # 上游jsonl分片数量或标注条数不对说明上游没跑完,继续跑只会静默少样本对
    annotation_file_error_message_list = []
    for per_series_name, per_expected_annotation_file_num in EXPECTED_SERIES_ANNOTATION_FILE_NUM_DICT.items(
    ):
        per_annotation_file_num = series_annotation_file_count_dict.get(
            per_series_name, 0)
        if per_annotation_file_num != per_expected_annotation_file_num:
            annotation_file_error_message_list.append(
                f'{per_series_name} annotation file num not match {per_annotation_file_num} != {per_expected_annotation_file_num}'
            )

    for per_series_name, per_expected_row_count in EXPECTED_SERIES_ROW_COUNT_DICT.items(
    ):
        per_series_annotation_count = series_annotation_count_dict.get(
            per_series_name, 0)
        if per_series_annotation_count != per_expected_row_count:
            annotation_file_error_message_list.append(
                f'{per_series_name} annotation count not match {per_series_annotation_count} != {per_expected_row_count}'
            )

    if total_annotation_count != EXPECTED_TOTAL_ROW_COUNT:
        annotation_file_error_message_list.append(
            f'total annotation count not match {total_annotation_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )

    if len(annotation_file_error_message_list) > 0:
        raise Exception(
            f'check annotation file failed {annotation_file_error_message_list}'
        )

    # 上游对账报告是0缺图/0非法图像名/0非text_to_image,这三类在本脚本里正常一条
    # 都不该命中: 一旦出现非0,要么上游产物被改动过,要么本脚本的路径/字段口径写错了
    # (比如图像路径少拼一层images就会让589万条全部变成缺图),
    # 这种情况继续跑只会静默产出一个残缺甚至完全空的数据集,所以直接硬拦
    load_annotation_error_message_list = []
    if load_annotation_failed_count > 0:
        load_annotation_error_message_list.append(
            f'load annotation failed count {load_annotation_failed_count}')
    if missing_image_count > 0:
        load_annotation_error_message_list.append(
            f'missing image count {missing_image_count}')
    if invalid_image_name_count > 0:
        load_annotation_error_message_list.append(
            f'invalid image name count {invalid_image_name_count}')
    if invalid_task_type_count > 0:
        load_annotation_error_message_list.append(
            f'invalid task type count {invalid_task_type_count}')

    if len(load_annotation_error_message_list) > 0:
        raise Exception(
            f'load annotation failed {load_annotation_error_message_list}')

    image_annotation_pair_list, duplicate_image_name_prefix_list = get_deduplicated_image_annotation_pair(
        image_annotation_pair_list)

    print('1111', len(image_annotation_pair_list),
          len(duplicate_image_name_prefix_list))
    if len(duplicate_image_name_prefix_list) > 0:
        print('1111', duplicate_image_name_prefix_list[:10])

    # 一个合格样本都不剩一定是硬错误(过滤口径写错/上游产物不对),
    # 不能让后面的切分和收尾自校验在空列表上"全部通过"
    if len(image_annotation_pair_list) == 0:
        raise Exception('no valid image annotation pair found')

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
        'series_annotation_file_count_dict':
        series_annotation_file_count_dict,
        'series_annotation_count_dict':
        series_annotation_count_dict,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/FLUX-Reason-6M'
    save_dataset_path = r'/root/autodl-tmp/t2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
