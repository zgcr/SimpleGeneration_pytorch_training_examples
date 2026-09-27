import os
import re
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'megastyle'

# 上游目录名叫MegaStyle-1.4M,但实测100片parquet合计8,000,000行,
# 是README里v2.0(MegaStyle++-8M)那一版(1M细粒度风格 x 每风格8张图)。
# 这里按**实际数据规模**命名,不沿用会误导人的1.4M
SAVE_DATASET_DIR_NAME = 'MegaStyle-8M'

# ==============================================================================
# 【数据集类型判定】MegaStyle(tencent/MegaStyle-1.4M,实际是MegaStyle++-8M)
# 是纯文生图(text-to-image)数据集,**不能处理成图像编辑数据集**,
# 所以本目录下这个数据集只有这一个t2i脚本、没有对应的resave ti2i脚本。
#
# 判定依据(实测,不是照抄上游注释):
# 1) 上游013解出的jsonl每行固定15个字段(抽10个分片x2000行,key组合100%一致),
#    里面**没有任何reference_image / input_image / mask / edit_instruction字段**,
#    task_type全量8000000行恒为text_to_image(上游对账报告
#    dataset_task_type=text_to_image、invalid_sample_pair_count=0),
#    README的task_categories也只写text-to-image。
# 2) 唯一可能拼出编辑对的路线是"同一条content在A风格下的图 -> 在B风格下的图"
#    (即风格迁移编辑对),**这条路线实测被证伪**:
#    取train-00000与train-00050共有的8208个content_index,对其中4对图做
#    8x8网格灰度对比,平均绝对灰度差分别是 46.3 / 68.9 / 88.3 / 89.8。
#    真正同构的风格迁移前后对这个指标应在个位数到十几。
#    也就是说Qwen-Image在不同风格提示下是**重新构图重新生成**,两张图除了
#    "文字描述的场景相同"以外,主体位置/数量/朝向/构图全都对不上,
#    根本不是"编辑前图/编辑后图"的同构关系。硬当编辑对训练等于教模型
#    "按指令把画面整个重画一遍",是错误监督。
# 3) 配对量本身也不够: 抽4个分片32万行只有25.7万个唯一content_index,
#    重复分布为 1次 202877 / 2次 46034 / 3次 7148 / 4次 788 / 5次 87 / 6次 4,
#    绝大多数content在库里只出现一两次。
#
# 【上游013解出的目录规格(读上游对账报告 + 实测抽查,不是推测)】
# MegaStyle-1.4M/
# ├── unzip_annotations/train-{00000..00099}.jsonl  100个,合计8,000,000行
# ├── images/train-{00000..00099}/{0000..0007}/<sample_id>.png
# │                                                 8,000,000张512x512 PNG(2.4T)
# ├── unzip_style_attributes/train-*.jsonl          100个,风格属性切片(本脚本不读)
# ├── unzip_source_annotations/{metadata.csv,style_indices.pkl}  原始真值(本脚本不读)
# └── unzip_check_missing_images.json               上游对账报告:
#         total_valid_sample_pair_count=8000000 / total_duplicate_sample_count=0 /
#         invalid_sample_pair_count=0 / check_error_count=0 /
#         image_suffix全是.png / image_shape全是512x512
#
# 本脚本只读这100个jsonl定位样本对,**绝不os.walk图像目录**:
# 上游一共解出800万个png,扫一遍目录树在NAS上不可接受。
#
# 【jsonl每行固定15个字段(实测抽10个分片x2000行,key组合100%一致、无缺字段)】
#   image_path / sample_id / parquet_name / row_index / global_row_index /
#   style_index / content_index / task_type / t2i_caption / content / style /
#   style_attribute_dict / image_width / image_height / image_suffix
#
# 【全量/抽样实测结论】
# - 8,000,000行,sample_id(形如s1_c26226)抽48万行100%唯一、0重复、
#   100%匹配^s\d+_c\d+$,上游报告duplicate也是0;
# - image_path形如 train-00000/0000/s1_c26226.png(相对images/,不含images/这一段),
#   第一段恒等于jsonl文件名(分片名)、第二段是4位桶号(每片8万张按1万一桶分8个桶);
# - t2i_caption(strip后)长度抽48万行 min 164 / p50 269 / max 766,
#   >400只有597条、>700只有16条,没有一条小于164;
# - 图像抽1000张读PNG头部: 100%是 bitdepth=8 / colortype=2(真RGB,
#   无alpha/灰度/调色板) / 非隔行 / 512x512,尾部IEND完整;
# - 标注宽高全是512x512,即"短边<64"和"宽高比>8"这两条过滤全量一条都不会命中
#   (仍照写兜底,且过滤一律以真实解码shape为准,不信标注里的宽高);
# - 上游有2条warning: train-00054的row 11607(s541451_c695615.png)与
#   row 43393(s545425_c105369.png),jsonl里image_width/image_height被写成0。
#   实测这两个文件大小都是327339字节、PNG头部IHDR正常解出512x512x8bit RGB、
#   尾部IEND完整,是上游在内存BytesIO上用PIL解码时的瞬时失败,图像本身没坏。
#   按方案这两条**不写特例**: 宽高一律以cv2实际解码的shape为准,能解码就正常保留
#   (见ANNOTATION_ZERO_IMAGE_SIZE的说明)。
#
# 【caption口径: 直接取上游已经拼好的t2i_caption,本脚本不再重拼】
# 该数据集的图是用"内容提示content x 风格提示style"合成出来的,
# 拼接后的完整提示才是当初真正喂给Qwen-Image的prompt,也是唯一能无歧义决定
# 这张图的文本(只用content时同一条caption对应8种画风的图、只用style时对应
# 8个不相干场景,两种都是互相矛盾的监督信号)。
# 上游013已经按 content(不以.!?结尾就补句点) + ' ' + style 拼好写进t2i_caption,
# 本脚本核对过样本,拼接口径正确,所以**直接取用、不再重拼**,避免两处口径漂移。
#
# 【上游标注里另外这些属性,按方案确认全部丢弃,新标注只留
#   width/height/t2i_caption/t2i_caption_length。落盘后**永久丢失**,
#   无法从新数据集恢复(上游产物会保留,可回头重跑)】
# style_attribute_dict : 9个结构化风格属性(overall_artistic_style/dominant_colors/
#                supporting_colors/light/visual_pattern/surface_status/medium/
#                brushwork/edge_rendering)。实测缺失率 brushwork 47.0% /
#                visual_pattern 64.4% / supporting_colors 10.7% /
#                edge_rendering 14.3% / medium 0.04%。
#                这是本数据集相对普通t2i数据集最独有的标注,丢弃后无法再按风格属性
#                做条件训练或数据均衡采样
# style_index  : 0..999999的风格分组键。丢弃后**无法再按style成组采样做风格一致性
#                正则、也无法做风格dropout的风格CFG**,而这正是该数据集的核心价值。
#                注意: sample_id以保存图像名后半段的形式保留(形如
#                megastyle_train_000_s1_c26226.jpg),其中的s{N}就是style_index+1,
#                所以风格分组关系其实还能从**文件名**反解,但9个结构化属性是真没了
# content_index: 内容提示分组键(同一条content会在多个风格下复用),同样只能从
#                文件名里的c{N}反解
# content / style : 拼接前的两段原文。丢弃后只剩拼好的整串,
#                **无法再做"只丢style保content"的风格dropout**
# parquet_name / row_index / global_row_index / image_path / image_suffix :
#                上游定位用。分片归属信息落盘后无法恢复,下游不能再"只训某几个分片"
# task_type    : 恒为text_to_image,只用于校验,不落盘
# image_width / image_height : 由真实解码后的shape得到的width/height承载
# ==============================================================================

# 上游013解压脚本按parquet分片另存的汇总标注目录
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_SUFFIX = '.jsonl'

# 上游把图像按 <分片名>/<桶号>/<sample_id>.png 落在这里,
# jsonl里的image_path是相对这个目录的路径(不含images/这一段)
LOAD_IMAGE_DIR_NAME = 'images'

# 本数据集没有官方的train/val/test划分(原始只有train-*.parquet一种分片),
# 100个分片只是parquet切片、不是语义子集,所以整个数据集只有一个系列train,
# 8000000条过滤后按sample_id全局排序统一重切
LOAD_SERIES_NAME = 'train'

# 实测上游jsonl分片数(100个parquet各一个),数量不对说明上游没跑完
EXPECTED_ANNOTATION_FILE_NUM = 100

# 实测上游有效样本对数,与上游对账报告里的total_valid_sample_pair_count完全一致,
# 直接当完整性ground truth: 缺分片/少行都能拦住
EXPECTED_TOTAL_ROW_COUNT = 8000000

ANNOTATION_IMAGE_KEY_NAME = 'image_path'

# sample_id形如s1_c26226(s{风格号}_c{内容号}),等于落盘图像的文件名前缀,
# 实测抽48万行100%唯一(上游报告duplicate_sample_count也是0),
# 所以它可以直接当去重键,也直接当保存图像名的后半段
ANNOTATION_IMAGE_NAME_KEY_NAME = 'sample_id'

# jsonl里另存的分片名,必须和jsonl文件名一致,只用于校验成员是否错位,不进新标注
ANNOTATION_PARQUET_NAME_KEY_NAME = 'parquet_name'

# 上游已经拼好的完整文生图提示(content + ' ' + style),是本数据集唯一的训练文本
ANNOTATION_CAPTION_KEY_NAME = 't2i_caption'

# 上游解码PNG头部拿到的宽高。本脚本不拿它做分辨率过滤(万一它和实际图像不一致
# 就会出偏差),只在解码后顺手比对一次,不一致的条数只上报不丢样本
ANNOTATION_IMAGE_WIDTH_KEY_NAME = 'image_width'

ANNOTATION_IMAGE_HEIGHT_KEY_NAME = 'image_height'

# 上游解码失败时把宽高写成0(实测全量只有train-00054的2条)。
# 这种行不参与"标注宽高 vs 实际宽高"的比对(0 vs 512不是真错位,是上游解码瞬时失败),
# 单独计一个数上报;样本本身照常走真实解码校验,能解码就保留、解不出才被判掉
ANNOTATION_ZERO_IMAGE_SIZE = 0

# 上游给每行都标了task_type,实测恒为text_to_image。
# 一旦出现别的取值说明上游规格变了(比如混进了编辑样本),这种样本不能进t2i数据集,
# 直接丢弃并计数上报
ANNOTATION_TASK_TYPE_KEY_NAME = 'task_type'

EXPECTED_ANNOTATION_TASK_TYPE = 'text_to_image'

SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 原始图像名前缀必须是sample_id那种s{数字}_c{数字}的组合(形如s1_c26226),
# 不符的样本没法保证保存图像名全局唯一(可能去覆盖别的样本),直接丢弃并上报
VALID_IMAGE_NAME_PREFIX_PATTERN = re.compile(r'^s\d+_c\d+$')

# jsonl文件名(去掉后缀)就是parquet分片名,形如train-00042
VALID_PARQUET_GROUP_PATTERN = re.compile(r'^train-\d{5}$')

# image_path中间那一段是桶号,上游每片8万张按1万一桶分成8个子目录(0000..0007)
VALID_IMAGE_SUB_DIR_PATTERN = re.compile(r'^\d{4}$')

# image_path固定是 <分片名>/<桶号>/<sample_id>.png 三段,段数不对说明上游规格变了
EXPECTED_IMAGE_RELATIVE_PATH_NAME_NUM = 3

# 只保留RGB三通道图,灰度图/P图/RGBA图/CMYK图等一律过滤掉。
# 实测抽1000张PNG头部100%是colortype=2(真RGB),这条实际不会命中,照写兜底
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 本机128核,这里和001/002/010保持一致取32。本数据集要对800万张图各解码两遍
# (jsonl扫描阶段校验一次、写盘阶段重编码一次),想跑快可以直接调大这个常量
PROCESS_NUM = 32

PER_FOLDER_IMAGE_NUM = 10000

# 每个子集目录放100个文件夹(即100万张图)。8000000张切出800个文件夹,
# 全塞进一个目录下NAS元数据压力太大,所以每100个文件夹再归入一个子集目录,
# 按方案确认取100,会切出train_000..train_007共8个子集目录
PER_SET_FOLDER_NUM = 100

# train系列按PER_SET_FOLDER_NUM切成带编号的子集目录。
# 即使只出一个子集目录也带编号,保证和002/008/009/010的子集目录命名规格一致
SPLIT_SET_DIR_SERIES_NAME_LIST = [
    'train',
]

# 实测800万张图全部是512x512,所以这两条过滤实际一条都不会命中,但仍然照写:
# 只要有一张图不符合就必须被判掉,不能靠"实测都合格"这个假设去省掉校验
# (而且过滤一律以真实解码出来的shape为准,不信标注里的宽高)
MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 实测抽48万条t2i_caption没有一条strip后长度小于164,所以这条实际也不会命中,
# 同样照写兜底
MIN_CAPTION_LENGTH = 10

# 实测抽48万条t2i_caption(strip后)长度 min 164 / p50 269 / max 766,
# 超过400的只有597条(0.12%)、超过700的只有16条(0.003%)。
# 所以阈值取1024时一条都不会丢,既和002/010保持同一个量级口径,又不会误砍正常样本
MAX_CAPTION_LENGTH = 1024


def get_set_name(per_series_name, per_set_index):
    """子集目录名: train按每PER_SET_FOLDER_NUM个文件夹切成train_000/train_001/...

    子集目录名同时是保存图像名的中间段,所以它必须在切分完成后才能确定,
    这也是后面排序键只能用sample_id而不能用保存图像名的原因。
    本数据集800个文件夹会切出train_000..train_007共8个子集目录。
    """
    if per_series_name not in SPLIT_SET_DIR_SERIES_NAME_LIST:
        return per_series_name

    return f'{per_series_name}_{per_set_index:03d}'


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上,逐样本打一次os.path.exists就是一次网络往返,
    800万条标注就要打800万次,这一步本身就能占掉整个扫描阶段的大头。
    实测同一个jsonl里的图像全部落在同一个分片目录下的8个桶子目录里
    (train-00000.jsonl的image_path都是train-00000/000{0..7}/xxx.png),
    所以这里按目录缓存一次os.listdir的结果,之后只做集合查表,
    网络往返次数从"标注条数"(8万/片)降到"桶目录数"(8/片,全量800次)。
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

    这个worker把文本层过滤(image_path段数/分片名/桶号不合规、parquet_name与
    jsonl文件名不一致、缺图、图像名与sample_id不一致、图像名非法、
    task_type不是text_to_image、描述为空或过短、描述过长)和图像层过滤
    (能否解码、是否RGB、短边、宽高比)一次做完。
    图像校验没有像001那样单独再开一个Pool,是因为本数据集有800万条标注:
    分两个Pool的话主进程要先攒800万条记录、再逐条发给check worker、再收回800万条,
    光进程间序列化就要来回搬上百GB,而合并进来之后IPC只传存活样本。
    判定逻辑、过滤口径、日志编号和001/002/010完全一致,图像也一样是解码两遍
    (这里校验一遍、写盘时重编码再解一遍),没有为了省时间跳过任何一道校验。
    """
    per_jsonl_path, root_dataset_path, per_series_name, per_parquet_group_name = annotation_file_pair

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    annotation_zero_image_size_count = 0
    image_annotation_pair_list = []

    # 每个worker只处理一个jsonl,缓存里只有该分片的8个桶目录,内存开销可忽略
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
            annotation_zero_image_size_count,
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

            # 上游image_path形如 train-00000/0000/s1_c26226.png(相对images/),
            # 第一段必须是本jsonl对应的分片名、第二段必须是4位桶号,
            # 不一致说明上游成员错位,这种样本定位到的图像不属于本片,直接丢弃
            per_image_relative_path = per_image_relative_path.replace(
                '\\', '/').lstrip('/')
            per_image_relative_path_name_list = per_image_relative_path.split(
                '/')
            if len(per_image_relative_path_name_list
                   ) != EXPECTED_IMAGE_RELATIVE_PATH_NAME_NUM:
                invalid_image_name_count += 1
                print('2222', per_jsonl_path, per_image_relative_path)
                continue

            if per_image_relative_path_name_list[
                    0] != per_parquet_group_name or not VALID_IMAGE_SUB_DIR_PATTERN.match(
                        per_image_relative_path_name_list[1]):
                invalid_image_name_count += 1
                print('2222', per_jsonl_path, per_image_relative_path)
                continue

            # jsonl里另存的parquet_name也必须和jsonl文件名一致,兜一道底
            per_annotation_parquet_name = per_annotation.get(
                ANNOTATION_PARQUET_NAME_KEY_NAME, '')
            if not isinstance(per_annotation_parquet_name, str):
                per_annotation_parquet_name = ''
            if per_annotation_parquet_name.strip() != per_parquet_group_name:
                invalid_image_name_count += 1
                print('2222', per_jsonl_path, per_annotation_parquet_name)
                continue

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

            # 上游已经拼好的 content + ' ' + style,本脚本不再重拼
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
            # (上游已校验content与style全部非空,实测抽48万条最短164)
            if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                invalid_caption_count += 1
                print('3333', per_image_path, len(per_t2i_caption))
                continue

            # 过长描述同样视为不合格样本对(实测最长766,这条一条都不会命中)
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
            # 记录的宽高,不一致只计数上报、不丢样本。
            # 上游解码失败写成0的那2条(train-00054)不算真错位,单独计数,
            # 不参与比对,也不丢样本(图像本身实测完好,能正常解出512x512)
            per_annotation_image_w = per_annotation.get(
                ANNOTATION_IMAGE_WIDTH_KEY_NAME, per_image_w)
            per_annotation_image_h = per_annotation.get(
                ANNOTATION_IMAGE_HEIGHT_KEY_NAME, per_image_h)
            if per_annotation_image_w == ANNOTATION_ZERO_IMAGE_SIZE or per_annotation_image_h == ANNOTATION_ZERO_IMAGE_SIZE:
                annotation_zero_image_size_count += 1
                print('2222', per_image_path, per_image_w, per_image_h,
                      per_annotation_image_w, per_annotation_image_h)
            elif per_annotation_image_w != per_image_w or per_annotation_image_h != per_image_h:
                annotation_image_size_not_match_count += 1
                print('2222', per_image_path, per_image_w, per_image_h,
                      per_annotation_image_w, per_annotation_image_h)

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
        annotation_zero_image_size_count,
    ]


def get_all_image_annotation_pair(root_dataset_path):
    """扫描上游解压好的jsonl标注,多进程组装图像路径和t2i描述的样本对列表

    这里只listdir unzip_annotations一层拿到100个jsonl路径,
    绝不去os.walk图像目录: 上游解出800万个png,扫目录树在NAS上不可接受。
    每个jsonl里的图像全在同一个分片目录下的8个桶里,worker只需要对这8个目录
    各listdir一次。
    最后按[系列名, sample_id]统一排序,保证输出顺序与串行版本完全一致。
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

            # jsonl文件名去掉后缀就是parquet分片名,由主进程算好后带给worker,
            # worker只认标注文件、数据集根目录、系列名、分片名这四个入参
            per_parquet_group_name = per_jsonl_name[:-len(
                LOAD_ANNOTATION_FILE_SUFFIX)]
            if not VALID_PARQUET_GROUP_PATTERN.match(per_parquet_group_name):
                # 分片名规格变了说明上游产物不是本脚本认识的那一份,必须显式感知
                invalid_annotation_file_name_list.append(per_jsonl_name)
                print('2222', root_annotation_path, per_jsonl_name)
                continue

            annotation_file_pair_list.append([
                os.path.join(root_annotation_path, per_jsonl_name),
                root_dataset_path,
                LOAD_SERIES_NAME,
                per_parquet_group_name,
            ])

    annotation_file_count = len(annotation_file_pair_list)

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    annotation_zero_image_size_count = 0
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
            annotation_zero_image_size_count += per_load_result[11]

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
        annotation_zero_image_size_count,
    ]


def get_deduplicated_image_annotation_pair(image_annotation_pair_list):
    """按sample_id全局去重,每个sample_id只保留排序后的第一条

    sample_id形如s1_c26226,是"风格号 + 内容号"的组合,实测抽48万行100%唯一、
    上游的对账报告duplicate_sample_count也是0,所以这里正常应该一条都不丢,
    只是兜一道底: 撞名会让后写的图像覆盖先写的、静默丢样本。
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
    """把过滤后的合格样本按系列分组,排序后每10000张切成一个文件夹、每PER_SET_FOLDER_NUM个文件夹归一个子集目录

    切分必须在过滤全部完成之后做,且切分前先按sample_id排序,这样才能保证每个文件夹
    都是满10000张(只有每个系列全局最后一个文件夹允许不满)。
    排序键用sample_id而不是保存图像名: 保存图像名里含子集目录名,而子集目录名恰恰
    由排序后的位置决定,存在循环依赖;同一个子集目录内所有图像名前缀完全相同,
    所以按sample_id排序与按保存图像名排序结果完全等价。
    上游的100个分片只是parquet切片、不是语义子集,所以这里把它们并成同一个train系列
    统一重切,800万张会切出800个文件夹、8个子集目录(train_000..train_007)。
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
                # 保存图像名统一全小写,形如megastyle_train_000_s1_c26226.jpg
                per_save_image_name = f'{DATASET_NAME}_{per_set_name}_{per_image_name_prefix}{SAVE_IMAGE_NAME_SUFFIX}'
                per_folder_save_pair_list.append([
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

            # 一个文件夹就是一个写盘任务,worker写完这10000张后直接写出该文件夹的json,
            # 主进程只收计数,不用把800万条记录再攒一遍
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
    以文件夹为任务粒度而不是以单张图为粒度: 本数据集800万张图,逐图收结果的话
    主进程要再攒一份800万条的列表,而且中途挂了只能从头再来;按文件夹收之后
    主进程内存只和文件夹数(800)相关,且json已经写全的文件夹可以直接跳过、支持断点续跑。
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

    和001的差别只在"满10000张"的口径: 001里子集本身就是切分单位,所以每个子集的最后
    一个文件夹都允许不满;这里子集目录只是PER_SET_FOLDER_NUM个文件夹的容器,
    本数据集800个文件夹会切成train_000..train_007,所以只有整个train系列全局最后
    一个文件夹允许不满,train_000..train_006下的每个文件夹都必须是满10000张。
    同理每个系列只有最后一个子集目录允许不满PER_SET_FOLDER_NUM个文件夹。
    800个文件夹每个都要listdir一万个文件再load一份json,串行跑在NAS上太久,
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

    image_annotation_pair_list, annotation_file_count, invalid_annotation_file_name_list, total_annotation_count, load_annotation_failed_count, missing_image_count, invalid_image_name_count, invalid_task_type_count, invalid_caption_count, too_long_caption_count, invalid_image_count, annotation_image_size_not_match_count, annotation_zero_image_size_count = get_all_image_annotation_pair(
        root_dataset_path)

    print('1111', annotation_file_count,
          len(invalid_annotation_file_name_list), total_annotation_count,
          load_annotation_failed_count, missing_image_count,
          invalid_image_name_count, invalid_task_type_count,
          invalid_caption_count, too_long_caption_count, invalid_image_count,
          annotation_image_size_not_match_count,
          annotation_zero_image_size_count, len(image_annotation_pair_list))

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
          annotation_image_size_not_match_count, 'annotation zero image size:',
          annotation_zero_image_size_count, 'duplicate image name:',
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
        'annotation_zero_image_size_count':
        annotation_zero_image_size_count,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/MegaStyle-1.4M'
    save_dataset_path = r'/root/autodl-tmp/t2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
