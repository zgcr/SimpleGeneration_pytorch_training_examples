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
# 数据集: ScaleEdit-12M(InternVL-U/ScaleEdit-12M)
#
# 【数据集类型】纯图像编辑(instruction-based image editing)数据集，不是文生图数据集。
# README原文 task_categories: image-to-image，tags: image-editing/instruction-based-editing，
# 每行固定是"1张参考图(编辑前图) + 1条英文编辑指令 + 1张编辑后图"，
# 没有第二张视觉条件图、没有mask、没有任何"只有caption+一张图"的纯生成样本。
# 所以下游只能走ti2i_dataset.py那条链路，不能当t2i(文生图)数据用。
#
# 【root_dataset_path实测原始保存规格(共6.7T / 251个parquet / 11845023行)】
# ScaleEdit-12M/
# ├── 23个 <category_id>_<task_name>/ 任务族子目录(见SUBSET_ROOT_DIR_NAME_LIST)
# │   └── <task_name>_%04d.parquet    每目录的分片编号实测0..N-1**全部连号**，
# │                                   单片最多50000行(约31GB)，**每片只有1个row group**
# ├── README.md         数据集说明(无用)
# ├── .gitattributes    git lfs配置(无用)
# └── .cache/           huggingface下载缓存，548个文件，**残留39个*.incomplete**(无用)
#
# 完整性已用huggingface下载缓存里的仓库文件清单
# .cache/huggingface/trees/*.json 对账过: 清单里共253个文件
# (251个parquet + README.md + .gitattributes)，**磁盘上全部存在且字节数100%一致**，
# 所以.cache里那39个*.incomplete只是缓存残渣，数据集本体是完整的。
#
# 251片的schema**完全一致**(15列 / SNAPPY / parquet-cpp-arrow version 15.0.1)，
# 且逐片用footer统计量验证过:
# - edit_task恒等于所在目录名去掉<category_id>_前缀后的task_name(min==max);
# - id是**子集内**从0开始的连续编号，片内 num_rows == id_max - id_min + 1(251/251成立)，
#   且同一子集内相邻分片的id区间首尾相接、覆盖0..该子集行数-1，
#   所以 <subset_name>/<id> 就是全局唯一样本key，不需要再做跨片去重。
#
# 【单行parquet的全部15列(全部是有用信息)】
#   id                        : int64，子集内唯一样本id                    -> 有用(样本唯一id/图像名前缀)
#   edit_task                 : 任务族名(style_transfer/object_addition等) -> 有用(按任务族采样/加权)
#   edit_instruction          : **英文编辑指令**，251片null=0             -> 有用(训练主文本，必需)
#   source_image              : 参考图(编辑前图)原始字节，抽样恒为JPEG      -> 有用(参考图，必需)
#                               **但有702078行为null**，见下面的规格坑
#   source_image_url          : 源图URL，仅当source_image为null时非空       -> 有用(回捞源图)
#   source_image_sha256       : 源图字节的sha256，仅当source_image为null时非空 -> 有用(回捞后校验同一份内容)
#   source_image_fetch_date   : 采集时间(ISO-8601)，仅当source_image为null时非空 -> 有用(合规溯源)
#   edited_image              : 编辑后图原始字节，**null=0恒存在**，抽样恒为JPEG -> 有用(编辑后图，必需)
#   source_image_width/height : 参考图原始宽高                             -> 有用(分辨率分桶可不解码图像)
#   edited_image_width/height : 编辑后图原始宽高                           -> 有用(同上)
#                               抽样16244行，声明宽高与真实解码尺寸**100%一致**
#   instruction_following_score / editing_consistency_score /
#   generation_quality_score  : 三维质量分(1-3)                            -> 有用(质量过滤/加权)
#                               实测全库 IF恒为3、EC与GQ都落在[2,3]，
#                               与README"只保留IF=3, EC>=2, GQ>=2"的口径一致
#
# 【无用信息(一律不整理进训练目录)】
# .cache/(548个文件，含39个*.incomplete) / .gitattributes / README.md /
# .DS_Store / CACHEDIR.TAG 这类目录元数据垃圾文件。
#
# 【**必须显式感知的规格坑: 702078行只有源图URL、没有源图字节**】
# README原文: 一部分源图来自公开网络抓取，过了隐私/安全/水印审查后
# **只以URL形式发布、不再rehost字节**，此时:
#   source_image == null 且 source_image_url/_sha256/_fetch_date 三者非空。
# 实测 source_image 的null数 = 702078(5.93%)，
# 而 source_image_url/_sha256/_fetch_date 三列的null数**都恰好等于11142945**，
# 与702078严格互补(11845023 - 702078 = 11142945)，即两种行严格二分、无交叉。
# 这702078行**没有参考图字节，所以不是"包含完整有用信息的编辑样本对"**，
# 但它的编辑后图/指令/URL/sha256/采集时间/宽高/3个打分**全都是有用信息，一条都不能丢**。
# 本脚本的口径(与用户确认过):
#   - 完整编辑对(11142945行): 参考图 + 编辑后图都落盘，标注写
#     unzip_annotations/<subset>/<parquet>.jsonl，可直接训练;
#   - 仅有URL的对(702078行): **编辑后图照样落盘**(有用信息不丢)，
#     全部属性(含url/sha256/fetch_date)写
#     unzip_url_source_annotations/<subset>/<parquet>.jsonl，
#     标 source_image_state='url_only'，等后续按URL+sha256回捞源图即可直接补齐;
#   - 两个清单条数与行数三方硬对账: 片内 完整对 + url_only对 + invalid对 == 行数,
#     全局 11142945 + 702078 == 11845023，且逐子集与实测ground truth比对，
#     任何一处不等立即抛异常。
#
# 【本脚本的处理口径】
# - 解包前预检(硬失败，不过就不白跑几十小时):
#   根目录条目白名单(多出未知文件/未知子集目录立即上报)、
#   23个子集目录必须都存在、每目录parquet数与实测ground truth逐一比对、
#   分片名必须是<task_name>_%04d.parquet且task_name与目录名后缀一致、
#   分片编号必须0..N-1连号、每片PAR1头尾魔数(O(1)读，拦下载截断)、
#   只读footer校验15列schema一致 + edit_task取值 + id区间自洽 +
#   逐子集行数/逐子集source_image为null的行数/全局总行数全部硬对账;
# - 并行单位 = 单个parquet(251个任务，Pool(32))，
#   pq.iter_batches(batch_size=32)流式读，**绝不整片进内存**
#   (实测单片28GB的parquet流式读峰值RSS只有约2.8GB);
# - 图像落盘 unzip_images/<subset>/<parquet>/<id>_source.jpg 与 <id>_edited.jpg,
#   直接写原始字节，**unzip阶段绝不引入二次编解码**，
#   resize/转格式/分辨率分桶留给preprocessing2的resave脚本;
# - 每张图写盘后**立刻校验落盘大小 == len(bytes)**(比只看存在性强，能挡住写半截/写0字节);
#   已存在且大小一致就计skip并跳过，保证脚本可以断点续跑;
# - 片内四方硬对账: 遍历行数 == footer num_rows、
#   编辑后图成员数 == 行数、参考图成员数 == 行数 - url_only行数、
#   extract+skip+not_save+fail == 两类图像成员数之和、
#   完整对 + url_only对 + invalid对 == 行数、片内id无重复且区间与footer一致;
# - 绝不静默丢样本对: 指令为空/编辑后图字节为空/参考图字节为空且URL三件套不全/
#   写盘失败/片内id重名，全部分门别类记进对应隔离清单并在汇总报告里上报;
#   片内id重名时改写到 unzip_duplicate_samples/ 独立目录保留数据，不互相覆盖;
# - 拷贝/解包/校验任一环出错都汇总后抛异常，不再静默跑过。
#
# 【跑之前务必确认目标盘扛得住】
# - EXTRACT_IMAGE_FILE_FLAG=True 时输出小文件数约 **2300万张jpg**
#   (11845023张编辑后图 + 11142945张参考图)，约6.7T，NAS上inode与元数据压力大;
# - 只想先建索引可把 EXTRACT_IMAGE_FILE_FLAG 置False，
#   图像继续留在原parquet里，样本对信息一样完整。
# ==============================================================================

DATASET_TASK_TYPE = 'image_edit'

DATASET_LICENSE_NAME = 'cc-by-nc-sa-4.0'

PARQUET_FILE_NAME_PATTERN = re.compile(r'^(?P<prefix>.+)\.parquet$')

# 带分片编号的parquet名(style_transfer_0000.parquet)，用于分片完整性预检
PARQUET_SHARD_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>(?P<task_name>.+)_(?P<index>\d{4}))\.parquet$')

# 子集目录名规格: <category_id>_<task_name>(1.1_style_transfer)
SUBSET_ROOT_DIR_NAME_PATTERN = re.compile(
    r'^(?P<category_id>\d+\.\d+)_(?P<task_name>.+)$')

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(548个文件，含39个*.incomplete)
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

# 过滤掉无用信息后根目录只应该剩这23个任务族子集目录
SUBSET_ROOT_DIR_NAME_LIST = [
    '1.1_style_transfer',
    '1.2_tone_adjustment',
    '1.3_viewpoint_transformation',
    '1.4_background_replacement',
    '2.1_object_addition',
    '2.2_object_removal',
    '2.3_object_replacement',
    '2.4_action_editing',
    '2.5_part_extraction',
    '3.1_color_change',
    '3.2_material_change',
    '3.3_visual_beautification',
    '3.4_count_change',
    '3.5_size_change',
    '4.1_movie_poster_text_editing',
    '4.2_gui_interface_text_editing',
    '4.3_object_surface_text_editing',
    '4.4_building_surface_text_editing',
    '5.1_perceptual_reasoning',
    '5.2_symbolic_reasoning',
    '5.3_social_reasoning',
    '5.4_scientific_reasoning',
    '6.1_compositional_editing',
]

# 实测每子集parquet分片数(合计251)，数量不对说明下载不全
EXPECTED_SUBSET_PARQUET_NUM_DICT = {
    '1.1_style_transfer': 16,
    '1.2_tone_adjustment': 12,
    '1.3_viewpoint_transformation': 1,
    '1.4_background_replacement': 24,
    '2.1_object_addition': 33,
    '2.2_object_removal': 30,
    '2.3_object_replacement': 51,
    '2.4_action_editing': 5,
    '2.5_part_extraction': 11,
    '3.1_color_change': 24,
    '3.2_material_change': 7,
    '3.3_visual_beautification': 2,
    '3.4_count_change': 1,
    '3.5_size_change': 1,
    '4.1_movie_poster_text_editing': 8,
    '4.2_gui_interface_text_editing': 3,
    '4.3_object_surface_text_editing': 8,
    '4.4_building_surface_text_editing': 7,
    '5.1_perceptual_reasoning': 1,
    '5.2_symbolic_reasoning': 1,
    '5.3_social_reasoning': 1,
    '5.4_scientific_reasoning': 1,
    '6.1_compositional_editing': 3,
}

# 实测每子集parquet footer行数总和(合计11845023)，直接当作完整性ground truth,
# 缺子集/缺分片/少行都能拦住
EXPECTED_SUBSET_ROW_COUNT_DICT = {
    '1.1_style_transfer': 760806,
    '1.2_tone_adjustment': 590702,
    '1.3_viewpoint_transformation': 872,
    '1.4_background_replacement': 1197765,
    '2.1_object_addition': 1621863,
    '2.2_object_removal': 1486445,
    '2.3_object_replacement': 2544193,
    '2.4_action_editing': 214211,
    '2.5_part_extraction': 506789,
    '3.1_color_change': 1163602,
    '3.2_material_change': 319472,
    '3.3_visual_beautification': 64544,
    '3.4_count_change': 458,
    '3.5_size_change': 2019,
    '4.1_movie_poster_text_editing': 384822,
    '4.2_gui_interface_text_editing': 107557,
    '4.3_object_surface_text_editing': 392555,
    '4.4_building_surface_text_editing': 330578,
    '5.1_perceptual_reasoning': 4089,
    '5.2_symbolic_reasoning': 3908,
    '5.3_social_reasoning': 3479,
    '5.4_scientific_reasoning': 8806,
    '6.1_compositional_editing': 135488,
}

# 实测每子集"source_image为null(只有源图URL)"的行数(合计702078)。
# 这是上游数据集自身的发布规格(网络抓取源图只给URL不rehost字节)，不是本地下载缺失，
# 所以只能当ground truth对账，不能当成缺图错误
EXPECTED_SUBSET_URL_SOURCE_ONLY_ROW_COUNT_DICT = {
    '1.1_style_transfer': 44208,
    '1.2_tone_adjustment': 43042,
    '1.3_viewpoint_transformation': 441,
    '1.4_background_replacement': 79239,
    '2.1_object_addition': 102145,
    '2.2_object_removal': 83223,
    '2.3_object_replacement': 88071,
    '2.4_action_editing': 42704,
    '2.5_part_extraction': 40634,
    '3.1_color_change': 81106,
    '3.2_material_change': 33731,
    '3.3_visual_beautification': 15478,
    '3.4_count_change': 0,
    '3.5_size_change': 0,
    '4.1_movie_poster_text_editing': 0,
    '4.2_gui_interface_text_editing': 0,
    '4.3_object_surface_text_editing': 1124,
    '4.4_building_surface_text_editing': 0,
    '5.1_perceptual_reasoning': 2754,
    '5.2_symbolic_reasoning': 2055,
    '5.3_social_reasoning': 1675,
    '5.4_scientific_reasoning': 2631,
    '6.1_compositional_editing': 37817,
}

EXPECTED_TOTAL_ROW_COUNT = 11845023

EXPECTED_TOTAL_URL_SOURCE_ONLY_ROW_COUNT = 702078

# 参考图与编辑后图都齐备的完整编辑样本对数 = 总行数 - 只有源图URL的行数
EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT = 11142945

# 实测单片最大行数(绝大多数片都恰好是50000，每子集最后一片是余数)，
# 只做软校验(打印告警)，因为官方没承诺分片切分规格，不能拿来当硬性失败条件
EXPECTED_PARQUET_ROW_COUNT = 50000

# 实测每片只有1个row group，多于1个不影响正确性(流式读一样)，只打印告警
EXPECTED_PARQUET_ROW_GROUP_NUM = 1

# parquet里应该齐备的全部15列列名，少列/多列说明上游数据规格变了，必须显式感知
PARQUET_COLUMN_NAME_LIST = [
    'id',
    'edit_task',
    'edit_instruction',
    'source_image',
    'source_image_url',
    'source_image_sha256',
    'source_image_fetch_date',
    'edited_image',
    'source_image_width',
    'source_image_height',
    'edited_image_width',
    'edited_image_height',
    'instruction_following_score',
    'editing_consistency_score',
    'generation_quality_score',
]

PARQUET_SAMPLE_ID_COLUMN_NAME = 'id'

PARQUET_EDIT_TASK_COLUMN_NAME = 'edit_task'

# 训练主文本 = edit_instruction(英文编辑指令)，实测251片null=0、抽样无空串,
# 为空则该样本对没有文本条件、不可训练，隔离上报
PARQUET_INSTRUCTION_COLUMN_NAME = 'edit_instruction'

# 参考图(编辑前图)与编辑后图的原始字节列，这两列是唯一需要落盘成图像文件的列
PARQUET_SOURCE_IMAGE_COLUMN_NAME = 'source_image'

PARQUET_EDITED_IMAGE_COLUMN_NAME = 'edited_image'

PARQUET_IMAGE_BYTES_COLUMN_NAME_LIST = [
    PARQUET_SOURCE_IMAGE_COLUMN_NAME,
    PARQUET_EDITED_IMAGE_COLUMN_NAME,
]

# source_image为null时必须齐备的源图溯源列(实测三列null数完全一致、与source_image严格互补)
PARQUET_URL_SOURCE_COLUMN_NAME_LIST = [
    'source_image_url',
    'source_image_sha256',
    'source_image_fetch_date',
]

PARQUET_SOURCE_IMAGE_SHAPE_COLUMN_NAME_LIST = [
    'source_image_width',
    'source_image_height',
]

PARQUET_EDITED_IMAGE_SHAPE_COLUMN_NAME_LIST = [
    'edited_image_width',
    'edited_image_height',
]

# 三维质量分列，实测全部齐备且落在[1, 3]内(实际取值IF恒为3、EC与GQ在[2,3])，
# 缺失或越界只上报不丢样本
PARQUET_SCORE_COLUMN_NAME_LIST = [
    'instruction_following_score',
    'editing_consistency_score',
    'generation_quality_score',
]

ANNOTATION_SCORE_VALUE_RANGE = [1, 3]

# 源图状态: 有原始字节 / 只有源图URL
SOURCE_IMAGE_STATE_EMBED_BYTES = 'embed_bytes'

SOURCE_IMAGE_STATE_URL_ONLY = 'url_only'

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_SOURCE_IMAGE_NAME_SUFFIX = '_source'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

# 只有源图URL、没有源图字节的样本对单独落一份索引，不混进可直接训练的标注里
SAVE_URL_SOURCE_ANNOTATION_DIR_NAME = 'unzip_url_source_annotations'

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
# True : 和其他数据集脚本口径一致，约2300万张jpg(约6.7T)，
#        NAS上inode和元数据压力极大，务必确认目标盘扛得住再跑;
# False: 只解析非图像列生成 unzip_annotations/*.jsonl 与
#        unzip_url_source_annotations/*.jsonl 索引(几小时即可跑完)，
#        图像继续留在原parquet里，训练时按parquet顺序读，样本对信息一样是完整的。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解包后再os.walk一遍输出目录做二次对账。
# 默认False: 2300万个小文件的os.walk在NAS上要跑非常久，而解包时已经做了
# "写盘后立刻校验落盘大小 == len(bytes)" + "两类图像成员数与行数四方对账"两道对账，
# 已经能保证每一行的图都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 是否解码图像header拿真实宽高，并与parquet里声明的宽高交叉校验。
# 默认True: 只解header不解像素(PIL的Image.open是惰性的)，代价可忽略，
# 实测抽样16244行声明宽高与真实尺寸100%一致，不一致只记warning不丢样本。
PARSE_IMAGE_SHAPE_FLAG = True

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

# 单片最大约31GB且只有1个row group，必须小batch流式读:
# 实测batch_size=32时单进程峰值RSS约2.8G，32进程约90G，机器内存2T足够
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

    该数据集parquet里只存裸图像字节、不带文件名，实测source/edited图全部是JPEG魔数。
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


def get_parquet_subset_name(per_parquet_relative_dir):
    """parquet所在的相对目录名就是子集名(1.1_style_transfer等)"""
    per_parquet_relative_dir = per_parquet_relative_dir.replace('\\', '/')

    return per_parquet_relative_dir.split('/')[0]


def get_subset_category_id_and_task_name(per_subset_name):
    """把子集目录名拆成<category_id>与<task_name>，拆不开时返回空串由调用方上报"""
    per_match_result = SUBSET_ROOT_DIR_NAME_PATTERN.match(per_subset_name)
    if not per_match_result:
        return '', ''

    return per_match_result.group('category_id'), per_match_result.group(
        'task_name')


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

    实测251个parquet全部满足，说明当前数据集是完整的。
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
    """只读单个parquet的footer拿行数/列名/edit_task取值/id区间/源图null数，不碰任何图像字节

    每片只有1个row group但这里仍按全部row group累加，规格变了也不会算错。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    per_subset_name = get_parquet_subset_name(per_parquet_relative_dir)
    per_parquet_relative_path = f'{per_subset_name}/{per_parquet_group_name}'

    per_metadata_dict = {
        'parquet_group_name': per_parquet_group_name,
        'parquet_relative_dir': per_parquet_relative_dir,
        'parquet_relative_path': per_parquet_relative_path,
        'subset_name': per_subset_name,
        'row_count': 0,
        'row_group_num': 0,
        'column_name_list': [],
        'edit_task_value_list': [],
        'sample_id_min': -1,
        'sample_id_max': -1,
        'url_source_only_row_count': -1,
        'edited_image_null_count': -1,
        'instruction_null_count': -1,
    }

    try:
        load_parquet_file = pq.ParquetFile(per_parquet_path)
        per_parquet_metadata = load_parquet_file.metadata
        per_column_name_list = list(load_parquet_file.schema_arrow.names)
    except Exception as e:
        print('7777', per_parquet_relative_path, e)

        return per_metadata_dict, [
            f'read parquet metadata failed {per_parquet_relative_path} {e}',
        ]

    error_message_list = []

    per_metadata_dict['row_count'] = per_parquet_metadata.num_rows
    per_metadata_dict['row_group_num'] = per_parquet_metadata.num_row_groups
    per_metadata_dict['column_name_list'] = per_column_name_list

    if per_column_name_list != PARQUET_COLUMN_NAME_LIST:
        error_message_list.append(
            f'{per_parquet_relative_path} column name not match {per_column_name_list}'
        )

        return per_metadata_dict, error_message_list

    per_column_index_dict = {
        per_column_name: per_column_index
        for per_column_index, per_column_name in enumerate(
            per_column_name_list)
    }

    per_edit_task_value_dict = {}
    per_sample_id_min, per_sample_id_max = None, None
    per_url_source_only_row_count = 0
    per_edited_image_null_count, per_instruction_null_count = 0, 0
    for per_row_group_index in range(per_parquet_metadata.num_row_groups):
        per_row_group_metadata = per_parquet_metadata.row_group(
            per_row_group_index)

        for per_column_name, per_column_index in per_column_index_dict.items():
            per_column_statistics = per_row_group_metadata.column(
                per_column_index).statistics
            if per_column_statistics is None:
                # footer里没有统计量就没法做O(1)预检，必须显式感知
                error_message_list.append(
                    f'{per_parquet_relative_path} row group {per_row_group_index} column {per_column_name} statistics not exist'
                )
                continue

            if per_column_name == PARQUET_EDIT_TASK_COLUMN_NAME:
                per_edit_task_value_dict[per_column_statistics.min] = 1
                per_edit_task_value_dict[per_column_statistics.max] = 1
            elif per_column_name == PARQUET_SAMPLE_ID_COLUMN_NAME:
                per_sample_id_min = per_column_statistics.min if per_sample_id_min is None else min(
                    per_sample_id_min, per_column_statistics.min)
                per_sample_id_max = per_column_statistics.max if per_sample_id_max is None else max(
                    per_sample_id_max, per_column_statistics.max)
            elif per_column_name == PARQUET_SOURCE_IMAGE_COLUMN_NAME:
                per_url_source_only_row_count += per_column_statistics.null_count
            elif per_column_name == PARQUET_EDITED_IMAGE_COLUMN_NAME:
                per_edited_image_null_count += per_column_statistics.null_count
            elif per_column_name == PARQUET_INSTRUCTION_COLUMN_NAME:
                per_instruction_null_count += per_column_statistics.null_count

    per_metadata_dict['edit_task_value_list'] = sorted(
        per_edit_task_value_dict.keys())
    per_metadata_dict[
        'sample_id_min'] = -1 if per_sample_id_min is None else per_sample_id_min
    per_metadata_dict[
        'sample_id_max'] = -1 if per_sample_id_max is None else per_sample_id_max
    per_metadata_dict[
        'url_source_only_row_count'] = per_url_source_only_row_count
    per_metadata_dict['edited_image_null_count'] = per_edited_image_null_count
    per_metadata_dict['instruction_null_count'] = per_instruction_null_count

    return per_metadata_dict, error_message_list


def check_parquet_shard_complete(parquet_group_list):
    """解包前预检: 每子集的分片数、分片名规格、分片编号连号，缺片直接中止"""
    error_message_list = []

    subset_shard_index_dict = {}
    for per_parquet_group_name, per_parquet_relative_dir, per_parquet_path in parquet_group_list:
        per_subset_name = get_parquet_subset_name(per_parquet_relative_dir)
        per_parquet_name = os.path.basename(per_parquet_path)

        if per_parquet_relative_dir.replace('\\', '/') != per_subset_name:
            # 子集目录下不应该再有下一层目录
            error_message_list.append(
                f'unknown parquet relative dir {per_parquet_relative_dir}')
            continue

        _, per_subset_task_name = get_subset_category_id_and_task_name(
            per_subset_name)
        if not per_subset_task_name:
            error_message_list.append(f'unknown subset dir {per_subset_name}')
            continue

        per_match_result = PARQUET_SHARD_FILE_NAME_PATTERN.match(
            per_parquet_name)
        if not per_match_result:
            error_message_list.append(
                f'unknown parquet name {per_subset_name}/{per_parquet_name}')
            continue

        if per_match_result.group('task_name') != per_subset_task_name:
            # 分片名里的task_name必须和目录名后缀一致，不一致说明文件放错目录
            error_message_list.append(
                f'parquet task name not match subset dir {per_subset_name}/{per_parquet_name}'
            )
            continue

        subset_shard_index_dict.setdefault(per_subset_name, {})[int(
            per_match_result.group('index'))] = per_parquet_group_name

    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        per_shard_index_dict = subset_shard_index_dict.get(per_subset_name, {})
        per_expected_parquet_num = EXPECTED_SUBSET_PARQUET_NUM_DICT[
            per_subset_name]

        print('1111', per_subset_name, 'parquet:', len(per_shard_index_dict),
              'expected parquet:', per_expected_parquet_num)

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
                f'{per_subset_name} parquet index not continuous, missing index {per_missing_shard_index_list[:10]}'
            )

    for per_subset_name in sorted(subset_shard_index_dict.keys()):
        if per_subset_name not in EXPECTED_SUBSET_PARQUET_NUM_DICT:
            error_message_list.append(
                f'unknown subset dir in parquet group {per_subset_name}')

    return error_message_list


def check_parquet_metadata_complete(parquet_group_list):
    """只读footer按子集与实测ground truth逐项硬对账

    对账项: 15列schema一致、edit_task取值 == 目录名后缀、
    片内 num_rows == id_max - id_min + 1、同一子集内分片id区间首尾相接且覆盖0..行数-1、
    编辑后图与编辑指令的null数必须为0、
    逐子集行数 / 逐子集只有源图URL的行数 / 全局总行数全部与写死的实测值相等。
    """
    error_message_list = []

    total_row_count, total_url_source_only_row_count = 0, 0
    per_parquet_metadata_dict = {}
    subset_row_count_dict = collections.Counter()
    subset_url_source_only_row_count_dict = collections.Counter()
    subset_parquet_num_dict = collections.Counter()
    subset_sample_id_range_dict = {}

    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_file_metadata, parquet_group_list),
                                     total=len(parquet_group_list)):
            per_metadata_dict, per_error_message_list = per_check_result
            error_message_list.extend(per_error_message_list)

            per_parquet_relative_path = per_metadata_dict[
                'parquet_relative_path']
            per_subset_name = per_metadata_dict['subset_name']
            per_row_count = per_metadata_dict['row_count']

            per_parquet_metadata_dict[
                per_parquet_relative_path] = per_metadata_dict
            subset_row_count_dict[per_subset_name] += per_row_count
            subset_url_source_only_row_count_dict[per_subset_name] += max(
                per_metadata_dict['url_source_only_row_count'], 0)
            subset_parquet_num_dict[per_subset_name] += 1
            total_row_count += per_row_count
            total_url_source_only_row_count += max(
                per_metadata_dict['url_source_only_row_count'], 0)

            if per_metadata_dict[
                    'row_group_num'] != EXPECTED_PARQUET_ROW_GROUP_NUM:
                # 多个row group不影响流式读的正确性，只打印告警
                print('2222', per_parquet_relative_path, 'row group num',
                      per_metadata_dict['row_group_num'])

            if per_row_count > EXPECTED_PARQUET_ROW_COUNT:
                print('2222', per_parquet_relative_path,
                      'row count larger than', EXPECTED_PARQUET_ROW_COUNT,
                      per_row_count)

            _, per_subset_task_name = get_subset_category_id_and_task_name(
                per_subset_name)
            if per_metadata_dict['edit_task_value_list'] != [
                    per_subset_task_name
            ]:
                error_message_list.append(
                    f'{per_parquet_relative_path} edit task value not match {per_metadata_dict["edit_task_value_list"]} != [{per_subset_task_name}]'
                )

            if per_metadata_dict['edited_image_null_count'] != 0:
                # 编辑后图是每一行都必须有的，为null就是不完整样本对
                error_message_list.append(
                    f'{per_parquet_relative_path} edited image null count {per_metadata_dict["edited_image_null_count"]}'
                )
            if per_metadata_dict['instruction_null_count'] != 0:
                error_message_list.append(
                    f'{per_parquet_relative_path} edit instruction null count {per_metadata_dict["instruction_null_count"]}'
                )

            per_sample_id_min = per_metadata_dict['sample_id_min']
            per_sample_id_max = per_metadata_dict['sample_id_max']
            if per_sample_id_min < 0 or per_sample_id_max < 0:
                error_message_list.append(
                    f'{per_parquet_relative_path} sample id range not exist')
            elif per_sample_id_max - per_sample_id_min + 1 != per_row_count:
                # id片内连续，不连续说明分片内容被改过
                error_message_list.append(
                    f'{per_parquet_relative_path} sample id range not match row count {per_sample_id_min} {per_sample_id_max} {per_row_count}'
                )
            else:
                subset_sample_id_range_dict.setdefault(
                    per_subset_name, []).append([
                        per_sample_id_min,
                        per_sample_id_max,
                        per_parquet_relative_path,
                    ])

    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        per_expected_row_count = EXPECTED_SUBSET_ROW_COUNT_DICT[
            per_subset_name]
        per_expected_url_source_only_row_count = EXPECTED_SUBSET_URL_SOURCE_ONLY_ROW_COUNT_DICT[
            per_subset_name]
        per_row_count = subset_row_count_dict.get(per_subset_name, 0)
        per_url_source_only_row_count = subset_url_source_only_row_count_dict.get(
            per_subset_name, 0)

        print('1111', per_subset_name, 'parquet:',
              subset_parquet_num_dict.get(per_subset_name,
                                          0), 'row:', per_row_count,
              'expected row:', per_expected_row_count, 'url source only row:',
              per_url_source_only_row_count, 'expected url source only row:',
              per_expected_url_source_only_row_count)

        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_subset_name} row count not match {per_row_count} != {per_expected_row_count}'
            )
        if per_url_source_only_row_count != per_expected_url_source_only_row_count:
            error_message_list.append(
                f'{per_subset_name} url source only row count not match {per_url_source_only_row_count} != {per_expected_url_source_only_row_count}'
            )

        # 同一子集内所有分片的id区间必须首尾相接且刚好覆盖0..该子集行数-1，
        # 这样 <subset_name>/<id> 才是全局唯一key，下游可以直接靠它对账去重
        per_sample_id_range_list = sorted(subset_sample_id_range_dict.get(
            per_subset_name, []),
                                          key=lambda x: x[0])
        if len(per_sample_id_range_list) != subset_parquet_num_dict.get(
                per_subset_name, 0):
            error_message_list.append(
                f'{per_subset_name} sample id range num not match {len(per_sample_id_range_list)} != {subset_parquet_num_dict.get(per_subset_name, 0)}'
            )
            continue

        if len(per_sample_id_range_list) == 0:
            continue

        if per_sample_id_range_list[0][0] != 0:
            error_message_list.append(
                f'{per_subset_name} sample id min not 0 {per_sample_id_range_list[0][0]}'
            )
        if per_sample_id_range_list[-1][1] != per_row_count - 1:
            error_message_list.append(
                f'{per_subset_name} sample id max not match {per_sample_id_range_list[-1][1]} != {per_row_count - 1}'
            )
        for per_range_index in range(len(per_sample_id_range_list) - 1):
            if per_sample_id_range_list[per_range_index][
                    1] + 1 != per_sample_id_range_list[per_range_index + 1][0]:
                error_message_list.append(
                    f'{per_subset_name} sample id range not continuous {per_sample_id_range_list[per_range_index][2]} {per_sample_id_range_list[per_range_index + 1][2]}'
                )

    print('1111', 'total row:', total_row_count, 'expected total row:',
          EXPECTED_TOTAL_ROW_COUNT, 'total url source only row:',
          total_url_source_only_row_count,
          'expected total url source only row:',
          EXPECTED_TOTAL_URL_SOURCE_ONLY_ROW_COUNT, 'parquet:',
          len(per_parquet_metadata_dict))

    if total_row_count != EXPECTED_TOTAL_ROW_COUNT:
        error_message_list.append(
            f'total row count not match {total_row_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )
    if total_url_source_only_row_count != EXPECTED_TOTAL_URL_SOURCE_ONLY_ROW_COUNT:
        error_message_list.append(
            f'total url source only row count not match {total_url_source_only_row_count} != {EXPECTED_TOTAL_URL_SOURCE_ONLY_ROW_COUNT}'
        )
    if total_row_count - total_url_source_only_row_count != EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total valid sample pair count not match {total_row_count - total_url_source_only_row_count} != {EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT}'
        )

    return total_row_count, per_parquet_metadata_dict, error_message_list


def check_required_subset_complete(root_dataset_path, parquet_group_list):
    """解包前预检: 根目录条目白名单、子集目录、parquet数量与连号、PAR1魔数、footer对账

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


def get_single_row_image_bytes(per_row_dict, per_image_column_name):
    """取出一行里某个图像列的原始字节，None/空字节都当成缺失"""
    per_image_bytes = per_row_dict.get(per_image_column_name, None)
    if not isinstance(per_image_bytes, bytes) or len(per_image_bytes) == 0:
        return None

    return per_image_bytes


def get_single_row_source_image_state(per_row_dict, per_source_image_bytes):
    """判定一行的源图状态: 有原始字节 / 只有源图URL

    source_image为null时，source_image_url/_sha256/_fetch_date三列必须齐备
    (实测三列null数与source_image严格互补)，缺任一列说明连URL都拿不到、
    该行既没有参考图也没法回捞，属于不完整样本对，必须隔离上报。
    """
    if per_source_image_bytes is not None:
        return SOURCE_IMAGE_STATE_EMBED_BYTES, []

    invalid_reason_list = []
    for per_column_name in PARQUET_URL_SOURCE_COLUMN_NAME_LIST:
        if not get_stripped_text_value(per_row_dict.get(per_column_name,
                                                        None)):
            invalid_reason_list.append(f'empty {per_column_name}')

    return SOURCE_IMAGE_STATE_URL_ONLY, invalid_reason_list


def get_single_row_score_warning_message_list(per_row_dict):
    """校验三维质量分是否齐备且在[1, 3]内，只上报不丢样本"""
    warning_message_list = []
    for per_column_name in PARQUET_SCORE_COLUMN_NAME_LIST:
        if per_column_name not in per_row_dict:
            warning_message_list.append(f'miss key {per_column_name}')
            continue

        per_score_value = per_row_dict[per_column_name]
        if not isinstance(per_score_value, int) or isinstance(
                per_score_value, bool):
            warning_message_list.append(
                f'{per_column_name} not a int {per_score_value}')
            continue

        if not ANNOTATION_SCORE_VALUE_RANGE[
                0] <= per_score_value <= ANNOTATION_SCORE_VALUE_RANGE[1]:
            warning_message_list.append(
                f'{per_column_name} out of range {per_score_value}')

    return warning_message_list


def get_single_row_image_shape_warning_message_list(per_image_shape,
                                                    per_row_dict,
                                                    per_shape_column_name_list,
                                                    per_image_name):
    """交叉校验真实解码宽高与parquet里声明的宽高，只上报不丢样本

    实测抽样16244行两者100%一致，一旦不一致说明上游宽高列不可信，
    下游做分辨率分桶就不能再免解码，必须显式感知。
    """
    warning_message_list = []
    if not PARSE_IMAGE_SHAPE_FLAG:
        return warning_message_list

    if per_image_shape[0] <= 0 or per_image_shape[1] <= 0:
        return warning_message_list

    per_declare_image_shape = [
        per_row_dict.get(per_column_name, None)
        for per_column_name in per_shape_column_name_list
    ]
    if per_declare_image_shape != per_image_shape:
        warning_message_list.append(
            f'{per_image_name} image shape not match declare shape {per_image_shape} != {per_declare_image_shape}'
        )

    return warning_message_list


def get_single_sample_pair_annotation(
        per_row_dict, per_sample_key, per_subset_name, per_category_id,
        per_parquet_group_name, per_row_index, per_source_image_state,
        per_source_image_relative_path, per_edited_image_relative_path,
        per_instruction, per_source_image_shape, per_edited_image_shape):
    """拼一条完整样本对标注: 保留原parquet全部15列的有用属性 + 落盘路径等补充属性

    下游可以直接按行取样本，不需要为了拿指令去扫2300万个小文件。
    """
    per_source_image_path_list = [per_source_image_relative_path
                                  ] if per_source_image_relative_path else []

    per_save_annotation = {
        'dataset_task_type': DATASET_TASK_TYPE,
        'sample_key': per_sample_key,
        'subset_name': per_subset_name,
        'category_id': per_category_id,
        'parquet_name': per_parquet_group_name,
        'row_index': per_row_index,
        'source_image_state': per_source_image_state,
        'instruction': per_instruction,
        'reference_image_path_list': per_source_image_path_list,
        'reference_image_num': len(per_source_image_path_list),
        'edited_image_path': per_edited_image_relative_path,
        'source_image_shape': per_source_image_shape,
        'edited_image_shape': per_edited_image_shape,
    }

    for per_column_name, per_column_value in per_row_dict.items():
        if per_column_name in PARQUET_IMAGE_BYTES_COLUMN_NAME_LIST:
            # 图像字节已经落盘成图像文件，标注里只留路径
            continue
        per_save_annotation[per_column_name] = get_json_serializable_value(
            per_column_value)

    return per_save_annotation


def process_single_parquet_file(parquet_group, save_dataset_path,
                                save_annotation_dir_path,
                                save_url_source_annotation_dir_path):
    """流式解开单个parquet，把内嵌图像字节写成图像文件、非图像列写成jsonl汇总标注

    落盘结构:
      unzip_images/<subset>/<分片名>/<id>_source.jpg
      unzip_images/<subset>/<分片名>/<id>_edited.jpg
      unzip_annotations/<subset>/<分片名>.jsonl              参考图+编辑后图都齐备的完整编辑对
      unzip_url_source_annotations/<subset>/<分片名>.jsonl   只有源图URL的样本对

    parquet按iter_batches流式读，绝不整片进内存(单片最大约31GB)。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    per_subset_name = get_parquet_subset_name(per_parquet_relative_dir)
    per_category_id, _ = get_subset_category_id_and_task_name(per_subset_name)
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
    save_url_source_annotation_path = os.path.join(
        save_url_source_annotation_dir_path, per_subset_name,
        f'{per_parquet_group_name}.jsonl')

    if EXTRACT_IMAGE_FILE_FLAG:
        os.makedirs(save_image_dir_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_url_source_annotation_path),
                exist_ok=True)

    row_count = 0
    valid_sample_pair_count, url_source_only_sample_pair_count = 0, 0
    source_image_member_count, edited_image_member_count = 0, 0
    extract_image_count, skip_image_count = 0, 0
    not_save_image_count, save_image_fail_count = 0, 0
    duplicate_sample_count = 0
    edit_task_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    edited_image_shape_count_dict = collections.Counter()
    score_count_dict = collections.Counter()
    invalid_sample_pair_list, warning_message_list = [], []
    error_message_list = []
    reach_parquet_end = False

    sample_id_dict = {}

    try:
        load_parquet_file = pq.ParquetFile(per_parquet_path)
        expected_row_count = load_parquet_file.metadata.num_rows

        with open(save_annotation_path, 'w',
                  encoding='UTF-8') as save_annotation_file:
            with open(save_url_source_annotation_path, 'w',
                      encoding='UTF-8') as save_url_source_annotation_file:
                for per_record_batch in load_parquet_file.iter_batches(
                        batch_size=PARQUET_ROW_BATCH_SIZE):
                    for per_row_dict in per_record_batch.to_pylist():
                        per_row_index = row_count
                        row_count += 1

                        per_sample_id = per_row_dict.get(
                            PARQUET_SAMPLE_ID_COLUMN_NAME, None)
                        per_sample_id_name = f'{per_sample_id}' if per_sample_id is not None else ''

                        # 同一片里出现重名id时按名写盘会互相覆盖，
                        # 这里改写到独立目录保留数据并上报，不能静默丢样本
                        per_sample_is_duplicate = bool(
                            per_sample_id_name
                        ) and per_sample_id_name in sample_id_dict
                        if per_sample_is_duplicate:
                            duplicate_sample_count += 1
                            error_message_list.append(
                                f'duplicate sample id {per_sample_id_name} row {per_row_index}'
                            )
                        elif per_sample_id_name:
                            sample_id_dict[per_sample_id_name] = per_row_index

                        # id为空或片内重名时退化用分片名+行号，保证图像名不冲突
                        per_image_name_prefix = per_sample_id_name if per_sample_id_name else f'{per_parquet_group_name}_{per_row_index:08d}'
                        if per_sample_is_duplicate:
                            per_image_name_prefix = f'{per_image_name_prefix}_{per_row_index:08d}'

                        per_sample_key = f'{per_subset_name}/{per_image_name_prefix}'

                        per_invalid_reason_list = []

                        per_instruction = get_stripped_text_value(
                            per_row_dict.get(PARQUET_INSTRUCTION_COLUMN_NAME,
                                             None))
                        if not per_instruction:
                            # 图像编辑样本对必须有编辑指令，没有指令的图不可训练
                            per_invalid_reason_list.append('empty instruction')

                        per_source_image_bytes = get_single_row_image_bytes(
                            per_row_dict, PARQUET_SOURCE_IMAGE_COLUMN_NAME)
                        per_edited_image_bytes = get_single_row_image_bytes(
                            per_row_dict, PARQUET_EDITED_IMAGE_COLUMN_NAME)

                        per_source_image_state, per_source_invalid_reason_list = get_single_row_source_image_state(
                            per_row_dict, per_source_image_bytes)
                        per_invalid_reason_list.extend(
                            per_source_invalid_reason_list)

                        if per_edited_image_bytes is None:
                            # 编辑后图实测251片null=0，恒存在，为空就是不完整样本对
                            per_invalid_reason_list.append(
                                'empty edited image bytes')

                        warning_message_list.extend([
                            f'row {per_row_index} {per_warning_message}'
                            for per_warning_message in
                            get_single_row_score_warning_message_list(
                                per_row_dict)
                        ])

                        per_save_image_dir_path = save_duplicate_dir_path if per_sample_is_duplicate else save_image_dir_path
                        per_save_image_relative_dir = f'{SAVE_DUPLICATE_SAMPLE_DIR_NAME}/{per_subset_name}/{per_parquet_group_name}' if per_sample_is_duplicate else f'{SAVE_IMAGE_DIR_NAME}/{per_subset_name}/{per_parquet_group_name}'

                        per_source_image_relative_path = ''
                        per_edited_image_relative_path = ''
                        per_source_image_shape = [0, 0]
                        per_edited_image_shape = [0, 0]

                        for per_image_bytes, per_image_name_suffix, per_shape_column_name_list in [
                            [
                                per_source_image_bytes,
                                SAVE_SOURCE_IMAGE_NAME_SUFFIX,
                                PARQUET_SOURCE_IMAGE_SHAPE_COLUMN_NAME_LIST,
                            ],
                            [
                                per_edited_image_bytes,
                                SAVE_EDITED_IMAGE_NAME_SUFFIX,
                                PARQUET_EDITED_IMAGE_SHAPE_COLUMN_NAME_LIST,
                            ],
                        ]:
                            if per_image_bytes is None:
                                continue

                            per_is_source_image = per_image_name_suffix == SAVE_SOURCE_IMAGE_NAME_SUFFIX
                            if per_is_source_image:
                                source_image_member_count += 1
                            else:
                                edited_image_member_count += 1

                            per_image_suffix = get_image_bytes_suffix(
                                per_image_bytes)
                            image_suffix_count_dict[per_image_suffix] += 1

                            per_save_image_name = f'{per_image_name_prefix}{per_image_name_suffix}{per_image_suffix}'
                            per_image_relative_path = f'{per_save_image_relative_dir}/{per_save_image_name}'

                            per_image_shape, per_image_shape_error_message = get_image_shape(
                                per_image_bytes)
                            if per_image_shape_error_message:
                                warning_message_list.append(
                                    f'row {per_row_index} {per_save_image_name} {per_image_shape_error_message}'
                                )
                            else:
                                warning_message_list.extend([
                                    f'row {per_row_index} {per_warning_message}'
                                    for per_warning_message in
                                    get_single_row_image_shape_warning_message_list(
                                        per_image_shape, per_row_dict,
                                        per_shape_column_name_list,
                                        per_save_image_name)
                                ])

                            if not EXTRACT_IMAGE_FILE_FLAG:
                                # 只建索引模式: 图像继续留在原parquet里，样本对信息一样完整
                                not_save_image_count += 1
                            else:
                                per_write_flag, per_skip_flag, per_save_error_message = save_single_image_bytes(
                                    os.path.join(per_save_image_dir_path,
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

                            if per_is_source_image:
                                per_source_image_relative_path = per_image_relative_path
                                per_source_image_shape = per_image_shape
                            else:
                                per_edited_image_relative_path = per_image_relative_path
                                per_edited_image_shape = per_image_shape

                        if not per_edited_image_relative_path:
                            per_invalid_reason_list.append(
                                'missing edited image')

                        if len(per_invalid_reason_list) > 0:
                            # 信息不完整的样本对不写进任何有效标注，但必须留痕，不能静默消失
                            invalid_sample_pair_list.append({
                                'sample_key':
                                per_sample_key,
                                'subset_name':
                                per_subset_name,
                                'parquet_name':
                                per_parquet_group_name,
                                'row_index':
                                per_row_index,
                                'source_image_state':
                                per_source_image_state,
                                'invalid_reason':
                                ','.join(per_invalid_reason_list),
                            })
                            continue

                        per_save_annotation = get_single_sample_pair_annotation(
                            per_row_dict, per_sample_key, per_subset_name,
                            per_category_id, per_parquet_group_name,
                            per_row_index, per_source_image_state,
                            per_source_image_relative_path,
                            per_edited_image_relative_path, per_instruction,
                            per_source_image_shape, per_edited_image_shape)
                        per_save_annotation_line = json.dumps(
                            per_save_annotation, ensure_ascii=False)

                        if per_source_image_state == SOURCE_IMAGE_STATE_URL_ONLY:
                            # 只有源图URL的样本对: 编辑后图已照常落盘，
                            # 全部属性(含url/sha256/fetch_date)单独落一份索引，
                            # 等后续按URL+sha256回捞源图即可直接补齐成完整编辑对
                            save_url_source_annotation_file.write(
                                f'{per_save_annotation_line}\n')
                            url_source_only_sample_pair_count += 1
                        else:
                            save_annotation_file.write(
                                f'{per_save_annotation_line}\n')
                            valid_sample_pair_count += 1

                        edit_task_count_dict[str(
                            per_row_dict.get(PARQUET_EDIT_TASK_COLUMN_NAME,
                                             None))] += 1
                        if per_edited_image_shape[0] > 0:
                            edited_image_shape_count_dict[
                                f'{per_edited_image_shape[0]}x{per_edited_image_shape[1]}'] += 1
                        score_count_dict[','.join([
                            f'{per_row_dict.get(per_column_name, None)}' for
                            per_column_name in PARQUET_SCORE_COLUMN_NAME_LIST
                        ])] += 1

        reach_parquet_end = True

        # 核心对账之一: 实际遍历到的行数必须等于footer里数出来的行数，
        # 否则说明流式读的时候有batch被静默吞掉了
        if row_count != expected_row_count:
            error_message_list.append(
                f'{per_parquet_relative_path} row count not match {row_count} != {expected_row_count}'
            )
    except Exception as e:
        # parquet损坏或NAS读失败时保留已解出的图像和标注，但必须上报，不能静默少样本
        print('7777', per_parquet_relative_path, e)
        error_message_list.append(f'read parquet failed {e}')

    if not reach_parquet_end:
        error_message_list.append(
            'not reach parquet row batch end, parquet may be truncated')

    # 核心对账之二: 每行必有一张编辑后图，参考图只有非url_only的行才有
    if edited_image_member_count != row_count:
        error_message_list.append(
            f'{per_parquet_relative_path} edited image member count not match {edited_image_member_count} != {row_count}'
        )
    if source_image_member_count + url_source_only_sample_pair_count + len(
            invalid_sample_pair_list) < row_count:
        # 参考图成员数 + url_only对数 + 隔离对数 至少要覆盖全部行，
        # 少了说明有行的参考图既没落盘也没被隔离上报
        error_message_list.append(
            f'{per_parquet_relative_path} source image member count not match {source_image_member_count} + {url_source_only_sample_pair_count} + {len(invalid_sample_pair_list)} < {row_count}'
        )

    # 核心对账之三: 每张图像成员都必须有明确归属(新写/跳过/不落盘/写失败)
    if extract_image_count + skip_image_count + not_save_image_count + save_image_fail_count != source_image_member_count + edited_image_member_count:
        error_message_list.append(
            f'{per_parquet_relative_path} process image count not match: {extract_image_count} + {skip_image_count} + {not_save_image_count} + {save_image_fail_count} != {source_image_member_count} + {edited_image_member_count}'
        )
    if save_image_fail_count > 0:
        error_message_list.append(
            f'{per_parquet_relative_path} save image fail count {save_image_fail_count}'
        )

    # 核心对账之四: 每一行都必须有归属，要么是完整编辑对、要么是只有源图URL的对、
    # 要么进隔离清单，一条都不会凭空消失
    if valid_sample_pair_count + url_source_only_sample_pair_count + len(
            invalid_sample_pair_list) != row_count:
        error_message_list.append(
            f'{per_parquet_relative_path} sample pair count not match {valid_sample_pair_count} + {url_source_only_sample_pair_count} + {len(invalid_sample_pair_list)} != {row_count}'
        )
    # 实测每行的指令与编辑后图都齐备，所以隔离清单应该恒为空，一旦非空必须显式感知
    if len(invalid_sample_pair_list) > 0:
        error_message_list.append(
            f'{per_parquet_relative_path} invalid sample pair count {len(invalid_sample_pair_list)}'
        )
    if valid_sample_pair_count != source_image_member_count:
        # 完整编辑对数必须等于参考图成员数(每个完整对恰好一张参考图)
        error_message_list.append(
            f'{per_parquet_relative_path} valid sample pair count not match source image member count {valid_sample_pair_count} != {source_image_member_count}'
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
        'url_source_only_sample_pair_count':
        url_source_only_sample_pair_count,
        'source_image_member_count':
        source_image_member_count,
        'edited_image_member_count':
        edited_image_member_count,
        'extract_image_count':
        extract_image_count,
        'skip_image_count':
        skip_image_count,
        'not_save_image_count':
        not_save_image_count,
        'save_image_fail_count':
        save_image_fail_count,
        'duplicate_sample_count':
        duplicate_sample_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_subset_name}/{per_parquet_group_name}.jsonl',
        'save_url_source_annotation_relative_path':
        f'{SAVE_URL_SOURCE_ANNOTATION_DIR_NAME}/{per_subset_name}/{per_parquet_group_name}.jsonl',
        'edit_task_count_dict':
        dict(edit_task_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'edited_image_shape_count_dict':
        dict(edited_image_shape_count_dict),
        'score_count_dict':
        dict(score_count_dict),
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
        # .cache里有548个下载缓存文件(含39个*.incomplete)，
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
    """可选的二次对账: os.walk单个分片的图像目录，核对落盘图像数与两份标注条数"""
    per_parquet_relative_path, per_image_dir_path, per_annotation_path, per_url_source_annotation_path, per_expected_valid_sample_pair_count, per_expected_url_source_only_sample_pair_count = parquet_check_pair

    error_message_list = []

    if not os.path.exists(per_image_dir_path):
        error_message_list.append(
            f'{per_parquet_relative_path} image dir not exist')

        return [per_parquet_relative_path, 0, 0, 0, error_message_list]

    image_name_dict = {}
    unknown_suffix_file_count = 0
    for per_root_path, _, per_file_name_list in os.walk(per_image_dir_path):
        for per_file_name in per_file_name_list:
            if check_image_file_suffix(per_file_name):
                image_name_dict[per_file_name] = 1
            else:
                # 图像目录里不该出现非图像文件，必须上报，不能默认当图像统计
                unknown_suffix_file_count += 1

    annotation_count, url_source_annotation_count = 0, 0
    missing_image_count = 0
    for per_load_annotation_path, per_is_url_source_annotation in [
        [per_annotation_path, False],
        [per_url_source_annotation_path, True],
    ]:
        try:
            with open(per_load_annotation_path, 'r',
                      encoding='UTF-8') as load_annotation_file:
                for per_annotation_line in load_annotation_file:
                    per_annotation_line = per_annotation_line.strip()
                    if not per_annotation_line:
                        continue

                    per_annotation = json.loads(per_annotation_line)
                    if per_is_url_source_annotation:
                        url_source_annotation_count += 1
                    else:
                        annotation_count += 1

                    per_image_path_list = list(
                        per_annotation.get('reference_image_path_list', []))
                    if per_annotation.get('edited_image_path', ''):
                        per_image_path_list.append(
                            per_annotation['edited_image_path'])

                    for per_image_path in per_image_path_list:
                        if os.path.basename(
                                per_image_path) not in image_name_dict:
                            missing_image_count += 1
        except Exception as e:
            error_message_list.append(
                f'{per_parquet_relative_path} load annotation failed {per_load_annotation_path} {e}'
            )

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_parquet_relative_path} unknown suffix file num {unknown_suffix_file_count}'
        )
    if annotation_count != per_expected_valid_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_relative_path} on disk annotation count not match {annotation_count} != {per_expected_valid_sample_pair_count}'
        )
    if url_source_annotation_count != per_expected_url_source_only_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_relative_path} on disk url source annotation count not match {url_source_annotation_count} != {per_expected_url_source_only_sample_pair_count}'
        )
    # 完整对贡献2张图(参考图+编辑后图)，只有源图URL的对只贡献1张编辑后图
    per_expected_image_count = per_expected_valid_sample_pair_count * 2 + per_expected_url_source_only_sample_pair_count
    if len(image_name_dict) != per_expected_image_count:
        error_message_list.append(
            f'{per_parquet_relative_path} on disk image count not match {len(image_name_dict)} != {per_expected_image_count}'
        )
    if missing_image_count > 0:
        error_message_list.append(
            f'{per_parquet_relative_path} on disk missing image count {missing_image_count}'
        )

    return [
        per_parquet_relative_path,
        len(image_name_dict),
        annotation_count,
        url_source_annotation_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, save_annotation_dir_path,
                             save_url_source_annotation_dir_path,
                             parquet_result_list):
    """可选的二次对账: 遍历输出目录核对每个分片的落盘图像数与两份标注条数"""
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
            os.path.join(save_url_source_annotation_dir_path,
                         per_parquet_result['subset_name'],
                         f'{per_parquet_result["parquet_name"]}.jsonl'),
            per_parquet_result['valid_sample_pair_count'],
            per_parquet_result['url_source_only_sample_pair_count'],
        ])

    error_message_list = []
    total_image_count = 0
    total_annotation_count, total_url_source_annotation_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_dir_on_disk, parquet_check_pair_list),
                                     total=len(parquet_check_pair_list)):
            (_, per_image_count, per_annotation_count,
             per_url_source_annotation_count,
             per_error_message_list) = per_check_result
            total_image_count += per_image_count
            total_annotation_count += per_annotation_count
            total_url_source_annotation_count += per_url_source_annotation_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk image:', total_image_count, 'on disk annotation:',
          total_annotation_count, 'on disk url source annotation:',
          total_url_source_annotation_count)

    return error_message_list


def save_check_result(save_dataset_path, parquet_result_list,
                      expected_total_row_count):
    """汇总所有parquet的解包与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_row_count = 0
    total_valid_sample_pair_count, total_url_source_only_sample_pair_count = 0, 0
    total_source_image_member_count, total_edited_image_member_count = 0, 0
    total_extract_image_count, total_skip_image_count = 0, 0
    total_not_save_image_count, total_save_image_fail_count = 0, 0
    total_duplicate_sample_count = 0
    subset_row_count_dict = collections.Counter()
    subset_sample_pair_count_dict = collections.Counter()
    subset_url_source_only_sample_pair_count_dict = collections.Counter()
    edit_task_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    edited_image_shape_count_dict = collections.Counter()
    score_count_dict = collections.Counter()
    parquet_sample_pair_count_dict = {}
    all_invalid_sample_pair_list, all_warning_message_list = [], []
    error_message_list = []

    for per_parquet_result in parquet_result_list:
        per_parquet_relative_path = per_parquet_result['parquet_relative_path']
        per_subset_name = per_parquet_result['subset_name']

        total_row_count += per_parquet_result['row_count']
        total_valid_sample_pair_count += per_parquet_result[
            'valid_sample_pair_count']
        total_url_source_only_sample_pair_count += per_parquet_result[
            'url_source_only_sample_pair_count']
        total_source_image_member_count += per_parquet_result[
            'source_image_member_count']
        total_edited_image_member_count += per_parquet_result[
            'edited_image_member_count']
        total_extract_image_count += per_parquet_result['extract_image_count']
        total_skip_image_count += per_parquet_result['skip_image_count']
        total_not_save_image_count += per_parquet_result[
            'not_save_image_count']
        total_save_image_fail_count += per_parquet_result[
            'save_image_fail_count']
        total_duplicate_sample_count += per_parquet_result[
            'duplicate_sample_count']

        subset_row_count_dict[per_subset_name] += per_parquet_result[
            'row_count']
        subset_sample_pair_count_dict[per_subset_name] += per_parquet_result[
            'valid_sample_pair_count']
        subset_url_source_only_sample_pair_count_dict[
            per_subset_name] += per_parquet_result[
                'url_source_only_sample_pair_count']
        edit_task_count_dict.update(per_parquet_result['edit_task_count_dict'])
        image_suffix_count_dict.update(
            per_parquet_result['image_suffix_count_dict'])
        edited_image_shape_count_dict.update(
            per_parquet_result['edited_image_shape_count_dict'])
        score_count_dict.update(per_parquet_result['score_count_dict'])
        parquet_sample_pair_count_dict[per_parquet_relative_path] = [
            per_parquet_result['valid_sample_pair_count'],
            per_parquet_result['url_source_only_sample_pair_count'],
        ]

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
          total_valid_sample_pair_count, 'total url source only sample pair:',
          total_url_source_only_sample_pair_count, 'source image member:',
          total_source_image_member_count, 'edited image member:',
          total_edited_image_member_count, 'extract image:',
          total_extract_image_count, 'skip image:', total_skip_image_count,
          'not save image:', total_not_save_image_count, 'save image fail:',
          total_save_image_fail_count, 'duplicate sample:',
          total_duplicate_sample_count, 'invalid sample pair:',
          len(all_invalid_sample_pair_list), 'warning:',
          len(all_warning_message_list))
    print('3333', 'subset sample pair:', dict(subset_sample_pair_count_dict))
    print('3333', 'subset url source only sample pair:',
          dict(subset_url_source_only_sample_pair_count_dict))
    print('3333', 'edit task:', dict(edit_task_count_dict))
    print('3333', 'image suffix:', dict(image_suffix_count_dict))
    print('3333', 'score(IF,EC,GQ):', dict(score_count_dict))
    print('3333', 'edited image shape top10:',
          dict(edited_image_shape_count_dict.most_common(10)))

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
        'total_url_source_only_sample_pair_count':
        total_url_source_only_sample_pair_count,
        'total_source_image_member_count':
        total_source_image_member_count,
        'total_edited_image_member_count':
        total_edited_image_member_count,
        'total_extract_image_count':
        total_extract_image_count,
        'total_skip_image_count':
        total_skip_image_count,
        'total_not_save_image_count':
        total_not_save_image_count,
        'total_save_image_fail_count':
        total_save_image_fail_count,
        'total_duplicate_sample_count':
        total_duplicate_sample_count,
        'invalid_sample_pair_count':
        len(all_invalid_sample_pair_list),
        'warning_message_count':
        len(all_warning_message_list),
        'subset_row_count_dict':
        dict(subset_row_count_dict),
        'subset_sample_pair_count_dict':
        dict(subset_sample_pair_count_dict),
        'subset_url_source_only_sample_pair_count_dict':
        dict(subset_url_source_only_sample_pair_count_dict),
        'edit_task_count_dict':
        dict(edit_task_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'score_count_dict':
        dict(score_count_dict),
        'edited_image_shape_count_dict':
        dict(edited_image_shape_count_dict),
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

    # 全量硬对账: 每一行都必须有归属，完整编辑对数与只有源图URL的对数都必须
    # 等于预检时从footer数出来的实测ground truth，
    # 少一个都说明有样本对在"读parquet->写图->写jsonl"这条链路上消失了
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
    if total_url_source_only_sample_pair_count != EXPECTED_TOTAL_URL_SOURCE_ONLY_ROW_COUNT:
        error_message_list.append(
            f'total url source only sample pair count not match {total_url_source_only_sample_pair_count} != {EXPECTED_TOTAL_URL_SOURCE_ONLY_ROW_COUNT}'
        )
    if total_valid_sample_pair_count + total_url_source_only_sample_pair_count + len(
            all_invalid_sample_pair_list) != total_row_count:
        error_message_list.append(
            f'total sample pair count not match {total_valid_sample_pair_count} + {total_url_source_only_sample_pair_count} + {len(all_invalid_sample_pair_list)} != {total_row_count}'
        )
    if total_edited_image_member_count != total_row_count:
        error_message_list.append(
            f'total edited image member count not match {total_edited_image_member_count} != {total_row_count}'
        )
    if total_source_image_member_count != EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'total source image member count not match {total_source_image_member_count} != {EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT}'
        )
    if total_extract_image_count + total_skip_image_count + total_not_save_image_count + total_save_image_fail_count != total_source_image_member_count + total_edited_image_member_count:
        error_message_list.append(
            f'total process image count not match {total_extract_image_count} + {total_skip_image_count} + {total_not_save_image_count} + {total_save_image_fail_count} != {total_source_image_member_count} + {total_edited_image_member_count}'
        )
    if total_save_image_fail_count > 0:
        error_message_list.append(
            f'total save image fail count {total_save_image_fail_count}')
    if len(all_invalid_sample_pair_list) > 0:
        error_message_list.append(
            f'invalid sample pair count {len(all_invalid_sample_pair_list)}')
    if total_duplicate_sample_count > 0:
        error_message_list.append(
            f'duplicate sample count {total_duplicate_sample_count}')

    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        per_expected_row_count = EXPECTED_SUBSET_ROW_COUNT_DICT[
            per_subset_name]
        per_expected_url_source_only_sample_pair_count = EXPECTED_SUBSET_URL_SOURCE_ONLY_ROW_COUNT_DICT[
            per_subset_name]
        per_expected_valid_sample_pair_count = per_expected_row_count - per_expected_url_source_only_sample_pair_count

        per_row_count = subset_row_count_dict.get(per_subset_name, 0)
        per_valid_sample_pair_count = subset_sample_pair_count_dict.get(
            per_subset_name, 0)
        per_url_source_only_sample_pair_count = subset_url_source_only_sample_pair_count_dict.get(
            per_subset_name, 0)

        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_subset_name} row count not match {per_row_count} != {per_expected_row_count}'
            )
        if per_valid_sample_pair_count != per_expected_valid_sample_pair_count:
            error_message_list.append(
                f'{per_subset_name} valid sample pair count not match {per_valid_sample_pair_count} != {per_expected_valid_sample_pair_count}'
            )
        if per_url_source_only_sample_pair_count != per_expected_url_source_only_sample_pair_count:
            error_message_list.append(
                f'{per_subset_name} url source only sample pair count not match {per_url_source_only_sample_pair_count} != {per_expected_url_source_only_sample_pair_count}'
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

    expected_parquet_group_num = sum(EXPECTED_SUBSET_PARQUET_NUM_DICT.values())
    if len(parquet_group_list) != expected_parquet_group_num:
        raise Exception(
            f'parquet group num not match {len(parquet_group_list)} != {expected_parquet_group_num}'
        )

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    save_url_source_annotation_dir_path = os.path.join(
        save_dataset_path, SAVE_URL_SOURCE_ANNOTATION_DIR_NAME)
    os.makedirs(save_url_source_annotation_dir_path, exist_ok=True)

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
    extract_func = partial(
        process_single_parquet_file,
        save_dataset_path=save_dataset_path,
        save_annotation_dir_path=save_annotation_dir_path,
        save_url_source_annotation_dir_path=save_url_source_annotation_dir_path
    )
    with Pool(processes=PROCESS_NUM) as pool:
        for per_parquet_result in tqdm(pool.imap_unordered(
                extract_func, parquet_group_list),
                                       total=len(parquet_group_list)):
            parquet_result_list.append(per_parquet_result)

            print('2222', per_parquet_result['parquet_relative_path'], 'row:',
                  per_parquet_result['row_count'], 'valid sample pair:',
                  per_parquet_result['valid_sample_pair_count'],
                  'url source only sample pair:',
                  per_parquet_result['url_source_only_sample_pair_count'],
                  'extract image:', per_parquet_result['extract_image_count'],
                  'skip image:', per_parquet_result['skip_image_count'],
                  'not save image:',
                  per_parquet_result['not_save_image_count'],
                  'save image fail:',
                  per_parquet_result['save_image_fail_count'],
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
            save_dataset_path, save_annotation_dir_path,
            save_url_source_annotation_dir_path, parquet_result_list)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解包/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/ScaleEdit-12M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
