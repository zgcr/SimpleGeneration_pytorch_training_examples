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
# 数据集: UnicEdit-10M(xiaotanhua/UnicEdit-10M, 论文CVPR2026 arXiv:2512.02790)
#
# 【数据集类型】纯图像编辑(instruction-based image editing)数据集，不是文生图数据集。
# README原文 task_categories: image-to-image，"This is a large-scale image editing dataset",
# 每行固定是"1张参考图(编辑前图src_image) + 1条编辑指令(中英双语prompt_cn/prompt_en) +
# 1张编辑后图(edit_image)"，没有第二张视觉条件图、没有mask、
# 也没有任何"只有caption+一张图"的纯生成样本，
# 所以下游只能走ti2i_dataset.py那条链路，不能当t2i(文生图)数据用。
# 抽样1040对逐行解码验证: src_image与edit_image的字节md5**没有一对相同**,
# 确实是真实编辑对而不是同图复制; 同一对的输入输出分辨率实测恒相同。
#
# 【root_dataset_path实测原始保存规格(共5.2T / 135个parquet / 2025000行)】
# UnicEdit-10M/
# ├── data/             135个 train-%05d-of-00681.parquet  (5.2T)  -> 唯一有用数据
# ├── README.md         数据集说明(无用，授权信息已记进校验报告的dataset_license_name字段)
# ├── .gitattributes    git lfs配置(无用)
# └── .cache/           huggingface下载缓存(无用，残留0个*.incomplete)
#
# 【必须显式感知的四个规格坑】
# 1) **文件名声称681片，上游仓库实际只上传了135片**(编号0..240且严重不连续):
#    在片  0-25,30-35,37-60,62-67,69-70,75-113,115-127,129-140,160-162,227-229,240
#    缺片  26-29,36,61,68,71-74,114,128,141-159,163-226,230-239,241-680(共546个编号)
#    已三方对账证实这是**上游仓库自身没上传**、不是本地下载缺失:
#      a. huggingface下载缓存里的仓库文件清单 .cache/huggingface/trees/*.json 共137个条目
#         (135个parquet + README.md + .gitattributes)，磁盘上**全部存在且字节数100%一致**;
#      b. .cache/huggingface/download/data/ 下正好135个*.metadata，**0个*.incomplete**;
#      c. 线上API /api/datasets/xiaotanhua/UnicEdit-10M/tree/main/data?recursive=1
#         返回的也正好是这135个文件。
#    所以按"root_dataset_path已是完整数据集"的前提，把这135个编号写死成
#    EXPECTED_PARQUET_INDEX_LIST 当ground truth，**多一个少一个都硬失败中止**,
#    绝不允许"少了几片但整体报成功"。
#    实际总行数 **2025000**(README标题说10M，是按上游完整681片规模宣称的，不是本地实际值)。
# 2) **key(32位小写hex md5)不是全局唯一**: 2025000行里有 **5181个key各出现2次**(共10362行),
#    其中1040对甚至落在**同一个parquet内部**(其余4141对跨parquet)。
#    逐行核对过重复key的内容: 两行的src_image字节相同、但prompt与edit_image不同
#    (同一张源图的多条不同编辑)，**两行都是完整有效的独立编辑样本对，一条都不能丢**。
#    ⇒ **绝不能用key当落盘文件名**，否则这5181对会互相覆盖、静默丢掉5181个样本对。
#    落盘key改由 <parquet名>_<片内行号> 合成(parquet名全局唯一 + 片内行号唯一
#    => 合成key全局唯一，天然不会重名，不需要重名隔离目录),
#    原始md5照样写进jsonl的md5_key字段，溯源不丢，并在校验报告里统计重复key数。
# 3) **edit_subtask有1行脏数据**: train-00067-of-00681 第7069行(key=5fc645e8f5a676b5e1c3d934c0e7b92f)
#    的edit_subtask被误填成一整句中文prompt("修复精灵球的破裂部分，并在其旁边添加一个小型的、漂浮的精灵球。")。
#    该行的key/edit_task/两张图/中英指令全部齐备，**是完整有用样本对，绝不丢弃**:
#    落盘时归到 <edit_task>/unknown_subtask/ 兜底目录(避免中文长句变成目录名),
#    原始脏值完整写进jsonl的edit_subtask字段并进warning清单上报。
# 4) src_image.path / edit_image.path 两个叶子列**全库2025000行恒为null**(恒空占位列)
#    -> 无用，不落盘也不进标注。
#
# 【单行parquet的全部7列/9个叶子列(135片schema完全一致, SNAPPY, parquet-cpp-arrow 20.0.0)】
# 每片恒为15000行 / 150个row group × 100行(**135/135片完全一致**，没有尾片不足的情况)
#   key          : 32位小写hex md5，**全库null=0、无空串**，但有5181个重复(见坑2)
#                                                             -> 有用(溯源/去重，不可当文件名)
#   edit_task    : 4类主编辑类别，全库null=0                    -> 有用(粗粒度任务族)
#   edit_subtask : 20类子编辑类别 + 1行脏值(见坑3)，全库null=0   -> 有用(细粒度任务族，
#                  分布跨4个数量级，长尾任务靠它才能做过采样/限流)
#   src_image    : struct<bytes: binary, path: string>
#                  .bytes = **参考图(编辑前图)原始字节，全库null=0**   -> 有用(参考图，必需)
#                  .path  = **全库2025000行全部为null**，恒空占位列    -> 无用
#   edit_image   : struct<bytes: binary, path: string>
#                  .bytes = **编辑后图原始字节，全库null=0**           -> 有用(编辑后图，必需)
#                  .path  = 同上，恒空                                 -> 无用
#   prompt_cn    : 中文编辑指令，**全库null=0、无空串**，长10~154       -> 有用(中文训练文本)
#   prompt_en    : 英文编辑指令，**全库null=0、无空串**，长30~542       -> 有用(英文训练主文本)
# 除这7列外该数据集**没有任何其他单样本属性**(无宽高/无打分/无mask/无授权列),
# 所以宽高只能在解包时顺手从图像header里读出来存进jsonl(见PARSE_IMAGE_SHAPE_FLAG)。
# 文本列实测**没有任何前后空白**(strip前后完全一致)。
#
# 【edit_task -> edit_subtask 实测是严格的1:N嵌套(21个组合，每个subtask只归属唯一一个task)】
#   Attribute Editing (676977): Color Alteration 520303 / Material Modification 86647 /
#                               Texture Editing 46322 / Shape-Size Alteration 19254 /
#                               Motion Change 4451
#   Scene Editing     (548362): Background Change 528872 / Tone Transformation 14856 /
#                               Style Transfer 4473 / Viewpoint Transformation 161
#   Object Editing    (518729): Subject Addition 471956 / Portrait Editing 32763 /
#                               Subject Replacement 11686 / Text Modification 1673 /
#                               Subject Removal 375 / Object Extraction 151 /
#                               Counting Change 125
#   Reasoning Editing (280932): Spatial Reasoning Edits 206924 /
#                               Multi-object Coordination 53516 /
#                               Compound Operation Edits 19698 / Relation Change 793 /
#                               **1行脏值(见坑3)**
# 所以落盘按 <edit_task>/<edit_subtask>/<parquet>/ 两级分层，粗细两档粒度都能直接按目录用,
# 不会有归属歧义。
#
# 【图像格式实测(抽样1040行逐行解码 + 3片各300行)】
# src_image与edit_image**全部是PNG、全部RGB三通道**，
# 分辨率**全部是16的倍数**(1024x1024最多，其次832x1248 / 1184x880 / 752x1392 / 688x1504等),
# 同一对的输入输出分辨率恒相同。
# 落盘后缀仍然按字节魔数逐张判定，不写死.png(上游哪天换成JPEG也不会存错后缀)。
#
# 【无用信息(一律不整理进训练目录)】
# .cache/(huggingface下载缓存) / .gitattributes / README.md /
# src_image.path与edit_image.path(全null的占位列) /
# .gitignore / .DS_Store / CACHEDIR.TAG 这类目录元数据垃圾文件。
#
# 【本脚本的处理口径】
# - 解包前三段预检(硬失败，不过就不白跑几十小时):
#   a. 根目录条目白名单(只允许data/，多出未知文件或未知目录立即上报)、
#      data/下必须全部是 train-%05d-of-00681.parquet 且"of"总片数恒为681、
#      **分片编号集合必须严格等于写死的135个编号**(缺号与多号都硬失败)、
#      每片PAR1头尾魔数(O(1)读，拦下载截断);
#   b. 只读footer校验: 7列/9个叶子列schema逐片一致、每片恒15000行、
#      key/edit_task/edit_subtask/两个bytes/两个prompt 七个叶子列null数必须为0、
#      两个path列必须恒null(变了只告警)、总行数硬对账2025000;
#   c. **只读5个小文本列**(parquet列裁剪，完全不碰图像字节，实测单片1.3秒)硬对账:
#      逐edit_task行数、逐edit_subtask行数、(task,subtask,parquet)组合数=2556、
#      prompt_en/prompt_cn空串数必须为0、key空串数必须为0,
#      并统计全库重复md5 key(实测5181个，只告警不判失败);
# - 并行单位 = 单个parquet(135个任务，Pool(32)),
#   pq.iter_batches(batch_size=32)流式读，**绝不整片进内存**(单片约42GB);
# - 图像落盘 unzip_images/<edit_task>/<edit_subtask>/<parquet>/<parquet>_%08d_src.png 与 _edit.png,
#   **直接写原始字节，unzip阶段绝不引入二次编解码**,
#   resize/转格式/分辨率分桶留给preprocessing2的resave脚本;
# - 每张图写盘后**立刻校验落盘大小 == len(bytes)**(比只看存在性强，能挡住写半截/写0字节);
#   已存在且大小一致就计skip并跳过，保证脚本可以断点续跑;
# - 汇总标注落 unzip_annotations/<edit_task>/<edit_subtask>/<parquet>.jsonl，
#   每行一个完整编辑样本对，**中英双语指令都完整保留**,
#   并保留md5_key/edit_task/edit_subtask全部有用属性,
#   再补上落盘路径、样本key、片内行号、真实宽高与后缀(字段名与016脚本对齐，
#   preprocessing2可直接复用同一套读取口径);
#   一个parquet的15000行是混合任务的，jsonl按(task,subtask)在内存里攒好
#   (15000行约15MB，可忽略)再逐个落盘，避免同时开21个文件句柄交叉写;
# - 片内五方硬对账: 遍历行数 == footer num_rows、
#   参考图成员数 == 编辑后图成员数 == 行数、
#   extract+skip+not_save+fail == 两类图像成员数之和、
#   有效样本对 + 隔离样本对 == 行数、
#   落盘jsonl总行数 == 有效样本对数;
# - 绝不静默丢样本对: 中英指令同时为空/参考图字节为空/编辑后图字节为空/写盘失败,
#   全部分门别类记进隔离清单并在汇总报告里上报(实测应恒为空，一旦非空必须显式感知);
#   edit_subtask脏值、重复md5 key、单语指令缺失只进warning清单,
#   **样本本身照常完整保存，绝不因为这些非致命异常丢样本对**;
# - 全局硬对账: 总行数 == 2025000、有效样本对 == 2025000、图像成员总数 == 4050000、
#   逐edit_task行数与实测值逐一比对、逐edit_subtask行数与实测值逐一比对、
#   落盘jsonl文件数 == 2556、jsonl总行数 == 2025000，少一条立即抛异常;
# - 拷贝/解包/校验任一环出错都汇总后抛异常，不再静默跑过。
#
# 【跑之前务必确认目标盘扛得住】
# - EXTRACT_IMAGE_FILE_FLAG=True 时输出小文件数 **4050000张图**(约5.2T),
# - 只想先建索引可把 EXTRACT_IMAGE_FILE_FLAG 置False,
#   图像继续留在原parquet里，样本对信息一样完整。
# ==============================================================================

DATASET_TASK_TYPE = 'image_edit'

DATASET_LICENSE_NAME = 'apache-2.0'

PARQUET_FILE_NAME_PATTERN = re.compile(r'^(?P<prefix>.+)\.parquet$')

# 带分片编号的parquet名(train-00000-of-00681.parquet)，用于分片完整性预检
PARQUET_SHARD_FILE_NAME_PATTERN = re.compile(
    r'^(?P<prefix>(?P<split>train)-(?P<index>\d{5})-of-(?P<total>\d{5}))\.parquet$'
)

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(0个*.incomplete)
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

PARQUET_SHARD_SPLIT_NAME = 'train'

# parquet文件名里"-of-"后面声称的总片数，恒为681(上游只上传了其中135片，见坑1)
EXPECTED_PARQUET_SHARD_TOTAL_NUM = 681

# **上游仓库实际上传的135个分片编号**(已用仓库文件清单 + 下载缓存 + 线上tree API三方对账),
# 编号0..240且严重不连续，缺的546个编号是上游自身没上传、不是本地下载缺失。
# 磁盘上的编号集合必须**严格等于**这个列表，缺号与多号都硬失败中止,
# 否则就是静默少几万个样本对
EXPECTED_PARQUET_INDEX_LIST = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    30,
    31,
    32,
    33,
    34,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    69,
    70,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    160,
    161,
    162,
    227,
    228,
    229,
    240,
]

EXPECTED_TOTAL_PARQUET_NUM = 135

# 实测每片恒为15000行(135/135片完全一致，没有尾片不足的情况)，
# 所以这里可以当硬性条件用，行数不对说明该片没下全
EXPECTED_PARQUET_ROW_COUNT = 15000

# 实测每片恒为150个row group、每组恒100行(单片约42GB，必须iter_batches流式读)
EXPECTED_PARQUET_ROW_GROUP_NUM = 150

EXPECTED_PARQUET_ROW_GROUP_ROW_COUNT = 100

EXPECTED_TOTAL_ROW_COUNT = 2025000

# 实测每行的两张图/两条指令/key/两个任务名全部齐备(footer里七个叶子列null数全为0),
# 所以完整编辑样本对数 == 总行数，少一个都说明链路上丢了样本
EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT = 2025000

# 每个完整编辑对贡献2张图(参考图 + 编辑后图)
EXPECTED_TOTAL_IMAGE_COUNT = 4050000

# 实测(edit_task, edit_subtask, parquet)组合数，即落盘jsonl文件数:
# 15个subtask在135片里都出现 + Relation Change 134片 + Subject Removal 128片 +
# Viewpoint Transformation 96片 + Object Extraction 92片 + Counting Change 80片 +
# 1片里的1行subtask脏值(归到unknown_subtask) = 2025 + 531 = 2556
EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 2556

# parquet里应该齐备的全部7个顶层列名，少列/多列说明上游数据规格变了，必须显式感知
PARQUET_COLUMN_NAME_LIST = [
    'key',
    'edit_task',
    'edit_subtask',
    'src_image',
    'edit_image',
    'prompt_cn',
    'prompt_en',
]

# footer统计量是按叶子列组织的(struct会被展平成 <列名>.<字段名>)
PARQUET_LEAF_COLUMN_NAME_LIST = [
    'key',
    'edit_task',
    'edit_subtask',
    'src_image.bytes',
    'src_image.path',
    'edit_image.bytes',
    'edit_image.path',
    'prompt_cn',
    'prompt_en',
]

# 这七个叶子列的null数实测全库为0，非0就说明存在不完整样本对，必须硬失败
PARQUET_NOT_NULL_LEAF_COLUMN_NAME_LIST = [
    'key',
    'edit_task',
    'edit_subtask',
    'src_image.bytes',
    'edit_image.bytes',
    'prompt_cn',
    'prompt_en',
]

# 这两个叶子列实测全库恒为null(恒空占位列)，不落盘也不进标注;
# 万一上游哪天填了值，只打印告警提醒，不判失败
PARQUET_ALL_NULL_LEAF_COLUMN_NAME_LIST = [
    'src_image.path',
    'edit_image.path',
]

# 预检时只读这5个小文本列做全量对账(parquet列裁剪，完全不碰图像字节，实测单片1.3秒)
PARQUET_TEXT_COLUMN_NAME_LIST = [
    'key',
    'edit_task',
    'edit_subtask',
    'prompt_cn',
    'prompt_en',
]

# 样本唯一id(32位小写hex md5)，**全库有5181个重复，不可当落盘文件名**(见坑2)
PARQUET_MD5_KEY_COLUMN_NAME = 'key'

MD5_KEY_LENGTH = 32

MD5_KEY_CHAR_NAME_LIST = '0123456789abcdef'

# 参考图(编辑前图)与编辑后图的struct列，这两列是唯一需要落盘成图像文件的列
PARQUET_SRC_IMAGE_COLUMN_NAME = 'src_image'

PARQUET_EDIT_IMAGE_COLUMN_NAME = 'edit_image'

PARQUET_IMAGE_STRUCT_BYTES_KEY_NAME = 'bytes'

# 训练主文本 = prompt_en(英文编辑指令)，prompt_cn(中文编辑指令)同样完整落进jsonl;
# **两条指令同时为空**才算该样本对没有文本条件、不可训练，隔离上报
PARQUET_PROMPT_EN_COLUMN_NAME = 'prompt_en'

PARQUET_PROMPT_CN_COLUMN_NAME = 'prompt_cn'

PARQUET_EDIT_TASK_COLUMN_NAME = 'edit_task'

PARQUET_EDIT_SUBTASK_COLUMN_NAME = 'edit_subtask'

# 4个主编辑类别(粗粒度任务族)
EDIT_TASK_NAME_LIST = [
    'Attribute Editing',
    'Object Editing',
    'Reasoning Editing',
    'Scene Editing',
]

# 20个子编辑类别(细粒度任务族)，落盘目录用的就是这一档粒度
EDIT_SUBTASK_NAME_LIST = [
    'Background Change',
    'Color Alteration',
    'Compound Operation Edits',
    'Counting Change',
    'Material Modification',
    'Motion Change',
    'Multi-object Coordination',
    'Object Extraction',
    'Portrait Editing',
    'Relation Change',
    'Shape-Size Alteration',
    'Spatial Reasoning Edits',
    'Style Transfer',
    'Subject Addition',
    'Subject Removal',
    'Subject Replacement',
    'Text Modification',
    'Texture Editing',
    'Tone Transformation',
    'Viewpoint Transformation',
]

# 实测 edit_task -> edit_subtask 是严格的1:N嵌套(每个subtask只归属唯一一个task),
# 嵌套关系被破坏说明上游标注错位，必须显式感知(只告警，不丢样本)
EDIT_TASK_SUBTASK_NAME_DICT = {
    'Attribute Editing': [
        'Color Alteration',
        'Material Modification',
        'Motion Change',
        'Shape-Size Alteration',
        'Texture Editing',
    ],
    'Object Editing': [
        'Counting Change',
        'Object Extraction',
        'Portrait Editing',
        'Subject Addition',
        'Subject Removal',
        'Subject Replacement',
        'Text Modification',
    ],
    'Reasoning Editing': [
        'Compound Operation Edits',
        'Multi-object Coordination',
        'Relation Change',
        'Spatial Reasoning Edits',
    ],
    'Scene Editing': [
        'Background Change',
        'Style Transfer',
        'Tone Transformation',
        'Viewpoint Transformation',
    ],
}

# 实测每个主编辑类别的行数(合计2025000)，直接当作完整性ground truth
EXPECTED_EDIT_TASK_ROW_COUNT_DICT = {
    'Attribute Editing': 676977,
    'Object Editing': 518729,
    'Reasoning Editing': 280932,
    'Scene Editing': 548362,
}

# 实测每个子编辑类别的行数(合计2024999，剩下1行是subtask脏值，见坑3),
# 分布跨4个数量级(528872 ~ 125)，长尾任务靠这一档粒度才能做过采样/限流
EXPECTED_EDIT_SUBTASK_ROW_COUNT_DICT = {
    'Background Change': 528872,
    'Color Alteration': 520303,
    'Compound Operation Edits': 19698,
    'Counting Change': 125,
    'Material Modification': 86647,
    'Motion Change': 4451,
    'Multi-object Coordination': 53516,
    'Object Extraction': 151,
    'Portrait Editing': 32763,
    'Relation Change': 793,
    'Shape-Size Alteration': 19254,
    'Spatial Reasoning Edits': 206924,
    'Style Transfer': 4473,
    'Subject Addition': 471956,
    'Subject Removal': 375,
    'Subject Replacement': 11686,
    'Text Modification': 1673,
    'Texture Editing': 46322,
    'Tone Transformation': 14856,
    'Viewpoint Transformation': 161,
}

# edit_task取值不在白名单时的兜底目录名(实测恒不会走到，出现即告警)
UNKNOWN_EDIT_TASK_DIR_NAME = 'unknown_task'

# edit_subtask取值不在白名单时的兜底目录名。
# 实测恰好有1行走这里(train-00067-of-00681第7069行的subtask被误填成一整句中文prompt),
# **该行其余字段齐备，是完整有用样本对，照常完整落盘，只是目录名用兜底名**
UNKNOWN_EDIT_SUBTASK_DIR_NAME = 'unknown_subtask'

EXPECTED_UNKNOWN_EDIT_TASK_ROW_COUNT = 0

EXPECTED_UNKNOWN_EDIT_SUBTASK_ROW_COUNT = 1

# 实测全库有5181个md5 key各出现2次(共10362行)，两行都是完整有效的独立编辑样本对。
# 这里统计的是"重复多出来的行数"(5181 = 10362 - 2019819个唯一key里重复key占的5181个位置):
#   片内重复贡献1040、跨片重复贡献4141。
# 因为落盘文件名不用md5 key(改用<parquet>_<行号>合成)，所以重复key不会导致覆盖丢样本,
# 这里只做软校验(打印告警)，数量变了说明上游数据换版了
EXPECTED_DUPLICATE_MD5_KEY_COUNT = 5181

EXPECTED_IN_FILE_DUPLICATE_MD5_KEY_COUNT = 1040

EXPECTED_CROSS_FILE_DUPLICATE_MD5_KEY_COUNT = 4141

EXPECTED_UNIQUE_MD5_KEY_COUNT = 2019819

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_SRC_IMAGE_NAME_SUFFIX = '_src'

SAVE_EDIT_IMAGE_NAME_SUFFIX = '_edit'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

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

# 实测两类图像全部是PNG，出现别的格式只打印告警(后缀按魔数逐张判定，不会存错)
EXPECTED_IMAGE_FILE_SUFFIX = '.png'

PARQUET_FILE_MAGIC_BYTES = b'PAR1'

# 图像成员是否落盘。
# True : 和其他数据集脚本口径一致，4050000张图(约5.2T),
#        NAS上inode和元数据压力大，务必确认目标盘扛得住再跑;
# False: 只解析非图像列生成 unzip_annotations/*.jsonl 索引(几分钟即可跑完),
#        图像继续留在原parquet里，训练时按parquet顺序读，样本对信息一样是完整的。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解包后再os.walk一遍输出目录做二次对账。
# 默认False: 405万个小文件的os.walk在NAS上要跑非常久，而解包时已经做了
# "写盘后立刻校验落盘大小 == len(bytes)" + "两类图像成员数与行数五方对账"两道对账,
# 已经能保证每一行的图都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 是否解码图像header拿真实宽高写进标注。
# 默认True: 只解header不解像素(PIL的Image.open是惰性的)，代价可忽略;
# 该数据集parquet里**没有任何宽高列**，不解header下游就只能在训练时逐张打开图才能分桶,
# 所以这里顺手把宽高存进jsonl，省掉后续一次405万张图的全量扫盘。
PARSE_IMAGE_SHAPE_FLAG = True

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

# 单片约42GB且每个row group有100行，必须小batch流式读，绝不整片进内存
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
    实测两类图像全部是PNG，但仍逐张按魔数判定，上游哪天换成JPEG也不会存错后缀。
    """
    for per_magic_bytes, per_magic_suffix in IMAGE_BYTES_MAGIC_SUFFIX_LIST:
        if per_image_bytes.startswith(per_magic_bytes):
            return per_magic_suffix

    return EXPECTED_IMAGE_FILE_SUFFIX


def get_stripped_text_value(per_column_value):
    """文本列统一转成strip后的字符串，None/空白都当成缺失"""
    if not isinstance(per_column_value, str):
        return ''

    return per_column_value.strip()


def get_normalized_name(per_name):
    """把任务族名里的空格归一成下划线，用作落盘目录名

    Attribute Editing -> Attribute_Editing, Shape-Size Alteration -> Shape-Size_Alteration。
    空格进目录名会让后续shell/训练脚本各种踩坑，
    原始edit_task/edit_subtask原值照样写进jsonl，溯源不丢。
    """
    per_name = per_name.replace('\\', '/').replace('/', '_')

    return '_'.join(per_name.split())


def get_md5_key_error_message(per_md5_key):
    """校验md5 key规格(32位小写hex)，不合规只上报不丢样本(key不参与落盘命名)"""
    if not per_md5_key:
        return 'empty md5 key'

    if len(per_md5_key) != MD5_KEY_LENGTH:
        return f'md5 key length not match {per_md5_key}'

    for per_md5_key_char_name in per_md5_key:
        if per_md5_key_char_name not in MD5_KEY_CHAR_NAME_LIST:
            return f'md5 key not lower hex {per_md5_key}'

    return ''


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

    实测135个parquet的字节数与仓库清单100%一致、头尾魔数全部正常，说明当前数据集是完整的。
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
    """只读单个parquet的footer拿行数/列名/各叶子列null数，不碰任何图像字节

    实测每片恒150个row group、每组恒100行，这里按全部row group累加，规格变了也不会算错。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    per_metadata_dict = {
        'parquet_group_name': per_parquet_group_name,
        'parquet_relative_dir': per_parquet_relative_dir,
        'row_count': 0,
        'row_group_num': 0,
        'row_group_row_count_list': [],
        'column_name_list': [],
        'leaf_column_name_list': [],
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

    per_row_group_row_count_list = []
    per_leaf_column_null_count_dict = collections.Counter()
    for per_row_group_index in range(per_parquet_metadata.num_row_groups):
        per_row_group_metadata = per_parquet_metadata.row_group(
            per_row_group_index)
        per_row_group_row_count_list.append(per_row_group_metadata.num_rows)

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

    per_metadata_dict[
        'row_group_row_count_list'] = per_row_group_row_count_list
    per_metadata_dict['leaf_column_null_count_dict'] = dict(
        per_leaf_column_null_count_dict)

    return per_metadata_dict, error_message_list


def check_single_parquet_text_column(parquet_group):
    """只读单个parquet的5个小文本列做全量对账，完全不碰图像字节

    parquet是列式存储，这里只读key/edit_task/edit_subtask/prompt_cn/prompt_en,
    单片42GB实测只要1.3秒，所以可以在跑几十小时解包前先把
    "逐任务族行数 / 空指令数 / 重复md5 key / (task,subtask,parquet)组合数"全部对账掉。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    per_text_column_dict = {
        'parquet_group_name': per_parquet_group_name,
        'row_count': 0,
        'md5_key_list': [],
        'edit_task_count_dict': {},
        'edit_subtask_count_dict': {},
        'edit_task_subtask_name_list': [],
        'unknown_edit_task_name_list': [],
        'unknown_edit_subtask_name_list': [],
        'not_nested_edit_task_subtask_name_list': [],
        'invalid_md5_key_message_list': [],
        'empty_prompt_en_count': 0,
        'empty_prompt_cn_count': 0,
        'empty_prompt_count': 0,
        'in_file_duplicate_md5_key_count': 0,
    }

    try:
        load_parquet_table = pq.read_table(
            per_parquet_path, columns=PARQUET_TEXT_COLUMN_NAME_LIST)
        per_md5_key_list = load_parquet_table.column(
            PARQUET_MD5_KEY_COLUMN_NAME).to_pylist()
        per_edit_task_name_list = load_parquet_table.column(
            PARQUET_EDIT_TASK_COLUMN_NAME).to_pylist()
        per_edit_subtask_name_list = load_parquet_table.column(
            PARQUET_EDIT_SUBTASK_COLUMN_NAME).to_pylist()
        per_prompt_en_list = load_parquet_table.column(
            PARQUET_PROMPT_EN_COLUMN_NAME).to_pylist()
        per_prompt_cn_list = load_parquet_table.column(
            PARQUET_PROMPT_CN_COLUMN_NAME).to_pylist()
    except Exception as e:
        print('7777', per_parquet_group_name, e)

        return per_text_column_dict, [
            f'read parquet text column failed {per_parquet_group_name} {e}',
        ]

    error_message_list = []

    per_row_count = len(per_md5_key_list)
    if not (len(per_edit_task_name_list) == per_row_count
            and len(per_edit_subtask_name_list) == per_row_count
            and len(per_prompt_en_list) == per_row_count
            and len(per_prompt_cn_list) == per_row_count):
        # 各列长度不一致说明读出来的列已经错位了，后面所有对账都不可信
        error_message_list.append(
            f'{per_parquet_group_name} text column row count not match')

        return per_text_column_dict, error_message_list

    per_md5_key_count_dict = collections.Counter()
    per_edit_task_count_dict = collections.Counter()
    per_edit_subtask_count_dict = collections.Counter()
    per_edit_task_subtask_name_dict = {}
    per_unknown_edit_task_name_dict, per_unknown_edit_subtask_name_dict = {}, {}
    per_not_nested_edit_task_subtask_name_dict = {}
    per_invalid_md5_key_message_list = []
    per_empty_prompt_en_count, per_empty_prompt_cn_count = 0, 0
    per_empty_prompt_count = 0

    for per_row_index in range(per_row_count):
        per_md5_key = get_stripped_text_value(per_md5_key_list[per_row_index])
        per_edit_task_name = get_stripped_text_value(
            per_edit_task_name_list[per_row_index])
        per_edit_subtask_name = get_stripped_text_value(
            per_edit_subtask_name_list[per_row_index])
        per_prompt_en = get_stripped_text_value(
            per_prompt_en_list[per_row_index])
        per_prompt_cn = get_stripped_text_value(
            per_prompt_cn_list[per_row_index])

        per_md5_key_count_dict[per_md5_key] += 1
        per_md5_key_error_message = get_md5_key_error_message(per_md5_key)
        if per_md5_key_error_message:
            per_invalid_md5_key_message_list.append(
                f'row {per_row_index} {per_md5_key_error_message}')

        per_edit_task_count_dict[per_edit_task_name] += 1
        per_edit_subtask_count_dict[per_edit_subtask_name] += 1
        per_edit_task_subtask_name_dict[(per_edit_task_name,
                                         per_edit_subtask_name)] = 1

        if per_edit_task_name not in EDIT_TASK_NAME_LIST:
            per_unknown_edit_task_name_dict[per_edit_task_name] = 1
        if per_edit_subtask_name not in EDIT_SUBTASK_NAME_LIST:
            per_unknown_edit_subtask_name_dict[per_edit_subtask_name] = 1
        elif per_edit_subtask_name not in EDIT_TASK_SUBTASK_NAME_DICT.get(
                per_edit_task_name, []):
            per_not_nested_edit_task_subtask_name_dict[(
                per_edit_task_name, per_edit_subtask_name)] = 1

        if not per_prompt_en:
            per_empty_prompt_en_count += 1
        if not per_prompt_cn:
            per_empty_prompt_cn_count += 1
        if not per_prompt_en and not per_prompt_cn:
            # 图像编辑样本对必须有编辑指令，中英都没有的样本对不可训练
            per_empty_prompt_count += 1

    per_text_column_dict['row_count'] = per_row_count
    per_text_column_dict['md5_key_list'] = sorted(
        per_md5_key_count_dict.keys())
    per_text_column_dict['edit_task_count_dict'] = dict(
        per_edit_task_count_dict)
    per_text_column_dict['edit_subtask_count_dict'] = dict(
        per_edit_subtask_count_dict)
    per_text_column_dict['edit_task_subtask_name_list'] = sorted(
        [[per_edit_task_name, per_edit_subtask_name] for per_edit_task_name,
         per_edit_subtask_name in per_edit_task_subtask_name_dict.keys()])
    per_text_column_dict['unknown_edit_task_name_list'] = sorted(
        per_unknown_edit_task_name_dict.keys())
    per_text_column_dict['unknown_edit_subtask_name_list'] = sorted(
        per_unknown_edit_subtask_name_dict.keys())
    per_text_column_dict['not_nested_edit_task_subtask_name_list'] = sorted(
        [[per_edit_task_name, per_edit_subtask_name]
         for per_edit_task_name, per_edit_subtask_name in
         per_not_nested_edit_task_subtask_name_dict.keys()])
    per_text_column_dict[
        'invalid_md5_key_message_list'] = per_invalid_md5_key_message_list[:
                                                                           MAX_SAVE_PROBLEM_ITEM_NUM]
    per_text_column_dict['empty_prompt_en_count'] = per_empty_prompt_en_count
    per_text_column_dict['empty_prompt_cn_count'] = per_empty_prompt_cn_count
    per_text_column_dict['empty_prompt_count'] = per_empty_prompt_count
    per_text_column_dict[
        'in_file_duplicate_md5_key_count'] = per_row_count - len(
            per_md5_key_count_dict)

    if per_empty_prompt_count != 0:
        # 实测全库0条，非0说明存在不可训练的样本对，必须硬失败
        error_message_list.append(
            f'{per_parquet_group_name} empty prompt count {per_empty_prompt_count}'
        )
    if len(per_invalid_md5_key_message_list) > 0:
        # 实测全库0条(全部是32位小写hex)，非0说明上游id规格变了
        error_message_list.append(
            f'{per_parquet_group_name} invalid md5 key count {len(per_invalid_md5_key_message_list)} {per_invalid_md5_key_message_list[:3]}'
        )

    return per_text_column_dict, error_message_list


def check_parquet_shard_complete(parquet_group_list):
    """解包前预检: parquet名规格、分片编号集合、"of"总片数，缺片或多片直接中止

    上游只上传了135片(编号0..240且严重不连续)，已三方对账过,
    所以磁盘编号集合必须**严格等于**EXPECTED_PARQUET_INDEX_LIST，缺号与多号都硬失败。
    """
    error_message_list = []

    shard_index_dict = {}
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

        if per_match_result.group('split') != PARQUET_SHARD_SPLIT_NAME:
            error_message_list.append(
                f'unknown parquet split name {per_parquet_name}')
            continue

        per_shard_total_num = int(per_match_result.group('total'))
        if per_shard_total_num != EXPECTED_PARQUET_SHARD_TOTAL_NUM:
            error_message_list.append(
                f'parquet shard total num not match {per_parquet_name} {per_shard_total_num} != {EXPECTED_PARQUET_SHARD_TOTAL_NUM}'
            )
            continue

        per_shard_index = int(per_match_result.group('index'))
        if per_shard_index in shard_index_dict:
            # 同一个编号出现两次说明文件名规格变了，按名解析会漏统计
            error_message_list.append(
                f'duplicate parquet index {per_parquet_name}')
            continue

        shard_index_dict[per_shard_index] = per_parquet_group_name

    expected_shard_index_set = set(EXPECTED_PARQUET_INDEX_LIST)
    missing_shard_index_list = sorted(expected_shard_index_set -
                                      set(shard_index_dict.keys()))
    unexpected_shard_index_list = sorted(
        set(shard_index_dict.keys()) - expected_shard_index_set)

    print('1111', 'parquet:', len(shard_index_dict), 'expected parquet:',
          len(expected_shard_index_set), 'missing index:',
          len(missing_shard_index_list), 'unexpected index:',
          len(unexpected_shard_index_list))

    if len(shard_index_dict) != EXPECTED_TOTAL_PARQUET_NUM:
        error_message_list.append(
            f'parquet num not match {len(shard_index_dict)} != {EXPECTED_TOTAL_PARQUET_NUM}'
        )
    if len(missing_shard_index_list) > 0:
        # 上游本来就只有这135片，缺任何一片都是本地没下全，必须硬失败
        error_message_list.append(
            f'parquet index missing {len(missing_shard_index_list)} {missing_shard_index_list[:10]}'
        )
    if len(unexpected_shard_index_list) > 0:
        # 多出编号说明上游补传了新分片，必须显式感知后更新ground truth
        error_message_list.append(
            f'parquet index unexpected {len(unexpected_shard_index_list)} {unexpected_shard_index_list[:10]}'
        )

    return error_message_list


def check_parquet_metadata_complete(parquet_group_list):
    """只读footer按片与实测ground truth逐项硬对账

    对账项: 7列/9个叶子列schema一致、每片恒15000行、恒150个row group且每组恒100行、
    七个有用叶子列的null数必须为0、两个path列必须恒null(只告警)、
    全局总行数必须等于2025000。
    """
    error_message_list = []

    total_row_count = 0
    per_parquet_metadata_dict = {}
    leaf_column_null_count_dict = collections.Counter()

    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_file_metadata, parquet_group_list),
                                     total=len(parquet_group_list)):
            per_metadata_dict, per_error_message_list = per_check_result
            error_message_list.extend(per_error_message_list)

            per_parquet_group_name = per_metadata_dict['parquet_group_name']
            per_row_count = per_metadata_dict['row_count']

            per_parquet_metadata_dict[
                per_parquet_group_name] = per_metadata_dict
            total_row_count += per_row_count
            leaf_column_null_count_dict.update(
                per_metadata_dict['leaf_column_null_count_dict'])

            if per_row_count != EXPECTED_PARQUET_ROW_COUNT:
                # 实测135/135片恒15000行，行数不对说明该片没下全
                error_message_list.append(
                    f'{per_parquet_group_name} row count not match {per_row_count} != {EXPECTED_PARQUET_ROW_COUNT}'
                )
            if per_metadata_dict[
                    'row_group_num'] != EXPECTED_PARQUET_ROW_GROUP_NUM:
                error_message_list.append(
                    f'{per_parquet_group_name} row group num not match {per_metadata_dict["row_group_num"]} != {EXPECTED_PARQUET_ROW_GROUP_NUM}'
                )
            for per_row_group_index, per_row_group_row_count in enumerate(
                    per_metadata_dict['row_group_row_count_list']):
                if per_row_group_row_count != EXPECTED_PARQUET_ROW_GROUP_ROW_COUNT:
                    error_message_list.append(
                        f'{per_parquet_group_name} row group {per_row_group_index} row count not match {per_row_group_row_count} != {EXPECTED_PARQUET_ROW_GROUP_ROW_COUNT}'
                    )

            for per_leaf_column_name in PARQUET_NOT_NULL_LEAF_COLUMN_NAME_LIST:
                per_null_count = per_metadata_dict[
                    'leaf_column_null_count_dict'].get(per_leaf_column_name, 0)
                if per_null_count != 0:
                    # 两张图/两条指令/key/两个任务名任一为null都是不完整样本对
                    error_message_list.append(
                        f'{per_parquet_group_name} {per_leaf_column_name} null count {per_null_count}'
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


def check_parquet_text_column_complete(parquet_group_list):
    """只读5个小文本列做全量硬对账(完全不碰图像字节，实测单片1.3秒)

    对账项: 逐edit_task行数、逐edit_subtask行数、(task,subtask,parquet)组合数、
    空指令数必须为0、md5 key规格; 并统计全库重复md5 key(只告警)。
    """
    error_message_list = []

    total_row_count = 0
    total_empty_prompt_count = 0
    total_empty_prompt_en_count, total_empty_prompt_cn_count = 0, 0
    total_in_file_duplicate_md5_key_count = 0
    md5_key_count_dict = collections.Counter()
    edit_task_count_dict = collections.Counter()
    edit_subtask_count_dict = collections.Counter()
    edit_task_subtask_parquet_name_dict = {}
    unknown_edit_task_name_dict, unknown_edit_subtask_name_dict = {}, {}
    not_nested_edit_task_subtask_name_dict = {}
    invalid_md5_key_message_list = []

    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_parquet_text_column, parquet_group_list),
                                     total=len(parquet_group_list)):
            per_text_column_dict, per_error_message_list = per_check_result
            error_message_list.extend(per_error_message_list)

            per_parquet_group_name = per_text_column_dict['parquet_group_name']

            total_row_count += per_text_column_dict['row_count']
            total_empty_prompt_count += per_text_column_dict[
                'empty_prompt_count']
            total_empty_prompt_en_count += per_text_column_dict[
                'empty_prompt_en_count']
            total_empty_prompt_cn_count += per_text_column_dict[
                'empty_prompt_cn_count']
            total_in_file_duplicate_md5_key_count += per_text_column_dict[
                'in_file_duplicate_md5_key_count']

            edit_task_count_dict.update(
                per_text_column_dict['edit_task_count_dict'])
            edit_subtask_count_dict.update(
                per_text_column_dict['edit_subtask_count_dict'])

            # 单文件内去重后的md5 key，用于统计跨文件重复(片内重复单独统计)
            for per_md5_key in per_text_column_dict['md5_key_list']:
                md5_key_count_dict[per_md5_key] += 1

            for per_edit_task_name, per_edit_subtask_name in per_text_column_dict[
                    'edit_task_subtask_name_list']:
                edit_task_subtask_parquet_name_dict[(
                    per_edit_task_name, per_edit_subtask_name,
                    per_parquet_group_name)] = 1

            for per_edit_task_name in per_text_column_dict[
                    'unknown_edit_task_name_list']:
                unknown_edit_task_name_dict[per_edit_task_name] = 1
            for per_edit_subtask_name in per_text_column_dict[
                    'unknown_edit_subtask_name_list']:
                unknown_edit_subtask_name_dict[per_edit_subtask_name] = 1
            for per_edit_task_name, per_edit_subtask_name in per_text_column_dict[
                    'not_nested_edit_task_subtask_name_list']:
                not_nested_edit_task_subtask_name_dict[(
                    per_edit_task_name, per_edit_subtask_name)] = 1

            invalid_md5_key_message_list.extend([
                f'{per_parquet_group_name} {per_invalid_md5_key_message}'
                for per_invalid_md5_key_message in
                per_text_column_dict['invalid_md5_key_message_list']
            ])

    # 重复md5 key = 片内重复 + 跨文件重复(实测5181个key各出现2次、共10362行)
    cross_file_duplicate_md5_key_count = sum([
        per_md5_key_count - 1
        for per_md5_key_count in md5_key_count_dict.values()
        if per_md5_key_count > 1
    ])
    duplicate_md5_key_count = cross_file_duplicate_md5_key_count + total_in_file_duplicate_md5_key_count

    unknown_edit_task_row_count = sum([
        per_row_count
        for per_edit_task_name, per_row_count in edit_task_count_dict.items()
        if per_edit_task_name not in EDIT_TASK_NAME_LIST
    ])
    unknown_edit_subtask_row_count = sum([
        per_row_count for per_edit_subtask_name, per_row_count in
        edit_subtask_count_dict.items()
        if per_edit_subtask_name not in EDIT_SUBTASK_NAME_LIST
    ])

    print('1111', 'text column total row:', total_row_count, 'unique md5 key:',
          len(md5_key_count_dict), 'duplicate md5 key:',
          duplicate_md5_key_count, 'empty prompt:', total_empty_prompt_count,
          'empty prompt en:', total_empty_prompt_en_count, 'empty prompt cn:',
          total_empty_prompt_cn_count, 'annotation file:',
          len(edit_task_subtask_parquet_name_dict))
    print('1111', 'edit task row:', dict(edit_task_count_dict))
    print('1111', 'edit subtask row:', dict(edit_subtask_count_dict))
    print('1111', 'unknown edit task:',
          sorted(unknown_edit_task_name_dict.keys()), 'row:',
          unknown_edit_task_row_count)
    print('1111', 'unknown edit subtask:',
          sorted(unknown_edit_subtask_name_dict.keys()), 'row:',
          unknown_edit_subtask_row_count)

    if total_row_count != EXPECTED_TOTAL_ROW_COUNT:
        error_message_list.append(
            f'text column total row count not match {total_row_count} != {EXPECTED_TOTAL_ROW_COUNT}'
        )
    if total_empty_prompt_count != 0:
        error_message_list.append(
            f'total empty prompt count {total_empty_prompt_count}')
    if len(invalid_md5_key_message_list) > 0:
        error_message_list.append(
            f'total invalid md5 key count {len(invalid_md5_key_message_list)} {invalid_md5_key_message_list[:3]}'
        )

    # 逐edit_task行数硬对账
    for per_edit_task_name in EDIT_TASK_NAME_LIST:
        per_expected_row_count = EXPECTED_EDIT_TASK_ROW_COUNT_DICT[
            per_edit_task_name]
        per_row_count = edit_task_count_dict.get(per_edit_task_name, 0)
        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_edit_task_name} text column row count not match {per_row_count} != {per_expected_row_count}'
            )

    # 逐edit_subtask行数硬对账(20个合法子类)
    for per_edit_subtask_name in EDIT_SUBTASK_NAME_LIST:
        per_expected_row_count = EXPECTED_EDIT_SUBTASK_ROW_COUNT_DICT[
            per_edit_subtask_name]
        per_row_count = edit_subtask_count_dict.get(per_edit_subtask_name, 0)
        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_edit_subtask_name} text column row count not match {per_row_count} != {per_expected_row_count}'
            )

    # 20个合法子类行数 + 兜底子类行数必须刚好等于总行数，一行都不能漏归类
    if sum(EXPECTED_EDIT_SUBTASK_ROW_COUNT_DICT.values()
           ) + unknown_edit_subtask_row_count != total_row_count:
        error_message_list.append(
            f'edit subtask row count not match {sum(EXPECTED_EDIT_SUBTASK_ROW_COUNT_DICT.values())} + {unknown_edit_subtask_row_count} != {total_row_count}'
        )
    if len(edit_task_subtask_parquet_name_dict
           ) != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        error_message_list.append(
            f'annotation file count not match {len(edit_task_subtask_parquet_name_dict)} != {EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}'
        )

    warning_message_list = []
    if unknown_edit_task_row_count != EXPECTED_UNKNOWN_EDIT_TASK_ROW_COUNT:
        # 实测恒为0，出现未知主类只告警(样本照常完整落到unknown_task兜底目录)
        warning_message_list.append(
            f'unknown edit task row count {unknown_edit_task_row_count} != {EXPECTED_UNKNOWN_EDIT_TASK_ROW_COUNT} {sorted(unknown_edit_task_name_dict.keys())[:3]}'
        )
    if unknown_edit_subtask_row_count != EXPECTED_UNKNOWN_EDIT_SUBTASK_ROW_COUNT:
        # 实测恰好1行(train-00067第7069行的subtask是脏值)，
        # 只告警(样本照常完整落到unknown_subtask兜底目录，绝不丢弃)
        warning_message_list.append(
            f'unknown edit subtask row count {unknown_edit_subtask_row_count} != {EXPECTED_UNKNOWN_EDIT_SUBTASK_ROW_COUNT} {sorted(unknown_edit_subtask_name_dict.keys())[:3]}'
        )
    if len(not_nested_edit_task_subtask_name_dict) > 0:
        warning_message_list.append(
            f'edit task subtask not nested {sorted(not_nested_edit_task_subtask_name_dict.keys())[:3]}'
        )
    if duplicate_md5_key_count != EXPECTED_DUPLICATE_MD5_KEY_COUNT:
        # 落盘文件名不用md5 key，所以重复key不会覆盖丢样本，只做软校验
        warning_message_list.append(
            f'duplicate md5 key count {duplicate_md5_key_count} != {EXPECTED_DUPLICATE_MD5_KEY_COUNT}'
        )
    if total_in_file_duplicate_md5_key_count != EXPECTED_IN_FILE_DUPLICATE_MD5_KEY_COUNT:
        warning_message_list.append(
            f'in file duplicate md5 key count {total_in_file_duplicate_md5_key_count} != {EXPECTED_IN_FILE_DUPLICATE_MD5_KEY_COUNT}'
        )
    if cross_file_duplicate_md5_key_count != EXPECTED_CROSS_FILE_DUPLICATE_MD5_KEY_COUNT:
        warning_message_list.append(
            f'cross file duplicate md5 key count {cross_file_duplicate_md5_key_count} != {EXPECTED_CROSS_FILE_DUPLICATE_MD5_KEY_COUNT}'
        )
    if len(md5_key_count_dict) != EXPECTED_UNIQUE_MD5_KEY_COUNT:
        warning_message_list.append(
            f'unique md5 key count {len(md5_key_count_dict)} != {EXPECTED_UNIQUE_MD5_KEY_COUNT}'
        )
    # 唯一key数 + 重复多出来的行数必须刚好等于总行数，否则说明key统计口径算错了
    if len(md5_key_count_dict) + duplicate_md5_key_count != total_row_count:
        error_message_list.append(
            f'md5 key count not match {len(md5_key_count_dict)} + {duplicate_md5_key_count} != {total_row_count}'
        )

    if total_empty_prompt_en_count > 0 or total_empty_prompt_cn_count > 0:
        warning_message_list.append(
            f'single language empty prompt count en {total_empty_prompt_en_count} cn {total_empty_prompt_cn_count}'
        )

    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    text_column_check_dict = {
        'text_column_total_row_count':
        total_row_count,
        'unique_md5_key_count':
        len(md5_key_count_dict),
        'duplicate_md5_key_count':
        duplicate_md5_key_count,
        'in_file_duplicate_md5_key_count':
        total_in_file_duplicate_md5_key_count,
        'cross_file_duplicate_md5_key_count':
        cross_file_duplicate_md5_key_count,
        'empty_prompt_count':
        total_empty_prompt_count,
        'empty_prompt_en_count':
        total_empty_prompt_en_count,
        'empty_prompt_cn_count':
        total_empty_prompt_cn_count,
        'text_column_edit_task_row_count_dict':
        dict(edit_task_count_dict),
        'text_column_edit_subtask_row_count_dict':
        dict(edit_subtask_count_dict),
        'text_column_annotation_file_count':
        len(edit_task_subtask_parquet_name_dict),
        'unknown_edit_task_name_list':
        sorted(unknown_edit_task_name_dict.keys()),
        'unknown_edit_subtask_name_list':
        sorted(unknown_edit_subtask_name_dict.keys()),
        'unknown_edit_task_row_count':
        unknown_edit_task_row_count,
        'unknown_edit_subtask_row_count':
        unknown_edit_subtask_row_count,
        'invalid_md5_key_message_list':
        invalid_md5_key_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'text_column_warning_message_list':
        warning_message_list,
    }

    return text_column_check_dict, error_message_list


def check_required_subset_complete(root_dataset_path, parquet_group_list):
    """解包前预检: 根目录条目白名单、data目录、分片编号集合、PAR1魔数、footer与文本列对账

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
    total_row_count, _, metadata_error_message_list = check_parquet_metadata_complete(
        parquet_group_list)
    error_message_list.extend(metadata_error_message_list)

    print('1111', 'check parquet text column:', len(parquet_group_list))
    text_column_check_dict, text_column_error_message_list = check_parquet_text_column_complete(
        parquet_group_list)
    error_message_list.extend(text_column_error_message_list)

    return total_row_count, text_column_check_dict, error_message_list


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

    单张图写盘异常不能让整片parquet的循环中断，否则该片后面上万行既不解图也不进标注。
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
        per_sample_key, per_md5_key, per_edit_task_name, per_edit_subtask_name,
        per_normalized_edit_task_name, per_normalized_edit_subtask_name,
        per_parquet_group_name, per_row_index, per_prompt_en, per_prompt_cn,
        per_src_image_relative_path, per_edit_image_relative_path,
        per_src_image_shape, per_edit_image_shape, per_src_image_suffix,
        per_edit_image_suffix):
    """拼一条完整编辑样本对标注

    该数据集parquet只有7列且两列是图像字节、两列path恒空，
    所以有用属性就是md5 key/edit_task/edit_subtask/中英双语指令，
    其余全是本脚本补的落盘路径与定位信息,
    下游可以直接按行取样本，不需要为了拿指令去扫405万个小文件。
    字段名与016脚本对齐(caption字段用instruction，参考图用reference_image_path_list),
    preprocessing2可以直接复用同一套读取口径。
    """
    per_save_annotation = {
        'dataset_task_type': DATASET_TASK_TYPE,
        'sample_key': per_sample_key,
        'md5_key': per_md5_key,
        'task_name': per_normalized_edit_task_name,
        'subtask_name': per_normalized_edit_subtask_name,
        'edit_task': per_edit_task_name,
        'edit_subtask': per_edit_subtask_name,
        'parquet_name': per_parquet_group_name,
        'row_index': per_row_index,
        'instruction': per_prompt_en,
        'prompt_en': per_prompt_en,
        'prompt_cn': per_prompt_cn,
        'reference_image_path_list': [per_src_image_relative_path],
        'reference_image_num': 1,
        'edited_image_path': per_edit_image_relative_path,
        'input_image_shape': per_src_image_shape,
        'edited_image_shape': per_edit_image_shape,
        'input_image_suffix': per_src_image_suffix,
        'edited_image_suffix': per_edit_image_suffix,
    }

    return per_save_annotation


def process_single_parquet_file(parquet_group, save_dataset_path,
                                save_annotation_dir_path):
    """流式解开单个parquet，把内嵌图像字节写成图像文件、非图像列写成jsonl汇总标注

    落盘结构(任务族名里的空格统一归一成下划线):
      unzip_images/<edit_task>/<edit_subtask>/<parquet>/<parquet>_%08d_src.png
      unzip_images/<edit_task>/<edit_subtask>/<parquet>/<parquet>_%08d_edit.png
      unzip_annotations/<edit_task>/<edit_subtask>/<parquet>.jsonl

    该数据集的key(md5)有5181个重复(见坑2)，绝不能当落盘文件名，
    样本key改由 <edit_task>/<edit_subtask>/<parquet>_<片内行号> 合成:
    parquet名全局唯一 + 片内行号唯一 => 合成key全局唯一，天然不会重名，
    所以不需要重名隔离目录。

    一个parquet的15000行是混合任务的，jsonl按(task, subtask)在内存里攒好
    (15000行约15MB，可忽略)再逐个落盘，避免同时开21个文件句柄交叉写。

    parquet按iter_batches流式读，绝不整片进内存(单片约42GB)。
    """
    per_parquet_group_name, per_parquet_relative_dir, per_parquet_path = parquet_group

    row_count, valid_sample_pair_count = 0, 0
    src_image_member_count, edit_image_member_count = 0, 0
    extract_image_count, skip_image_count = 0, 0
    not_save_image_count, save_image_fail_count = 0, 0
    unknown_edit_task_row_count, unknown_edit_subtask_row_count = 0, 0
    edit_task_count_dict = collections.Counter()
    edit_subtask_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    edited_image_shape_count_dict = collections.Counter()
    md5_key_count_dict = collections.Counter()
    annotation_line_dict = {}
    invalid_sample_pair_list, warning_message_list = [], []
    error_message_list = []
    reach_parquet_end = False

    try:
        load_parquet_file = pq.ParquetFile(per_parquet_path)
        expected_row_count = load_parquet_file.metadata.num_rows

        for per_record_batch in load_parquet_file.iter_batches(
                batch_size=PARQUET_ROW_BATCH_SIZE):
            for per_row_dict in per_record_batch.to_pylist():
                per_row_index = row_count
                row_count += 1

                per_invalid_reason_list = []

                per_md5_key = get_stripped_text_value(
                    per_row_dict.get(PARQUET_MD5_KEY_COLUMN_NAME, None))
                md5_key_count_dict[per_md5_key] += 1
                per_md5_key_error_message = get_md5_key_error_message(
                    per_md5_key)
                if per_md5_key_error_message:
                    # md5 key只用于溯源、不参与落盘命名，不合规只告警不丢样本
                    warning_message_list.append(
                        f'row {per_row_index} {per_md5_key_error_message}')

                per_edit_task_name = get_stripped_text_value(
                    per_row_dict.get(PARQUET_EDIT_TASK_COLUMN_NAME, None))
                per_edit_subtask_name = get_stripped_text_value(
                    per_row_dict.get(PARQUET_EDIT_SUBTASK_COLUMN_NAME, None))

                if per_edit_task_name in EDIT_TASK_NAME_LIST:
                    per_normalized_edit_task_name = get_normalized_name(
                        per_edit_task_name)
                else:
                    # 未知主类实测恒不会出现，出现时落到兜底目录并告警，样本照常完整保存
                    unknown_edit_task_row_count += 1
                    per_normalized_edit_task_name = UNKNOWN_EDIT_TASK_DIR_NAME
                    warning_message_list.append(
                        f'row {per_row_index} unknown edit task {per_edit_task_name}'
                    )

                if per_edit_subtask_name in EDIT_SUBTASK_NAME_LIST:
                    per_normalized_edit_subtask_name = get_normalized_name(
                        per_edit_subtask_name)
                    if per_edit_subtask_name not in EDIT_TASK_SUBTASK_NAME_DICT.get(
                            per_edit_task_name, []):
                        # 主类与子类的嵌套关系被破坏说明上游标注错位，必须显式感知
                        warning_message_list.append(
                            f'row {per_row_index} edit task subtask not nested {per_edit_task_name} {per_edit_subtask_name}'
                        )
                else:
                    # 实测train-00067第7069行的subtask被误填成一整句中文prompt,
                    # 中文长句不能当目录名，落到兜底目录并告警,
                    # **原始脏值完整写进jsonl，样本对照常完整保存，绝不丢弃**
                    unknown_edit_subtask_row_count += 1
                    per_normalized_edit_subtask_name = UNKNOWN_EDIT_SUBTASK_DIR_NAME
                    warning_message_list.append(
                        f'row {per_row_index} unknown edit subtask {per_edit_subtask_name[:50]}'
                    )

                per_prompt_en = get_stripped_text_value(
                    per_row_dict.get(PARQUET_PROMPT_EN_COLUMN_NAME, None))
                per_prompt_cn = get_stripped_text_value(
                    per_row_dict.get(PARQUET_PROMPT_CN_COLUMN_NAME, None))
                if not per_prompt_en and not per_prompt_cn:
                    # 图像编辑样本对必须有编辑指令，中英都没有的样本对不可训练
                    per_invalid_reason_list.append(
                        'empty prompt_en and prompt_cn')
                elif not per_prompt_en:
                    # 只缺一种语言时样本对仍然可训练，只告警不隔离
                    warning_message_list.append(
                        f'row {per_row_index} empty prompt_en')
                elif not per_prompt_cn:
                    warning_message_list.append(
                        f'row {per_row_index} empty prompt_cn')

                per_src_image_bytes = get_single_row_image_bytes(
                    per_row_dict, PARQUET_SRC_IMAGE_COLUMN_NAME)
                per_edit_image_bytes = get_single_row_image_bytes(
                    per_row_dict, PARQUET_EDIT_IMAGE_COLUMN_NAME)

                if per_src_image_bytes is None:
                    # 参考图实测全库null=0，恒存在，为空就是不完整样本对
                    per_invalid_reason_list.append('empty src image bytes')
                if per_edit_image_bytes is None:
                    per_invalid_reason_list.append('empty edit image bytes')

                per_image_name_prefix = f'{per_parquet_group_name}_{per_row_index:08d}'
                per_sample_key = f'{per_normalized_edit_task_name}/{per_normalized_edit_subtask_name}/{per_image_name_prefix}'

                save_image_dir_path = os.path.join(
                    save_dataset_path, SAVE_IMAGE_DIR_NAME,
                    per_normalized_edit_task_name,
                    per_normalized_edit_subtask_name, per_parquet_group_name)
                save_image_relative_dir = f'{SAVE_IMAGE_DIR_NAME}/{per_normalized_edit_task_name}/{per_normalized_edit_subtask_name}/{per_parquet_group_name}'

                per_src_image_relative_path = ''
                per_edit_image_relative_path = ''
                per_src_image_shape = [0, 0]
                per_edit_image_shape = [0, 0]
                per_src_image_suffix = ''
                per_edit_image_suffix = ''

                for per_image_bytes, per_image_name_suffix in [
                    [
                        per_src_image_bytes,
                        SAVE_SRC_IMAGE_NAME_SUFFIX,
                    ],
                    [
                        per_edit_image_bytes,
                        SAVE_EDIT_IMAGE_NAME_SUFFIX,
                    ],
                ]:
                    if per_image_bytes is None:
                        continue

                    per_is_src_image = per_image_name_suffix == SAVE_SRC_IMAGE_NAME_SUFFIX
                    if per_is_src_image:
                        src_image_member_count += 1
                    else:
                        edit_image_member_count += 1

                    per_image_suffix = get_image_bytes_suffix(per_image_bytes)
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
                                         per_save_image_name), per_image_bytes)
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

                    if per_is_src_image:
                        per_src_image_relative_path = per_image_relative_path
                        per_src_image_shape = per_image_shape
                        per_src_image_suffix = per_image_suffix
                    else:
                        per_edit_image_relative_path = per_image_relative_path
                        per_edit_image_shape = per_image_shape
                        per_edit_image_suffix = per_image_suffix

                if not per_src_image_relative_path:
                    per_invalid_reason_list.append('missing src image')
                if not per_edit_image_relative_path:
                    per_invalid_reason_list.append('missing edit image')

                if len(per_invalid_reason_list) > 0:
                    # 信息不完整的样本对不写进有效标注，但必须留痕，不能静默消失
                    invalid_sample_pair_list.append({
                        'sample_key':
                        per_sample_key,
                        'md5_key':
                        per_md5_key,
                        'task_name':
                        per_normalized_edit_task_name,
                        'subtask_name':
                        per_normalized_edit_subtask_name,
                        'parquet_name':
                        per_parquet_group_name,
                        'row_index':
                        per_row_index,
                        'invalid_reason':
                        ','.join(per_invalid_reason_list),
                    })
                    continue

                per_save_annotation = get_single_sample_pair_annotation(
                    per_sample_key, per_md5_key, per_edit_task_name,
                    per_edit_subtask_name, per_normalized_edit_task_name,
                    per_normalized_edit_subtask_name, per_parquet_group_name,
                    per_row_index, per_prompt_en, per_prompt_cn,
                    per_src_image_relative_path, per_edit_image_relative_path,
                    per_src_image_shape, per_edit_image_shape,
                    per_src_image_suffix, per_edit_image_suffix)

                annotation_line_dict.setdefault(
                    (per_normalized_edit_task_name,
                     per_normalized_edit_subtask_name), []).append(
                         json.dumps(per_save_annotation, ensure_ascii=False))
                valid_sample_pair_count += 1

                edit_task_count_dict[per_edit_task_name] += 1
                edit_subtask_count_dict[per_edit_subtask_name] += 1
                if per_edit_image_shape[0] > 0:
                    edited_image_shape_count_dict[
                        f'{per_edit_image_shape[0]}x{per_edit_image_shape[1]}'] += 1

        reach_parquet_end = True

        # 核心对账之一: 实际遍历到的行数必须等于footer里数出来的行数，
        # 否则说明流式读的时候有batch被静默吞掉了
        if row_count != expected_row_count:
            error_message_list.append(
                f'{per_parquet_group_name} row count not match {row_count} != {expected_row_count}'
            )
    except Exception as e:
        # parquet损坏或NAS读失败时保留已解出的图像，但必须上报，不能静默少样本
        print('7777', per_parquet_group_name, e)
        error_message_list.append(f'read parquet failed {e}')

    if not reach_parquet_end:
        error_message_list.append(
            'not reach parquet row batch end, parquet may be truncated')

    # 该片的汇总标注按(edit_task, edit_subtask)逐个落盘
    save_annotation_relative_path_list = []
    save_annotation_line_count = 0
    for per_normalized_edit_task_name, per_normalized_edit_subtask_name in sorted(
            annotation_line_dict.keys()):
        per_annotation_line_list = annotation_line_dict[(
            per_normalized_edit_task_name, per_normalized_edit_subtask_name)]

        per_save_annotation_relative_path = f'{SAVE_ANNOTATION_DIR_NAME}/{per_normalized_edit_task_name}/{per_normalized_edit_subtask_name}/{per_parquet_group_name}{SAVE_ANNOTATION_FILE_NAME_SUFFIX}'
        per_save_annotation_path = os.path.join(
            save_annotation_dir_path, per_normalized_edit_task_name,
            per_normalized_edit_subtask_name,
            f'{per_parquet_group_name}{SAVE_ANNOTATION_FILE_NAME_SUFFIX}')

        try:
            os.makedirs(os.path.dirname(per_save_annotation_path),
                        exist_ok=True)
            with open(per_save_annotation_path, 'w',
                      encoding='UTF-8') as save_annotation_file:
                for per_annotation_line in per_annotation_line_list:
                    save_annotation_file.write(f'{per_annotation_line}\n')
        except Exception as e:
            print('6666', per_save_annotation_path, e)
            error_message_list.append(
                f'{per_parquet_group_name} save annotation failed {per_save_annotation_relative_path} {e}'
            )
            continue

        save_annotation_relative_path_list.append(
            per_save_annotation_relative_path)
        save_annotation_line_count += len(per_annotation_line_list)

    # 核心对账之二: 每一行都恰好有一张参考图和一张编辑后图
    if src_image_member_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} src image member count not match {src_image_member_count} != {row_count}'
        )
    if edit_image_member_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} edit image member count not match {edit_image_member_count} != {row_count}'
        )

    # 核心对账之三: 每张图像成员都必须有明确归属(新写/跳过/不落盘/写失败)
    if extract_image_count + skip_image_count + not_save_image_count + save_image_fail_count != src_image_member_count + edit_image_member_count:
        error_message_list.append(
            f'{per_parquet_group_name} process image count not match: {extract_image_count} + {skip_image_count} + {not_save_image_count} + {save_image_fail_count} != {src_image_member_count} + {edit_image_member_count}'
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
    # 实测每行的两条指令与两张图都齐备，所以隔离清单应该恒为空，一旦非空必须显式感知
    if len(invalid_sample_pair_list) > 0:
        error_message_list.append(
            f'{per_parquet_group_name} invalid sample pair count {len(invalid_sample_pair_list)}'
        )
    if valid_sample_pair_count != row_count:
        error_message_list.append(
            f'{per_parquet_group_name} valid sample pair count not match {valid_sample_pair_count} != {row_count}'
        )

    # 核心对账之五: 落盘jsonl的总行数必须等于有效样本对数，
    # 否则说明有样本对在"攒标注->写jsonl"这一步消失了
    if save_annotation_line_count != valid_sample_pair_count:
        error_message_list.append(
            f'{per_parquet_group_name} save annotation line count not match {save_annotation_line_count} != {valid_sample_pair_count}'
        )

    return {
        'parquet_group_name':
        per_parquet_group_name,
        'row_count':
        row_count,
        'valid_sample_pair_count':
        valid_sample_pair_count,
        'src_image_member_count':
        src_image_member_count,
        'edit_image_member_count':
        edit_image_member_count,
        'extract_image_count':
        extract_image_count,
        'skip_image_count':
        skip_image_count,
        'not_save_image_count':
        not_save_image_count,
        'save_image_fail_count':
        save_image_fail_count,
        'unknown_edit_task_row_count':
        unknown_edit_task_row_count,
        'unknown_edit_subtask_row_count':
        unknown_edit_subtask_row_count,
        'in_file_duplicate_md5_key_count':
        row_count - len(md5_key_count_dict),
        'save_annotation_line_count':
        save_annotation_line_count,
        'save_annotation_relative_path_list':
        save_annotation_relative_path_list,
        'edit_task_count_dict':
        dict(edit_task_count_dict),
        'edit_subtask_count_dict':
        dict(edit_subtask_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'edited_image_shape_count_dict':
        dict(edited_image_shape_count_dict),
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
        # .cache里全是huggingface下载缓存文件，直接在遍历时剪掉整棵子树，不要走进去
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


def check_single_annotation_file_on_disk(annotation_check_pair):
    """可选的二次对账: 逐条读jsonl，核对每张图是否真的在落盘目录里"""
    per_save_dataset_path, per_annotation_relative_path = annotation_check_pair

    error_message_list = []

    per_annotation_path = os.path.join(per_save_dataset_path,
                                       per_annotation_relative_path)
    if not os.path.exists(per_annotation_path):
        error_message_list.append(
            f'{per_annotation_relative_path} annotation file not exist')

        return [per_annotation_relative_path, 0, 0, error_message_list]

    annotation_count, image_count, missing_image_count = 0, 0, 0
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_annotation_file:
            for per_annotation_line in load_annotation_file:
                per_annotation_line = per_annotation_line.strip()
                if not per_annotation_line:
                    continue

                per_annotation = json.loads(per_annotation_line)
                annotation_count += 1

                per_image_relative_path_list = list(
                    per_annotation.get('reference_image_path_list', []))
                if per_annotation.get('edited_image_path', ''):
                    per_image_relative_path_list.append(
                        per_annotation['edited_image_path'])

                for per_image_relative_path in per_image_relative_path_list:
                    per_image_path = os.path.join(per_save_dataset_path,
                                                  per_image_relative_path)
                    if not check_image_file_suffix(per_image_path):
                        # 标注里的图像路径后缀必须是图像后缀，不是就说明写标注时错位了
                        error_message_list.append(
                            f'{per_annotation_relative_path} unknown suffix image path {per_image_relative_path}'
                        )
                        continue

                    if os.path.exists(per_image_path):
                        image_count += 1
                    else:
                        missing_image_count += 1
    except Exception as e:
        error_message_list.append(
            f'{per_annotation_relative_path} load annotation failed {e}')

    # 每个完整编辑对贡献2张图(参考图 + 编辑后图)
    if image_count != annotation_count * 2:
        error_message_list.append(
            f'{per_annotation_relative_path} on disk image count not match {image_count} != {annotation_count} * 2'
        )
    if missing_image_count > 0:
        error_message_list.append(
            f'{per_annotation_relative_path} on disk missing image count {missing_image_count}'
        )

    return [
        per_annotation_relative_path,
        annotation_count,
        image_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, parquet_result_list):
    """可选的二次对账: 逐条读所有jsonl，核对每条标注引用的两张图都真的在盘上"""
    annotation_check_pair_list = []
    for per_parquet_result in parquet_result_list:
        for per_annotation_relative_path in per_parquet_result[
                'save_annotation_relative_path_list']:
            annotation_check_pair_list.append([
                save_dataset_path,
                per_annotation_relative_path,
            ])

    error_message_list = []
    total_annotation_count, total_image_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_annotation_file_on_disk,
                annotation_check_pair_list),
                                     total=len(annotation_check_pair_list)):
            _, per_annotation_count, per_image_count, per_error_message_list = per_check_result
            total_annotation_count += per_annotation_count
            total_image_count += per_image_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk annotation:', total_annotation_count,
          'on disk image:', total_image_count)

    if total_annotation_count != EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT:
        error_message_list.append(
            f'on disk annotation count not match {total_annotation_count} != {EXPECTED_TOTAL_VALID_SAMPLE_PAIR_COUNT}'
        )
    if total_image_count != EXPECTED_TOTAL_IMAGE_COUNT:
        error_message_list.append(
            f'on disk image count not match {total_image_count} != {EXPECTED_TOTAL_IMAGE_COUNT}'
        )

    return error_message_list


def save_check_result(save_dataset_path, parquet_result_list,
                      expected_total_row_count, text_column_check_dict):
    """汇总所有parquet的解包与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_row_count, total_valid_sample_pair_count = 0, 0
    total_src_image_member_count, total_edit_image_member_count = 0, 0
    total_extract_image_count, total_skip_image_count = 0, 0
    total_not_save_image_count, total_save_image_fail_count = 0, 0
    total_unknown_edit_task_row_count = 0
    total_unknown_edit_subtask_row_count = 0
    total_in_file_duplicate_md5_key_count = 0
    total_annotation_line_count, total_annotation_file_count = 0, 0
    edit_task_count_dict = collections.Counter()
    edit_subtask_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    edited_image_shape_count_dict = collections.Counter()
    parquet_sample_pair_count_dict = {}
    all_invalid_sample_pair_list, all_warning_message_list = [], []
    error_message_list, warning_message_list = [], []

    for per_parquet_result in parquet_result_list:
        per_parquet_group_name = per_parquet_result['parquet_group_name']

        total_row_count += per_parquet_result['row_count']
        total_valid_sample_pair_count += per_parquet_result[
            'valid_sample_pair_count']
        total_src_image_member_count += per_parquet_result[
            'src_image_member_count']
        total_edit_image_member_count += per_parquet_result[
            'edit_image_member_count']
        total_extract_image_count += per_parquet_result['extract_image_count']
        total_skip_image_count += per_parquet_result['skip_image_count']
        total_not_save_image_count += per_parquet_result[
            'not_save_image_count']
        total_save_image_fail_count += per_parquet_result[
            'save_image_fail_count']
        total_unknown_edit_task_row_count += per_parquet_result[
            'unknown_edit_task_row_count']
        total_unknown_edit_subtask_row_count += per_parquet_result[
            'unknown_edit_subtask_row_count']
        total_in_file_duplicate_md5_key_count += per_parquet_result[
            'in_file_duplicate_md5_key_count']
        total_annotation_line_count += per_parquet_result[
            'save_annotation_line_count']
        total_annotation_file_count += len(
            per_parquet_result['save_annotation_relative_path_list'])

        edit_task_count_dict.update(per_parquet_result['edit_task_count_dict'])
        edit_subtask_count_dict.update(
            per_parquet_result['edit_subtask_count_dict'])
        image_suffix_count_dict.update(
            per_parquet_result['image_suffix_count_dict'])
        edited_image_shape_count_dict.update(
            per_parquet_result['edited_image_shape_count_dict'])
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
          total_valid_sample_pair_count, 'src image member:',
          total_src_image_member_count, 'edit image member:',
          total_edit_image_member_count, 'extract image:',
          total_extract_image_count, 'skip image:', total_skip_image_count,
          'not save image:', total_not_save_image_count, 'save image fail:',
          total_save_image_fail_count, 'invalid sample pair:',
          len(all_invalid_sample_pair_list), 'warning:',
          len(all_warning_message_list))
    print('3333', 'annotation file:', total_annotation_file_count,
          'annotation line:', total_annotation_line_count,
          'unknown edit task row:', total_unknown_edit_task_row_count,
          'unknown edit subtask row:', total_unknown_edit_subtask_row_count,
          'in file duplicate md5 key:', total_in_file_duplicate_md5_key_count)
    print('3333', 'edit task row:', dict(edit_task_count_dict))
    print('3333', 'edit subtask row:', dict(edit_subtask_count_dict))
    print('3333', 'image suffix:', dict(image_suffix_count_dict))
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
        'expected_parquet_index_num':
        len(EXPECTED_PARQUET_INDEX_LIST),
        'parquet_shard_total_num_in_file_name':
        EXPECTED_PARQUET_SHARD_TOTAL_NUM,
        'total_row_count':
        total_row_count,
        'expected_total_row_count':
        expected_total_row_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_src_image_member_count':
        total_src_image_member_count,
        'total_edit_image_member_count':
        total_edit_image_member_count,
        'total_extract_image_count':
        total_extract_image_count,
        'total_skip_image_count':
        total_skip_image_count,
        'total_not_save_image_count':
        total_not_save_image_count,
        'total_save_image_fail_count':
        total_save_image_fail_count,
        'total_annotation_file_count':
        total_annotation_file_count,
        'total_annotation_line_count':
        total_annotation_line_count,
        'total_unknown_edit_task_row_count':
        total_unknown_edit_task_row_count,
        'total_unknown_edit_subtask_row_count':
        total_unknown_edit_subtask_row_count,
        'total_in_file_duplicate_md5_key_count':
        total_in_file_duplicate_md5_key_count,
        'invalid_sample_pair_count':
        len(all_invalid_sample_pair_list),
        'warning_message_count':
        len(all_warning_message_list),
        'edit_task_row_count_dict':
        dict(edit_task_count_dict),
        'edit_subtask_row_count_dict':
        dict(edit_subtask_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'edited_image_shape_count_dict':
        dict(edited_image_shape_count_dict),
        'parquet_sample_pair_count_dict':
        parquet_sample_pair_count_dict,
        'invalid_sample_pair_list':
        all_invalid_sample_pair_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        sorted(set(all_warning_message_list))[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    save_check_result_dict.update(text_column_check_dict)

    # 全量硬对账: 每一行都必须有归属，完整编辑对数必须等于预检时从footer数出来的
    # 实测ground truth，少一个都说明有样本对在"读parquet->写图->写jsonl"这条链路上消失了
    if len(parquet_result_list) != EXPECTED_TOTAL_PARQUET_NUM:
        error_message_list.append(
            f'total parquet count not match {len(parquet_result_list)} != {EXPECTED_TOTAL_PARQUET_NUM}'
        )
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
    if total_src_image_member_count != total_row_count:
        error_message_list.append(
            f'total src image member count not match {total_src_image_member_count} != {total_row_count}'
        )
    if total_edit_image_member_count != total_row_count:
        error_message_list.append(
            f'total edit image member count not match {total_edit_image_member_count} != {total_row_count}'
        )
    if total_src_image_member_count + total_edit_image_member_count != EXPECTED_TOTAL_IMAGE_COUNT:
        error_message_list.append(
            f'total image member count not match {total_src_image_member_count} + {total_edit_image_member_count} != {EXPECTED_TOTAL_IMAGE_COUNT}'
        )
    if total_extract_image_count + total_skip_image_count + total_not_save_image_count + total_save_image_fail_count != total_src_image_member_count + total_edit_image_member_count:
        error_message_list.append(
            f'total process image count not match {total_extract_image_count} + {total_skip_image_count} + {total_not_save_image_count} + {total_save_image_fail_count} != {total_src_image_member_count} + {total_edit_image_member_count}'
        )
    if total_save_image_fail_count > 0:
        error_message_list.append(
            f'total save image fail count {total_save_image_fail_count}')
    if len(all_invalid_sample_pair_list) > 0:
        error_message_list.append(
            f'invalid sample pair count {len(all_invalid_sample_pair_list)}')
    if total_annotation_line_count != total_valid_sample_pair_count:
        error_message_list.append(
            f'total annotation line count not match {total_annotation_line_count} != {total_valid_sample_pair_count}'
        )
    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        error_message_list.append(
            f'total annotation file count not match {total_annotation_file_count} != {EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}'
        )

    # 逐edit_task行数硬对账
    for per_edit_task_name in EDIT_TASK_NAME_LIST:
        per_expected_row_count = EXPECTED_EDIT_TASK_ROW_COUNT_DICT[
            per_edit_task_name]
        per_row_count = edit_task_count_dict.get(per_edit_task_name, 0)
        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_edit_task_name} valid sample pair count not match {per_row_count} != {per_expected_row_count}'
            )

    # 逐edit_subtask行数硬对账(20个合法子类)
    for per_edit_subtask_name in EDIT_SUBTASK_NAME_LIST:
        per_expected_row_count = EXPECTED_EDIT_SUBTASK_ROW_COUNT_DICT[
            per_edit_subtask_name]
        per_row_count = edit_subtask_count_dict.get(per_edit_subtask_name, 0)
        if per_row_count != per_expected_row_count:
            error_message_list.append(
                f'{per_edit_subtask_name} valid sample pair count not match {per_row_count} != {per_expected_row_count}'
            )

    # 20个合法子类行数 + 兜底子类行数必须刚好等于有效样本对数，一行都不能漏归类
    if sum(EXPECTED_EDIT_SUBTASK_ROW_COUNT_DICT.values(
    )) + total_unknown_edit_subtask_row_count != total_valid_sample_pair_count:
        error_message_list.append(
            f'edit subtask valid sample pair count not match {sum(EXPECTED_EDIT_SUBTASK_ROW_COUNT_DICT.values())} + {total_unknown_edit_subtask_row_count} != {total_valid_sample_pair_count}'
        )

    if total_unknown_edit_task_row_count != EXPECTED_UNKNOWN_EDIT_TASK_ROW_COUNT:
        # 实测恒为0，出现未知主类只告警(样本照常完整落到unknown_task兜底目录)
        warning_message_list.append(
            f'total unknown edit task row count {total_unknown_edit_task_row_count} != {EXPECTED_UNKNOWN_EDIT_TASK_ROW_COUNT}'
        )
    if total_unknown_edit_subtask_row_count != EXPECTED_UNKNOWN_EDIT_SUBTASK_ROW_COUNT:
        # 实测恰好1行(train-00067第7069行的subtask是脏值),
        # 只告警(样本照常完整落到unknown_subtask兜底目录，绝不丢弃)
        warning_message_list.append(
            f'total unknown edit subtask row count {total_unknown_edit_subtask_row_count} != {EXPECTED_UNKNOWN_EDIT_SUBTASK_ROW_COUNT}'
        )
    for per_image_suffix in sorted(image_suffix_count_dict.keys()):
        if per_image_suffix != EXPECTED_IMAGE_FILE_SUFFIX:
            # 实测两类图像全部是PNG，出现别的格式只告警(后缀按魔数判定，不会存错)
            warning_message_list.append(
                f'unexpected image suffix {per_image_suffix} {image_suffix_count_dict[per_image_suffix]}'
            )

    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_dict['check_warning_message_list'] = warning_message_list
    save_check_result_dict[
        'check_error_message_list'] = error_message_list[:
                                                         MAX_SAVE_PROBLEM_ITEM_NUM]
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

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

    expected_total_row_count, text_column_check_dict, precheck_error_message_list = check_required_subset_complete(
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

            print(
                '2222', per_parquet_result['parquet_group_name'], 'row:',
                per_parquet_result['row_count'], 'valid sample pair:',
                per_parquet_result['valid_sample_pair_count'],
                'extract image:', per_parquet_result['extract_image_count'],
                'skip image:', per_parquet_result['skip_image_count'],
                'not save image:', per_parquet_result['not_save_image_count'],
                'save image fail:',
                per_parquet_result['save_image_fail_count'],
                'annotation file:',
                len(per_parquet_result['save_annotation_relative_path_list']),
                'invalid sample pair:',
                len(per_parquet_result['invalid_sample_pair_list']))

    check_error_message_list = save_check_result(save_dataset_path,
                                                 parquet_result_list,
                                                 expected_total_row_count,
                                                 text_column_check_dict)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG and EXTRACT_IMAGE_FILE_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, parquet_result_list)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解包/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/UnicEdit-10M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
