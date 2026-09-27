import io
import os
import re
import json
import shutil
import tarfile
import collections

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: ConceptEdit-12M(inclusionAI/ConceptEdit-12M, 论文arXiv:2608.16812)
#
# 【数据集类型】纯图像编辑(instruction-based image editing)数据集，不是文生图数据集。
# README原文 task_categories: image-to-image，tags: image-editing/instruction-based-editing,
# "Each sample is stored as a triplet: a source image, an edited image, a JSON metadata file"。
# 每个样本对固定是"1张参考图(编辑前图<id>_source.jpg) + 1条编辑指令(中英双语、粗细两档共4条) +
# 1张编辑后图(<id>_edit.png)"，没有第二张视觉条件图、没有mask,
# 也没有任何"只有caption + 一张图"的纯生成样本，
# 所以下游只能走ti2i_dataset.py那条链路，不能当t2i(文生图)数据用。
# (json里的original_simple_caption是"源图的简短描述"，是给编辑任务做上下文用的，
#  它描述的是参考图而不是编辑后图，**不能**拿来当文生图的caption使用。)
#
# 【root_dataset_path实测原始保存规格(共21.4T / 616个tar)】
# ConceptEdit-12M/
# ├── enhanced_prompt_square_resolution/  154个 batch_{0..153}.tar   (5.51T) -> 有用
# ├── enhanced_prompt_random_resolution/  162个 batch_{0..161}.tar   (5.57T) -> 有用
# ├── original_prompt_square_resolution/  131个 batch_{0..130}.tar   (4.52T) -> 有用
# ├── original_prompt_random_resolution/  169个 batch_{0..168}.tar   (5.81T) -> 有用
# ├── README.md         数据集说明(无用，授权信息已记进校验报告的dataset_license_name字段)
# ├── .gitattributes    git lfs配置(无用)
# └── .cache/           huggingface下载缓存，660个文件，**残留39个*.incomplete**(无用)
#
# 四个子集名就是两个正交维度的组合，**必须原样保留成属性**(下游按需过滤/配比):
#   enhanced_prompt / original_prompt : 编辑指令是否被增强改写过
#   square_resolution / random_resolution : 图像分辨率是正方形还是任意长宽比
#
# 【完整性实测(已三方对账，当前数据集是完整的)】
# a. 4个子集的tar编号都是 0..N-1 **严格连号、无缺号**(154/162/131/169，合计616);
# b. 616个tar **全部512字节对齐且尾部1024字节EOF块完好**(并行O(1)读尾部实测0个异常);
# c. huggingface下载缓存里的仓库文件清单 .cache/huggingface/trees/*.json 共618个条目
#    (616个tar + README.md + .gitattributes)，磁盘上**全部存在且字节数100%一致**,
#    磁盘上也没有清单外的多余文件;
#    .cache里那39个*.incomplete只是断点续传残渣，数据集本体是完整的。
#
# 【tar内部规格(实测抽样8个tar + 2个tar全量遍历)】
# 每个tar内只有**一个顶层目录**且目录名 == tar名(batch_130.tar -> batch_130/),
# 目录下是平铺的三元组文件，成员按"json -> _edit.png -> _source.jpg"三个一组顺序排列:
#   batch_130/<id>.json
#   batch_130/<id>_edit.png
#   batch_130/<id>_source.jpg
# 实测 batch_130 = 31809个成员 = 10603 × 3，json/source/edit三方id集合**完全相等**,
# id无重复、无孤立成员、无目录成员、无非三元组文件;
# **每个tar的样本数不固定**(实测batch_130是10603、batch_129是19687),
# 所以绝不能把"每片N条"当硬性条件，只能拿每个tar自己的成员总数当ground truth。
#
# 【必须显式感知的规格坑】
# 1) **同一个id会在4个子集里各出现一次，但它们是完全不同的编辑样本**。
#    实测 id=1093_0_9 在三个子集的batch_0里都有，但
#      enhanced_square : "Apply a silhouette effect..." 剪影效果   source md5 8656e68805
#      enhanced_random : "Press down the brightness..." 背景压暗   source md5 efb3e7965d
#      original_square : "Add a light snowfall effect." 小雪       source md5 565855e16a
#    三者的参考图字节、指令、编辑后图**全不相同**(源图都来自Fine-T2I的同一张原图，
#    但做了不同的预处理与不同的编辑)，**每一条都是独立完整的有效样本对，一条都不能丢**。
#    ⇒ 落盘必须按 <subset>/<archive>/ 两级分层，sample_key也必须带subset与archive,
#      否则这些同名id会互相覆盖、静默丢掉几十万个样本对。
#    同一个子集内部**跨tar的id无交集**(实测batch_129与batch_130交集为0),
#    tar内部id也无重复，所以加上这两级目录后落盘文件名天然全局唯一，不需要重编号。
# 2) **id不全是纯数字**: 形如 <数字>_<数字>_<数字> 或 <数字>_<数字>_auto
#    (实测id里出现过的字符只有 0-9 _ a o t u，即数字与"auto"),
#    所以解析成员名时**绝不能按下划线切分取数字**，只能按 _source / _edit / .json 后缀切。
# 3) **edit_concept.category的取值不是固定枚举**: 主要是7类中文大类
#    (人像与人体专属/高阶应用与垂类/通用物体与实体编辑/全局画质与氛围/生成与画面重构/
#     文字与平面设计/环境与风格)，但抽样6000条里混有十几条英文写法
#    (Global Atmosphere / General Object Editing / Portrait & Body ...)与长尾中文类目。
#    ⇒ **绝不能把category写死成白名单去过滤样本**(会静默丢掉这些长尾样本),
#      这里只统计分布、不做任何过滤，落盘目录也不按category分层。
# 4) **recaption_prompt_en/zh大量为空串**(抽样6000条里5095条为空)，
#    它是VQA发现指令与实际编辑效果有偏差时给出的"重写指令"，为空是正常的,
#    ⇒ 只能当可选属性保留，**不能拿它当训练主文本**。
#
# 【单个样本对的全部有用信息(抽样6000条json，6个顶层字段100%齐备)】
#   <id>_source.jpg : 参考图(编辑前图)，实测全部JPEG、RGB三通道        -> 有用(参考图，必需)
#   <id>_edit.png   : 编辑后图，实测全部PNG、RGB三通道                 -> 有用(编辑后图，必需)
#   <id>.json 的字段:
#     id                    : 样本id，== 三个文件名的公共prefix(实测100%一致) -> 有用(溯源)
#     images.source/edited  : tar内相对路径，实测与真实成员名100%一致
#                             -> 有用(交叉校验用；落盘路径以真实成员名为准)
#     edit_concept          : category / sub_category / task / detail 四级编辑分类,
#                             实测四个字段都非空                       -> 有用(任务族采样/加权)
#     instruction           : short_en / short_zh / detailed_en / detailed_zh,
#                             **实测6000/6000条四条全部非空**，中英双语粗细两档
#                             -> 有用(训练主文本，默认取detailed_en)
#     evaluation.overall_vqa_score : 实测只有0.8与1.0两种取值            -> 有用(质量过滤/加权)
#     evaluation.keep              : **实测抽样全部为true**(上游已按keep过滤过)-> 有用
#     evaluation.wrong_count       : 未通过的VQA维度数                   -> 有用
#     evaluation.recaption_prompt_en/zh : VQA纠偏后的重写指令，大量为空   -> 有用(可选)
#     evaluation.vqa               : 恒5个维度(1_primary_edit_success /
#                                    2_logic_and_physics / 3_key_details /
#                                    4_specific_preservation /
#                                    5_artifacts_and_anatomy),
#                                    每个含 question_en/question_zh/expected_answer/passed
#                                    -> 有用(细粒度质量信号)
#     original_simple_caption      : **源图**的简短英文描述，实测全部非空  -> 有用(源图上下文)
# 除这些之外没有任何其他单样本属性(无宽高列/无授权列/无mask),
# 所以宽高只能在解包时顺手从图像header里读出来存进jsonl(见PARSE_IMAGE_SHAPE_FLAG)。
#
# 【无用信息(一律不整理进训练目录)】
# .cache/(660个文件，含39个*.incomplete) / .gitattributes / README.md /
# .gitignore / .DS_Store / CACHEDIR.TAG 这类目录元数据垃圾文件。
# 该数据集没有demo图、没有评测统计npz，过滤掉上面这些之后根目录只剩4个子集目录。
#
# 【本脚本的处理口径】
# - 解包前预检(硬失败，不过就不白跑几十小时):
#   a. 根目录条目白名单(只允许4个子集目录，多出未知文件或未知目录立即上报);
#   b. 每个子集的tar数必须等于实测值(154/162/131/169)、
#      tar名必须是 batch_<index>.tar、编号必须严格是 0..N-1 连号(缺号与多号都硬失败);
#   c. 616个tar逐个O(1)读尾部，校验512字节对齐 + 1024字节全0的EOF块(拦下载截断);
# - 并行单位 = 单个tar(616个任务，Pool(32)),
#   MultiPartArchiveReader + tarfile 'r|*' 流式读，**绝不整包进内存**(单包最大约38GB);
# - 落盘 unzip_images/<subset>/<archive>/<id>.json|<id>_source.jpg|<id>_edit.png,
#   **图像直接写原始字节，unzip阶段绝不引入二次编解码**,
#   resize/转格式/分辨率分桶留给preprocessing2的resave脚本;
#   原始json也原样落盘，保证VQA问题原文等冗长信息一条不丢(jsonl里只存精简后的可训练字段);
# - 每个成员写盘后**立刻校验落盘大小 == tar头里的size**(比只看存在性强，能挡住写半截/写0字节);
#   已存在且大小一致就计skip并跳过，保证脚本可以断点续跑
#   (续跑时图像宽高改从磁盘文件header读，jsonl依然是完整的);
# - 汇总标注落 unzip_annotations/<subset>/<archive>.jsonl，每行一个完整编辑样本对,
#   **中英双语、粗细两档共4条指令全部保留**，并保留四级编辑分类与全部VQA质量字段,
#   再补上落盘路径、样本key、所属子集与tar、两张图的真实宽高与后缀
#   (字段名与013/015/016脚本对齐，preprocessing2可直接复用同一套读取口径);
# - 片内多方硬对账，**任何一条不过都抛异常，绝不静默少样本对**:
#   a. 必须正常读到tar流末尾(截断/NAS读失败不再像老脚本那样只print);
#   b. extract + skip + not_save + fail + duplicate == tar头里数出来的成员总数;
#   c. json成员数 == 参考图成员数 == 编辑后图成员数 == 有效样本对数,
#      且 成员总数 == 有效样本对数 × 3;
#   d. json里的id与文件名prefix、images.source/edited与真实成员名逐条交叉校验;
#   e. **致命问题**(缺图/孤儿图/四条指令全空/json解析失败/写盘失败/重名成员)
#      逐条进隔离清单且**硬失败**，因为它们都意味着"有样本对没能完整落进jsonl";
#      **非致命告警**(某个可选字段为空、vqa维度数变了、json里的图路径与成员名对不上等)
#      只进warning清单打印，**不判失败**——这些样本对本身依然完整有效、照样落盘,
#      硬失败反而会把"数据能用"误报成"数据有问题";
#   f. tar内重名成员写进 unzip_duplicate_members/ 隔离目录保留并上报，绝不互相覆盖;

# - 全局校验报告 unzip_check_missing_images.json: 总样本对数、逐子集样本对数、
#   编辑分类分布、VQA分数与未通过维度分布、图像后缀与分辨率TopN、各类问题清单;
#   官方只声称"12M"没给精确条数，所以总数只做**软校验**(落在区间外打印告警，不判失败)。
#
# 【跑之前务必确认目标盘扛得住】
# - EXTRACT_IMAGE_FILE_FLAG=True 时输出小文件数约 **1200万样本对 × 3 = 3600万个文件**(约21T),
#   NAS上inode与元数据压力极大，务必确认目标盘扛得住再跑;
# - 只想先建索引可把 EXTRACT_IMAGE_FILE_FLAG 置False，
#   图像继续留在原tar里按webdataset方式读，样本对信息一样完整。
# ==============================================================================

DATASET_TASK_TYPE = 'image_edit'

DATASET_LICENSE_NAME = 'apache-2.0'

ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.tar$'),
]

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(660个文件，含39个*.incomplete)
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

# 过滤掉无用信息后根目录只剩这4个子集目录
SUBSET_ROOT_DIR_NAME_LIST = [
    'enhanced_prompt_square_resolution',
    'enhanced_prompt_random_resolution',
    'original_prompt_square_resolution',
    'original_prompt_random_resolution',
]

# 实测每个子集的tar分片数(README声明值与实测值完全一致，合计616)，数量不对说明下载不全
EXPECTED_SUBSET_ARCHIVE_NUM_DICT = {
    'enhanced_prompt_square_resolution': 154,
    'enhanced_prompt_random_resolution': 162,
    'original_prompt_square_resolution': 131,
    'original_prompt_random_resolution': 169,
}

EXPECTED_TOTAL_ARCHIVE_NUM = 616

# 子集名拆出来的两个正交属性，原样写进每条标注，下游可按需过滤或配比
SUBSET_PROMPT_TYPE_DICT = {
    'enhanced_prompt_square_resolution': 'enhanced_prompt',
    'enhanced_prompt_random_resolution': 'enhanced_prompt',
    'original_prompt_square_resolution': 'original_prompt',
    'original_prompt_random_resolution': 'original_prompt',
}

SUBSET_RESOLUTION_TYPE_DICT = {
    'enhanced_prompt_square_resolution': 'square_resolution',
    'enhanced_prompt_random_resolution': 'random_resolution',
    'original_prompt_square_resolution': 'square_resolution',
    'original_prompt_random_resolution': 'random_resolution',
}

# tar名(不含后缀): batch_0 ... batch_168，编号必须0..N-1连号
ARCHIVE_SHARD_NAME_PATTERN = re.compile(r'^batch_(?P<index>\d+)$')

ANNOTATION_FILE_SUFFIX = '.json'

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.bmp',
    '.gif',
    '.tif',
]

# 三元组成员名规格。
# id里可能带"auto"(如1114_0_auto)，**绝不能按下划线切分取数字**，只能按尾部后缀切。
ANNOTATION_MEMBER_NAME_PATTERN = re.compile(r'^(?P<sample_id>.+)\.json$')

SOURCE_IMAGE_MEMBER_NAME_PATTERN = re.compile(
    r'^(?P<sample_id>.+)_source(?P<suffix>\.[A-Za-z0-9]+)$')

EDITED_IMAGE_MEMBER_NAME_PATTERN = re.compile(
    r'^(?P<sample_id>.+)_edit(?P<suffix>\.[A-Za-z0-9]+)$')

MEMBER_KIND_ANNOTATION = 'annotation'

MEMBER_KIND_SOURCE_IMAGE = 'source_image'

MEMBER_KIND_EDITED_IMAGE = 'edited_image'

# json里必须齐备的6个顶层字段(实测抽样6000/6000条全部齐备),
# 缺失只上报不丢样本(除非训练主文本也拿不到)
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    'id',
    'images',
    'edit_concept',
    'instruction',
    'evaluation',
    'original_simple_caption',
]

ANNOTATION_SAMPLE_ID_KEY_NAME = 'id'

ANNOTATION_IMAGES_KEY_NAME = 'images'

ANNOTATION_IMAGES_SOURCE_KEY_NAME = 'source'

ANNOTATION_IMAGES_EDITED_KEY_NAME = 'edited'

ANNOTATION_EDIT_CONCEPT_KEY_NAME = 'edit_concept'

ANNOTATION_EDIT_CONCEPT_SUB_KEY_NAME_LIST = [
    'category',
    'sub_category',
    'task',
    'detail',
]

ANNOTATION_INSTRUCTION_KEY_NAME = 'instruction'

# 4条指令(中英双语 × 粗细两档)，实测全部非空，全部保留进jsonl
ANNOTATION_INSTRUCTION_SUB_KEY_NAME_LIST = [
    'short_en',
    'short_zh',
    'detailed_en',
    'detailed_zh',
]

# 训练主文本取值优先级: 详细英文 -> 简短英文 -> 详细中文 -> 简短中文。
# 四条全空才算"没有文本条件"，该样本对进隔离清单(实测不应出现)
ANNOTATION_MAIN_INSTRUCTION_KEY_NAME_LIST = [
    'detailed_en',
    'short_en',
    'detailed_zh',
    'short_zh',
]

ANNOTATION_EVALUATION_KEY_NAME = 'evaluation'

ANNOTATION_EVALUATION_SUB_KEY_NAME_LIST = [
    'overall_vqa_score',
    'keep',
    'wrong_count',
    'recaption_prompt_en',
    'recaption_prompt_zh',
    'vqa',
]

ANNOTATION_EVALUATION_VQA_KEY_NAME = 'vqa'

ANNOTATION_VQA_DIMENSION_KEY_NAME = 'dimension'

ANNOTATION_VQA_PASSED_KEY_NAME = 'passed'

ANNOTATION_ORIGINAL_CAPTION_KEY_NAME = 'original_simple_caption'

# 实测恒为这5个VQA维度，多出/少掉只打印告警，不判失败(不影响样本对完整性)
EXPECTED_VQA_DIMENSION_NAME_LIST = [
    '1_primary_edit_success',
    '2_logic_and_physics',
    '3_key_details',
    '4_specific_preservation',
    '5_artifacts_and_anatomy',
]

EXPECTED_VQA_DIMENSION_NUM = 5

SAVE_IMAGE_DIR_NAME = 'unzip_images'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_DUPLICATE_MEMBER_DIR_NAME = 'unzip_duplicate_members'

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

# README只声称"12M"级别，官方没给精确条数，
# 按21.4T总大小与实测单样本对约1.8MB换算成宽松区间，只做软校验(打印告警)，不判失败
EXPECTED_TOTAL_SAMPLE_PAIR_COUNT_RANGE = [10000000, 14000000]

TAR_BLOCK_SIZE = 512

TAR_EOF_BLOCK_SIZE = 1024

# 图像成员是否落盘。
# True : 和其他数据集脚本口径一致，约1200万样本对 × 3个文件 = 约3600万个小文件、21T,
#        NAS上inode和元数据压力极大，务必确认目标盘扛得住再跑;
# False: 只解析json生成 unzip_annotations/*.jsonl 索引，图像继续留在原tar里,
#        训练时按webdataset方式顺序读，样本对信息一样是完整的。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解包后再os.walk一遍输出目录做二次对账。
# 默认False: 3600万个小文件的os.walk在NAS上要跑非常久，而解包时已经做了
# "写盘后立刻校验落盘大小 == tar头size" + "三类成员数与样本对数多方对账"两道对账，
# 已经能保证每个成员都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

# 是否解码图像header拿真实宽高写进标注。
# 默认True: 只解header不解像素(PIL的Image.open是惰性的)，代价可忽略;
# 该数据集json里**没有任何宽高字段**，不解header下游就只能在训练时逐张打开图才能分桶,
# 所以这里顺手把宽高存进jsonl，省掉后续一次3600万张图的全量扫盘。
PARSE_IMAGE_SHAPE_FLAG = True

# 是否把VQA的5个问题原文(question_en/question_zh/expected_answer/passed)整份内嵌进jsonl。
# 默认False: 每条约1.5KB，1200万条会让jsonl从约20G涨到约40G，而原始json已经原样落盘
# (路径记在每条标注的annotation_path字段里)，信息一条没丢，需要时随时能读回来;
# jsonl里默认只保留"维度总数/通过数/未通过的维度名列表"这几个可直接用于过滤的精简信号。
SAVE_FULL_VQA_DETAIL_FLAG = False

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

MAX_SAVE_IMAGE_SHAPE_ITEM_NUM = 100

PROCESS_NUM = 32

COPY_FILE_BLOCK_SIZE = 16 * 1024 * 1024

EXTRACT_FILE_BLOCK_SIZE = 4 * 1024 * 1024


class MultiPartArchiveReader:
    """把按字节切分的多个分片压缩包拼接成一个只读的连续字节流

    该数据集每个tar都是独立完整的(不是分卷)，这里保留多分片能力只是为了和其他脚本口径一致。
    """

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


def get_image_bytes_suffix(per_image_bytes, per_default_suffix):
    """用图像字节的魔数推断真实后缀

    实测参考图恒为JPEG、编辑后图恒为PNG，和文件名后缀一致;
    这里仍然逐张按魔数判定，上游哪天换了编码也不会存错后缀。
    拿不到字节(不落盘且不解header)时退回文件名后缀。
    """
    if not per_image_bytes:
        return per_default_suffix

    for per_magic_bytes, per_magic_suffix in IMAGE_BYTES_MAGIC_SUFFIX_LIST:
        if per_image_bytes.startswith(per_magic_bytes):
            return per_magic_suffix

    return per_default_suffix


def get_stripped_text_value(per_value):
    """文本字段统一转成strip后的字符串，None/非字符串/空白都当成缺失"""
    if not isinstance(per_value, str):
        return ''

    return per_value.strip()


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


def get_image_shape_from_file(per_image_path):
    """断点续跑时图像字节没有过手，改从磁盘上已落盘的图像header里读宽高

    只有这样"跳过已存在文件"的那部分样本对在jsonl里才依然带真实宽高，
    不会因为续跑就退化成[0, 0]。
    """
    if not PARSE_IMAGE_SHAPE_FLAG:
        return [0, 0], ''

    try:
        with Image.open(per_image_path) as load_image:
            per_image_width, per_image_height = load_image.size
    except Exception as e:
        return [0, 0], f'parse image shape from file failed {e}'

    return [per_image_width, per_image_height], ''


def get_archive_part_sort_key(per_archive_part_index):
    """分片编号排序key: 纯数字编号按数值排序，字母编号按位数优先再按字典序排序

    返回值统一是[编号类型, 数值编号, 字母编号]三元组，保证不同命名风格之间也能比较。
    不能直接按字符串排序: part-9 会排到 part-10 后面；
    也不能只按[位数, 字符串]排序: part-100 会排到 part-89 前面导致整个tar流错位。
    """
    per_archive_part_index = per_archive_part_index or ''
    if per_archive_part_index.isdigit():
        return [0, int(per_archive_part_index), '']

    return [1, len(per_archive_part_index), per_archive_part_index]


def get_member_kind_and_sample_id(per_member_base_name):
    """把tar成员名解析成[成员类型, 样本id, 后缀]

    三元组固定是 <id>.json / <id>_source.jpg / <id>_edit.png,
    id里可能带"auto"(如1114_0_auto)，所以只能按尾部后缀切，绝不能按下划线切分。
    解析不出来返回['', '', '']，由调用方当未知成员上报(绝不默认当图像处理)。
    """
    per_member_base_name = per_member_base_name.replace('\\', '/')

    per_match_result = ANNOTATION_MEMBER_NAME_PATTERN.match(
        per_member_base_name)
    if per_match_result:
        return [
            MEMBER_KIND_ANNOTATION,
            per_match_result.group('sample_id'),
            ANNOTATION_FILE_SUFFIX,
        ]

    per_match_result = SOURCE_IMAGE_MEMBER_NAME_PATTERN.match(
        per_member_base_name)
    if per_match_result and per_match_result.group(
            'suffix').lower() in IMAGE_FILE_SUFFIX_LIST:
        return [
            MEMBER_KIND_SOURCE_IMAGE,
            per_match_result.group('sample_id'),
            per_match_result.group('suffix').lower(),
        ]

    per_match_result = EDITED_IMAGE_MEMBER_NAME_PATTERN.match(
        per_member_base_name)
    if per_match_result and per_match_result.group(
            'suffix').lower() in IMAGE_FILE_SUFFIX_LIST:
        return [
            MEMBER_KIND_EDITED_IMAGE,
            per_match_result.group('sample_id'),
            per_match_result.group('suffix').lower(),
        ]

    return ['', '', '']


def check_single_archive_tar_tail(per_archive_path):
    """O(1)预检单个tar是否被截断: 长度必须512字节对齐，且结尾必须有1024字节全0的EOF块

    实测616个tar(154 + 162 + 131 + 169)全部满足，说明当前数据集是完整的。
    如果下载不全，流式解包只会在读到一半时抛异常，必须在跑21T解包前先拦住。
    """
    error_message_list = []
    try:
        per_archive_size = os.path.getsize(per_archive_path)
        if per_archive_size <= 0:
            error_message_list.append(f'empty archive {per_archive_path}')

            return error_message_list

        if per_archive_size % TAR_BLOCK_SIZE != 0:
            error_message_list.append(
                f'archive size not 512 aligned {per_archive_path} {per_archive_size}'
            )

        if per_archive_size < TAR_EOF_BLOCK_SIZE:
            error_message_list.append(
                f'archive size too small {per_archive_path} {per_archive_size}'
            )

            return error_message_list

        with open(per_archive_path, 'rb') as load_archive_file:
            load_archive_file.seek(per_archive_size - TAR_EOF_BLOCK_SIZE)
            per_archive_tail_bytes = load_archive_file.read(TAR_EOF_BLOCK_SIZE)

        if per_archive_tail_bytes != b'\x00' * TAR_EOF_BLOCK_SIZE:
            error_message_list.append(
                f'archive tar eof block broken(truncated archive) {per_archive_path}'
            )
    except Exception as e:
        error_message_list.append(
            f'read archive tail failed {per_archive_path} {e}')

    return error_message_list


def check_required_subset_complete(root_dataset_path):
    """解包前预检: 子集目录、tar数量、tar编号连号、每个tar的EOF完整性

    数据集本身不完整就没必要跑几十小时解包，也避免"少了几个tar但整体报成功"。
    """
    error_message_list = []

    if not os.path.exists(root_dataset_path):
        error_message_list.append(
            f'root dataset path not exist {root_dataset_path}')

        return error_message_list

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

    archive_path_check_list = []
    for per_subset_name in SUBSET_ROOT_DIR_NAME_LIST:
        per_subset_path = os.path.join(root_dataset_path, per_subset_name)
        if not os.path.exists(per_subset_path):
            error_message_list.append(
                f'subset dir not exist {per_subset_name}')
            continue

        per_archive_name_list, per_archive_index_list = [], []
        for per_file_name in sorted(os.listdir(per_subset_path)):
            if check_skip_file_or_dir(per_file_name):
                continue

            per_file_name_prefix, per_file_name_suffix = os.path.splitext(
                per_file_name)
            if per_file_name_suffix.lower() != '.tar':
                error_message_list.append(
                    f'unknown file in subset dir {per_subset_name}/{per_file_name}'
                )
                continue

            per_archive_name_list.append(per_file_name_prefix)
            archive_path_check_list.append(
                os.path.join(per_subset_path, per_file_name))

            per_match_result = ARCHIVE_SHARD_NAME_PATTERN.match(
                per_file_name_prefix)
            if not per_match_result:
                error_message_list.append(
                    f'unknown archive name {per_subset_name}/{per_file_name}')
                continue

            per_archive_index_list.append(int(per_match_result.group('index')))

        per_expected_archive_num = EXPECTED_SUBSET_ARCHIVE_NUM_DICT[
            per_subset_name]
        print('1111', per_subset_name, 'archive:', len(per_archive_name_list),
              'expected archive:', per_expected_archive_num)

        if len(per_archive_name_list) != per_expected_archive_num:
            error_message_list.append(
                f'{per_subset_name} archive num not match {len(per_archive_name_list)} != {per_expected_archive_num}'
            )

        # tar编号必须是0..N-1连号，缺号说明有分片没下载下来
        per_missing_index_list = sorted(
            set(range(0, per_expected_archive_num)) -
            set(per_archive_index_list))
        if len(per_missing_index_list) > 0:
            error_message_list.append(
                f'{per_subset_name} archive index not continuous, missing index {per_missing_index_list[:10]}'
            )

        # 多出编号(比如编号超出0..N-1范围或重名)同样必须上报
        per_unexpected_index_list = sorted(
            set(per_archive_index_list) -
            set(range(0, per_expected_archive_num)))
        if len(per_unexpected_index_list) > 0:
            error_message_list.append(
                f'{per_subset_name} unexpected archive index {per_unexpected_index_list[:10]}'
            )

    print('1111', 'check archive tar tail:', len(archive_path_check_list))
    with Pool(processes=PROCESS_NUM) as pool:
        for per_tail_error_message_list in tqdm(
                pool.imap_unordered(check_single_archive_tar_tail,
                                    archive_path_check_list),
                total=len(archive_path_check_list)):
            error_message_list.extend(per_tail_error_message_list)

    return error_message_list


def process_single_file_copy(file_copy_pair, save_dataset_path):
    """把数据集中的非压缩包文件原样拷贝到目标目录，保持相对路径不变

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

    if os.path.getsize(save_file_path) != os.path.getsize(per_file_path):
        print('4444', per_file_path, 'copy file size not match')

        return [per_file_relative_path, 'copy file size not match']

    return [per_file_relative_path, '']


def save_single_member_bytes(save_member_path, per_member_bytes,
                             per_member_size):
    """把内存里的成员字节流落盘并立刻校验落盘大小，返回错误信息(空串表示成功)"""
    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_member_path, 'wb') as save_member_file:
            save_member_file.write(per_member_bytes)
    except Exception as e:
        return f'write member failed {save_member_path} {e}'

    if os.path.getsize(save_member_path) != per_member_size:
        return f'write member size not match {save_member_path}'

    return ''


def save_single_member_file(load_member_file, save_member_path,
                            per_member_size):
    """流式把tar成员写盘并立刻校验落盘大小，返回错误信息(空串表示成功)"""
    try:
        os.makedirs(os.path.dirname(save_member_path), exist_ok=True)
        with open(save_member_path, 'wb') as save_member_file:
            shutil.copyfileobj(load_member_file, save_member_file,
                               EXTRACT_FILE_BLOCK_SIZE)
    except Exception as e:
        return f'write member failed {save_member_path} {e}'

    if os.path.getsize(save_member_path) != per_member_size:
        return f'write member size not match {save_member_path}'

    return ''


def get_single_annotation_check_message_list(per_annotation, per_sample_id,
                                             per_source_image_member_name,
                                             per_edited_image_member_name):
    """校验单条json的有用信息是否齐备，返回[主指令, 非致命告警信息列表]

    这里产出的**全部是非致命告警**(某个可选字段为空、vqa维度数变了、
    json里记的图路径与真实成员名对不上等): 这些样本对的三张文件都在、指令也有，
    依然是完整有效的样本对，照样落盘、照样进jsonl，只是把异常显式记进报告让人能感知。
    唯一的致命情况"四条指令全空"由调用方根据返回的空主指令来隔离。
    """
    check_message_list = []

    for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            check_message_list.append(
                f'{per_sample_id} miss key {per_key_name}')

    # json里的id必须等于三元组文件名的公共prefix，不一致说明tar内成员错位
    per_annotation_sample_id = get_stripped_text_value(
        per_annotation.get(ANNOTATION_SAMPLE_ID_KEY_NAME, ''))
    if per_annotation_sample_id and per_annotation_sample_id != per_sample_id:
        check_message_list.append(
            f'{per_sample_id} json id not match {per_annotation_sample_id}')

    # json里记的两张图相对路径必须和真实成员名对得上(只校验basename，忽略顶层目录名)
    per_images = per_annotation.get(ANNOTATION_IMAGES_KEY_NAME, None)
    if not isinstance(per_images, dict):
        check_message_list.append(f'{per_sample_id} images not a dict')
    else:
        per_json_source_name = os.path.basename(
            get_stripped_text_value(
                per_images.get(ANNOTATION_IMAGES_SOURCE_KEY_NAME,
                               '')).replace('\\', '/'))
        per_json_edited_name = os.path.basename(
            get_stripped_text_value(
                per_images.get(ANNOTATION_IMAGES_EDITED_KEY_NAME,
                               '')).replace('\\', '/'))
        if per_json_source_name != per_source_image_member_name:
            check_message_list.append(
                f'{per_sample_id} json source image name not match {per_json_source_name} != {per_source_image_member_name}'
            )
        if per_json_edited_name != per_edited_image_member_name:
            check_message_list.append(
                f'{per_sample_id} json edited image name not match {per_json_edited_name} != {per_edited_image_member_name}'
            )

    per_edit_concept = per_annotation.get(ANNOTATION_EDIT_CONCEPT_KEY_NAME,
                                          None)
    if not isinstance(per_edit_concept, dict):
        check_message_list.append(f'{per_sample_id} edit_concept not a dict')
    else:
        for per_key_name in ANNOTATION_EDIT_CONCEPT_SUB_KEY_NAME_LIST:
            if not get_stripped_text_value(
                    per_edit_concept.get(per_key_name, '')):
                check_message_list.append(
                    f'{per_sample_id} empty edit_concept {per_key_name}')

    per_instruction = per_annotation.get(ANNOTATION_INSTRUCTION_KEY_NAME, None)
    if not isinstance(per_instruction, dict):
        check_message_list.append(f'{per_sample_id} instruction not a dict')
        per_instruction = {}
    else:
        for per_key_name in ANNOTATION_INSTRUCTION_SUB_KEY_NAME_LIST:
            if not get_stripped_text_value(
                    per_instruction.get(per_key_name, '')):
                check_message_list.append(
                    f'{per_sample_id} empty instruction {per_key_name}')

    per_main_instruction = ''
    for per_key_name in ANNOTATION_MAIN_INSTRUCTION_KEY_NAME_LIST:
        per_main_instruction = get_stripped_text_value(
            per_instruction.get(per_key_name, ''))
        if per_main_instruction:
            break

    per_evaluation = per_annotation.get(ANNOTATION_EVALUATION_KEY_NAME, None)
    if not isinstance(per_evaluation, dict):
        check_message_list.append(f'{per_sample_id} evaluation not a dict')
    else:
        for per_key_name in ANNOTATION_EVALUATION_SUB_KEY_NAME_LIST:
            if per_key_name not in per_evaluation:
                check_message_list.append(
                    f'{per_sample_id} miss evaluation key {per_key_name}')

        per_vqa_list = per_evaluation.get(ANNOTATION_EVALUATION_VQA_KEY_NAME,
                                          None)
        if not isinstance(per_vqa_list, list):
            check_message_list.append(f'{per_sample_id} vqa not a list')
        elif len(per_vqa_list) != EXPECTED_VQA_DIMENSION_NUM:
            # 实测恒为5个维度，多出/少掉只上报，不影响样本对本身的完整性
            check_message_list.append(
                f'{per_sample_id} vqa dimension num {len(per_vqa_list)} != {EXPECTED_VQA_DIMENSION_NUM}'
            )

    if not get_stripped_text_value(
            per_annotation.get(ANNOTATION_ORIGINAL_CAPTION_KEY_NAME, '')):
        check_message_list.append(
            f'{per_sample_id} empty {ANNOTATION_ORIGINAL_CAPTION_KEY_NAME}')

    return per_main_instruction, check_message_list


def get_single_save_annotation(
        per_annotation, per_main_instruction, per_sample_id, per_subset_name,
        per_archive_group_name, per_annotation_relative_path,
        per_source_image_relative_path, per_edited_image_relative_path,
        per_source_image_shape, per_edited_image_shape,
        per_source_image_suffix, per_edited_image_suffix):
    """把单条json整理成一行可直接训练的编辑样本对标注

    落盘路径都是相对save_dataset_path的相对路径，
    字段名与013/015/016脚本对齐(reference_image_path_list / edited_image_path / instruction),
    preprocessing2可以直接复用同一套读取口径。
    """
    per_instruction = per_annotation.get(ANNOTATION_INSTRUCTION_KEY_NAME, None)
    if not isinstance(per_instruction, dict):
        per_instruction = {}

    per_edit_concept = per_annotation.get(ANNOTATION_EDIT_CONCEPT_KEY_NAME,
                                          None)
    if not isinstance(per_edit_concept, dict):
        per_edit_concept = {}

    per_evaluation = per_annotation.get(ANNOTATION_EVALUATION_KEY_NAME, None)
    if not isinstance(per_evaluation, dict):
        per_evaluation = {}

    per_vqa_list = per_evaluation.get(ANNOTATION_EVALUATION_VQA_KEY_NAME, None)
    if not isinstance(per_vqa_list, list):
        per_vqa_list = []

    per_vqa_passed_num = 0
    per_not_passed_vqa_dimension_name_list = []
    for per_vqa in per_vqa_list:
        if not isinstance(per_vqa, dict):
            continue

        if per_vqa.get(ANNOTATION_VQA_PASSED_KEY_NAME, None) is True:
            per_vqa_passed_num += 1
            continue

        per_not_passed_vqa_dimension_name_list.append(
            get_stripped_text_value(
                per_vqa.get(ANNOTATION_VQA_DIMENSION_KEY_NAME, '')))

    per_save_annotation = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        # subset + archive + id 三段合成，天然全局唯一:
        # 同一个id会在4个子集里各出现一次但内容完全不同，只用id会互相覆盖
        'sample_key':
        f'{per_subset_name}/{per_archive_group_name}/{per_sample_id}',
        'sample_id':
        per_sample_id,
        'subset_name':
        per_subset_name,
        'archive_name':
        per_archive_group_name,
        'prompt_type':
        SUBSET_PROMPT_TYPE_DICT.get(per_subset_name, ''),
        'resolution_type':
        SUBSET_RESOLUTION_TYPE_DICT.get(per_subset_name, ''),
        'instruction':
        per_main_instruction,
        'short_en_instruction':
        get_stripped_text_value(per_instruction.get('short_en', '')),
        'short_zh_instruction':
        get_stripped_text_value(per_instruction.get('short_zh', '')),
        'detailed_en_instruction':
        get_stripped_text_value(per_instruction.get('detailed_en', '')),
        'detailed_zh_instruction':
        get_stripped_text_value(per_instruction.get('detailed_zh', '')),
        'original_simple_caption':
        get_stripped_text_value(
            per_annotation.get(ANNOTATION_ORIGINAL_CAPTION_KEY_NAME, '')),
        'edit_category':
        get_stripped_text_value(per_edit_concept.get('category', '')),
        'edit_sub_category':
        get_stripped_text_value(per_edit_concept.get('sub_category', '')),
        'edit_task':
        get_stripped_text_value(per_edit_concept.get('task', '')),
        'edit_detail':
        get_stripped_text_value(per_edit_concept.get('detail', '')),
        'overall_vqa_score':
        per_evaluation.get('overall_vqa_score', None),
        'keep':
        per_evaluation.get('keep', None),
        'wrong_count':
        per_evaluation.get('wrong_count', None),
        'recaption_prompt_en':
        get_stripped_text_value(per_evaluation.get('recaption_prompt_en', '')),
        'recaption_prompt_zh':
        get_stripped_text_value(per_evaluation.get('recaption_prompt_zh', '')),
        'vqa_dimension_num':
        len(per_vqa_list),
        'vqa_passed_num':
        per_vqa_passed_num,
        'not_passed_vqa_dimension_name_list':
        per_not_passed_vqa_dimension_name_list,
        'reference_image_path_list': [per_source_image_relative_path],
        'reference_image_num':
        1,
        'edited_image_path':
        per_edited_image_relative_path,
        'annotation_path':
        per_annotation_relative_path,
        'reference_image_shape':
        per_source_image_shape,
        'edited_image_shape':
        per_edited_image_shape,
        'reference_image_suffix':
        per_source_image_suffix,
        'edited_image_suffix':
        per_edited_image_suffix,
    }

    if SAVE_FULL_VQA_DETAIL_FLAG:
        # 需要VQA问题原文时才整份内嵌(jsonl体积会翻倍),
        # 默认不嵌也不丢信息: 原始json已原样落盘在annotation_path
        per_save_annotation['vqa_detail_list'] = per_vqa_list

    return per_save_annotation


def process_single_archive_group(archive_group, save_dataset_path,
                                 save_annotation_dir_path):
    """流式解包单个tar，同时把json成员解析成该tar的jsonl汇总标注

    落盘结构:
      unzip_images/<subset>/<archive>/<id>.json
      unzip_images/<subset>/<archive>/<id>_source.jpg
      unzip_images/<subset>/<archive>/<id>_edit.png
      unzip_annotations/<subset>/<archive>.jsonl
    tar内成员本来就带一层与tar同名的顶层目录(batch_130/xxx)，
    这里把这层目录剥掉再按 <subset>/<archive>/ 重建，避免出现 batch_130/batch_130/ 双层嵌套;
    顶层目录名与tar名不一致时会上报，但仍然按basename落盘，绝不因此丢样本。

    json只有几KB，流式解包时内容正好在手上，顺手解析出来生成汇总标注,
    比解包完再去扫1200万个小json便宜几个数量级。
    """
    per_archive_group_name, per_archive_relative_dir, per_archive_part_path_list = archive_group

    per_subset_name = per_archive_relative_dir.replace('\\', '/').split('/')[0]
    per_archive_relative_path = f'{per_archive_relative_dir}/{per_archive_group_name}'

    save_archive_dir_path = os.path.join(save_dataset_path,
                                         SAVE_IMAGE_DIR_NAME,
                                         per_archive_relative_dir,
                                         per_archive_group_name)
    save_archive_relative_dir = f'{SAVE_IMAGE_DIR_NAME}/{per_archive_relative_dir}/{per_archive_group_name}'
    os.makedirs(save_archive_dir_path, exist_ok=True)

    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_MEMBER_DIR_NAME,
                                           per_archive_relative_dir,
                                           per_archive_group_name)

    extract_file_count, skip_file_count, not_save_file_count = 0, 0, 0
    fail_file_count, duplicate_member_count = 0, 0
    total_file_member_count = 0
    unknown_suffix_member_name_list = []
    reach_tar_end = False
    error_message_list = []

    # sample_id -> [json成员名, 解析出来的dict或None]
    # 三元组凑齐后立刻生成jsonl行并把dict置None释放内存
    # (单tar最多约2万个样本对，完整json约3KB一条，不释放的话32进程会吃掉2G内存)
    annotation_dict = {}
    # sample_id -> [成员名, 宽高, 后缀]
    source_image_dict, edited_image_dict = {}, {}
    completed_sample_id_set = set()

    valid_annotation_line_list = []
    # 致命问题(json解析失败 / 四条指令全空): 这些样本对没能进jsonl，必须硬失败
    invalid_annotation_message_list = []
    # 非致命告警(可选字段为空 / vqa维度数变了 / json里的图路径与成员名对不上):
    # 样本对照样完整落盘、照样进jsonl，只上报不判失败
    annotation_warning_message_list = []
    image_shape_count_dict = collections.Counter()

    image_suffix_count_dict = collections.Counter()
    edit_category_count_dict = collections.Counter()
    edit_sub_category_count_dict = collections.Counter()
    edit_task_count_dict = collections.Counter()
    overall_vqa_score_count_dict = collections.Counter()
    keep_count_dict = collections.Counter()
    not_passed_vqa_dimension_count_dict = collections.Counter()

    def build_single_sample_pair_annotation(per_sample_id):
        """三元组凑齐后生成一行标注，并顺手统计各类分布"""
        per_annotation_member_name, per_annotation = annotation_dict[
            per_sample_id]
        per_source_member_name, per_source_image_shape, per_source_image_suffix = source_image_dict[
            per_sample_id]
        per_edited_member_name, per_edited_image_shape, per_edited_image_suffix = edited_image_dict[
            per_sample_id]

        completed_sample_id_set.add(per_sample_id)
        # json内容已经用不到了，立刻释放，只留成员名用于对账
        annotation_dict[per_sample_id] = [per_annotation_member_name, None]

        if per_annotation is None:
            invalid_annotation_message_list.append(
                f'{per_archive_relative_path}/{per_annotation_member_name} load annotation failed'
            )

            return

        per_main_instruction, per_check_message_list = get_single_annotation_check_message_list(
            per_annotation, per_sample_id, per_source_member_name,
            per_edited_member_name)
        if len(per_check_message_list) > 0:
            # 非致命告警: 样本对本身完整有效，照样落盘进jsonl，只记进warning清单
            annotation_warning_message_list.extend([
                f'{per_archive_relative_path} {per_check_message}'
                for per_check_message in per_check_message_list
            ])

        if not per_main_instruction:
            # 四条指令全空的样本对没有文本条件、不可训练，隔离上报(实测不应出现)
            invalid_annotation_message_list.append(
                f'{per_archive_relative_path}/{per_sample_id} empty instruction'
            )

            return

        per_save_annotation = get_single_save_annotation(
            per_annotation, per_main_instruction, per_sample_id,
            per_subset_name, per_archive_group_name,
            f'{save_archive_relative_dir}/{per_annotation_member_name}',
            f'{save_archive_relative_dir}/{per_source_member_name}',
            f'{save_archive_relative_dir}/{per_edited_member_name}',
            per_source_image_shape, per_edited_image_shape,
            per_source_image_suffix, per_edited_image_suffix)

        valid_annotation_line_list.append(
            json.dumps(per_save_annotation, ensure_ascii=False))

        edit_category_count_dict[per_save_annotation['edit_category']] += 1
        edit_sub_category_count_dict[
            per_save_annotation['edit_sub_category']] += 1
        edit_task_count_dict[per_save_annotation['edit_task']] += 1
        overall_vqa_score_count_dict[str(
            per_save_annotation['overall_vqa_score'])] += 1
        keep_count_dict[str(per_save_annotation['keep'])] += 1
        for per_dimension_name in per_save_annotation[
                'not_passed_vqa_dimension_name_list']:
            not_passed_vqa_dimension_count_dict[per_dimension_name] += 1
        image_suffix_count_dict[f'source{per_source_image_suffix}'] += 1
        image_suffix_count_dict[f'edited{per_edited_image_suffix}'] += 1
        if PARSE_IMAGE_SHAPE_FLAG:
            image_shape_count_dict[
                f'source_{per_source_image_shape[0]}x{per_source_image_shape[1]}'] += 1
            image_shape_count_dict[
                f'edited_{per_edited_image_shape[0]}x{per_edited_image_shape[1]}'] += 1

        return

    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    try:
        with tarfile.open(fileobj=archive_reader, mode='r|*') as load_tar_file:
            for per_member in load_tar_file:
                per_member_name = per_member.name.replace('\\',
                                                          '/').lstrip('/')
                per_member_name = os.path.normpath(per_member_name).replace(
                    '\\', '/')
                if per_member_name.startswith('..'):
                    print('5555', per_archive_group_name, per_member.name)
                    error_message_list.append(
                        f'illegal member name {per_member.name}')
                    continue

                if check_skip_file_or_dir(per_member_name):
                    continue

                if per_member.isdir():
                    # tar内只有一层与tar同名的顶层目录，落盘时被 <subset>/<archive>/ 取代，
                    # 这里不需要单独建目录
                    continue

                if not per_member.isfile():
                    # 实测只有普通文件，出现链接等类型必须显式上报
                    print('5555', per_archive_group_name, per_member.name,
                          'not a regular file')
                    error_message_list.append(
                        f'not a regular file {per_member.name}')
                    continue

                total_file_member_count += 1

                per_member_dir_name = os.path.dirname(per_member_name)
                per_member_base_name = os.path.basename(per_member_name)
                if per_member_dir_name != per_archive_group_name:
                    # 顶层目录名与tar名不一致(或成员嵌得更深)，上报后仍按basename落盘，不丢样本
                    error_message_list.append(
                        f'unexpected member dir {per_member_name}')

                per_member_kind, per_sample_id, per_member_suffix = get_member_kind_and_sample_id(
                    per_member_base_name)
                if not per_member_kind:
                    # 既不是json也不是三元组图像的成员必须上报，不能默认当图像统计
                    unknown_suffix_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'unknown member name {per_member_name}')
                    fail_file_count += 1
                    continue

                if per_member_kind == MEMBER_KIND_ANNOTATION:
                    per_member_is_duplicate = per_sample_id in annotation_dict
                elif per_member_kind == MEMBER_KIND_SOURCE_IMAGE:
                    per_member_is_duplicate = per_sample_id in source_image_dict
                else:
                    per_member_is_duplicate = per_sample_id in edited_image_dict

                if per_member_is_duplicate:
                    # 同一个tar里出现重名成员时按名写盘会互相覆盖，
                    # 这里改写到独立目录保留数据并上报，不能静默丢样本
                    duplicate_member_count += 1
                    error_message_list.append(
                        f'duplicate member name {per_member_name}')

                    save_member_path = os.path.join(
                        save_duplicate_dir_path, f'{duplicate_member_count}',
                        per_member_base_name)
                    try:
                        load_member_file = load_tar_file.extractfile(
                            per_member)
                        if load_member_file is None:
                            raise Exception('extractfile return None')

                        per_save_error_message = save_single_member_file(
                            load_member_file, save_member_path,
                            per_member.size)
                    except Exception as e:
                        per_save_error_message = f'save duplicate member failed {per_member_name} {e}'

                    if per_save_error_message:
                        print('6666', per_archive_group_name,
                              per_save_error_message)
                        error_message_list.append(per_save_error_message)
                    continue

                save_member_path = os.path.join(save_archive_dir_path,
                                                per_member_base_name)

                if per_member_kind == MEMBER_KIND_ANNOTATION:
                    # json必须读进内存用于生成汇总标注(只有几KB，代价可忽略)
                    load_member_file = load_tar_file.extractfile(per_member)
                    if load_member_file is None:
                        print('6666', per_archive_group_name, per_member.name)
                        error_message_list.append(
                            f'extract member failed {per_member.name}')
                        fail_file_count += 1
                        continue

                    per_member_bytes = load_member_file.read()
                    if len(per_member_bytes) != per_member.size:
                        error_message_list.append(
                            f'member data truncated {per_member_name} {len(per_member_bytes)} != {per_member.size}'
                        )
                        fail_file_count += 1
                        continue

                    try:
                        per_annotation = json.loads(
                            per_member_bytes.decode('UTF-8'))
                    except Exception as e:
                        error_message_list.append(
                            f'load annotation failed {per_member_name} {e}')
                        per_annotation = None

                    if per_annotation is not None and not isinstance(
                            per_annotation, dict):
                        error_message_list.append(
                            f'annotation not a dict {per_member_name}')
                        per_annotation = None

                    annotation_dict[per_sample_id] = [
                        per_member_base_name,
                        per_annotation,
                    ]

                    if os.path.exists(save_member_path) and os.path.getsize(
                            save_member_path) == per_member.size:
                        skip_file_count += 1
                    else:
                        per_save_error_message = save_single_member_bytes(
                            save_member_path, per_member_bytes,
                            per_member.size)
                        if per_save_error_message:
                            print('6666', per_archive_group_name,
                                  per_save_error_message)
                            error_message_list.append(per_save_error_message)
                            fail_file_count += 1
                        else:
                            extract_file_count += 1

                    if per_sample_id in source_image_dict and per_sample_id in edited_image_dict:
                        build_single_sample_pair_annotation(per_sample_id)

                    continue

                # 参考图 / 编辑后图成员
                per_image_shape, per_image_suffix = [0, 0], per_member_suffix
                per_member_already_saved = (
                    EXTRACT_IMAGE_FILE_FLAG
                    and os.path.exists(save_member_path)
                    and os.path.getsize(save_member_path) == per_member.size)

                if per_member_already_saved:
                    # 断点续跑: 图像字节不再过手，宽高改从已落盘文件的header里读
                    skip_file_count += 1
                    per_image_shape, per_shape_error_message = get_image_shape_from_file(
                        save_member_path)
                    if per_shape_error_message:
                        error_message_list.append(
                            f'{per_member_name} {per_shape_error_message}')
                elif not EXTRACT_IMAGE_FILE_FLAG and not PARSE_IMAGE_SHAPE_FLAG:
                    # 只建索引且不要宽高时，图像字节完全不用读
                    not_save_file_count += 1
                else:
                    load_member_file = load_tar_file.extractfile(per_member)
                    if load_member_file is None:
                        print('6666', per_archive_group_name, per_member.name)
                        error_message_list.append(
                            f'extract member failed {per_member.name}')
                        fail_file_count += 1
                        continue

                    per_member_bytes = load_member_file.read()
                    if len(per_member_bytes) != per_member.size:
                        error_message_list.append(
                            f'member data truncated {per_member_name} {len(per_member_bytes)} != {per_member.size}'
                        )
                        fail_file_count += 1
                        continue

                    per_image_suffix = get_image_bytes_suffix(
                        per_member_bytes, per_member_suffix)
                    per_image_shape, per_shape_error_message = get_image_shape(
                        per_member_bytes)
                    if per_shape_error_message:
                        error_message_list.append(
                            f'{per_member_name} {per_shape_error_message}')

                    if not EXTRACT_IMAGE_FILE_FLAG:
                        # 只建索引模式: 图像继续留在原tar里，样本对信息一样完整
                        not_save_file_count += 1
                    else:
                        per_save_error_message = save_single_member_bytes(
                            save_member_path, per_member_bytes,
                            per_member.size)
                        if per_save_error_message:
                            print('6666', per_archive_group_name,
                                  per_save_error_message)
                            error_message_list.append(per_save_error_message)
                            fail_file_count += 1
                            continue

                        extract_file_count += 1

                if per_member_kind == MEMBER_KIND_SOURCE_IMAGE:
                    source_image_dict[per_sample_id] = [
                        per_member_base_name,
                        per_image_shape,
                        per_image_suffix,
                    ]
                else:
                    edited_image_dict[per_sample_id] = [
                        per_member_base_name,
                        per_image_shape,
                        per_image_suffix,
                    ]

                if per_sample_id in annotation_dict and per_sample_id in source_image_dict and per_sample_id in edited_image_dict:
                    build_single_sample_pair_annotation(per_sample_id)

        reach_tar_end = True
    except Exception as e:
        # tar截断或NAS读失败时保留已解包出的文件，但必须上报，不能静默少样本对
        print('7777', per_archive_group_name, len(per_archive_part_path_list),
              e)
        error_message_list.append(f'read archive failed {e}')
    finally:
        archive_reader.close()

    if not reach_tar_end:
        error_message_list.append(
            'not reach tar stream end, archive may be truncated')

    if extract_file_count + skip_file_count + not_save_file_count + fail_file_count + duplicate_member_count != total_file_member_count:
        error_message_list.append(
            f'process file count not match: {extract_file_count} + {skip_file_count} + {not_save_file_count} + {fail_file_count} + {duplicate_member_count} != {total_file_member_count}'
        )

    # 三元组没凑齐的样本对必须逐条上报，绝不静默丢弃
    missing_image_sample_key_list, orphan_image_relative_path_list = [], []
    for per_sample_id in sorted(annotation_dict.keys()):
        if per_sample_id in completed_sample_id_set:
            continue

        per_missing_kind_list = []
        if per_sample_id not in source_image_dict:
            per_missing_kind_list.append(MEMBER_KIND_SOURCE_IMAGE)
        if per_sample_id not in edited_image_dict:
            per_missing_kind_list.append(MEMBER_KIND_EDITED_IMAGE)
        missing_image_sample_key_list.append(
            f'{per_archive_relative_path}/{per_sample_id} miss {per_missing_kind_list}'
        )

    for per_sample_id in sorted(
            set(source_image_dict.keys()) | set(edited_image_dict.keys())):
        if per_sample_id in annotation_dict:
            continue

        # 图有json没有: 没有编辑指令的图不可训练，只能算orphan
        orphan_image_relative_path_list.append(
            f'{per_archive_relative_path}/{per_sample_id}')

    save_annotation_path = os.path.join(save_annotation_dir_path,
                                        per_archive_relative_dir,
                                        f'{per_archive_group_name}.jsonl')
    try:
        os.makedirs(os.path.dirname(save_annotation_path), exist_ok=True)
        with open(save_annotation_path, 'w',
                  encoding='UTF-8') as save_json_file:
            for per_valid_annotation_line in valid_annotation_line_list:
                save_json_file.write(f'{per_valid_annotation_line}\n')
    except Exception as e:
        error_message_list.append(
            f'{per_archive_relative_path} save annotation failed {e}')

    per_annotation_member_count = len(annotation_dict)
    per_source_image_member_count = len(source_image_dict)
    per_edited_image_member_count = len(edited_image_dict)
    per_valid_sample_pair_count = len(valid_annotation_line_list)

    # 核心对账: tar内json数 == 参考图数 == 编辑后图数 == 有效样本对数,
    # 且成员总数刚好是样本对数的3倍。
    # 三元组是"一起丢"的，只比对落盘文件同名配对的老校验永远查不出缺样本对，
    # 必须拿tar头里数出来的成员数当ground truth。
    if per_annotation_member_count != per_source_image_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} annotation member count {per_annotation_member_count} != source image member count {per_source_image_member_count}'
        )
    if per_annotation_member_count != per_edited_image_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} annotation member count {per_annotation_member_count} != edited image member count {per_edited_image_member_count}'
        )
    if total_file_member_count % 3 != 0:
        error_message_list.append(
            f'{per_archive_relative_path} tar file member count not triple {total_file_member_count}'
        )
    if per_valid_sample_pair_count * 3 != total_file_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} valid sample pair count not match {per_valid_sample_pair_count} * 3 != {total_file_member_count}'
        )

    return {
        'archive_relative_path':
        per_archive_relative_path,
        'subset_name':
        per_subset_name,
        'archive_name':
        per_archive_group_name,
        'extract_file_count':
        extract_file_count,
        'skip_file_count':
        skip_file_count,
        'not_save_file_count':
        not_save_file_count,
        'fail_file_count':
        fail_file_count,
        'duplicate_member_count':
        duplicate_member_count,
        'total_file_member_count':
        total_file_member_count,
        'annotation_member_count':
        per_annotation_member_count,
        'source_image_member_count':
        per_source_image_member_count,
        'edited_image_member_count':
        per_edited_image_member_count,
        'valid_sample_pair_count':
        per_valid_sample_pair_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_archive_relative_dir}/{per_archive_group_name}.jsonl',
        'edit_category_count_dict':
        dict(edit_category_count_dict),
        'edit_sub_category_count_dict':
        dict(edit_sub_category_count_dict),
        'edit_task_count_dict':
        dict(edit_task_count_dict),
        'overall_vqa_score_count_dict':
        dict(overall_vqa_score_count_dict),
        'keep_count_dict':
        dict(keep_count_dict),
        'not_passed_vqa_dimension_count_dict':
        dict(not_passed_vqa_dimension_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'image_shape_count_dict':
        dict(image_shape_count_dict),
        'unknown_suffix_member_name_list':
        unknown_suffix_member_name_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'missing_image_sample_key_list':
        missing_image_sample_key_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'orphan_image_relative_path_list':
        orphan_image_relative_path_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'invalid_annotation_message_list':
        invalid_annotation_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'annotation_warning_message_list':
        annotation_warning_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'missing_image_count':
        len(missing_image_sample_key_list),
        'orphan_image_count':
        len(orphan_image_relative_path_list),
        'invalid_annotation_count':
        len(invalid_annotation_message_list),
        'annotation_warning_count':
        len(annotation_warning_message_list),
        'error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'error_message_count':
        len(error_message_list),
    }


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表和按分片归组后的压缩包列表"""
    file_copy_pair_list = []
    archive_part_path_dict = {}
    for per_root_path, per_dir_name_list, per_file_name_list in os.walk(
            root_dataset_path):
        # .cache里有660个文件，直接在遍历时剪掉整棵子树，不要走进去
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


def check_single_archive_dir_on_disk(archive_check_pair):
    """可选的二次对账: os.walk单个tar的输出目录，核对落盘文件数和三元组同名配对"""
    per_archive_relative_path, per_archive_dir_path, per_expected_file_member_count, per_expected_sample_pair_count = archive_check_pair

    error_message_list = []
    if not os.path.exists(per_archive_dir_path):
        error_message_list.append(
            f'{per_archive_relative_path} archive dir not exist')

        return [per_archive_relative_path, 0, 0, error_message_list]

    annotation_sample_id_set = set()
    source_image_sample_id_set, edited_image_sample_id_set = set(), set()
    unknown_suffix_file_count = 0
    for per_root_path, _, per_file_name_list in os.walk(per_archive_dir_path):
        for per_file_name in per_file_name_list:
            per_member_kind, per_sample_id, _ = get_member_kind_and_sample_id(
                per_file_name)
            if per_member_kind == MEMBER_KIND_ANNOTATION:
                annotation_sample_id_set.add(per_sample_id)
            elif per_member_kind == MEMBER_KIND_SOURCE_IMAGE:
                source_image_sample_id_set.add(per_sample_id)
            elif per_member_kind == MEMBER_KIND_EDITED_IMAGE:
                edited_image_sample_id_set.add(per_sample_id)
            else:
                # 既不是json也不是三元组图像的文件必须上报，不能默认当图像统计
                unknown_suffix_file_count += 1

    per_annotation_count = len(annotation_sample_id_set)
    per_image_count = len(source_image_sample_id_set) + len(
        edited_image_sample_id_set)
    per_matched_count = len(annotation_sample_id_set
                            & source_image_sample_id_set
                            & edited_image_sample_id_set)

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_archive_relative_path} unknown suffix file num {unknown_suffix_file_count}'
        )
    if per_annotation_count + per_image_count != per_expected_file_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} unzip file count not match {per_annotation_count + per_image_count} != {per_expected_file_member_count}'
        )
    if per_matched_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_relative_path} matched sample pair count not match {per_matched_count} != {per_expected_sample_pair_count}'
        )

    return [
        per_archive_relative_path,
        per_annotation_count,
        per_image_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, archive_result_list):
    """可选的二次对账: 遍历输出目录核对每个tar目录里的落盘文件数和三元组同名配对"""
    archive_check_pair_list = []
    for per_archive_result in archive_result_list:
        archive_check_pair_list.append([
            per_archive_result['archive_relative_path'],
            os.path.join(save_dataset_path, SAVE_IMAGE_DIR_NAME,
                         per_archive_result['subset_name'],
                         per_archive_result['archive_name']),
            per_archive_result['total_file_member_count'],
            per_archive_result['valid_sample_pair_count'],
        ])

    error_message_list = []
    total_annotation_count, total_image_count = 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_archive_dir_on_disk, archive_check_pair_list),
                                     total=len(archive_check_pair_list)):
            _, per_annotation_count, per_image_count, per_error_message_list = per_check_result
            total_annotation_count += per_annotation_count
            total_image_count += per_image_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk annotation:', total_annotation_count,
          'on disk image:', total_image_count)

    return error_message_list


def save_check_result(save_dataset_path, archive_result_list):
    """汇总所有tar的解包与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_file_member_count, total_valid_sample_pair_count = 0, 0
    total_extract_file_count, total_skip_file_count = 0, 0
    total_not_save_file_count, total_fail_file_count = 0, 0
    total_duplicate_member_count = 0
    total_annotation_member_count = 0
    total_source_image_member_count, total_edited_image_member_count = 0, 0
    total_missing_image_count, total_orphan_image_count = 0, 0
    total_invalid_annotation_count, total_annotation_warning_count = 0, 0
    subset_sample_pair_count_dict = collections.Counter()

    subset_archive_count_dict = collections.Counter()
    edit_category_count_dict = collections.Counter()
    edit_sub_category_count_dict = collections.Counter()
    edit_task_count_dict = collections.Counter()
    overall_vqa_score_count_dict = collections.Counter()
    keep_count_dict = collections.Counter()
    not_passed_vqa_dimension_count_dict = collections.Counter()
    image_suffix_count_dict = collections.Counter()
    image_shape_count_dict = collections.Counter()
    archive_sample_pair_count_dict = {}
    missing_image_sample_key_list, orphan_image_relative_path_list = [], []
    invalid_annotation_message_list, unknown_suffix_member_name_list = [], []
    annotation_warning_message_list = []
    error_message_list, warning_message_list = [], []

    for per_archive_result in archive_result_list:
        per_archive_relative_path = per_archive_result['archive_relative_path']

        total_file_member_count += per_archive_result[
            'total_file_member_count']
        total_valid_sample_pair_count += per_archive_result[
            'valid_sample_pair_count']
        total_extract_file_count += per_archive_result['extract_file_count']
        total_skip_file_count += per_archive_result['skip_file_count']
        total_not_save_file_count += per_archive_result['not_save_file_count']
        total_fail_file_count += per_archive_result['fail_file_count']
        total_duplicate_member_count += per_archive_result[
            'duplicate_member_count']
        total_annotation_member_count += per_archive_result[
            'annotation_member_count']
        total_source_image_member_count += per_archive_result[
            'source_image_member_count']
        total_edited_image_member_count += per_archive_result[
            'edited_image_member_count']
        total_missing_image_count += per_archive_result['missing_image_count']
        total_orphan_image_count += per_archive_result['orphan_image_count']
        total_invalid_annotation_count += per_archive_result[
            'invalid_annotation_count']
        total_annotation_warning_count += per_archive_result[
            'annotation_warning_count']

        subset_sample_pair_count_dict[per_archive_result[
            'subset_name']] += per_archive_result['valid_sample_pair_count']
        subset_archive_count_dict[per_archive_result['subset_name']] += 1
        edit_category_count_dict.update(
            per_archive_result['edit_category_count_dict'])
        edit_sub_category_count_dict.update(
            per_archive_result['edit_sub_category_count_dict'])
        edit_task_count_dict.update(per_archive_result['edit_task_count_dict'])
        overall_vqa_score_count_dict.update(
            per_archive_result['overall_vqa_score_count_dict'])
        keep_count_dict.update(per_archive_result['keep_count_dict'])
        not_passed_vqa_dimension_count_dict.update(
            per_archive_result['not_passed_vqa_dimension_count_dict'])
        image_suffix_count_dict.update(
            per_archive_result['image_suffix_count_dict'])
        image_shape_count_dict.update(
            per_archive_result['image_shape_count_dict'])
        archive_sample_pair_count_dict[
            per_archive_relative_path] = per_archive_result[
                'valid_sample_pair_count']

        missing_image_sample_key_list.extend(
            per_archive_result['missing_image_sample_key_list'])
        orphan_image_relative_path_list.extend(
            per_archive_result['orphan_image_relative_path_list'])
        invalid_annotation_message_list.extend(
            per_archive_result['invalid_annotation_message_list'])
        annotation_warning_message_list.extend(
            per_archive_result['annotation_warning_message_list'])
        unknown_suffix_member_name_list.extend([
            f'{per_archive_relative_path}/{per_member_name}'
            for per_member_name in
            per_archive_result['unknown_suffix_member_name_list']
        ])

        if per_archive_result['error_message_count'] > 0:
            print('7777', per_archive_relative_path,
                  per_archive_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_archive_relative_path} error num {per_archive_result["error_message_count"]} {per_archive_result["error_message_list"][:3]}'
            )

    for per_subset_name, per_expected_archive_num in EXPECTED_SUBSET_ARCHIVE_NUM_DICT.items(
    ):
        per_archive_count = subset_archive_count_dict.get(per_subset_name, 0)
        if per_archive_count != per_expected_archive_num:
            error_message_list.append(
                f'{per_subset_name} processed archive num {per_archive_count} != {per_expected_archive_num}'
            )

    if not EXPECTED_TOTAL_SAMPLE_PAIR_COUNT_RANGE[
            0] <= total_valid_sample_pair_count <= EXPECTED_TOTAL_SAMPLE_PAIR_COUNT_RANGE[
                1]:
        # 官方只声称12M、没给精确条数，所以这里只做软校验，打印告警不判失败
        warning_message_list.append(
            f'total sample pair count {total_valid_sample_pair_count} not in {EXPECTED_TOTAL_SAMPLE_PAIR_COUNT_RANGE}'
        )

    for per_dimension_name in not_passed_vqa_dimension_count_dict.keys():
        if per_dimension_name not in EXPECTED_VQA_DIMENSION_NAME_LIST:
            warning_message_list.append(
                f'unknown vqa dimension name {per_dimension_name}')

    print('3333', 'total tar file member:', total_file_member_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'total annotation member:', total_annotation_member_count,
          'total source image member:', total_source_image_member_count,
          'total edited image member:', total_edited_image_member_count,
          'extract:', total_extract_file_count, 'skip:', total_skip_file_count,
          'not save:', total_not_save_file_count, 'fail:',
          total_fail_file_count, 'duplicate member:',
          total_duplicate_member_count, 'missing image:',
          total_missing_image_count, 'orphan image:', total_orphan_image_count,
          'invalid annotation:', total_invalid_annotation_count,
          'annotation warning:', total_annotation_warning_count)

    print('3333', 'subset sample pair:', dict(subset_sample_pair_count_dict))
    print('3333', 'edit category:',
          dict(edit_category_count_dict.most_common(20)))
    print('3333', 'overall vqa score:', dict(overall_vqa_score_count_dict))
    print('3333', 'keep:', dict(keep_count_dict))
    print('3333', 'not passed vqa dimension:',
          dict(not_passed_vqa_dimension_count_dict))
    print('3333', 'image suffix:', dict(image_suffix_count_dict))
    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        DATASET_TASK_TYPE,
        'dataset_license_name':
        DATASET_LICENSE_NAME,
        'extract_image_file_flag':
        EXTRACT_IMAGE_FILE_FLAG,
        'parse_image_shape_flag':
        PARSE_IMAGE_SHAPE_FLAG,
        'save_full_vqa_detail_flag':
        SAVE_FULL_VQA_DETAIL_FLAG,
        'total_archive_count':
        len(archive_result_list),
        'total_tar_file_member_count':
        total_file_member_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_annotation_member_count':
        total_annotation_member_count,
        'total_source_image_member_count':
        total_source_image_member_count,
        'total_edited_image_member_count':
        total_edited_image_member_count,
        'total_extract_file_count':
        total_extract_file_count,
        'total_skip_file_count':
        total_skip_file_count,
        'total_not_save_file_count':
        total_not_save_file_count,
        'total_fail_file_count':
        total_fail_file_count,
        'total_duplicate_member_count':
        total_duplicate_member_count,
        'missing_image_count':
        total_missing_image_count,
        'orphan_image_count':
        total_orphan_image_count,
        'invalid_annotation_count':
        total_invalid_annotation_count,
        'annotation_warning_count':
        total_annotation_warning_count,
        'subset_archive_count_dict':
        dict(subset_archive_count_dict),
        'subset_sample_pair_count_dict':
        dict(subset_sample_pair_count_dict),
        'edit_category_count_dict':
        dict(edit_category_count_dict),
        'edit_sub_category_count_dict':
        dict(edit_sub_category_count_dict),
        'edit_task_count_dict':
        dict(edit_task_count_dict),
        'overall_vqa_score_count_dict':
        dict(overall_vqa_score_count_dict),
        'keep_count_dict':
        dict(keep_count_dict),
        'not_passed_vqa_dimension_count_dict':
        dict(not_passed_vqa_dimension_count_dict),
        'image_suffix_count_dict':
        dict(image_suffix_count_dict),
        'image_shape_count_dict':
        dict(
            image_shape_count_dict.most_common(MAX_SAVE_IMAGE_SHAPE_ITEM_NUM)),
        'archive_sample_pair_count_dict':
        archive_sample_pair_count_dict,
        'missing_image_sample_key_list':
        sorted(missing_image_sample_key_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'orphan_image_relative_path_list':
        sorted(orphan_image_relative_path_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'invalid_annotation_message_list':
        sorted(invalid_annotation_message_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'annotation_warning_message_list':
        sorted(annotation_warning_message_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'unknown_suffix_member_name_list':
        sorted(unknown_suffix_member_name_list)[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'warning_message_list':
        warning_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
        'check_error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    # 硬失败条件: 任何一条都意味着"有完整有用信息的样本对没有被完整处理并保存"
    if total_valid_sample_pair_count == 0:
        error_message_list.append('no valid sample pair found')
    if total_missing_image_count > 0:
        error_message_list.append(
            f'missing image count {total_missing_image_count}')
    if total_orphan_image_count > 0:
        error_message_list.append(
            f'orphan image count {total_orphan_image_count}')
    if total_invalid_annotation_count > 0:
        error_message_list.append(
            f'invalid annotation count {total_invalid_annotation_count}')
    if total_fail_file_count > 0:
        error_message_list.append(f'fail file count {total_fail_file_count}')
    if total_duplicate_member_count > 0:
        error_message_list.append(
            f'duplicate member count {total_duplicate_member_count}')
    if total_annotation_member_count != total_source_image_member_count:
        error_message_list.append(
            f'total annotation member count not match source image {total_annotation_member_count} != {total_source_image_member_count}'
        )
    if total_annotation_member_count != total_edited_image_member_count:
        error_message_list.append(
            f'total annotation member count not match edited image {total_annotation_member_count} != {total_edited_image_member_count}'
        )
    if total_valid_sample_pair_count * 3 != total_file_member_count:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} * 3 != {total_file_member_count}'
        )

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    subset_error_message_list = check_required_subset_complete(
        root_dataset_path)
    if len(subset_error_message_list) > 0:
        # 数据集本身不完整就没必要跑几十小时解包
        raise Exception(
            f'check subset failed {subset_error_message_list[:20]}')

    save_dataset_path = os.path.join(save_dataset_path,
                                     os.path.basename(root_dataset_path))
    os.makedirs(save_dataset_path, exist_ok=True)

    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    file_copy_pair_list, archive_group_list = get_all_file_and_archive_group(
        root_dataset_path)

    print('1111', len(file_copy_pair_list), len(archive_group_list))
    if len(file_copy_pair_list) > 0:
        print('1111', file_copy_pair_list[0])
    if len(archive_group_list) > 0:
        print('1111', archive_group_list[0][0], archive_group_list[0][1],
              len(archive_group_list[0][2]))

    if len(archive_group_list) != EXPECTED_TOTAL_ARCHIVE_NUM:
        raise Exception(
            f'archive group num not match {len(archive_group_list)} != {EXPECTED_TOTAL_ARCHIVE_NUM}'
        )

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

    archive_result_list = []
    extract_func = partial(process_single_archive_group,
                           save_dataset_path=save_dataset_path,
                           save_annotation_dir_path=save_annotation_dir_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_archive_result in tqdm(pool.imap_unordered(
                extract_func, archive_group_list),
                                       total=len(archive_group_list)):
            archive_result_list.append(per_archive_result)

            print('2222', per_archive_result['archive_relative_path'],
                  'extract:', per_archive_result['extract_file_count'],
                  'skip:', per_archive_result['skip_file_count'], 'not save:',
                  per_archive_result['not_save_file_count'], 'fail:',
                  per_archive_result['fail_file_count'], 'tar file member:',
                  per_archive_result['total_file_member_count'],
                  'valid sample pair:',
                  per_archive_result['valid_sample_pair_count'],
                  'duplicate member:',
                  per_archive_result['duplicate_member_count'])

    check_error_message_list = save_check_result(save_dataset_path,
                                                 archive_result_list)

    on_disk_error_message_list = []
    if CHECK_UNZIP_FILE_ON_DISK_FLAG and EXTRACT_IMAGE_FILE_FLAG:
        on_disk_error_message_list = check_unzip_file_on_disk(
            save_dataset_path, archive_result_list)

    all_error_message_list = copy_error_message_list + check_error_message_list + on_disk_error_message_list
    if len(all_error_message_list) > 0:
        # 拷贝/解包/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/ConceptEdit-12M'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
