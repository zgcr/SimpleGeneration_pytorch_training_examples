import os
import re
import json
import shutil
import tarfile
import collections

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# ==============================================================================
# 数据集: Fine-T2I(ma-xu/fine-t2i)
#
# 【数据集类型】纯文生图(text-to-image)数据集，不是图像编辑数据集。
# README原文 task_categories: text-to-image / image-to-text，
# tags: t2i / image-generation / image caption，
# 每个样本对固定是"1张目标图 + 1条prompt"，
# 没有任何参考图/输入图/编辑指令/mask/第二张视觉条件图，
# 所以下游只能走t2i_dataset.py那条链路，不能当ti2i(编辑)数据用。
#
# 【root_dataset_path实测原始保存规格(共2.2T / 6356个tar / 6363个仓库文件)】
# fine-t2i/
# ├── curated/                                     192个  train-{000000..000191}.tar (260G)
# ├── synthetic_enhanced_prompt_random_resolution/ 1621个 train-{000000..001620}.tar (476G)
# ├── synthetic_enhanced_prompt_square_resolution/ 1544个 train-{000000..001543}.tar (517G)
# ├── synthetic_original_prompt_random_resolution/ 1688个 train-{000000..001687}.tar (479G)
# ├── synthetic_original_prompt_square_resolution/ 1311个 train-{000000..001310}.tar (436G)
# ├── z_assets/       5张README示意图(teaser/distribution/aesthetic/
# │                   prompt_length/example_one_sample.jpg)，不是样本数据(无用)
# ├── README.md       数据集说明(无用)
# ├── .gitattributes  git lfs配置(无用)
# └── .cache/         huggingface下载缓存，12745个文件，残留16个*.incomplete(无用)
#
# 5个子集目录的tar编号实测**0..N-1全部连号**，每个tar都是**独立完整的webdataset**
# (不是按字节切分的分卷)，实测长度全部512字节对齐且结尾1024字节EOF块完好。
#
# 每个tar内部**没有顶层目录**，成员严格按"jpg -> json -> txt"三件套交替排列:
#   <uuid>.jpg
#   <uuid>.json
#   <uuid>.txt
# 实测每个tar的成员数 == 样本对数 * 3，每个prefix恰好出现3次，无重名、无孤立成员
# (synthetic每片恒为1000对，curated每片583~1024对不等，最后一片也不是特例);
# 抽样curated/enhanced/original三类共4600+条json，**15个字段100%齐备**、
# json里的id与文件名prefix**100%一致**、跨tar跨子集的uuid无交集;
# 抽样jpg的魔数恒为FFD8FF(JPEG)。
#
# 完整性已用huggingface下载缓存里的**官方仓库文件清单**
# .cache/huggingface/trees/*.json 对账过: 清单里共6363个文件
# (6356个tar + 5张z_assets图 + README.md + .gitattributes)，
# **磁盘上全部存在、字节数100%一致、磁盘上也没有清单之外的多余文件**，
# 所以.cache里那16个*.incomplete只是缓存残渣，数据集本体是完整的。
#
# 【单个样本对的全部成员(三件套，都是有用信息)】
#   <uuid>.jpg : **生成后图/目标图**，实测恒为JPEG                -> 有用(必需)
#   <uuid>.txt : **真正用于生成该图的那条prompt**                 -> 有用(训练主文本，必需)
#                实测 synthetic_enhanced_* 里逐字节等于enhanced_prompt,
#                synthetic_original_* 与 curated 里逐字节等于prompt,
#                与json里的image_generated_with_enhanced_prompt完全自洽。
#                README原文也是"The prompt for generating synthetic image is saved in txt"
#   <uuid>.json: 15个字段，全部有用(见下)                         -> 有用
#
# 【单条json的全部15个字段(实测抽样4600+/4600+条都齐备)】
#   id             : uuid，样本唯一id，等于文件名prefix       -> 有用(去重/对账/图像名)
#   prompt         : 原始短prompt                             -> 有用(双prompt标注之一)
#   enhanced_prompt: 增强长prompt                             -> 有用(双prompt标注之一)
#   length / enhanced_length : 两条prompt的词数               -> 有用(按长度采样/分桶)
#   prompt_generator / enhancer : 生成/增强prompt的模型名      -> 有用(来源溯源/加权)
#   style          : 11种视觉风格(Anime/General & Photorealistic等) -> 有用(风格采样/加权)
#   prompt_category: 32类prompt类别(People: Emotions等)        -> 有用(类别采样/加权)
#   task           : 任务组合list(Colors/Counting/Reasoning等，可为null) -> 有用
#   image_aspect_ratio : 宽高比(1:1 / 9:16 ...)                -> 有用(分辨率分桶)
#   image_resolution   : [w, h]                                -> 有用(分桶可不解码图像)
#   image_generator    : Z-Image-Turbo / FLUX.2-dev / unsplash_lite / pexels -> 有用
#   image_generated_with_enhanced_prompt : bool，指示txt里存的是哪条prompt -> 有用
#   aesthetic_predictor_v_2_5_score      : 美学分              -> 有用(质量过滤/加权)
# 注: curated子集(真实摄影图)的 style / prompt_category / task / image_aspect_ratio
# 实测恒为null，这是该子集的正常规格(真实图没有这些属性)，**不算缺字段、不能丢样本**，
# 本脚本按"key必须存在"校验，值为null只做分桶统计。
#
# 【无用信息(一律不整理进训练目录)】
# .cache/(12745个文件，含16个*.incomplete) / .gitattributes / README.md /
# z_assets/(5张README示意图) / .DS_Store / CACHEDIR.TAG 这类目录元数据垃圾文件。
#
# 【本脚本的处理口径: 保证每个包含完整有用信息的样本对都被处理且完整保存】
# - 解压前预检(硬失败，不过就不白跑几十小时):
#   根目录条目白名单(多出未知文件/未知子集目录立即上报)、5个子集目录必须都存在、
#   每个子集的tar数与实测ground truth逐一比对、tar名必须是train-%06d.tar且
#   编号0..N-1连号、每个tar的512对齐+1024字节EOF块(O(1)读尾部，拦下载截断)、
#   再读**官方仓库文件清单**.cache/huggingface/trees/*.json 逐个核对tar存在性与字节数
#   (这是"没漏下载任何一个tar"的最强证据，可用开关关闭);
# - 并行单位 = 单个tar(6356个任务，Pool(32))，流式tarfile r|*，绝不整包进内存;
# - json/txt只有几百字节，且流式解压时内容正好在手上，顺手解析生成
#   unzip_annotations/<subset>/<archive>.jsonl 汇总标注(保留原json全部属性),
#   比解压完再去扫1900万个小文件便宜几个数量级;
# - 图像直接写原始字节，**unzip阶段绝不引入二次编解码**，
#   resize/转格式/分辨率分桶留给preprocessing2的resave脚本;
# - 每个成员写盘后**立刻校验落盘大小 == tar头里的大小**(比只看存在性强，
#   能挡住写半截/写0字节)，已存在且大小一致就计skip，脚本可断点续跑;
# - 片内五方硬对账: jpg成员数 == json成员数 == txt成员数 == 有效样本对数、
#   总成员数 % 3 == 0、有效样本对数 * 3 == 总成员数、
#   extract + skip + not_save == 总成员数、且必须正常读到tar流结束;
#   三件套是"成对一起丢"的，所以只比对落盘文件同名配对永远查不出缺样本，
#   必须拿tar头里数出来的成员数当ground truth;
# - 单条样本校验: json里的id必须等于文件名prefix(不等说明tar内成员错位)、
#   txt必须非空、15个key逐个查缺失、txt与prompt/enhanced_prompt的匹配来源
#   与image_generated_with_enhanced_prompt交叉核对(不一致只上报告警，**绝不丢样本**);
# - 绝不静默丢样本对: 片内prefix重名时改写到unzip_duplicate_members/独立目录
#   保留数据并上报，不互相覆盖; 缺件/空txt/坏json全部分门别类记进隔离清单并汇总上报;
# - 拷贝/解压/校验任一环出错都汇总后抛异常，不再静默跑过。
#
# 【跑之前务必确认目标盘扛得住】
# - EXTRACT_IMAGE_FILE_FLAG=True 时输出小文件数约 **1900万个**
#   (约630万jpg + 630万json + 630万txt)、约2.2T，NAS上inode与元数据压力大;
# - 只想先建索引可把 EXTRACT_IMAGE_FILE_FLAG 置False，
#   图像继续留在原tar里(训练时用webdataset方式按tar顺序读)，样本对信息一样完整。
# ==============================================================================

DATASET_TASK_TYPE = 'text_to_image'

DATASET_LICENSE_NAME = 'apache-2.0'

ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.tar$'),
]

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(12745个文件，含16个*.incomplete)
# .gitattributes   git lfs配置
# README.md        数据集说明(授权信息已记进校验报告的dataset_license_name字段)
# z_assets/        README里的5张示意图(teaser/distribution/aesthetic/
#                  prompt_length/example_one_sample.jpg)，不是样本数据
# .gitignore/.DS_Store/CACHEDIR.TAG  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    'README.md',
    'z_assets',
    '.gitignore',
    '.DS_Store',
    'CACHEDIR.TAG',
]

ANNOTATION_FILE_SUFFIX = '.json'

TEXT_FILE_SUFFIX = '.txt'

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.bmp',
]

# 该数据集只有这五个子集目录(全部是train，没有val/test)
SUBSET_ROOT_DIR_NAME_LIST = [
    'curated',
    'synthetic_enhanced_prompt_random_resolution',
    'synthetic_enhanced_prompt_square_resolution',
    'synthetic_original_prompt_random_resolution',
    'synthetic_original_prompt_square_resolution',
]

# 实测tar分片数(与官方仓库文件清单一致)，数量不对说明下载不全
EXPECTED_SUBSET_ARCHIVE_NUM_DICT = {
    'curated': 192,
    'synthetic_enhanced_prompt_random_resolution': 1621,
    'synthetic_enhanced_prompt_square_resolution': 1544,
    'synthetic_original_prompt_random_resolution': 1688,
    'synthetic_original_prompt_square_resolution': 1311,
}

ARCHIVE_SHARD_NAME_PATTERN = re.compile(r'^train-(?P<index>\d{6})$')

# 每个样本对的文本提示直接存在同名.txt里(json里的两条prompt也都保留)
ANNOTATION_TEXT_KEY_NAME_LIST = [
    'prompt',
    'enhanced_prompt',
]

# 每条json里必须齐备的15个有用属性，缺失只上报不丢样本
# (实测抽样4600+条全部齐备，一旦出现缺失说明数据规格变了，必须显式感知;
#  curated子集里style/prompt_category/task/image_aspect_ratio的值恒为null，
#  这是正常规格，本脚本只校验key存在性，不校验值非空)
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    'id',
    'prompt',
    'enhanced_prompt',
    'length',
    'enhanced_length',
    'prompt_generator',
    'enhancer',
    'style',
    'prompt_category',
    'task',
    'image_aspect_ratio',
    'image_resolution',
    'image_generator',
    'image_generated_with_enhanced_prompt',
    'aesthetic_predictor_v_2_5_score',
]

# huggingface下载缓存里的官方仓库文件清单目录，
# 用来核对"6356个tar一个都没漏下载、字节数完全一致"
HUGGINGFACE_TREE_DIR_RELATIVE_PATH = '.cache/huggingface/trees'

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_DUPLICATE_MEMBER_DIR_NAME = 'unzip_duplicate_members'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

# README声明的各子集样本数，与实测tar容量对不上
# (如enhanced_random声明1615592，但1621片*1000已经大于这个数)，
# 官方没给精确的逐片条数，所以只按声明值上下浮动5%做**软校验**(打印告警)，
# 不能拿来当硬性失败条件; 真正的硬对账靠"tar头成员数 == 样本对数 * 3"
EXPECTED_SUBSET_SAMPLE_PAIR_COUNT_RANGE_DICT = {
    'curated': [160000, 177000],
    'synthetic_enhanced_prompt_random_resolution': [1534000, 1697000],
    'synthetic_enhanced_prompt_square_resolution': [1461000, 1616000],
    'synthetic_original_prompt_random_resolution': [1602000, 1771000],
    'synthetic_original_prompt_square_resolution': [1240000, 1371000],
}

# 图像成员是否落盘。
# True : 和其他数据集脚本口径一致，5个子集共约630万张jpg + 630万json + 630万txt
#        = 约1900万个小文件、2.2T，NAS上inode和元数据压力极大
# False: 只解析json/txt生成 unzip_annotations/*.jsonl 索引(几小时即可跑完)，
#        图像继续留在原tar里，训练时用webdataset方式按tar顺序读，
#        样本对信息一样是完整的。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否读官方仓库文件清单(.cache/huggingface/trees/*.json)核对每个tar的存在性与字节数。
# 默认True: 这是"没漏下载任何一个tar"的最强证据，只读6363条元信息，代价极小;
# 如果之后手工搬运过数据导致.cache不在了，把这个开关置False即可(其余预检照常生效)。
CHECK_HUGGINGFACE_TREE_FILE_FLAG = True

# 是否在解压后再os.walk一遍输出目录做二次对账。
# 默认False: 1900万个小文件的os.walk在NAS上要跑非常久，而解压时已经做了
# "写盘后立刻校验落盘大小 == tar头大小" + "extract+skip+not_save == tar成员总数"两道对账，
# 已经能保证每个成员都被处理且完整落盘。
CHECK_UNZIP_FILE_ON_DISK_FLAG = False

TAR_BLOCK_SIZE = 512

TAR_EOF_BLOCK_SIZE = 1024

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
    """过滤掉.cache、.gitattributes、README.md、z_assets等无用文件或目录"""
    per_file_relative_path = per_file_relative_path.replace('\\', '/')
    for per_path_name in per_file_relative_path.split('/'):
        if per_path_name in SKIP_FILE_OR_DIR_NAME_LIST:
            return True

    return False


def check_image_file_suffix(per_file_name):
    """判断是否是图像文件后缀"""
    per_file_suffix = os.path.splitext(per_file_name)[1].lower()

    return per_file_suffix in IMAGE_FILE_SUFFIX_LIST


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


def check_single_archive_tar_tail(per_archive_path):
    """O(1)预检单个tar是否被截断: 长度必须512字节对齐，且结尾必须有1024字节全0的EOF块

    实测6356个tar全部满足，说明当前数据集是完整的。
    如果下载不全，流式解压只会在读到一半时抛异常，必须在跑2.2T解压前先拦住。
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


def check_huggingface_tree_file(root_dataset_path):
    """读官方仓库文件清单，逐个核对仓库里的每个文件在磁盘上都存在且字节数一致

    huggingface下载时会把整个仓库的文件清单(路径 + size + lfs_sha256)缓存在
    .cache/huggingface/trees/<commit>.json，这是"6356个tar一个都没漏下载"的最强证据，
    只读6363条元信息，代价极小。
    实测清单6363个文件(6356个tar + 5张z_assets图 + README.md + .gitattributes)
    与磁盘100%一致，且磁盘上没有清单之外的多余文件。
    """
    error_message_list = []

    root_tree_path = os.path.join(root_dataset_path,
                                  HUGGINGFACE_TREE_DIR_RELATIVE_PATH)
    if not os.path.exists(root_tree_path):
        error_message_list.append(
            f'huggingface tree dir not exist {root_tree_path}')

        return error_message_list

    all_tree_file_size_dict = {}
    for per_tree_file_name in sorted(os.listdir(root_tree_path)):
        if not per_tree_file_name.endswith('.json'):
            continue

        per_tree_file_path = os.path.join(root_tree_path, per_tree_file_name)
        try:
            with open(per_tree_file_path, 'r',
                      encoding='UTF-8') as load_json_file:
                per_tree_dict = json.load(load_json_file)
        except Exception as e:
            error_message_list.append(
                f'load huggingface tree file failed {per_tree_file_path} {e}')
            continue

        for per_file_relative_path, per_file_info in per_tree_dict.get(
                'files', {}).items():
            per_file_relative_path = per_file_relative_path.replace('\\', '/')
            # 多个commit的清单同时存在时，同一路径按最大声明字节数取严
            all_tree_file_size_dict[per_file_relative_path] = max(
                all_tree_file_size_dict.get(per_file_relative_path, 0),
                int(per_file_info.get('size', 0)))

    if len(all_tree_file_size_dict) == 0:
        error_message_list.append(
            f'huggingface tree file list empty {root_tree_path}')

        return error_message_list

    tree_archive_relative_path_dict = {}
    for per_file_relative_path, per_file_size in all_tree_file_size_dict.items(
    ):
        # 清单里的README.md/.gitattributes/z_assets是无用信息，不参与完整性对账
        if check_skip_file_or_dir(per_file_relative_path):
            continue

        per_file_path = os.path.join(root_dataset_path, per_file_relative_path)
        if not os.path.exists(per_file_path):
            error_message_list.append(
                f'huggingface tree file not exist on disk {per_file_relative_path}'
            )
            continue

        if os.path.getsize(per_file_path) != per_file_size:
            error_message_list.append(
                f'huggingface tree file size not match {per_file_relative_path} {os.path.getsize(per_file_path)} != {per_file_size}'
            )
            continue

        tree_archive_relative_path_dict[per_file_relative_path] = per_file_size

    print('1111', 'huggingface tree file:', len(all_tree_file_size_dict),
          'matched archive file:', len(tree_archive_relative_path_dict))

    per_expected_archive_num = sum(EXPECTED_SUBSET_ARCHIVE_NUM_DICT.values())
    if len(tree_archive_relative_path_dict) != per_expected_archive_num:
        error_message_list.append(
            f'huggingface tree archive num not match {len(tree_archive_relative_path_dict)} != {per_expected_archive_num}'
        )

    return error_message_list


def check_required_subset_complete(root_dataset_path):
    """解压前预检: 子集目录、tar数量、tar编号连号、每个tar的EOF完整性、官方清单对账

    数据集本身不完整就没必要跑几十小时解压，也避免"少了几个tar但整体报成功"。
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

    if CHECK_HUGGINGFACE_TREE_FILE_FLAG:
        error_message_list.extend(
            check_huggingface_tree_file(root_dataset_path))

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


def get_single_annotation_error_message_list(per_annotation, per_text,
                                             per_sample_key):
    """校验单条样本的有用信息是否完整，返回(文本来源, 错误信息列表)

    txt里存的是真正用于生成该图的那条prompt，是训练主文本，必须非空;
    json的15个属性缺key只上报不丢样本(curated子集里部分属性值为null是正常规格)。
    """
    error_message_list = []

    if not per_text:
        # 文生图样本对必须有文本提示，没有prompt的图不可训练
        error_message_list.append(f'{per_sample_key} empty text prompt')

    for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            error_message_list.append(
                f'{per_sample_key} miss key {per_key_name}')

    if per_annotation.get('id', per_sample_key) != per_sample_key:
        # json里的id就是文件名prefix，不一致说明tar内成员错位
        error_message_list.append(
            f'{per_sample_key} json id not match {per_annotation.get("id", None)}'
        )

    # txt来自prompt还是enhanced_prompt，要和image_generated_with_enhanced_prompt自洽
    per_text_source_name = 'other'
    for per_text_key_name in ANNOTATION_TEXT_KEY_NAME_LIST:
        per_annotation_text = per_annotation.get(per_text_key_name, None)
        if isinstance(per_annotation_text,
                      str) and per_annotation_text.strip() == per_text:
            per_text_source_name = per_text_key_name
            break

    per_enhanced_prompt_flag = per_annotation.get(
        'image_generated_with_enhanced_prompt', None)
    per_expect_text_source_name = 'enhanced_prompt' if per_enhanced_prompt_flag else 'prompt'
    if per_text and per_text_source_name != per_expect_text_source_name:
        # 只上报不丢样本: txt本身就是训练用的那条文本，即使和json对不上也照样可训练
        error_message_list.append(
            f'{per_sample_key} text source not match {per_text_source_name} != {per_expect_text_source_name}'
        )

    return per_text_source_name, error_message_list


def process_single_archive_group(archive_group, save_dataset_path,
                                 save_annotation_dir_path):
    """流式解压单个tar，同时把json/txt成员内容解析成该tar的jsonl汇总标注

    该数据集的tar内没有顶层目录，且不同tar之间成员名(uuid)虽然实测不重复，
    但仍额外建一层以tar名命名的子目录，保证任何情况下都不会互相覆盖。

    json/txt只有几百字节，且流式解压时内容正好在手上，顺手解析出来生成汇总标注，
    比解压完再去扫1900万个小文件便宜几个数量级。
    """
    per_archive_group_name, per_archive_relative_dir, per_archive_part_path_list = archive_group

    per_archive_relative_path = f'{per_archive_relative_dir}/{per_archive_group_name}'

    save_archive_dir_path = os.path.join(save_dataset_path,
                                         per_archive_relative_dir,
                                         per_archive_group_name)
    os.makedirs(save_archive_dir_path, exist_ok=True)

    save_duplicate_dir_path = os.path.join(save_dataset_path,
                                           SAVE_DUPLICATE_MEMBER_DIR_NAME,
                                           per_archive_relative_dir,
                                           per_archive_group_name)

    extract_file_count, skip_file_count, not_save_file_count = 0, 0, 0
    total_file_member_count, duplicate_member_count = 0, 0
    unknown_suffix_member_name_list = []
    reach_tar_end = False
    error_message_list = []

    annotation_dict, text_dict, image_relative_path_dict = {}, {}, {}

    archive_reader = MultiPartArchiveReader(per_archive_part_path_list)
    try:
        with tarfile.open(fileobj=archive_reader, mode='r|*') as load_tar_file:
            for per_member in load_tar_file:
                per_member_name = per_member.name.replace('\\',
                                                          '/').lstrip('/')
                per_member_name = os.path.normpath(per_member_name)
                if per_member_name.startswith('..'):
                    print('5555', per_archive_group_name, per_member.name)
                    error_message_list.append(
                        f'illegal member name {per_member.name}')
                    continue

                if check_skip_file_or_dir(per_member_name):
                    continue

                if per_member.isdir():
                    os.makedirs(os.path.join(save_archive_dir_path,
                                             per_member_name),
                                exist_ok=True)
                    continue

                if not per_member.isfile():
                    # 实测只有普通文件，出现链接等类型必须显式上报
                    print('5555', per_archive_group_name, per_member.name,
                          'not a regular file')
                    error_message_list.append(
                        f'not a regular file {per_member.name}')
                    continue

                total_file_member_count += 1

                per_member_name_prefix, per_member_name_suffix = os.path.splitext(
                    per_member_name)
                per_member_name_prefix = per_member_name_prefix.replace(
                    '\\', '/')
                per_member_name_suffix = per_member_name_suffix.lower()

                per_member_is_annotation = per_member_name_suffix == ANNOTATION_FILE_SUFFIX
                per_member_is_text = per_member_name_suffix == TEXT_FILE_SUFFIX
                per_member_is_image = per_member_name_suffix in IMAGE_FILE_SUFFIX_LIST

                if not per_member_is_annotation and not per_member_is_text and not per_member_is_image:
                    # 既不是json/txt也不是图像的成员必须上报，不能默认当图像统计
                    unknown_suffix_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'unknown suffix member {per_member_name}')
                    continue

                per_member_is_duplicate = (
                    per_member_is_annotation
                    and per_member_name_prefix in annotation_dict) or (
                        per_member_is_text and per_member_name_prefix
                        in text_dict) or (per_member_is_image
                                          and per_member_name_prefix
                                          in image_relative_path_dict)

                if per_member_is_duplicate:
                    # 同一个tar里出现重名成员时按名写盘会互相覆盖，
                    # 这里改写到独立目录保留数据并上报，不能静默丢样本
                    duplicate_member_count += 1
                    error_message_list.append(
                        f'duplicate member name {per_member_name}')
                    save_member_path = os.path.join(
                        save_duplicate_dir_path, f'{duplicate_member_count}',
                        per_member_name)
                else:
                    save_member_path = os.path.join(save_archive_dir_path,
                                                    per_member_name)

                if per_member_is_annotation or per_member_is_text:
                    # json/txt必须读进内存用于生成汇总标注(几百字节，代价可忽略)
                    load_member_file = load_tar_file.extractfile(per_member)
                    if load_member_file is None:
                        print('6666', per_archive_group_name, per_member.name)
                        error_message_list.append(
                            f'extract member failed {per_member.name}')
                        continue

                    per_member_bytes = load_member_file.read()
                    if len(per_member_bytes) != per_member.size:
                        error_message_list.append(
                            f'member data truncated {per_member_name} {len(per_member_bytes)} != {per_member.size}'
                        )
                        continue

                    if not per_member_is_duplicate and per_member_is_annotation:
                        try:
                            per_annotation = json.loads(
                                per_member_bytes.decode('UTF-8'))
                        except Exception as e:
                            error_message_list.append(
                                f'load annotation failed {per_member_name} {e}'
                            )
                            per_annotation = None

                        if per_annotation is not None and not isinstance(
                                per_annotation, dict):
                            error_message_list.append(
                                f'annotation not a dict {per_member_name}')
                            per_annotation = None

                        annotation_dict[per_member_name_prefix] = [
                            per_member_name,
                            per_annotation,
                        ]

                    if not per_member_is_duplicate and per_member_is_text:
                        try:
                            per_text = per_member_bytes.decode('UTF-8').strip()
                        except Exception as e:
                            error_message_list.append(
                                f'load text failed {per_member_name} {e}')
                            per_text = None

                        text_dict[per_member_name_prefix] = [
                            per_member_name,
                            per_text,
                        ]

                    if os.path.exists(save_member_path) and os.path.getsize(
                            save_member_path) == per_member.size:
                        skip_file_count += 1
                        continue

                    per_save_error_message = save_single_member_bytes(
                        save_member_path, per_member_bytes, per_member.size)
                    if per_save_error_message:
                        print('6666', per_archive_group_name,
                              per_save_error_message)
                        error_message_list.append(per_save_error_message)
                        continue

                    extract_file_count += 1
                    continue

                # 图像成员
                if not per_member_is_duplicate:
                    image_relative_path_dict[
                        per_member_name_prefix] = per_member_name

                if not EXTRACT_IMAGE_FILE_FLAG:
                    # 只建索引模式: 图像继续留在原tar里，样本对信息一样完整
                    not_save_file_count += 1
                    continue

                if os.path.exists(save_member_path) and os.path.getsize(
                        save_member_path) == per_member.size:
                    skip_file_count += 1
                    continue

                load_member_file = load_tar_file.extractfile(per_member)
                if load_member_file is None:
                    print('6666', per_archive_group_name, per_member.name)
                    error_message_list.append(
                        f'extract member failed {per_member.name}')
                    continue

                per_save_error_message = save_single_member_file(
                    load_member_file, save_member_path, per_member.size)
                if per_save_error_message:
                    print('6666', per_archive_group_name,
                          per_save_error_message)
                    error_message_list.append(per_save_error_message)
                    continue

                extract_file_count += 1

        reach_tar_end = True
    except Exception as e:
        # tar截断或NAS读失败时保留已解压出的文件，但必须上报，不能静默少样本
        print('7777', per_archive_group_name, len(per_archive_part_path_list),
              e)
        error_message_list.append(f'read archive failed {e}')
    finally:
        archive_reader.close()

    if not reach_tar_end:
        error_message_list.append(
            'not reach tar stream end, archive may be truncated')

    if extract_file_count + skip_file_count + not_save_file_count != total_file_member_count:
        error_message_list.append(
            f'process file count not match: {extract_file_count} + {skip_file_count} + {not_save_file_count} != {total_file_member_count}'
        )

    # 生成该tar的汇总标注: 只有"生成后图 + 非空文本提示 + json属性"三件套齐备的
    # 样本对才算包含完整有用信息
    valid_annotation_line_list = []
    missing_image_sample_key_list, missing_text_sample_key_list = [], []
    orphan_image_relative_path_list = []
    invalid_annotation_message_list = []
    style_name_count_dict = collections.Counter()
    prompt_category_count_dict = collections.Counter()
    task_name_count_dict = collections.Counter()
    image_generator_count_dict = collections.Counter()
    text_source_count_dict = collections.Counter()

    for per_sample_key in sorted(annotation_dict.keys()):
        per_annotation_relative_path, per_annotation = annotation_dict[
            per_sample_key]

        if per_annotation is None:
            invalid_annotation_message_list.append(
                f'{per_archive_relative_path}/{per_annotation_relative_path} load annotation failed'
            )
            continue

        if per_sample_key not in image_relative_path_dict:
            # json有图没有: 样本对不完整
            missing_image_sample_key_list.append(
                f'{per_archive_relative_path}/{per_sample_key}')
            continue

        if per_sample_key not in text_dict:
            # json有txt没有: 没有文本提示的图不可训练
            missing_text_sample_key_list.append(
                f'{per_archive_relative_path}/{per_sample_key}')
            continue

        per_image_relative_path = image_relative_path_dict[per_sample_key]
        per_text_relative_path, per_text = text_dict[per_sample_key]

        per_text_source_name, per_annotation_error_message_list = get_single_annotation_error_message_list(
            per_annotation, per_text, per_sample_key)
        if len(per_annotation_error_message_list) > 0:
            invalid_annotation_message_list.extend([
                f'{per_archive_relative_path} {per_annotation_error_message}'
                for per_annotation_error_message in
                per_annotation_error_message_list
            ])

        if not per_text:
            missing_text_sample_key_list.append(
                f'{per_archive_relative_path}/{per_sample_key}')
            continue

        # 完整有用信息的样本对: 保留原json的全部15个属性(双prompt/风格/类别/任务/
        # 宽高比/分辨率/生成模型/美学分等)，再补上落盘路径、样本key、所属子集与tar，
        # 以及txt里那条真正用于生成该图的文本，方便下游直接按行取样本
        per_save_annotation = {
            'image_path':
            f'{per_archive_relative_dir}/{per_archive_group_name}/{per_image_relative_path}',
            'annotation_path':
            f'{per_archive_relative_dir}/{per_archive_group_name}/{per_annotation_relative_path}',
            'text_path':
            f'{per_archive_relative_dir}/{per_archive_group_name}/{per_text_relative_path}',
            'sample_key': per_sample_key,
            'subset_name': per_archive_relative_dir,
            'archive_name': per_archive_group_name,
            'caption': per_text,
            'caption_source_name': per_text_source_name,
        }
        for per_annotation_key, per_annotation_value in per_annotation.items():
            # json里出现同名key时不能覆盖上面刚写好的落盘路径/样本key，
            # 否则下游按image_path取图会取错，这里改名保留原值并上报
            if per_annotation_key in per_save_annotation:
                invalid_annotation_message_list.append(
                    f'{per_archive_relative_path} {per_sample_key} annotation key conflict {per_annotation_key}'
                )
                per_annotation_key = f'annotation_{per_annotation_key}'
            per_save_annotation[per_annotation_key] = per_annotation_value

        valid_annotation_line_list.append(
            json.dumps(per_save_annotation, ensure_ascii=False))

        style_name_count_dict[str(per_annotation.get('style', None))] += 1
        prompt_category_count_dict[str(
            per_annotation.get('prompt_category', None))] += 1
        image_generator_count_dict[str(
            per_annotation.get('image_generator', None))] += 1
        text_source_count_dict[per_text_source_name] += 1

        per_task_name_list = per_annotation.get('task', None)
        if isinstance(per_task_name_list, list):
            for per_task_name in per_task_name_list:
                task_name_count_dict[str(per_task_name)] += 1
        else:
            task_name_count_dict[str(per_task_name_list)] += 1

    for per_sample_key, per_image_relative_path in image_relative_path_dict.items(
    ):
        if per_sample_key not in annotation_dict or per_sample_key not in text_dict:
            # 图有但json/txt没有: 没有文本提示的图不可训练，只能算orphan
            orphan_image_relative_path_list.append(
                f'{per_archive_relative_path}/{per_image_relative_path}')

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
    per_text_member_count = len(text_dict)
    per_image_member_count = len(image_relative_path_dict)
    per_valid_sample_pair_count = len(valid_annotation_line_list)

    # 核心对账: tar内json数 == txt数 == 图数 == 有效样本对数，
    # 且成员总数刚好是样本对数的3倍。
    # 三件套是"成对一起丢"的，所以只比对落盘文件同名配对的旧校验永远查不出缺样本，
    # 必须拿tar头里数出来的成员数当ground truth。
    if per_annotation_member_count != per_image_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} annotation member count {per_annotation_member_count} != image member count {per_image_member_count}'
        )
    if per_annotation_member_count != per_text_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} annotation member count {per_annotation_member_count} != text member count {per_text_member_count}'
        )
    if total_file_member_count % 3 != 0:
        error_message_list.append(
            f'{per_archive_relative_path} tar file member count not divisible by 3 {total_file_member_count}'
        )
    if per_valid_sample_pair_count * 3 != total_file_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} valid sample pair count not match {per_valid_sample_pair_count} * 3 != {total_file_member_count}'
        )

    return {
        'archive_relative_path': per_archive_relative_path,
        'subset_name': per_archive_relative_dir,
        'archive_name': per_archive_group_name,
        'extract_file_count': extract_file_count,
        'skip_file_count': skip_file_count,
        'not_save_file_count': not_save_file_count,
        'total_file_member_count': total_file_member_count,
        'duplicate_member_count': duplicate_member_count,
        'annotation_member_count': per_annotation_member_count,
        'text_member_count': per_text_member_count,
        'image_member_count': per_image_member_count,
        'valid_sample_pair_count': per_valid_sample_pair_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_archive_relative_dir}/{per_archive_group_name}.jsonl',
        'style_name_count_dict': dict(style_name_count_dict),
        'prompt_category_count_dict': dict(prompt_category_count_dict),
        'task_name_count_dict': dict(task_name_count_dict),
        'image_generator_count_dict': dict(image_generator_count_dict),
        'text_source_count_dict': dict(text_source_count_dict),
        'unknown_suffix_member_name_list': unknown_suffix_member_name_list,
        'missing_image_sample_key_list': missing_image_sample_key_list,
        'missing_text_sample_key_list': missing_text_sample_key_list,
        'orphan_image_relative_path_list': orphan_image_relative_path_list,
        'invalid_annotation_message_list': invalid_annotation_message_list,
        'error_message_list': error_message_list,
    }


def get_all_file_and_archive_group(root_dataset_path):
    """扫描数据集，收集非压缩包文件列表和按分片归组后的压缩包列表"""
    file_copy_pair_list = []
    archive_part_path_dict = {}
    for per_root_path, per_dir_name_list, per_file_name_list in os.walk(
            root_dataset_path):
        # .cache里有12745个文件，直接在遍历时剪掉整棵子树，不要走进去
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
    """可选的二次对账: os.walk单个tar的输出目录，核对落盘文件数和同名三件套配对"""
    per_archive_relative_path, per_archive_dir_path, per_expected_file_member_count, per_expected_sample_pair_count = archive_check_pair

    error_message_list = []
    if not os.path.exists(per_archive_dir_path):
        error_message_list.append(
            f'{per_archive_relative_path} archive dir not exist')

        return [per_archive_relative_path, 0, 0, 0, error_message_list]

    annotation_name_prefix_dict, text_name_prefix_dict = {}, {}
    image_name_prefix_dict = {}
    unknown_suffix_file_count = 0
    for per_root_path, _, per_file_name_list in os.walk(per_archive_dir_path):
        for per_file_name in per_file_name_list:
            per_file_path = os.path.join(per_root_path, per_file_name)
            per_file_relative_path = os.path.relpath(per_file_path,
                                                     per_archive_dir_path)
            per_file_relative_path = per_file_relative_path.replace('\\', '/')

            per_file_name_prefix, per_file_name_suffix = os.path.splitext(
                per_file_relative_path)
            if per_file_name_suffix.lower() == ANNOTATION_FILE_SUFFIX:
                annotation_name_prefix_dict[
                    per_file_name_prefix] = per_file_relative_path
            elif per_file_name_suffix.lower() == TEXT_FILE_SUFFIX:
                text_name_prefix_dict[
                    per_file_name_prefix] = per_file_relative_path
            elif check_image_file_suffix(per_file_relative_path):
                image_name_prefix_dict[
                    per_file_name_prefix] = per_file_relative_path
            else:
                # 既不是json/txt也不是图像的文件必须上报，不能默认当图像统计
                unknown_suffix_file_count += 1

    per_annotation_count = len(annotation_name_prefix_dict)
    per_text_count = len(text_name_prefix_dict)
    per_image_count = len(image_name_prefix_dict)
    per_matched_count = len(
        set(annotation_name_prefix_dict.keys())
        & set(text_name_prefix_dict.keys())
        & set(image_name_prefix_dict.keys()))

    if unknown_suffix_file_count > 0:
        error_message_list.append(
            f'{per_archive_relative_path} unknown suffix file num {unknown_suffix_file_count}'
        )
    if per_annotation_count + per_text_count + per_image_count != per_expected_file_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} unzip file count not match {per_annotation_count + per_text_count + per_image_count} != {per_expected_file_member_count}'
        )
    if per_matched_count != per_expected_sample_pair_count:
        error_message_list.append(
            f'{per_archive_relative_path} matched sample pair count not match {per_matched_count} != {per_expected_sample_pair_count}'
        )

    return [
        per_archive_relative_path,
        per_annotation_count,
        per_text_count,
        per_image_count,
        error_message_list,
    ]


def check_unzip_file_on_disk(save_dataset_path, archive_result_list):
    """可选的二次对账: 遍历输出目录核对每个tar目录里的落盘文件数和同名三件套配对"""
    archive_check_pair_list = []
    for per_archive_result in archive_result_list:
        archive_check_pair_list.append([
            per_archive_result['archive_relative_path'],
            os.path.join(save_dataset_path, per_archive_result['subset_name'],
                         per_archive_result['archive_name']),
            per_archive_result['total_file_member_count'],
            per_archive_result['valid_sample_pair_count'],
        ])

    error_message_list = []
    total_annotation_count, total_text_count, total_image_count = 0, 0, 0
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_single_archive_dir_on_disk, archive_check_pair_list),
                                     total=len(archive_check_pair_list)):
            _, per_annotation_count, per_text_count, per_image_count, per_error_message_list = per_check_result
            total_annotation_count += per_annotation_count
            total_text_count += per_text_count
            total_image_count += per_image_count
            error_message_list.extend(per_error_message_list)

    print('3333', 'on disk annotation:', total_annotation_count,
          'on disk text:', total_text_count, 'on disk image:',
          total_image_count)

    return error_message_list


def save_check_result(save_dataset_path, archive_result_list):
    """汇总所有tar的解压与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_file_member_count, total_valid_sample_pair_count = 0, 0
    total_extract_file_count, total_skip_file_count = 0, 0
    total_not_save_file_count, total_duplicate_member_count = 0, 0
    total_annotation_member_count, total_text_member_count = 0, 0
    total_image_member_count = 0
    subset_sample_pair_count_dict = collections.Counter()
    style_name_count_dict = collections.Counter()
    prompt_category_count_dict = collections.Counter()
    task_name_count_dict = collections.Counter()
    image_generator_count_dict = collections.Counter()
    text_source_count_dict = collections.Counter()
    archive_sample_pair_count_dict = {}
    missing_image_sample_key_list, missing_text_sample_key_list = [], []
    orphan_image_relative_path_list = []
    invalid_annotation_message_list, unknown_suffix_member_name_list = [], []
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
        total_duplicate_member_count += per_archive_result[
            'duplicate_member_count']
        total_annotation_member_count += per_archive_result[
            'annotation_member_count']
        total_text_member_count += per_archive_result['text_member_count']
        total_image_member_count += per_archive_result['image_member_count']

        subset_sample_pair_count_dict[per_archive_result[
            'subset_name']] += per_archive_result['valid_sample_pair_count']
        style_name_count_dict.update(
            per_archive_result['style_name_count_dict'])
        prompt_category_count_dict.update(
            per_archive_result['prompt_category_count_dict'])
        task_name_count_dict.update(per_archive_result['task_name_count_dict'])
        image_generator_count_dict.update(
            per_archive_result['image_generator_count_dict'])
        text_source_count_dict.update(
            per_archive_result['text_source_count_dict'])
        archive_sample_pair_count_dict[
            per_archive_relative_path] = per_archive_result[
                'valid_sample_pair_count']

        missing_image_sample_key_list.extend(
            per_archive_result['missing_image_sample_key_list'])
        missing_text_sample_key_list.extend(
            per_archive_result['missing_text_sample_key_list'])
        orphan_image_relative_path_list.extend(
            per_archive_result['orphan_image_relative_path_list'])
        invalid_annotation_message_list.extend(
            per_archive_result['invalid_annotation_message_list'])
        unknown_suffix_member_name_list.extend([
            f'{per_archive_relative_path}/{per_member_name}'
            for per_member_name in
            per_archive_result['unknown_suffix_member_name_list']
        ])

        if len(per_archive_result['error_message_list']) > 0:
            print('7777', per_archive_relative_path,
                  per_archive_result['error_message_list'][:5])
            error_message_list.append(
                f'{per_archive_relative_path} error num {len(per_archive_result["error_message_list"])} {per_archive_result["error_message_list"][:3]}'
            )

    for per_subset_name, per_sample_pair_count_range in EXPECTED_SUBSET_SAMPLE_PAIR_COUNT_RANGE_DICT.items(
    ):
        per_sample_pair_count = subset_sample_pair_count_dict.get(
            per_subset_name, 0)
        if not per_sample_pair_count_range[
                0] <= per_sample_pair_count <= per_sample_pair_count_range[1]:
            # 官方README的声明条数与实测tar容量对不上，只做软校验，打印告警不判失败
            warning_message_list.append(
                f'{per_subset_name} sample pair count {per_sample_pair_count} not in {per_sample_pair_count_range}'
            )

    print('3333', 'total tar file member:', total_file_member_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'total annotation member:', total_annotation_member_count,
          'total text member:', total_text_member_count, 'total image member:',
          total_image_member_count, 'extract:', total_extract_file_count,
          'skip:', total_skip_file_count, 'not save:',
          total_not_save_file_count, 'duplicate member:',
          total_duplicate_member_count, 'missing image:',
          len(missing_image_sample_key_list), 'missing text:',
          len(missing_text_sample_key_list), 'orphan image:',
          len(orphan_image_relative_path_list), 'invalid annotation:',
          len(invalid_annotation_message_list))
    print('3333', 'subset sample pair:', dict(subset_sample_pair_count_dict))
    print('3333', 'text source:', dict(text_source_count_dict))
    print('3333', 'image generator:', dict(image_generator_count_dict))
    print('3333', 'style name:', dict(style_name_count_dict))
    print('3333', 'prompt category:', dict(prompt_category_count_dict))
    print('3333', 'task name:', dict(task_name_count_dict))
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
        'total_archive_count':
        len(archive_result_list),
        'total_tar_file_member_count':
        total_file_member_count,
        'total_valid_sample_pair_count':
        total_valid_sample_pair_count,
        'total_annotation_member_count':
        total_annotation_member_count,
        'total_text_member_count':
        total_text_member_count,
        'total_image_member_count':
        total_image_member_count,
        'total_extract_file_count':
        total_extract_file_count,
        'total_skip_file_count':
        total_skip_file_count,
        'total_not_save_file_count':
        total_not_save_file_count,
        'total_duplicate_member_count':
        total_duplicate_member_count,
        'missing_image_count':
        len(missing_image_sample_key_list),
        'missing_text_count':
        len(missing_text_sample_key_list),
        'orphan_image_count':
        len(orphan_image_relative_path_list),
        'invalid_annotation_count':
        len(invalid_annotation_message_list),
        'subset_sample_pair_count_dict':
        dict(subset_sample_pair_count_dict),
        'text_source_count_dict':
        dict(text_source_count_dict),
        'image_generator_count_dict':
        dict(image_generator_count_dict),
        'style_name_count_dict':
        dict(style_name_count_dict),
        'prompt_category_count_dict':
        dict(prompt_category_count_dict),
        'task_name_count_dict':
        dict(task_name_count_dict),
        'archive_sample_pair_count_dict':
        archive_sample_pair_count_dict,
        'missing_image_sample_key_list':
        sorted(missing_image_sample_key_list)[:10000],
        'missing_text_sample_key_list':
        sorted(missing_text_sample_key_list)[:10000],
        'orphan_image_relative_path_list':
        sorted(orphan_image_relative_path_list)[:10000],
        'invalid_annotation_message_list':
        sorted(invalid_annotation_message_list)[:10000],
        'unknown_suffix_member_name_list':
        sorted(unknown_suffix_member_name_list)[:10000],
        'warning_message_list':
        warning_message_list,
        'check_error_message_list':
        error_message_list[:10000],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    if total_valid_sample_pair_count == 0:
        error_message_list.append('no valid sample pair found')
    if len(missing_image_sample_key_list) > 0:
        error_message_list.append(
            f'missing image count {len(missing_image_sample_key_list)}')
    if len(missing_text_sample_key_list) > 0:
        error_message_list.append(
            f'missing text count {len(missing_text_sample_key_list)}')
    if len(orphan_image_relative_path_list) > 0:
        error_message_list.append(
            f'orphan image count {len(orphan_image_relative_path_list)}')
    if len(invalid_annotation_message_list) > 0:
        error_message_list.append(
            f'invalid annotation count {len(invalid_annotation_message_list)}')
    if total_valid_sample_pair_count * 3 != total_file_member_count:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} * 3 != {total_file_member_count}'
        )

    return error_message_list


def preprocess_dataset(root_dataset_path, save_dataset_path):
    subset_error_message_list = check_required_subset_complete(
        root_dataset_path)
    if len(subset_error_message_list) > 0:
        # 数据集本身不完整就没必要跑几十小时解压
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

    expected_archive_group_num = sum(EXPECTED_SUBSET_ARCHIVE_NUM_DICT.values())
    if len(archive_group_list) != expected_archive_group_num:
        raise Exception(
            f'archive group num not match {len(archive_group_list)} != {expected_archive_group_num}'
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
                  per_archive_result['not_save_file_count'],
                  'tar file member:',
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
        # 拷贝/解压/校验任一环出错都必须让上层感知，不能静默少样本对
        raise Exception(
            f'preprocess dataset error num {len(all_error_message_list)} {all_error_message_list[:20]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/fine-t2i'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
