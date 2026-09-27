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
# 数据集: GPIC(Giant Permissive Image Corpus, stanford-vision-lab)
#
# 【数据集类型】纯文生图(text-to-image)数据集，不是图像编辑数据集。
# 每个样本对只有"一张目标图 + 一条caption"，没有任何参考图/输入图/编辑指令/mask，
# 所以下游只能用t2i_dataset.py那条链路，不能当ti2i(编辑)数据用。
#
# 【root_dataset_path实测原始保存规格】
# gpic/
# ├── train/                8000个 gpic_train_{00000-07999}.tar  (12T，编号0..7999连续无缺号)
# │   └── test.txt          14字节的"Hello, world!"，作者上传时留下的垃圾文件(无用)
# ├── val/                  32个   gpic_val_{00000-00031}.tar    (24G)
# ├── test/                 128个  gpic_test_{00000-00127}.tar   (119G)
# ├── reference_stats/      5个669MB的npz(full/lite/nano/test/val)，是FID等指标的参考统计量，
# │                         不是样本数据(无用，共3.2G，原样拷贝纯属浪费NAS空间)
# ├── figures/              gpic_graphics.svg / gpic_stats.jpeg，README里的示意图(无用)
# ├── README.md             数据集说明(无用)
# ├── .gitattributes        git lfs配置(无用)
# └── .cache/               huggingface下载缓存，16351个文件，里面还残留8个*.incomplete(无用)
#
# 每个tar内部是webdataset规格，没有顶层目录，成员按"json在前、图在后"交替排列:
#   <sha256>.json
#   <sha256>.jpg 或 <sha256>.png
# 实测train_00000: 12639个json + 11042个jpg + 1597个png = 12639个样本对，成员数刚好偶数，
# 每个prefix正好出现2次(1个json+1个图)，无重名、无孤立成员；
# 抽样的train/val/test共26793条json全部字段齐备、caption全部非空、json里的key字段
# 与文件名prefix 100%一致；跨tar(train0/train1/val0/test0)的key无交集。
#
# 【单条json的全部字段(实测26793/26793条都齐备)】
#   caption      : 文本提示描述(唯一的训练文本，必需)          -> 有用
#   caption_type : tag/short/medium/long，caption粒度           -> 有用(可做caption采样/加权)
#   split        : ["nano"]/["lite"]/["full"]/["val"]/["test"]  -> 有用(train内部还分nano/lite/full
#                  三档规模子集，实测shard0-1是nano、shard100/500是lite、shard1000+是full，
#                  只有这个字段能区分，丢了就没法只训小规模子集)
#   img_width / img_height : 原图宽高                            -> 有用(分辨率分桶可不解码图像)
#   key          : sha256，样本唯一id，等于文件名prefix           -> 有用(去重/对账)
#   license / license_url / attribution / retrieved_at : 授权与来源 -> 有用(合规溯源，需保留)
# 生成后图: 与json同名的 <key>.jpg / <key>.png (一张，无参考图)   -> 有用
#
# 【原脚本(修改前)的问题，两个问题的答案都是"否"】
# 1) 没有任何汇总标注: 只把tar原样摊成小文件，100M个json+100M张图共约2亿个小文件，
#    校验也只比对"同名prefix配对"，从来没打开过json。
#    后果: caption/caption_type/split/img_width/img_height/license这些"单个样本的其他所有属性"
#    虽然文件还在，但没有被整理成可训练的样本对索引，dataloader要扫2亿个小文件才能拿到caption，
#    等于"有用信息没有被完整保存下来"。
# 2) 会静默丢样本对且查不出来:
#    a. tarfile读取异常只print('7777')，preprocess_dataset不抛异常、不返回非零码，
#       某个tar截断/NAS读失败时那几千个样本对直接消失，主流程照常"成功"；
#    b. 解压前没有任何预检(8000个tar是否连号、每个tar的1024字节EOF块是否完整)，
#       而.cache里实测残留8个*.incomplete，说明下载确实中断过，风险是真实存在的；
#    c. 没有和tar头里的真实成员数对账(extract+skip == member),
#       json和图是"成对一起丢"的，所以只比对落盘文件的同名配对校验永远发现不了缺样本;
#    d. check_single_subset_dir把任何非.json文件都当图统计，且靠os.listdir(save/train)
#       取8000个tar目录，多出来的垃圾文件会污染统计;
#    e. figures/reference_stats/README这些无用信息被原样拷贝(3.2G)。
#
# 【修改后】
# - 解压前预检: 子集目录、tar数量、tar编号连号、每个tar的512对齐+1024字节EOF块(O(1)读尾部);
# - 解压时顺手把json成员内容读出来(json只有几百字节，且流式解压时正好在手上，不需要二次扫盘),
#   按tar落一份 unzip_annotations/<subset>/<archive>.jsonl 汇总标注，保留原json全部属性;
# - 每个tar内部严格对账: json成员数 == 图成员数 == 样本对数，
#   且 extract+skip+not_save == tar头里的成员总数，且必须正常读到tar流结束;
# - 每个成员写盘后立即校验落盘大小 == tar头里的大小(避免二次os.walk 2亿个文件);
# - 无用信息(.cache/.gitattributes/README.md/figures/reference_stats/train/test.txt)全部不拷贝;
# - 任何一环出错都汇总后抛异常并sys.exit(1)，不再静默跑过。
# ==============================================================================

ARCHIVE_FILE_NAME_PATTERN_LIST = [
    re.compile(r'^(?P<prefix>.+)\.tar$'),
]

# 无用信息，不整理进训练目录:
# .cache/          huggingface下载缓存(16351个文件，含8个*.incomplete)
# .gitattributes   git lfs配置
# README.md        数据集说明
# figures/         README里的示意图(gpic_graphics.svg / gpic_stats.jpeg)
# reference_stats/ FID等指标的参考统计npz(5*669MB)，评测用，不是样本数据；
#                  如果之后要跑官方评测协议，单独拷这个目录，不要混进训练数据目录
# test.txt         train/目录下14字节的"Hello, world!"，作者上传时留下的垃圾文件
# .DS_Store/CACHEDIR.TAG  目录元数据垃圾文件
SKIP_FILE_OR_DIR_NAME_LIST = [
    '.cache',
    '.gitattributes',
    'README.md',
    'figures',
    'reference_stats',
    'test.txt',
    '.DS_Store',
    'CACHEDIR.TAG',
]

ANNOTATION_FILE_SUFFIX = '.json'

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.png',
    '.webp',
    '.bmp',
]

# 该数据集只有这三个子集目录
SUBSET_ROOT_DIR_NAME_LIST = [
    'train',
    'val',
    'test',
]

# 实测tar分片数(README声明值和实测值一致)，数量不对说明下载不全
EXPECTED_SUBSET_ARCHIVE_NUM_DICT = {
    'train': 8000,
    'val': 32,
    'test': 128,
}

ARCHIVE_SHARD_NAME_PATTERN = re.compile(
    r'^gpic_(?P<subset>train|val|test)_(?P<index>\d{5})$')

# 每条json里的文本提示字段(文生图数据集没有参考图，生成后图靠同名文件配对)
ANNOTATION_TEXT_KEY_NAME_LIST = [
    'caption',
]

# 每条json里除caption外必须齐备的有用属性，缺失只上报不丢样本
# (实测26793/26793条全部齐备，一旦出现缺失说明数据规格变了，必须显式感知)
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    'key',
    'caption_type',
    'split',
    'img_width',
    'img_height',
    'license',
    'license_url',
    'attribution',
    'retrieved_at',
]

SAVE_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_DUPLICATE_MEMBER_DIR_NAME = 'unzip_duplicate_members'

SAVE_CHECK_RESULT_FILE_NAME = 'unzip_check_missing_images.json'

# README声明train约100M/val约200K/test约1M，按实测每个tar约12500/6100/7900条换算成宽松区间，
# 只做软校验(打印告警)，因为官方没给精确条数，不能拿来当硬性失败条件
EXPECTED_SUBSET_SAMPLE_PAIR_COUNT_RANGE_DICT = {
    'train': [95000000, 105000000],
    'val': [190000, 215000],
    'test': [950000, 1060000],
}

# 图像成员是否落盘。
# True : 和其他数据集脚本口径一致，train+val+test共约1亿张图 + 1亿个json = 约2亿个小文件、12T，
#        NAS上inode和元数据压力极大，务必确认目标盘扛得住再跑;
# False: 只解析json生成 unzip_annotations/*.jsonl 索引(几小时即可跑完)，图像继续留在原tar里，
#        训练时用webdataset方式按tar顺序读，样本对信息一样是完整的。
EXTRACT_IMAGE_FILE_FLAG = True

# 是否在解压后再os.walk一遍输出目录做二次对账。
# 默认False: 2亿个小文件的os.walk在NAS上要跑非常久，而解压时已经做了
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
    """过滤掉.cache、README.md、figures、reference_stats、test.txt等无用文件或目录"""
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

    实测8160个tar(8000 train + 32 val + 128 test)全部满足，说明当前数据集是完整的。
    如果下载不全，流式解压只会在读到一半时抛异常，必须在跑12T解压前先拦住。
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
    """解压前预检: 子集目录、tar数量、tar编号连号、每个tar的EOF完整性

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
            # train/test.txt这种垃圾文件在这里被跳过，不算未知文件
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

            if per_match_result.group('subset') != per_subset_name:
                error_message_list.append(
                    f'archive subset name not match {per_subset_name}/{per_file_name}'
                )
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

    该数据集过滤掉无用信息后这里基本是空列表，保留这一步只是为了兼容后续新增文件。
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


def get_single_annotation_error_message_list(per_annotation, per_sample_key):
    """校验单条json的有用信息是否完整: caption必须非空，其余有用属性缺失只上报"""
    error_message_list = []

    per_caption = ''
    for per_text_key_name in ANNOTATION_TEXT_KEY_NAME_LIST:
        per_caption = per_annotation.get(per_text_key_name, '')
        if isinstance(per_caption, str) and len(per_caption.strip()) > 0:
            break
        per_caption = ''

    if not per_caption:
        # 文生图样本对必须有文本提示，没有caption的图不可训练
        error_message_list.append(f'{per_sample_key} empty caption')

    for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            error_message_list.append(
                f'{per_sample_key} miss key {per_key_name}')

    if per_annotation.get('key', per_sample_key) != per_sample_key:
        # json里的key就是文件名prefix，不一致说明tar内成员错位
        error_message_list.append(
            f'{per_sample_key} json key not match {per_annotation.get("key", None)}'
        )

    return per_caption, error_message_list


def process_single_archive_group(archive_group, save_dataset_path,
                                 save_annotation_dir_path):
    """流式解压单个tar，同时把json成员内容解析成该tar的jsonl汇总标注

    该数据集的tar内没有顶层目录，且不同tar之间成员名(sha256)虽然实测不重复，
    但仍额外建一层以tar名命名的子目录，保证任何情况下都不会互相覆盖。

    json只有几百字节，且流式解压时内容正好在手上，顺手解析出来生成汇总标注，
    比解压完再去扫1亿个小json便宜几个数量级。
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

    annotation_dict, image_relative_path_dict = {}, {}

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
                per_member_is_image = per_member_name_suffix in IMAGE_FILE_SUFFIX_LIST

                if not per_member_is_annotation and not per_member_is_image:
                    # 既不是json也不是图像的成员必须上报，不能默认当图像统计
                    unknown_suffix_member_name_list.append(per_member_name)
                    error_message_list.append(
                        f'unknown suffix member {per_member_name}')
                    continue

                per_member_is_duplicate = (
                    per_member_is_annotation
                    and per_member_name_prefix in annotation_dict) or (
                        per_member_is_image
                        and per_member_name_prefix in image_relative_path_dict)

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

                if per_member_is_annotation:
                    # json必须读进内存用于生成汇总标注(几百字节，代价可忽略)
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

                    if not per_member_is_duplicate:
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

    # 生成该tar的汇总标注: 只有"生成后图 + 非空caption"齐备的样本对才算完整有用信息
    valid_annotation_line_list = []
    missing_image_sample_key_list, orphan_image_relative_path_list = [], []
    invalid_annotation_message_list = []
    caption_type_count_dict = collections.Counter()
    split_name_count_dict = collections.Counter()

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

        per_image_relative_path = image_relative_path_dict[per_sample_key]

        per_caption, per_annotation_error_message_list = get_single_annotation_error_message_list(
            per_annotation, per_sample_key)
        if len(per_annotation_error_message_list) > 0:
            invalid_annotation_message_list.extend([
                f'{per_archive_relative_path} {per_annotation_error_message}'
                for per_annotation_error_message in
                per_annotation_error_message_list
            ])

        if not per_caption:
            continue

        # 完整有用信息的样本对: 保留原json的全部属性(caption_type/split/宽高/授权溯源等),
        # 再补上落盘路径、样本key、所属子集与tar，方便下游直接按行取样本
        per_save_annotation = {
            'image_path':
            f'{per_archive_relative_dir}/{per_archive_group_name}/{per_image_relative_path}',
            'annotation_path':
            f'{per_archive_relative_dir}/{per_archive_group_name}/{per_annotation_relative_path}',
            'sample_key': per_sample_key,
            'subset_name': per_archive_relative_dir,
            'archive_name': per_archive_group_name,
            'caption': per_caption,
        }
        for per_annotation_key, per_annotation_value in per_annotation.items():
            if per_annotation_key in ANNOTATION_TEXT_KEY_NAME_LIST:
                continue
            per_save_annotation[per_annotation_key] = per_annotation_value

        valid_annotation_line_list.append(
            json.dumps(per_save_annotation, ensure_ascii=False))

        caption_type_count_dict[str(per_annotation.get('caption_type',
                                                       None))] += 1
        per_split_name_list = per_annotation.get('split', None)
        if isinstance(per_split_name_list, list):
            for per_split_name in per_split_name_list:
                split_name_count_dict[str(per_split_name)] += 1
        else:
            split_name_count_dict[str(per_split_name_list)] += 1

    for per_sample_key, per_image_relative_path in image_relative_path_dict.items(
    ):
        if per_sample_key not in annotation_dict:
            # 图有json没有: 没有文本提示的图不可训练，只能算orphan
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
    per_image_member_count = len(image_relative_path_dict)
    per_valid_sample_pair_count = len(valid_annotation_line_list)

    # 核心对账: tar内json数 == 图数 == 有效样本对数，且成员总数刚好是样本对数的2倍。
    # json和图是"成对一起丢"的，所以只比对落盘文件同名配对的旧校验永远查不出缺样本，
    # 必须拿tar头里数出来的成员数当ground truth。
    if per_annotation_member_count != per_image_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} annotation member count {per_annotation_member_count} != image member count {per_image_member_count}'
        )
    if total_file_member_count % 2 != 0:
        error_message_list.append(
            f'{per_archive_relative_path} tar file member count not even {total_file_member_count}'
        )
    if per_valid_sample_pair_count * 2 != total_file_member_count:
        error_message_list.append(
            f'{per_archive_relative_path} valid sample pair count not match {per_valid_sample_pair_count} * 2 != {total_file_member_count}'
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
        'image_member_count': per_image_member_count,
        'valid_sample_pair_count': per_valid_sample_pair_count,
        'save_annotation_relative_path':
        f'{SAVE_ANNOTATION_DIR_NAME}/{per_archive_relative_dir}/{per_archive_group_name}.jsonl',
        'caption_type_count_dict': dict(caption_type_count_dict),
        'split_name_count_dict': dict(split_name_count_dict),
        'unknown_suffix_member_name_list': unknown_suffix_member_name_list,
        'missing_image_sample_key_list': missing_image_sample_key_list,
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
        # .cache里有16351个文件，直接在遍历时剪掉整棵子树，不要走进去
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
    """可选的二次对账: os.walk单个tar的输出目录，核对落盘文件数和同名配对"""
    per_archive_relative_path, per_archive_dir_path, per_expected_file_member_count, per_expected_sample_pair_count = archive_check_pair

    error_message_list = []
    if not os.path.exists(per_archive_dir_path):
        error_message_list.append(
            f'{per_archive_relative_path} archive dir not exist')

        return [per_archive_relative_path, 0, 0, error_message_list]

    annotation_name_prefix_dict, image_name_prefix_dict = {}, {}
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
            elif check_image_file_suffix(per_file_relative_path):
                image_name_prefix_dict[
                    per_file_name_prefix] = per_file_relative_path
            else:
                # 既不是json也不是图像的文件必须上报，不能默认当图像统计
                unknown_suffix_file_count += 1

    per_annotation_count = len(annotation_name_prefix_dict)
    per_image_count = len(image_name_prefix_dict)
    per_matched_count = len(
        set(annotation_name_prefix_dict.keys())
        & set(image_name_prefix_dict.keys()))

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
    """可选的二次对账: 遍历输出目录核对每个tar目录里的落盘文件数和同名配对"""
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
    """汇总所有tar的解压与校验结果，落盘一份校验报告并返回错误信息列表"""
    total_file_member_count, total_valid_sample_pair_count = 0, 0
    total_extract_file_count, total_skip_file_count = 0, 0
    total_not_save_file_count, total_duplicate_member_count = 0, 0
    total_annotation_member_count, total_image_member_count = 0, 0
    subset_sample_pair_count_dict = collections.Counter()
    caption_type_count_dict = collections.Counter()
    split_name_count_dict = collections.Counter()
    archive_sample_pair_count_dict = {}
    missing_image_sample_key_list, orphan_image_relative_path_list = [], []
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
        total_image_member_count += per_archive_result['image_member_count']

        subset_sample_pair_count_dict[per_archive_result[
            'subset_name']] += per_archive_result['valid_sample_pair_count']
        caption_type_count_dict.update(
            per_archive_result['caption_type_count_dict'])
        split_name_count_dict.update(
            per_archive_result['split_name_count_dict'])
        archive_sample_pair_count_dict[
            per_archive_relative_path] = per_archive_result[
                'valid_sample_pair_count']

        missing_image_sample_key_list.extend(
            per_archive_result['missing_image_sample_key_list'])
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
            # 官方没给精确条数，只按README规模做软校验，打印告警不判失败
            warning_message_list.append(
                f'{per_subset_name} sample pair count {per_sample_pair_count} not in {per_sample_pair_count_range}'
            )

    print('3333', 'total tar file member:', total_file_member_count,
          'total valid sample pair:', total_valid_sample_pair_count,
          'total annotation member:', total_annotation_member_count,
          'total image member:', total_image_member_count, 'extract:',
          total_extract_file_count, 'skip:', total_skip_file_count,
          'not save:', total_not_save_file_count, 'duplicate member:',
          total_duplicate_member_count, 'missing image:',
          len(missing_image_sample_key_list), 'orphan image:',
          len(orphan_image_relative_path_list), 'invalid annotation:',
          len(invalid_annotation_message_list))
    print('3333', 'subset sample pair:', dict(subset_sample_pair_count_dict))
    print('3333', 'split name:', dict(split_name_count_dict))
    print('3333', 'caption type:', dict(caption_type_count_dict))
    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type':
        'text_to_image',
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
        'orphan_image_count':
        len(orphan_image_relative_path_list),
        'invalid_annotation_count':
        len(invalid_annotation_message_list),
        'subset_sample_pair_count_dict':
        dict(subset_sample_pair_count_dict),
        'split_name_count_dict':
        dict(split_name_count_dict),
        'caption_type_count_dict':
        dict(caption_type_count_dict),
        'archive_sample_pair_count_dict':
        archive_sample_pair_count_dict,
        'missing_image_sample_key_list':
        sorted(missing_image_sample_key_list)[:10000],
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
    if len(orphan_image_relative_path_list) > 0:
        error_message_list.append(
            f'orphan image count {len(orphan_image_relative_path_list)}')
    if len(invalid_annotation_message_list) > 0:
        error_message_list.append(
            f'invalid annotation count {len(invalid_annotation_message_list)}')
    if total_valid_sample_pair_count * 2 != total_file_member_count:
        error_message_list.append(
            f'total valid sample pair count not match {total_valid_sample_pair_count} * 2 != {total_file_member_count}'
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets/gpic'
    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip'
    preprocess_dataset(root_dataset_path, save_dataset_path)
