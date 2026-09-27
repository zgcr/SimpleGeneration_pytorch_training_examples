import os
import re
import json
import numpy as np
import cv2

from fractions import Fraction
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'imgedit'

SAVE_DATASET_DIR_NAME = 'ImgEdit'

# 上游006解压脚本把37个parquet解成annotations/<parquet名>.jsonl，每行是一个完整的
# 图像编辑对(1~2张参考图 + 1张编辑后图 + 1条编辑指令)，图像统一落盘到
# images/<subset_name>/<样本目录>/<文件名>，标注里的图像路径就是相对images的规范路径
LOAD_ANNOTATION_DIR_NAME_LIST = [
    'annotations',
]

LOAD_IMAGE_DIR_NAME_LIST = [
    'images',
]

# 上游还按子集另存了image_index/<subset_name>.txt(35个文件、合计407万行)，
# 每行就是一条已落盘图像的规范相对路径，和标注里的路径完全同构。
# 本脚本用它做图像存在性判定，不用os.path.exists也不用os.listdir:
# ImgEdit是"一个样本目录只放一个样本的2~4张图"的结构(36万个样本目录)，
# 按目录缓存listdir会退化成36万次NAS网络往返，而读索引只要35次顺序读
LOAD_IMAGE_INDEX_DIR_NAME_LIST = [
    'image_index',
]

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

LOAD_IMAGE_INDEX_FILE_NAME_SUFFIX = '.txt'

# 三个多轮子集(合计314208个编辑对，占全量21.1%)按方案确认整体不处理:
# content_memory_part2(61722对/2轮): 每行自带global_prompt全局约束(实测61722条
#     全非空，形如"All subsequent edits must use vibrant tropical fruits")，
#     丢掉全局约束后指令不完整;
# content_understanding_part2(126417对/3轮): 后续轮指令全是代词指代，
#     turn1形如"adjust its glass panels to..."、turn2有11921条就是"remove it"，
#     脱离history_prompt_list根本无法执行;
# version_backtracking_part0(126069对/3轮): turn2指令直接引用轮次，形如
#     "withdraw the previous round of modifications, adjust the green vest in round1"。
# 新标注只保留单条ti2i_caption、不保留global_prompt/history_prompt_list，
# 所以这三个子集必须在这一步整体跳过，不能只靠指令长度过滤兜底
SKIP_MULTITURN_PARQUET_NAME_LIST = [
    'content_memory_part2',
    'content_understanding_part2',
    'version_backtracking_part0',
]

# 上游标注里的parquet名，形如add_part0/style_transfer，去掉_partN后缀就是任务类型。
# 注意不能用标注里的task_type字段: 实测1492359条全是"image_edit"这一个值，没有区分力;
# 也不能用metadata里的result.edit_type: style_transfer两个分片全是null、
# action四个分片的metadata文件是0字节(上游没解出result.json)，覆盖不全
ANNOTATION_PARQUET_NAME_KEY_NAME = 'parquet_name'

# 上游图像落盘用的子集目录名(如results_add_laion_part0)，只用来定位对应的图像索引文件，
# 不参与新数据集的子集划分(它带分片号且和parquet名大面积不一致)
ANNOTATION_SUBSET_NAME_KEY_NAME = 'subset_name'

# 全局唯一样本键，形如add_part0_00000000、style_transfer_00000000。
# 图像名前缀必须用它: 上游图像原始名前缀只有十几种(编辑后图是
# result/result_0/result_1/result_2，参考图是original/origin_0/origin_1/result_2)，
# 直接用原始名前缀会让117万对塌缩成十几个同名文件夹、绝大部分样本被覆盖丢失;
# 样本目录名(如00049_00040_000401278)也不行，实测唯一值只有364429个、跨子集大面积重名。
# sample_id本身已内含parquet名，所以保存图像名里不再重复拼子集名
# (实测单轮1178151对拼出1178151个唯一编辑后图像名、0重名)
ANNOTATION_SAMPLE_ID_KEY_NAME = 'sample_id'

# 编辑后的图像，实测单轮1178151条全非空(其中png 1333351张、jpg 159008张，
# 本脚本统一重编码成jpg)
ANNOTATION_EDITED_IMAGE_KEY_NAME = 'target_image_path'

# 参考图列表，上游已按固定顺序存好: 第0张永远是编辑前原图，
# 只有reference_replace_part1/part7是2张(原图 + 被抽取出来的物体图)，其余32个分片都是1张
ANNOTATION_REFERENCE_IMAGE_LIST_KEY_NAME = 'reference_image_path_list'

ANNOTATION_CAPTION_KEY_NAME = 'prompt'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 实测单轮子集的sample_id和图像原始名前缀100%满足(action子集的视频帧名形如
# kgrdxxd90yk_segment_151_frame_164也满足)，最长保存名77字符，远低于文件系统上限
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 上游标注里剩下的这些属性，按方案确认全部丢弃、不另存索引:
# task_type          : 1492359条全是image_edit，废字段
# subset_name        : 上游图像目录名，只用于定位图像索引文件
# parquet_name       : 只用于推导子集名
# row_index          : parquet里的行号，已被sample_id内含
# total_turn_num     : 单轮子集恒为1
# sample_dir         : 样本目录，跨子集重名，无法当唯一键
# turn_index         : 单轮子集恒为0
# global_prompt      : 单轮子集恒为空(非空的61722条全在已跳过的content_memory里)
# history_prompt_list: 单轮子集恒为空
# reference_image_num: 不直接采信，一律由reference_image这个list的长度现算
# metadata/*.jsonl(1006441行)里的样本自带属性也全部丢弃:
#   result.original_path / result.resolution{width,height} / result.edit_type /
#   result.edit_prompt / result.style(style_transfer专有画风描述) /
#   result.edit_obj{class_name,bbox,mask,score,clip_score,aes_score} /
#   result.edit_result / result.round1_prompt等多轮字段 /
#   judge.score(184995条) / judge_2scores(237382条)
# metadata/all_dataset_gpt_score.jsonl(824333条/296MB)的GPT质量打分同样全部丢弃:
#   覆盖率只有55%、action四个分片0覆盖、且多轮子集是按样本目录打分而非按轮次打分，
#   拿它做质量过滤会造成各子集口径严重不一致
# benchmark/(287个成员的评测集)与训练标注体系无关，不处理
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃
VALID_IMAGE_MODE_LIST = [
    'RGB',
    'L',
]

# reference_replace这个子集是双参考图([编辑前原图, 被抽取出的物体图])，
# 其余9个子集都只有编辑前原图这一张参考图，用于收尾自校验的交叉对账
DOUBLE_REFERENCE_IMAGE_SET_NAME_LIST = [
    'reference_replace',
]

# 【按方案确认整体丢弃的子集】
# reference_extract(118900对，扣掉上游缺图的part7整批41993对后剩76907对):
#   这个子集的**编辑后图本身就是从原图里被抠出来的那个物体图**(名字就是"提取"，
#   不是"编辑整张图")，所以编辑后图与编辑前原图的长宽比天生就不一样。
#   而本次改动要求"reference_image[0]必须与编辑后图长宽比严格相同、否则整对丢弃"，
#   这个子集会被逐条判掉、几乎清零，留着只会在日志里刷满
#   different_aspect_ratio，所以按方案在解析阶段就整体跳过。
#   注意与它同源的reference_replace(把物体换成参考图里的物体)**保留**:
#   那个子集的编辑后图是整张图、与原图同构，第2张物体参考图走长边对齐即可。
# 丢弃判定放在"逐parquet与逐子集条数统计之后"，
# 所以EXPECTED_SINGLE_TURN_ANNOTATION_COUNT_DICT /
# EXPECTED_SET_ANNOTATION_COUNT_DICT /
# EXPECTED_TOTAL_SINGLE_TURN_ANNOTATION_COUNT这三个硬对账口径全部保持原值不变
SKIP_SET_NAME_LIST = [
    'reference_extract',
]

# 被整体丢弃的子集合计条数(reference_extract实测118900条)，解析阶段硬对账。
# 这个数字守护的是"丢弃范围没被改动过"，改了子集白名单就会立刻失败
EXPECTED_SKIP_SET_COUNT = 118900

# 上游37个parquet里的34个单轮parquet及其实测标注条数(去掉3个多轮parquet后合计1178151条)。
# 解析阶段按parquet硬对账，少一条都说明上游annotations被改动过或没跑完
EXPECTED_SINGLE_TURN_ANNOTATION_COUNT_DICT = {
    'action_part1': 37224,
    'action_part2': 37416,
    'action_part3': 40294,
    'action_part4': 44074,
    'add_part0': 32233,
    'add_part1': 77039,
    'add_part4': 35127,
    'add_part5': 31068,
    'adjust_canny_part0': 42030,
    'adjust_canny_part2': 7756,
    'adjust_canny_part3': 42904,
    'adjust_canny_part4': 42819,
    'background_part0': 14135,
    'background_part2': 13991,
    'background_part3': 13493,
    'background_part5': 14095,
    'background_part7': 2376,
    'hybrid_part0': 10267,
    'hybrid_part2': 8656,
    'hybrid_part6': 9467,
    'reference_extract_part1': 76907,
    'reference_extract_part7': 41993,
    'reference_replace_part1': 76907,
    'reference_replace_part7': 41993,
    'remove_part0': 32222,
    'remove_part1': 76981,
    'remove_part4': 7698,
    'remove_part5': 42745,
    'replace_part0': 32209,
    'replace_part1': 75626,
    'replace_part4': 39912,
    'replace_part5': 11648,
    'style_transfer': 28374,
    'style_transfer_part0': 36472,
}

# 34个单轮parquet去掉_partN后聚合出的10个子集(即10个任务类型)及其实测标注条数
EXPECTED_SET_ANNOTATION_COUNT_DICT = {
    'action': 159008,
    'add': 175467,
    'adjust_canny': 135509,
    'background': 58090,
    'hybrid': 28390,
    'reference_extract': 118900,
    'reference_replace': 118900,
    'remove': 159646,
    'replace': 159395,
    'style_transfer': 64846,
}

EXPECTED_TOTAL_SINGLE_TURN_ANNOTATION_COUNT = 1178151

# 上游tar缺片导致图像没解出来的编辑对数(实测91963对):
# reference_extract_part7整批41993对缺result_1/result_2、
# reference_replace_part7整批41993对缺、remove_part0缺7968对、hybrid_part0缺9对。
# 这不是本脚本引入的问题，所以只上报数量、不作为致命错误
EXPECTED_MISSING_IMAGE_EDIT_PAIR_COUNT = 91963

# parquet名去掉分片号后缀就是子集名(任务类型)，style_transfer本身不带分片号
PARQUET_NAME_PART_SUFFIX_PATTERN = re.compile(r'_part\d+$')

PROCESS_NUM = 32

PER_FOLDER_EDIT_PAIR_NUM = 10000

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# ==============================================================================
# 【参考图与编辑后图的尺寸对齐规格(全部ti2i数据集统一口径)】
# 编辑后图**原分辨率落盘、不做任何缩放**，它是这个样本对唯一的尺寸基准
# (json里的width/height就是它)。参考图按下面的规则对齐:
#
#   reference_image[0](编辑前原图):
#       长宽比与编辑后图严格相同 -> LANCZOS resize到编辑后图尺寸(等比缩放、零形变)
#       长宽比不同               -> **整个样本对丢弃**(resize会把画面拉伸变形)
#
#   reference_image[k>=1](第二张视觉条件图/主体图/物体图):
#       长宽比与编辑后图严格相同 -> resize到编辑后图尺寸
#       长宽比不同               -> 按**长边对齐**等比resize(不裁剪、不形变、不丢弃)
#
# 长宽比判定用Fraction最简分数比、**不留任何容差**:
# 留容差会让"几乎一样但不严格相等"的样本(如910x512 vs 896x512)被各向异性拉伸落盘。
# 落盘后还会把参考图的真实shape与目标shape硬对账一次，不等则整对丢弃;
# 收尾自校验再真解一次reference_image[0]、硬校验它严格等于json里的width/height。
#
# 【为什么第一张参考图必须与编辑后图尺寸严格一致】
# 下游TorchAspectRatioBucketResize会按"这张参考图解码后的宽高比是否等于编辑后图
# 解码后的宽高比"逐张走两条分支: 相等走"与GT完全相同的各向异性resize到bucket
# 分辨率"(h/w RoPE逐像素对齐，结构/ID保持类编辑的关键)，不相等则退化成"保住自身
# 宽高比、长边对齐bucket长边"的独立主体图分支。
# 也就是说尺寸不一致的编辑样本会被静默降级成主体参考样本来训，
# 所以这个不变式必须在resave阶段就硬保证。
# ==============================================================================

# 参考图resize到目标尺寸时用的重采样方式。
# 按方案确认用PIL的LANCZOS而不是cv2.resize: 与项目既定口径(017.0/017.1)保持一致
SAVE_IMAGE_RESIZE_RESAMPLING = Image.Resampling.LANCZOS

# 豁免尺寸对齐的子集: 这些子集的编辑后图与全部参考图**完全原样落盘**，
# 不判长宽比、不resize、不丢弃、也不做长边对齐。
# 本数据集没有任何子集需要豁免(全部子集都要求参考图与编辑后图像素对齐)，
# 所以这里是空列表(保留这个常量只为与001等脚本口径一致)
EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST = []

# 收尾自校验时是否真解一次reference_image[0]、硬校验它的shape等于json里的
# width/height。按方案确认置True: "第一张参考图与编辑后图尺寸必须一致"是本次改动的
# 核心诉求，而只对账json里的数字是查不出resize有没有真的生效的，必须真解一次图。
# 代价是收尾自校验要多解一遍全部参考图，在NAS上会明显变慢
CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG = True

# 实测单轮1178151条指令strip后min 9、p50 73、p90 156、p99 198、p999 251、max 2184，
# 没有一条小于10(长度9的11921条"remove it"全在已跳过的content_understanding里)
MIN_CAPTION_LENGTH = 10

# 这个数据集的指令是带方位/尺寸描述的长编辑指令，各子集分布差异很大
# (add子集p50就有163、p99是207，remove子集p99是217)，
# 阈值取200会砍掉12456条完全正常的长指令(add 3803/remove 3159/hybrid 574/action 58
# 加上已跳过的多轮子集)，属于砍正常样本而不是砍异常值;
# 真正的异常长尾在512以上(单轮里只有463条: add 339 + remove 124，max 2184)，
# 按方案确认取512。上限判定放在指令归一化之后，和写进json的口径完全一致
MAX_CAPTION_LENGTH = 512

# 带编号的视觉参考图占位符，编号从"非原图的第1张参考图"起算:
# [V1*]指代reference_image[1]、[V2*]指代reference_image[2]...[VN*]指代
# reference_image[N]，其中N == reference_image_num - 1。
# 编辑前原图reference_image[0]永远隐式、不写进指令、不占编号。
# 这套写法与002.resave_anyedit_split_dataset.py完全一致，保证跨数据集口径统一
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# ImgEdit的双参考图子集reference_replace用自然语言指代第二张参考图，
# 指令形如"Replace sports car with the reference image."，没有任何编号占位符，
# 落盘前统一改写成带编号的[V1*]，保证新数据集里只存在带编号一种形态。
# 这个改写的100%正确性已全量实测验证(118900条，非抽样):
# 1. 句式100%单一: 全部严格匹配^Replace (.+) with the reference image\.$，0条不匹配;
# 2. 短语出现次数恒为1: 每条指令里"the reference image"恰好出现1次，
#    不存在出现0次或2次以上的情况，所以str.replace不会误替多处;
# 3. 改写后100%自洽: 过滤缺图后的76907条用check_invalid_caption逐条校验全部通过、0失败;
# 4. 无污染风险: 单轮1178151条指令里含"reference"一词的只有这118900条、
#    含"[V"或"<image>"的0条，所以对单参考图子集执行同一个replace是空操作，不会误伤;
# 5. 改写后长度变化为-14(19字符换成5字符)，该子集max长度105，不触碰任何长度阈值
CAPTION_REFERENCE_IMAGE_PHRASE = 'the reference image'

CAPTION_FIRST_VISUAL_PLACEHOLDER = '[V1*]'

# 同一个编号在一条指令里最多允许重复出现的次数。
# 实测reference_replace的118900条指令里每条只出现1次，这个上限只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2


def get_set_name(per_parquet_name):
    """把parquet名去掉_partN后缀，得到子集名(即图像编辑任务类型)

    34个单轮parquet聚合成10个子集: action/add/adjust_canny/background/hybrid/
    reference_extract/reference_replace/remove/replace/style_transfer。
    style_transfer这个parquet本身不带分片号，正则匹配不上，原样返回即可。
    """
    per_parquet_name = str(per_parquet_name).strip()

    return PARQUET_NAME_PART_SUFFIX_PATTERN.sub('', per_parquet_name)


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量(reference_replace是2，其余是1)"""
    if per_set_name in DOUBLE_REFERENCE_IMAGE_SET_NAME_LIST:
        return 2

    return 1


def check_skip_set(per_set_name):
    """判定这个子集是不是要整体丢弃的子集，返回True表示丢弃

    见SKIP_SET_NAME_LIST的注释: reference_extract的编辑后图就是被抠出来的物体图、
    与编辑前原图长宽比天生不同，按"第一张参考图必须与编辑后图长宽比相同"这条规则
    会被逐条判掉，所以在解析阶段就整体跳过。
    """
    return per_set_name in SKIP_SET_NAME_LIST


def get_normalized_ti2i_caption(per_ti2i_caption):
    """把上游自然语言指代的"the reference image"归一化成带编号的[V1*]

    ImgEdit的视觉参考图只有1张(只在reference_replace子集里)，上游指令用自然语言
    指代它。新数据集统一按"编号从非原图的第1张参考图起算"的写法保存，
    所以这里把"the reference image"改写成"[V1*]"。
    单参考图子集的指令里不含这个短语，执行到这里是空操作。
    已经带编号的占位符原样保留，不做任何改动(便于后续多参考图数据集复用本函数)。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    return per_ti2i_caption.replace(CAPTION_REFERENCE_IMAGE_PHRASE,
                                    CAPTION_FIRST_VISUAL_PLACEHOLDER)


def check_invalid_caption(per_ti2i_caption, per_reference_image_num):
    """判定占位符编号与参考图数量不自洽的坏指令，返回True表示这条指令不合格

    参考图里第0张永远是编辑前原图(隐式、不占编号)，所以一条指令应该带的占位符编号
    正好是1...N，其中N = reference_image_num - 1。这里做三条校验:
    1. 同一个编号最多重复2次，超过就是逐字符插占位符的坏指令;
    2. 最大编号必须正好等于N，多了就是指代了不存在的参考图;
    3. 1...N每个编号都必须至少出现一次，不允许跳号，也不允许有图没被指代。
    N为0时(9个单参考图子集)要求指令里完全没有占位符。
    另外还禁止无编号与带编号混用: 归一化后本不该出现，这里只做防御性拦截。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    per_placeholder_index_list = CAPTION_VISUAL_PLACEHOLDER_PATTERN.findall(
        per_ti2i_caption)

    # 归一化之后不允许再出现无编号的[V*]
    if '' in per_placeholder_index_list:
        return True

    per_placeholder_index_count_dict = {}
    for per_placeholder_index in per_placeholder_index_list:
        per_placeholder_index = int(per_placeholder_index)
        per_placeholder_index_count_dict[
            per_placeholder_index] = per_placeholder_index_count_dict.get(
                per_placeholder_index, 0) + 1

    # 校验1: 同一个编号最多重复MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM次
    for per_placeholder_index, per_placeholder_count in per_placeholder_index_count_dict.items(
    ):
        if per_placeholder_count > MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM:
            return True

    per_expect_placeholder_num = per_reference_image_num - 1
    if per_expect_placeholder_num < 0:
        per_expect_placeholder_num = 0

    # 校验2: 最大编号必须正好等于N(N为0时不允许有任何占位符)
    per_max_placeholder_index = max(per_placeholder_index_count_dict.keys(
    )) if len(per_placeholder_index_count_dict) > 0 else 0
    if per_max_placeholder_index != per_expect_placeholder_num:
        return True

    # 校验3: 1...N每个编号都必须至少出现一次，不允许跳号
    for per_placeholder_index in range(1, per_expect_placeholder_num + 1):
        if per_placeholder_index not in per_placeholder_index_count_dict:
            return True

    return False


def check_image_file_exists(per_image_relative_path, per_subset_name,
                            root_image_path, root_image_index_path,
                            subset_image_path_set_cache_dict):
    """用上游按子集另存的图像索引替代逐样本os.path.exists

    上游图像都放在NAS上，逐样本打一次os.path.exists就是一次网络往返，
    而本数据集每条标注要判2~3张图、合计约270万次往返;
    改用os.listdir按目录缓存也没用: ImgEdit是"一个样本目录只放一个样本的图"的结构，
    36万个样本目录意味着36万次listdir，比逐个exists好不了多少。
    上游已经在image_index/<subset_name>.txt里存好了每个子集的全部落盘图像相对路径
    (35个文件、合计407万行，路径写法和标注里完全同构)，所以这里按子集读一次索引建集合，
    之后只做集合查表，网络往返次数直接降到"子集数"。
    实测同一个jsonl里的图像全部落在同一个子集下，所以每个worker的缓存里通常只有一个集合。
    读索引失败(文件不存在/无权限)时回退到os.path.exists逐个判，
    保证判定结果和没有索引时完全一致。
    """
    if per_subset_name not in subset_image_path_set_cache_dict:
        per_image_index_path = os.path.join(
            root_image_index_path,
            f'{per_subset_name}{LOAD_IMAGE_INDEX_FILE_NAME_SUFFIX}')
        try:
            per_image_path_set = set()
            with open(per_image_index_path, 'r',
                      encoding='UTF-8') as load_index_file:
                for per_line in load_index_file:
                    per_line = per_line.strip()
                    if per_line:
                        per_image_path_set.add(per_line)
            subset_image_path_set_cache_dict[
                per_subset_name] = per_image_path_set
        except Exception:
            subset_image_path_set_cache_dict[per_subset_name] = None

    per_image_path_set = subset_image_path_set_cache_dict[per_subset_name]
    if per_image_path_set is None:
        return os.path.exists(
            os.path.join(root_image_path, per_image_relative_path))

    return per_image_relative_path in per_image_path_set


def get_save_image_name_prefix(per_image_relative_path):
    """取上游图像的原始名前缀(小写、不带后缀)，用来区分同一个样本对里的多张参考图

    reference_replace子集的两张参考图原始名前缀分别是original(编辑前原图)和
    result_1(被抽取出来的物体图)，其余子集只有一张参考图。
    实测单轮子集里没有任何一个样本对内部出现重复的原始名前缀(0例)，
    所以拼进保存名后，编辑后图像 + 全部参考图合计2475202个保存名100%唯一、0重名。
    """
    per_image_name = os.path.basename(
        str(per_image_relative_path).replace('\\', '/'))

    return os.path.splitext(per_image_name)[0].strip().lower()


def process_single_annotation_file(annotation_file_pair):
    """解析单个上游jsonl标注文件，组装图像编辑对(参考图+编辑后图+编辑指令)的列表

    这一步只做纯文本层面的过滤(缺字段、缺图、图不存在、保存名非法、指令为空或过短、
    指令过长、指令是坏占位符指令)，图像本身的解码校验和分辨率过滤留到后面多进程里做。
    """

    per_annotation_path, root_image_path, root_image_index_path = annotation_file_pair

    per_parquet_file_name = os.path.basename(per_annotation_path)
    per_parquet_name = per_parquet_file_name[:-len(
        LOAD_ANNOTATION_FILE_NAME_SUFFIX)]

    annotation_list = []
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_jsonl_file:
            for per_line in load_jsonl_file:
                per_line = per_line.strip()
                if not per_line:
                    continue
                annotation_list.append(json.loads(per_line))
    except Exception as e:
        print('2222', per_annotation_path, e)

    total_annotation_count = len(annotation_list)
    missing_image_count, invalid_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    skip_set_count = 0
    edit_annotation_pair_list = []

    # 每个worker只处理一个标注文件，同一个标注文件里的图像全部落在同一个子集下，
    # 所以缓存里通常只有一个子集的图像路径集合
    subset_image_path_set_cache_dict = {}

    for per_annotation in annotation_list:

        # 子集名一律由parquet名现推，不采信标注里的subset_name(它是上游图像目录名，
        # 带分片号且和parquet名大面积不一致)
        per_annotation_parquet_name = per_annotation.get(
            ANNOTATION_PARQUET_NAME_KEY_NAME, '')
        if not isinstance(per_annotation_parquet_name, str):
            per_annotation_parquet_name = ''
        per_annotation_parquet_name = per_annotation_parquet_name.strip()
        if not per_annotation_parquet_name:
            per_annotation_parquet_name = per_parquet_name

        per_set_name = get_set_name(per_annotation_parquet_name)

        # 整体丢弃的子集在这里就跳过(见SKIP_SET_NAME_LIST的注释)。
        # 这一步刻意排在total_annotation_count统计之后、任何图像与指令过滤之前:
        # 逐parquet与逐子集的条数硬对账用的是total_annotation_count这个口径，
        # 所以那三个EXPECTED_*常量全部保持原值不变;
        # 而后面那些过滤计数则只统计保留下来的子集
        if check_skip_set(per_set_name):
            skip_set_count += 1
            continue

        per_subset_name = per_annotation.get(ANNOTATION_SUBSET_NAME_KEY_NAME,
                                             '')
        if not isinstance(per_subset_name, str):
            per_subset_name = ''
        per_subset_name = per_subset_name.strip()

        per_sample_id = per_annotation.get(ANNOTATION_SAMPLE_ID_KEY_NAME, '')
        if not isinstance(per_sample_id, str):
            per_sample_id = ''
        per_sample_id = per_sample_id.strip().lower()

        # 任务类型决定子集名、sample_id决定保存图像名，缺任意一个都无法安全落盘
        if not per_set_name or not per_sample_id or not per_subset_name:
            missing_image_count += 1
            continue

        per_edited_image_relative_path = per_annotation.get(
            ANNOTATION_EDITED_IMAGE_KEY_NAME, '')
        if not isinstance(per_edited_image_relative_path, str):
            per_edited_image_relative_path = ''
        per_edited_image_relative_path = per_edited_image_relative_path.replace(
            '\\', '/').strip().lstrip('/')
        if not per_edited_image_relative_path:
            missing_image_count += 1
            continue

        if not check_image_file_exists(per_edited_image_relative_path,
                                       per_subset_name, root_image_path,
                                       root_image_index_path,
                                       subset_image_path_set_cache_dict):
            missing_image_count += 1
            continue

        per_edited_image_path = os.path.join(root_image_path,
                                             per_edited_image_relative_path)

        # 保存图像名前缀一定用全局唯一的sample_id，不能用上游图像原始名前缀。
        # sample_id本身已内含parquet名，所以这里不再重复拼子集名
        per_save_edited_image_name = f'{DATASET_NAME}_{per_sample_id}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
        # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名的前缀
        per_save_pair_folder_name = os.path.splitext(
            per_save_edited_image_name)[0]

        if not VALID_IMAGE_NAME_PATTERN.match(per_save_edited_image_name):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_save_edited_image_name)
            continue

        # 参考图顺序完全沿用上游的顺序，第0张一定是编辑前原图
        per_reference_image_relative_path_list = per_annotation.get(
            ANNOTATION_REFERENCE_IMAGE_LIST_KEY_NAME, None) or []
        if not isinstance(per_reference_image_relative_path_list, list):
            per_reference_image_relative_path_list = [
                per_reference_image_relative_path_list
            ]

        per_reference_image_path_list = []
        per_save_reference_image_name_list = []
        per_missing_reference_image_count = 0
        per_invalid_save_reference_image_name_count = 0
        for per_reference_image_relative_path in per_reference_image_relative_path_list:
            if not isinstance(per_reference_image_relative_path, str):
                per_reference_image_relative_path = ''
            per_reference_image_relative_path = per_reference_image_relative_path.replace(
                '\\', '/').strip().lstrip('/')

            if not per_reference_image_relative_path:
                per_missing_reference_image_count += 1
                continue

            if not check_image_file_exists(per_reference_image_relative_path,
                                           per_subset_name, root_image_path,
                                           root_image_index_path,
                                           subset_image_path_set_cache_dict):
                per_missing_reference_image_count += 1
                continue

            # 保存名里带上图像原始名前缀(original/result_1等)，
            # 保证同一个样本对的多张参考图不会撞名
            per_save_reference_image_name = (
                f'{DATASET_NAME}_{per_sample_id}_'
                f'{get_save_image_name_prefix(per_reference_image_relative_path)}'
                f'{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}')
            if not VALID_IMAGE_NAME_PATTERN.match(
                    per_save_reference_image_name):
                per_invalid_save_reference_image_name_count += 1
                continue

            per_reference_image_path_list.append(
                os.path.join(root_image_path,
                             per_reference_image_relative_path))
            per_save_reference_image_name_list.append(
                per_save_reference_image_name)

        if per_invalid_save_reference_image_name_count > 0:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path,
                  per_save_reference_image_name_list)
            continue

        # 同一个样本对里两张参考图撞名会互相覆盖，整对丢弃
        if len(set(per_save_reference_image_name_list)) != len(
                per_save_reference_image_name_list):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path,
                  per_save_reference_image_name_list)
            continue

        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)
        # 参考图缺任意一张都会让这个编辑对的条件信息不完整，整对丢弃
        if per_missing_reference_image_count > 0 or len(
                per_reference_image_path_list
        ) != per_expect_reference_image_num:
            missing_image_count += 1
            continue

        per_ti2i_caption = per_annotation.get(ANNOTATION_CAPTION_KEY_NAME, '')
        if isinstance(per_ti2i_caption, (list, tuple)):
            per_ti2i_caption = per_ti2i_caption[0] if len(
                per_ti2i_caption) > 0 else ''
        if not isinstance(per_ti2i_caption, str):
            per_ti2i_caption = ''
        per_ti2i_caption = per_ti2i_caption.strip()

        # 空指令、全空格指令、过短指令都视为不合格图像编辑对。
        # 长度下限判定用归一化之前的原文，避免占位符改写少掉的14个字符影响阈值
        if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
            invalid_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 把自然语言指代的"the reference image"归一化成带编号的[V1*]，
        # 写进json的一定是归一化后的指令
        per_ti2i_caption = get_normalized_ti2i_caption(per_ti2i_caption)

        # 过长指令同样视为不合格图像编辑对，按方案用归一化后的指令判定，
        # 和写进json的指令口径完全一致，收尾自校验直接量json里的长度就能复检
        # (实测单轮117万条里只有463条超过512: add 339 + remove 124)
        if len(per_ti2i_caption) > MAX_CAPTION_LENGTH:
            too_long_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 占位符编号与参考图数量不自洽的指令也视为不合格图像编辑对(同一编号重复超限、
        # 最大编号不等于参考图数量-1、编号跳号或缺失)，
        # 这类样本在过滤阶段就丢掉，后面的分组切分才能保证每个文件夹都是满10000对
        if check_invalid_caption(per_ti2i_caption,
                                 per_expect_reference_image_num):
            invalid_placeholder_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:100])
            continue

        edit_annotation_pair_list.append([
            per_set_name,
            per_save_pair_folder_name,
            per_edited_image_path,
            per_save_edited_image_name,
            per_reference_image_path_list,
            per_save_reference_image_name_list,
            per_ti2i_caption,
            per_expect_reference_image_num,
        ])

    return [
        edit_annotation_pair_list,
        per_parquet_name,
        total_annotation_count,
        missing_image_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
        skip_set_count,
    ]


def get_all_edit_annotation_pair(root_dataset_path):
    """扫描上游解压好的34个单轮jsonl标注，多进程组装全部图像编辑对的列表

    上游一共37个jsonl，其中3个多轮标注按方案整体跳过，剩下34个合计117万条，
    逐条还要判2~3张图像文件是否存在，所以这里按标注文件粒度开多进程解析，最后再统一排序。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        *LOAD_ANNOTATION_DIR_NAME_LIST)
    root_image_path = os.path.join(root_dataset_path,
                                   *LOAD_IMAGE_DIR_NAME_LIST)
    root_image_index_path = os.path.join(root_dataset_path,
                                         *LOAD_IMAGE_INDEX_DIR_NAME_LIST)

    annotation_file_pair_list = []
    skip_multiturn_annotation_file_count = 0
    for per_annotation_name in sorted(os.listdir(root_annotation_path)):
        if not per_annotation_name.endswith(LOAD_ANNOTATION_FILE_NAME_SUFFIX):
            continue

        per_parquet_name = per_annotation_name[:-len(
            LOAD_ANNOTATION_FILE_NAME_SUFFIX)]

        # 三个多轮标注整体跳过，它们的全局约束和历史指令无法塞进单条ti2i_caption
        if per_parquet_name in SKIP_MULTITURN_PARQUET_NAME_LIST:
            skip_multiturn_annotation_file_count += 1
            continue

        annotation_file_pair_list.append([
            os.path.join(root_annotation_path, per_annotation_name),
            root_image_path,
            root_image_index_path,
        ])

    total_annotation_count = 0
    missing_image_count, invalid_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    skip_set_count = 0
    parquet_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])
            parquet_annotation_count_dict[
                per_load_result[1]] = per_load_result[2]
            total_annotation_count += per_load_result[2]
            missing_image_count += per_load_result[3]
            invalid_caption_count += per_load_result[4]
            too_long_caption_count += per_load_result[5]
            invalid_placeholder_caption_count += per_load_result[6]
            invalid_save_image_name_count += per_load_result[7]
            skip_set_count += per_load_result[8]

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        skip_multiturn_annotation_file_count,
        parquet_annotation_count_dict,
        total_annotation_count,
        missing_image_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
        skip_set_count,
    ]


def check_load_annotation_count(parquet_annotation_count_dict,
                                total_annotation_count, skip_set_count):
    """解析完标注后按parquet和子集两级硬对账: 文件名、每个文件的条数、每个子集的条数

    上游annotations是一次性解出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账能额外拦住"分片被并进错误子集"这种parquet级对账看不出来的问题。

    注意这里的三个EXPECTED_*口径都是"上游标注侧"的，与
    "reference_extract整体丢弃"这个下游决策无关: 丢弃发生在条数统计之后，
    所以这几个数字保持原值。被丢弃的条数单独用EXPECTED_SKIP_SET_COUNT硬对账。
    """
    check_error_message_list = []

    # 被整体丢弃的子集条数硬对账，守住"丢弃范围没被改动过"这条口径
    if skip_set_count != EXPECTED_SKIP_SET_COUNT:
        check_error_message_list.append(
            f'skip set count not match '
            f'{skip_set_count} != {EXPECTED_SKIP_SET_COUNT}')

    set_annotation_count_dict = {}

    for per_parquet_name in sorted(parquet_annotation_count_dict.keys()):
        if per_parquet_name not in EXPECTED_SINGLE_TURN_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown single turn annotation file {per_parquet_name}')
            continue

        per_expect_annotation_count = EXPECTED_SINGLE_TURN_ANNOTATION_COUNT_DICT[
            per_parquet_name]
        if parquet_annotation_count_dict[
                per_parquet_name] != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_parquet_name} annotation count not match '
                f'{parquet_annotation_count_dict[per_parquet_name]} != '
                f'{per_expect_annotation_count}')

        per_set_name = get_set_name(per_parquet_name)
        set_annotation_count_dict[per_set_name] = set_annotation_count_dict.get(
            per_set_name, 0) + parquet_annotation_count_dict[per_parquet_name]

    for per_parquet_name in sorted(
            EXPECTED_SINGLE_TURN_ANNOTATION_COUNT_DICT.keys()):
        if per_parquet_name not in parquet_annotation_count_dict:
            check_error_message_list.append(
                f'missing single turn annotation file {per_parquet_name}')

    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in EXPECTED_SET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown set {per_set_name}')
            continue

        per_expect_set_annotation_count = EXPECTED_SET_ANNOTATION_COUNT_DICT[
            per_set_name]
        if set_annotation_count_dict[
                per_set_name] != per_expect_set_annotation_count:
            check_error_message_list.append(
                f'{per_set_name} set annotation count not match '
                f'{set_annotation_count_dict[per_set_name]} != '
                f'{per_expect_set_annotation_count}')

    for per_set_name in sorted(EXPECTED_SET_ANNOTATION_COUNT_DICT.keys()):
        if per_set_name not in set_annotation_count_dict:
            check_error_message_list.append(f'missing set {per_set_name}')

    # 要整体丢弃的子集必须真的在上游标注里存在(不存在说明子集名写错了，
    # 那样丢弃就是空操作、reference_extract会被静默保留下来)
    for per_skip_set_name in SKIP_SET_NAME_LIST:
        if per_skip_set_name not in set_annotation_count_dict:
            check_error_message_list.append(
                f'skip set {per_skip_set_name} not in upstream annotation')

    if total_annotation_count != EXPECTED_TOTAL_SINGLE_TURN_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total single turn annotation count not match '
            f'{total_annotation_count} != '
            f'{EXPECTED_TOTAL_SINGLE_TURN_ANNOTATION_COUNT}')

    return check_error_message_list


def check_same_image_aspect_ratio(per_reference_image_shape,
                                  per_edited_image_shape):
    """判定参考图与编辑后图的长宽比是否严格相同，返回True表示相同(可以等比resize)

    两个入参都是[宽, 高]。
    用Fraction的最简分数比精确判定、**不留任何容差**，不用浮点相除:
    浮点比较要么因为精度误差把本该相同的判成不同(如1056/1584与832/1248)，
    要么需要引入一个人为的容差阈值，而容差一旦放开就会让"几乎一样但不严格相等"的
    样本(如910x512与896x512)被各向异性拉伸落盘。最简分数比是精确的、可复现的。
    """
    return Fraction(per_reference_image_shape[0],
                    per_reference_image_shape[1]) == Fraction(
                        per_edited_image_shape[0], per_edited_image_shape[1])


def get_long_side_aligned_shape(per_reference_image_shape,
                                per_edited_image_shape):
    """按长边与编辑后图长边对齐，算出参考图应该被resize到的[宽, 高]

    只有reference_image[k>=1](第二张视觉条件图/主体图/物体图)在长宽比与编辑后图
    不同时才会走到这里: 这类参考图是独立主体/材质样例，本来就不要求与编辑后图
    像素对齐，硬resize到编辑后图尺寸会把画面拉伸变形，所以改成保持它自己的长宽比、
    只把长边缩放到与编辑后图长边相同(等比缩放、零形变、不裁剪)。
    """
    per_reference_image_w, per_reference_image_h = per_reference_image_shape
    per_edited_image_w, per_edited_image_h = per_edited_image_shape

    per_scale = max(per_edited_image_w, per_edited_image_h) / max(
        per_reference_image_w, per_reference_image_h)

    per_save_image_w = max(1, int(round(per_reference_image_w * per_scale)))
    per_save_image_h = max(1, int(round(per_reference_image_h * per_scale)))

    return [
        per_save_image_w,
        per_save_image_h,
    ]


def get_save_reference_image_shape(per_reference_image_index,
                                   per_reference_image_shape,
                                   per_edited_image_shape):
    """算出这张参考图应该被resize到的[宽, 高]，返回None表示整个样本对必须丢弃

    统一口径(见文件头的尺寸对齐规格):
      长宽比与编辑后图严格相同 -> 一律resize到编辑后图尺寸(等比缩放、零形变);
      长宽比不同且是reference_image[0](编辑前原图) -> 返回None，整对丢弃;
      长宽比不同且是reference_image[k>=1]          -> 按长边对齐resize。
    """
    if check_same_image_aspect_ratio(per_reference_image_shape,
                                     per_edited_image_shape):
        return list(per_edited_image_shape)

    # 第一张参考图是编辑前原图，它必须与编辑后图像素对齐(这是编辑类样本的根本要求)，
    # 长宽比不同就无法在不形变的前提下对齐，整对丢弃
    if per_reference_image_index == 0:
        return None

    return get_long_side_aligned_shape(per_reference_image_shape,
                                       per_edited_image_shape)


def check_single_image(per_image_path):
    """校验单张图像能否正常解码，并过滤非RGB图和极端分辨率图

    返回图像宽高只用于统计，最终写进json的宽高一定取自实际写盘图像的shape。
    """
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把P图/CMYK图静默转成3通道，
    # 所以必须先用PIL读原始mode才能把灰度图/P图/CMYK图判出来
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

    # 检查图像宽高比
    per_image_aspect_ratio = max(per_image_w / per_image_h,
                                 per_image_h / per_image_w)
    if per_image_aspect_ratio > MAX_IMAGE_ASPECT_RATIO:
        print('7777', per_image_path, per_image_w, per_image_h)
        return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_edit_pair_check(edit_annotation_pair):
    """校验单个图像编辑对，并算出每张参考图应该被resize到的目标尺寸

    返回值是[状态字符串, 子集名, 编辑对]:
      'ok'                     -> 第三项是带上每张参考图目标尺寸的编辑对
      'invalid_image'          -> 有图解不开/mode不在白名单/短边或宽高比越界，
                                  第三项为None
      'different_aspect_ratio' -> reference_image[0]与编辑后图长宽比不同(整对丢弃)，
                                  第三项为None。带上子集名是为了在主流程里逐子集统计

    短边和宽高比的过滤按方案只以编辑后图像为准判定，参考图只要求能正常解码且
    mode命中白名单。

    这里之所以能零额外IO地判长宽比: check_single_image本来就已经把编辑后图和
    每张参考图都真解码了一遍并返回了[宽, 高]，改造前只是把返回值丢掉了。
    现在接住这些shape，既能判长宽比、又能把"每张参考图的目标尺寸"一路带到落盘阶段，
    落盘时不必再重算一次。

    豁免子集(EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST)的目标尺寸一律给None，
    表示编辑后图与全部参考图都完全原样落盘。
    """
    per_set_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_ti2i_caption, per_expect_reference_image_num = edit_annotation_pair

    per_edited_image_shape = check_single_image(per_edited_image_path)
    if per_edited_image_shape is None:
        return ['invalid_image', per_set_name, None]

    per_exempt_aspect_ratio_align_flag = per_set_name in EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST

    per_save_reference_image_shape_list = []
    for per_reference_image_index, per_reference_image_path in enumerate(
            per_reference_image_path_list):
        per_reference_image_shape = check_single_image(
            per_reference_image_path)
        if per_reference_image_shape is None:
            return ['invalid_image', per_set_name, None]

        # 豁免子集完全原样落盘: 不判长宽比、不resize、不丢弃、也不做长边对齐
        if per_exempt_aspect_ratio_align_flag:
            per_save_reference_image_shape_list.append(None)
            continue

        per_save_reference_image_shape = get_save_reference_image_shape(
            per_reference_image_index, per_reference_image_shape,
            per_edited_image_shape)
        # 只有reference_image[0]长宽比不同才会拿到None，此时整对丢弃
        if per_save_reference_image_shape is None:
            return ['different_aspect_ratio', per_set_name, None]

        per_save_reference_image_shape_list.append(
            per_save_reference_image_shape)

    return [
        'ok',
        per_set_name,
        [
            per_set_name,
            per_save_pair_folder_name,
            per_edited_image_path,
            per_save_edited_image_name,
            per_reference_image_path_list,
            per_save_reference_image_name_list,
            per_save_reference_image_shape_list,
            per_ti2i_caption,
            per_expect_reference_image_num,
        ],
    ]


def get_all_edit_pair_save_folder_pair(edit_annotation_pair_list,
                                       save_dataset_path):
    """把过滤后的合格图像编辑对按子集分组，排序后每10000对切成一个文件夹

    切分必须在过滤全部完成之后做，且切分前先按保存的编辑后图像名排序，这样才能保证
    每个文件夹都是满10000对(最后一个文件夹允许不满)。每个图像编辑对在文件夹里再独占
    一个子文件夹，该对的编辑后图像和所有参考图像都存在这个子文件夹里。
    """
    per_set_edit_annotation_pair_dict = {}
    for per_edit_annotation_pair in edit_annotation_pair_list:
        per_set_name = per_edit_annotation_pair[0]
        if per_set_name not in per_set_edit_annotation_pair_dict:
            per_set_edit_annotation_pair_dict[per_set_name] = []
        per_set_edit_annotation_pair_dict[per_set_name].append(
            per_edit_annotation_pair)

    edit_pair_save_folder_pair_list = []
    set_folder_count_dict = {}
    for per_set_name in sorted(per_set_edit_annotation_pair_dict.keys()):
        per_set_edit_annotation_pair_list = sorted(
            per_set_edit_annotation_pair_dict[per_set_name],
            key=lambda x: x[3])

        per_set_folder_count = 0
        for per_folder_start_index in range(
                0, len(per_set_edit_annotation_pair_list),
                PER_FOLDER_EDIT_PAIR_NUM):
            per_folder_edit_annotation_pair_list = per_set_edit_annotation_pair_list[
                per_folder_start_index:per_folder_start_index +
                PER_FOLDER_EDIT_PAIR_NUM]

            per_folder_name = f'{per_set_name}_{per_set_folder_count:05d}'

            for per_edit_annotation_pair in per_folder_edit_annotation_pair_list:
                _, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_save_reference_image_shape_list, per_ti2i_caption, per_expect_reference_image_num = per_edit_annotation_pair

                per_pair_folder_path = os.path.join(save_dataset_path,
                                                    per_set_name,
                                                    per_folder_name,
                                                    per_save_pair_folder_name)
                os.makedirs(per_pair_folder_path, exist_ok=True)

                edit_pair_save_folder_pair_list.append([
                    per_set_name,
                    per_folder_name,
                    per_save_pair_folder_name,
                    per_edited_image_path,
                    per_save_edited_image_name,
                    per_reference_image_path_list,
                    per_save_reference_image_name_list,
                    per_save_reference_image_shape_list,
                    per_ti2i_caption,
                    per_expect_reference_image_num,
                ])

            per_set_folder_count += 1

        set_folder_count_dict[per_set_name] = per_set_folder_count

    return edit_pair_save_folder_pair_list, set_folder_count_dict


def resize_single_image(per_image, per_save_image_shape):
    """把BGR的numpy图像resize到指定的[宽, 高]，返回resize后的BGR numpy图像

    按方案确认用PIL的LANCZOS而不是cv2.resize(与017.0/017.1口径一致):
    先BGR->RGB转成PIL、resize、再转回numpy并RGB->BGR，
    中间的颜色通道转换不能省，否则落盘图像的R与B通道会互换。
    """
    per_save_image_w, per_save_image_h = per_save_image_shape

    per_pil_image = Image.fromarray(cv2.cvtColor(per_image, cv2.COLOR_BGR2RGB))
    per_pil_image = per_pil_image.resize((per_save_image_w, per_save_image_h),
                                         SAVE_IMAGE_RESIZE_RESAMPLING)

    return cv2.cvtColor(np.asarray(per_pil_image), cv2.COLOR_RGB2BGR)


def resave_single_image(per_image_path,
                        save_image_path,
                        save_image_shape=None):
    """重新编码保存单张图像，只在显式给了目标尺寸时才resize

    save_image_shape为None(编辑后图与豁免子集的参考图走这条): 图像原分辨率多少
    保存时还是多少，不做任何缩放;
    给了[宽, 高](非豁免子集的参考图走这条): 先LANCZOS resize到这个尺寸再落盘。
    目标尺寸由process_single_edit_pair_check按统一规则算好并一路带下来:
    reference_image[0]一定是编辑后图尺寸(长宽比不同的样本对已经在那一步整对丢弃了)，
    reference_image[k>=1]是编辑后图尺寸或长边对齐后的尺寸。

    上游有133万张png(绝大多数子集的编辑前后图都是png，只有action子集是jpg)，
    这里统一重编码成jpg，只换编码格式不换像素尺寸。
    """
    try:
        per_image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
    except Exception as e:
        print('8888', per_image_path, e)
        return None

    if per_image is None or per_image.ndim != 3 or per_image.shape[2] != 3:
        print('8888', per_image_path)
        return None

    # 只有非豁免子集的参考图会带目标尺寸，且只在尺寸真的不一样时才resize
    if save_image_shape is not None:
        per_save_image_w, per_save_image_h = save_image_shape
        if per_save_image_w <= 0 or per_save_image_h <= 0:
            print('8888', per_image_path, save_image_shape)
            return None

        if per_image.shape[1] != per_save_image_w or per_image.shape[
                0] != per_save_image_h:
            try:
                per_image = resize_single_image(per_image, save_image_shape)
            except Exception as e:
                print('8888', per_image_path, save_image_shape, e)
                return None

        # resize之后必须真的等于目标尺寸，不等说明resize没生效
        if per_image.shape[1] != per_save_image_w or per_image.shape[
                0] != per_save_image_h:
            print('8888', per_image_path, per_image.shape, save_image_shape)
            return None

    # 宽高直接取自这个即将被编码写盘的数组的shape，
    # jpg编解码不改变像素尺寸，所以宽高一定和保存图像一致
    per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    if not os.path.exists(save_image_path):
        try:
            cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
        except Exception as e:
            print('8888', save_image_path, e)
            return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_edit_pair(edit_pair_save_folder_pair, save_dataset_path):
    """重新编码保存单个图像编辑对的编辑后图像和所有参考图像，任意一张失败则整对丢弃"""
    per_set_name, per_folder_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_save_reference_image_shape_list, per_ti2i_caption, per_expect_reference_image_num = edit_pair_save_folder_pair

    per_pair_folder_path = os.path.join(save_dataset_path, per_set_name,
                                        per_folder_name,
                                        per_save_pair_folder_name)

    save_edited_image_path = os.path.join(per_pair_folder_path,
                                          per_save_edited_image_name)
    per_edited_image_shape = resave_single_image(per_edited_image_path,
                                                 save_edited_image_path)
    if per_edited_image_shape is None:
        return None

    per_edited_image_w, per_edited_image_h = per_edited_image_shape

    for per_reference_image_path, per_save_reference_image_name, per_save_reference_image_shape in zip(
            per_reference_image_path_list, per_save_reference_image_name_list,
            per_save_reference_image_shape_list):
        save_reference_image_path = os.path.join(
            per_pair_folder_path, per_save_reference_image_name)
        # per_save_reference_image_shape为None只出现在豁免子集，表示原样落盘
        per_save_shape = resave_single_image(per_reference_image_path,
                                             save_reference_image_path,
                                             per_save_reference_image_shape)
        if per_save_shape is None:
            return None

        # 落盘后的shape必须与目标尺寸严格相等，不等说明resize链路有问题。
        # 第一张参考图的目标尺寸就是编辑后图尺寸，这一条同时也就是
        # "第一张参考图与编辑后图尺寸必须严格一致"这个核心不变式的落盘期硬校验
        if per_save_reference_image_shape is not None and per_save_shape != list(
                per_save_reference_image_shape):
            print('8888', save_reference_image_path, per_save_shape,
                  per_save_reference_image_shape)
            return None

    return [
        per_folder_name,
        per_save_edited_image_name,
        list(per_save_reference_image_name_list),
        per_edited_image_w,
        per_edited_image_h,
        per_ti2i_caption,
        per_expect_reference_image_num,
    ]


def save_all_folder_annotation_json(save_result_list, save_dataset_path,
                                    set_folder_count_dict):
    """按文件夹汇总标注并写出与文件夹同名的json文件

    上游标注里还剩subset_name/parquet_name/task_type/row_index/total_turn_num/
    sample_dir/turn_index/global_prompt/history_prompt_list这些属性，
    以及metadata里的result/judge/judge_2scores和all_dataset_gpt_score打分，
    按方案全部丢弃、不另存索引: task_type全是image_edit没有区分力，
    turn_index/global_prompt/history_prompt_list在单轮子集里恒为0或空，
    gpt打分覆盖率只有55%且action子集0覆盖，拿来做质量过滤会造成子集间口径不一致。
    ti2i_caption写的是归一化后的指令，双参考图子集的占位符一定是带编号的[V1*]形态。
    """

    folder_annotation_dict = {}
    reference_image_num_mismatch_count = 0
    for per_save_result in save_result_list:
        per_folder_name, per_save_edited_image_name, per_save_reference_image_name_list, per_edited_image_w, per_edited_image_h, per_ti2i_caption, per_expect_reference_image_num = per_save_result
        if per_folder_name not in folder_annotation_dict:
            folder_annotation_dict[per_folder_name] = {}

        # reference_image_num一定由reference_image这个list的长度现算，
        # 保证写进json的数值永远和list长度对得上
        per_reference_image_num = len(per_save_reference_image_name_list)
        # 再和该子集应有的参考图数量(reference_replace是2，其余是1)交叉对账，
        # 不一致只上报不改数值
        if per_expect_reference_image_num >= 0 and per_reference_image_num != per_expect_reference_image_num:
            reference_image_num_mismatch_count += 1
            print('9999', per_save_edited_image_name, per_reference_image_num,
                  per_expect_reference_image_num)

        # ti2i_caption_length直接取即将写进json的这个字符串的长度，
        # 保证记录的长度和ti2i_caption永远自洽(该字符串已strip并归一化过占位符)
        folder_annotation_dict[per_folder_name][per_save_edited_image_name] = {
            'reference_image': per_save_reference_image_name_list,
            'edited_image': per_save_edited_image_name,
            'reference_image_num': per_reference_image_num,
            'width': per_edited_image_w,
            'height': per_edited_image_h,
            'ti2i_caption': per_ti2i_caption,
            'ti2i_caption_length': len(per_ti2i_caption),
        }

    folder_edit_pair_count_dict = {}
    for per_set_name in sorted(set_folder_count_dict.keys()):
        for per_folder_index in range(set_folder_count_dict[per_set_name]):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            if per_folder_name not in folder_annotation_dict:
                print('9999', per_folder_name)
                continue

            per_folder_annotation_dict = folder_annotation_dict[
                per_folder_name]
            per_folder_annotation_dict = {
                per_save_edited_image_name:
                per_folder_annotation_dict[per_save_edited_image_name]
                for per_save_edited_image_name in sorted(
                    per_folder_annotation_dict.keys())
            }

            save_json_path = os.path.join(save_dataset_path, per_set_name,
                                          f'{per_folder_name}.json')
            with open(save_json_path, 'w', encoding='UTF-8') as save_json_file:
                json.dump(per_folder_annotation_dict,
                          save_json_file,
                          ensure_ascii=False)

            folder_edit_pair_count_dict[per_folder_name] = len(
                per_folder_annotation_dict)

            print('2222', per_folder_name, len(per_folder_annotation_dict))

    return folder_edit_pair_count_dict, reference_image_num_mismatch_count


def check_save_dataset(save_dataset_path, set_folder_count_dict):
    """全部落盘后的收尾自校验: 文件夹容量、json与磁盘一一对应、指令与参考图数量

    每个子集除最后一个文件夹外都必须是满10000对，json里的每个key都必须在磁盘上有
    对应的样本对文件夹且文件恰好等于编辑后图像 + 所有参考图像，磁盘上也不允许有
    json没记录的残留样本对文件夹。另外还要校验ti2i_caption里的占位符编号集合与
    reference_image这个list的长度必须自洽(编号只能是1...len(reference_image)-1
    且每个编号至少出现一次)，保证文本指代与参考图一一对应。
    """

    check_error_message_list = []
    total_edit_pair_count = 0
    for per_set_name in sorted(set_folder_count_dict.keys()):
        per_set_dir_path = os.path.join(save_dataset_path, per_set_name)
        per_set_folder_count = set_folder_count_dict[per_set_name]
        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)
        per_exempt_aspect_ratio_align_flag = per_set_name in EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST
        for per_folder_index in range(per_set_folder_count):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            per_json_path = os.path.join(per_set_dir_path,
                                         f'{per_folder_name}.json')
            if not os.path.isfile(per_json_path):
                check_error_message_list.append(
                    f'{per_folder_name} json not exists')
                continue

            with open(per_json_path, 'r', encoding='UTF-8') as load_json_file:
                per_folder_annotation_dict = json.load(load_json_file)

            total_edit_pair_count += len(per_folder_annotation_dict)

            # 除每个子集最后一个文件夹外都必须是满10000对
            if per_folder_index < per_set_folder_count - 1 and len(
                    per_folder_annotation_dict) != PER_FOLDER_EDIT_PAIR_NUM:
                check_error_message_list.append(
                    f'{per_folder_name} edit pair num not match {len(per_folder_annotation_dict)} != {PER_FOLDER_EDIT_PAIR_NUM}'
                )

            per_folder_path = os.path.join(per_set_dir_path, per_folder_name)
            per_exist_pair_folder_name_list = sorted([
                per_save_pair_folder_name
                for per_save_pair_folder_name in os.listdir(per_folder_path)
                if os.path.isdir(
                    os.path.join(per_folder_path, per_save_pair_folder_name))
            ])
            per_expect_pair_folder_name_list = sorted([
                per_save_edited_image_name.removesuffix('.jpg')
                for per_save_edited_image_name in
                per_folder_annotation_dict.keys()
            ])
            if per_exist_pair_folder_name_list != per_expect_pair_folder_name_list:
                check_error_message_list.append(
                    f'{per_folder_name} pair folder not match {len(per_exist_pair_folder_name_list)} != {len(per_expect_pair_folder_name_list)}'
                )

            for per_save_edited_image_name in sorted(
                    per_folder_annotation_dict.keys()):
                per_annotation = per_folder_annotation_dict[
                    per_save_edited_image_name]

                # 每条标注的字段集合必须和约定的七个key严格一致，不能多也不能少
                if sorted(per_annotation.keys()) != sorted(
                        SAVE_ANNOTATION_KEY_NAME_LIST):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} annotation key not match'
                    )
                    continue

                if per_save_edited_image_name != per_annotation[
                        'edited_image']:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name not match'
                    )
                if not per_save_edited_image_name.endswith(
                        SAVE_EDITED_IMAGE_NAME_SUFFIX):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name suffix not match'
                    )
                if not isinstance(per_annotation['reference_image'], list):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image not a list'
                    )
                if per_annotation['reference_image_num'] != len(
                        per_annotation['reference_image']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match'
                    )
                # reference_replace子集必须是双参考图，其余9个子集必须是单参考图
                if per_annotation[
                        'reference_image_num'] != per_expect_reference_image_num:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match set {per_annotation["reference_image_num"]} != {per_expect_reference_image_num}'
                    )
                # ti2i_caption的占位符编号集合必须与reference_image这个list的
                # 长度自洽: 编号只能是1...len(reference_image)-1且每个都要出现
                if check_invalid_caption(
                        per_annotation['ti2i_caption'],
                        len(per_annotation['reference_image'])):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} caption placeholder index not match reference image num {len(per_annotation["reference_image"])}'
                    )
                # 归一化后的指令里不允许再残留自然语言指代的"the reference image"
                if CAPTION_REFERENCE_IMAGE_PHRASE in per_annotation[
                        'ti2i_caption']:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} caption still has an unnormalized reference image phrase'
                    )
                # json里存的就是归一化后的指令，上限过滤也是按归一化后判定的，
                # 两者口径一致，这里直接量json里的长度复检
                if len(per_annotation['ti2i_caption'].strip()
                       ) < MIN_CAPTION_LENGTH:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still an invalid caption'
                    )
                if len(per_annotation['ti2i_caption'].strip()
                       ) > MAX_CAPTION_LENGTH:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a too long caption'
                    )
                # 记录的指令长度必须和指令字符串的实际长度对得上
                if per_annotation['ti2i_caption_length'] != len(
                        per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} ti2i caption length not match'
                    )

                per_save_pair_folder_name = per_save_edited_image_name.removesuffix(
                    '.jpg')
                per_pair_folder_path = os.path.join(per_folder_path,
                                                    per_save_pair_folder_name)
                per_expect_file_name_list = sorted(
                    [per_save_edited_image_name] +
                    list(per_annotation['reference_image']))
                per_exist_file_name_list = sorted(
                    os.listdir(per_pair_folder_path)) if os.path.isdir(
                        per_pair_folder_path) else []
                if per_exist_file_name_list != per_expect_file_name_list:
                    check_error_message_list.append(
                        f'{per_save_pair_folder_name} pair image file not match'
                    )

                # 真解一次落盘后的reference_image[0]，硬校验它的shape严格等于
                # json里的width/height(也就是编辑后图的宽高)。
                # 这一条是"第一张参考图与编辑后图尺寸必须一致"这个核心不变式的最终
                # 验收: 只对账json里的数字是查不出resize有没有真的生效的。
                # 非豁免子集才校验: 豁免子集是完全原样落盘的，尺寸本来就可以不等
                if CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG and not per_exempt_aspect_ratio_align_flag and len(
                        per_annotation['reference_image']) > 0:
                    per_check_save_reference_image_path = os.path.join(
                        per_pair_folder_path,
                        per_annotation['reference_image'][0])
                    try:
                        per_check_save_reference_image_w, per_check_save_reference_image_h = Image.open(
                            per_check_save_reference_image_path).size
                    except Exception as e:
                        per_check_save_reference_image_w, per_check_save_reference_image_h = 0, 0
                        print('4444', per_check_save_reference_image_path, e)

                    if per_check_save_reference_image_w != per_annotation[
                            'width'] or per_check_save_reference_image_h != per_annotation[
                                'height']:
                        check_error_message_list.append(
                            f'{per_save_edited_image_name} first reference image shape not match edited image shape '
                            f'{per_check_save_reference_image_w} {per_check_save_reference_image_h} != '
                            f'{per_annotation["width"]} {per_annotation["height"]}'
                        )

    print('3333', 'check total edit pair:', total_edit_pair_count,
          'check error:', len(check_error_message_list))

    return check_error_message_list, total_edit_pair_count


def check_save_dataset_path_empty(save_dataset_path):
    """落盘前断言输出目录必须为空(或不存在)，非空直接报错

    这一条是resize链路的前提: resave_single_image里有
    "if not os.path.exists(save_image_path)"这个跳过重复写盘的短路，
    如果输出目录里还留着上一轮(未resize口径)跑出来的老图，
    这个短路会跳过写盘、但仍然返回内存里resize之后的shape，
    结果json记的宽高与磁盘上的实际文件不一致、收尾自校验也会大面积报错。
    所以本次改动之后这几个数据集必须落到全新的输出目录(或先手动删掉旧目录)重跑，
    不能在旧产物上增量跑。
    """
    if not os.path.exists(save_dataset_path):
        return

    per_exist_name_list = os.listdir(save_dataset_path)
    if len(per_exist_name_list) > 0:
        raise Exception(
            f'save dataset path not empty {save_dataset_path} '
            f'{len(per_exist_name_list)} {sorted(per_exist_name_list)[:10]}, '
            f'must remove the old resave result first')

    return


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path, SAVE_DATASET_DIR_NAME)
    # 必须落到全新的输出目录: 旧产物是未resize口径的，增量跑会让json宽高与磁盘错位
    check_save_dataset_path_empty(save_dataset_path)
    os.makedirs(save_dataset_path, exist_ok=True)

    edit_annotation_pair_list, total_annotation_file_count, skip_multiturn_annotation_file_count, parquet_annotation_count_dict, total_annotation_count, missing_image_count, invalid_caption_count, too_long_caption_count, invalid_placeholder_caption_count, invalid_save_image_name_count, skip_set_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count,
          skip_multiturn_annotation_file_count, total_annotation_count,
          skip_set_count, missing_image_count, invalid_caption_count,
          too_long_caption_count, invalid_placeholder_caption_count,
          invalid_save_image_name_count, len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        parquet_annotation_count_dict, total_annotation_count, skip_set_count)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if load_annotation_check_error_message_list:
        raise Exception(
            f'load annotation check failed {load_annotation_check_error_message_list[:20]}'
        )

    check_edit_annotation_pair_list = []
    invalid_image_count, different_aspect_ratio_count = 0, 0
    set_different_aspect_ratio_count_dict = {}
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result, per_check_set_name, per_check_edit_annotation_pair in tqdm(
                pool.imap(process_single_edit_pair_check,
                          edit_annotation_pair_list),
                total=len(edit_annotation_pair_list)):
            if per_check_result == 'invalid_image':
                invalid_image_count += 1
                continue
            # reference_image[0]与编辑后图长宽比不同的样本对在这里整对丢弃，
            # 逐子集记一份数字，方便看清是哪些任务类型天生对不齐
            if per_check_result == 'different_aspect_ratio':
                different_aspect_ratio_count += 1
                set_different_aspect_ratio_count_dict[
                    per_check_set_name] = set_different_aspect_ratio_count_dict.get(
                        per_check_set_name, 0) + 1
                continue
            check_edit_annotation_pair_list.append(
                per_check_edit_annotation_pair)

    print('1111', len(check_edit_annotation_pair_list), invalid_image_count,
          different_aspect_ratio_count)

    edit_pair_save_folder_pair_list, set_folder_count_dict = get_all_edit_pair_save_folder_pair(
        check_edit_annotation_pair_list, save_dataset_path)

    print('1111', len(edit_pair_save_folder_pair_list),
          len(set_folder_count_dict))
    if len(edit_pair_save_folder_pair_list) > 0:
        print('1111', edit_pair_save_folder_pair_list[0])

    save_result_list = []
    process_func = partial(process_single_edit_pair,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_save_result in tqdm(
                pool.imap(process_func, edit_pair_save_folder_pair_list),
                total=len(edit_pair_save_folder_pair_list)):
            if per_save_result is None:
                continue
            save_result_list.append(per_save_result)

    save_edit_pair_failed_count = len(edit_pair_save_folder_pair_list) - len(
        save_result_list)

    folder_edit_pair_count_dict, reference_image_num_mismatch_count = save_all_folder_annotation_json(
        save_result_list, save_dataset_path, set_folder_count_dict)

    total_save_reference_image_count = sum(
        [len(per_save_result[2]) for per_save_result in save_result_list])

    check_error_message_list, check_total_edit_pair_count = check_save_dataset(
        save_dataset_path, set_folder_count_dict)

    print('3333', 'total annotation file:', total_annotation_file_count,
          'skip multiturn annotation file:',
          skip_multiturn_annotation_file_count, 'total annotation:',
          total_annotation_count, 'skip set:', skip_set_count,
          'missing image:', missing_image_count, 'invalid caption:',
          invalid_caption_count, 'too long caption:', too_long_caption_count,
          'invalid placeholder caption:', invalid_placeholder_caption_count,
          'invalid save image name:', invalid_save_image_name_count,
          'invalid image:', invalid_image_count, 'different aspect ratio:',
          different_aspect_ratio_count, 'save edit pair failed:',
          save_edit_pair_failed_count, 'total save edit pair:',
          len(save_result_list), 'total save reference image:',
          total_save_reference_image_count, 'reference image num mismatch:',
          reference_image_num_mismatch_count, 'total save set:',
          len(set_folder_count_dict), 'total save folder:',
          len(folder_edit_pair_count_dict), 'check total edit pair:',
          check_total_edit_pair_count, 'check error:',
          len(check_error_message_list))

    save_check_result_path = os.path.join(save_dataset_path,
                                          'resave_check_result.json')
    save_check_result_dict = {
        'total_annotation_file_count': total_annotation_file_count,
        'skip_multiturn_annotation_file_count':
        skip_multiturn_annotation_file_count,
        'total_annotation_count': total_annotation_count,
        # 按方案整体丢弃的子集(reference_extract)的条数与子集名
        'skip_set_count': skip_set_count,
        'skip_set_name_list': SKIP_SET_NAME_LIST,
        'missing_image_count': missing_image_count,
        'expected_missing_image_count': EXPECTED_MISSING_IMAGE_EDIT_PAIR_COUNT,
        'invalid_caption_count': invalid_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'invalid_image_count': invalid_image_count,
        # reference_image[0]与编辑后图长宽比不同而被整对丢弃的条数。
        # 第一轮跑完后可以把实测值回填成EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT
        # 再上硬对账，守住"哪些样本被resize对齐、哪些被丢弃"这条口径
        'different_aspect_ratio_count': different_aspect_ratio_count,
        'set_different_aspect_ratio_count_dict':
        set_different_aspect_ratio_count_dict,
        # 落盘口径标记，便于下游一眼看出这份产物是不是"参考图已对齐"的版本
        'resize_reference_image_to_edited_image_shape_flag': True,
        'long_side_align_extra_reference_image_flag': True,
        'exempt_aspect_ratio_align_set_name_list':
        EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST,
        'check_save_reference_image_shape_flag':
        CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG,
        'save_edit_pair_failed_count': save_edit_pair_failed_count,
        'total_save_edit_pair_count': len(save_result_list),
        'total_save_reference_image_count': total_save_reference_image_count,
        'reference_image_num_mismatch_count':
        reference_image_num_mismatch_count,
        'total_save_set_count': len(set_folder_count_dict),
        'total_save_folder_count': len(folder_edit_pair_count_dict),
        'check_total_edit_pair_count': check_total_edit_pair_count,
        'check_error_count': len(check_error_message_list),
        'parquet_annotation_count_dict': parquet_annotation_count_dict,
        'set_folder_count_dict': set_folder_count_dict,
        'folder_edit_pair_count_dict': folder_edit_pair_count_dict,
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    if check_total_edit_pair_count != len(save_result_list):
        check_error_message_list.append(
            f'check total edit pair count not match {check_total_edit_pair_count} != {len(save_result_list)}'
        )
    if len(check_error_message_list) > 0:
        # 收尾自校验不通过必须让上层感知，不能静默留下坏样本对或不满的文件夹
        raise Exception(
            f'check save dataset error num {len(check_error_message_list)} {check_error_message_list[:10]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/ImgEdit'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
