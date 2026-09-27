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

DATASET_NAME = 'x2edit'

SAVE_DATASET_DIR_NAME = 'X2Edit'

# 上游008解压脚本把1396个tar解成
# X2Edit_data/<构造模型名>/<分片编号>/<压缩包编号>/<样本编号前缀>.<后缀>这种四层结构，
# 一个样本对固定是同前缀的4个文件(textflux是5个):
#   <prefix>.1.0.jpg  参考图(编辑前原图)
#   <prefix>.1.1.jpg  第二参考图(只有textflux有，是文字前景mask)
#   <prefix>.2.jpg    编辑后图
#   <prefix>.json     该样本的全部属性(instruction/task/各类质量分等)
#   <prefix>.txt      编辑指令
# 上游解压自校验实测6026214个样本对全部完整、0个残缺样本(unzip_check_sample_result.json)
LOAD_DATA_DIR_NAME_LIST = [
    'X2Edit_data',
]

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.json'

LOAD_EDITED_IMAGE_FILE_NAME_SUFFIX = '.2.jpg'

LOAD_REFERENCE_IMAGE_FILE_NAME_SUFFIX = '.1.0.jpg'

# 只有textflux这一个构造模型目录有第二参考图，且该目录整体跳过不处理，
# 所以这个后缀在本脚本里只用于统计上报，不参与任何样本对的组装
LOAD_SECOND_REFERENCE_IMAGE_FILE_NAME_SUFFIX = '.1.1.jpg'

# textflux(257101对)按方案确认整体跳过不处理:
# 该子集的第二参考图.1.1.jpg是"文字前景mask"(实测黑底白字、90%以上像素是纯黑、
# 与原图同尺寸)，是渲染文字所必需的空间条件图，丢掉它这个子集的编辑对就不完整;
# 但它的指令(形如"将“好无聊”改为“很无聊”。")里没有任何可以改写成[V1*]编号占位符的
# 自然语言指代短语，保留它就必须人工拼一句本不存在的引导语。
# 另外该子集编辑后图与原图尺寸还常差几十像素(910x512 vs 896x512)，与其余子集口径不同。
# 所以整体跳过，不做半成品处理
SKIP_MODEL_DIR_NAME_LIST = [
    'textflux',
]

# 图像编辑任务类型，实测5769113条(跳过textflux后)全非空、无一条缺失。
# 直接用它分子集，所以不需要mix兜底子集。
# 注意不能用model字段分子集: 它就是上游的构造模型目录名(bagel/kontext/lama等)，
# 描述的是"这条数据是谁造的"而不是"这是什么编辑任务"，
# 同一个model目录里混着十几种任务(如step1x-edit有11种)
ANNOTATION_TASK_KEY_NAME = 'task'

# 【按方案确认整体丢弃的3个子集】
# personalized_generation(524745对) / portrait(856081对) /
# portrait_editing(102841对)，合计1483667对:
#   这三个子集来自上游的kontext_subject等"主体驱动生成"构造模型，
#   参考图是一张**独立的主体图/人像**、编辑后图是模型按主体重新生成的整图，
#   两者的分辨率与长宽比彼此无关(不是"对同一张图做局部编辑")。
#   本次改动要求"reference_image[0]必须与编辑后图长宽比严格相同、否则整对丢弃"，
#   这三个子集会被大比例判掉; 而且它们本质上是"主体参考生成"而不是"图像编辑"，
#   与其留一批被判剩的残样本，不如按方案在解析阶段整体跳过。
#
# 【为什么要用独立的SKIP_SET_NAME_LIST而不是直接从SAVE_SET_NAME_LIST里删掉】
# check_skip_tail_set的判定是"不在SAVE_SET_NAME_LIST里就算长尾"，
# 而长尾任务名的个数有一条硬对账EXPECTED_SKIP_TAIL_SET_COUNT = 568。
# 如果直接把这三个子集从白名单里删掉，它们就会被算成长尾、
# 让长尾任务名个数变成571而硬对账失败(还会把它们的条数混进长尾统计里、
# 看不出到底丢了多少)。所以这里单独出一个丢弃名单、单独计数、单独硬对账，
# EXPECTED_SKIP_TAIL_SET_COUNT保持568不变。
# 丢弃判定必须排在长尾判定**之前**，否则这三个子集会先被长尾分支吃掉
SKIP_SET_NAME_LIST = [
    'personalized_generation',
    'portrait',
    'portrait_editing',
]

# 被整体丢弃的3个子集的实测**原始标注条数**合计，解析阶段硬对账。
# 这个数字守护的是"丢弃范围没被改动过"。
# 【注意口径】check_skip_set排在所有指令过滤之前，所以skip_set_count统计的是
# 这三个子集在json里的原始条数1507995，而不是它们过滤后的落盘条数1483667
# (= 524745 + 856081 + 102841)。两者相差24328条，正好是这三个子集内部
# 会被空指令/null指令/无文字指令/过短指令/过长指令判掉的那一批样本。
# 早期版本这里误填了过滤后的1483667，导致解析阶段硬对账必然失败
EXPECTED_SKIP_SET_COUNT = 1507995

# 只保留这12个规范英文任务名作为子集，即最终产出12个子集目录。
# 上游task字段实测共583个不同取值，这15个规范值(含下面按方案整体丢弃的3个)
# 占了5679352对(99.79%)，扣掉那3个后本脚本只保留这12个，
# 其余568个是上游标注者随手写的长尾任务名(中文如"证件照生成"/"打光"，
# 英文拼写变体如complex_reasoning/camrea_move_editing/-tone_transfer)，
# 合计只有11876对、其中大量取值只对应1对样本，
# 数据量太少不足以支撑任务学习，按方案确认这些任务连带其样本整体丢弃、不保存。
# 丢弃判定放在"取到task之后、任何图像与指令过滤之前"，
# 所以下面所有指令过滤计数都只统计这12个保留子集的样本
SAVE_SET_NAME_LIST = [
    'action_change',
    'background_change',
    'camera_movement',
    'color_change',
    'material_change',
    'reasoning',
    'style_change',
    'subject_addition',
    'subject_deletion',
    'subject_replacement',
    'text_change',
    'tone_transform',
]

# 编辑指令，也是本数据集唯一一个真正的图像编辑指令字段。
# README官方定义: "instruction: Editing instruction, it could be Chinese or English."
# 实测跳过textflux后的9个构造模型目录100%都有这个key(中英文混杂: kontext/lama/
# kontext_subject/ominiconsistencey是英文，step1x-edit/qwen两目录是中文，
# bagel/gpt4o中英混杂)，语言不做任何过滤，原样保存
ANNOTATION_CAPTION_KEY_NAME = 'instruction'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 新标注固定只存这七个key，多一个少一个都在收尾自校验里报错。
# 上游json里剩下的属性按方案确认全部丢弃、不另存索引:
# 【其它文本字段】(实测逐条比对后确认都不是可直接用的编辑指令)
#   instruction_en : lama的这个字段与instruction逐字100%相同(纯冗余);
#                    gpt4o的只有19.9%覆盖率且是instruction的中译英
#   instruction_zh : lama/kontext_subject是instruction的中译;
#                    kontext的94.5%是中译、但另有5.7%是同语言英文改写
#                    ("taking During Sunrise"->"taken during sunrise")，口径不统一
#   instruction_ori: qwen两个目录专有，README明确写它是"description of the image"，
#                    实测确认是"编辑前原图"的生成prompt而不是编辑指令
#                    (instruction_ori="身材健硕的男人在健身房举铁" vs
#                     instruction="身材健硕的男人在海边沙滩上晒太阳")，
#                    误用它会让参考图与指令完全对不上
#   caption_en / caption_zh / caption_ori / caption_ori_en / caption_ori_zh:
#                    编辑前后整图的图像描述，不是编辑指令，且覆盖率19%~100%各子集不一
#                    (kontext_subject与qwen两目录完全没有这几个字段)
# 【质量分字段】按方案确认一律不用于过滤、也不保存:
#   aesthetic_score / aesthetic_score_v2_5(_edit) / aesthetic_score_ori /
#   aesthetic_score_delete / liqe_score(_edit) / liqe_score_clip(_edit) /
#   liqe_score_ori / liqe_score_delete / liqe_score_clip_ori /
#   liqe_score_clip_delete / score / score_7b / clip_score / clip_score_blip /
#   watermark / watermark_score / IQA_score / IAA_score / clipI / clipT / dino /
#   shuttle_pro / confidence_in / confidence_out / race_conf
#   这些分数各子集覆盖率差异极大(score_7b只有6个目录有、IQA_score只有gpt4o与
#   step1x-edit的部分样本有、qwen两目录一个都没有)，拿它们做阈值过滤会造成
#   子集之间的质量口径严重不一致
# 【其它属性】
#   model    : 上游构造模型目录名，已内含在保存图像名里
#   task     : 只用于推导子集名
#   label    : qwen两目录恒为qwen-image
#   race     : qwen两目录的人种标签，与图像编辑任务无关
#   font     : textflux的OCR文字与四点坐标(该目录已整体跳过)
#   width / height / is_512 / is_1024: 上游记录的原图尺寸，
#              本脚本一律以实际写盘数组的shape为准，不采信上游数值
#   caption_keywords / caption_qwen(_en) / caption_qwenvl_zh / caption_qwenvl_en /
#   ocr_area / ocr_threshold: 少数目录才有的中间产物字段
# 【.txt文件】README标注它是"Editing instruction"，实测8个目录与json的instruction
#   逐字100%一致(纯冗余); qwen两个目录的.txt则是
#   "['<instruction_ori>', '<instruction>']"这种list字符串(把原图prompt和编辑指令
#   拼在了一起)，json里已经拆成两个字段，所以本脚本只读json、完全不读.txt
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# 本数据集每个编辑对只有.1.0.jpg这一张参考图(唯一有第二参考图的textflux已整体跳过)，
# 所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 上游task字段里的中文长尾值(如"证件照生成"/"打光")所在的任务已整体丢弃，
# 保留的12个任务名全是ASCII，所以图像名里不会出现CJK字符，
# 这个白名单与002/006/007完全一致。
# 实测5679352个保存名100%满足这个模式、0个非法名，最长80字符
# (x2edit_personalized_generation_kontext_subject_7_00080_000004999_dup1_edited.jpg)，
# 远低于文件系统单文件名255字节的上限
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 跳过textflux后9个构造模型目录的实测样本对数，合计5769113对。
# 解析阶段按构造模型目录硬对账，少一对都说明上游008没跑完或产物被改动过
EXPECTED_MODEL_ANNOTATION_COUNT_DICT = {
    'bagel': 502582,
    'gpt4o': 232176,
    'kontext': 1628010,
    'kontext_subject': 390561,
    'lama': 911789,
    'ominiconsistencey': 269988,
    'qwen-image-edit-Asian-portrait': 467256,
    'qwen-image-edit-NonAsian-portrait': 388843,
    'step1x-edit': 977908,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = 5769113

# 上游全量6026214对里textflux占257101对，跳过后剩5769113对
EXPECTED_SKIP_MODEL_ANNOTATION_COUNT = 257101

# 5769113对先丢掉按方案整体丢弃的3个子集的1507995对(原始条数口径)、
# 再丢掉568个长尾任务的11876对，再经文本层过滤丢掉53557对(12子集口径)，
# 最终进入图像校验阶段的是4195685对:
#   5769113 - 1507995 - 11876 - 53557 = 4195685
# 注意这个数字是"文本过滤后"的条数，不是最终落盘条数:
# 后面还会按长宽比规则再丢一批(different_aspect_ratio_count)。
# 换个口径也能对上: 5679352(旧的15子集落盘数) - 1483667(3个子集过滤后条数) = 4195685
EXPECTED_VALID_ANNOTATION_COUNT = 4195685

# 被长尾任务过滤丢弃的实测精确数字，解析阶段硬对账。
# 这一步发生在任何图像与指令过滤之前，所以下面那些指令过滤计数
# 统计的都只是12个保留子集里的样本
EXPECTED_SKIP_TAIL_SET_COUNT = 568

EXPECTED_SKIP_TAIL_SET_ANNOTATION_COUNT = 11876

# 文本层各类不合格指令的实测精确条数，解析阶段逐项硬对账。
# 【注意口径】这五个计数**只统计12个保留子集**: check_skip_set与
# check_skip_tail_set都排在指令过滤之前，被整体丢弃的3个主体驱动生成子集与
# 568个长尾任务的样本根本走不到这里。
# 旧的15子集口径是67065/4396/54/6338/32(合计77885)，里面含那3个子集的24328条，
# 换成12子集口径后合计53557条。早期版本漏改这五个常量会让解析阶段硬对账必然失败。
# empty_caption仍占大头且几乎全在bagel，对应上游.txt也是0字节的那批样本，
# 属原始数据自带缺陷
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 44720,
    'null_like_caption_count': 2646,
    'no_word_char_caption_count': 38,
    'too_short_caption_count': 6126,
    'too_long_caption_count': 27,
}

# 12个保留子集的实测条数，解析阶段逐个硬对账。
# 原来是15个子集合计5679352对，扣掉按方案整体丢弃的
# personalized_generation(524745) + portrait(856081) + portrait_editing(102841)
# = 1483667对之后，剩12个子集合计4195685对
EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT = {
    'action_change': 182051,
    'background_change': 163486,
    'camera_movement': 105853,
    'color_change': 214572,
    'material_change': 221572,
    'reasoning': 412396,
    'style_change': 655910,
    'subject_addition': 268900,
    'subject_deletion': 1101247,
    'subject_replacement': 306948,
    'text_change': 183962,
    'tone_transform': 378788,
}

# 最终产出的子集数，必须与SAVE_SET_NAME_LIST严格一一对应(不多也不少)。
# 原来是15个，扣掉按方案整体丢弃的3个主体驱动生成子集后是12个。
# 12个子集全部都在10万对以上，所以每个子集都会切出多个满10000对的文件夹
EXPECTED_SAVE_SET_COUNT = 12

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

# 本数据集有一半样本(约300万对)是中文指令，中文单字信息量远高于英文单词，
# 同一条语义的中文指令字符数只有英文的三分之一左右
# (step1x-edit全中文指令p50只有12、"油画"这种合法的风格类指令只有2个字符)，
# 沿用前面几个纯英文数据集的阈值10会砍掉约17万条完全正常的中文短指令，
# 属于砍正常样本而不是砍异常值，所以按方案确认这个数据集取4。
# 实测12个保留子集里strip后长度小于4的只有6126条(15子集口径是6338条)
MIN_CAPTION_LENGTH = 4

# 实测指令长度p50=29、p90=79、p99=134、p999=195、max=506，
# 长尾很短，12个保留子集里超过512的只有27条(全在kontext，15子集口径是32条)，
# 按方案确认取512，超长整对丢弃
MAX_CAPTION_LENGTH = 512

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 12个保留子集里实测有38条指令strip后只剩标点(如"...")，
# 这类指令没有任何可训练的语义，整对丢弃
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则。
# 12个保留子集里实测被判掉的2646条中绝大多数就是"None"/"None."/"无"这三种写法
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)。
# 分两类:
# 1. null字面量: 上游构造流程失败时把空值写成了字符串，实测有None/无/Empty/Blank等;
# 2. 明确表示"不做任何修改"的指令: no change/不变/无需修改这类，
#    语法上是合法句子但编辑前后图应该几乎相同，当图像编辑训练样本是纯噪声，
#    按方案确认一并丢弃。
# 这里只做全串精确匹配、不做任何启发式猜测，因为上游有大量合法的单名词短指令
# (油画/卡通/水彩画/吉卜力/线描艺术，对应style change任务)，
# 一旦按"是不是单名词"来判会误伤十几万条正常样本
NULL_LIKE_CAPTION_LIST = [
    'null',
    'none',
    'nan',
    'n/a',
    'na',
    'nil',
    'undefined',
    'unknown',
    'empty',
    'blank',
    'no change',
    'no changes',
    'nochange',
    '无',
    '空',
    '无指令',
    '无变化',
    '不变',
    '没有',
    '无需修改',
    '未修改',
]

# 带编号的视觉参考图占位符，编号从"非原图的第1张参考图"起算:
# [V1*]指代reference_image[1]、[V2*]指代reference_image[2]...[VN*]指代
# reference_image[N]，其中N == reference_image_num - 1。
# 编辑前原图reference_image[0]永远隐式、不写进指令、不占编号。
# 这套写法与002/006/007完全一致，保证跨数据集口径统一。
# 本数据集处理的9个构造模型目录都是单参考图，指令里本就不该出现任何占位符
# (实测含[Vn*]形态的0条)，这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2


def get_set_name(per_task_name):
    """把上游task字段归一化成子集名(即图像编辑任务类型)

    统一小写并把空格换成下划线(style change -> style_change)。
    实测583个取值归一化后仍是583个、没有任何两个不同的task被归并到同一个子集名，
    其中只有12个规范英文值会被SAVE_SET_NAME_LIST保留下来
    (另有3个规范值在SKIP_SET_NAME_LIST里被整体丢弃)。
    """
    return str(per_task_name).strip().lower().replace(' ', '_')


def check_skip_set(per_set_name):
    """判定这个子集是不是按方案整体丢弃的3个主体驱动生成子集，返回True表示丢弃

    见SKIP_SET_NAME_LIST的注释: personalized_generation / portrait /
    portrait_editing的参考图是独立主体图、与编辑后图长宽比彼此无关，
    按"第一张参考图必须与编辑后图长宽比相同"这条规则会被大比例判掉，
    所以在解析阶段就整体跳过。
    这个判定必须排在check_skip_tail_set之前，否则这三个子集会先被长尾分支吃掉、
    污染长尾任务名个数与条数的硬对账。
    """
    return per_set_name in SKIP_SET_NAME_LIST


def check_skip_tail_set(per_set_name):
    """判定这个子集名是不是要整体丢弃的长尾任务，返回True表示丢弃

    只保留SAVE_SET_NAME_LIST里的12个规范英文任务名，
    其余568个长尾任务名(合计仅11876对)连带其样本整体丢弃。
    注意按方案整体丢弃的那3个主体驱动生成子集(见SKIP_SET_NAME_LIST)已经在
    check_skip_set里先被跳掉了，不会走到这里、也不会混进长尾统计。
    """
    return per_set_name not in SAVE_SET_NAME_LIST and per_set_name not in SKIP_SET_NAME_LIST


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    本数据集唯一有第二参考图的textflux已经整体跳过，剩下的12个子集全都是
    "只有编辑前原图这一张参考图"，所以这里恒为1。
    保留这个函数是为了和002/006/007的收尾自校验保持同一套交叉对账写法。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def check_invalid_caption(per_ti2i_caption, per_reference_image_num):
    """判定占位符编号与参考图数量不自洽的坏指令，返回True表示这条指令不合格

    参考图里第0张永远是编辑前原图(隐式、不占编号)，所以一条指令应该带的占位符编号
    正好是1...N，其中N = reference_image_num - 1。这里做三条校验:
    1. 同一个编号最多重复2次，超过就是逐字符插占位符的坏指令;
    2. 最大编号必须正好等于N，多了就是指代了不存在的参考图;
    3. 1...N每个编号都必须至少出现一次，不允许跳号，也不允许有图没被指代。
    本数据集N恒为0，即要求指令里完全没有占位符。
    另外还禁止无编号与带编号混用，只做防御性拦截。
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


def check_null_like_caption(per_ti2i_caption):
    """判定指令是不是null字面量或"不做任何修改"这类无意义指令

    先小写化再剥掉两端标点空白，然后与黑名单做全串精确匹配，
    这样"None"/"None."/"无"/"无。"能被同一条规则一次判掉。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().strip(
        CAPTION_STRIP_CHAR)

    return per_ti2i_caption in NULL_LIKE_CAPTION_LIST


def process_single_archive_dir(archive_dir_pair):
    """解析单个压缩包目录里的全部json标注，组装图像编辑对的列表

    这一步只做纯文本层面的过滤(缺任务类型、长尾任务、缺图、指令为空、
    指令是null字面量、指令没有任何文字字符、指令过短、指令过长、
    指令是坏占位符指令、保存名非法)，
    图像本身的解码校验和分辨率过滤留到后面多进程里做。
    长尾任务的丢弃排在最前面(取到task就立刻判)，
    所以后面那些指令过滤计数统计的都只是12个保留子集里的样本。

    这个数据集的图像和标注都散落在1341个压缩包目录里(每个目录约4300个样本对)，
    上游又都放在NAS上，所以这里每个目录只打一次os.listdir拿全量文件名建集合，
    之后判图像是否存在只做集合查表。
    如果按样本逐个os.path.exists，光这一步就是约1150万次网络往返。
    """

    per_model_name, per_shard_name, per_archive_name, per_archive_dir_path = archive_dir_pair

    try:
        file_name_set = set(os.listdir(per_archive_dir_path))
    except Exception as e:
        print('2222', per_archive_dir_path, e)
        file_name_set = set()

    sample_name_prefix_list = sorted([
        per_file_name[:-len(LOAD_ANNOTATION_FILE_NAME_SUFFIX)]
        for per_file_name in file_name_set
        if per_file_name.endswith(LOAD_ANNOTATION_FILE_NAME_SUFFIX)
    ])

    total_annotation_count = len(sample_name_prefix_list)
    load_annotation_failed_count, missing_task_count = 0, 0
    skip_set_count = 0
    skip_tail_set_count = 0
    skip_tail_set_name_count_dict = {}
    missing_image_count = 0
    second_reference_image_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    for per_sample_name_prefix in sample_name_prefix_list:
        per_annotation_path = os.path.join(
            per_archive_dir_path,
            f'{per_sample_name_prefix}{LOAD_ANNOTATION_FILE_NAME_SUFFIX}')

        try:
            with open(per_annotation_path, 'r',
                      encoding='UTF-8') as load_json_file:
                per_annotation = json.load(load_json_file)
        except Exception as e:
            load_annotation_failed_count += 1
            print('2222', per_annotation_path, e)
            continue

        if not isinstance(per_annotation, dict):
            load_annotation_failed_count += 1
            print('2222', per_annotation_path)
            continue

        per_task_name = per_annotation.get(ANNOTATION_TASK_KEY_NAME, '')
        if not isinstance(per_task_name, str):
            per_task_name = ''
        per_task_name = per_task_name.strip()

        # 任务类型决定子集名，缺了就无法安全落盘(实测0条缺失，这里只做防御)
        if not per_task_name:
            missing_task_count += 1
            continue

        per_set_name = get_set_name(per_task_name)

        # 按方案整体丢弃的3个主体驱动生成子集在这里先跳掉(见SKIP_SET_NAME_LIST)。
        # 这一步必须排在长尾判定之前: 否则这三个子集会被算成长尾任务、
        # 把长尾任务名个数从568顶到571而硬对账失败，也看不出到底丢了多少
        if check_skip_set(per_set_name):
            skip_set_count += 1
            continue

        # 长尾任务连带其样本整体丢弃，这一步排在所有图像与指令过滤之前，
        # 保证后面每一项过滤计数都只统计12个保留子集里的样本
        if check_skip_tail_set(per_set_name):
            skip_tail_set_count += 1
            skip_tail_set_name_count_dict[
                per_set_name] = skip_tail_set_name_count_dict.get(
                    per_set_name, 0) + 1
            continue

        per_edited_image_name = f'{per_sample_name_prefix}{LOAD_EDITED_IMAGE_FILE_NAME_SUFFIX}'
        per_reference_image_name = f'{per_sample_name_prefix}{LOAD_REFERENCE_IMAGE_FILE_NAME_SUFFIX}'

        # 编辑后图和参考图缺任意一张，这个编辑对的信息都不完整，整对丢弃
        if per_edited_image_name not in file_name_set or per_reference_image_name not in file_name_set:
            missing_image_count += 1
            continue

        # 只统计不使用: 有第二参考图的textflux已整体跳过，这里应该恒为0
        if f'{per_sample_name_prefix}{LOAD_SECOND_REFERENCE_IMAGE_FILE_NAME_SUFFIX}' in file_name_set:
            second_reference_image_count += 1

        per_edited_image_path = os.path.join(per_archive_dir_path,
                                             per_edited_image_name)
        per_reference_image_path = os.path.join(per_archive_dir_path,
                                                per_reference_image_name)

        per_ti2i_caption = per_annotation.get(ANNOTATION_CAPTION_KEY_NAME,
                                              None)
        if isinstance(per_ti2i_caption, (list, tuple)):
            per_ti2i_caption = per_ti2i_caption[0] if len(
                per_ti2i_caption) > 0 else ''
        if not isinstance(per_ti2i_caption, str):
            per_ti2i_caption = ''
        per_ti2i_caption = per_ti2i_caption.strip()

        # 空指令、全空格指令视为不合格图像编辑对
        # (12个保留子集里实测44720条，绝大多数在bagel)
        if not per_ti2i_caption:
            empty_caption_count += 1
            continue

        # null字面量与"不做任何修改"这类无意义指令同样丢弃(12子集口径实测2646条)
        if check_null_like_caption(per_ti2i_caption):
            null_like_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:50])
            continue

        # 只剩标点、没有任何数字/字母/汉字的指令也丢弃
        # (12子集口径实测38条，形如"...")
        if not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption):
            no_word_char_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:50])
            continue

        # 过短指令视为不合格图像编辑对(12子集口径实测6126条)
        if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
            too_short_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 过长指令同样视为不合格图像编辑对(12子集口径实测27条，全在kontext)
        if len(per_ti2i_caption) > MAX_CAPTION_LENGTH:
            too_long_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 占位符编号与参考图数量不自洽的指令也丢弃。
        # 本数据集是单参考图，即要求指令里完全没有[Vn*]占位符(实测0条命中)
        if check_invalid_caption(per_ti2i_caption, EXPECT_REFERENCE_IMAGE_NUM):
            invalid_placeholder_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:100])
            continue

        # 保存图像名必须拼出全局唯一键: 上游样本编号前缀只是压缩包目录内的序号
        # (形如000000000)，全局只有51371个不同取值、跨1341个目录大面积重复，
        # 直接用它会让569万对塌缩成5万个同名文件夹、绝大部分样本被覆盖丢失。
        # 这里拼上任务类型 + 构造模型名 + 分片编号 + 压缩包编号:
        # 前三者定位到唯一的压缩包目录，目录内的前缀又由上游008用_dupN后缀去过重名，
        # 所以实测5679352个保存名100%唯一、0重名
        per_save_image_name_prefix = f'{DATASET_NAME}_{per_set_name}_{per_model_name.lower()}_{per_shard_name}_{per_archive_name}_{per_sample_name_prefix}'
        per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
        per_save_reference_image_name = f'{per_save_image_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}'

        # 保存名里出现路径分隔符或其它异常字符会写坏目录结构，整对丢弃(实测0条)
        if not VALID_IMAGE_NAME_PATTERN.match(
                per_save_edited_image_name
        ) or not VALID_IMAGE_NAME_PATTERN.match(per_save_reference_image_name):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_save_edited_image_name)
            continue

        # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的前缀
        # (即带_edited那一段)，和002/006/007的写法保持一致，
        # 收尾自校验也是按edited_image去掉.jpg来反推这个文件夹名的
        per_save_pair_folder_name = os.path.splitext(
            per_save_edited_image_name)[0]

        set_annotation_count_dict[
            per_set_name] = set_annotation_count_dict.get(per_set_name, 0) + 1

        edit_annotation_pair_list.append([
            per_set_name,
            per_save_pair_folder_name,
            per_edited_image_path,
            per_save_edited_image_name,
            [per_reference_image_path],
            [per_save_reference_image_name],
            per_ti2i_caption,
            EXPECT_REFERENCE_IMAGE_NUM,
        ])

    return [
        edit_annotation_pair_list,
        per_model_name,
        total_annotation_count,
        load_annotation_failed_count,
        missing_task_count,
        skip_tail_set_count,
        skip_tail_set_name_count_dict,
        missing_image_count,
        second_reference_image_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
        set_annotation_count_dict,
        skip_set_count,
    ]


def get_all_archive_dir_pair(root_dataset_path):
    """扫描上游解压产物，收集需要处理的全部压缩包目录

    上游是X2Edit_data/<构造模型名>/<分片编号>/<压缩包编号>这种四层结构，
    全量1396个压缩包目录，跳过textflux的55个后剩1341个。
    """
    root_data_path = os.path.join(root_dataset_path, *LOAD_DATA_DIR_NAME_LIST)

    archive_dir_pair_list = []
    skip_model_dir_count = 0
    for per_model_name in sorted(os.listdir(root_data_path)):
        per_model_path = os.path.join(root_data_path, per_model_name)
        if not os.path.isdir(per_model_path):
            continue

        # textflux整体跳过，它的第二参考图无法在单参考图口径下完整表达
        if per_model_name in SKIP_MODEL_DIR_NAME_LIST:
            skip_model_dir_count += 1
            continue

        for per_shard_name in sorted(os.listdir(per_model_path)):
            per_shard_path = os.path.join(per_model_path, per_shard_name)
            if not os.path.isdir(per_shard_path):
                continue

            for per_archive_name in sorted(os.listdir(per_shard_path)):
                per_archive_path = os.path.join(per_shard_path,
                                                per_archive_name)
                if not os.path.isdir(per_archive_path):
                    continue

                archive_dir_pair_list.append([
                    per_model_name,
                    per_shard_name,
                    per_archive_name,
                    per_archive_path,
                ])

    return archive_dir_pair_list, skip_model_dir_count


def get_all_edit_annotation_pair(root_dataset_path):
    """扫描上游解压好的1341个压缩包目录，多进程组装全部图像编辑对的列表

    上游合计576万个json标注，逐条还要判两张图像文件是否存在，
    所以这里按压缩包目录粒度开多进程解析(每个目录约4300个样本对)，最后再统一排序。
    """
    archive_dir_pair_list, skip_model_dir_count = get_all_archive_dir_pair(
        root_dataset_path)

    total_annotation_count = 0
    load_annotation_failed_count, missing_task_count = 0, 0
    skip_set_count = 0
    skip_tail_set_count = 0
    skip_tail_set_name_count_dict = {}
    missing_image_count = 0
    second_reference_image_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    model_annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(archive_dir_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_archive_dir, archive_dir_pair_list),
                                    total=len(archive_dir_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            per_model_name = per_load_result[1]
            model_annotation_count_dict[
                per_model_name] = model_annotation_count_dict.get(
                    per_model_name, 0) + per_load_result[2]

            total_annotation_count += per_load_result[2]
            load_annotation_failed_count += per_load_result[3]
            missing_task_count += per_load_result[4]
            skip_tail_set_count += per_load_result[5]

            for per_set_name, per_set_count in per_load_result[6].items():
                skip_tail_set_name_count_dict[
                    per_set_name] = skip_tail_set_name_count_dict.get(
                        per_set_name, 0) + per_set_count

            missing_image_count += per_load_result[7]
            second_reference_image_count += per_load_result[8]
            empty_caption_count += per_load_result[9]
            null_like_caption_count += per_load_result[10]
            no_word_char_caption_count += per_load_result[11]
            too_short_caption_count += per_load_result[12]
            too_long_caption_count += per_load_result[13]
            invalid_placeholder_caption_count += per_load_result[14]
            invalid_save_image_name_count += per_load_result[15]

            for per_set_name, per_set_count in per_load_result[16].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

            skip_set_count += per_load_result[17]

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(archive_dir_pair_list),
        skip_model_dir_count,
        model_annotation_count_dict,
        set_annotation_count_dict,
        skip_tail_set_name_count_dict,
        total_annotation_count,
        load_annotation_failed_count,
        missing_task_count,
        skip_tail_set_count,
        missing_image_count,
        second_reference_image_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
        skip_set_count,
    ]


def check_load_annotation_count(
        model_annotation_count_dict, set_annotation_count_dict,
        skip_tail_set_name_count_dict, total_annotation_count,
        valid_annotation_count, skip_tail_set_count, skip_set_count,
        invalid_caption_count_dict, other_filter_count_dict):
    """解析完标注后硬对账: 构造模型目录、长尾任务丢弃量、指令过滤条数、子集条数

    上游解压产物是一次性解出来的确定结果，条数对不上说明上游008没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账还能额外拦住"任务类型归一化写法被改动"这种模型级对账看不出来的问题。
    被丢弃的长尾任务有568个、逐个硬编码不现实，所以对它们只校验
    "被丢弃的任务个数"和"被丢弃的总条数"，完整明细会写进resave_check_result.json。
    """
    check_error_message_list = []

    for per_model_name in sorted(model_annotation_count_dict.keys()):
        if per_model_name not in EXPECTED_MODEL_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown model {per_model_name}')
            continue

        per_expect_annotation_count = EXPECTED_MODEL_ANNOTATION_COUNT_DICT[
            per_model_name]
        if model_annotation_count_dict[
                per_model_name] != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_model_name} annotation count not match '
                f'{model_annotation_count_dict[per_model_name]} != '
                f'{per_expect_annotation_count}')

    for per_model_name in sorted(EXPECTED_MODEL_ANNOTATION_COUNT_DICT.keys()):
        if per_model_name not in model_annotation_count_dict:
            check_error_message_list.append(f'missing model {per_model_name}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    # 按方案整体丢弃的3个主体驱动生成子集的条数硬对账，
    # 守住"丢弃范围没被改动过"这条口径。
    # 注意这个数字与长尾统计完全分开: 那三个子集在check_skip_set里先被跳掉，
    # 不会走到长尾分支，所以EXPECTED_SKIP_TAIL_SET_COUNT仍然是568
    if skip_set_count != EXPECTED_SKIP_SET_COUNT:
        check_error_message_list.append(
            f'skip set count not match '
            f'{skip_set_count} != {EXPECTED_SKIP_SET_COUNT}')

    # 被整体丢弃的长尾任务: 只校验任务个数与总条数两个汇总量
    if len(skip_tail_set_name_count_dict) != EXPECTED_SKIP_TAIL_SET_COUNT:
        check_error_message_list.append(
            f'skip tail set count not match '
            f'{len(skip_tail_set_name_count_dict)} != '
            f'{EXPECTED_SKIP_TAIL_SET_COUNT}')

    if skip_tail_set_count != EXPECTED_SKIP_TAIL_SET_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'skip tail set annotation count not match '
            f'{skip_tail_set_count} != '
            f'{EXPECTED_SKIP_TAIL_SET_ANNOTATION_COUNT}')

    if skip_tail_set_count != sum(skip_tail_set_name_count_dict.values()):
        check_error_message_list.append(
            f'skip tail set annotation count not self consistent '
            f'{skip_tail_set_count} != '
            f'{sum(skip_tail_set_name_count_dict.values())}')

    # 被丢弃的任务名里不允许出现任何一个应该保留的子集名
    for per_set_name in sorted(skip_tail_set_name_count_dict.keys()):
        if per_set_name in SAVE_SET_NAME_LIST:
            check_error_message_list.append(
                f'save set {per_set_name} wrongly skipped')
        # 按方案整体丢弃的那3个子集也不允许混进长尾统计里(那说明判定顺序被改错了)
        if per_set_name in SKIP_SET_NAME_LIST:
            check_error_message_list.append(
                f'skip set {per_set_name} wrongly counted as a tail set')

    # 保留白名单与整体丢弃名单不允许有交集
    for per_skip_set_name in SKIP_SET_NAME_LIST:
        if per_skip_set_name in SAVE_SET_NAME_LIST:
            check_error_message_list.append(
                f'skip set {per_skip_set_name} also in save set name list')

    for per_count_name in sorted(EXPECTED_INVALID_CAPTION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_CAPTION_COUNT_DICT[per_count_name]
        if invalid_caption_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_caption_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    if valid_annotation_count != EXPECTED_VALID_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'valid annotation count not match '
            f'{valid_annotation_count} != {EXPECTED_VALID_ANNOTATION_COUNT}')

    # 【过滤链路恒等式自校验】总条数减去每一项被丢弃的条数必须正好等于保留条数。
    # 这一条不依赖任何硬编码的期望值，纯粹校验"各项计数之间自洽"，
    # 专门用来拦住"某个EXPECTED_*常量口径被改过/没跟着改"这类问题:
    # 上面那些逐项对账各自都可能因为口径漂移而误报或漏报，
    # 但只要这个恒等式不成立，就一定是计数逻辑或统计口径出了问题。
    # (历史教训: EXPECTED_SKIP_SET_COUNT曾误填成"过滤后"的条数、
    #  EXPECTED_INVALID_CAPTION_COUNT_DICT曾停留在15子集口径，
    #  两边的偏差刚好互相抵消，导致valid总数反而对得上、掩盖了问题)
    per_all_filter_count = (skip_set_count + skip_tail_set_count +
                            sum(invalid_caption_count_dict.values()) +
                            sum(other_filter_count_dict.values()))
    if total_annotation_count - per_all_filter_count != valid_annotation_count:
        check_error_message_list.append(
            f'annotation filter count not self consistent '
            f'{total_annotation_count} - {per_all_filter_count} != '
            f'{valid_annotation_count}')

    # 12个保留子集逐个硬对账
    for per_set_name in sorted(EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT.keys()):
        per_expect_set_annotation_count = EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT[
            per_set_name]
        if per_set_name not in set_annotation_count_dict:
            check_error_message_list.append(f'missing save set {per_set_name}')
            continue
        if set_annotation_count_dict[
                per_set_name] != per_expect_set_annotation_count:
            check_error_message_list.append(
                f'{per_set_name} set annotation count not match '
                f'{set_annotation_count_dict[per_set_name]} != '
                f'{per_expect_set_annotation_count}')

    # 保留下来的子集集合必须与白名单严格一一对应，不允许多出任何一个子集
    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in SAVE_SET_NAME_LIST:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

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
    本数据集会产出12个子集(长尾任务与3个主体驱动生成子集已在解析阶段整体丢弃)，
    每个子集都在10万对以上，
    所以每个子集都会切出多个满10000对的文件夹，实测合计576个文件夹。
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

    上游图像本来就全是jpg，这里重新编码一遍只是为了统一编码参数，
    像素尺寸不做任何改动。
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

    每条标注固定只有SAVE_ANNOTATION_KEY_NAME_LIST这七个key，
    上游json里剩下的图像描述字段(caption_*)、指令的其它语言版本(instruction_en/
    instruction_zh)、原图生成prompt(instruction_ori)以及二十多个质量分字段
    全部丢弃，理由见文件开头SAVE_ANNOTATION_KEY_NAME_LIST的注释。
    ti2i_caption写的就是上游instruction字段strip后的原文(中英文都原样保留)。
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
        # 再和该子集应有的参考图数量(本数据集恒为1)交叉对账，不一致只上报不改数值
        if per_expect_reference_image_num >= 0 and per_reference_image_num != per_expect_reference_image_num:
            reference_image_num_mismatch_count += 1
            print('9999', per_save_edited_image_name, per_reference_image_num,
                  per_expect_reference_image_num)

        # ti2i_caption_length直接取即将写进json的这个字符串的长度，
        # 保证记录的长度和ti2i_caption永远自洽(该字符串已strip过)
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
    json没记录的残留样本对文件夹。另外还要复检ti2i_caption: 占位符编号集合必须与
    reference_image这个list的长度自洽、长度必须在阈值区间内、不能是null字面量或
    只剩标点的无意义指令、记录的长度必须与字符串实际长度一致。
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
                if not VALID_IMAGE_NAME_PATTERN.match(
                        per_save_edited_image_name):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image name not match pattern'
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
                # 本数据集全部12个子集都必须是单参考图
                if per_annotation[
                        'reference_image_num'] != per_expect_reference_image_num:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match set {per_annotation["reference_image_num"]} != {per_expect_reference_image_num}'
                    )
                # ti2i_caption的占位符编号集合必须与reference_image这个list的
                # 长度自洽: 单参考图即要求指令里完全没有占位符
                if check_invalid_caption(
                        per_annotation['ti2i_caption'],
                        len(per_annotation['reference_image'])):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} caption placeholder index not match reference image num {len(per_annotation["reference_image"])}'
                    )
                # 落盘后的指令里不允许再残留null字面量或"不做任何修改"这类无意义指令
                if check_null_like_caption(per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a null like caption'
                    )
                # 也不允许残留只剩标点、没有任何数字/字母/汉字的指令
                if not CAPTION_WORD_CHAR_PATTERN.search(
                        per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a no word char caption'
                    )
                # json里存的就是过滤时判定的那个字符串，两者口径一致，
                # 这里直接量json里的长度复检
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

    edit_annotation_pair_list, total_archive_dir_count, skip_model_dir_count, model_annotation_count_dict, set_annotation_count_dict, skip_tail_set_name_count_dict, total_annotation_count, load_annotation_failed_count, missing_task_count, skip_tail_set_count, missing_image_count, second_reference_image_count, empty_caption_count, null_like_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, invalid_placeholder_caption_count, invalid_save_image_name_count, skip_set_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_archive_dir_count, skip_model_dir_count,
          total_annotation_count, load_annotation_failed_count,
          missing_task_count, skip_set_count, skip_tail_set_count,
          len(skip_tail_set_name_count_dict), missing_image_count,
          second_reference_image_count, empty_caption_count,
          null_like_caption_count, no_word_char_caption_count,
          too_short_caption_count, too_long_caption_count,
          invalid_placeholder_caption_count, invalid_save_image_name_count,
          len(set_annotation_count_dict), len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    invalid_caption_count_dict = {
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
    }

    # 除了子集丢弃与指令过滤之外剩下的几项丢弃计数，
    # 只给check_load_annotation_count里的过滤链路恒等式自校验用。
    # 这四项实测都是0，但必须一并算进恒等式，否则上游数据一旦出现缺图/缺任务，
    # 恒等式会误报成"计数不自洽"
    other_filter_count_dict = {
        'load_annotation_failed_count': load_annotation_failed_count,
        'missing_task_count': missing_task_count,
        'missing_image_count': missing_image_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        model_annotation_count_dict, set_annotation_count_dict,
        skip_tail_set_name_count_dict, total_annotation_count,
        len(edit_annotation_pair_list), skip_tail_set_count, skip_set_count,
        invalid_caption_count_dict, other_filter_count_dict)

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

    print('3333', 'total archive dir:', total_archive_dir_count,
          'skip model dir:', skip_model_dir_count, 'total annotation:',
          total_annotation_count, 'load annotation failed:',
          load_annotation_failed_count, 'missing task:', missing_task_count,
          'skip set:', skip_set_count, 'skip tail set:',
          skip_tail_set_count, 'skip tail set num:',
          len(skip_tail_set_name_count_dict), 'missing image:',
          missing_image_count, 'second reference image:',
          second_reference_image_count, 'empty caption:', empty_caption_count,
          'null like caption:', null_like_caption_count,
          'no word char caption:', no_word_char_caption_count,
          'too short caption:', too_short_caption_count, 'too long caption:',
          too_long_caption_count, 'invalid placeholder caption:',
          invalid_placeholder_caption_count, 'invalid save image name:',
          invalid_save_image_name_count, 'invalid image:', invalid_image_count,
          'different aspect ratio:', different_aspect_ratio_count,
          'save edit pair failed:',
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
        'total_archive_dir_count': total_archive_dir_count,
        'skip_model_dir_count': skip_model_dir_count,
        'skip_model_annotation_count': EXPECTED_SKIP_MODEL_ANNOTATION_COUNT,
        'total_annotation_count': total_annotation_count,
        'load_annotation_failed_count': load_annotation_failed_count,
        'missing_task_count': missing_task_count,
        # 按方案整体丢弃的3个主体驱动生成子集的条数与子集名
        'skip_set_count': skip_set_count,
        'skip_set_name_list': SKIP_SET_NAME_LIST,
        'skip_tail_set_count': skip_tail_set_count,
        'skip_tail_set_name_count': len(skip_tail_set_name_count_dict),
        'missing_image_count': missing_image_count,
        'second_reference_image_count': second_reference_image_count,
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
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
        'model_annotation_count_dict': model_annotation_count_dict,
        'set_annotation_count_dict': set_annotation_count_dict,
        'skip_tail_set_name_count_dict': skip_tail_set_name_count_dict,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/X2Edit-Dataset'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
