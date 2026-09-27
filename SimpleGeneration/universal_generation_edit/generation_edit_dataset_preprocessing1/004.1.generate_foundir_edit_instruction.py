# coding:utf8
import os
import re
import json
import random
import hashlib
import itertools
import collections

from tqdm import tqdm
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool

# ==============================================================================
# 为009.unzip_foundir_dataset.py解压出来的FoundIR数据集补齐图像编辑文本指令
#
# 【为什么需要这个脚本】
# 009那一步已经把920224个样本对(LQ退化图 + GT清晰图)全部解压落盘并写出
# unzip_annotations/<group_name>/<块号>.jsonl，但FoundIR原始数据里一个文本字都没有，
# 标注里只有group_name和degradation_type_list两个客观标签可以分辨具体是哪种复原任务。
# 下游ti2i训练必须有ti2i_caption，所以这里把"退化类型标签"翻译成
# "人写风格的图像编辑指令"。
#
# 【本脚本完全离线: 不调用任何大模型API】
# 图像复原是"固定操作 + 封闭退化词表"的模板类任务，指令的全部信息量就是
# "对哪几种退化做复原"，而这几种退化已经由group_name唯一确定了。
# 也就是说指令文本可以完全由本地词表组合出来，不需要任何模型参与:
#   骨架一: 动词短语表 × 该组每种退化的名词措辞 × 位置短语表
#           like "remove the blur from the image"
#   骨架二: 修复类动词表 × 该组每种退化的形容词措辞 × 图像名词表
#           like "fix this blurry photo"
# 笛卡尔积之后逐条过validate_single_instruction硬校验，实测每组合法候选:
#   单退化组 4895~5031条、双退化组 38273~39075条、三退化组 303350条，
# 全部远超INSTRUCTION_POOL_TARGET_NUM(4000)，所以每组池子必定能攒满。
# 相比调模型的好处: 秒级完成、零网络依赖、零token成本、结果完全可复现，
# 而且指令在"构造时"就已满足全部风格约束，不存在模型跑偏产出废句的情况。
#
# 【指令风格的对标数据集: anyedit-split】
# 实测/root/autodl-tmp/ti2i_datasets/anyedit-split里25个编辑任务
# (每任务随机抽300条、扫描648~20000条)的ti2i_caption规格:
#   长度      : 绝大多数任务中位27~41字符 / 4~8个单词;
#               只有implicit_change(56字符)和visual_*系列(53~65字符)偏长;
#   详细程度  : 一句祈使句，只说"动作 + 对象/属性"，不描写画面内容、不解释目的、
#               没有"so that / in order to / to restore ..."这种目的从句，
#               绝大多数不带逗号;
#   大小写标点: 模板类任务(add/remove/replace/color_alter/tone_transfer/
#               material_change/movement/action_change)首字母小写、句末无句号;
#               visual_*/style_change/rotation_change/outpaint首字母大写但仍无句号;
#               只有人写的counting/relation才带句号;
#   多样性    : 2~4个同义动词轮换 + 一个封闭属性词表，扫描级unique率
#               outpaint 0.0002、style_change 0.002、rotation_change 0.002、
#               tone_transfer 0.024、background_change 0.061、material_change 0.294、
#               add 0.657，也就是说"同一条指令被成千上万个样本复用"是这个数据集的常态。
# 图像复原本身就是"固定操作 + 封闭退化词表"的模板类任务(和tone_transfer/
# style_change同档)，所以这里的目标规格定为:
#   单退化组 4~10词/18~55字符，双退化组 ≤14词/≤80字符，三退化组 ≤18词/≤105字符，
#   首字母小写、句末无句号、不含逗号分号冒号、不描写画面内容、不写目的从句。
#
# 【为什么是"指令池 + 哈希分配"而不是"每条样本单独生成"】
# 样本对有920224个，但指令的信息量只有"退化组合"这一个维度(共17种取值)，
# 逐样本生成没有任何意义，只会把同样的17种内容重复生成92万次。
# 本脚本:
#   1) 按17个退化组(8种退化基元的17种组合)各建一个指令池，
#      本地词表笛卡尔积产出全部候选 -> 定序后用固定种子shuffle ->
#      逐条硬校验并去重 -> 攒够INSTRUCTION_POOL_TARGET_NUM条;
#   2) 指令池落盘seed_instruction_pool/<group_name>.json，可复用可审查
#      (目录名沿用历史命名，下游004.resave_foundir_ti2i_dataset.py按此名读取);
#   3) 本地按md5(group_name + sample_key)取模从池里选一条分配给样本，
#      同一个sample_key每次跑都拿到同一条指令(可重复、可增量、可对账)。
# 这样每组unique率约0.027~0.10，正好落在anyedit的模板类任务区间里。
#
# 【指令与样本任务如何保证正确匹配】
# - 指令池是按group_name建的，一个组的池子只会分给这个组的样本，不存在跨任务串味;
# - 候选是按该组degradation_type_list逐项取词组合出来的，天然"每种退化都提到、
#   且绝不会提到该组没有的退化"，这一点由构造方式本身保证，不依赖任何后验过滤;
# - 脚本侧validate_single_instruction再硬查一遍:
#   该组每种退化必须命中至少一个同义词(inclusion)，
#   非该组的退化专属词一个都不许出现(exclusion)，
#   例如11Raindrop只能说droplet/raindrop不能说rain streak，
#   14Lowlight只能说brighten/underexposed不能说night/noise;
# - 校验不过的候选一律丢弃并计入invalid，不会被写进池子，也不会分配给样本。
#
# 【输出目录规格】
# <save_dataset_path>/FoundIR/
# ├── seed_instruction_pool/<group_name>.json               每组的指令池(可复用)
# ├── seed_instruction_annotations/<group_name>/<块号>.jsonl 每行=1个样本对+ti2i_caption
# │                                                         文件名与unzip_annotations一一对应
# └── seed_instruction_check_result.json                     全量对账报告
#
# 【本脚本如何保证每条标注样本都拿到正确指令】
# 1) 输入以unzip_annotations为唯一真值: 每个组的输入文件名集合、每个文件的行数、
#    每组总行数都必须与009实测期望值(GROUP_CONFIG_DICT扣掉
#    KNOWN_UPSTREAM_MISSING_SAMPLE_KEY_DICT里已知的上游缺失)对上，
#    总行数必须等于920224 - 上游已知缺失总数(实测920223);
# 2) 输出文件名集合必须与输入文件名集合完全一致，多出来的陈旧文件直接删掉，
#    不允许上一版跑一半留下的文件混在训练目录里;
# 3) 每行输出都要求ti2i_caption非空且再过一遍validate_single_instruction，
#    不合规计入invalid_caption并硬失败;
# 4) 每组指令池数量不足MIN_INSTRUCTION_POOL_NUM直接失败，避免几万个样本共用几十条指令;
# 5) 任一对账不过汇总后抛异常并sys.exit(1)，不静默跑过。
# ==============================================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 每个退化组的[退化类型列表, 009实测样本对数]。
# 退化类型列表就是009里从组名机械拆出来的客观标签，指令池只能按这个建;
# 样本对数是对账用的ground truth，输入标注行数和输出标注行数都必须和它一致。
GROUP_CONFIG_DICT = {
    '01Blur': [['blur'], 109480],
    '02Blur_Noise': [['blur', 'noise'], 29950],
    '03Blur_JPEG': [['blur', 'jpeg'], 29940],
    '04Blur_Noise_JPEG': [['blur', 'noise', 'jpeg'], 29950],
    '05Noise': [['noise'], 58015],
    '06JPEG': [['jpeg'], 59950],
    '07Noise_JPEG': [['noise', 'jpeg'], 29950],
    '08Haze': [['haze'], 79800],
    '09Lowlight_Haze': [['lowlight', 'haze'], 79800],
    '10Rain': [['rain'], 39900],
    '11Raindrop': [['raindrop'], 44828],
    '12NightRain': [['night', 'rain'], 40111],
    '13Rain_Haze': [['rain', 'haze'], 79750],
    '14Lowlight': [['lowlight'], 39962],
    '15Lowlight_Blur': [['lowlight', 'blur'], 85893],
    '16Lowlight_Noise': [['lowlight', 'noise'], 52995],
    '17Lowlight_JPEG': [['lowlight', 'jpeg'], 29950],
}

# 17组样本对数首尾相接，合计920224
EXPECTED_TOTAL_SAMPLE_PAIR_COUNT = 920224

# 009解压时已知失败、因此unzip_annotations里本来就没有的样本主干。
# 实测013Rain_Haze/0673776这一个成员在原始分卷zip里就是坏的:
#   009的unzip_check_result.json里明确记了
#   '13Rain_Haze/21 error num 1 [13Rain_Haze/0673776 member size not match
#    0673776.jpg 1179648 != 2212797]'
#   incomplete_sample_pair_list长度1、13Rain_Haze只写出79749行、总行数920223。
# 也就是说这个洞是上游数据本身的问题(下载的分卷被截断)，本脚本无法也不应该补，
# 但必须显式登记在这里，否则:
#   a) 不登记就把期望值改成79749/920223 -> 之后真丢样本了对账查不出来;
#   b) 不登记又保留79750/920224 -> 每次跑都必然失败，脚本永远跑不完。
# 登记之后期望值按组自动扣减，任何"不在这张表里"的缺失仍然会硬失败。
# 上游重新下载补齐13Rain_Haze分卷并重跑009之后，把这里清空即可。
KNOWN_UPSTREAM_MISSING_SAMPLE_KEY_DICT = {
    '13Rain_Haze': ['0673776'],
}

# 每种退化类型允许使用的措辞(inclusion词表)。
# 一条候选指令必须命中该组每种退化的至少一个词，否则说明模型漏了一种退化，
# 指令就和样本任务对不上，必须丢弃。
DEGRADATION_TYPE_INCLUDE_WORD_DICT = {
    'blur': [
        'blur',
        'blurry',
        'blurred',
        'blurriness',
        'defocus',
        'out-of-focus',
        'out of focus',
        'unfocused',
        'sharpen',
        'unsharp',
        'smear',
    ],
    'noise': [
        'noise',
        'noisy',
        'denoise',
        'noisegranule',
        'speckle',
        'speckled',
    ],
    'jpeg': [
        'jpeg',
        'compression',
        'compressed',
        'blocky',
        'blockiness',
        'banding',
        'macroblock',
    ],
    'haze': [
        'haze',
        'hazy',
        'dehaze',
        'fog',
        'foggy',
        'mist',
        'misty',
    ],
    'lowlight': [
        'low light',
        'low-light',
        'lowlight',
        'underexposed',
        'underexposure',
        'brighten',
        'brightness',
        'brighter',
        'dark',
        'darkness',
        'dim',
        'dimly lit',
        'exposure',
        'illuminate',
        'lighten',
        'light up',
    ],
    'rain': [
        'rain',
        'rainy',
        'rainfall',
        'raining',
        'derain',
        'downpour',
        'rain streak',
    ],
    'raindrop': [
        'dropletmark',
    ],
    'night': [
        'night',
        'nighttime',
        'night-time',
        'nocturnal',
    ],
}

# 每种退化类型的专属措辞(exclusion词表)。
# 这些词只能出现在"该组确实有这种退化"的指令里;
# 出现在别的组说明模型编了一种这个样本根本没有的退化，指令会误导训练，必须丢弃。
# lowlight的exclusion故意不放dark/dim/brightness这类泛用亮度词(会误伤rain/haze组的
# 正常措辞)，只留真正指代"低照度退化"的词。
DEGRADATION_TYPE_EXCLUDE_WORD_DICT = {
    'blur': [
        'blur',
        'blurry',
        'blurred',
        'blurriness',
        'defocus',
        'out-of-focus',
        'out of focus',
        'unfocused',
        'sharpen',
        'unsharp',
    ],
    'noise': [
        'noise',
        'noisy',
        'denoise',
        'noisegranule',
        'speckle',
        'speckled',
    ],
    'jpeg': [
        'jpeg',
        'compression',
        'compressed',
        'blocky',
        'blockiness',
        'banding',
        'macroblock',
    ],
    'haze': [
        'haze',
        'hazy',
        'dehaze',
        'fog',
        'foggy',
        'mist',
        'misty',
    ],
    'lowlight': [
        'low light',
        'low-light',
        'lowlight',
        'underexposed',
        'underexposure',
        'brighten',
    ],
    'rain': [
        'rain',
        'rainy',
        'rainfall',
        'raining',
        'derain',
        'downpour',
    ],
    'raindrop': [
        'dropletmark',
    ],
    'night': [
        'night',
        'nighttime',
        'night-time',
        'nocturnal',
    ],
}

# rain这个词面被别的合法措辞包含，直接做子串匹配会大面积误判(实测踩过):
#   "raindrop"里带"rain"  -> 11Raindrop的正常指令被判成"提到了没有的rain退化",
#                            该组一条都收不上来;
#   "grain"里也带"rain"   -> 05Noise说"colour grain"(prompt里就是这么描述噪声的)
#                            同样被判成提到了雨。
# 所以校验前先把雨滴措辞和噪点颗粒措辞各归一成一个占位词，
# 占位词本身必须不含"rain"子串(raindropmark/grainmark都还带着rain，不能用)，
# 归一之后rain的匹配就只会命中真正的雨条措辞。
RAINDROP_NORMALIZE_WORD_LIST = [
    'raindrops',
    'raindrop',
    'rain drops',
    'rain drop',
    'water droplets',
    'water droplet',
    'water drops',
    'water drop',
    'waterdrops',
    'waterdrop',
    'droplets',
    'droplet',
    'dew drops',
    'dew drop',
]

RAINDROP_NORMALIZE_MARK_WORD = 'dropletmark'

NOISE_GRAIN_NORMALIZE_WORD_LIST = [
    'colour graininess',
    'color graininess',
    'graininess',
    'colour grains',
    'color grains',
    'colour grain',
    'color grain',
    'grainy',
    'grains',
    'grain',
]

NOISE_GRAIN_NORMALIZE_MARK_WORD = 'noisegranule'

# ------------------------------------------------------------------------------
# 指令池的本地词表。两套骨架各用一组词表，笛卡尔积之后就是该组的全部候选指令。
#
# 骨架一(NOUN_PATTERN_INDEX): 动词短语 + 退化名词短语(多退化用and连接) + 位置短语
#   like "remove the blur from the image" / "clean up the noise and the haze"
# 骨架二(ADJECTIVE_PATTERN_INDEX): 修复动词 + this + 退化形容词 + 图像名词
#   like "fix this blurry photo" / "clean up this dark noisy image"
#
# 所有措辞都必须落在DEGRADATION_TYPE_INCLUDE_WORD_DICT的词面里，
# 否则validate_single_instruction会判"miss degradation type"直接丢弃。
# ------------------------------------------------------------------------------

# 骨架一的动词短语: anyedit模板类任务都是同义动词轮换(remove/erase/delete)，
# 这里取30个同档的"去除/修复"动词，保证池子的动词分布像anyedit那样集中但不单一。
LOCAL_VERB_PHRASE_LIST = [
    'remove',
    'erase',
    'delete',
    'clear',
    'clear out',
    'clean up',
    'clean off',
    'clean',
    'wipe out',
    'wipe off',
    'get rid of',
    'take out',
    'take away',
    'strip out',
    'eliminate',
    'cut out',
    'cut down',
    'reduce',
    'kill',
    'kill off',
    'knock out',
    'scrub off',
    'scrub out',
    'purge',
    'blot out',
    'fix',
    'correct',
    'repair',
    'undo',
    'cancel out',
]

# 骨架一的退化名词短语: 每种退化8个措辞，用来在"同一个动作"下变换退化的说法。
# 每个短语自带the，组合时多退化之间用and连接。
LOCAL_DEGRADATION_NOUN_PHRASE_DICT = {
    'blur': [
        'the blur',
        'the blurriness',
        'the motion blur',
        'the defocus blur',
        'the out of focus blur',
        'the blurry smear',
        'the unsharp blur',
        'the smeared blur',
    ],
    'noise': [
        'the noise',
        'the sensor noise',
        'the colour noise',
        'the noise grain',
        'the speckle noise',
        'the noisy speckle',
        'the random noise',
        'the grainy noise',
    ],
    'jpeg': [
        'the jpeg artifacts',
        'the compression artifacts',
        'the jpeg blocks',
        'the jpeg blockiness',
        'the compression banding',
        'the jpeg macroblocks',
        'the jpeg damage',
        'the blocky jpeg artifacts',
    ],
    'haze': [
        'the haze',
        'the fog',
        'the hazy veil',
        'the mist',
        'the foggy haze',
        'the misty fog',
        'the hazy fog',
        'the haze veil',
    ],
    'lowlight': [
        'the low light',
        'the underexposure',
        'the low light cast',
        'the lowlight dimness',
        'the underexposed look',
        'the low-light murk',
        'the dim low light',
        'the low light gloom',
    ],
    'rain': [
        'the rain',
        'the rain streaks',
        'the rainfall',
        'the heavy rain',
        'the falling rain',
        'the rainy streaks',
        'the rain lines',
        'the downpour',
    ],
    'raindrop': [
        'the raindrops',
        'the water droplets',
        'the raindrop marks',
        'the stuck droplets',
        'the dew drops',
        'the water drops',
        'the lens droplets',
        'the blocking raindrops',
    ],
    'night': [
        'the night murk',
        'the night gloom',
        'the night darkness',
        'the nighttime murk',
        'the nocturnal gloom',
        'the nighttime gloom',
        'the night dimness',
        'the nighttime darkness',
    ],
}

# 骨架一的位置短语: 对应anyedit里"remove the cow from the field"的尾部成分。
# 空串表示"只说动作+对象"这种最短的形态(anyedit里占比最高)。
LOCAL_SENTENCE_TAIL_LIST = [
    '',
    'from the image',
    'from the photo',
    'from this image',
    'from this photo',
    'in the image',
    'in the photo',
    'in this image',
    'in this photo',
    'on the image',
    'on the photo',
    'across the image',
    'across the photo',
    'over the image',
    'all over the image',
    'from the whole image',
    'in the whole photo',
    'from this picture',
    'in this picture',
    'on this picture',
]

# 骨架二的动词: "fix this blurry photo"这种句式里只能用修复类动词，
# 不能用remove(remove this blurry photo语义就错了)。
LOCAL_FIX_VERB_PHRASE_LIST = [
    'fix',
    'clean up',
    'restore',
    'repair',
    'correct',
    'clear up',
    'sort out',
    'rescue',
    'touch up',
    'clean',
]

# 骨架二的退化形容词: 直接修饰图像名词。
LOCAL_DEGRADATION_ADJECTIVE_DICT = {
    'blur': [
        'blurry',
        'blurred',
        'unfocused',
        'out of focus',
        'defocused',
        'smeared',
    ],
    'noise': [
        'noisy',
        'speckled',
        'grainy',
    ],
    'jpeg': [
        'blocky',
        'compressed',
        'jpeg damaged',
    ],
    'haze': [
        'hazy',
        'foggy',
        'misty',
    ],
    'lowlight': [
        'underexposed',
        'dark',
        'dim',
        'dimly lit',
        'low light',
        'low-light',
    ],
    'rain': [
        'rainy',
        'rain streaked',
        'rain hit',
    ],
    'raindrop': [
        'raindrop covered',
        'droplet covered',
        'droplet blocked',
    ],
    'night': [
        'nighttime',
        'nocturnal',
        'night time',
    ],
}

LOCAL_IMAGE_NOUN_LIST = [
    'image',
    'photo',
    'picture',
    'shot',
]

# 两套骨架在指令池元素里的编号，落盘后写进instruction_sentence_pattern_index，
# 方便事后按骨架统计池子的句式分布。
NOUN_SENTENCE_PATTERN_INDEX = 0

ADJECTIVE_SENTENCE_PATTERN_INDEX = 1

SAVE_IMAGE_DIR_NAME = 'images'

LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

SAVE_INSTRUCTION_POOL_DIR_NAME = 'seed_instruction_pool'

SAVE_ANNOTATION_DIR_NAME = 'seed_instruction_annotations'

SAVE_CHECK_RESULT_FILE_NAME = 'seed_instruction_check_result.json'

# 指令来源: 本地词表组合，不涉及任何大模型API。
# 这个值会写进每行标注的ti2i_caption_source字段，必须如实反映来源。
INSTRUCTION_SOURCE_NAME = 'local_template'

INSTRUCTION_STYLE_REFERENCE_NAME = 'anyedit-split'

TASK_TYPE_NAME = 'image_restoration_edit'

# 每组指令池的目标条数。
# 4000条时最大的01Blur(109480对)unique率约0.037、最小的03Blur_JPEG(29940对)约0.13，
# 正好落在anyedit模板类任务的unique率区间(0.0002~0.294)里。
# 本地词表实测最小的组(单退化组)也能产出4895条合法候选，所以每组必定能攒满。
INSTRUCTION_POOL_TARGET_NUM = 4000

# 每组指令池的最低条数。本地组合是确定性的、正常情况下每组都会攒满到目标值，
# 这一项留着兜底: 之后有人改窄词表或改严校验规则时能立刻发现池子塌了，
# 而不是让几万个样本静默共用几十条指令。
MIN_INSTRUCTION_POOL_NUM = 1500

# 单条指令的长度门槛。单退化组对齐anyedit中位27~41字符/4~8词的规格，
# 每多一种退化放宽一档(必须多提一种退化，句子必然更长)。
MIN_INSTRUCTION_WORD_NUM = 3

MAX_INSTRUCTION_WORD_NUM_PER_DEGRADATION_TYPE = [10, 14, 18]

MIN_INSTRUCTION_CHAR_NUM = 15

MAX_INSTRUCTION_CHAR_NUM_PER_DEGRADATION_TYPE = [55, 80, 105]

# anyedit的祈使句模板里不会出现的东西: 目的从句、解释、称呼、逗号分号冒号。
# 命中任意一项说明这条指令的详细程度已经超出anyedit规格，直接丢弃。
INSTRUCTION_FORBID_WORD_LIST = [
    'so that',
    'in order to',
    'so as to',
    'to restore',
    'to recover',
    'to bring back',
    'to make it',
    'to produce',
    'to reveal',
    'to get',
    'which',
    'while',
    'because',
    'please',
    'you ',
    'your ',
    'should',
    'must',
    'let ',
    'instruction',
    'image editing',
    'as you can',
    'high quality',
    'high-quality',
    'photorealistic',
]

INSTRUCTION_FORBID_CHAR_LIST = [
    ',',
    ';',
    ':',
    '"',
    "'",
    '`',
    '*',
    '#',
    '(',
    ')',
    '[',
    ']',
    '{',
    '}',
    '/',
    '\\',
    '|',
    '=',
    '<',
    '>',
    '_',
]

# 行首编号/项目符号。本地组合不会产出这种前缀，但clean_single_instruction
# 同时也用来清洗已落盘池子里的历史指令，所以这个兜底保留。
INSTRUCTION_LINE_PREFIX_PATTERN = re.compile(r'^\s*(?:[-*•>]+|\d+[\.\)、])\s*')

MAX_SAVE_PROBLEM_ITEM_NUM = 10000

# 写标注是纯磁盘IO任务，线程池比进程池合适:
# 不用给每个worker复制一份920224行的标注数据。
WRITE_ANNOTATION_THREAD_NUM = 32

# True: 已有seed_instruction_pool/<group_name>.json时直接复用，不再重新组合。
# 指令池复用 + 哈希分配是确定性的，所以重跑一遍每个sample_key拿到的指令完全不变。
# 注意本地组合本身也是确定性的(固定种子)，所以这个开关只影响"是否读盘"，
# 置False重新组合出来的池子和盘上的完全一致。
REUSE_INSTRUCTION_POOL_FLAG = True

# True: 输出目录里出现输入目录没有的陈旧jsonl(上一版跑一半留下的)就删掉。
# 不删的话训练目录里会混进风格不一致的旧标注。
REMOVE_STALE_ANNOTATION_FILE_FLAG = True

# 池子里的指令至少要有这个比例被真正分配给样本。
# 不能要求100%: md5是均匀分配但仍然是随机撞桶，按coupon collector，
# 池子4000条、组内样本29940个时，期望有4000*(1-1/4000)^29940 ≈ 2条一次都没被撞到，
# 要求严格相等会在最小的几个组上必然误报。
# 低于这个比例才是真有问题(池子读错、哈希退化成常数)。
MIN_USED_INSTRUCTION_RATIO = 0.99

# 只跑前几个组、每组只跑一个文件用来验风格。0表示全量。
DEBUG_GROUP_NUM = 0

DEBUG_ANNOTATION_FILE_NUM_PER_GROUP = 0


def get_group_expected_sample_pair_count(per_group_name):
    """该组在unzip_annotations里应该有多少行

    = 009中央目录里数出来的样本对数 - 009解压失败因此本来就没写出来的样本数。
    扣减项只认KNOWN_UPSTREAM_MISSING_SAMPLE_KEY_DICT里登记过的主干，
    没登记的缺失一律算错。
    """
    return GROUP_CONFIG_DICT[per_group_name][1] - len(
        KNOWN_UPSTREAM_MISSING_SAMPLE_KEY_DICT.get(per_group_name, []))


def get_total_expected_sample_pair_count():
    """全量应该有多少行 = 920224 - 上游已知缺失总数"""
    return EXPECTED_TOTAL_SAMPLE_PAIR_COUNT - sum(
        len(x) for x in KNOWN_UPSTREAM_MISSING_SAMPLE_KEY_DICT.values())


def get_normalized_check_text(per_instruction):
    """把指令归一成用于词表匹配的文本

    全小写 + 把雨滴措辞和噪点颗粒措辞各换成一个不含"rain"子串的占位词。
    不归一的话"remove the raindrops"和"remove the colour grain"里的rain
    都会被当成雨条退化，11Raindrop和05Noise的正常指令会被大面积误杀。
    雨滴要先归一: 先换grain的话"rain drop"这类写法还留着rain。
    """
    per_check_text = f' {per_instruction.lower()} '
    for per_raindrop_word in RAINDROP_NORMALIZE_WORD_LIST:
        per_check_text = per_check_text.replace(
            per_raindrop_word, f' {RAINDROP_NORMALIZE_MARK_WORD} ')
    for per_noise_grain_word in NOISE_GRAIN_NORMALIZE_WORD_LIST:
        per_check_text = per_check_text.replace(
            per_noise_grain_word, f' {NOISE_GRAIN_NORMALIZE_MARK_WORD} ')

    return per_check_text


def clean_single_instruction(per_instruction):
    """把模型输出的一行清成一条干净指令

    剥掉行首编号/项目符号、首尾引号星号、句末句号，首字母改小写(anyedit的模板类
    任务全是小写开头)，但保留JPEG这种全大写缩写不动。
    """
    per_instruction = per_instruction.replace('\u00a0', ' ').strip()
    per_instruction = INSTRUCTION_LINE_PREFIX_PATTERN.sub('', per_instruction)
    per_instruction = per_instruction.strip(' \t"\'`*.。!！?？-')
    per_instruction = re.sub(r'\s+', ' ', per_instruction).strip()

    if not per_instruction:
        return ''

    per_first_word = per_instruction.split(' ')[0]
    if not per_first_word.isupper():
        per_instruction = per_instruction[0].lower() + per_instruction[1:]

    return per_instruction


def validate_single_instruction(per_instruction, degradation_type_list):
    """逐条硬校验一条候选指令，返回错误信息(空串表示合规)

    校验的每一项都对应anyedit实测规格或"指令必须和样本任务匹配"这条底线:
      长度/词数     -> 详细程度对齐anyedit(单退化4~10词、每多一种退化放宽一档);
      小写开头/无句号/无逗号分号 -> 文本编写风格对齐anyedit模板类任务;
      禁用词        -> 挡掉目的从句、解释、称呼这些anyedit里不会出现的成分;
      inclusion     -> 该组每种退化都必须被提到，漏一种指令就和样本任务对不上;
      exclusion     -> 不许提该组没有的退化，否则会教模型做这张图上根本不存在的操作。
    """
    if not per_instruction:
        return 'empty instruction'

    if len(degradation_type_list) < 1:
        return 'empty degradation type list'

    per_max_word_num = MAX_INSTRUCTION_WORD_NUM_PER_DEGRADATION_TYPE[
        min(len(degradation_type_list),
            len(MAX_INSTRUCTION_WORD_NUM_PER_DEGRADATION_TYPE)) - 1]
    per_max_char_num = MAX_INSTRUCTION_CHAR_NUM_PER_DEGRADATION_TYPE[
        min(len(degradation_type_list),
            len(MAX_INSTRUCTION_CHAR_NUM_PER_DEGRADATION_TYPE)) - 1]

    per_word_num = len(per_instruction.split(' '))
    if per_word_num < MIN_INSTRUCTION_WORD_NUM or per_word_num > per_max_word_num:
        return f'word num out of range {per_word_num}'

    if len(per_instruction) < MIN_INSTRUCTION_CHAR_NUM or len(
            per_instruction) > per_max_char_num:
        return f'char num out of range {len(per_instruction)}'

    if '\n' in per_instruction or '\r' in per_instruction:
        return 'multi line instruction'

    if per_instruction[0].isupper():
        return 'first char not lower'

    if per_instruction.endswith('.'):
        return 'end with period'

    for per_forbid_char in INSTRUCTION_FORBID_CHAR_LIST:
        if per_forbid_char in per_instruction:
            return f'forbid char {per_forbid_char}'

    # 非ascii(中文/emoji/全角标点)一律不要: anyedit全是纯英文祈使句
    if not all(ord(per_char) < 128 for per_char in per_instruction):
        return 'non ascii char'

    per_check_text = get_normalized_check_text(per_instruction)

    for per_forbid_word in INSTRUCTION_FORBID_WORD_LIST:
        if per_forbid_word in per_check_text:
            return f'forbid word {per_forbid_word.strip()}'

    for per_degradation_type in degradation_type_list:
        per_include_word_list = DEGRADATION_TYPE_INCLUDE_WORD_DICT[
            per_degradation_type]
        if not any(per_include_word in per_check_text
                   for per_include_word in per_include_word_list):
            return f'miss degradation type {per_degradation_type}'

    for per_degradation_type in sorted(
            DEGRADATION_TYPE_EXCLUDE_WORD_DICT.keys()):
        if per_degradation_type in degradation_type_list:
            continue

        per_exclude_word_list = DEGRADATION_TYPE_EXCLUDE_WORD_DICT[
            per_degradation_type]
        for per_exclude_word in per_exclude_word_list:
            if per_exclude_word in per_check_text:
                return f'extra degradation type {per_degradation_type} word {per_exclude_word}'

    return ''


def build_single_group_candidate_instruction_list(degradation_type_list):
    """用本地词表组合出一个退化组的全部候选指令

    返回[[指令, 动词表下标, 骨架下标], ...]，顺序是确定的(词表顺序 + 笛卡尔积顺序)，
    调用方再用固定种子shuffle，保证每次跑出来的池子完全一致。

    两套骨架:
      骨架一 动词短语 + 退化名词短语(多退化and连接) + 位置短语
             like "remove the blur from the image"
      骨架二 修复动词 + this + 退化形容词 + 图像名词
             like "fix this blurry photo"
    退化名词/形容词都是按degradation_type_list逐项取的，所以"该组每种退化都被提到、
    且绝不会提到该组没有的退化"由构造方式本身保证，不靠后验过滤。
    """
    candidate_instruction_list = []

    # ====== 骨架一: 动词 + 退化名词短语 + 位置短语 ======
    per_noun_phrase_group_list = [
        LOCAL_DEGRADATION_NOUN_PHRASE_DICT[per_degradation_type]
        for per_degradation_type in degradation_type_list
    ]
    for per_noun_phrase_tuple in itertools.product(
            *per_noun_phrase_group_list):
        per_noun_phrase_text = ' and '.join(per_noun_phrase_tuple)
        for per_verb_index, per_verb_phrase in enumerate(
                LOCAL_VERB_PHRASE_LIST):
            for per_sentence_tail in LOCAL_SENTENCE_TAIL_LIST:
                per_instruction = f'{per_verb_phrase} {per_noun_phrase_text}'
                if per_sentence_tail:
                    per_instruction = f'{per_instruction} {per_sentence_tail}'

                candidate_instruction_list.append([
                    per_instruction,
                    per_verb_index,
                    NOUN_SENTENCE_PATTERN_INDEX,
                ])

    # ====== 骨架二: 修复动词 + this + 退化形容词 + 图像名词 ======
    per_adjective_group_list = [
        LOCAL_DEGRADATION_ADJECTIVE_DICT[per_degradation_type]
        for per_degradation_type in degradation_type_list
    ]
    for per_adjective_tuple in itertools.product(*per_adjective_group_list):
        per_adjective_text = ' '.join(per_adjective_tuple)
        for per_verb_index, per_fix_verb_phrase in enumerate(
                LOCAL_FIX_VERB_PHRASE_LIST):
            for per_image_noun in LOCAL_IMAGE_NOUN_LIST:
                per_instruction = (f'{per_fix_verb_phrase} this '
                                   f'{per_adjective_text} {per_image_noun}')

                candidate_instruction_list.append([
                    per_instruction,
                    per_verb_index,
                    ADJECTIVE_SENTENCE_PATTERN_INDEX,
                ])

    return candidate_instruction_list


def build_single_group_instruction_pool_list(per_group_name):
    """为一个退化组攒出去重 + 硬校验后的指令池

    流程: 本地组合出全部候选 -> 用固定种子shuffle(打散词表顺序带来的聚集，
    让池子里的动词/句式分布均匀) -> 逐条clean + validate + 去重 ->
    攒够INSTRUCTION_POOL_TARGET_NUM条就停。

    shuffle用独立的random.Random(按组名派生种子)而不是全局random，
    这样某个组的候选数变化不会影响别的组的结果，单组可独立复现。
    """
    per_degradation_type_list = GROUP_CONFIG_DICT[per_group_name][0]

    candidate_instruction_list = build_single_group_candidate_instruction_list(
        per_degradation_type_list)

    # 种子只由组名决定: 同一个组任何时候跑都得到同一个池子
    per_random = random.Random(
        int(hashlib.md5(per_group_name.encode('UTF-8')).hexdigest(), 16) %
        (2**32))
    per_random.shuffle(candidate_instruction_list)

    instruction_pool_list, instruction_key_set = [], set()
    invalid_instruction_count = 0
    invalid_error_message_count_dict = collections.Counter()

    for (per_raw_instruction, per_verb_style_index,
         per_sentence_pattern_index) in candidate_instruction_list:
        if len(instruction_pool_list) >= INSTRUCTION_POOL_TARGET_NUM:
            break

        per_instruction = clean_single_instruction(per_raw_instruction)

        per_error_message = validate_single_instruction(
            per_instruction, per_degradation_type_list)
        if per_error_message:
            invalid_instruction_count += 1
            invalid_error_message_count_dict[per_error_message] += 1
            continue

        if per_instruction in instruction_key_set:
            continue

        instruction_key_set.add(per_instruction)
        instruction_pool_list.append([
            per_instruction,
            per_verb_style_index,
            per_sentence_pattern_index,
        ])

    return {
        'group_name': per_group_name,
        'instruction_pool_list': instruction_pool_list,
        'candidate_instruction_count': len(candidate_instruction_list),
        'invalid_instruction_count': invalid_instruction_count,
        'invalid_error_message_count_dict':
        dict(invalid_error_message_count_dict),
    }


def build_group_instruction_pool_dict(group_name_list):
    """为每个退化组建指令池(全本地，不发任何网络请求)

    本地组合是纯CPU计算、17个组一共只有几十万个候选，串行跑完也是秒级，
    所以这里不用线程池: 少一层并发就少一处不确定性，池子结果完全可复现。
    """
    group_instruction_pool_dict = {}
    group_candidate_count_dict, group_invalid_count_dict = {}, {}
    invalid_error_message_count_dict = collections.Counter()

    for per_group_name in tqdm(group_name_list, desc='Building pool'):
        per_build_result = build_single_group_instruction_pool_list(
            per_group_name)

        group_instruction_pool_dict[per_group_name] = per_build_result[
            'instruction_pool_list']
        group_candidate_count_dict[per_group_name] = per_build_result[
            'candidate_instruction_count']
        group_invalid_count_dict[per_group_name] = per_build_result[
            'invalid_instruction_count']
        for per_error_message, per_count in per_build_result[
                'invalid_error_message_count_dict'].items():
            invalid_error_message_count_dict[per_error_message] += per_count

        print('2222', per_group_name, 'degradation type',
              GROUP_CONFIG_DICT[per_group_name][0], 'candidate',
              per_build_result['candidate_instruction_count'], 'invalid',
              per_build_result['invalid_instruction_count'], 'pool num',
              len(per_build_result['instruction_pool_list']))

    build_pool_stat_dict = {
        'timestamp': TIMESTAMP,
        'instruction_source': INSTRUCTION_SOURCE_NAME,
        'reuse_instruction_pool_flag': False,
        'instruction_pool_target_num': INSTRUCTION_POOL_TARGET_NUM,
        'local_verb_phrase_num': len(LOCAL_VERB_PHRASE_LIST),
        'local_sentence_tail_num': len(LOCAL_SENTENCE_TAIL_LIST),
        'local_fix_verb_phrase_num': len(LOCAL_FIX_VERB_PHRASE_LIST),
        'local_image_noun_num': len(LOCAL_IMAGE_NOUN_LIST),
        'group_candidate_instruction_count_dict': group_candidate_count_dict,
        'group_invalid_instruction_count_dict': group_invalid_count_dict,
        'group_instruction_pool_num_dict': {
            per_group_name: len(group_instruction_pool_dict[per_group_name])
            for per_group_name in group_name_list
        },
        'invalid_error_message_count_dict':
        dict(invalid_error_message_count_dict),
    }

    print('3333', 'total candidate:', sum(group_candidate_count_dict.values()),
          'total invalid:', sum(group_invalid_count_dict.values()),
          'total pool:',
          sum(len(x) for x in group_instruction_pool_dict.values()))

    return group_instruction_pool_dict, build_pool_stat_dict


def load_group_instruction_pool_dict(save_instruction_pool_dir_path,
                                     group_name_list):
    """复用已落盘的指令池

    只有"文件在 + 条数达标 + 逐条重新过一遍校验都合规"才认，
    否则当成没有并重新本地组合一份，避免历史版本(例如早期带场景描写的
    176~246字符长指令)被静默复用到训练目录里。
    """
    group_instruction_pool_dict = {}
    for per_group_name in group_name_list:
        per_instruction_pool_path = os.path.join(
            save_instruction_pool_dir_path, f'{per_group_name}.json')
        if not os.path.exists(per_instruction_pool_path):
            return {}

        try:
            with open(per_instruction_pool_path, 'r',
                      encoding='UTF-8') as load_json_file:
                per_instruction_pool_dict = json.load(load_json_file)
        except Exception:
            return {}

        per_instruction_pool_list = per_instruction_pool_dict.get(
            'instruction_pool_list', [])
        if len(per_instruction_pool_list) < MIN_INSTRUCTION_POOL_NUM:
            return {}

        per_degradation_type_list = GROUP_CONFIG_DICT[per_group_name][0]
        for per_instruction_item in per_instruction_pool_list:
            if validate_single_instruction(per_instruction_item[0],
                                           per_degradation_type_list):
                return {}

        group_instruction_pool_dict[per_group_name] = [
            list(x) for x in per_instruction_pool_list
        ]

    return group_instruction_pool_dict


def save_group_instruction_pool_dict(save_instruction_pool_dir_path,
                                     group_instruction_pool_dict):
    """指令池落盘: 一个组一个json，带长度统计方便人工审查风格是否对齐anyedit"""
    for per_group_name in sorted(group_instruction_pool_dict.keys()):
        per_instruction_pool_list = group_instruction_pool_dict[per_group_name]
        per_instruction_length_list = sorted(
            [len(x[0]) for x in per_instruction_pool_list])
        per_instruction_word_num_list = sorted(
            [len(x[0].split(' ')) for x in per_instruction_pool_list])

        per_save_instruction_pool_dict = {
            'group_name':
            per_group_name,
            'degradation_type_list':
            GROUP_CONFIG_DICT[per_group_name][0],
            'task_type':
            TASK_TYPE_NAME,
            'instruction_source':
            INSTRUCTION_SOURCE_NAME,
            'instruction_style_reference':
            INSTRUCTION_STYLE_REFERENCE_NAME,
            'instruction_pool_num':
            len(per_instruction_pool_list),
            'instruction_char_num_min':
            per_instruction_length_list[0]
            if per_instruction_length_list else 0,
            'instruction_char_num_median':
            per_instruction_length_list[len(per_instruction_length_list) // 2]
            if per_instruction_length_list else 0,
            'instruction_char_num_max':
            per_instruction_length_list[-1]
            if per_instruction_length_list else 0,
            'instruction_word_num_median':
            per_instruction_word_num_list[len(per_instruction_word_num_list) //
                                          2]
            if per_instruction_word_num_list else 0,
            'instruction_pool_list':
            per_instruction_pool_list,
        }

        per_instruction_pool_path = os.path.join(
            save_instruction_pool_dir_path, f'{per_group_name}.json')
        per_temp_instruction_pool_path = f'{per_instruction_pool_path}.tmp'
        with open(per_temp_instruction_pool_path, 'w',
                  encoding='UTF-8') as save_json_file:
            json.dump(per_save_instruction_pool_dict,
                      save_json_file,
                      ensure_ascii=False)
        os.replace(per_temp_instruction_pool_path, per_instruction_pool_path)

    return


def get_instruction_pool_index(per_group_name, per_sample_key,
                               per_instruction_pool_num):
    """按md5(group_name + sample_key)取模选指令池下标

    用哈希而不是随机/轮转的原因: 同一个sample_key每次跑都必须拿到同一条指令，
    这样断点重跑、增量补数、和unzip_annotations对账时结果完全可复现，
    而且md5的均匀性能保证池子里每条指令被分到的样本数基本相同。
    """
    per_hash_value = hashlib.md5(
        f'{per_group_name}_{per_sample_key}'.encode('UTF-8')).hexdigest()

    return int(per_hash_value, 16) % per_instruction_pool_num


def process_single_annotation_file(annotation_file_task):
    """给单个标注文件的每一行分配指令并落盘(线程池worker)

    输入行原封不动保留009写出的全部客观字段，只追加指令相关字段;
    临时文件写完再os.replace，中途挂掉不会留下半个文件被下游读到。
    """
    (per_group_name, per_annotation_file_name, per_load_annotation_file_path,
     per_save_annotation_file_path,
     per_instruction_pool_list) = annotation_file_task

    error_message_list = []
    input_line_count, write_line_count = 0, 0
    invalid_caption_count = 0
    caption_length_sum = 0
    caption_length_min, caption_length_max = 0, 0
    used_instruction_index_dict = collections.Counter()

    per_degradation_type_list = GROUP_CONFIG_DICT[per_group_name][0]
    per_instruction_pool_num = len(per_instruction_pool_list)

    per_temp_save_annotation_file_path = f'{per_save_annotation_file_path}.tmp'

    try:
        os.makedirs(os.path.dirname(per_save_annotation_file_path),
                    exist_ok=True)

        with open(per_load_annotation_file_path, 'r',
                  encoding='UTF-8') as load_json_file, open(
                      per_temp_save_annotation_file_path,
                      'w',
                      encoding='UTF-8') as save_json_file:
            for per_line in load_json_file:
                if not per_line.strip():
                    continue

                input_line_count += 1

                try:
                    per_annotation_dict = json.loads(per_line)
                except Exception:
                    error_message_list.append(
                        f'{per_annotation_file_name} line {input_line_count} broken json'
                    )
                    continue

                per_sample_key = per_annotation_dict.get('sample_key', '')
                if not per_sample_key:
                    error_message_list.append(
                        f'{per_annotation_file_name} line {input_line_count} empty sample key'
                    )
                    continue

                if per_annotation_dict.get('group_name', '') != per_group_name:
                    error_message_list.append(
                        f'{per_annotation_file_name} line {input_line_count} group name not match'
                    )
                    continue

                if per_annotation_dict.get('degradation_type_list',
                                           []) != per_degradation_type_list:
                    error_message_list.append(
                        f'{per_annotation_file_name} line {input_line_count} degradation type not match'
                    )
                    continue

                per_instruction_pool_index = get_instruction_pool_index(
                    per_group_name, per_sample_key, per_instruction_pool_num)
                (per_instruction, per_verb_style_index,
                 per_sentence_pattern_index
                 ) = per_instruction_pool_list[per_instruction_pool_index]

                per_caption_error_message = validate_single_instruction(
                    per_instruction, per_degradation_type_list)
                if per_caption_error_message:
                    invalid_caption_count += 1
                    error_message_list.append(
                        f'{per_annotation_file_name} {per_sample_key} invalid caption '
                        f'{per_caption_error_message}')
                    continue

                per_annotation_dict['ti2i_caption'] = per_instruction
                per_annotation_dict['ti2i_caption_length'] = len(
                    per_instruction)
                per_annotation_dict[
                    'ti2i_caption_source'] = INSTRUCTION_SOURCE_NAME
                per_annotation_dict[
                    'instruction_style_reference'] = INSTRUCTION_STYLE_REFERENCE_NAME
                per_annotation_dict[
                    'instruction_pool_index'] = per_instruction_pool_index
                per_annotation_dict[
                    'instruction_pool_num'] = per_instruction_pool_num
                per_annotation_dict[
                    'instruction_verb_style_index'] = per_verb_style_index
                per_annotation_dict[
                    'instruction_sentence_pattern_index'] = per_sentence_pattern_index

                save_json_file.write(
                    f'{json.dumps(per_annotation_dict, ensure_ascii=False)}\n')

                write_line_count += 1
                used_instruction_index_dict[per_instruction_pool_index] += 1
                caption_length_sum += len(per_instruction)
                if caption_length_min == 0 or len(
                        per_instruction) < caption_length_min:
                    caption_length_min = len(per_instruction)
                if len(per_instruction) > caption_length_max:
                    caption_length_max = len(per_instruction)

        if write_line_count != input_line_count:
            os.remove(per_temp_save_annotation_file_path)
            error_message_list.append(
                f'{per_annotation_file_name} write line count not match '
                f'{write_line_count} != {input_line_count}')

            return {
                'group_name': per_group_name,
                'annotation_file_name': per_annotation_file_name,
                'input_line_count': input_line_count,
                'write_line_count': 0,
                'invalid_caption_count': invalid_caption_count,
                'caption_length_sum': 0,
                'caption_length_min': 0,
                'caption_length_max': 0,
                'used_instruction_index_dict': {},
                'error_message_list': error_message_list,
            }

        os.replace(per_temp_save_annotation_file_path,
                   per_save_annotation_file_path)
    except Exception as e:
        if os.path.exists(per_temp_save_annotation_file_path):
            os.remove(per_temp_save_annotation_file_path)
        error_message_list.append(
            f'{per_annotation_file_name} write annotation failed {e}')

        return {
            'group_name': per_group_name,
            'annotation_file_name': per_annotation_file_name,
            'input_line_count': input_line_count,
            'write_line_count': 0,
            'invalid_caption_count': invalid_caption_count,
            'caption_length_sum': 0,
            'caption_length_min': 0,
            'caption_length_max': 0,
            'used_instruction_index_dict': {},
            'error_message_list': error_message_list,
        }

    return {
        'group_name': per_group_name,
        'annotation_file_name': per_annotation_file_name,
        'input_line_count': input_line_count,
        'write_line_count': write_line_count,
        'invalid_caption_count': invalid_caption_count,
        'caption_length_sum': caption_length_sum,
        'caption_length_min': caption_length_min,
        'caption_length_max': caption_length_max,
        'used_instruction_index_dict': dict(used_instruction_index_dict),
        'error_message_list': error_message_list,
    }


def get_all_annotation_file_task(load_annotation_dir_path,
                                 save_annotation_dir_path,
                                 group_instruction_pool_dict):
    """收集所有要处理的标注文件任务，并清掉输出目录里的陈旧文件

    输入以unzip_annotations为唯一真值: 输出文件名与输入一一对应，
    输入没有的输出文件就是上一版跑一半留下的垃圾，必须删掉，
    否则训练目录里会混进风格不一致的旧标注(实测上一版留了81行200+字符的长指令)。
    """
    annotation_file_task_list = []
    error_message_list = []

    for per_group_name in sorted(group_instruction_pool_dict.keys()):
        per_load_group_dir_path = os.path.join(load_annotation_dir_path,
                                               per_group_name)
        per_save_group_dir_path = os.path.join(save_annotation_dir_path,
                                               per_group_name)

        if not os.path.isdir(per_load_group_dir_path):
            error_message_list.append(
                f'{per_group_name} load annotation dir not exist')
            continue

        per_load_annotation_file_name_list = sorted([
            per_file_name
            for per_file_name in os.listdir(per_load_group_dir_path)
            if per_file_name.endswith('.jsonl')
        ])
        if len(per_load_annotation_file_name_list) < 1:
            error_message_list.append(
                f'{per_group_name} load annotation file not exist')
            continue

        if DEBUG_ANNOTATION_FILE_NUM_PER_GROUP > 0:
            per_load_annotation_file_name_list = per_load_annotation_file_name_list[:
                                                                                    DEBUG_ANNOTATION_FILE_NUM_PER_GROUP]

        os.makedirs(per_save_group_dir_path, exist_ok=True)

        if REMOVE_STALE_ANNOTATION_FILE_FLAG and DEBUG_ANNOTATION_FILE_NUM_PER_GROUP <= 0:
            per_load_annotation_file_name_set = set(
                per_load_annotation_file_name_list)
            for per_file_name in sorted(os.listdir(per_save_group_dir_path)):
                if per_file_name in per_load_annotation_file_name_set:
                    continue

                try:
                    os.remove(
                        os.path.join(per_save_group_dir_path, per_file_name))
                    print('2222', 'remove stale annotation file',
                          per_group_name, per_file_name)
                except Exception as e:
                    error_message_list.append(
                        f'{per_group_name} remove stale file {per_file_name} failed {e}'
                    )

        for per_annotation_file_name in per_load_annotation_file_name_list:
            annotation_file_task_list.append([
                per_group_name,
                per_annotation_file_name,
                os.path.join(per_load_group_dir_path,
                             per_annotation_file_name),
                os.path.join(per_save_group_dir_path,
                             per_annotation_file_name),
                group_instruction_pool_dict[per_group_name],
            ])

    return annotation_file_task_list, error_message_list


def save_check_result(save_dataset_path, group_instruction_pool_dict,
                      build_pool_stat_dict, annotation_file_result_list):
    """全量对账并落盘校验报告，返回错误信息列表

    对账口径: unzip_annotations是ground truth。
      每组输出行数 == 输入行数 == 009实测样本对数 - 该组上游已知缺失数;
      全量输出行数 == 920224 - 上游已知缺失总数;
      invalid_caption / broken line 必须为0;
      每组指令池条数 >= MIN_INSTRUCTION_POOL_NUM;
      每组实际用到的指令条数 >= min(池子条数, 该组样本对数) * MIN_USED_INSTRUCTION_RATIO
        (哈希分配退化成常数或池子读错时这一项会掉到很低);
      指令长度必须全部落在该组的长度门槛内。
    任一不满足都汇总后抛异常，不静默跑过。
    """
    total_input_line_count, total_write_line_count = 0, 0
    total_invalid_caption_count = 0
    group_input_line_count_dict = collections.Counter()
    group_write_line_count_dict = collections.Counter()
    group_caption_length_sum_dict = collections.Counter()
    group_caption_length_min_dict, group_caption_length_max_dict = {}, {}
    group_used_instruction_index_dict = {}
    degradation_type_count_dict = collections.Counter()
    error_message_list, warning_message_list = [], []

    for per_annotation_file_result in annotation_file_result_list:
        per_group_name = per_annotation_file_result['group_name']

        total_input_line_count += per_annotation_file_result[
            'input_line_count']
        total_write_line_count += per_annotation_file_result[
            'write_line_count']
        total_invalid_caption_count += per_annotation_file_result[
            'invalid_caption_count']

        group_input_line_count_dict[
            per_group_name] += per_annotation_file_result['input_line_count']
        group_write_line_count_dict[
            per_group_name] += per_annotation_file_result['write_line_count']
        group_caption_length_sum_dict[
            per_group_name] += per_annotation_file_result['caption_length_sum']

        per_caption_length_min = per_annotation_file_result[
            'caption_length_min']
        if per_caption_length_min > 0:
            if per_group_name not in group_caption_length_min_dict or per_caption_length_min < group_caption_length_min_dict[
                    per_group_name]:
                group_caption_length_min_dict[
                    per_group_name] = per_caption_length_min
        per_caption_length_max = per_annotation_file_result[
            'caption_length_max']
        if per_caption_length_max > group_caption_length_max_dict.get(
                per_group_name, 0):
            group_caption_length_max_dict[
                per_group_name] = per_caption_length_max

        if per_group_name not in group_used_instruction_index_dict:
            group_used_instruction_index_dict[
                per_group_name] = collections.Counter()
        for per_instruction_pool_index, per_use_count in per_annotation_file_result[
                'used_instruction_index_dict'].items():
            group_used_instruction_index_dict[per_group_name][int(
                per_instruction_pool_index)] += per_use_count

        if len(per_annotation_file_result['error_message_list']) > 0:
            print('7777', per_group_name,
                  per_annotation_file_result['annotation_file_name'],
                  per_annotation_file_result['error_message_list'][:5])
            error_message_list.extend(
                per_annotation_file_result['error_message_list'][:5])

    group_check_result_dict = {}
    for per_group_name in sorted(group_instruction_pool_dict.keys()):
        per_degradation_type_list = GROUP_CONFIG_DICT[per_group_name][0]
        per_expected_sample_pair_count = get_group_expected_sample_pair_count(
            per_group_name)
        per_instruction_pool_num = len(
            group_instruction_pool_dict[per_group_name])
        per_write_line_count = group_write_line_count_dict.get(
            per_group_name, 0)
        per_input_line_count = group_input_line_count_dict.get(
            per_group_name, 0)
        per_used_instruction_num = len(
            group_used_instruction_index_dict.get(per_group_name, {}))
        per_max_char_num = MAX_INSTRUCTION_CHAR_NUM_PER_DEGRADATION_TYPE[
            min(len(per_degradation_type_list),
                len(MAX_INSTRUCTION_CHAR_NUM_PER_DEGRADATION_TYPE)) - 1]

        group_check_result_dict[per_group_name] = {
            'degradation_type_list':
            per_degradation_type_list,
            'expected_sample_pair_count':
            per_expected_sample_pair_count,
            'upstream_missing_sample_key_list':
            KNOWN_UPSTREAM_MISSING_SAMPLE_KEY_DICT.get(per_group_name, []),
            'input_line_count':
            per_input_line_count,
            'write_line_count':
            per_write_line_count,
            'instruction_pool_num':
            per_instruction_pool_num,
            'used_instruction_num':
            per_used_instruction_num,
            'unique_caption_ratio':
            round(per_used_instruction_num /
                  per_write_line_count, 6) if per_write_line_count > 0 else 0,
            'caption_length_min':
            group_caption_length_min_dict.get(per_group_name, 0),
            'caption_length_max':
            group_caption_length_max_dict.get(per_group_name, 0),
            'caption_length_mean':
            round(
                group_caption_length_sum_dict.get(per_group_name, 0) /
                per_write_line_count, 2) if per_write_line_count > 0 else 0,
        }

        for per_degradation_type in per_degradation_type_list:
            degradation_type_count_dict[
                per_degradation_type] += per_write_line_count

        if per_instruction_pool_num < MIN_INSTRUCTION_POOL_NUM:
            error_message_list.append(
                f'{per_group_name} instruction pool num too small '
                f'{per_instruction_pool_num} < {MIN_INSTRUCTION_POOL_NUM}')

        if per_write_line_count != per_input_line_count:
            error_message_list.append(
                f'{per_group_name} write line count not match input line count '
                f'{per_write_line_count} != {per_input_line_count}')

        if DEBUG_GROUP_NUM <= 0 and DEBUG_ANNOTATION_FILE_NUM_PER_GROUP <= 0:
            if per_input_line_count != per_expected_sample_pair_count:
                error_message_list.append(
                    f'{per_group_name} input line count not match expected sample pair count '
                    f'{per_input_line_count} != {per_expected_sample_pair_count}'
                )

            # 池子里绝大部分指令都必须真的被分配出去，
            # 掉到MIN_USED_INSTRUCTION_RATIO以下说明哈希分配或池子读取写错了
            per_expected_used_instruction_num = min(per_instruction_pool_num,
                                                    per_write_line_count)
            per_min_used_instruction_num = int(
                per_expected_used_instruction_num * MIN_USED_INSTRUCTION_RATIO)
            if per_used_instruction_num < per_min_used_instruction_num:
                error_message_list.append(
                    f'{per_group_name} used instruction num too small '
                    f'{per_used_instruction_num} < {per_min_used_instruction_num}'
                )
            elif per_used_instruction_num != per_expected_used_instruction_num:
                # 差几条是md5撞桶的正常统计波动，不是错误，只打印告警
                warning_message_list.append(
                    f'{per_group_name} used instruction num not reach pool num '
                    f'{per_used_instruction_num} != {per_expected_used_instruction_num}'
                )

        per_caption_length_max = group_caption_length_max_dict.get(
            per_group_name, 0)
        if per_caption_length_max > per_max_char_num:
            error_message_list.append(
                f'{per_group_name} caption length out of range '
                f'{per_caption_length_max} > {per_max_char_num}')
        per_caption_length_min = group_caption_length_min_dict.get(
            per_group_name, 0)
        if per_write_line_count > 0 and per_caption_length_min < MIN_INSTRUCTION_CHAR_NUM:
            error_message_list.append(
                f'{per_group_name} caption length out of range '
                f'{per_caption_length_min} < {MIN_INSTRUCTION_CHAR_NUM}')

        if per_instruction_pool_num < INSTRUCTION_POOL_TARGET_NUM:
            # 池子没攒满不影响指令正确性(每条都过了校验)，只打印告警不判失败
            warning_message_list.append(
                f'{per_group_name} instruction pool num not reach target '
                f'{per_instruction_pool_num} < {INSTRUCTION_POOL_TARGET_NUM}')

    print('3333', 'total input line:', total_input_line_count,
          'total write line:', total_write_line_count, 'invalid caption:',
          total_invalid_caption_count)
    for per_group_name in sorted(group_check_result_dict.keys()):
        per_group_check_result = group_check_result_dict[per_group_name]
        print('3333', per_group_name, 'write line',
              per_group_check_result['write_line_count'], 'pool',
              per_group_check_result['instruction_pool_num'], 'used',
              per_group_check_result['used_instruction_num'], 'unique ratio',
              per_group_check_result['unique_caption_ratio'], 'caption len',
              per_group_check_result['caption_length_min'], '~',
              per_group_check_result['caption_length_max'], 'mean',
              per_group_check_result['caption_length_mean'])
    for per_warning_message in warning_message_list:
        print('2222', per_warning_message)

    save_check_result_path = os.path.join(save_dataset_path,
                                          SAVE_CHECK_RESULT_FILE_NAME)
    save_check_result_dict = {
        'dataset_task_type': 'image_edit',
        'dataset_sub_task_type': TASK_TYPE_NAME,
        'has_original_text_prompt': False,
        'ti2i_caption_source': INSTRUCTION_SOURCE_NAME,
        'instruction_style_reference': INSTRUCTION_STYLE_REFERENCE_NAME,
        'instruction_assign_method':
        'md5(group_name + sample_key) % instruction_pool_num',
        'total_expected_sample_pair_count':
        get_total_expected_sample_pair_count(),
        'known_upstream_missing_sample_key_dict':
        KNOWN_UPSTREAM_MISSING_SAMPLE_KEY_DICT,
        'total_group_count': len(group_instruction_pool_dict),
        'total_annotation_file_count': len(annotation_file_result_list),
        'total_input_line_count': total_input_line_count,
        'total_write_line_count': total_write_line_count,
        'total_invalid_caption_count': total_invalid_caption_count,
        'degradation_type_count_dict': dict(degradation_type_count_dict),
        'group_check_result_dict': group_check_result_dict,
        'build_pool_stat_dict': build_pool_stat_dict,
        'warning_message_list': warning_message_list,
        'check_error_message_list':
        error_message_list[:MAX_SAVE_PROBLEM_ITEM_NUM],
    }
    try:
        with open(save_check_result_path, 'w',
                  encoding='UTF-8') as save_json_file:
            json.dump(save_check_result_dict,
                      save_json_file,
                      ensure_ascii=False)
    except Exception as e:
        error_message_list.append(f'save check result failed {e}')

    if total_write_line_count == 0:
        error_message_list.append('no annotation line written')
    if total_invalid_caption_count > 0:
        error_message_list.append(
            f'invalid caption count {total_invalid_caption_count}')
    if total_input_line_count != total_write_line_count:
        error_message_list.append(
            f'total input line count not match total write line count '
            f'{total_input_line_count} != {total_write_line_count}')
    if DEBUG_GROUP_NUM <= 0 and DEBUG_ANNOTATION_FILE_NUM_PER_GROUP <= 0:
        if total_write_line_count != get_total_expected_sample_pair_count():
            error_message_list.append(
                f'total write line count not match {total_write_line_count} != '
                f'{get_total_expected_sample_pair_count()}')

    return error_message_list


def generate_dataset_instruction(save_dataset_path):
    if not os.path.exists(save_dataset_path):
        raise Exception(f'save dataset path not exist {save_dataset_path}')

    load_annotation_dir_path = os.path.join(save_dataset_path,
                                            LOAD_ANNOTATION_DIR_NAME)
    if not os.path.exists(load_annotation_dir_path):
        raise Exception(
            f'load annotation dir path not exist {load_annotation_dir_path}')

    save_instruction_pool_dir_path = os.path.join(
        save_dataset_path, SAVE_INSTRUCTION_POOL_DIR_NAME)
    save_annotation_dir_path = os.path.join(save_dataset_path,
                                            SAVE_ANNOTATION_DIR_NAME)
    os.makedirs(save_instruction_pool_dir_path, exist_ok=True)
    os.makedirs(save_annotation_dir_path, exist_ok=True)

    group_name_list = sorted(GROUP_CONFIG_DICT.keys())
    if DEBUG_GROUP_NUM > 0:
        group_name_list = group_name_list[:DEBUG_GROUP_NUM]

    print('1111', 'group', len(group_name_list), 'instruction pool target',
          INSTRUCTION_POOL_TARGET_NUM)

    group_instruction_pool_dict, build_pool_stat_dict = {}, {}
    if REUSE_INSTRUCTION_POOL_FLAG:
        group_instruction_pool_dict = load_group_instruction_pool_dict(
            save_instruction_pool_dir_path, group_name_list)
        if group_instruction_pool_dict:
            build_pool_stat_dict = {
                'instruction_source': INSTRUCTION_SOURCE_NAME,
                'reuse_instruction_pool_flag': True,
                'group_instruction_pool_num_dict': {
                    per_group_name:
                    len(group_instruction_pool_dict[per_group_name])
                    for per_group_name in group_name_list
                },
            }
            print('1111', 'reuse instruction pool',
                  build_pool_stat_dict['group_instruction_pool_num_dict'])

    if not group_instruction_pool_dict:
        group_instruction_pool_dict, build_pool_stat_dict = build_group_instruction_pool_dict(
            group_name_list)

        for per_group_name in group_name_list:
            if len(group_instruction_pool_dict[per_group_name]
                   ) < MIN_INSTRUCTION_POOL_NUM:
                # 池子太小就别往下写标注了: 几万个样本共用几十条指令等于没有多样性
                raise Exception(
                    f'{per_group_name} instruction pool num too small '
                    f'{len(group_instruction_pool_dict[per_group_name])} < {MIN_INSTRUCTION_POOL_NUM}'
                )

        save_group_instruction_pool_dict(save_instruction_pool_dir_path,
                                         group_instruction_pool_dict)

    for per_group_name in group_name_list:
        per_instruction_pool_list = group_instruction_pool_dict[per_group_name]
        print('1111', per_group_name, 'degradation type',
              GROUP_CONFIG_DICT[per_group_name][0], 'pool num',
              len(per_instruction_pool_list), 'example',
              [x[0] for x in per_instruction_pool_list[:3]])

    annotation_file_task_list, task_error_message_list = get_all_annotation_file_task(
        load_annotation_dir_path, save_annotation_dir_path,
        group_instruction_pool_dict)

    print('1111', 'annotation file task',
          len(annotation_file_task_list), 'task error',
          len(task_error_message_list), task_error_message_list[:20])
    if len(task_error_message_list) > 0:
        raise Exception(
            f'get annotation file task failed {task_error_message_list[:20]}')

    annotation_file_result_list = []
    pool = ThreadPool(WRITE_ANNOTATION_THREAD_NUM)
    for per_annotation_file_result in tqdm(
            pool.imap_unordered(process_single_annotation_file,
                                annotation_file_task_list),
            total=len(annotation_file_task_list),
            desc='Writing annotation'):
        annotation_file_result_list.append(per_annotation_file_result)
    pool.close()
    pool.join()

    check_error_message_list = save_check_result(save_dataset_path,
                                                 group_instruction_pool_dict,
                                                 build_pool_stat_dict,
                                                 annotation_file_result_list)

    print('3333', 'total error', len(check_error_message_list),
          check_error_message_list[:20])

    if len(check_error_message_list) > 0:
        # 指令生成/分配/对账任一环出错都必须让上层感知，不能静默给样本配错指令
        raise Exception(
            f'generate dataset instruction error num {len(check_error_message_list)} '
            f'{check_error_message_list[:20]}')

    return


if __name__ == '__main__':
    random.seed(0)

    save_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/FoundIR'
    generate_dataset_instruction(save_dataset_path)
