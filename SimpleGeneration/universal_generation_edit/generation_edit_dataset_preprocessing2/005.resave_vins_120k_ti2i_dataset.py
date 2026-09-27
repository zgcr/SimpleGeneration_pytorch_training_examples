import os
import re
import json
import shutil
import numpy as np
import cv2

from fractions import Fraction
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'vins120k'

SAVE_DATASET_DIR_NAME = 'VINS120K'

# 上游010.unzip_vins_120k_dataset.py把46个tar组解成
#   images/<子集名>/<标注相对路径>   212184张4K~8K的png
# 并把3个训练标注json各解成一个annotations/<子集名>.jsonl(每行1个完整样本对，
# 保留了原json的全部属性并补上落盘后的规范图像路径)。
# 上游自校验(unzip_check_result.json)实测: 标注总行数131035、隔离的不完整样本对158、
# 写出的有效样本对130877、缺图0张、引用唯一图212107张、每对固定1张参考图。
# 本脚本只读这3个jsonl，不读上游的image_index/(落盘图像索引，上游自校验用的中间产物)、
# 也不读unzip_check_result.json
LOAD_ANNOTATION_DIR_NAME = 'annotations'

# 只读这3个训练标注。benchmark.jsonl(509行/100张图)按方案不处理:
# 它是VINS-4KEval评测集，一张图配多条不同编辑类型的指令，**只有参考图没有编辑后图**
# (编辑后图要靠模型生成)，根本构不成"参考图 + 编辑后图 + 指令"的图像编辑对，
# 上游解压时也是单独解到benchmark/目录、不参与训练样本对统计的。
# 所以benchmark.jsonl和benchmark/目录本脚本都不读不搬
LOAD_ANNOTATION_FILE_NAME_LIST = [
    'nano-consistent.jsonl',
    'ultravideo.jsonl',
    'x2edit.jsonl',
]

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 上游标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如images/x2edit/3/00026/output/000000001.png)，不需要再拼images子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 上游标注里图像路径的公共第一段。
# 拼保存图像名时要先把这一段剥掉、再把剩下的路径拍平成名字: 实测130877行的路径
# 第一段100%都是images，留着它只会让每个保存名都多出无意义的7个字符
LOAD_IMAGE_ROOT_DIR_NAME = 'images'

# 上游只有3个标注文件(最大的x2edit.jsonl有83004行)，不拆分就只能开3个进程解析，
# 所以这里照007的写法先把3个jsonl按固定行数拆成分片，再按分片粒度开多进程，
# 分片文件落在输出目录下、解析完成后整个目录删除。
# 实测按2000行一片拆出67片(nano-consistent 12 + ultravideo 13 + x2edit 42)，
# 正好能把32个进程铺满两轮
SAVE_ANNOTATION_SHARD_DIR_NAME = 'annotation_shard'

PER_ANNOTATION_SHARD_LINE_NUM = 2000

# 分片文件名用双下划线分隔标注名和分片编号: 标注名本身带中划线(nano-consistent)，
# 用双下划线才能保证文件名可逆
ANNOTATION_SHARD_FILE_NAME_SEPARATOR = '__'

# 上游标注名(即上游的subset_name，两者实测恒等)，只用于分片对账，不写进新标注。
# 它描述的是"这条数据出自上游哪个标注json"而不是"这是什么编辑任务"
# (同一个标注里混着3~10种edit_type)，所以不能用它分子集
ANNOTATION_NAME_KEY_NAME = 'annotation_name'

# 图像编辑任务类型，实测130877行全非空、无一条缺失，共18个原始取值。
# 这就是本数据集能拿到的图像编辑任务类型，子集按它划分，所以不需要mix兜底子集。
# 注意不能用subset_name/annotation_name/archive_group_name分子集，
# 那三个描述的都是"数据打在哪个包里"而不是"这是什么编辑任务"
ANNOTATION_EDIT_TYPE_KEY_NAME = 'edit_type'

# 编辑指令，也是本数据集唯一一个真正的图像编辑指令字段，实测130877行全非空。
# 上游README官方定义就是instruction(英文编辑指令)，实测只有6条是中英混写
# (如"Pan the frame to the right, increase全体 color saturation")，
# 语言不做任何过滤、原样保存
ANNOTATION_CAPTION_KEY_NAME = 'instruction'

# 编辑前的原图(唯一的参考图)，上游是长度恒为1的list、实测130877行全非空
ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'reference_image_path_list'

# 编辑后的图像，实测130877行全非空、全是png
ANNOTATION_EDITED_IMAGE_KEY_NAME = 'target_image_path'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 新标注固定只存这七个key，多一个少一个都在收尾自校验里报错。
# 上游jsonl里剩下的属性按方案全部丢弃、不另存索引:
#   sample_id           : <标注名>_<8位行号>，只是上游的行级溯源id
#   subset_name         : 与annotation_name实测恒等(nano-consistent/ultravideo/
#                         x2edit)，是"上游哪个标注json"而不是编辑任务类型
#   annotation_name     : 同上，本脚本只用它做分片与解析对账，不写进新标注
#   archive_group_name  : 形如ultravideo/clips_short_12、x2edit/3，上游tar组名，
#                         只描述"这条数据打在哪个压缩包里"，与编辑任务无关
#   task_type           : 实测130877行恒为image_edit，整个数据集就一个值，无信息量
#   row_index           : 上游json里的行号，只用于溯源
#   reference_image_num : 上游恒为1; 新标注里的这个值一律由reference_image这个list
#                         的长度现算，不采信上游数值
#   edit_type           : 只用于推导子集名(子集目录名已完整体现)，不另存为字段
#   reference_image_path_list / target_image_path:
#                         只用于定位源图和拼保存图像名
# 另外要说明的是: 上游原始数据集(VINS-120K)每条标注**总共只有4个字段**
# (edit_type/input/output/instruction)，不像008那个数据集还带二十多个质量分和
# 多语言caption，所以本数据集没有"用美学分/质量分做阈值过滤"这条路可走，
# 只能靠下面的文本层规则 + 图像层解码与分辨率过滤
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# 本数据集每个编辑对固定只有"编辑前原图"这一张参考图(上游reference_image_num
# 实测130877行恒为1)，所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 归一化后的edit_type -> 子集名的归并表。
# 上游18个原始edit_type按strip().lower().replace(' ','_')归一化后剩16个
# (上游同一语义存在同义异写: "action change"与"action_change"、
#  "background change"与"background_change"并存)，再按下表归并成13个子集。
#
# 【为什么这3条要合并】实测(全量非抽样)动词类覆盖率与实际指令逐条比对确认
# object_*(全部来自ultravideo)与subject_*(全部来自x2edit)是**同一个编辑任务**，
# 只是两个上游子集叫法不同:
#   object_addition(1336)    add 96% | subject_addition(18787)    add 97%
#     "Add a clownfish peeking out from the sea anemone."
#     "Add a sailboat to the distant sea."
#   object_deletion(786)     remove 99% | subject_deletion(15026)  remove 95%
#     "Remove the clouds from the sky." / "Remove the fountain from the square."
#   object_replacement(621)  replace 88% | subject_replacement(10638) replace 80%
#     "Replace the clouds in the sky." / "Replace the waves with calm lakes."
# 【为什么归到subject_而不是object_】1) 已产出的X2Edit-Dataset用的就是
# subject_addition/subject_deletion/subject_replacement，跨数据集口径必须统一，
# 否则同一语义会变成两个任务名、后续按子集做任务采样时还要再加一层归并表;
# 2) subject_*侧是18787/15026/10638条、object_*侧只有1336/786/621条，
# 少数派向多数派归(与007里ultraedit/add归进omniedit/addition的add同一做法);
# 3) 本数据集这三类里被增删替换的既有物体也有人
# ("Add a person wearing a red jacket riding a bicycle")，subject覆盖面更准。
#
# 【为什么object_movement不合并】它没有同义的对手方(x2edit侧根本不存在"移动"任务)，
# 而与它唯一相邻的camera_movement是**不同任务**: object_movement是"移动画面里的
# 物体"(move类动词86%、镜头类词只有20%，如"Move the clownfish slightly down and
# to the left")，camera_movement是"移动镜头本身"(镜头类词99%，如"Pan the frame
# slightly to the right, slightly zoom in")，两者必须分开。所以沿用上游原名。
#
# 表里没有的edit_type归一化后直接就是子集名，任何新增取值都会在
# check_load_annotation_count里被"unknown set"硬拦下来
GET_SET_NAME_DICT = {
    'object_addition': 'subject_addition',
    'object_deletion': 'subject_deletion',
    'object_replacement': 'subject_replacement',
}

# 找不到任务类型时才用的兜底子集名。
# 本数据集edit_type实测130877行全非空，一条都不会落进mix，
# 保留这条路径只是为了和002/006/007/009的口径保持一致，
# 并防止上游之后写出空edit_type时被静默漏处理
MIX_SET_NAME = 'mix'

# 最终产出的13个子集(即13种图像编辑任务类型)。
# 保留下来的子集集合必须与这个白名单严格一一对应，不多也不少
SAVE_SET_NAME_LIST = [
    'action_change',
    'background_change',
    'camera_movement',
    'color_change',
    'material_change',
    'object_movement',
    'personalized_generation',
    'style_change',
    'subject_addition',
    'subject_deletion',
    'subject_replacement',
    'text_change',
    'tone_transform',
]

# 保存图像名里只允许小写字母/数字/下划线/中划线/点，这个白名单与002/006/007/008/009
# 完全一致。
# 本数据集实测有**56个**样本对的路径里带非ASCII字符(全是nano-consistent的
# images/nano-consistent/Image/output/background/Park_Güell_Barcelona/ 这个目录，
# 名字里有ü)，按方案确认这56对整对丢弃(计入invalid_save_image_name_count)，
# 不做字符替换。
# 剩下130740个保存名100%满足这个模式、0个非法名，最长188字符
# (远低于文件系统单文件名255字节的上限)
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 只保留RGB三通道图和灰度图，P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须命中这个白名单，任意一张不合格则整个图像编辑对丢弃。
# 实测抽样120张(3个上游子集各40对的编辑后图+参考图)全是RGB，
# 'L'只是和009一样多兜一层灰度图
VALID_IMAGE_MODE_LIST = [
    'RGB',
    'L',
]

# 上游3个标注jsonl的实测行数，合计130877行(即上游写出的有效样本对数)。
# 解析阶段按标注名硬对账，少一行都说明上游010没跑完或产物被改动过。
# 注意这130877是上游**已经隔离掉158个不完整样本对之后**的数字
# (上游原始标注131035行，其中nano-consistent有25行instruction是空串、
#  ultravideo有1行edit_type是null、132行是edit_type=NO_CHANGE的"无变化"对照样本，
#  这158行上游已上报进unzip_check_result.json的invalid_sample_pair_list、
#  没有写进jsonl)，所以本脚本里的empty_caption等计数天然应该是0
EXPECTED_ANNOTATION_COUNT_DICT = {
    'nano-consistent': 22199,
    'ultravideo': 25674,
    'x2edit': 83004,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = 130877

# 按PER_ANNOTATION_SHARD_LINE_NUM=2000拆分后的实测分片数
# (nano-consistent 12 + ultravideo 13 + x2edit 42)
EXPECTED_TOTAL_ANNOTATION_SHARD_COUNT = 67

# 文本层各类不合格样本对的实测精确条数，解析阶段逐项硬对账。
# 判定顺序就是process_single_annotation_file里的顺序，所以这些数字之间不重复计数:
#   missing_task                : 0     (上游edit_type全非空)
#   missing_image               : 0     (上游图像路径全非空、参考图恒1张，
#                                        且上游自校验实测缺图0张)
#   invalid_save_image_name     : 56    (Park_Güell_Barcelona目录名里带ü)
#   empty_caption               : 0     (上游已把25行空指令隔离掉了)
#   null_like_caption           : 50    (全在x2edit的subject deletion:
#                                        None×41/Empty×4/no change×4/Blank×1)
#   no_word_char_caption        : 0     (没有"只剩标点"的指令)
#   too_short_caption           : 12    (strip后长度<4，如'Ink'/'Sky'/'Sun')
#   too_long_caption            : 1     (strip后长度>1024，最长那条1042字符)
#   double_image_caption        : 18    (ultravideo 13 + x2edit 5)
#   invalid_placeholder_caption : 0     (指令里连'['和']'都没有，纯防御)
EXPECTED_INVALID_ANNOTATION_COUNT_DICT = {
    'missing_task_count': 0,
    'missing_image_count': 0,
    'invalid_save_image_name_count': 56,
    'empty_caption_count': 0,
    'null_like_caption_count': 50,
    'no_word_char_caption_count': 0,
    'too_short_caption_count': 12,
    'too_long_caption_count': 1,
    'double_image_caption_count': 18,
    'invalid_placeholder_caption_count': 0,
}

# 130877行经文本层过滤丢掉137对后，实测落盘130740对。
# 这是全量实测(非抽样)的精确数字，解析阶段硬对账
EXPECTED_VALID_ANNOTATION_COUNT = 130740

# 13个子集的实测条数，合计130740对，解析阶段逐个硬对账。
# 子集级对账能额外拦住"某个edit_type被归并进错误子集"这种
# 标注文件级对账看不出来的问题
EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT = {
    'action_change': 11565,
    'background_change': 5991,
    'camera_movement': 15176,
    'color_change': 7708,
    'material_change': 7767,
    'object_movement': 1472,
    'personalized_generation': 7204,
    'style_change': 11218,
    'subject_addition': 20121,
    'subject_deletion': 15749,
    'subject_replacement': 11259,
    'text_change': 2515,
    'tone_transform': 12995,
}

# 最终产出的子集数，必须与SAVE_SET_NAME_LIST严格一一对应(不多也不少)。
# 13个子集里object_movement/text_change等5个不足1万对、只切出1个文件夹，
# subject_addition最多切3个，实测合计21个文件夹
EXPECTED_SAVE_SET_COUNT = 13

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

# 这个数据集主打"全部不低于4K的超高分辨率"，实测抽样120张编辑后图与参考图
# 短边最小3008、最大7680、宽高比最大1.85(4K/8K横竖屏为主)，
# 短边和宽高比这两个阈值预计一条都砍不到，纯粹作为跨数据集统一的兜底

# 实测130877条指令strip后 min 3 / p50 48 / p99 698 / max 1042，
# 各上游子集差异极大: x2edit是短指令(p50 41、max 177)，
# nano-consistent是多句改写长指令(p50 86、p90就已经669)。
#
# 下限取4(与008一致): 取10会砍掉463条，但里面大量是完全合法的单名词风格类指令
# ('Sky'×10/'Paint'×36/'Pixar.'×13/'Plants'×17/'Trees'×7，对应style/tone任务)，
# 属于砍正常样本而不是砍异常值; 取4后只砍掉12条(如'Ink'/'Sun')
MIN_CAPTION_LENGTH = 4

# 上限取1024: 取512会砍掉6123条(4.7%)，而这6123条几乎全是nano-consistent那种
# 天然极长的多句改写指令(该子集p90就有669)，属于砍正常样本;
# 取800砍99条、取1024只砍1条(即最长的那条1042字符)。
# 上限判定放在指令归一化之后，和写进json的口径完全一致，
# 收尾自校验直接量json里的长度就能复检
MAX_CAPTION_LENGTH = 1024

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 只剩标点的指令(如"...")没有任何可训练的语义，整对丢弃。实测0条命中，纯防御
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)，与008完全一致。
# 分两类:
# 1. null字面量: 上游构造流程失败时把空值写成了字符串;
# 2. 明确表示"不做任何修改"的指令: 编辑前后图应该几乎相同，当图像编辑训练样本
#    是纯噪声。
# 这里只做全串精确匹配、不做任何启发式猜测，因为上游有大量合法的单名词短指令
# ('Paint'/'Pixar.'/'Trees'，对应style change任务)，
# 一旦按"是不是单名词"来判会误伤上千条正常样本。
# 实测命中50条，全在x2edit的subject deletion(None×41/Empty×4/no change×4/Blank×1)
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
# 这套写法与002/006/007/008/009完全一致，保证跨数据集口径统一。
# 本数据集恒为1张参考图(N == 0)，即指令里不允许出现任何占位符
# (实测含[Vn*]形态的0条、连'['和']'都是0条)，这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2

# 双图指代指令(与007完全一致的规则): 这类文本不是"对着1张参考图下达的编辑指令"，
# 而是"同时看着编辑前后两张图写的对比描述"，形如
#   "The first image displays an aerial view of a large, circular stadium…"
#   "Change the positions and poses of all seagulls to match their
#    configuration in the second image."
# 新数据集每个编辑对只给1张参考图，这类指令预设的"两张图"根本不存在，语义是错的，
# 按方案整对丢弃。实测命中18条(ultravideo 13 + x2edit 5)。
#
# 注意007里那条"描述式前缀过滤"(^(the|in the|this|it|these))本数据集**不启用**:
# 它在本数据集命中1643条，但逐条看绝大多数是nano-consistent的合法祈使改写
# ("The woman in the green dress **should change from** … **to** …"、
#  "The young woman … **should be repositioned to** …")，
# 启用会大面积误伤正常样本
CAPTION_DOUBLE_IMAGE_PATTERN = re.compile(
    r'(first image|second image|left image|right image|left panel|right panel|diptych|both images|the two images|edit_operation_description)',
    re.I)

# jpg重编码质量与色度采样方式，与009.resave_foundir_dataset.py取同一套
# (质量97 + 色度4:4:4不下采样)，而不是cv2.imencode('.jpg', img)的默认值
# (质量95 + 色度4:2:0)。
#
# 【实测依据】本数据集的源图**261754次引用全部是png**(无损源)，
# 分辨率3008~7680、单张png 22~48MB，是所有已处理数据集里对重编码最敏感的一个
# (009那个数据集有近一半源图本身已是jpg、存在量化表幂等性，本数据集没有这个问题,
#  png源是真无损、编码器损失可以被直接度量)。
# 抽样12张编辑后图实测(先随机选上游子集、再在子集内随机选行):
#   配置                  编辑后图重编码PSNR   单张均值   全量261480张落盘估算
#   q95 4:2:0(cv2默认)         44.63 dB        4.37 MB        1.14 TB
#   q97 4:4:4                  47.81 dB        7.49 MB        1.96 TB  <- 本脚本采用
#   q98 4:4:4                  48.66 dB        8.71 MB        2.28 TB
# 真正的瓶颈是色度下采样而不是质量值: cv2默认的4:2:0会把色度分辨率直接砍半，
# 对本数据集这种超高分辨率图的细节(text_change子集2515对要求改画面里的文字、
# material_change 7767对要求改材质纹理)损伤最直接。
# 磁盘余量231T，1.96TB不构成约束，所以按方案取q97 + 4:4:4。
# 参考图与编辑后图共用同一套编码参数，避免两条编码链路引入
# "参考图多一层压缩"这种参考图/编辑后图不对称的伪偏差
SAVE_IMAGE_JPEG_QUALITY = 97

# 色度采样方式取4:4:4(不对色度做下采样)。cv2.imencode默认是4:2:0
SAVE_IMAGE_JPEG_SAMPLING_FACTOR = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444

# 落盘时统一使用的jpg编码参数，编辑后图和所有参考图都走这一套
SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST = [
    int(cv2.IMWRITE_JPEG_QUALITY),
    int(SAVE_IMAGE_JPEG_QUALITY),
    int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
    int(SAVE_IMAGE_JPEG_SAMPLING_FACTOR),
]


def get_normalized_edit_type_name(per_edit_type):
    """把上游edit_type归一化成统一写法: 小写 + 空格换下划线

    上游18个原始取值里存在同义异写("action change"与"action_change"、
    "background change"与"background_change"、"camera movement"等并存)，
    归一化后剩16个，这一步必须做，否则同一个任务会被切成两个子集。
    """
    return str(per_edit_type).strip().lower().replace(' ', '_')


def get_set_name(per_edit_type):
    """按edit_type推导子集名(即图像编辑任务类型)

    先归一化写法，再按GET_SET_NAME_DICT把object_*那3类归并到语义相同的subject_*
    (合并依据见该表处的实测注释)，其余归一化后的名字直接就是子集名。
    本数据集edit_type实测130877行全非空，一条都不会落进mix;
    只有连edit_type都拿不到、彻底找不到任务类型时才归到mix子集。
    """
    per_normalized_edit_type_name = get_normalized_edit_type_name(
        per_edit_type)

    if not per_normalized_edit_type_name:
        return MIX_SET_NAME

    return GET_SET_NAME_DICT.get(per_normalized_edit_type_name,
                                 per_normalized_edit_type_name)


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    这个数据集每个编辑对只有编辑前原图这一张参考图(上游reference_image_num
    实测130877行恒为1)，所以13个子集全都是1。
    保留这个函数是为了和002/006/007/008/009的收尾自校验保持同一套交叉对账写法。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def get_save_image_name_prefix(per_set_name, per_edited_image_relative_path,
                               per_reference_image_relative_path):
    """拼出这个图像编辑对的保存图像名前缀(全小写)

    口径是 {数据集名}_{子集名}_{原始编辑后图像名前缀}_ref_{原始参考图像名前缀}，
    两段"原图名前缀"的取法都必须按下面的实测结论来，否则会大面积撞名丢样本:

    1) 原始编辑后图像名前缀取的是**上游标注路径剥掉公共第一段images/之后的完整
       相对路径**(去后缀、小写、'/'换'_')，而不是basename。
       因为上游basename前缀在全局大面积重名: 130877行的basename前缀唯一值只有
       20467个、最大重名274次(x2edit全是压缩包目录内的序号000000001，
       ultravideo全是帧号0000/0149)，只用basename会让保存名塌缩到62393个、
       68484对样本互相覆盖丢失。

    2) 还必须拼上参考图的basename前缀。
       因为同一张编辑后图会被多行引用、每次配的是**不同的**参考图:
       118165张唯一编辑后图对应130877行(最多1张被8行引用，如同一个clip的
       0059.png分别以0000.png和0010.png为参考图)，只用编辑后图的完整相对路径
       仍有8556对撞名(最大重名7次)。

    按这个口径实测130877个保存名100%唯一、0重名，最长188字符
    (含_reference.jpg后仍远低于文件系统单文件名255字节的上限)。
    """
    per_edited_image_name_prefix = strip_image_root_dir_name(
        per_edited_image_relative_path)
    per_edited_image_name_prefix = os.path.splitext(
        per_edited_image_name_prefix)[0].strip().lower().replace('/', '_')

    per_reference_image_name_prefix = os.path.splitext(
        os.path.basename(
            per_reference_image_relative_path))[0].strip().lower()

    return (f'{DATASET_NAME}_{per_set_name}_{per_edited_image_name_prefix}'
            f'_ref_{per_reference_image_name_prefix}')


def strip_image_root_dir_name(per_image_relative_path):
    """剥掉上游图像相对路径的公共第一段images/

    实测130877行的路径第一段100%都是images，剥掉之后剩下的部分才是真正区分样本的
    那一段; 路径第一段不是images时原样返回(不静默改路径)，这条分支只在上游改了
    目录结构时才会走到，而那时保存名唯一性校验会把问题暴露出来。
    """
    per_image_relative_path = str(per_image_relative_path).replace(
        '\\', '/').strip().lstrip('/')

    if per_image_relative_path.startswith(f'{LOAD_IMAGE_ROOT_DIR_NAME}/'):
        return per_image_relative_path[len(LOAD_IMAGE_ROOT_DIR_NAME) + 1:]

    return per_image_relative_path


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令

    这个数据集的指令里没有任何视觉参考图占位符(实测含[Vn*]形态的0条、
    连'['和']'都是0条)，也没有"the reference image"这类自然语言指代
    (只有1张参考图、指令从不指代它)，所以这里只做strip，不做任何占位符改写。
    已经带编号的占位符原样保留，不做任何改动(便于后续多参考图数据集复用本函数)。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    return per_ti2i_caption


def check_invalid_caption(per_ti2i_caption, per_reference_image_num):
    """判定占位符编号与参考图数量不自洽的坏指令，返回True表示这条指令不合格

    参考图里第0张永远是编辑前原图(隐式、不占编号)，所以一条指令应该带的占位符编号
    正好是1...N，其中N = reference_image_num - 1。这里做三条校验:
    1. 同一个编号最多重复2次，超过就是逐字符插占位符的坏指令;
    2. 最大编号必须正好等于N，多了就是指代了不存在的参考图;
    3. 1...N每个编号都必须至少出现一次，不允许跳号，也不允许有图没被指代。
    本数据集恒为1张参考图(N为0)，即要求指令里完全没有占位符。
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


def check_null_like_caption(per_ti2i_caption):
    """判定指令是不是null字面量或"不做任何修改"这类无意义指令

    先小写化再剥掉两端标点空白，然后与黑名单做全串精确匹配，
    这样"None"/"None."/"no change."能被同一条规则一次判掉。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().strip(
        CAPTION_STRIP_CHAR)

    return per_ti2i_caption in NULL_LIKE_CAPTION_LIST


def check_double_image_caption(per_ti2i_caption):
    """判定双图指代指令，返回True表示这条指令不合格

    指令里出现first/second image、left/right image、left/right panel、diptych、
    both images、the two images、EDIT_OPERATION_DESCRIPTION时，
    它预设读者能同时看到编辑前后两张图，而新数据集每个编辑对只给1张参考图，
    这类文本是"对比描述"而不是"能直接执行的编辑指令"，整对丢弃。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    return CAPTION_DOUBLE_IMAGE_PATTERN.search(per_ti2i_caption) is not None


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上，逐样本打一次os.path.exists就是一次网络往返，
    而本数据集每行要判2张图、合计约26万次往返。上游图像落在6264个目录里
    (nano-consistent 270 + ultravideo 5334 + x2edit 660)，
    单目录最多22199个文件，所以这里按目录缓存一次os.listdir的结果、
    之后只做集合查表，网络往返次数从"图像引用数"降到"目录数"。
    分片是按标注行顺序切的、同一片里的图像高度聚集在少数几个目录下，
    所以每个worker的缓存开销可忽略。
    listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判，
    保证判定结果和不加缓存时完全一致。
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


def split_all_annotation_shard_file(root_dataset_path, save_shard_dir_path):
    """把上游3个标注jsonl按固定行数拆成分片文件

    上游只有3个标注文件(最大的x2edit.jsonl有83004行)，不拆分就只能开3个进程解析。
    这里主进程顺序读一遍3个文件、每PER_ANNOTATION_SHARD_LINE_NUM行写出一个分片，
    之后按分片粒度开多进程解析，解析完成后分片目录会被整个删除。
    实测按2000行一片拆出67片。
    benchmark.jsonl不在LOAD_ANNOTATION_FILE_NAME_LIST里，所以不会被拆也不会被读。
    """
    if os.path.exists(save_shard_dir_path):
        shutil.rmtree(save_shard_dir_path, ignore_errors=True)
    os.makedirs(save_shard_dir_path, exist_ok=True)

    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    total_line_count, illegal_line_count = 0, 0
    annotation_line_count_dict = {}
    annotation_shard_pair_list = []
    for per_annotation_file_name in LOAD_ANNOTATION_FILE_NAME_LIST:
        per_annotation_name = per_annotation_file_name[:-len(
            LOAD_ANNOTATION_FILE_NAME_SUFFIX)]
        per_annotation_path = os.path.join(root_annotation_path,
                                           per_annotation_file_name)

        per_annotation_line_count, per_shard_index = 0, 0
        per_shard_line_list = []
        try:
            with open(per_annotation_path, 'r',
                      encoding='UTF-8') as load_jsonl_file:
                for per_line in tqdm(load_jsonl_file):
                    per_line = per_line.strip()
                    if not per_line:
                        continue

                    total_line_count += 1
                    per_annotation_line_count += 1

                    per_shard_line_list.append(per_line)
                    if len(per_shard_line_list
                           ) < PER_ANNOTATION_SHARD_LINE_NUM:
                        continue

                    annotation_shard_pair_list.append(
                        save_single_annotation_shard_file(
                            save_shard_dir_path, per_annotation_name,
                            per_shard_index, per_shard_line_list))
                    per_shard_index += 1
                    per_shard_line_list = []

            if len(per_shard_line_list) > 0:
                annotation_shard_pair_list.append(
                    save_single_annotation_shard_file(save_shard_dir_path,
                                                      per_annotation_name,
                                                      per_shard_index,
                                                      per_shard_line_list))
                per_shard_index += 1
        except Exception as e:
            illegal_line_count += 1
            print('2222', per_annotation_path, e)

        annotation_line_count_dict[
            per_annotation_name] = per_annotation_line_count

        print('1111', per_annotation_name, 'line', per_annotation_line_count,
              'shard', per_shard_index)

    annotation_shard_pair_list = sorted(annotation_shard_pair_list,
                                        key=lambda x: x[0])

    return [
        annotation_shard_pair_list,
        annotation_line_count_dict,
        total_line_count,
        illegal_line_count,
    ]


def save_single_annotation_shard_file(save_shard_dir_path, per_annotation_name,
                                      per_shard_index, per_shard_line_list):
    """把一个分片的标注行写成一个jsonl分片文件，返回该分片的解析任务"""
    per_shard_file_name = (
        f'{per_annotation_name}{ANNOTATION_SHARD_FILE_NAME_SEPARATOR}'
        f'{per_shard_index:05d}{LOAD_ANNOTATION_FILE_NAME_SUFFIX}')
    per_shard_path = os.path.join(save_shard_dir_path, per_shard_file_name)

    with open(per_shard_path, 'w', encoding='UTF-8') as save_jsonl_file:
        for per_line in per_shard_line_list:
            save_jsonl_file.write(f'{per_line}\n')

    return [
        per_shard_path,
        per_annotation_name,
    ]


def process_single_annotation_file(annotation_file_pair):
    """解析单个标注分片，组装图像编辑对(参考图+编辑后图+编辑指令)的列表

    这一步只做纯文本层面的过滤(json坏行、行内标注名与分片不一致、缺任务类型、
    缺图或图不存在、参考图数量不对、保存名非法、指令为空、指令是null字面量、
    指令没有任何文字字符、指令过短、指令过长、指令是双图指代指令、
    指令是坏占位符指令)，
    图像本身的解码校验和分辨率过滤留到后面多进程里做。
    这些过滤都必须在切分文件夹之前做完，后面每10000对切一个文件夹才能切得满。
    """

    per_annotation_path, per_annotation_name, root_image_path = annotation_file_pair

    annotation_list = []
    illegal_line_count = 0
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_jsonl_file:
            for per_line in load_jsonl_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                try:
                    annotation_list.append(json.loads(per_line))
                except Exception as e:
                    illegal_line_count += 1
                    print('2222', per_annotation_path, e)
                    continue
    except Exception as e:
        print('2222', per_annotation_path, e)

    total_annotation_count = len(annotation_list) + illegal_line_count
    missing_task_count, missing_image_count = 0, 0
    invalid_save_image_name_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    double_image_caption_count = 0
    invalid_placeholder_caption_count = 0
    annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    # 分片是按标注行顺序切的，同一片里的图像高度聚集在少数几个目录下，
    # 所以这个缓存里的目录数很少、内存开销可忽略
    dir_file_name_cache_dict = {}

    for per_annotation in annotation_list:
        if not isinstance(per_annotation, dict):
            illegal_line_count += 1
            print('2222', per_annotation_path)
            continue

        per_load_annotation_name = per_annotation.get(ANNOTATION_NAME_KEY_NAME,
                                                      '')
        if not isinstance(per_load_annotation_name, str):
            per_load_annotation_name = ''
        per_load_annotation_name = per_load_annotation_name.strip()

        # 行内标注名必须和这个分片来自的标注文件一致: 不一致说明上游产物被搬动过
        # 或分片写错了，继续跑会把条数记到错误的标注名下、让对账自说自话地对上
        if per_load_annotation_name != per_annotation_name:
            illegal_line_count += 1
            print('2222', per_annotation_path, per_load_annotation_name)
            continue

        annotation_count_dict[per_annotation_name] = annotation_count_dict.get(
            per_annotation_name, 0) + 1

        per_edit_type = per_annotation.get(ANNOTATION_EDIT_TYPE_KEY_NAME, '')
        if not isinstance(per_edit_type, str):
            per_edit_type = ''
        per_edit_type = per_edit_type.strip()

        # 任务类型决定子集名，缺了就无法安全落盘(实测0条缺失，这里只做防御)
        if not per_edit_type:
            missing_task_count += 1
            continue

        per_set_name = get_set_name(per_edit_type)

        per_edited_image_relative_path = per_annotation.get(
            ANNOTATION_EDITED_IMAGE_KEY_NAME, '')
        if not isinstance(per_edited_image_relative_path, str):
            per_edited_image_relative_path = ''
        per_edited_image_relative_path = per_edited_image_relative_path.replace(
            '\\', '/').strip().lstrip('/')
        if not per_edited_image_relative_path:
            missing_image_count += 1
            continue

        per_edited_image_path = os.path.join(root_image_path,
                                             per_edited_image_relative_path)
        if not check_image_file_exists(per_edited_image_path,
                                       dir_file_name_cache_dict):
            missing_image_count += 1
            continue

        # 参考图顺序固定，第0张一定是编辑前的原图(这个数据集也只有这一张)
        per_load_reference_image_relative_path_list = per_annotation.get(
            ANNOTATION_REFERENCE_IMAGE_KEY_NAME, [])
        if not isinstance(per_load_reference_image_relative_path_list,
                          (list, tuple)):
            per_load_reference_image_relative_path_list = []

        per_reference_image_relative_path_list = []
        per_missing_reference_image_count = 0
        for per_reference_image_relative_path in per_load_reference_image_relative_path_list:
            if not isinstance(per_reference_image_relative_path, str):
                per_missing_reference_image_count += 1
                continue

            per_reference_image_relative_path = per_reference_image_relative_path.replace(
                '\\', '/').strip().lstrip('/')
            if not per_reference_image_relative_path:
                per_missing_reference_image_count += 1
                continue

            if not check_image_file_exists(
                    os.path.join(root_image_path,
                                 per_reference_image_relative_path),
                    dir_file_name_cache_dict):
                per_missing_reference_image_count += 1
                continue

            per_reference_image_relative_path_list.append(
                per_reference_image_relative_path)

        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)
        # 参考图缺任意一张都会让这个编辑对的条件信息不完整，整对丢弃
        if per_missing_reference_image_count > 0 or len(
                per_reference_image_relative_path_list
        ) != per_expect_reference_image_num:
            missing_image_count += 1
            continue

        # 保存图像名前缀，取法见get_save_image_name_prefix的注释
        # (必须用编辑后图的完整相对路径 + 参考图的basename，否则大面积撞名丢样本)
        per_save_image_name_prefix = get_save_image_name_prefix(
            per_set_name, per_edited_image_relative_path,
            per_reference_image_relative_path_list[0])
        per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
        per_save_reference_image_name_list = [
            f'{per_save_image_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}'
        ]

        # 保存名里出现路径分隔符或其它异常字符会写坏目录结构，整对丢弃。
        # 实测命中56条，全是nano-consistent的Park_Güell_Barcelona目录(名字里带ü)
        per_invalid_save_image_name_flag = not VALID_IMAGE_NAME_PATTERN.match(
            per_save_edited_image_name)
        for per_save_reference_image_name in per_save_reference_image_name_list:
            if not VALID_IMAGE_NAME_PATTERN.match(
                    per_save_reference_image_name):
                per_invalid_save_image_name_flag = True
        # 同一个样本对里两张参考图撞名会互相覆盖，整对丢弃
        # (本数据集只有1张参考图，这里只做防御性拦截)
        if len(set(per_save_reference_image_name_list)) != len(
                per_save_reference_image_name_list):
            per_invalid_save_image_name_flag = True

        if per_invalid_save_image_name_flag:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_save_edited_image_name)
            continue

        # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的前缀
        # (即带_edited那一段)，和002/006/007/008/009的写法保持一致，
        # 收尾自校验也是按edited_image去掉.jpg来反推这个文件夹名的
        per_save_pair_folder_name = os.path.splitext(
            per_save_edited_image_name)[0]

        per_ti2i_caption = per_annotation.get(ANNOTATION_CAPTION_KEY_NAME, '')
        if isinstance(per_ti2i_caption, (list, tuple)):
            per_ti2i_caption = per_ti2i_caption[0] if len(
                per_ti2i_caption) > 0 else ''
        if not isinstance(per_ti2i_caption, str):
            per_ti2i_caption = ''
        per_ti2i_caption = per_ti2i_caption.strip()

        # 空指令、全空格指令视为不合格图像编辑对
        # (上游010已把25行空指令隔离掉了，这里实测0条命中)
        if not per_ti2i_caption:
            empty_caption_count += 1
            continue

        # null字面量与"不做任何修改"这类无意义指令同样丢弃(实测50条，全在x2edit)
        if check_null_like_caption(per_ti2i_caption):
            null_like_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:50])
            continue

        # 只剩标点、没有任何数字/字母/汉字的指令也丢弃(实测0条，纯防御)
        if not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption):
            no_word_char_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:50])
            continue

        # 过短指令视为不合格图像编辑对(实测12条，如'Ink'/'Sky'/'Sun')
        if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
            too_short_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 本数据集的指令不需要任何占位符改写，这里只做strip，
        # 写进json的一定是归一化后的指令
        per_ti2i_caption = get_normalized_ti2i_caption(per_ti2i_caption)

        # 过长指令同样视为不合格图像编辑对，按归一化后的指令判定，
        # 和写进json的指令口径完全一致，收尾自校验直接量json里的长度就能复检
        # (实测1条，即最长的那条1042字符)
        if len(per_ti2i_caption) > MAX_CAPTION_LENGTH:
            too_long_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 双图指代指令丢弃: 它预设能同时看到编辑前后两张图，而这里只给1张参考图
        # (实测18条: ultravideo 13 + x2edit 5)
        if check_double_image_caption(per_ti2i_caption):
            double_image_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:100])
            continue

        # 占位符编号与参考图数量不自洽的指令也丢弃。
        # 本数据集是单参考图，即要求指令里完全没有[Vn*]占位符(实测0条命中)
        if check_invalid_caption(per_ti2i_caption,
                                 per_expect_reference_image_num):
            invalid_placeholder_caption_count += 1
            print('3333', per_edited_image_path, per_ti2i_caption[:100])
            continue

        set_annotation_count_dict[
            per_set_name] = set_annotation_count_dict.get(per_set_name, 0) + 1

        edit_annotation_pair_list.append([
            per_set_name,
            per_save_pair_folder_name,
            per_edited_image_path,
            per_save_edited_image_name,
            [
                os.path.join(root_image_path,
                             per_reference_image_relative_path)
                for per_reference_image_relative_path in
                per_reference_image_relative_path_list
            ],
            per_save_reference_image_name_list,
            per_ti2i_caption,
            per_expect_reference_image_num,
        ])

    return [
        edit_annotation_pair_list,
        annotation_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        missing_task_count,
        missing_image_count,
        invalid_save_image_name_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        double_image_caption_count,
        invalid_placeholder_caption_count,
    ]


def get_all_edit_annotation_pair(root_dataset_path, save_shard_dir_path):
    """按标注分片粒度多进程组装全部图像编辑对的列表

    上游3个jsonl合计130877行、逐行还要判2张图像文件是否存在，
    所以先把3个jsonl拆成67个分片，再按分片开多进程解析，
    最后按保存的编辑后图像名统一排序。
    """
    root_image_path = os.path.join(root_dataset_path,
                                   *LOAD_IMAGE_DIR_NAME_LIST)

    annotation_shard_pair_list, annotation_line_count_dict, total_line_count, split_illegal_line_count = split_all_annotation_shard_file(
        root_dataset_path, save_shard_dir_path)

    print('1111', 'annotation file:',
          len(LOAD_ANNOTATION_FILE_NAME_LIST), 'annotation shard:',
          len(annotation_shard_pair_list), 'annotation line:',
          total_line_count)

    annotation_file_pair_list = [[
        per_annotation_shard_path,
        per_annotation_name,
        root_image_path,
    ] for per_annotation_shard_path, per_annotation_name in
                                 annotation_shard_pair_list]

    total_annotation_count = 0
    illegal_line_count = split_illegal_line_count
    missing_task_count, missing_image_count = 0, 0
    invalid_save_image_name_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    double_image_caption_count = 0
    invalid_placeholder_caption_count = 0
    annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            for per_annotation_name, per_annotation_count in per_load_result[
                    1].items():
                annotation_count_dict[
                    per_annotation_name] = annotation_count_dict.get(
                        per_annotation_name, 0) + per_annotation_count

            for per_set_name, per_set_count in per_load_result[2].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

            total_annotation_count += per_load_result[3]
            illegal_line_count += per_load_result[4]
            missing_task_count += per_load_result[5]
            missing_image_count += per_load_result[6]
            invalid_save_image_name_count += per_load_result[7]
            empty_caption_count += per_load_result[8]
            null_like_caption_count += per_load_result[9]
            no_word_char_caption_count += per_load_result[10]
            too_short_caption_count += per_load_result[11]
            too_long_caption_count += per_load_result[12]
            double_image_caption_count += per_load_result[13]
            invalid_placeholder_caption_count += per_load_result[14]

    # 分片只是解析用的中间产物，解析完立刻删掉，不留在输出目录里
    if os.path.exists(save_shard_dir_path):
        shutil.rmtree(save_shard_dir_path, ignore_errors=True)

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        annotation_line_count_dict,
        annotation_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        missing_task_count,
        missing_image_count,
        invalid_save_image_name_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        double_image_caption_count,
        invalid_placeholder_caption_count,
    ]


def check_load_annotation_count(
        annotation_count_dict, set_annotation_count_dict,
        total_annotation_shard_count, total_annotation_count,
        valid_annotation_count, illegal_line_count,
        invalid_annotation_count_dict, edit_annotation_pair_list):
    """解析完标注后按标注文件和子集两级硬对账，并检查保存图像名是否唯一

    上游标注是010一次性跑出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账能额外拦住"某个edit_type被归并进错误子集"这种
    标注文件级对账看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对
    (本数据集的保存名口径就是靠这一步实测出来的，见
     get_save_image_name_prefix的注释)。
    """
    check_error_message_list = []

    for per_annotation_name in sorted(annotation_count_dict.keys()):
        if per_annotation_name not in EXPECTED_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown annotation {per_annotation_name}')
            continue

        per_expect_annotation_count = EXPECTED_ANNOTATION_COUNT_DICT[
            per_annotation_name]
        if annotation_count_dict[
                per_annotation_name] != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_annotation_name} annotation count not match '
                f'{annotation_count_dict[per_annotation_name]} != '
                f'{per_expect_annotation_count}')

    for per_annotation_name in sorted(EXPECTED_ANNOTATION_COUNT_DICT.keys()):
        if per_annotation_name not in annotation_count_dict:
            check_error_message_list.append(
                f'missing annotation {per_annotation_name}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    if total_annotation_shard_count != EXPECTED_TOTAL_ANNOTATION_SHARD_COUNT:
        check_error_message_list.append(
            f'total annotation shard count not match '
            f'{total_annotation_shard_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_SHARD_COUNT}')

    # 上游jsonl是010用json.dumps一行一条写出来的，不允许有坏行
    if illegal_line_count > 0:
        check_error_message_list.append(
            f'illegal line count {illegal_line_count}')

    # 文本层各类不合格样本对逐项硬对账
    for per_count_name in sorted(
            EXPECTED_INVALID_ANNOTATION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_ANNOTATION_COUNT_DICT[
            per_count_name]
        if invalid_annotation_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_annotation_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    if valid_annotation_count != EXPECTED_VALID_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'valid annotation count not match '
            f'{valid_annotation_count} != {EXPECTED_VALID_ANNOTATION_COUNT}')

    # 保留下来的样本对数 + 各类被丢弃的样本对数 必须等于标注总行数，
    # 否则说明有样本对在解析链路上凭空消失了
    if valid_annotation_count + sum(invalid_annotation_count_dict.values(
    )) + illegal_line_count != total_annotation_count:
        check_error_message_list.append(
            f'valid + invalid annotation count not self consistent '
            f'{valid_annotation_count} + '
            f'{sum(invalid_annotation_count_dict.values())} + '
            f'{illegal_line_count} != {total_annotation_count}')

    # 13个保留子集逐个硬对账
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
    # (上游新增edit_type取值时会在这里被硬拦下来)
    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in SAVE_SET_NAME_LIST:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

    # 保存的编辑后图像名必须全局唯一，撞名会让两个样本对在磁盘和json里互相覆盖
    save_edited_image_name_set = set()
    duplicate_save_edited_image_name_list = []
    for per_edit_annotation_pair in edit_annotation_pair_list:
        per_save_edited_image_name = per_edit_annotation_pair[3]
        if per_save_edited_image_name in save_edited_image_name_set:
            duplicate_save_edited_image_name_list.append(
                per_save_edited_image_name)
            continue
        save_edited_image_name_set.add(per_save_edited_image_name)

    if len(duplicate_save_edited_image_name_list) > 0:
        check_error_message_list.append(
            f'duplicate save edited image name num '
            f'{len(duplicate_save_edited_image_name_list)} '
            f'{duplicate_save_edited_image_name_list[:5]}')

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
    # 所以必须先用PIL读原始mode才能把P图/CMYK图判出来
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
    本数据集会产出13个子集，其中object_movement/text_change等5个不足1万对、
    只切出1个文件夹，subject_addition最多切3个，实测合计21个文件夹。
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

    上游261754次图像引用全部是4K~8K的png，这里统一重编码成jpg，
    只换编码格式不换像素尺寸。
    编码参数显式用SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST(质量97 + 色度4:4:4)，
    而不是cv2的默认值(质量95 + 色度4:2:0): png源是真无损、编码器损失可以被直接
    度量，实测默认配置编辑后图只有44.63dB、本配置到47.81dB，
    而且真正的瓶颈是色度下采样不是质量值(默认的4:2:0把色度分辨率直接砍半，
    对text_change/material_change这类要求改画面文字与材质纹理的子集损伤最直接)，
    详见常量处的实测表。
    编辑后图和参考图共用同一套参数，避免两条编码链路引入不对称的伪偏差。
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
            cv2.imencode(
                '.jpg', per_image,
                SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST)[1].tofile(save_image_path)
        except Exception as e:
            print('8888', save_image_path, e)
            return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_edit_pair(edit_pair_save_folder_pair, save_dataset_path):
    """重新编码保存单个图像编辑对的编辑后图像和所有参考图像，任意一张失败则整对丢弃

    注意本数据集有一部分源图会被重复写盘: 118165张唯一编辑后图被130877行引用、
    99550张唯一参考图被130877行引用，即约4.4万张图会以不同的保存名写进不同的
    样本对文件夹。这是"每个图像编辑对独占一个文件夹、该对的图都存在里面"这个
    目录规范的必然结果，按方案不做去重(重复引用数会上报进resave_check_result.json)。
    """
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
    上游jsonl里剩下的sample_id/subset_name/annotation_name/archive_group_name/
    task_type/row_index/reference_image_num全部丢弃、不另存索引，
    理由见文件开头SAVE_ANNOTATION_KEY_NAME_LIST的注释。
    ti2i_caption写的就是上游instruction字段strip后的原文
    (含那6条中英混写的指令也原样保留)，本数据集恒1张参考图，
    指令里不含任何视觉参考图占位符。
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
        # 保证记录的长度和ti2i_caption永远自洽(该字符串已strip并归一化过)
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
    只剩标点的无意义指令、不能是双图指代指令、记录的长度必须与字符串实际长度一致。
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
                # 本数据集全部13个子集都必须是单参考图
                if per_annotation[
                        'reference_image_num'] != per_expect_reference_image_num:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match set {per_annotation["reference_image_num"]} != {per_expect_reference_image_num}'
                    )
                # 参考图名的前缀必须和编辑后图像名同一个前缀、后缀必须是_reference.jpg
                for per_save_reference_image_name in per_annotation[
                        'reference_image']:
                    if not per_save_reference_image_name.endswith(
                            SAVE_REFERENCE_IMAGE_NAME_SUFFIX):
                        check_error_message_list.append(
                            f'{per_save_reference_image_name} reference image name suffix not match'
                        )
                    if not VALID_IMAGE_NAME_PATTERN.match(
                            per_save_reference_image_name):
                        check_error_message_list.append(
                            f'{per_save_reference_image_name} reference image name not match pattern'
                        )
                    if per_save_reference_image_name.removesuffix(
                            SAVE_REFERENCE_IMAGE_NAME_SUFFIX
                    ) != per_save_edited_image_name.removesuffix(
                            SAVE_EDITED_IMAGE_NAME_SUFFIX):
                        check_error_message_list.append(
                            f'{per_save_reference_image_name} reference image name prefix not match {per_save_edited_image_name}'
                        )
                # 图像宽高必须是正数
                if per_annotation['width'] <= 0 or per_annotation[
                        'height'] <= 0:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} edited image shape not match {per_annotation["width"]} {per_annotation["height"]}'
                    )
                # 短边和宽高比必须仍然满足过滤阈值
                if min(per_annotation['width'],
                       per_annotation['height']) < MIN_IMAGE_SHORT_SIDE:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a too small image {per_annotation["width"]} {per_annotation["height"]}'
                    )
                if max(per_annotation['width'] / per_annotation['height'],
                       per_annotation['height'] /
                       per_annotation['width']) > MAX_IMAGE_ASPECT_RATIO:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still an extreme aspect ratio image {per_annotation["width"]} {per_annotation["height"]}'
                    )
                # ti2i_caption的占位符编号集合必须与reference_image这个list的
                # 长度自洽: 本数据集恒1张参考图，即不允许出现任何占位符
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
                # 也不允许残留双图指代指令
                if check_double_image_caption(per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a double image caption'
                    )
                # json里存的就是归一化后的指令，长度过滤也是按归一化后判定的，
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

    save_shard_dir_path = os.path.join(save_dataset_path,
                                       SAVE_ANNOTATION_SHARD_DIR_NAME)

    edit_annotation_pair_list, total_annotation_shard_count, annotation_line_count_dict, annotation_count_dict, set_annotation_count_dict, total_annotation_count, illegal_line_count, missing_task_count, missing_image_count, invalid_save_image_name_count, empty_caption_count, null_like_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, double_image_caption_count, invalid_placeholder_caption_count = get_all_edit_annotation_pair(
        root_dataset_path, save_shard_dir_path)

    print('1111', total_annotation_shard_count, total_annotation_count,
          illegal_line_count, missing_task_count, missing_image_count,
          invalid_save_image_name_count, empty_caption_count,
          null_like_caption_count, no_word_char_caption_count,
          too_short_caption_count, too_long_caption_count,
          double_image_caption_count, invalid_placeholder_caption_count,
          len(set_annotation_count_dict), len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    invalid_annotation_count_dict = {
        'missing_task_count': missing_task_count,
        'missing_image_count': missing_image_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'double_image_caption_count': double_image_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        annotation_count_dict, set_annotation_count_dict,
        total_annotation_shard_count, total_annotation_count,
        len(edit_annotation_pair_list), illegal_line_count,
        invalid_annotation_count_dict, edit_annotation_pair_list)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        # 上游标注条数对不上说明上游010没跑完或产物被改动过，
        # 继续往下跑只会得到一个悄悄少样本的新数据集
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
          'different aspect ratio:', different_aspect_ratio_count,
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

    print(
        '3333', 'total annotation file:', len(LOAD_ANNOTATION_FILE_NAME_LIST),
        'total annotation shard:', total_annotation_shard_count,
        'total annotation:', total_annotation_count, 'illegal line:',
        illegal_line_count, 'missing task:', missing_task_count,
        'missing image:', missing_image_count, 'invalid save image name:',
        invalid_save_image_name_count, 'empty caption:', empty_caption_count,
        'null like caption:', null_like_caption_count, 'no word char caption:',
        no_word_char_caption_count, 'too short caption:',
        too_short_caption_count, 'too long caption:', too_long_caption_count,
        'double image caption:', double_image_caption_count,
        'invalid placeholder caption:', invalid_placeholder_caption_count,
        'invalid image:', invalid_image_count, 'save edit pair failed:',
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
        'total_annotation_file_count': len(LOAD_ANNOTATION_FILE_NAME_LIST),
        'total_annotation_shard_count': total_annotation_shard_count,
        'total_annotation_count': total_annotation_count,
        'illegal_line_count': illegal_line_count,
        'missing_task_count': missing_task_count,
        'missing_image_count': missing_image_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'double_image_caption_count': double_image_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
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
        'save_image_jpeg_quality': SAVE_IMAGE_JPEG_QUALITY,
        'save_image_jpeg_sampling_factor_444_flag': True,
        'annotation_line_count_dict': annotation_line_count_dict,
        'annotation_count_dict': annotation_count_dict,
        'set_annotation_count_dict': set_annotation_count_dict,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/VINS-120K'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
