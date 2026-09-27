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

DATASET_NAME = 'inter_edit'

SAVE_DATASET_DIR_NAME = 'InterEdit'

# ==============================================================================
# 【这个数据集只能产出图像编辑数据集，不能产出文生图数据集】
# 上游016.unzip_inter_edit_train_dataset.py解出来的每一行有26个字段，
# 其中唯一的文本字段是instruction(编辑指令)，形如"添加一双发光的筷子"、
# "remove the plastic bag from the table"、"将其改为大理石纹理"。
# 整个数据集**没有任何一列是图像内容描述(caption)**: 既没有编辑前原图的描述、
# 也没有编辑后图的描述。编辑指令只说"要改什么"、不说"整张图是什么"，
# 拿它当t2i的prompt会得到完全错误的图文对(比如"移除"配一张桌子图)，
# 所以本数据集只走ti2i这一条链路，不另写t2i脚本。
# 上游016自己也把dataset_task_type写死成image_edit，
# 官方README的task_categories是image-to-image、tags是image-editing。
#
# 【本脚本 vs 017.1: 同一份上游数据的两个落盘版本】
# 上游每个样本对除了"参考图(编辑前source) + 编辑指令 + 编辑后图(target)"之外，
# 还带一张**编辑区域mask**(1099857张，8bit灰度PNG、实测100%二值{0,255}、
# 形状不规则贴合物体轮廓)。这是I^3E(交互式指令编辑)任务的空间条件。
#   本脚本(017.0): **不使用mask**，reference_image只有编辑前原图这1张，
#                  reference_image_num恒为1，ti2i_caption里**不含任何占位符**;
#                  因为没有空间条件，"移除"这类没有指代对象的通用指令必须整体丢弃
#                  (见下面的CAPTION_GENERIC_NO_TARGET_LIST)。
#   017.1        : 把mask当作第2张参考图，reference_image_num恒为2，
#                  ti2i_caption统一改写成带[V1*]占位符的形态，
#                  因此那批通用指令加上mask后语义完整、可以全部保留。
# 两个版本落盘到两个互不覆盖的根目录(Inter-Edit / Inter-Edit-Mask)。
#
# 【上游016的产物规格(实测)】
# Inter-Edit-Train/
# ├── manifest.json                                官方规格清单(本脚本不读)
# ├── unzip_annotations/train-{00000..00274}-of-00275.jsonl   275个，合计1099857行
# ├── unzip_images/sources/<source分片名>/source_{7位}.png     610156张(唯一源图)
# ├── unzip_images/targets/<asset分片名>/target_{7位}.png      1099857张(编辑后图)
# ├── unzip_images/masks/<asset分片名>/mask_{7位}.png          1099857张(编辑区域mask)
# └── unzip_check_missing_images.json              上游自校验报告(本脚本不读)
# 上游全量1099964条里有107条"clip后bbox仍退化"的样本对已被016丢弃、没有落盘，
# 所以标注只有1099857行，本脚本按1099857硬对账。
#
# 【全量实测规格(1099857行全部扫过，非抽样)】
# - 26个字段1099857/1099857条全部齐备，无缺字段、无多字段;
# - sample_key全局唯一(1099857个，7位数字)，所以保存图像名天然全局唯一;
# - edit_type只有4种: Local 408012 / Add 312261 / Remove 308044 / Texture 71540，
#   0条缺失，所以子集直接由它划分、不需要mix兜底(mix只作防御性保留);
# - instruction: 0条空、0条null字面量、0条纯标点、0条含[Vn*]占位符;
# - 指令语言: 中文624019(p50=7 / min=2 / max=191) + 英文475838(p50=28 / min=3 /
#   max=120)，中英混杂，按005(x2edit)口径原样保留、不做任何语言归一化;
# - 参考图恒为1张(reference_image_path_list长度恒为1);
# - 唯一源图610156张，被1099857个编辑对引用(复用1次230001 / 2次270609 /
#   3次109546)，所以同一张源图会被重复编码落盘2~3份。这是
#   "每个编辑对独占一个文件夹、该对的参考图与编辑后图都放里面"这个格式规范
#   决定的，按方案确认接受这个冗余;
# - 编辑后图分辨率只有7种(1024x1024 545512 / 896x1184 93008 / 1184x896 92799 /
#   832x1248 92675 / 1376x768 92539 / 1248x832 92176 / 768x1376 91148)，
#   **短边最小768、宽高比最大1.79**，所以下面的短边64/宽高比8两个阈值
#   实测一条都砍不到，纯粹作为跨数据集统一的兜底;
# - 抽样200张(编辑后图100 + 参考图100)实测100%是RGB模式的PNG。
#
# 【**必须显式感知的规格坑: 参考图与编辑后图分辨率100%不相等**】
# 全量1099857行实测 source_image_shape == edited_image_shape 的有 **0 条**。
# 这不是脏数据，而是上游的发布规格: 源图与编辑后图分别按两个不同的分辨率档位生成，
# 7种组合覆盖全量、非常规整:
#   源图 -> 编辑后图                 长宽比            条数
#   1328x1328 -> 1024x1024          相同(1:1)        545512   <- 保留(resize)
#   1056x1584 -> 832x1248           相同(2:3)         92675   <- 保留(resize)
#   1584x1056 -> 1248x832           相同(3:2)         92176   <- 保留(resize)
#   1104x1472 -> 896x1184           0.7500 vs 0.7568  93008   <- 丢弃
#   1472x1104 -> 1184x896           1.3333 vs 1.3214  92799   <- 丢弃
#   1664x928  -> 1376x768           1.7931 vs 1.7917  92539   <- 丢弃
#   928x1664  -> 768x1376           0.5577 vs 0.5581  91148   <- 丢弃
# 按方案确认的处理口径(两个脚本完全一致):
#   1. 下游要求"第一张参考图与生成图尺寸必须一致"。直接按"尺寸不等就丢弃"执行会
#      把整个数据集清空(0条满足)，所以改成**能对齐的就resize对齐、对不齐的才丢弃**;
#   2. **长宽比严格相同**(用Fraction最简分数比判定，避免浮点误差)的730363条:
#      把参考图resize到编辑后图的尺寸(等比缩小，不产生任何形变);
#   3. **长宽比不同**的369494条(上面那4种组合): resize会把画面拉伸变形、
#      按方案确认**整对丢弃**;
#   4. resize用PIL的LANCZOS(先BGR->RGB转PIL、resize、再转回numpy并RGB->BGR)，
#      而不是cv2.resize: 与项目既定口径保持一致。
# 落盘后**编辑后图与参考图尺寸严格相等**，
# 收尾自校验会**实读参考图的shape**与json里的width/height硬对账。
# ==============================================================================

# 本脚本只读这一套标注(275个jsonl、1099857行)。
# 绝不os.walk图像目录: 上游解出约330万个小文件，扫目录树在NAS上不可接受
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 上游标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如unzip_images/targets/asset-00000-of-00275/target_0000000.png)，
# 不需要再往前拼任何子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 样本主干id(7位数字字符串)，实测1099857个全局唯一，
# 保存图像名的"原始图像名前缀"取的就是它
ANNOTATION_SAMPLE_KEY_KEY_NAME = 'sample_key'

# 图像编辑任务类型，实测1099857行全非空、只有4种取值。
# 直接用它分子集，所以不需要mix兜底子集
ANNOTATION_EDIT_TYPE_KEY_NAME = 'edit_type'

# 编辑指令，也是本数据集唯一的文本字段。实测1099857行全非空
ANNOTATION_CAPTION_KEY_NAME = 'instruction'

# 参考图(编辑前原图source)的相对路径列表，本数据集恒为长度1的list
ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'reference_image_path_list'

# 编辑后图(target)的相对路径，实测1099857行全非空
ANNOTATION_EDITED_IMAGE_KEY_NAME = 'edited_image_path'

# 上游记录的参考图(源图)与编辑后图的[宽, 高]，实测1099857行全非空、全是2元int list。
# **只用来判长宽比是否相同**(据此决定resize还是丢弃)，
# 写进json的宽高一律以实际写盘数组的shape为准、不采信这两个字段。
# 放在文本过滤之后判，这样只需读标注、不用解图就能把369494条形变样本挡在解码之前
ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME = 'source_image_shape'

ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME = 'edited_image_shape'

# 上游落盘的图像名规格，用来和sample_key交叉对账:
#   targets/target_{sample_key}.png   编辑后图
#   sources/source_{source_id}.png    参考图(注意这里是source_id不是sample_key，
#                                     因为源图是全局去重后单独发布的)
LOAD_EDITED_IMAGE_NAME_PATTERN = re.compile(r'^target_(?P<index>\d{7})\.png$')

LOAD_REFERENCE_IMAGE_NAME_PATTERN = re.compile(
    r'^source_(?P<index>\d{7})\.png$')

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 新标注固定只存这七个key，多一个少一个都在收尾自校验里报错。
# 上游标注里剩下的属性按方案确认全部丢弃、不另存索引:
# 【mask相关3个字段 + 1099857张mask图】按方案确认本版本整体不使用、不落盘:
#   mask_image_path / mask_image_shape / mask_image_shape_state
#   mask是编辑区域的二值标注(实测100%二值{0,255}、形状不规则贴合物体轮廓、
#   前景占比Add p50 8.5% / Remove 9.1% / Local 16.0% / Texture 26.9%)，
#   本来可以做局部重绘监督，但七字段格式里没有位置放它:
#   放进reference_image会让reference_image_num变2、并要求指令里出现[V1*]占位符，
#   与本版本"单参考图 + 指令无占位符"的口径直接冲突。
#   需要mask的版本请用017.1，上游unzip_images/masks/会一直保留。
# 【bounding_box系列5个字段】按方案确认整体丢弃:
#   bounding_box / clipped_bounding_box / normalized_bounding_box /
#   bbox_reference_dimensions / bounding_box_state(normal 1083252 / clipped 16605)
#   这是I^3E任务的不精确空间提示，七字段格式同样无处安放。
#   丢掉它之后本数据集就退化成普通的instruction-based editing数据集，
#   这也正是下面必须丢弃"移除"这类通用指令的根本原因。
# 【better_data】质量标记(True 755304 / False 344553)。
#   按方案确认**不用于过滤、也不保存**，与005(x2edit)对二十多个质量分的口径一致。
# 【上游记录的宽高】source_image_shape / edited_image_shape / mask_image_shape:
#   前两个只用来判长宽比(决定resize还是丢弃)，
#   写进json的宽高一律以实际写盘数组的shape为准，不采信上游数值。
# 【定位与冗余字段】dataset_task_type(恒为image_edit)、metadata_name、
#   sample_id / source_id(与sample_key等价的int形态)、
#   source_archive / source_file / asset_archive / target_file / mask_file
#   (上游tar内的成员定位路径，落盘后已失效)、
#   reference_image_path_list / reference_image_num(恒为1，本脚本现算)。
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# 本版本每个编辑对只有编辑前原图这1张参考图(mask不使用)，
# 所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 上游edit_type -> 归一化后的任务名(即子集名)，最终产出4个子集。
# 这4个任务类型是数据集自带的、明确可知的，所以不需要mix兜底子集。
# 显式写死而不是每行现lower()的原因: 归一化规则一改就会静默把几十万个样本对
# 写进错误子集，写死之后任何上游取值变化都会在check_load_annotation_count里硬失败
GET_SET_NAME_DICT = {
    'Add': 'add',
    'Local': 'local',
    'Remove': 'remove',
    'Texture': 'texture',
}

# 找不到任务类型时才用的兜底子集名。
# 本数据集edit_type实测1099857行全非空、只有上面4种取值，一条都不会落进mix，
# 保留这条路径只是为了和002/005/006的口径保持一致，
# 并防止上游之后新增edit_type取值时被静默漏处理
# (新取值会被check_load_annotation_count的"unknown edit type"硬拦下来)
MIX_SET_NAME = 'mix'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 保存名形如inter_edit_add_0000000_edited.jpg(最长34字符)，
# 4个子集名全是ASCII小写、sample_key全是7位数字，
# 所以不会出现CJK字符或其它异常字符，也远低于文件系统单文件名255字节的上限
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 保存图像名里的原图名前缀(即sample_key)必须是7位数字，
# 上游016已硬校验过它全局唯一且覆盖0000000..1099963，这里做落盘前的最后一道拦截
VALID_SAMPLE_KEY_PATTERN = re.compile(r'^\d{7}$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃。
# 实测抽样200张(编辑后图100 + 参考图100)100%是RGB模式的PNG
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 上游标注文件数与总行数，解析阶段硬对账。
# 少一个分片/少一行都说明上游016没跑完或产物被改动过，
# 这时候继续往下跑只会得到一个悄悄少样本的新数据集
EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 275

EXPECTED_TOTAL_ANNOTATION_COUNT = 1099857

# 上游全量edit_type分布(合计1099857)，即"过滤之前"每个子集分到的条数，
# 解析阶段按edit_type逐项硬对账。
# 这能拦住"某个edit_type被映射进错误子集"这种总数级对账看不出来的问题
EXPECTED_EDIT_TYPE_COUNT_DICT = {
    'Add': 312261,
    'Local': 408012,
    'Remove': 308044,
    'Texture': 71540,
}

# 文本层各类不合格指令的实测精确条数，解析阶段逐项硬对账。
# 判定顺序严格按下面process_single_annotation_file里的顺序执行
# (empty -> null_like -> generic_no_target -> dirty -> no_word_char ->
#  too_short -> too_long -> different_aspect_ratio)，
# 换顺序会让这几个数字互相搬家，所以顺序不能改
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 0,
    'null_like_caption_count': 0,
    'generic_no_target_caption_count': 145315,
    'dirty_caption_count': 5596,
    'no_word_char_caption_count': 0,
    'too_short_caption_count': 1488,
    'too_long_caption_count': 0,
}

# 文本层7项过滤之后、长宽比过滤之前的样本对数:
# 1099857 - 145315 - 5596 - 1488 == 947458。
# 这个数只用于打印与写进resave_check_result.json，方便定位是哪一层砍掉的
EXPECTED_VALID_CAPTION_ANNOTATION_COUNT = 947458

# 被"参考图与编辑后图长宽比不同"这条规则丢弃的实测条数(在文本过滤之后统计)。
# 对应上面文件头列出的那4种尺寸组合，它们resize到编辑后图尺寸会把画面拉伸变形，
# 按方案确认整对丢弃
EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT = 318252

# 947458 - 318252 == 629206，这是最终落盘的样本对数。
# 全量实测(非抽样)的精确数字，解析阶段硬对账
EXPECTED_VALID_ANNOTATION_COUNT = 629206

# 4个子集过滤后的实测条数，合计629206，解析阶段逐个硬对账。
# remove子集掉得最多(308044 -> 105246): 先被通用指令"移除"/"remove"砍一刀，
# 再被长宽比规则砍一刀
EXPECTED_SET_ANNOTATION_COUNT_DICT = {
    'add': 199603,
    'local': 272916,
    'remove': 105246,
    'texture': 51441,
}

# 最终产出的子集数，必须与GET_SET_NAME_DICT严格一一对应(不多也不少)。
# 4个子集全部都在5万对以上，所以每个子集都会切出多个满10000对的文件夹，
# 实测合计65个文件夹(local 28 + add 20 + remove 11 + texture 6)
EXPECTED_SAVE_SET_COUNT = 4

PROCESS_NUM = 32

PER_FOLDER_EDIT_PAIR_NUM = 10000

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 指令长度阈值。
# 本数据集56%是中文指令，中文单字信息量远高于英文单词
# (中文p50只有7、min 2; 英文p50 28、min 3)，
# 沿用006(FoundIR)那种纯英文口径的10会砍掉43.6%的正常中文短指令，
# 属于砍正常样本而不是砍异常值，所以按方案确认取4(与005 x2edit的中文口径一致)。
# 实测丢掉通用指令与脏指令之后，长度小于4的还剩1488条
# (形如"移除锁"/"移除桶"/"关闭它"，是有明确指代对象的3字中文指令，
#  按方案确认一并砍掉，量级只占0.16%)。
# 上限200: 实测max=191，一条都不会被砍，只作兜底
MIN_CAPTION_LENGTH = 4

MAX_CAPTION_LENGTH = 200

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 实测0条命中，只作防御性拦截
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)，与005/015口径一致。
# 实测本数据集0条命中，只作防御性拦截
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

# 【本版本特有的过滤: 没有指代对象的通用指令，实测145315条(13.2%)整体丢弃】
# 这个数据集是I^3E(交互式指令编辑)任务: 指令可以很泛，因为"具体改哪个物体"
# 是由bounding_box和mask这两个空间提示指定的。全量精确统计:
#   移除 109849 / remove 20776 / remove it 14432 / remove the object 156 /
#   移除它 13 / erase 6 / 删除 3 / 去除 38 / 擦掉 27 / 擦除 15 / ...
#   合计145315条，其中Remove类占绝大多数。
# 本版本(017.0)不使用mask、也不保存bbox，所以这些指令落盘后就变成
# "给一张图 + 一句'移除'"——模型完全无法知道该移除哪个物体，是纯噪声监督，
# 按方案确认整体丢弃。
# 需要保留这批样本请用017.1(把mask当第2张参考图，指令改写成
# "参考给定的掩码 [V1*] ，移除"，mask精确圈出了要移除的物体，语义就完整了)。
#
# 这里只做**小写化 + 剥掉句末标点后的全串精确匹配**，不做任何启发式猜测:
# 实测"移除锁"/"移除桶"/"移除书"这类**有明确指代对象**的3字中文短指令共1568条，
# 一旦按"是不是太短"或"是不是只有一个动词"来判会误伤它们
# (它们会在后面的MIN_CAPTION_LENGTH那一步被砍，那是另一条独立规则)
CAPTION_GENERIC_NO_TARGET_STRIP_CHAR = '.。!！?？ '

CAPTION_GENERIC_NO_TARGET_LIST = [
    '移除',
    '移除它',
    '删除',
    '删除它',
    '去除',
    '去掉',
    '擦掉',
    '擦除',
    'remove',
    'remove it',
    'remove this',
    'remove that',
    'remove the object',
    'delete',
    'delete it',
    'erase',
    'erase it',
]

# 【本版本特有的过滤: LLM思维链残留的脏指令，实测5596条整体丢弃】
# 上游用LLM精简编辑指令时，把**中间过程文本**一起写进了instruction字段:
#   含'→'(原文 → 精简结果) 4961条:
#     '在椅子上添加一条毯子 → 添加一条毯子'
#   含'**'(markdown加粗)   2190条:
#     '在窗台上添加一杯热气腾腾的咖啡 → **添加一杯热气腾腾的咖啡**'
#   含换行                  594条:
#     '在椅子上添加一杯热咖啡 简化为：  \n\n**添加一杯热咖啡**'
#   含'简化为'/'精简后'等    409条:
#     '在已知原图和编辑区域的情况下，精简后的编辑指令为：\n\n"添加一条毯子"'
#   去重合计5596条。
# 按方案确认整体丢弃、**不做自动截取**: 这些串同时带'→'、'**'、换行和引号，
# 而且个别是'change the penguin into a polar bear → change it into a polar bear'
# 这种把具体指代反向改成it的精简，截取'→'后半段反而得到更差的指令。
# 另外'['和']'这两个字符也放进黑名单: 它们同时保证了原始指令里不会混入伪占位符，
# 这是017.1做[V1*]占位符改写的前提(两个脚本用同一套黑名单，口径完全一致)
CAPTION_DIRTY_CHAR_PATTERN = re.compile(r'[\*\[\]\u2192\n\r]')

CAPTION_DIRTY_WORD_PATTERN = re.compile(r'简化为|精简后|精简为|改写为|输出为')

# jpg重编码质量与色度采样方式，与006(FoundIR)完全一致。
# 001~005/007~008那几个脚本用的是cv2.imencode('.jpg', img)的默认值
# (质量95 + 色度4:2:0)，本数据集显式改成"质量97 + 色度4:4:4(不下采样)":
# 本数据集的**源图100%是无损PNG**(不像005 x2edit的源图本来就是jpg、
# 存在量化表幂等性)，用默认的4:2:0会把色度分辨率直接砍半，
# 是无损源图上重编码损失的主要来源。
# 006的全量实测表(510对/1020张测质量档 + 340对/680张测色度采样)已经证明:
# 真正的瓶颈是色度下采样而不是质量值(q100+4:2:0的色度保真55.47dB
# 还不如q95+4:4:4的55.52dB)，png源图上q95->q97的PSNR是45.43->46.47dB。
# 编辑后图与参考图共用同一套编码参数，避免两条编码链路引入
# "参考图多一层压缩"这种不对称的伪偏差
SAVE_IMAGE_JPEG_QUALITY = 97

SAVE_IMAGE_JPEG_SAMPLING_FACTOR = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444

SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST = [
    int(cv2.IMWRITE_JPEG_QUALITY),
    int(SAVE_IMAGE_JPEG_QUALITY),
    int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
    int(SAVE_IMAGE_JPEG_SAMPLING_FACTOR),
]

# 带编号的视觉参考图占位符，编号从"非原图的第1张参考图"起算:
# [V1*]指代reference_image[1]、[V2*]指代reference_image[2]...[VN*]指代
# reference_image[N]，其中N == reference_image_num - 1。
# 编辑前原图reference_image[0]永远隐式、不写进指令、不占编号。
# 这套写法与001/003/004/005/006/007完全一致，保证跨数据集口径统一。
# 本版本恒为1张参考图(N == 0)，即要求指令里完全没有占位符
# (实测上游1099857条里含[Vn*]形态的0条，而且带'['/']'的行已经在
#  脏指令黑名单里被判掉了，这里只做防御性拦截)
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本版本用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2

# 参考图resize到编辑后图尺寸时用的重采样方式。
# 按方案确认用PIL的LANCZOS而不是cv2.resize: 与项目既定口径保持一致。
# 只有长宽比严格相同的样本才会走到resize(等比缩小、零形变)，
# 长宽比不同的已经在解析阶段整对丢弃了
SAVE_IMAGE_RESIZE_RESAMPLING = Image.Resampling.LANCZOS

# 收尾自校验时是否**实读参考图**、把它的真实shape与json里的width/height硬对账。
# 按方案确认置True: "第一张参考图与生成图尺寸必须一致"是本次改动的核心诉求，
# 只对账json里的数字是查不出resize有没有真的生效的，必须真解一次图。
# 代价是收尾自校验要多解62万张参考图，在NAS上会明显变慢，
# 但相对前面几十小时的重编码可以接受
CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG = True


def get_set_name(per_edit_type):
    """按上游edit_type推导子集名(即图像编辑任务类型)

    本数据集4种edit_type的任务类型全部可知，所以4个子集就是4种图像编辑任务，
    一条都不会落进mix; 只有连edit_type都拿不到、彻底找不到任务类型时
    才归到mix子集(实测0条)。
    """
    per_edit_type = str(per_edit_type).strip()

    if per_edit_type in GET_SET_NAME_DICT:
        return GET_SET_NAME_DICT[per_edit_type]

    return MIX_SET_NAME


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    本版本不使用mask，每个编辑对只有编辑前原图这1张参考图，所有子集恒为1。
    保留这个函数是为了和002/005/006的收尾自校验保持同一套交叉对账写法。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def get_annotation_image_shape(per_image_shape):
    """把上游标注里的[宽, 高]取出来并校验，取不到或非法返回None

    上游source_image_shape / edited_image_shape实测1099857行全是2元int list，
    这里的类型校验只做防御性拦截。
    """
    if not isinstance(per_image_shape,
                      (list, tuple)) or len(per_image_shape) != 2:
        return None

    per_image_w, per_image_h = per_image_shape[0], per_image_shape[1]
    if not isinstance(per_image_w, int) or isinstance(per_image_w, bool):
        return None
    if not isinstance(per_image_h, int) or isinstance(per_image_h, bool):
        return None
    if per_image_w <= 0 or per_image_h <= 0:
        return None

    return [per_image_w, per_image_h]


def check_same_image_aspect_ratio(per_reference_image_shape,
                                  per_edited_image_shape):
    """判定参考图与编辑后图的长宽比是否严格相同，返回True表示相同(可以等比resize)

    用Fraction的最简分数比精确判定，不用浮点相除: 浮点比较要么因为精度误差把
    真正相同的判成不同，要么要引入一个拍脑袋的容差阈值。
    实测730363条相同(1328x1328->1024x1024、1056x1584->832x1248、
    1584x1056->1248x832这3种组合)、369494条不同。
    长宽比相同的才会被resize到编辑后图尺寸(等比缩小、零形变)，
    不同的整对丢弃(resize会把画面拉伸变形)。
    """
    if not per_reference_image_shape or not per_edited_image_shape:
        return False

    return Fraction(per_reference_image_shape[0],
                    per_reference_image_shape[1]) == Fraction(
                        per_edited_image_shape[0], per_edited_image_shape[1])


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令

    本版本不使用mask、不引入任何视觉条件图，指令里既没有占位符
    也没有"the reference image"这类自然语言指代(只有1张参考图、指令从不指代它)，
    所以这里只做strip，不做任何占位符改写。
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
    本版本N恒为0，即要求指令里完全没有占位符。
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
    实测本数据集0条命中，只作防御性拦截。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().strip(
        CAPTION_STRIP_CHAR)

    return per_ti2i_caption in NULL_LIKE_CAPTION_LIST


def check_generic_no_target_caption(per_ti2i_caption):
    """判定指令是不是"移除"/"remove"这类没有指代对象的通用指令

    只做小写化 + 剥掉句末标点后的**全串精确匹配**，不做任何启发式猜测，
    这样"移除"/"移除。"/"Remove."/"remove it"能被同一条规则一次判掉，
    而"移除锁"/"移除咖啡杯"这类有明确指代对象的短指令不会被误伤。
    本版本没有mask和bbox这两个空间提示，这类指令落盘后是纯噪声监督，整对丢弃。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().rstrip(
        CAPTION_GENERIC_NO_TARGET_STRIP_CHAR)

    return per_ti2i_caption in CAPTION_GENERIC_NO_TARGET_LIST


def check_dirty_caption(per_ti2i_caption):
    """判定指令里是不是混进了LLM精简指令时的思维链中间过程文本

    命中任意一条即判掉: 含 * [ ] → 换行 这几个字符，或含
    简化为/精简后/精简为/改写为/输出为 这几个词。
    实测5596条命中，全部整对丢弃、不做自动截取(理由见常量处注释)。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip()

    if CAPTION_DIRTY_CHAR_PATTERN.search(per_ti2i_caption):
        return True

    if CAPTION_DIRTY_WORD_PATTERN.search(per_ti2i_caption):
        return True

    return False


def process_single_annotation_file(annotation_file_pair):
    """解析单个标注文件，组装图像编辑对(参考图+编辑后图+编辑指令)的列表

    这一步只做纯文本层面 + 标注里已有宽高的过滤，判定顺序**严格固定**为:
      edit_type未知 -> 指令为空 -> null字面量 -> 通用无指代指令 -> 脏指令 ->
      无文字字符 -> 指令过短 -> 指令过长 -> **参考图与编辑后图长宽比不同** ->
      缺图 -> 保存名非法 -> 坏占位符指令
    换顺序会让EXPECTED_INVALID_CAPTION_COUNT_DICT与
    EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT里那几个数字互相搬家，所以顺序不能改。
    长宽比这条只读标注里的source_image_shape / edited_image_shape就能判、
    不用解图，所以能把369494条形变样本挡在几十小时的解码重编码之前。
    图像本身的解码校验和分辨率过滤留到后面多进程里做。

    组装出来的每个编辑对都会带上"参考图应该被resize到的目标尺寸"
    (即编辑后图的宽高)，落盘时参考图按它resize，保证落盘后两张图尺寸严格相等。

    图像是否存在这里用os.path.isfile逐个判，没有按目录缓存os.listdir:
    上游图像分散在275个target分片目录 + 245个source分片目录里，
    每个目录约4000/2500个文件，缓存收益有限; 而且每张图后面都要真解码一遍，
    真缺图在解码阶段一定会被判出来，这里的存在性判定只是为了把"缺图"和"图坏"
    分开统计。
    """

    per_annotation_path, root_image_path = annotation_file_pair

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
    unknown_edit_type_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    generic_no_target_caption_count, dirty_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_annotation_image_shape_count = 0
    different_aspect_ratio_count = 0
    missing_image_count = 0
    invalid_save_image_name_count = 0
    invalid_placeholder_caption_count = 0
    edit_type_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    for per_annotation in annotation_list:
        if not isinstance(per_annotation, dict):
            illegal_line_count += 1
            print('2222', per_annotation_path)
            continue

        per_edit_type = per_annotation.get(ANNOTATION_EDIT_TYPE_KEY_NAME, '')
        if not isinstance(per_edit_type, str):
            per_edit_type = ''
        per_edit_type = per_edit_type.strip()

        # 任务类型决定子集名，取值不在白名单里就无法安全落盘(实测0条命中)
        if per_edit_type not in GET_SET_NAME_DICT:
            unknown_edit_type_count += 1
            print('3333', per_annotation_path, per_edit_type)
            continue

        # 按上游edit_type统计"过滤之前"的分布，与实测ground truth硬对账
        edit_type_count_dict[per_edit_type] = edit_type_count_dict.get(
            per_edit_type, 0) + 1

        per_set_name = get_set_name(per_edit_type)
        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)

        per_ti2i_caption = per_annotation.get(ANNOTATION_CAPTION_KEY_NAME, '')
        if isinstance(per_ti2i_caption, (list, tuple)):
            per_ti2i_caption = per_ti2i_caption[0] if len(
                per_ti2i_caption) > 0 else ''
        if not isinstance(per_ti2i_caption, str):
            per_ti2i_caption = ''
        per_ti2i_caption = per_ti2i_caption.strip()

        # 空指令、全空格指令视为不合格图像编辑对(实测0条，只作防御)
        if not per_ti2i_caption:
            empty_caption_count += 1
            continue

        # null字面量与"不做任何修改"这类无意义指令同样丢弃(实测0条，只作防御)
        if check_null_like_caption(per_ti2i_caption):
            null_like_caption_count += 1
            print('3333', per_annotation_path, per_ti2i_caption[:50])
            continue

        # 没有指代对象的通用指令整体丢弃(实测145315条)。
        # 本版本没有mask和bbox这两个空间提示，这类指令是纯噪声监督
        if check_generic_no_target_caption(per_ti2i_caption):
            generic_no_target_caption_count += 1
            continue

        # LLM思维链残留的脏指令整体丢弃(实测5596条)
        if check_dirty_caption(per_ti2i_caption):
            dirty_caption_count += 1
            continue

        # 只剩标点、没有任何数字/字母/汉字的指令也丢弃(实测0条，只作防御)
        if not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption):
            no_word_char_caption_count += 1
            print('3333', per_annotation_path, per_ti2i_caption[:50])
            continue

        # 过短指令视为不合格图像编辑对(实测1488条)
        if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
            too_short_caption_count += 1
            continue

        # 过长指令同样视为不合格图像编辑对(实测0条，只作兜底)
        if len(per_ti2i_caption) > MAX_CAPTION_LENGTH:
            too_long_caption_count += 1
            print('3333', per_annotation_path, len(per_ti2i_caption))
            continue

        # 本版本的指令不需要任何占位符改写，这里只做strip，
        # 写进json的一定是归一化后的指令
        per_ti2i_caption = get_normalized_ti2i_caption(per_ti2i_caption)

        # 【参考图与编辑后图必须能对齐到同一尺寸】
        # 上游两张图的分辨率100%不相等(见文件头规格坑)，
        # 长宽比相同的等比resize到编辑后图尺寸(零形变、保留)，
        # 长宽比不同的resize会把画面拉伸变形，整对丢弃(实测318252条)。
        # 这一步只读标注里的宽高、不解图，所以能挡在解码重编码之前
        per_annotation_reference_image_shape = get_annotation_image_shape(
            per_annotation.get(ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME,
                               None))
        per_annotation_edited_image_shape = get_annotation_image_shape(
            per_annotation.get(ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME, None))

        # 两个宽高字段实测1099857行全非空全合法，取不到只可能是上游规格变了
        if per_annotation_reference_image_shape is None or per_annotation_edited_image_shape is None:
            invalid_annotation_image_shape_count += 1
            print(
                '3333', per_annotation_path,
                per_annotation.get(ANNOTATION_REFERENCE_IMAGE_SHAPE_KEY_NAME,
                                   None),
                per_annotation.get(ANNOTATION_EDITED_IMAGE_SHAPE_KEY_NAME,
                                   None))
            continue

        if not check_same_image_aspect_ratio(
                per_annotation_reference_image_shape,
                per_annotation_edited_image_shape):
            different_aspect_ratio_count += 1
            continue

        per_sample_key = per_annotation.get(ANNOTATION_SAMPLE_KEY_KEY_NAME, '')
        if not isinstance(per_sample_key, str):
            per_sample_key = ''
        per_sample_key = per_sample_key.strip()

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
        if not os.path.isfile(per_edited_image_path):
            missing_image_count += 1
            continue

        # 编辑后图的上游文件名必须是target_{sample_key}.png:
        # 保存图像名的前缀取自sample_key，一旦它和真正被读取的那张图对不上，
        # 落盘的图和标注就张冠李戴了，所以这里必须交叉比对
        per_edited_image_name = os.path.basename(
            per_edited_image_relative_path)
        per_edited_image_match_result = LOAD_EDITED_IMAGE_NAME_PATTERN.match(
            per_edited_image_name)
        if not VALID_SAMPLE_KEY_PATTERN.match(
                per_sample_key
        ) or not per_edited_image_match_result or per_edited_image_match_result.group(
                'index') != per_sample_key:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_sample_key)
            continue

        # 保存图像名前缀用{数据集名}_{子集名}_{原图名前缀}。
        # 原图名前缀就是7位样本主干sample_key，上游016已硬校验过它全局唯一
        # (覆盖0000000..1099963、无缺号无重叠)，所以629206个保存名100%唯一，
        # 不需要像005那样再往名字里塞分片编号
        per_save_image_name_prefix = (f'{DATASET_NAME}_{per_set_name}_'
                                      f'{per_sample_key}')
        per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
        # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的前缀
        # (即带_edited那一段)，和002/005/006的写法保持一致，
        # 收尾自校验也是按edited_image去掉.jpg来反推这个文件夹名的
        per_save_pair_folder_name = os.path.splitext(
            per_save_edited_image_name)[0]

        if not VALID_IMAGE_NAME_PATTERN.match(per_save_edited_image_name):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_save_edited_image_name)
            continue

        # 参考图顺序固定，第0张一定是编辑前的原图source(本版本也只有这一张)
        per_reference_image_relative_path_list = per_annotation.get(
            ANNOTATION_REFERENCE_IMAGE_KEY_NAME, [])
        if not isinstance(per_reference_image_relative_path_list,
                          (list, tuple)):
            per_reference_image_relative_path_list = []

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

            per_reference_image_path = os.path.join(
                root_image_path, per_reference_image_relative_path)
            if not os.path.isfile(per_reference_image_path):
                per_missing_reference_image_count += 1
                continue

            # 参考图的上游文件名必须是source_{7位}.png。
            # 注意这里的7位数字是source_id而不是sample_key(源图是全局去重后
            # 单独发布的、一张源图最多被3个样本对复用)，所以只校验名字格式、
            # 不和sample_key比对
            if not LOAD_REFERENCE_IMAGE_NAME_PATTERN.match(
                    os.path.basename(per_reference_image_relative_path)):
                per_invalid_save_reference_image_name_count += 1
                continue

            per_save_reference_image_name = f'{per_save_image_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}'
            if not VALID_IMAGE_NAME_PATTERN.match(
                    per_save_reference_image_name):
                per_invalid_save_reference_image_name_count += 1
                continue

            per_reference_image_path_list.append(per_reference_image_path)
            per_save_reference_image_name_list.append(
                per_save_reference_image_name)

        if per_invalid_save_reference_image_name_count > 0:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path,
                  per_save_reference_image_name_list)
            continue

        # 同一个样本对里两张参考图撞名会互相覆盖，整对丢弃
        # (本版本只有1张参考图，这里只做防御性拦截)
        if len(set(per_save_reference_image_name_list)) != len(
                per_save_reference_image_name_list):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path,
                  per_save_reference_image_name_list)
            continue

        # 参考图缺任意一张都会让这个编辑对的条件信息不完整，整对丢弃
        if per_missing_reference_image_count > 0 or len(
                per_reference_image_path_list
        ) != per_expect_reference_image_num:
            missing_image_count += 1
            continue

        # 占位符编号与参考图数量不自洽的指令也丢弃。
        # 本版本是单参考图，即要求指令里完全没有[Vn*]占位符(实测0条命中，
        # 带'['/']'的行已经在脏指令那一步被判掉了)
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
            per_reference_image_path_list,
            per_save_reference_image_name_list,
            per_ti2i_caption,
            per_expect_reference_image_num,
        ])

    return [
        edit_annotation_pair_list,
        edit_type_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        unknown_edit_type_count,
        empty_caption_count,
        null_like_caption_count,
        generic_no_target_caption_count,
        dirty_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_annotation_image_shape_count,
        different_aspect_ratio_count,
        missing_image_count,
        invalid_save_image_name_count,
        invalid_placeholder_caption_count,
    ]


def get_all_annotation_file_pair(root_dataset_path):
    """收集上游全部标注文件，返回标注文件任务列表

    上游标注是275个平铺的jsonl(每片最多4000个样本对)，
    这里按"标注文件"这一粒度出任务，正好能把多进程铺满。
    """
    root_image_path = os.path.join(root_dataset_path,
                                   *LOAD_IMAGE_DIR_NAME_LIST)

    load_annotation_dir_path = os.path.join(root_dataset_path,
                                            LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    for per_annotation_file_name in sorted(
            os.listdir(load_annotation_dir_path)):
        if not per_annotation_file_name.endswith(
                LOAD_ANNOTATION_FILE_NAME_SUFFIX):
            continue

        per_annotation_path = os.path.join(load_annotation_dir_path,
                                           per_annotation_file_name)
        if not os.path.isfile(per_annotation_path):
            continue

        annotation_file_pair_list.append([
            per_annotation_path,
            root_image_path,
        ])

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list


def get_all_edit_annotation_pair(root_dataset_path):
    """按标注文件粒度多进程组装全部图像编辑对的列表

    上游有275个标注文件、合计1099857行，逐行还要判2张图像文件是否存在，
    所以这里按标注文件开多进程解析，最后按保存的编辑后图像名统一排序。
    """
    annotation_file_pair_list = get_all_annotation_file_pair(root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list))

    total_annotation_count = 0
    illegal_line_count, unknown_edit_type_count = 0, 0
    empty_caption_count, null_like_caption_count = 0, 0
    generic_no_target_caption_count, dirty_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_annotation_image_shape_count, different_aspect_ratio_count = 0, 0
    missing_image_count = 0
    invalid_save_image_name_count = 0
    invalid_placeholder_caption_count = 0
    edit_type_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            for per_edit_type, per_edit_type_count in per_load_result[1].items(
            ):
                edit_type_count_dict[per_edit_type] = edit_type_count_dict.get(
                    per_edit_type, 0) + per_edit_type_count

            for per_set_name, per_set_count in per_load_result[2].items():
                set_annotation_count_dict[
                    per_set_name] = set_annotation_count_dict.get(
                        per_set_name, 0) + per_set_count

            total_annotation_count += per_load_result[3]
            illegal_line_count += per_load_result[4]
            unknown_edit_type_count += per_load_result[5]
            empty_caption_count += per_load_result[6]
            null_like_caption_count += per_load_result[7]
            generic_no_target_caption_count += per_load_result[8]
            dirty_caption_count += per_load_result[9]
            no_word_char_caption_count += per_load_result[10]
            too_short_caption_count += per_load_result[11]
            too_long_caption_count += per_load_result[12]
            invalid_annotation_image_shape_count += per_load_result[13]
            different_aspect_ratio_count += per_load_result[14]
            missing_image_count += per_load_result[15]
            invalid_save_image_name_count += per_load_result[16]
            invalid_placeholder_caption_count += per_load_result[17]

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        edit_type_count_dict,
        set_annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        unknown_edit_type_count,
        empty_caption_count,
        null_like_caption_count,
        generic_no_target_caption_count,
        dirty_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_annotation_image_shape_count,
        different_aspect_ratio_count,
        missing_image_count,
        invalid_save_image_name_count,
        invalid_placeholder_caption_count,
    ]


def check_load_annotation_count(
        total_annotation_file_count, total_annotation_count,
        edit_type_count_dict, set_annotation_count_dict,
        invalid_caption_count_dict, invalid_annotation_image_shape_count,
        different_aspect_ratio_count, edit_annotation_pair_list):
    """解析完标注后按edit_type与子集两级硬对账，并检查保存图像名是否唯一

    上游标注是016一次性跑出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账能额外拦住"某个edit_type被映射进错误子集"这种
    edit_type级对账看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对。
    """
    check_error_message_list = []

    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    # edit_type级对账: 统计的是"过滤之前"的分布，必须与上游全量实测逐项相等
    for per_edit_type in sorted(edit_type_count_dict.keys()):
        if per_edit_type not in EXPECTED_EDIT_TYPE_COUNT_DICT:
            check_error_message_list.append(
                f'unknown edit type {per_edit_type}')
            continue

        per_expect_edit_type_count = EXPECTED_EDIT_TYPE_COUNT_DICT[
            per_edit_type]
        if edit_type_count_dict[per_edit_type] != per_expect_edit_type_count:
            check_error_message_list.append(
                f'{per_edit_type} edit type count not match '
                f'{edit_type_count_dict[per_edit_type]} != '
                f'{per_expect_edit_type_count}')

    for per_edit_type in sorted(EXPECTED_EDIT_TYPE_COUNT_DICT.keys()):
        if per_edit_type not in edit_type_count_dict:
            check_error_message_list.append(
                f'missing edit type {per_edit_type}')

    # 文本层各类不合格指令的条数逐项硬对账。
    # 判定顺序一改这几个数字就会互相搬家，所以这一组对账同时也在守护判定顺序
    for per_count_name in sorted(EXPECTED_INVALID_CAPTION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_CAPTION_COUNT_DICT[per_count_name]
        if invalid_caption_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_caption_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    # 上游两个宽高字段实测全非空全合法，取不到一条就说明上游规格变了
    if invalid_annotation_image_shape_count > 0:
        check_error_message_list.append(
            f'invalid annotation image shape count '
            f'{invalid_annotation_image_shape_count} != 0')

    # 被长宽比规则丢弃的条数硬对账。
    # 这个数字守护的是"哪些样本被resize、哪些被丢弃"这条最关键的口径
    if different_aspect_ratio_count != EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT:
        check_error_message_list.append(
            f'different aspect ratio count not match '
            f'{different_aspect_ratio_count} != '
            f'{EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT}')

    # 文本过滤之后、长宽比过滤之前的条数也要自洽
    per_valid_caption_annotation_count = (
        EXPECTED_TOTAL_ANNOTATION_COUNT -
        sum(EXPECTED_INVALID_CAPTION_COUNT_DICT.values()))
    if per_valid_caption_annotation_count != EXPECTED_VALID_CAPTION_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'valid caption annotation count not self consistent '
            f'{per_valid_caption_annotation_count} != '
            f'{EXPECTED_VALID_CAPTION_ANNOTATION_COUNT}')

    if len(edit_annotation_pair_list) != EXPECTED_VALID_ANNOTATION_COUNT:
        check_error_message_list.append(f'valid annotation count not match '
                                        f'{len(edit_annotation_pair_list)} != '
                                        f'{EXPECTED_VALID_ANNOTATION_COUNT}')

    # 4个子集逐个硬对账
    for per_set_name in sorted(EXPECTED_SET_ANNOTATION_COUNT_DICT.keys()):
        per_expect_set_annotation_count = EXPECTED_SET_ANNOTATION_COUNT_DICT[
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
    # (mix子集一旦出现就说明上游新增了edit_type取值，必须显式感知)
    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in EXPECTED_SET_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

    # 保存的编辑后图像名必须全局唯一(sample_key全局唯一时天然满足)，
    # 撞名会让两个样本对在磁盘和json里互相覆盖
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
    """校验单个图像编辑对里的编辑后图像和所有参考图像，任意一张不合格则整对丢弃

    短边和宽高比的过滤按方案只以编辑后图像为准判定，参考图只要求能正常解码且是RGB
    (参考图会被resize到编辑后图的尺寸，所以它自己的原始短边与宽高比没有意义)。
    实测编辑后图只有7种分辨率(短边最小768、宽高比最大1.79)，
    这两个阈值一条都砍不到，只作兜底。
    """
    per_set_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_ti2i_caption, per_expect_reference_image_num = edit_annotation_pair

    if check_single_image(per_edited_image_path) is None:
        return None

    for per_reference_image_path in per_reference_image_path_list:
        if check_single_image(per_reference_image_path) is None:
            return None

    return [
        per_set_name,
        per_save_pair_folder_name,
        per_edited_image_path,
        per_save_edited_image_name,
        per_reference_image_path_list,
        per_save_reference_image_name_list,
        per_ti2i_caption,
        per_expect_reference_image_num,
    ]


def get_all_edit_pair_save_folder_pair(edit_annotation_pair_list,
                                       save_dataset_path):
    """把过滤后的合格图像编辑对按子集分组，排序后每10000对切成一个文件夹

    切分必须在过滤全部完成之后做，且切分前先按保存的编辑后图像名排序，这样才能保证
    每个文件夹都是满10000对(最后一个文件夹允许不满)。每个图像编辑对在文件夹里再独占
    一个子文件夹，该对的编辑后图像和所有参考图像都存在这个子文件夹里。
    本版本会产出4个子集，每个子集都在5万对以上，
    所以每个子集都会切出多个满10000对的文件夹，实测合计65个文件夹。
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
                _, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_ti2i_caption, per_expect_reference_image_num = per_edit_annotation_pair

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
                    per_ti2i_caption,
                    per_expect_reference_image_num,
                ])

            per_set_folder_count += 1

        set_folder_count_dict[per_set_name] = per_set_folder_count

    return edit_pair_save_folder_pair_list, set_folder_count_dict


def resize_single_image(per_image, per_save_image_shape):
    """把BGR的numpy图像resize到指定的[宽, 高]，返回resize后的BGR numpy图像

    按方案确认用PIL的LANCZOS而不是cv2.resize:
    先BGR->RGB转成PIL、resize、再转回numpy并RGB->BGR，
    中间的颜色通道转换不能省，否则落盘图像的R与B通道会互换。
    只有长宽比严格相同的样本才会走到这里(等比缩小、零形变)，
    长宽比不同的已经在解析阶段整对丢弃。
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

    save_image_shape为None(编辑后图走这条): 图像原分辨率多少保存时还是多少，
    不做任何缩放; 给了[宽, 高](参考图走这条): 先LANCZOS resize到这个尺寸再落盘，
    保证落盘后参考图与编辑后图尺寸严格相等。

    上游1258412张图(629206张编辑后图 + 629206张参考图，参考图有重复源图)
    全部是无损PNG，这里统一重编码成jpg。
    编码参数显式用SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST(质量97 + 色度4:4:4)，
    而不是cv2的默认值(质量95 + 色度4:2:0): 源图是无损PNG，
    默认的4:2:0会把色度分辨率直接砍半，是这类源图上重编码损失的主要来源。
    编辑后图和参考图共用同一套编码参数，避免两条编码链路引入不对称的伪偏差。
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

    # 只有参考图会带目标尺寸，且只在尺寸真的不一样时才resize
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

    编辑后图**原分辨率落盘、不做任何缩放**，先落盘并拿到它的真实宽高;
    参考图再按这个宽高resize后落盘，保证落盘后两张图尺寸严格相等。
    参考图落盘后返回的宽高会与编辑后图的宽高再对账一次，不等则整对丢弃。
    """
    per_set_name, per_folder_name, per_save_pair_folder_name, per_edited_image_path, per_save_edited_image_name, per_reference_image_path_list, per_save_reference_image_name_list, per_ti2i_caption, per_expect_reference_image_num = edit_pair_save_folder_pair

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

    for per_reference_image_path, per_save_reference_image_name in zip(
            per_reference_image_path_list, per_save_reference_image_name_list):
        save_reference_image_path = os.path.join(
            per_pair_folder_path, per_save_reference_image_name)
        # 参考图统一resize到编辑后图的尺寸
        per_save_reference_image_shape = resave_single_image(
            per_reference_image_path, save_reference_image_path,
            per_edited_image_shape)
        if per_save_reference_image_shape is None:
            return None

        # 落盘后的参考图尺寸必须与编辑后图严格相等，不等说明resize链路有问题
        if per_save_reference_image_shape != per_edited_image_shape:
            print('8888', save_reference_image_path,
                  per_save_reference_image_shape, per_edited_image_shape)
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
    上游26个字段里剩下的属性(mask系列3个字段与1099857张mask图、
    bounding_box系列5个字段、better_data、上游记录的三张图宽高、
    以及各种tar内成员定位路径)全部丢弃，
    理由见文件开头SAVE_ANNOTATION_KEY_NAME_LIST的注释。
    width/height取的是编辑后图实际写盘数组的shape，
    而参考图已经被resize到同一尺寸，所以这一组宽高同时也是参考图的宽高。
    ti2i_caption写的就是上游instruction字段strip后的原文(中英文都原样保留)，
    本版本是单参考图，指令里不含任何视觉参考图占位符。
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
        # 再和该子集应有的参考图数量(本版本恒为1)交叉对账，不一致只上报不改数值
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


def get_save_image_shape(per_save_image_path):
    """实读一张落盘图像的[宽, 高]，读不到返回None

    收尾自校验用它复检"参考图与编辑后图尺寸严格相等"。
    只用PIL读图像头拿size、不解像素，代价远低于cv2.imdecode。
    """
    try:
        with Image.open(per_save_image_path) as per_image:
            per_image_w, per_image_h = per_image.size
    except Exception as e:
        print('9999', per_save_image_path, e)
        return None

    return [per_image_w, per_image_h]


def check_save_dataset(save_dataset_path, set_folder_count_dict):
    """全部落盘后的收尾自校验: 文件夹容量、json与磁盘一一对应、指令与参考图数量

    每个子集除最后一个文件夹外都必须是满10000对，json里的每个key都必须在磁盘上有
    对应的样本对文件夹且文件恰好等于编辑后图像 + 所有参考图像，磁盘上也不允许有
    json没记录的残留样本对文件夹。另外还要复检ti2i_caption: 占位符编号集合必须与
    reference_image这个list的长度自洽、长度必须在阈值区间内、不能是null字面量、
    不能是通用无指代指令、不能是脏指令、不能只剩标点、记录的长度必须与字符串
    实际长度一致。
    最后在CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG为True时**实读编辑后图与参考图**，
    硬校验两者的真实shape都等于json里的width/height——
    "第一张参考图与生成图尺寸必须一致"是本次改动的核心诉求，
    只对账json里的数字查不出resize有没有真的生效，必须真解一次图。
    """

    check_error_message_list = []
    total_edit_pair_count = 0
    for per_set_name in sorted(set_folder_count_dict.keys()):
        per_set_dir_path = os.path.join(save_dataset_path, per_set_name)
        per_set_folder_count = set_folder_count_dict[per_set_name]
        per_expect_reference_image_num = get_expect_reference_image_num(
            per_set_name)
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
                        f'{per_save_edited_image_name} edited image name not a valid name'
                    )
                if not isinstance(per_annotation['reference_image'], list):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image not a list'
                    )
                    continue
                if per_annotation['reference_image_num'] != len(
                        per_annotation['reference_image']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match'
                    )
                # 本版本全部4个子集都必须是单参考图
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
                            f'{per_save_reference_image_name} reference image name not a valid name'
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
                # 长度自洽: 单参考图即要求指令里完全没有占位符
                if check_invalid_caption(
                        per_annotation['ti2i_caption'],
                        len(per_annotation['reference_image'])):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} caption placeholder index not match reference image num {len(per_annotation["reference_image"])}'
                    )
                # 落盘后的指令里不允许再残留null字面量
                if check_null_like_caption(per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a null like caption'
                    )
                # 也不允许残留没有指代对象的通用指令
                if check_generic_no_target_caption(
                        per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a generic no target caption'
                    )
                # 也不允许残留LLM思维链的中间过程文本
                if check_dirty_caption(per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a dirty caption')
                # 也不允许残留只剩标点、没有任何数字/字母/汉字的指令
                if not CAPTION_WORD_CHAR_PATTERN.search(
                        per_annotation['ti2i_caption']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a no word char caption'
                    )
                # json里存的就是归一化后的指令，长度过滤也是按归一化后判定的，
                # 两者口径一致，这里直接量json里的长度复检
                if len(per_annotation['ti2i_caption'].strip()
                       ) < MIN_CAPTION_LENGTH:
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} still a too short caption'
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
                    continue

                # 【核心复检】实读编辑后图与参考图，
                # 硬校验两者的真实shape都等于json里记录的width/height。
                # 只对账json里的数字是查不出resize有没有真的生效的
                if not CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG:
                    continue

                per_expect_save_image_shape = [
                    per_annotation['width'],
                    per_annotation['height'],
                ]

                for per_check_save_image_name in per_expect_file_name_list:
                    per_check_save_image_path = os.path.join(
                        per_pair_folder_path, per_check_save_image_name)
                    per_check_save_image_shape = get_save_image_shape(
                        per_check_save_image_path)
                    if per_check_save_image_shape is None:
                        check_error_message_list.append(
                            f'{per_check_save_image_name} load save image failed'
                        )
                        continue

                    if per_check_save_image_shape != per_expect_save_image_shape:
                        check_error_message_list.append(
                            f'{per_check_save_image_name} save image shape not match {per_check_save_image_shape} != {per_expect_save_image_shape}'
                        )

    print('3333', 'check total edit pair:', total_edit_pair_count,
          'check error:', len(check_error_message_list))

    return check_error_message_list, total_edit_pair_count


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path, SAVE_DATASET_DIR_NAME)
    os.makedirs(save_dataset_path, exist_ok=True)

    edit_annotation_pair_list, total_annotation_file_count, edit_type_count_dict, set_annotation_count_dict, total_annotation_count, illegal_line_count, unknown_edit_type_count, empty_caption_count, null_like_caption_count, generic_no_target_caption_count, dirty_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, invalid_annotation_image_shape_count, different_aspect_ratio_count, missing_image_count, invalid_save_image_name_count, invalid_placeholder_caption_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          illegal_line_count, unknown_edit_type_count, empty_caption_count,
          null_like_caption_count, generic_no_target_caption_count,
          dirty_caption_count, no_word_char_caption_count,
          too_short_caption_count, too_long_caption_count,
          invalid_annotation_image_shape_count, different_aspect_ratio_count,
          missing_image_count, invalid_save_image_name_count,
          invalid_placeholder_caption_count, len(edit_type_count_dict),
          len(set_annotation_count_dict), len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    invalid_caption_count_dict = {
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'generic_no_target_caption_count': generic_no_target_caption_count,
        'dirty_caption_count': dirty_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        total_annotation_file_count, total_annotation_count,
        edit_type_count_dict, set_annotation_count_dict,
        invalid_caption_count_dict, invalid_annotation_image_shape_count,
        different_aspect_ratio_count, edit_annotation_pair_list)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        raise Exception(
            f'check load annotation count error num {len(load_annotation_check_error_message_list)} {load_annotation_check_error_message_list[:20]}'
        )

    check_edit_annotation_pair_list = []
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap(process_single_edit_pair_check,
                                               edit_annotation_pair_list),
                                     total=len(edit_annotation_pair_list)):
            if per_check_result is None:
                continue
            check_edit_annotation_pair_list.append(per_check_result)

    invalid_image_count = len(edit_annotation_pair_list) - len(
        check_edit_annotation_pair_list)

    print('1111', len(check_edit_annotation_pair_list), invalid_image_count)

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
          'total annotation:', total_annotation_count, 'illegal line:',
          illegal_line_count, 'unknown edit type:', unknown_edit_type_count,
          'empty caption:', empty_caption_count, 'null like caption:',
          null_like_caption_count, 'generic no target caption:',
          generic_no_target_caption_count, 'dirty caption:',
          dirty_caption_count, 'no word char caption:',
          no_word_char_caption_count, 'too short caption:',
          too_short_caption_count, 'too long caption:', too_long_caption_count,
          'invalid annotation image shape:',
          invalid_annotation_image_shape_count, 'different aspect ratio:',
          different_aspect_ratio_count, 'missing image:', missing_image_count,
          'invalid save image name:', invalid_save_image_name_count,
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
        'total_annotation_file_count': total_annotation_file_count,
        'total_annotation_count': total_annotation_count,
        'illegal_line_count': illegal_line_count,
        'unknown_edit_type_count': unknown_edit_type_count,
        'empty_caption_count': empty_caption_count,
        'null_like_caption_count': null_like_caption_count,
        'generic_no_target_caption_count': generic_no_target_caption_count,
        'dirty_caption_count': dirty_caption_count,
        'no_word_char_caption_count': no_word_char_caption_count,
        'too_short_caption_count': too_short_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'valid_caption_annotation_count':
        EXPECTED_VALID_CAPTION_ANNOTATION_COUNT,
        'invalid_annotation_image_shape_count':
        invalid_annotation_image_shape_count,
        'different_aspect_ratio_count': different_aspect_ratio_count,
        'missing_image_count': missing_image_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
        'invalid_image_count': invalid_image_count,
        'save_edit_pair_failed_count': save_edit_pair_failed_count,
        'total_save_edit_pair_count': len(save_result_list),
        'total_save_reference_image_count': total_save_reference_image_count,
        'reference_image_num_mismatch_count':
        reference_image_num_mismatch_count,
        'total_save_set_count': len(set_folder_count_dict),
        'total_save_folder_count': len(folder_edit_pair_count_dict),
        'check_total_edit_pair_count': check_total_edit_pair_count,
        'check_error_count': len(check_error_message_list),
        'use_mask_as_reference_image_flag': False,
        'resize_reference_image_to_edited_image_shape_flag': True,
        'check_save_reference_image_shape_flag':
        CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG,
        'save_image_jpeg_quality': SAVE_IMAGE_JPEG_QUALITY,
        'save_image_jpeg_sampling_factor_444_flag': True,
        'edit_type_count_dict': edit_type_count_dict,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/Inter-Edit-Train'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
