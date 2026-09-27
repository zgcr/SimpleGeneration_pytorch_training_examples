import os
import re
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'fine_t2i'

SAVE_DATASET_DIR_NAME = 'fine-t2i'

# ==============================================================================
# 【数据集类型判定】Fine-T2I(ma-xu/fine-t2i)是纯文生图(text-to-image)数据集,
# **不能处理成图像编辑数据集**,所以本目录下这个数据集只有这一个t2i脚本、
# 没有对应的resave ti2i脚本。
#
# 判定依据(实测,不是照抄上游注释):
# 1) 上游012解出的jsonl每行固定23个字段(抽5个子集共24238行,key组合100%一致),
#    里面**没有任何reference_image / input_image / source_image / mask /
#    edit_instruction / edit_type字段**,每个样本严格是"1张目标图 + 1条prompt"
#    三件套(jpg/json/txt),上游对账报告dataset_task_type=text_to_image,
#    README的task_categories也只写text-to-image / image-to-text。
# 2) 唯一可能拼出编辑对的路线是"同一个uuid在A子集下的图 -> 在B子集下的图",
#    **这条路线实测被证伪**: 把全量6313671条sample_key导出做全局对账,
#    唯一key 3087649个,出现1次1210214 / 2次907479 / 3次591325 / 4次378631,
#    即同一条prompt(同一个uuid)最多在4个synthetic子集里各出一张图;
#    但抽查这些同key样本,**分辨率不同、生成模型不同(Z-Image-Turbo vs FLUX.2-dev)、
#    caption来源不同(prompt vs enhanced_prompt)**,例如d209d89e-…:
#    enhanced_random是Z-Image-Turbo/2048x2048/1020字符enhanced_prompt,
#    original_square是FLUX.2-dev/2048x2048/123字符prompt。
#    这是"同一条内容用不同提示/不同模型/不同分辨率**各自从零重画一遍**",
#    不是"编辑前图 -> 编辑后图"的同构关系; 而且数据集里根本不存在任何一条
#    编辑指令可以当ti2i_caption。硬当编辑对训练等于教模型
#    "按指令把画面整个重画一遍",是错误监督。
#
# 【上游012解出的目录规格(读上游对账报告 + 实测抽查,不是推测)】
# fine-t2i/
# ├── curated/train-{000000..000191}/<uuid或数字id>.jpg|.json|.txt
# ├── synthetic_enhanced_prompt_random_resolution/train-{000000..001620}/...
# ├── synthetic_enhanced_prompt_square_resolution/train-{000000..001543}/...
# ├── synthetic_original_prompt_random_resolution/train-{000000..001687}/...
# ├── synthetic_original_prompt_square_resolution/train-{000000..001310}/...
# ├── unzip_annotations/<子集名>/train-%06d.jsonl  6356个,合计6,313,671行(17G)
# └── unzip_check_missing_images.json  上游对账报告:
#         total_valid_sample_pair_count=6313671 / total_duplicate_member_count=0 /
#         missing_image_count=0 / orphan_image_count=0 /
#         missing_text_count=446 / invalid_annotation_count=446
#
# 本脚本只读这6356个jsonl定位样本对,**绝不os.walk图像目录**:
# 上游一共解出约1900万个小文件(630万jpg + 630万json + 630万txt、2.2T),
# 扫一遍目录树在NAS上不可接受。
#
# 【jsonl每行固定23个字段(实测抽样24238行,key组合100%一致、无缺字段)】
#   image_path / annotation_path / text_path / sample_key / subset_name /
#   archive_name / caption / caption_source_name / id / prompt /
#   enhanced_prompt / length / enhanced_length / prompt_generator / enhancer /
#   style / prompt_category / task / image_aspect_ratio / image_resolution /
#   image_generator / image_generated_with_enhanced_prompt /
#   aesthetic_predictor_v_2_5_score
#
# 【全量/抽样实测结论】
# - 6,313,671行(= 上游总样本对6,314,117 - 446条空txt样本,那446条上游**已经没有
#   写进jsonl**,所以本脚本天然不会碰到; 上游报告里的28条check_error也是这446条
#   分布在28个tar上导致的"valid pair * 3 != member数"派生提示,不影响样本完整性);
# - 逐子集行数 curated 167979 / enhanced_random 1615592 / enhanced_square 1538252 /
#   original_random 1686498 / original_square 1305350,与上游报告完全一致;
# - sample_key**在每个子集内部**: 4个synthetic子集100%唯一;
#   curated 167979行里唯一167683个,有296个key重复,且重复的两条是**完全不同的两张图**
#   (生成源不同pixabay vs pexels、分辨率不同、caption不同,例如key=1118895:
#    train-000020是pixabay 1280x853的羚羊图、train-000058是pexels 6000x4000的人物图),
#   按方案确认按"子集内图名前缀"去重,每个前缀只留排序后第一条(见去重函数注释);
# - sample_key形态有两种: uuid(8-4-4-4-12小写十六进制,4个synthetic子集与curated里
#   unsplash_lite来源的18343条) 和 纯数字(curated里pexels 117069 + pixabay 32567条),
#   实测无第三种形态、无大写字母;
# - image_path形如 <子集名>/<tar名>/<sample_key>.jpg,第一段恒等于jsonl所在子集目录名、
#   第二段恒等于jsonl文件名(tar名),文件名前缀恒等于sample_key;
# - 图像后缀全是.jpg,但**curated里有一部分文件的字节内容不是JPEG**:
#   全量扫curated 167979个文件头,能在前1KB定出SOF的34742个里
#   png(colortype2 真RGB) 1041 / png(colortype3 调色板) 21 /
#   灰度JPEG(1分量) 11 / CMYK JPEG(4分量) 6 / 其余为标准3分量JPEG
#   (剩下133236个是EXIF太长、SOF超出1KB,全量读取复核2000个全是3分量JPEG且EOI完好);
#   按比例外推curated里约有5000张真PNG、约50张灰度、约30张CMYK。
#   4个synthetic子集抽32000张100%是3分量baseline JPEG。
#   cv2/PIL都按内容而不是后缀解码,所以这些"挂着.jpg的PNG"能正常读,
#   本脚本按解码后的真实mode过滤(见VALID_IMAGE_MODE_LIST);
# - image_resolution与文件头解出的真实宽高抽样对账0例不符;
# - 短边实测最小512(4个synthetic子集恒为512起) / 576~640(curated),
#   宽高比实测最大2.44,即"短边<64"和"宽高比>8"这两条过滤全量一条都不会命中
#   (仍照写兜底,且过滤一律以真实解码shape为准,不信标注里的宽高)。
#
# 【caption口径: 按方案R逐子集选取"真正与图像一致"的那条prompt,
#   不直接沿用上游拼好的caption字段】
# 每条标注里有两条文本: prompt(短) 和 enhanced_prompt(长),实测两条永远不同
# (prompt == enhanced_prompt的样本0条)。它们的可信度在不同子集里**不一样**:
# - curated(16.8万张真实摄影图): prompt_generator与enhancer都是同一个**视觉captioner**
#   Qwen2.5-VL-7B-Captioner-Relaxed,两条都是**看着图**写出来的,所以enhanced_prompt是
#   图像grounded的更详细真描述(抽样核对: prompt"阴天海岸边两条破旧渔船…",
#   enhanced是同一张图的广角细描,前景木船剥漆/中景浑绿海湾/远景低矮丘陵/厚积灰云,
#   细节全部能在图里对上)。-> 取**enhanced_prompt**
# - synthetic_enhanced_*(约315万张): 图就是用enhanced_prompt跑出来的,
#   文本与图由构造保证一致。-> 取**enhanced_prompt**
# - synthetic_original_*(约299万张): 图是用短prompt跑出来的,而enhanced_prompt是
#   promptenhancer-7b**纯文本臆想**出来的(它没看过图,图也不是照它生成的),
#   里面的材质/光线/机位/背景细节大概率图里根本没有,拿它当训练文本就是在教模型
#   "文本说的东西可以不出现在图里",直接损害prompt-following。-> 取**prompt**
# 脚本里不写死子集名(上游改规格时会静默出错),而是按标注属性判定:
#   image_generated_with_enhanced_prompt为真 -> enhanced_prompt;
#   image_generator是真实图来源(unsplash_lite/pexels/pixabay,此时两条都是VLM看图写的)
#     -> enhanced_prompt;
#   其余(合成图且用短prompt生成) -> prompt。
# 选完后再与上游的caption_source_name / image_generated_with_enhanced_prompt
# 交叉核对一次,不自洽只计数上报、**绝不丢样本**(txt里那条本身就是可训练文本)。
#
# 该口径与"直接用上游caption字段"的唯一差别是curated从短caption升级成长caption
# (167979条,strip后长度p50 253 -> 1065),这正好贴合训练目标: 推理时用户的原始提示
# 会被别的LLM改写扩写,训练文本越接近那种长而详细的改写提示越好。
#
# 【上游标注里另外这些属性,按方案确认全部丢弃,新标注只留
#   width/height/t2i_caption/t2i_caption_length。落盘后**永久丢失**,
#   无法从新数据集恢复(上游产物会保留,可回头重跑)】
# prompt / enhanced_prompt 中**未被采用**的那一条 : 丢弃后无法再做
#                "短提示 <-> 长提示"配对训练、也无法做提示长度dropout
# caption / caption_source_name : 上游按txt还原的训练文本与其来源,
#                本脚本按方案R自己重选,这两个字段只用于交叉核对,不落盘
# length / enhanced_length      : 两条prompt的词数(注意是词数不是字符数)
# prompt_generator / enhancer   : 生成/增强prompt的模型名(Qwen2.5-VL-7B-Captioner-
#                Relaxed / Llama-3.1-8B-Instruct / promptenhancer-7b),
#                丢弃后无法再按文本来源溯源或加权
# style          : 11种视觉风格(Anime/General & Photorealistic/3D & CGI等,
#                curated恒为null)。丢弃后无法再按风格做条件训练或均衡采样
# prompt_category: 32类prompt类别(People: Portrait / Text rendering: Long text等,
#                curated恒为null)。丢弃后无法再单独加权"文字渲染"这类难样本
# task           : 任务组合list(Colors/Counting/Position/Reasoning,可为null)。
#                丢弃后无法再做组合泛化能力的定向采样
# image_aspect_ratio : 宽高比字符串(1:1 / 9:16 …,curated恒为null),
#                由真实解码后的width/height承载
# image_resolution   : [w, h],由真实解码后的width/height承载
# image_generator    : Z-Image-Turbo(458万) / FLUX.2-dev(156万) / pexels(11.7万) /
#                pixabay(3.3万) / unsplash_lite(1.8万)。丢弃后无法再区分
#                "真实照片 vs 合成图",也无法按生成模型加权
# image_generated_with_enhanced_prompt : bool,只用于选caption和交叉核对,不落盘
# aesthetic_predictor_v_2_5_score : 美学分。按方案确认**不做美学过滤、也不落盘**,
#                所以之后无法再按美学分筛样本(想筛只能回头重跑)
# id / sample_key : 以保存图像名的后半段形式保留(形如
#                fine_t2i_curated_000_<uuid或数字id>.jpg)
# subset_name    : 以子集目录名的前半段形式保留(形如curated_000 /
#                synthetic_enhanced_prompt_random_resolution_001),
#                所以"curated / enhanced / original / random / square"这个维度**不会丢**
# archive_name   : 上游tar归属。丢弃后不能再"只训某几个tar"
# image_path / annotation_path / text_path : 上游定位用
# ==============================================================================

# 上游012解压脚本按tar另存的汇总标注目录
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_SUFFIX = '.jsonl'

# 上游解出的5个子集目录(全部是train,没有val/test),unzip_annotations下也是这5个同名目录。
# 这5个目录既是上游的子集名,也是本脚本的系列名(切分后再变成带编号的子集目录名)
LOAD_SERIES_DIR_NAME_LIST = [
    'curated',
    'synthetic_enhanced_prompt_random_resolution',
    'synthetic_enhanced_prompt_square_resolution',
    'synthetic_original_prompt_random_resolution',
    'synthetic_original_prompt_square_resolution',
]

# 实测上游jsonl分片数(与上游012的tar数一一对应,合计6356个),数量不对说明上游没跑完
EXPECTED_SERIES_ANNOTATION_FILE_NUM_DICT = {
    'curated': 192,
    'synthetic_enhanced_prompt_random_resolution': 1621,
    'synthetic_enhanced_prompt_square_resolution': 1544,
    'synthetic_original_prompt_random_resolution': 1688,
    'synthetic_original_prompt_square_resolution': 1311,
}

# 实测上游每个子集的jsonl行数(与上游对账报告的subset_sample_pair_count_dict完全一致),
# 直接当完整性ground truth: 缺分片/少行都能拦住。
# 注意合计6,313,671 = 上游总样本对6,314,117 - 446条空txt样本(上游未写进jsonl)
EXPECTED_SERIES_ROW_COUNT_DICT = {
    'curated': 167979,
    'synthetic_enhanced_prompt_random_resolution': 1615592,
    'synthetic_enhanced_prompt_square_resolution': 1538252,
    'synthetic_original_prompt_random_resolution': 1686498,
    'synthetic_original_prompt_square_resolution': 1305350,
}

# jsonl里的image_path是相对上游数据集根目录的路径,形如
# curated/train-000000/002a702a-2638-433e-9553-3fe2b2585855.jpg
ANNOTATION_IMAGE_KEY_NAME = 'image_path'

# sample_key等于落盘图像的文件名前缀,是子集内的唯一样本id,
# 也直接当保存图像名的后半段
ANNOTATION_IMAGE_NAME_KEY_NAME = 'sample_key'

# jsonl里另存的子集名,必须和jsonl所在的子集目录名一致,只用于校验成员是否错位,不进新标注
ANNOTATION_SUBSET_NAME_KEY_NAME = 'subset_name'

# 上游按txt还原的训练文本与其来源,只用于和本脚本按方案R选出来的那条交叉核对
ANNOTATION_CAPTION_KEY_NAME = 'caption'

ANNOTATION_CAPTION_SOURCE_KEY_NAME = 'caption_source_name'

# 两条候选文本: 短prompt与长enhanced_prompt(实测两条永远不同)
ANNOTATION_PROMPT_KEY_NAME = 'prompt'

ANNOTATION_ENHANCED_PROMPT_KEY_NAME = 'enhanced_prompt'

# bool,指示这张图当初是用enhanced_prompt还是prompt生成的
ANNOTATION_ENHANCED_PROMPT_FLAG_KEY_NAME = 'image_generated_with_enhanced_prompt'

ANNOTATION_IMAGE_GENERATOR_KEY_NAME = 'image_generator'

# 上游标注里记录的原图宽高,形如[6000, 4000]。本脚本不拿它做分辨率过滤
# (万一它和实际图像不一致就会出偏差),只在解码后顺手比对一次,
# 不一致的条数只上报不丢样本(实测抽样对账0例不符)
ANNOTATION_IMAGE_RESOLUTION_KEY_NAME = 'image_resolution'

# 每条标注里必须齐备的属性(实测抽样24238行全部齐备)。
# 缺key说明上游规格变了,这种样本没法可靠地选caption,直接丢弃并计数,
# 扫描阶段结束后立刻抛异常,不等跑完几百万张图才发现
ANNOTATION_EXPECTED_KEY_NAME_LIST = [
    ANNOTATION_IMAGE_KEY_NAME,
    ANNOTATION_IMAGE_NAME_KEY_NAME,
    ANNOTATION_SUBSET_NAME_KEY_NAME,
    ANNOTATION_CAPTION_KEY_NAME,
    ANNOTATION_CAPTION_SOURCE_KEY_NAME,
    ANNOTATION_PROMPT_KEY_NAME,
    ANNOTATION_ENHANCED_PROMPT_KEY_NAME,
    ANNOTATION_ENHANCED_PROMPT_FLAG_KEY_NAME,
    ANNOTATION_IMAGE_GENERATOR_KEY_NAME,
    ANNOTATION_IMAGE_RESOLUTION_KEY_NAME,
]

# 真实图片来源(不是生成模型)。这三类样本的prompt与enhanced_prompt都是
# 视觉captioner(Qwen2.5-VL-7B-Captioner-Relaxed)**看着图**写出来的,
# 所以更长的enhanced_prompt是图像grounded的更详细真描述,应当取enhanced_prompt。
# 实测curated恒为这三种之一(pexels 117069 / pixabay 32567 / unsplash_lite 18343),
# 4个synthetic子集恒为Z-Image-Turbo或FLUX.2-dev
REAL_IMAGE_GENERATOR_NAME_LIST = [
    'unsplash_lite',
    'pexels',
    'pixabay',
]

SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 原始图像名前缀实测只有两种形态: uuid(8-4-4-4-12小写十六进制) 和 纯数字id。
# 不符合的样本没法保证保存图像名在子集内唯一(可能去覆盖别的样本),直接丢弃并上报
# 注意纯数字这条必须写[0-9]而不是\d: \d在Python3默认匹配Unicode数字
# (阿拉伯-印度数字等),那种字符进保存图像名会被收尾自校验的
# VALID_IMAGE_NAME_PATTERN判成非法字符,不如在这里就拦住
VALID_IMAGE_NAME_PREFIX_PATTERN_LIST = [
    re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    re.compile(r'^[0-9]+$'),
]

# image_path固定是 <子集名>/<tar名>/<sample_key>.jpg 三段,段数不对说明上游规格变了
EXPECTED_IMAGE_RELATIVE_PATH_NAME_NUM = 3

# 按方案确认与002 GPIC口径一致: 保留RGB与灰度图(cv2.IMREAD_COLOR会把灰度图
# 复制成3通道),把调色板P图/RGBA图/CMYK图等过滤掉。
# 实测4个synthetic子集100%是RGB; curated里按比例外推约有50张灰度(保留)、
# 约100张调色板PNG与约30张CMYK JPEG(过滤掉)
VALID_IMAGE_MODE_LIST = [
    'RGB',
    'L',
]

# 本机128核,这里和001/002保持一致取32。本数据集要对约631万张图各解码两遍
# (jsonl扫描阶段校验一次、写盘阶段重编码一次),想跑快可以直接调大这个常量
PROCESS_NUM = 32

PER_FOLDER_IMAGE_NUM = 10000

# 每个子集目录放100个文件夹(即100万张图)。全塞进一个目录下NAS元数据压力太大,
# 所以每100个文件夹再归入一个子集目录。按方案确认取100,预计切出
# curated_000(17个文件夹) + enhanced_random_000/_001 + enhanced_square_000/_001 +
# original_random_000/_001 + original_square_000/_001 共9个子集目录、约632个文件夹
PER_SET_FOLDER_NUM = 100

# 5个系列全部按PER_SET_FOLDER_NUM切成带编号的子集目录(curated只会出一个,
# 但仍按编号命名,保证与002/008/009/010/013的子集目录命名规格一致)
SPLIT_SET_DIR_SERIES_NAME_LIST = [
    'curated',
    'synthetic_enhanced_prompt_random_resolution',
    'synthetic_enhanced_prompt_square_resolution',
    'synthetic_original_prompt_random_resolution',
    'synthetic_original_prompt_square_resolution',
]

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 实测(curated全量167979条 + 4个synthetic子集各抽5万条)按方案R选出来的那条文本
# strip后长度: curated min 56 / p50 1065 / max 3135,
# enhanced_random min 286 / p50 1020 / max 2032,
# enhanced_square min 299 / p50 1018 / max 2109,
# original_random min 14 / p50 142 / max 933,
# original_square min 17 / p50 140 / max 941。
# 没有一条小于10,所以这条阈值只是兜底
MIN_CAPTION_LENGTH = 10

# 按方案确认取1536。注意**不能沿用002的1024**: 那会砍掉两个enhanced子集里约48%的
# 样本(约150万条),那是在砍一整个正常类别而不是砍异常值。
# 取1536的实测代价: curated约1.80%(3024/167979)、enhanced_random约0.42%、
# enhanced_square约0.40%、original_*(用短prompt)0条,
# 合计约1.6万条/631万条 ≈ 0.26%
MAX_CAPTION_LENGTH = 1536


def get_set_name(per_series_name, per_set_index):
    """子集目录名: 每个系列按每100个文件夹切成<系列名>_000/<系列名>_001/...

    子集目录名同时是保存图像名的中间段,所以它必须在切分完成后才能确定,
    这也是后面排序键只能用图名前缀而不能用保存图像名的原因。
    """
    if per_series_name not in SPLIT_SET_DIR_SERIES_NAME_LIST:
        return per_series_name

    return f'{per_series_name}_{per_set_index:03d}'


def check_image_name_prefix(per_image_name_prefix):
    """原始图像名前缀实测只有uuid和纯数字id两种形态,不符合的样本直接丢弃"""
    for per_image_name_prefix_pattern in VALID_IMAGE_NAME_PREFIX_PATTERN_LIST:
        if per_image_name_prefix_pattern.match(per_image_name_prefix):
            return True

    return False


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上,逐样本打一次os.path.exists就是一次网络往返,
    631万条标注就要打631万次,这一步本身就能占掉整个扫描阶段的大头。
    实测同一个jsonl里的图像全部落在同一个tar目录下(curated/train-000000.jsonl的
    image_path都是curated/train-000000/xxx.jpg),所以这里按目录缓存一次
    os.listdir的结果,之后只做集合查表,网络往返次数从"标注条数"降到"tar目录数"
    (6356次)。listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判,
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


def get_single_annotation_prompt(per_annotation, per_prompt_key_name):
    """取出标注里的一条prompt并strip,上游固定是str,这里兼容list和str两种形式"""
    per_prompt = per_annotation.get(per_prompt_key_name, '')
    if isinstance(per_prompt, (list, tuple)):
        per_prompt = per_prompt[0] if len(per_prompt) > 0 else ''
    if not isinstance(per_prompt, str):
        per_prompt = ''

    return per_prompt.strip()


def get_single_t2i_caption(per_annotation):
    """按方案R选出"真正与图像一致"的那条prompt,返回[文本, 选中来源, 上游期望来源]

    - image_generated_with_enhanced_prompt为真: 图就是用enhanced_prompt生成的,
      文本与图由构造保证一致 -> 取enhanced_prompt;
    - image_generator是真实图来源(unsplash_lite/pexels/pixabay): 图是真实照片,
      prompt与enhanced_prompt都是同一个视觉captioner**看着图**写的,
      更长的那条是图像grounded的更详细真描述 -> 取enhanced_prompt;
    - 其余(合成图且用短prompt生成): enhanced_prompt是纯文本增强模型
      **没看过图**臆想出来的,里面的细节图里大概率没有,拿它训练会损害
      prompt-following -> 取prompt。
    这里刻意不写死子集名: 上游一旦改子集划分,按属性判定仍然成立,
    而写死子集名会静默取错文本。
    入参的image_generated_with_enhanced_prompt必须已经被调用方校验成bool
    (非bool时整条样本在调用方就被丢掉了),所以这里可以直接当真假用。
    """
    per_enhanced_prompt_flag = per_annotation.get(
        ANNOTATION_ENHANCED_PROMPT_FLAG_KEY_NAME, None)

    per_image_generator_name = per_annotation.get(
        ANNOTATION_IMAGE_GENERATOR_KEY_NAME, '')
    if not isinstance(per_image_generator_name, str):
        per_image_generator_name = ''
    per_image_generator_name = per_image_generator_name.strip()

    if per_enhanced_prompt_flag:
        per_caption_source_name = ANNOTATION_ENHANCED_PROMPT_KEY_NAME
    elif per_image_generator_name in REAL_IMAGE_GENERATOR_NAME_LIST:
        per_caption_source_name = ANNOTATION_ENHANCED_PROMPT_KEY_NAME
    else:
        per_caption_source_name = ANNOTATION_PROMPT_KEY_NAME

    per_t2i_caption = get_single_annotation_prompt(per_annotation,
                                                   per_caption_source_name)

    # 上游的caption字段取自txt,txt里存的是当初真正用于生成该图的那条prompt,
    # 所以上游来源必然由image_generated_with_enhanced_prompt决定。
    # 真实图(curated)那批上游来源是prompt,而本脚本按方案R主动改取enhanced_prompt,
    # 这是**有意的口径差异**,所以核对的是"上游自身是否自洽",不是"和本脚本是否一致"
    per_expect_caption_source_name = ANNOTATION_ENHANCED_PROMPT_KEY_NAME if per_enhanced_prompt_flag else ANNOTATION_PROMPT_KEY_NAME

    return [
        per_t2i_caption,
        per_caption_source_name,
        per_expect_caption_source_name,
    ]


def process_single_image_check(per_image_path):
    """校验单张图像能否正常解码,并过滤非法mode图和极端分辨率图

    返回的宽高只用于和上游标注比对,最终写进json的宽高一定取自实际写盘图像的shape。
    """
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把RGBA图静默丢掉alpha通道、
    # 把CMYK图与调色板P图静默转成3通道,所以必须先用PIL读原始mode才能把这些图判出来。
    # 注意curated里有一部分文件挂着.jpg后缀但内容是PNG,PIL按内容解码,不受后缀影响
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

    # 检查图像宽高比,取长短边之比,宽高比大于8和小于1/8这两种极端样本一起判掉
    per_image_aspect_ratio = max(per_image_w / per_image_h,
                                 per_image_h / per_image_w)
    if per_image_aspect_ratio > MAX_IMAGE_ASPECT_RATIO:
        print('7777', per_image_path, per_image_w, per_image_h)
        return None

    return [
        per_image_w,
        per_image_h,
    ]


def process_single_annotation_file(annotation_file_pair):
    """解析单个上游jsonl标注,组装图像路径和t2i描述的样本对列表

    这个worker把文本层过滤(缺key、缺图、图像名非法、成员错位、描述为空或过短、
    描述过长)和图像层过滤(能否解码、mode是否合法、短边、宽高比)一次做完。
    图像校验没有像001那样单独再开一个Pool,是因为本数据集有约631万条标注:
    分两个Pool的话主进程要先攒631万条记录、再逐条发给check worker、再收回,
    光进程间序列化就要来回搬几十GB,而合并进来之后IPC只传存活样本。
    判定逻辑、过滤口径、日志编号和002完全一致,图像也一样是解码两遍
    (这里校验一遍、写盘时重编码再解一遍),没有为了省时间跳过任何一道校验。
    """
    per_jsonl_path, root_dataset_path, per_series_name = annotation_file_pair

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_annotation_key_count, subset_name_not_match_count = 0, 0
    invalid_enhanced_prompt_flag_count = 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    caption_source_not_match_count = 0
    image_annotation_pair_list = []

    # 每个worker只处理一个jsonl,缓存里通常只有一个tar目录,内存开销可忽略
    dir_file_name_cache_dict = {}

    try:
        load_jsonl_file = open(per_jsonl_path, 'r', encoding='UTF-8')
    except Exception as e:
        print('2222', per_jsonl_path, e)

        return [
            image_annotation_pair_list,
            per_series_name,
            total_annotation_count,
            1,
            missing_annotation_key_count,
            subset_name_not_match_count,
            invalid_enhanced_prompt_flag_count,
            missing_image_count,
            invalid_image_name_count,
            invalid_caption_count,
            too_long_caption_count,
            invalid_image_count,
            annotation_image_size_not_match_count,
            caption_source_not_match_count,
        ]

    with load_jsonl_file:
        for per_line in load_jsonl_file:
            per_line = per_line.strip()
            if not per_line:
                continue

            total_annotation_count += 1

            try:
                per_annotation = json.loads(per_line)
            except Exception as e:
                load_annotation_failed_count += 1
                print('2222', per_jsonl_path, e)
                continue

            if not isinstance(per_annotation, dict):
                load_annotation_failed_count += 1
                print('2222', per_jsonl_path, 'annotation not a dict')
                continue

            # 缺key说明上游规格变了,这种样本没法可靠地选caption,直接丢弃
            per_miss_key_flag = False
            for per_key_name in ANNOTATION_EXPECTED_KEY_NAME_LIST:
                if per_key_name not in per_annotation:
                    per_miss_key_flag = True
                    print('2222', per_jsonl_path, 'miss key', per_key_name)
                    break

            if per_miss_key_flag:
                missing_annotation_key_count += 1
                continue

            # image_generated_with_enhanced_prompt是方案R选caption的第一判据,
            # 必须是真bool才可信: 若上游把它写成null/字符串,
            # 直接当真假用会把整批样本静默配上错误文本(比如把纯文本臆想出来的
            # enhanced_prompt配给用短prompt生成的图),所以非bool直接丢样本并计数,
            # 扫描阶段结束后立刻抛异常。实测抽样24238行100%是bool
            per_enhanced_prompt_flag = per_annotation.get(
                ANNOTATION_ENHANCED_PROMPT_FLAG_KEY_NAME, None)
            if not isinstance(per_enhanced_prompt_flag, bool):
                invalid_enhanced_prompt_flag_count += 1
                print('2222', per_jsonl_path,
                      'invalid image_generated_with_enhanced_prompt',
                      per_enhanced_prompt_flag)
                continue

            per_image_relative_path = per_annotation.get(
                ANNOTATION_IMAGE_KEY_NAME, '')

            if not isinstance(per_image_relative_path, str):
                per_image_relative_path = ''
            if not per_image_relative_path:
                missing_image_count += 1
                continue

            per_image_relative_path = per_image_relative_path.replace(
                '\\', '/').lstrip('/')

            # image_path固定是 <子集名>/<tar名>/<sample_key>.jpg 三段,
            # 且第一段必须等于jsonl所在的子集目录名、标注里的subset_name也必须一致,
            # 任一不符说明上游成员错位,这种样本的归属不可信,直接丢弃并上报
            per_image_relative_path_name_list = per_image_relative_path.split(
                '/')
            per_annotation_subset_name = per_annotation.get(
                ANNOTATION_SUBSET_NAME_KEY_NAME, '')
            if not isinstance(per_annotation_subset_name, str):
                per_annotation_subset_name = ''
            per_annotation_subset_name = per_annotation_subset_name.strip()

            if len(
                    per_image_relative_path_name_list
            ) != EXPECTED_IMAGE_RELATIVE_PATH_NAME_NUM or per_image_relative_path_name_list[
                    0] != per_series_name or per_annotation_subset_name != per_series_name:
                subset_name_not_match_count += 1
                print('2222', per_jsonl_path, per_image_relative_path,
                      per_annotation_subset_name)
                continue

            per_image_path = os.path.join(root_dataset_path,
                                          per_image_relative_path)
            if not check_image_file_exists(per_image_path,
                                           dir_file_name_cache_dict):
                missing_image_count += 1
                continue

            # 落盘图像的文件名前缀就是sample_key,不一致说明上游成员错位,
            # 这种样本没法保证保存图像名唯一,直接丢弃
            per_image_name_prefix = os.path.splitext(
                os.path.basename(per_image_relative_path))[0].lower()
            per_sample_key = per_annotation.get(ANNOTATION_IMAGE_NAME_KEY_NAME,
                                                '')
            if not isinstance(per_sample_key, str):
                per_sample_key = ''
            per_sample_key = per_sample_key.strip().lower()

            if not per_sample_key or per_sample_key != per_image_name_prefix:
                invalid_image_name_count += 1
                print('2222', per_image_path, per_sample_key)
                continue

            if not check_image_name_prefix(per_image_name_prefix):
                invalid_image_name_count += 1
                print('2222', per_image_path, per_image_name_prefix)
                continue

            per_t2i_caption, per_caption_source_name, per_expect_caption_source_name = get_single_t2i_caption(
                per_annotation)

            # 上游自身是否自洽的交叉核对(只计数上报、**绝不丢样本**,
            # 因为本脚本用的是自己按方案R选出来的那条文本):
            # 1) 上游记录的caption_source_name必须与
            #    image_generated_with_enhanced_prompt推出来的来源一致;
            # 2) 上游的caption(取自txt,即当初真正用于生成该图的那条文本)必须与
            #    上游声明来源的那条prompt逐字符一致。
            # 只核对来源名不核对内容是不够的: 上游成员错位时来源名照样对得上,
            # 但txt里的文本已经是别的样本的了
            per_annotation_caption_source_name = per_annotation.get(
                ANNOTATION_CAPTION_SOURCE_KEY_NAME, '')
            if not isinstance(per_annotation_caption_source_name, str):
                per_annotation_caption_source_name = ''
            per_annotation_caption_source_name = per_annotation_caption_source_name.strip(
            )

            per_annotation_caption = get_single_annotation_prompt(
                per_annotation, ANNOTATION_CAPTION_KEY_NAME)
            per_expect_caption = get_single_annotation_prompt(
                per_annotation, per_expect_caption_source_name)

            if per_annotation_caption_source_name != per_expect_caption_source_name or per_annotation_caption != per_expect_caption:
                caption_source_not_match_count += 1
                print('2222', per_image_path,
                      per_annotation_caption_source_name,
                      per_expect_caption_source_name,
                      len(per_annotation_caption), len(per_expect_caption))

            # 空描述、全空格描述、过短描述都视为不合格样本对
            # (实测按方案R选出来的文本没有一条strip后长度小于10)
            if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                invalid_caption_count += 1
                print('3333', per_image_path, per_caption_source_name,
                      len(per_t2i_caption))
                continue

            # 过长描述同样视为不合格样本对(实测合计约0.26%超1536)
            if len(per_t2i_caption) > MAX_CAPTION_LENGTH:
                too_long_caption_count += 1
                print('3333', per_image_path, per_caption_source_name,
                      len(per_t2i_caption))
                continue

            per_check_result = process_single_image_check(per_image_path)
            if per_check_result is None:
                invalid_image_count += 1
                continue

            per_image_w, per_image_h = per_check_result

            # 分辨率过滤一律用上面真实解码出来的shape,这里只是顺手核对一遍上游标注里
            # 记录的image_resolution,不一致只计数上报、不丢样本(实测抽样0例不符)
            per_annotation_image_resolution = per_annotation.get(
                ANNOTATION_IMAGE_RESOLUTION_KEY_NAME, None)
            if not isinstance(
                    per_annotation_image_resolution, (list, tuple)) or len(
                        per_annotation_image_resolution
                    ) != 2 or per_annotation_image_resolution[
                        0] != per_image_w or per_annotation_image_resolution[
                            1] != per_image_h:
                annotation_image_size_not_match_count += 1
                print('2222', per_image_path, per_image_w, per_image_h,
                      per_annotation_image_resolution)

            # 保存图像名要等切完子集目录才能拼出来,这里只带上原始图名前缀
            image_annotation_pair_list.append([
                per_series_name,
                per_image_path,
                per_image_name_prefix,
                per_t2i_caption,
            ])

    return [
        image_annotation_pair_list,
        per_series_name,
        total_annotation_count,
        load_annotation_failed_count,
        missing_annotation_key_count,
        subset_name_not_match_count,
        invalid_enhanced_prompt_flag_count,
        missing_image_count,
        invalid_image_name_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_image_count,
        annotation_image_size_not_match_count,
        caption_source_not_match_count,
    ]


def get_all_image_annotation_pair(root_dataset_path):
    """扫描上游解压好的jsonl标注,多进程组装图像路径和t2i描述的样本对列表

    这里只listdir unzip_annotations下的5个系列目录拿到6356个jsonl路径,
    绝不去os.walk图像目录: 上游解出约1900万个小文件,扫目录树在NAS上不可接受。
    每个jsonl里的图像全在同一个tar目录下,worker只需要对那个目录listdir一次。
    最后按[系列名, 图名前缀]统一排序,保证输出顺序与串行版本完全一致。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    series_annotation_file_count_dict = {}
    for per_series_name in LOAD_SERIES_DIR_NAME_LIST:
        per_series_annotation_path = os.path.join(root_annotation_path,
                                                  per_series_name)
        if not os.path.isdir(per_series_annotation_path):
            print('2222', per_series_annotation_path)
            series_annotation_file_count_dict[per_series_name] = 0
            continue

        per_series_annotation_file_count = 0
        for per_jsonl_name in sorted(os.listdir(per_series_annotation_path)):
            if not per_jsonl_name.endswith(LOAD_ANNOTATION_FILE_SUFFIX):
                continue

            # 系列名由主进程按目录名算好后带给worker,
            # worker只认标注文件、数据集根目录、系列名这三个入参
            annotation_file_pair_list.append([
                os.path.join(per_series_annotation_path, per_jsonl_name),
                root_dataset_path,
                per_series_name,
            ])
            per_series_annotation_file_count += 1

        series_annotation_file_count_dict[
            per_series_name] = per_series_annotation_file_count

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_annotation_key_count, subset_name_not_match_count = 0, 0
    invalid_enhanced_prompt_flag_count = 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_caption_count, too_long_caption_count = 0, 0
    invalid_image_count, annotation_image_size_not_match_count = 0, 0
    caption_source_not_match_count = 0
    # 逐系列的原始行数(不是存活样本数): worker数的是jsonl里的原始行,
    # 所以过滤本身不会把"上游少写了行"这种情况掩盖掉,可以直接和上游对账报告比
    series_row_count_dict = {
        per_series_name: 0
        for per_series_name in LOAD_SERIES_DIR_NAME_LIST
    }
    image_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            image_annotation_pair_list.extend(per_load_result[0])
            series_row_count_dict[
                per_load_result[1]] = series_row_count_dict.get(
                    per_load_result[1], 0) + per_load_result[2]
            total_annotation_count += per_load_result[2]
            load_annotation_failed_count += per_load_result[3]
            missing_annotation_key_count += per_load_result[4]
            subset_name_not_match_count += per_load_result[5]
            invalid_enhanced_prompt_flag_count += per_load_result[6]
            missing_image_count += per_load_result[7]
            invalid_image_name_count += per_load_result[8]
            invalid_caption_count += per_load_result[9]
            too_long_caption_count += per_load_result[10]
            invalid_image_count += per_load_result[11]
            annotation_image_size_not_match_count += per_load_result[12]
            caption_source_not_match_count += per_load_result[13]

    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[0], x[2]])

    return [
        image_annotation_pair_list,
        series_annotation_file_count_dict,
        series_row_count_dict,
        total_annotation_count,
        load_annotation_failed_count,
        missing_annotation_key_count,
        subset_name_not_match_count,
        invalid_enhanced_prompt_flag_count,
        missing_image_count,
        invalid_image_name_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_image_count,
        annotation_image_size_not_match_count,
        caption_source_not_match_count,
    ]


def get_deduplicated_image_annotation_pair(image_annotation_pair_list):
    """按[系列名, 图名前缀]去重,同一个系列里每个图名前缀只保留排序后的第一条

    去重键必须带系列名,**不能只按图名前缀全局去重**: 本数据集的sample_key是
    "内容uuid"而不是图像内容哈希,实测同一个uuid最多在4个synthetic子集里各出一张
    **完全不同的图**(分辨率不同、生成模型不同、用的prompt不同),
    全局去重会误删约300万条合法样本; 而这些跨系列同名样本的保存图像名里带着
    不同的子集目录名前缀,本来就不会互相覆盖。
    系列内则必须去重: 实测curated 167979行里有296个key重复,且重复的两条是两张
    完全不同的图(pixabay vs pexels、分辨率不同、caption不同),curated只会切出
    一个子集目录curated_000,两条会拼出同一个保存图像名、后写覆盖先写、静默丢样本,
    按方案确认只留排序后第一条并上报。
    排序键取[系列名, 图名前缀, 图像路径],保证同一个键留下的永远是同一条。
    """
    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[0], x[2], x[1]])

    duplicate_key_dict = {}
    duplicate_image_name_prefix_list = []
    deduplicated_image_annotation_pair_list = []
    for per_image_annotation_pair in image_annotation_pair_list:
        per_series_name, per_image_path, per_image_name_prefix, per_t2i_caption = per_image_annotation_pair

        per_duplicate_key = f'{per_series_name}/{per_image_name_prefix}'
        if per_duplicate_key in duplicate_key_dict:
            duplicate_image_name_prefix_list.append(per_duplicate_key)
            print('2222', per_image_path, per_duplicate_key)
            continue

        duplicate_key_dict[per_duplicate_key] = 1
        deduplicated_image_annotation_pair_list.append(
            per_image_annotation_pair)

    deduplicated_image_annotation_pair_list = sorted(
        deduplicated_image_annotation_pair_list, key=lambda x: [x[0], x[2]])

    return deduplicated_image_annotation_pair_list, duplicate_image_name_prefix_list


def get_all_image_save_folder_pair(image_annotation_pair_list,
                                   save_dataset_path):
    """把过滤后的合格样本按系列分组,排序后每10000张切成一个文件夹、每100个文件夹归一个子集目录

    切分必须在过滤全部完成之后做,且切分前先按图名前缀排序,这样才能保证每个文件夹都是
    满10000张(只有每个系列全局最后一个文件夹允许不满)。
    排序键用图名前缀而不是保存图像名: 保存图像名里含子集目录名,而子集目录名恰恰由排序后
    的位置决定,存在循环依赖;同一个子集目录内所有图像名前缀完全相同,
    所以按图名前缀排序与按保存图像名排序结果完全等价。
    """
    per_series_image_annotation_pair_dict = {}
    for per_image_annotation_pair in image_annotation_pair_list:
        per_series_name = per_image_annotation_pair[0]
        if per_series_name not in per_series_image_annotation_pair_dict:
            per_series_image_annotation_pair_dict[per_series_name] = []
        per_series_image_annotation_pair_dict[per_series_name].append(
            per_image_annotation_pair)

    image_save_folder_pair_list = []
    set_folder_count_dict, series_set_name_list_dict = {}, {}
    for per_series_name in sorted(
            per_series_image_annotation_pair_dict.keys()):
        per_series_image_annotation_pair_list = sorted(
            per_series_image_annotation_pair_dict[per_series_name],
            key=lambda x: x[2])

        per_series_folder_count = 0
        per_series_set_name_list = []
        for per_folder_start_index in range(
                0, len(per_series_image_annotation_pair_list),
                PER_FOLDER_IMAGE_NUM):
            per_folder_image_annotation_pair_list = per_series_image_annotation_pair_list[
                per_folder_start_index:per_folder_start_index +
                PER_FOLDER_IMAGE_NUM]

            # 全局文件夹序号先定子集目录,再定子集目录内的文件夹序号
            per_set_index = per_series_folder_count // PER_SET_FOLDER_NUM
            per_set_folder_index = per_series_folder_count % PER_SET_FOLDER_NUM

            per_set_name = get_set_name(per_series_name, per_set_index)
            per_folder_name = f'{per_set_name}_{per_set_folder_index:05d}'
            per_folder_image_path = os.path.join(save_dataset_path,
                                                 per_set_name, per_folder_name)
            os.makedirs(per_folder_image_path, exist_ok=True)

            per_folder_save_pair_list = []
            for per_image_annotation_pair in per_folder_image_annotation_pair_list:
                _, per_image_path, per_image_name_prefix, per_t2i_caption = per_image_annotation_pair
                # 保存图像名统一全小写,形如
                # fine_t2i_curated_000_<uuid或数字id>.jpg
                per_save_image_name = f'{DATASET_NAME}_{per_set_name}_{per_image_name_prefix}{SAVE_IMAGE_NAME_SUFFIX}'
                per_folder_save_pair_list.append([
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

            # 一个文件夹就是一个写盘任务,worker写完这10000张后直接写出该文件夹的json,
            # 主进程只收计数,不用把631万条记录再攒一遍
            image_save_folder_pair_list.append([
                per_set_name,
                per_folder_name,
                per_folder_save_pair_list,
            ])

            if per_set_name not in set_folder_count_dict:
                set_folder_count_dict[per_set_name] = 0
                per_series_set_name_list.append(per_set_name)
            set_folder_count_dict[per_set_name] += 1

            per_series_folder_count += 1

        series_set_name_list_dict[per_series_name] = per_series_set_name_list

    return image_save_folder_pair_list, set_folder_count_dict, series_set_name_list_dict


def process_single_image_folder(image_save_folder_pair, save_dataset_path):
    """重新编码保存一个文件夹的图像,并写出与文件夹同名的json标注

    图像原分辨率多少保存时还是多少,不做任何缩放。
    以文件夹为任务粒度而不是以单张图为粒度: 本数据集约631万张图,逐图收结果的话
    主进程要再攒一份631万条的列表,而且中途挂了只能从头再来;按文件夹收之后
    主进程内存只和文件夹数(约632)相关,且json已经写全的文件夹可以直接跳过、支持断点续跑。
    """
    per_set_name, per_folder_name, per_folder_save_pair_list = image_save_folder_pair

    save_folder_path = os.path.join(save_dataset_path, per_set_name,
                                    per_folder_name)
    save_json_path = os.path.join(save_dataset_path, per_set_name,
                                  f'{per_folder_name}.json')

    expect_save_image_name_list = sorted([
        per_save_image_name
        for _, per_save_image_name, _ in per_folder_save_pair_list
    ])

    # 断点续跑: json已经写全且记录的图像名与本次任务完全一致时整个文件夹跳过
    if os.path.isfile(save_json_path):
        try:
            with open(save_json_path, 'r', encoding='UTF-8') as load_json_file:
                per_folder_annotation_dict = json.load(load_json_file)
        except Exception as e:
            print('9999', save_json_path, e)
            per_folder_annotation_dict = {}

        if sorted(per_folder_annotation_dict.keys(
        )) == expect_save_image_name_list and sorted(
                os.listdir(save_folder_path)) == expect_save_image_name_list:
            return [
                per_set_name,
                per_folder_name,
                len(per_folder_annotation_dict),
                0,
            ]

    per_folder_annotation_dict = {}
    save_image_failed_count = 0
    for per_folder_save_pair in per_folder_save_pair_list:
        per_image_path, per_save_image_name, per_t2i_caption = per_folder_save_pair

        try:
            per_image = cv2.imdecode(
                np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            save_image_failed_count += 1
            print('8888', per_image_path, e)
            continue

        if per_image is None or per_image.ndim != 3 or per_image.shape[2] != 3:
            save_image_failed_count += 1
            print('8888', per_image_path)
            continue

        # json里的宽高直接取自这个即将被编码写盘的数组的shape,
        # 中间不做resize,jpg编解码也不改变像素尺寸,所以宽高一定和保存图像一致
        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

        save_image_path = os.path.join(save_folder_path, per_save_image_name)

        if not os.path.exists(save_image_path):
            try:
                cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)
            except Exception as e:
                save_image_failed_count += 1
                print('8888', save_image_path, e)
                continue

        # t2i_caption_length直接取即将写进json的这个字符串的长度,
        # 保证记录的长度和t2i_caption永远自洽(该字符串在过滤阶段已经strip过)
        per_folder_annotation_dict[per_save_image_name] = {
            'width': per_image_w,
            'height': per_image_h,
            't2i_caption': per_t2i_caption,
            't2i_caption_length': len(per_t2i_caption),
        }

    per_folder_annotation_dict = {
        per_save_image_name: per_folder_annotation_dict[per_save_image_name]
        for per_save_image_name in sorted(per_folder_annotation_dict.keys())
    }

    try:
        with open(save_json_path, 'w', encoding='UTF-8') as save_json_file:
            json.dump(per_folder_annotation_dict,
                      save_json_file,
                      ensure_ascii=False)
    except Exception as e:
        print('9999', save_json_path, e)

    return [
        per_set_name,
        per_folder_name,
        len(per_folder_annotation_dict),
        save_image_failed_count,
    ]


def check_single_save_folder(folder_check_pair, save_dataset_path):
    """校验单个文件夹: json与磁盘一一对应、图像名和描述合规、文件夹容量

    每个系列只有全局最后一个文件夹允许不满10000张,其余都必须是满10000张。
    """
    per_set_name, per_folder_name, per_is_series_last_folder = folder_check_pair

    check_error_message_list = []

    per_json_path = os.path.join(save_dataset_path, per_set_name,
                                 f'{per_folder_name}.json')
    if not os.path.isfile(per_json_path):
        check_error_message_list.append(f'{per_folder_name} json not exists')

        return [per_folder_name, 0, check_error_message_list]

    try:
        with open(per_json_path, 'r', encoding='UTF-8') as load_json_file:
            per_folder_annotation_dict = json.load(load_json_file)
    except Exception as e:
        check_error_message_list.append(
            f'{per_folder_name} load json failed {e}')

        return [per_folder_name, 0, check_error_message_list]

    # 除每个系列全局最后一个文件夹外都必须是满10000张
    if not per_is_series_last_folder and len(
            per_folder_annotation_dict) != PER_FOLDER_IMAGE_NUM:
        check_error_message_list.append(
            f'{per_folder_name} image num not match {len(per_folder_annotation_dict)} != {PER_FOLDER_IMAGE_NUM}'
        )

    per_folder_path = os.path.join(save_dataset_path, per_set_name,
                                   per_folder_name)
    per_exist_image_name_list = sorted(
        os.listdir(per_folder_path)) if os.path.isdir(per_folder_path) else []
    per_expect_image_name_list = sorted(per_folder_annotation_dict.keys())
    if per_exist_image_name_list != per_expect_image_name_list:
        check_error_message_list.append(
            f'{per_folder_name} image file not match {len(per_exist_image_name_list)} != {len(per_expect_image_name_list)}'
        )

    for per_save_image_name in per_expect_image_name_list:
        per_annotation = per_folder_annotation_dict[per_save_image_name]

        if not per_save_image_name.endswith(SAVE_IMAGE_NAME_SUFFIX):
            check_error_message_list.append(
                f'{per_save_image_name} image name suffix not match')
        if not per_save_image_name.startswith(
                f'{DATASET_NAME}_{per_set_name}_'):
            check_error_message_list.append(
                f'{per_save_image_name} image name prefix not match')
        if per_save_image_name != per_save_image_name.lower():
            check_error_message_list.append(
                f'{per_save_image_name} image name not all lower case')
        if not VALID_IMAGE_NAME_PATTERN.match(per_save_image_name):
            check_error_message_list.append(
                f'{per_save_image_name} image name has invalid char')
        if min(per_annotation['width'],
               per_annotation['height']) < MIN_IMAGE_SHORT_SIDE:
            check_error_message_list.append(
                f'{per_save_image_name} image short side not match')
        if max(
                per_annotation['width'] / per_annotation['height'],
                per_annotation['height'] / per_annotation['width'],
        ) > MAX_IMAGE_ASPECT_RATIO:
            check_error_message_list.append(
                f'{per_save_image_name} image aspect ratio not match')
        if len(per_annotation['t2i_caption'].strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_image_name} still an invalid caption')
        if len(per_annotation['t2i_caption'].strip()) > MAX_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_image_name} still a too long caption')
        # 记录的描述长度必须和描述字符串的实际长度对得上
        if per_annotation['t2i_caption_length'] != len(
                per_annotation['t2i_caption']):
            check_error_message_list.append(
                f'{per_save_image_name} t2i caption length not match')

    return [
        per_folder_name,
        len(per_folder_annotation_dict),
        check_error_message_list,
    ]


def check_save_dataset(save_dataset_path, set_folder_count_dict,
                       series_set_name_list_dict):
    """全部落盘后的收尾自校验: 子集目录容量、文件夹容量、json与磁盘一一对应、图像名和描述合规

    和001的差别只在"满10000张"的口径: 001里子集本身就是切分单位,所以每个子集的最后一个
    文件夹都允许不满;这里子集目录只是100个文件夹的容器,每个系列会被切成多个子集目录,
    所以只有整个系列全局最后一个文件夹允许不满,其余每个文件夹都必须是满10000张。
    同理每个系列只有最后一个子集目录允许不满100个文件夹。
    约632个文件夹每个都要listdir一万个文件再load一份json,串行跑在NAS上太久,
    所以这一步也按文件夹粒度开多进程。
    """
    check_error_message_list = []

    folder_check_pair_list = []
    for per_series_name in sorted(series_set_name_list_dict.keys()):
        per_series_set_name_list = series_set_name_list_dict[per_series_name]
        for per_set_index, per_set_name in enumerate(per_series_set_name_list):
            per_set_folder_count = set_folder_count_dict[per_set_name]

            # 每个系列只有最后一个子集目录允许不满100个文件夹
            if per_set_index < len(
                    per_series_set_name_list
            ) - 1 and per_set_folder_count != PER_SET_FOLDER_NUM:
                check_error_message_list.append(
                    f'{per_set_name} folder num not match {per_set_folder_count} != {PER_SET_FOLDER_NUM}'
                )

            for per_folder_index in range(per_set_folder_count):
                per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
                per_is_series_last_folder = (
                    per_set_index == len(per_series_set_name_list) - 1
                    and per_folder_index == per_set_folder_count - 1)
                folder_check_pair_list.append([
                    per_set_name,
                    per_folder_name,
                    per_is_series_last_folder,
                ])

    total_image_count = 0
    check_func = partial(check_single_save_folder,
                         save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_func, folder_check_pair_list),
                                     total=len(folder_check_pair_list)):
            _, per_folder_image_count, per_check_error_message_list = per_check_result
            total_image_count += per_folder_image_count
            check_error_message_list.extend(per_check_error_message_list)

    print('3333', 'check total image:', total_image_count, 'check error:',
          len(check_error_message_list))

    return check_error_message_list, total_image_count


def preprocess_dataset(root_dataset_path, save_dataset_path):
    save_dataset_path = os.path.join(save_dataset_path, SAVE_DATASET_DIR_NAME)
    os.makedirs(save_dataset_path, exist_ok=True)

    image_annotation_pair_list, series_annotation_file_count_dict, series_row_count_dict, total_annotation_count, load_annotation_failed_count, missing_annotation_key_count, subset_name_not_match_count, invalid_enhanced_prompt_flag_count, missing_image_count, invalid_image_name_count, invalid_caption_count, too_long_caption_count, invalid_image_count, annotation_image_size_not_match_count, caption_source_not_match_count = get_all_image_annotation_pair(
        root_dataset_path)

    print('1111', series_annotation_file_count_dict, series_row_count_dict,
          total_annotation_count, load_annotation_failed_count,
          missing_annotation_key_count, subset_name_not_match_count,
          invalid_enhanced_prompt_flag_count, missing_image_count,
          invalid_image_name_count, invalid_caption_count,
          too_long_caption_count, invalid_image_count,
          annotation_image_size_not_match_count,
          caption_source_not_match_count, len(image_annotation_pair_list))

    if len(image_annotation_pair_list) > 0:
        print('1111', image_annotation_pair_list[0])

    # 上游jsonl分片数量不对说明上游没跑完,继续跑只会静默少样本对
    annotation_file_error_message_list = []
    for per_series_name, per_expected_annotation_file_num in EXPECTED_SERIES_ANNOTATION_FILE_NUM_DICT.items(
    ):
        per_annotation_file_num = series_annotation_file_count_dict.get(
            per_series_name, 0)
        if per_annotation_file_num != per_expected_annotation_file_num:
            annotation_file_error_message_list.append(
                f'{per_series_name} annotation file num not match {per_annotation_file_num} != {per_expected_annotation_file_num}'
            )

    # 逐系列的jsonl行数必须和上游对账报告完全一致,少行说明标注被截断。
    # 这里比的是原始行数(不是存活样本数),所以过滤本身不会把缺行掩盖掉;
    # 逐系列比而不是只比总数: 只比总数时"A系列多B系列少"会互相抵消
    for per_series_name, per_expected_row_count in EXPECTED_SERIES_ROW_COUNT_DICT.items(
    ):
        per_row_count = series_row_count_dict.get(per_series_name, 0)
        if per_row_count != per_expected_row_count:
            annotation_file_error_message_list.append(
                f'{per_series_name} annotation row count not match {per_row_count} != {per_expected_row_count}'
            )

    expected_total_row_count = sum(EXPECTED_SERIES_ROW_COUNT_DICT.values())
    if total_annotation_count != expected_total_row_count:
        annotation_file_error_message_list.append(
            f'total annotation row count not match {total_annotation_count} != {expected_total_row_count}'
        )

    if len(annotation_file_error_message_list) > 0:
        raise Exception(
            f'check annotation file failed {annotation_file_error_message_list}'
        )

    if load_annotation_failed_count > 0:
        raise Exception(
            f'load annotation failed count {load_annotation_failed_count}')

    # 缺key、成员错位、flag类型不对都说明上游规格变了,必须显式感知,
    # 不能静默少样本对或静默配错文本。
    # 这几道检查放在扫描阶段结束、写盘开始之前,不会白跑几十小时
    if missing_annotation_key_count > 0:
        raise Exception(
            f'missing annotation key count {missing_annotation_key_count}')

    if subset_name_not_match_count > 0:
        raise Exception(
            f'subset name not match count {subset_name_not_match_count}')

    if invalid_enhanced_prompt_flag_count > 0:
        raise Exception(
            f'invalid enhanced prompt flag count {invalid_enhanced_prompt_flag_count}'
        )

    image_annotation_pair_list, duplicate_image_name_prefix_list = get_deduplicated_image_annotation_pair(
        image_annotation_pair_list)

    print('1111', len(image_annotation_pair_list),
          len(duplicate_image_name_prefix_list))
    if len(duplicate_image_name_prefix_list) > 0:
        print('1111', duplicate_image_name_prefix_list[:10])

    image_save_folder_pair_list, set_folder_count_dict, series_set_name_list_dict = get_all_image_save_folder_pair(
        image_annotation_pair_list, save_dataset_path)

    total_save_task_image_count = sum([
        len(per_folder_save_pair_list)
        for _, _, per_folder_save_pair_list in image_save_folder_pair_list
    ])

    print('1111', len(image_save_folder_pair_list), len(set_folder_count_dict),
          total_save_task_image_count)
    if len(image_save_folder_pair_list) > 0:
        print('1111', image_save_folder_pair_list[0][0],
              image_save_folder_pair_list[0][1],
              image_save_folder_pair_list[0][2][0])

    # 切分之后保存图像名必须全局唯一,撞名会让后写的图像覆盖先写的、静默丢样本,
    # 所以这里再兜一道,撞上就直接中止。
    # 跨系列同名的uuid在这里天然不会撞: 保存图像名里带着各自的子集目录名
    save_image_name_dict = {}
    conflict_save_image_name_list = []
    for _, _, per_folder_save_pair_list in image_save_folder_pair_list:
        for _, per_save_image_name, _ in per_folder_save_pair_list:
            if per_save_image_name in save_image_name_dict:
                conflict_save_image_name_list.append(per_save_image_name)
                continue
            save_image_name_dict[per_save_image_name] = 1

    if len(conflict_save_image_name_list) > 0:
        raise Exception(
            f'conflict save image name num {len(conflict_save_image_name_list)} {conflict_save_image_name_list[:10]}'
        )

    save_image_name_dict = {}

    folder_image_count_dict = {}
    save_image_failed_count = 0
    process_func = partial(process_single_image_folder,
                           save_dataset_path=save_dataset_path)
    with Pool(processes=PROCESS_NUM) as pool:
        for per_save_result in tqdm(pool.imap_unordered(
                process_func, image_save_folder_pair_list),
                                    total=len(image_save_folder_pair_list)):
            per_set_name, per_folder_name, per_folder_image_count, per_save_image_failed_count = per_save_result
            folder_image_count_dict[per_folder_name] = per_folder_image_count
            save_image_failed_count += per_save_image_failed_count

            print('2222', per_folder_name, per_folder_image_count,
                  per_save_image_failed_count)

    total_save_image_count = sum(folder_image_count_dict.values())

    check_error_message_list, check_total_image_count = check_save_dataset(
        save_dataset_path, set_folder_count_dict, series_set_name_list_dict)

    print('3333', 'total annotation:', total_annotation_count,
          'missing annotation key:', missing_annotation_key_count,
          'subset name not match:', subset_name_not_match_count,
          'invalid enhanced prompt flag:', invalid_enhanced_prompt_flag_count,
          'missing image:', missing_image_count, 'invalid image name:',
          invalid_image_name_count, 'invalid caption:', invalid_caption_count,
          'too long caption:', too_long_caption_count, 'invalid image:',
          invalid_image_count, 'annotation image size not match:',
          annotation_image_size_not_match_count, 'caption source not match:',
          caption_source_not_match_count, 'duplicate image name:',
          len(duplicate_image_name_prefix_list), 'save image failed:',
          save_image_failed_count, 'total save image:',
          total_save_image_count, 'total save folder:',
          len(folder_image_count_dict), 'total save set:',
          len(set_folder_count_dict),
          'check total image:', check_total_image_count, 'check error:',
          len(check_error_message_list))

    save_check_result_path = os.path.join(save_dataset_path,
                                          'resave_check_result.json')
    save_check_result_dict = {
        'series_annotation_file_count_dict':
        series_annotation_file_count_dict,
        'series_row_count_dict':
        series_row_count_dict,
        'total_annotation_count':
        total_annotation_count,
        'load_annotation_failed_count':
        load_annotation_failed_count,
        'missing_annotation_key_count':
        missing_annotation_key_count,
        'subset_name_not_match_count':
        subset_name_not_match_count,
        'invalid_enhanced_prompt_flag_count':
        invalid_enhanced_prompt_flag_count,
        'missing_image_count':
        missing_image_count,
        'invalid_image_name_count':
        invalid_image_name_count,
        'invalid_caption_count':
        invalid_caption_count,
        'too_long_caption_count':
        too_long_caption_count,
        'invalid_image_count':
        invalid_image_count,
        'annotation_image_size_not_match_count':
        annotation_image_size_not_match_count,
        'caption_source_not_match_count':
        caption_source_not_match_count,
        'duplicate_image_name_count':
        len(duplicate_image_name_prefix_list),
        'total_save_task_image_count':
        total_save_task_image_count,
        'save_image_failed_count':
        save_image_failed_count,
        'total_save_image_count':
        total_save_image_count,
        'total_save_folder_count':
        len(folder_image_count_dict),
        'total_save_set_count':
        len(set_folder_count_dict),
        'check_total_image_count':
        check_total_image_count,
        'check_error_count':
        len(check_error_message_list),
        'series_set_name_list_dict':
        series_set_name_list_dict,
        'set_folder_count_dict':
        set_folder_count_dict,
        'folder_image_count_dict':
        folder_image_count_dict,
        'duplicate_image_name_prefix_list':
        duplicate_image_name_prefix_list[:10000],
    }
    with open(save_check_result_path, 'w', encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    if total_save_image_count != total_save_task_image_count:
        check_error_message_list.append(
            f'total save image count not match {total_save_image_count} != {total_save_task_image_count}'
        )
    if check_total_image_count != total_save_image_count:
        check_error_message_list.append(
            f'check total image count not match {check_total_image_count} != {total_save_image_count}'
        )
    if save_image_failed_count > 0:
        check_error_message_list.append(
            f'save image failed count {save_image_failed_count}')
    if len(check_error_message_list) > 0:
        # 收尾自校验不通过必须让上层感知,不能静默留下坏样本或不满的文件夹
        raise Exception(
            f'check save dataset error num {len(check_error_message_list)} {check_error_message_list[:10]}'
        )

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/fine-t2i'
    save_dataset_path = r'/root/autodl-tmp/t2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
