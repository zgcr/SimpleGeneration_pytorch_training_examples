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

SAVE_DATASET_DIR_NAME = 'FoundIR'

# 上游009.unzip_foundir_dataset.py把17个分卷zip组解压成
#   images/<group_name>/GT/<7位主干>.<png|JPG|jpg>   编辑后图(清晰GT)
#   images/<group_name>/LQ/<7位主干>.<png|JPG|jpg>   参考图(退化LQ)
# 并写出unzip_annotations/<group_name>/<块号>.jsonl(每行1个完整样本对，但没有指令);
# 上游009.1.generate_foundir_edit_instruction_by_seed.py再把每一行原封不动复制、
# 只追加ti2i_caption等指令字段，写成seed_instruction_annotations/<group_name>/<块号>.jsonl。
# 也就是说seed_instruction_annotations是unzip_annotations的严格超集(实测字段完全包含、
# 行数与文件名一一对应、009.1自身已做过"输入文件名集合/每文件行数/每组行数/总行数"
# 四级对账)，所以本脚本只读这一套标注，不再读unzip_annotations做交叉对账。
# 实测17个组共464个jsonl、合计920223行
LOAD_ANNOTATION_DIR_NAME = 'seed_instruction_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如images/01Blur/LQ/0000001.png)，不需要再拼images子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 上游标注天然就是464个块文件(每块2000个样本对)，不像007那样只有一个大jsonl，
# 所以这里直接按标注文件粒度开多进程解析，不需要先切分片
ANNOTATION_GROUP_NAME_KEY_NAME = 'group_name'

# 退化组名机械拆出来的客观退化类型标签，实测920223行全非空、
# 8种退化基元(blur/noise/jpeg/haze/lowlight/rain/raindrop/night)组合出17种。
# 这就是本数据集能拿到的图像编辑任务类型，子集按它划分
ANNOTATION_DEGRADATION_TYPE_LIST_KEY_NAME = 'degradation_type_list'

# 样本主干id，实测全局唯一、7位数字、17组首尾相接覆盖0000001~0920224
ANNOTATION_SAMPLE_KEY_KEY_NAME = 'sample_key'

# 编辑指令，由上游009.1按"退化类型指令池 + md5(group_name+sample_key)哈希分配"生成，
# 实测920223行全非空、长度15~105字符、全小写开头、句末无句号、不含任何占位符
ANNOTATION_CAPTION_KEY_NAME = 'ti2i_caption'

# 编辑前的退化图LQ(唯一的参考图)，实测920223行全非空
ANNOTATION_REFERENCE_IMAGE_KEY_NAME = 'reference_image_path'

# 编辑后的清晰图GT，实测920223行全非空。
# 注意GT与LQ的后缀可以不一样(实测有png对JPG、png对png、jpg对jpg、jpg对png四种组合)，
# 所以两张图的路径必须各自独立取，不能拿一张的后缀去拼另一张
ANNOTATION_EDITED_IMAGE_KEY_NAME = 'target_image_path'

# 参考图顺序固定为[reference_image_path(编辑前退化图LQ)]。
# 这个数据集每个编辑对只有1张参考图，没有第二张视觉条件图，
# 所以reference_image恒为长度1的list、reference_image_num恒为1
SAVE_REFERENCE_IMAGE_KEY_NAME_LIST = [
    ANNOTATION_REFERENCE_IMAGE_KEY_NAME,
]

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 保存图像名前缀里的数据集名(全小写)
SAVE_IMAGE_NAME_DATASET_PREFIX = 'foundir'

# group_name -> 归一化后的任务名(即子集名)。
# 这个数据集的图像编辑任务类型是明确可知的: 17个退化组就是17种图像复原任务，
# 组名本身就是"编号 + 退化类型组合"，所以子集名直接取degradation_type_list
# 下划线拼接的结果(等价于组名去掉数字编号前缀并转小写下划线风格):
#   01Blur            -> blur
#   12NightRain       -> night_rain      (组名里NightRain是两种退化，拆成night+rain)
#   15Lowlight_Blur   -> lowlight_blur
# 实测17个子集名互不相同、与group_name严格一对一，不存在007那种"同一语义在不同来源
# 里叫法不同"需要跨组归并的情况，所以这里是显式的一对一映射表而不是归并表。
# 显式写死而不是每行现拆的原因: 拆错一个组就会静默把几万个样本对写进错误子集，
# 写死之后任何组名/退化标签变化都会在check_load_annotation_count里硬失败
GET_SET_NAME_DICT = {
    '01Blur': 'blur',
    '02Blur_Noise': 'blur_noise',
    '03Blur_JPEG': 'blur_jpeg',
    '04Blur_Noise_JPEG': 'blur_noise_jpeg',
    '05Noise': 'noise',
    '06JPEG': 'jpeg',
    '07Noise_JPEG': 'noise_jpeg',
    '08Haze': 'haze',
    '09Lowlight_Haze': 'lowlight_haze',
    '10Rain': 'rain',
    '11Raindrop': 'raindrop',
    '12NightRain': 'night_rain',
    '13Rain_Haze': 'rain_haze',
    '14Lowlight': 'lowlight',
    '15Lowlight_Blur': 'lowlight_blur',
    '16Lowlight_Noise': 'lowlight_noise',
    '17Lowlight_JPEG': 'lowlight_jpeg',
}

# 找不到任务类型时才用的兜底子集名。
# 本数据集17个组的任务类型全部可知(见GET_SET_NAME_DICT)，实测一条都不会落进mix，
# 保留这条路径只是为了和002/006/007的口径保持一致，
# 并防止上游之后新增退化组时被静默漏处理
MIX_SET_NAME = 'mix'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点。
# 本数据集保存名形如foundir_0000001_edited.jpg(固定27字符)，
# 因为sample_key是7位数字且全局唯一，920223个保存名100%唯一、0重名，
# 不需要像007那样再往名字里塞source和task
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 保存图像名里的原图名前缀(即sample_key)必须是7位数字，
# 上游009已硬校验过"主干全是数字且17组连号覆盖1~920224"，这里做落盘前的最后一道拦截
VALID_SAMPLE_KEY_PATTERN = re.compile(r'^\d{7}$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃。
# 实测抽样102张(17组各3对的GT+LQ)全是RGB
VALID_IMAGE_MODE_LIST = [
    'RGB',
    'L',
]

# 每个退化组(即每个子集)在上游seed_instruction_annotations里的实测标注行数，
# 合计920223条。解析阶段按组和按子集两级硬对账，少一条都说明上游009/009.1没跑完
# 或产物被改动过。
# 注意13Rain_Haze是79749而不是中央目录里数出来的79750: 13Rain_Haze/0673776这一个
# 成员在原始分卷zip里本身就是坏的(009解压时CRC/原始大小校验失败并硬上报，
# unzip_check_result.json里incomplete_sample_pair_list长度1)，
# 所以上游标注里本来就没有这一行。这是上游数据下载被截断导致的，本脚本无法也不应该补，
# 按方案直接把它当成"不存在的样本对"、期望值就取79749/920223;
# 上游重新下载补齐13Rain_Haze分卷并重跑009/009.1之后，把这两个数字改回79750/920224即可
EXPECTED_ANNOTATION_COUNT_DICT = {
    '01Blur': 109480,
    '02Blur_Noise': 29950,
    '03Blur_JPEG': 29940,
    '04Blur_Noise_JPEG': 29950,
    '05Noise': 58015,
    '06JPEG': 59950,
    '07Noise_JPEG': 29950,
    '08Haze': 79800,
    '09Lowlight_Haze': 79800,
    '10Rain': 39900,
    '11Raindrop': 44828,
    '12NightRain': 40111,
    '13Rain_Haze': 79749,
    '14Lowlight': 39962,
    '15Lowlight_Blur': 85893,
    '16Lowlight_Noise': 52995,
    '17Lowlight_JPEG': 29950,
}

# 17个group归一化出的17个子集(即17个图像复原任务类型)及其实测标注条数。
# 本数据集group与子集是一对一的，所以数值与EXPECTED_ANNOTATION_COUNT_DICT逐项相同，
# 但仍然分开写死: 子集级对账能额外拦住"某个组被映射进错误子集"这种
# 组级对账看不出来的问题
EXPECTED_SET_ANNOTATION_COUNT_DICT = {
    'blur': 109480,
    'blur_noise': 29950,
    'blur_jpeg': 29940,
    'blur_noise_jpeg': 29950,
    'noise': 58015,
    'jpeg': 59950,
    'noise_jpeg': 29950,
    'haze': 79800,
    'lowlight_haze': 79800,
    'rain': 39900,
    'raindrop': 44828,
    'night_rain': 40111,
    'rain_haze': 79749,
    'lowlight': 39962,
    'lowlight_blur': 85893,
    'lowlight_noise': 52995,
    'lowlight_jpeg': 29950,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = 920223

# 上游实测的标注文件总数(17组共464个块文件，每块最多2000个样本对)
EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 464

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
# 长宽比判定用Fraction最简分数比、**不留任何容差**。
# 落盘后还会把参考图的真实shape与目标shape硬对账一次，不等则整对丢弃。
#
# 【本数据集的实际情况】编辑后图GT与参考图LQ是像素对齐、同分辨率的
# (上游009实测920223对的same_size_flag全为True)，所以这里的resize在绝大多数样本上
# 是恒等操作、different_aspect_ratio_count预期为0。
# 加这套代码的意义是把"上游恰好一致"这个事实升级成"本脚本硬保证":
# 一旦上游产物换代、GT与LQ不再同尺寸，会立刻被判出来而不是静默落盘成错位样本对。
# ==============================================================================

# 参考图resize到目标尺寸时用的重采样方式。
# 按方案确认用PIL的LANCZOS而不是cv2.resize: 与项目既定口径(017.0/017.1)保持一致
SAVE_IMAGE_RESIZE_RESAMPLING = Image.Resampling.LANCZOS

# 豁免尺寸对齐的子集: 这些子集的编辑后图与全部参考图**完全原样落盘**，
# 不判长宽比、不resize、不丢弃、也不做长边对齐。
# 本数据集17个子集全部都是"参考图与编辑后图像素对齐"的图像复原任务，
# 没有任何子集需要豁免，所以这里是空列表(保留这个常量只为与001等脚本口径一致)
EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST = []

# 收尾自校验时是否真解一次reference_image[0]、硬校验它的shape等于json里的
# width/height。按方案确认置True: "第一张参考图与编辑后图尺寸必须一致"是本次改动的
# 核心诉求，而只对账json里的数字是查不出resize有没有真的生效的，必须真解一次图。
# 代价是收尾自校验要多解约92万张参考图，在NAS上会明显变慢
CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG = True

# 上游指令是模板类指令(退化类型指令池 + 哈希分配)，实测920223条长度15~105字符、
# 各组中位38~75字符，没有空指令也没有超长指令。
# 阈值仍按anyedit-split的口径取10/200(与002.resave_anyedit_split_dataset.py一致):
# 实测一条都不会被这两个阈值砍掉，纯粹作为跨数据集统一的兜底，
# 防止上游指令池换代后悄悄写进空指令或几百字符的长描述
MIN_CAPTION_LENGTH = 10

MAX_CAPTION_LENGTH = 200

# jpg重编码质量与色度采样方式。
# 001~008那8个脚本用的都是cv2.imencode('.jpg', img)的默认值(质量95 + 色度4:2:0)，
# 本数据集显式改成"质量97 + 色度4:4:4(不下采样)"，且按方案只改本脚本、
# 不动001~008(那5个数据集已产出，统一改要全部重跑;
# 本数据集是唯一的"图像复原"任务、编辑后图像GT是高清干净原图，对保真最敏感)。
#
# 【实测依据】17个退化组各随机抽样(先随机选块文件、再在块文件内随机选行，
# 覆盖每组整个主干区间; 不能只抽每组第一个块文件的前几行，那样全是主干id最小的
# 连号样本、同一场景同一台相机，结论有偏):
# 第一轮510对/1020张测质量档，第二轮340对/680张测色度采样。
#
# 1) 先要知道源图有近一半GT本身就是jpg(全量扫描464个标注文件920223行，非抽样):
#      编辑后图GT : png 487696(53.00%) / jpg 432527(47.00%)
#      参考图LQ   : png 307956(33.47%) / jpg 612267(66.53%)
#    jpg源图存在"量化表幂等性": 源图本来就是libjpeg用q≈95的标准量化表压过的，
#    用同一张表重编码时DCT系数落回原来的量化格子(近乎无损，60dB+)，
#    一旦换成q97的另一张表，系数被重新量化到不同格子反而引入新误差。
#    所以jpg源图上会出现"质量提高但PSNR反而下降"(实测510张jpg源里46.3%非单调，
#    如13Rain_Haze/0673803: q95=60.83dB -> q96=53.17 -> q97=54.11 -> q100=59.14)，
#    而png源图(真无损，才是编码器损失的真实度量)完全单调:
#      png源GT  q90 43.45dB / q95 45.43dB / q97 46.47dB / q100 48.79dB
#    衡量编码损失必须按源格式拆开看，混在一起平均会得到错误结论。
#
# 2) 按上面的源格式占比加权后的质量/体积前沿(GT指标只统计编辑后图):
#      配置          GT PSNR  GT SSIM  GT色度PSNR  全量总体积
#      q95 4:2:0     47.66    0.98933   52.98      1.63 TB   (001~008的现状)
#      q97 4:2:0     48.37    0.99165   53.49      2.00 TB
#      q95 4:4:4     48.96    0.99181   55.52      1.92 TB
#      q97 4:4:4     50.25    0.99413   56.90      2.42 TB   <- 本脚本采用
#      q98 4:4:4     51.45    0.99537   57.66      2.71 TB
#      q100 4:2:0    51.41    0.99586   55.47      3.13 TB
#      q100 4:4:4    55.11    0.99777   60.59      4.18 TB
#    关键结论: 真正的瓶颈是色度下采样而不是质量值。OpenCV默认4:2:0会把色度分辨率
#    直接砍半(解SOF确认采样因子是2,2)，光提高质量值救不回来——
#    q100 4:2:0花了+93%体积，色度保真(55.47)还不如q95 4:4:4(55.52)。
#    所以这里把预算优先花在关掉色度下采样上，再叠一档质量到97。
#
# 3) 本数据集有blur_jpeg/jpeg/noise_jpeg/blur_noise_jpeg/lowlight_jpeg这5个
#    "去jpeg压缩伪影"子集(合计179740对)。如果GT自己就带一层可见的jpeg块效应，
#    "把jpeg伪影去掉"这个任务的监督信号会自相矛盾(要求模型去伪影、却又拿带伪影的
#    图当目标)。这5个子集的GT实测也正是全数据集最脆弱的一档(比其余12个子集低约4dB):
#      配置        去jpeg5子集GT PSNR   其余12子集GT PSNR
#      q95 4:2:0        44.22               48.31
#      q97 4:2:0        45.41               48.96
#      q97 4:4:4        47.57               50.69
#    q97 4:4:4把这5个子集的GT从44.22dB拉到47.57dB(+3.35dB)，监督信号才干净。
#
# 4) 参考图LQ本身就是退化图、对编码质量不敏感，但仍然用与GT完全相同的编码参数，
#    避免GT和LQ走两条不同的编码链路引入"参考图多一层压缩"这种GT/LQ不对称的伪偏差。
# 5) 不取q100: 体积再翻倍(4.18TB)而收益递减，且jpeg即使q100仍是有损。
#    磁盘余量246TB，2.42TB不构成约束。
SAVE_IMAGE_JPEG_QUALITY = 97

# 色度采样方式取4:4:4(不对色度做下采样)。
# cv2.imencode默认是4:2:0，色度分辨率砍半，是本数据集重编码损失的主要来源(见上表)。
SAVE_IMAGE_JPEG_SAMPLING_FACTOR = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444

# 落盘时统一使用的jpg编码参数，GT和所有参考图都走这一套，保证编码链路完全一致
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
# 这套写法与002.resave_anyedit_split_dataset.py、006.resave_imgedit_dataset.py、
# 007.resave_gpt_image_edit_1_5m_dataset.py完全一致，保证跨数据集口径统一。
# 本数据集恒为1张参考图(N == 0)，所以指令里不允许出现任何占位符
# (上游009.1的指令池校验已禁掉了'['、']'、'*'等字符，这里只做防御性拦截)
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2


def check_same_image_aspect_ratio(per_reference_image_shape,
                                  per_edited_image_shape):
    """判定参考图与编辑后图的长宽比是否严格相同，返回True表示相同(可以等比resize)

    两个入参都是[宽, 高]。
    用Fraction的最简分数比精确判定、**不留任何容差**，不用浮点相除:
    浮点比较要么因为精度误差把本该相同的判成不同(如1056/1584与832/1248)，
    要么需要引入一个人为的容差阈值，而容差一旦放开就会让"几乎一样但不严格相等"的
    样本被各向异性拉伸落盘。最简分数比是精确的、可复现的。
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


def get_normalized_task_name(per_group_name, per_degradation_type_list):
    """把(group_name, degradation_type_list)映射成归一化后的任务名

    本数据集group与任务类型一对一，优先用写死的映射表(拆错一个组就会静默把几万个
    样本对写进错误子集); 表里没有的组退回到degradation_type_list下划线拼接，
    这条路径只在上游新增退化组时才会走到，且会在check_load_annotation_count里
    被"unknown group"硬拦下来。
    """
    per_group_name = str(per_group_name).strip()

    if per_group_name in GET_SET_NAME_DICT:
        return GET_SET_NAME_DICT[per_group_name]

    if isinstance(per_degradation_type_list,
                  (list, tuple)) and len(per_degradation_type_list) > 0:
        return '_'.join([
            str(per_degradation_type).strip().lower()
            for per_degradation_type in per_degradation_type_list
        ])

    return ''


def get_set_name(per_group_name, per_degradation_type_list):
    """按(group_name, degradation_type_list)推导子集名(即图像编辑任务类型)

    本数据集17个组的任务类型全部可知，所以17个子集就是17种图像复原任务，
    一条都不会落进mix; 只有连group_name和degradation_type_list都拿不到、
    彻底找不到任务类型时才归到mix子集。
    """
    per_normalized_task_name = get_normalized_task_name(
        per_group_name, per_degradation_type_list)

    if not per_normalized_task_name:
        return MIX_SET_NAME

    return per_normalized_task_name


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    这个数据集每个编辑对只有编辑前退化图LQ这一张参考图，所有子集恒为1
    (上游标注里的reference_image_num也恒为1)，
    保留这个函数是为了和002/006/007的收尾自校验口径保持一致。
    """
    return 1


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令

    这个数据集的指令是上游009.1按"退化类型指令池 + md5哈希分配"生成的模板类指令，
    里面没有任何视觉参考图占位符(上游校验已禁掉[ ] * 等字符)，也没有
    "the reference image"这类自然语言指代(只有1张参考图、指令从不指代它)，
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


def process_single_annotation_file(annotation_file_pair):
    """解析单个标注文件，组装图像编辑对(参考图+编辑后图+编辑指令)的列表

    这一步只做纯文本层面的过滤(json坏行、缺字段、组名与退化标签不自洽、图不存在、
    保存名非法、指令为空或过短、指令过长、指令是坏占位符指令)，
    图像本身的解码校验和分辨率过滤留到后面多进程里做。

    图像是否存在这里用os.path.isfile逐个判，没有照007那样按目录缓存os.listdir:
    本数据集的图像只落在34个目录里(17组各GT/LQ两个)，但单个目录最多有218960个文件，
    缓存一个目录的文件名集合就要十几MB、一个worker最多缓存两个组也要几十MB，
    32个worker叠起来反而更亏; 而且每张图后面都要真解码一遍，
    真缺图在解码阶段一定会被判出来，这里的存在性判定只是为了把"缺图"和"图坏"
    分开统计。
    """

    per_annotation_path, per_group_name, root_image_path = annotation_file_pair

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
    missing_image_count, invalid_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    annotation_count_dict = {}
    edit_annotation_pair_list = []

    for per_annotation in annotation_list:

        per_annotation_group_name = per_annotation.get(
            ANNOTATION_GROUP_NAME_KEY_NAME, '')
        if not isinstance(per_annotation_group_name, str):
            per_annotation_group_name = ''
        per_annotation_group_name = per_annotation_group_name.strip()

        per_degradation_type_list = per_annotation.get(
            ANNOTATION_DEGRADATION_TYPE_LIST_KEY_NAME, [])
        if not isinstance(per_degradation_type_list, (list, tuple)):
            per_degradation_type_list = []
        per_degradation_type_list = [
            str(per_degradation_type).strip().lower()
            for per_degradation_type in per_degradation_type_list
            if str(per_degradation_type).strip()
        ]

        # 组名和退化类型决定子集名，缺任意一个都无法定位子集。
        # 行内组名还必须和这个标注文件所在的组目录一致: 不一致说明上游产物被搬动过，
        # 继续跑会把样本对写进错误子集
        if not per_annotation_group_name or len(
                per_degradation_type_list
        ) < 1 or per_annotation_group_name != per_group_name:
            missing_image_count += 1
            print('3333', per_annotation_path, per_annotation_group_name,
                  per_degradation_type_list)
            continue

        annotation_count_dict[
            per_annotation_group_name] = annotation_count_dict.get(
                per_annotation_group_name, 0) + 1

        per_set_name = get_set_name(per_annotation_group_name,
                                    per_degradation_type_list)

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

        # 保存图像名前缀用{数据集名(全小写)}_{原图名前缀}。
        # 原图名前缀就是7位样本主干(sample_key)，上游009已硬校验过它全局唯一
        # (17组主干区间首尾相接、恰好覆盖0000001~0920224、无缺号无重叠)，
        # 所以920223个保存名100%唯一，不需要像007那样再往名字里塞source和task。
        # 这里仍然从编辑后图像的文件名现取前缀(而不是直接用sample_key字段)，
        # 保证保存名和真正被读取的那张图严格对应，再和sample_key交叉比对一次
        per_edited_image_name_prefix = os.path.splitext(
            os.path.basename(
                per_edited_image_relative_path))[0].strip().lower()

        per_sample_key = per_annotation.get(ANNOTATION_SAMPLE_KEY_KEY_NAME, '')
        if not isinstance(per_sample_key, str):
            per_sample_key = ''
        per_sample_key = per_sample_key.strip()

        if not VALID_SAMPLE_KEY_PATTERN.match(
                per_edited_image_name_prefix
        ) or per_edited_image_name_prefix != per_sample_key:
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_edited_image_name_prefix,
                  per_sample_key)
            continue

        per_save_image_name_prefix = (f'{SAVE_IMAGE_NAME_DATASET_PREFIX}_'
                                      f'{per_edited_image_name_prefix}')
        per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
        # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名的前缀
        per_save_pair_folder_name = os.path.splitext(
            per_save_edited_image_name)[0]

        if not VALID_IMAGE_NAME_PATTERN.match(per_save_edited_image_name):
            invalid_save_image_name_count += 1
            print('3333', per_edited_image_path, per_save_edited_image_name)
            continue

        # 参考图顺序固定，第0张一定是编辑前的退化图LQ(这个数据集也只有这一张)
        per_reference_image_path_list = []
        per_save_reference_image_name_list = []
        per_missing_reference_image_count = 0
        per_invalid_save_reference_image_name_count = 0
        for per_reference_image_key_name in SAVE_REFERENCE_IMAGE_KEY_NAME_LIST:
            per_reference_image_relative_path = per_annotation.get(
                per_reference_image_key_name, None)
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
        # (本数据集只有1张参考图，这里只做防御性拦截)
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

        # 空指令、全空格指令、过短指令都视为不合格图像编辑对
        # (上游指令实测最短15字符，这里一条都不会被砍，只作兜底)
        if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
            invalid_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 本数据集的指令不需要任何占位符改写，这里只做strip，
        # 写进json的一定是归一化后的指令
        per_ti2i_caption = get_normalized_ti2i_caption(per_ti2i_caption)

        # 过长指令同样视为不合格图像编辑对，按归一化后的指令判定，
        # 和写进json的指令口径完全一致，收尾自校验直接量json里的长度就能复检
        # (上游指令实测最长105字符，这里一条都不会被砍，只作兜底)
        if len(per_ti2i_caption) > MAX_CAPTION_LENGTH:
            too_long_caption_count += 1
            print('3333', per_edited_image_path, len(per_ti2i_caption))
            continue

        # 占位符编号与参考图数量不自洽的指令也视为不合格图像编辑对(本数据集恒1张
        # 参考图，即指令里不允许出现任何占位符)，
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
        annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        missing_image_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
    ]


def get_all_annotation_file_pair(root_dataset_path):
    """收集上游全部标注文件，返回[标注文件任务列表, 每组标注文件数]

    上游标注按组分目录、组内按块号分文件(实测17组共464个jsonl)，
    这里按"标注文件"这一粒度出任务，正好能把多进程铺满，不需要像007那样先切分片。
    """
    root_image_path = os.path.join(root_dataset_path,
                                   *LOAD_IMAGE_DIR_NAME_LIST)

    load_annotation_dir_path = os.path.join(root_dataset_path,
                                            LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    group_annotation_file_count_dict = {}
    for per_group_name in sorted(os.listdir(load_annotation_dir_path)):
        per_group_dir_path = os.path.join(load_annotation_dir_path,
                                          per_group_name)
        if not os.path.isdir(per_group_dir_path):
            continue

        per_group_annotation_file_name_list = sorted([
            per_annotation_file_name
            for per_annotation_file_name in os.listdir(per_group_dir_path) if
            per_annotation_file_name.endswith(LOAD_ANNOTATION_FILE_NAME_SUFFIX)
        ])

        for per_annotation_file_name in per_group_annotation_file_name_list:
            annotation_file_pair_list.append([
                os.path.join(per_group_dir_path, per_annotation_file_name),
                per_group_name,
                root_image_path,
            ])

        group_annotation_file_count_dict[per_group_name] = len(
            per_group_annotation_file_name_list)

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list, group_annotation_file_count_dict


def get_all_edit_annotation_pair(root_dataset_path):
    """按标注文件粒度多进程组装全部图像编辑对的列表

    上游有464个标注文件、合计920223行，逐行还要判2张图像文件是否存在，
    所以这里按标注文件开多进程解析，最后按保存的编辑后图像名统一排序。
    """
    annotation_file_pair_list, group_annotation_file_count_dict = get_all_annotation_file_pair(
        root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list),
          'annotation group:', len(group_annotation_file_count_dict))

    total_annotation_count = 0
    illegal_line_count = 0
    missing_image_count, invalid_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    annotation_count_dict = {}
    edit_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])
            for per_annotation_count_key, per_annotation_count in per_load_result[
                    1].items():
                annotation_count_dict[
                    per_annotation_count_key] = annotation_count_dict.get(
                        per_annotation_count_key, 0) + per_annotation_count
            total_annotation_count += per_load_result[2]
            illegal_line_count += per_load_result[3]
            missing_image_count += per_load_result[4]
            invalid_caption_count += per_load_result[5]
            too_long_caption_count += per_load_result[6]
            invalid_placeholder_caption_count += per_load_result[7]
            invalid_save_image_name_count += per_load_result[8]

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        group_annotation_file_count_dict,
        annotation_count_dict,
        total_annotation_count,
        illegal_line_count,
        missing_image_count,
        invalid_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
    ]


def check_load_annotation_count(annotation_count_dict, total_annotation_count,
                                total_annotation_file_count,
                                edit_annotation_pair_list):
    """解析完标注后按组和子集两级硬对账，并检查保存图像名是否唯一

    上游标注是009/009.1两步跑出来的确定产物，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    子集级对账能额外拦住"某个组被映射进错误子集"这种组级对账看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对。
    """
    check_error_message_list = []

    set_annotation_count_dict = {}

    for per_annotation_count_key in sorted(annotation_count_dict.keys()):
        if per_annotation_count_key not in EXPECTED_ANNOTATION_COUNT_DICT:
            check_error_message_list.append(
                f'unknown group {per_annotation_count_key}')
            continue

        per_expect_annotation_count = EXPECTED_ANNOTATION_COUNT_DICT[
            per_annotation_count_key]
        if annotation_count_dict[
                per_annotation_count_key] != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_annotation_count_key} annotation count not match '
                f'{annotation_count_dict[per_annotation_count_key]} != '
                f'{per_expect_annotation_count}')

        # 这里必须走和落盘时完全同一条子集名推导路径，否则"组->子集"映射改了
        # 而对账表没改时，子集级对账会用另一套映射自说自话地对上
        per_set_name = GET_SET_NAME_DICT.get(per_annotation_count_key,
                                             MIX_SET_NAME)
        set_annotation_count_dict[per_set_name] = set_annotation_count_dict.get(
            per_set_name, 0) + annotation_count_dict[per_annotation_count_key]

    for per_annotation_count_key in sorted(
            EXPECTED_ANNOTATION_COUNT_DICT.keys()):
        if per_annotation_count_key not in annotation_count_dict:
            check_error_message_list.append(
                f'missing group {per_annotation_count_key}')

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

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

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
    """校验单个图像编辑对，并算出每张参考图应该被resize到的目标尺寸

    返回值是[状态字符串, 子集名, 编辑对]:
      'ok'                     -> 第三项是带上每张参考图目标尺寸的编辑对
      'invalid_image'          -> 有图解不开/不是RGB/短边或宽高比越界，第三项为None
      'different_aspect_ratio' -> reference_image[0]与编辑后图长宽比不同(整对丢弃)，
                                  第三项为None。带上子集名是为了在主流程里逐子集统计

    短边和宽高比的过滤按方案只以编辑后图像为准判定，参考图只要求能正常解码且
    mode在白名单里(实测抽样119480对的编辑后图像短边最小720、宽高比最大1.778，
    极端样本预计为0，这两个阈值只作兜底)。

    这里之所以能零额外IO地判长宽比: check_single_image本来就已经把编辑后图和
    每张参考图都真解码了一遍并返回了[宽, 高]，改造前只是把返回值丢掉了。
    现在接住这些shape，既能判长宽比、又能把"每张参考图的目标尺寸"一路带到落盘阶段，
    落盘时不必再重算一次。

    豁免子集(EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST)的目标尺寸一律给None，
    表示编辑后图与全部参考图都完全原样落盘。本数据集没有豁免子集。
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

    上游1840446张图是png/JPG/jpg混着的(GT与LQ的后缀可以不一样)，
    这里统一重编码成jpg，只换编码格式不换像素尺寸。
    编码参数显式用SAVE_IMAGE_JPEG_ENCODE_PARAM_LIST(质量97 + 色度4:4:4)，
    而不是cv2的默认值(质量95 + 色度4:2:0):
    本数据集是图像复原任务、编辑后图像是高清干净GT，而且有5个子集专门要模型
    去掉jpeg压缩伪影，GT自己不能带可见的块效应和色度模糊。
    实测(510对/1020张测质量档 + 340对/680张测色度采样)默认配置GT只有47.66dB，
    本配置到50.25dB、去jpeg那5个子集从44.22dB提到47.57dB，
    而且真正的瓶颈是色度下采样不是质量值(q100+4:2:0的色度保真还不如q95+4:4:4)，
    详见常量处的实测表。
    GT和参考图共用同一套参数，避免两条编码链路引入GT/LQ不对称的伪偏差。
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

    编辑后图**原分辨率落盘、不做任何缩放**，先落盘并拿到它的真实宽高;
    参考图再按check阶段算好的目标尺寸resize后落盘。
    非豁免子集的reference_image[0]的目标尺寸就是编辑后图尺寸，
    所以落盘后还会额外硬对账一次"reference_image[0]的shape == 编辑后图的shape"，
    不等说明resize链路有问题、整对丢弃。
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

    上游标注里还剩sample_key/group_name/degradation_type_list/task_type/
    reference_image_path_list/reference_image_num/reference_image_width/
    reference_image_height/same_size_flag/reference_image_file_size/
    target_image_file_size/ti2i_caption_source/instruction_style_reference/
    instruction_pool_index/instruction_pool_num/instruction_verb_style_index/
    instruction_sentence_pattern_index这些属性，按方案全部丢弃、不另存索引:
    sample_key只用来生成保存图像名(保存名前缀就是foundir_<sample_key>)，
    group_name和degradation_type_list只用来推子集名，
    task_type恒为image_restoration_edit(整个数据集就一个值、已由子集名体现)，
    reference_image_path_list/reference_image_num恒为1张参考图，
    reference_image_width/height和same_size_flag没有意义(实测920223对
    same_size_flag全为True、参考图与编辑后图同分辨率，json只记编辑后图像的宽高)，
    两个file_size是原始png/jpg的字节数、重编码成jpg后已失效，
    ti2i_caption_source/instruction_style_reference/instruction_pool_*/
    instruction_verb_style_index/instruction_sentence_pattern_index都是上游
    生成指令时的过程元信息、不是样本属性。
    上游的unzip_annotations(seed版标注的真子集)、seed_instruction_pool、
    seed_instruction_logs、unzip_check_result.json、
    seed_instruction_check_result.json也都不读不搬。
    ti2i_caption写的就是上游009.1分配的编辑指令(已strip)，本数据集恒1张参考图，
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
    json没记录的残留样本对文件夹。另外还要校验ti2i_caption里的占位符编号集合与
    reference_image这个list的长度必须自洽(本数据集恒1张参考图，即指令里不允许有
    任何占位符)。

    CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG为True时还会真解一次落盘后的
    reference_image[0]，硬校验它的真实shape严格等于json里的width/height
    (即等于编辑后图的宽高)。这一条是"第一张参考图与编辑后图尺寸必须一致"
    这个核心不变式的最终验收: 只对账json里的数字是查不出resize有没有真的生效的。
    豁免子集(完全原样落盘)会跳过这一条。
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
                if per_annotation['reference_image_num'] != len(
                        per_annotation['reference_image']):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} reference image num not match'
                    )
                # 本数据集所有子集都必须是单参考图
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
                # 长度自洽: 本数据集恒1张参考图，即不允许出现任何占位符
                if check_invalid_caption(
                        per_annotation['ti2i_caption'],
                        len(per_annotation['reference_image'])):
                    check_error_message_list.append(
                        f'{per_save_edited_image_name} caption placeholder index not match reference image num {len(per_annotation["reference_image"])}'
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
    所以本次改动之后这8个数据集必须落到全新的输出目录(或先手动删掉旧目录)重跑，
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

    edit_annotation_pair_list, total_annotation_file_count, group_annotation_file_count_dict, annotation_count_dict, total_annotation_count, illegal_line_count, missing_image_count, invalid_caption_count, too_long_caption_count, invalid_placeholder_caption_count, invalid_save_image_name_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          illegal_line_count, missing_image_count, invalid_caption_count,
          too_long_caption_count, invalid_placeholder_caption_count,
          invalid_save_image_name_count, len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    load_annotation_check_error_message_list = check_load_annotation_count(
        annotation_count_dict, total_annotation_count,
        total_annotation_file_count, edit_annotation_pair_list)
    if len(load_annotation_check_error_message_list) > 0:
        # 上游标注条数对不上说明上游009/009.1没跑完或产物被改动过，
        # 继续往下跑只会得到一个悄悄少样本的新数据集
        raise Exception(
            f'check load annotation count error num {len(load_annotation_check_error_message_list)} {load_annotation_check_error_message_list[:10]}'
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
          'total annotation:', total_annotation_count, 'illegal line:',
          illegal_line_count, 'missing image:', missing_image_count,
          'invalid caption:', invalid_caption_count, 'too long caption:',
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
        'total_annotation_file_count': total_annotation_file_count,
        'total_annotation_count': total_annotation_count,
        'illegal_line_count': illegal_line_count,
        'missing_image_count': missing_image_count,
        'invalid_caption_count': invalid_caption_count,
        'too_long_caption_count': too_long_caption_count,
        'invalid_placeholder_caption_count': invalid_placeholder_caption_count,
        'invalid_save_image_name_count': invalid_save_image_name_count,
        'invalid_image_count': invalid_image_count,
        # reference_image[0]与编辑后图长宽比不同而被整对丢弃的条数。
        # 本数据集上游GT与LQ同分辨率，这个数字预期为0;
        # 第一轮跑完后可以把实测值回填成EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT再上硬对账
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
        'group_annotation_file_count_dict': group_annotation_file_count_dict,
        'annotation_count_dict': annotation_count_dict,
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/FoundIR'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
