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

DATASET_NAME = 'bm_6m'

SAVE_DATASET_DIR_NAME = 'BM-6M'

# ==============================================================================
# 【这个数据集为什么能出ti2i】
# BM-6M(ByteMorph-6M)官方README原话: "we present ByteMorph, a substantial benchmark
# specifically created for instruction-based image editing focused on non-rigid
# motions"，即它就是一个**基于指令的图像编辑**数据集，专攻非刚性运动
# (镜头移动/物体形变/人体articulation/复杂交互)。
# 上游015解压脚本已经把每个tar里的1024x512拼接图无损拆成了左右两张512x512单图:
#   unzip_images/<subset>/<kind>/<batch_i>/<sample_key>_reference.png  参考图(编辑前帧)
#   unzip_images/<subset>/<kind>/<batch_i>/<sample_key>_edited.png     编辑后图(编辑后帧)
# 每行标注固定是"1张参考图 + 1条祈使句英文编辑指令 + 1张编辑后图"，
# 没有第二张视觉条件图、没有mask，所以reference_image恒为长度1的list、
# reference_image_num恒为1。
# 这个数据集**同时也能出t2i**(每帧都带一条详细整图caption)，见同目录的
# 014.resave_bm_6m_t2i_dataset.py。
#
# 【上游015解压产物实测规格(全量扫过2120个jsonl、5878678行，非抽样)】
# BM-6M/
# ├── unzip_images/<subset-N>/<kind>/<batch_i>/<sample_key>_reference.png
# │                                            <sample_key>_edited.png  1175.7万张
# ├── unzip_annotations/<subset-N>/<kind>/<batch_i>.jsonl  2120个(15G)，共5878678行
# └── unzip_check_missing_images.json  上游对账报告(check_error_count=0、
#                                      missing/orphan/invalid_concat/save_image_error全0)
# 本脚本只读这2120个jsonl定位样本对，**绝不os.walk图像目录**:
# 上游解出1175.7万个小文件，扫一遍目录树在NAS上不可接受。
#
# 【实测关键规格】
# - 每行恒为37个字段(2120个分片keyset 100%一致、0缺字段);
# - 所有图像**恒为512x512 RGB**((edited_w,h,ref_w,h)全量只有(512,512,512,512)
#   这一种组合)，所以短边过滤与宽高比过滤一条都不会命中，只作兜底;
# - edit_instruction全量非空、edit_instruction_key_name恒为edit_rewrite
#   (0条回退到edit)，长度min 10 / p50约103 / p99约202 / max 562;
# - reference_image_num恒为1、reference_image_path_list长度恒为1;
# - 落盘图像名与sample_key严格对应(basename == <sample_key>_reference.png /
#   <sample_key>_edited.png，全量0例不符);
# - 行内subset_name/frame_kind_name/archive_name与所在目录100%一致(全量0例不符)。
#
# 【两个kind的关系(已用像素md5交叉验证20个batch/95组，100%命中)】
# 同一个batch内，frames的1个对与multi的3个对来自**同一段4帧视频**:
#   sampled_frames      : (帧0 | 帧3)
#   sampled_multi_frames: (帧0 | 帧1) (帧1 | 帧2) (帧2 | 帧3)
# 且 frames的帧0 == multi frame_0_1的帧0、frames的帧3 == multi frame_2_3的帧3、
# 帧1/帧2在相邻两对之间字节完全相同。
# 也就是说587万对落盘的1175.7万张图，去重后只有6068568个唯一帧(重复约1.94倍)。
# 按方案确认**两个kind全要**(上游注释也明确"两套不同粒度的编辑对都是有效样本对")，
# 代价是同一张帧会以不同的样本对名字重复落盘。
#
# 【848组跨batch/subset重复的sample_key(必须处理，否则静默覆盖)】
# 全量统计: 848个sample_key各出现2次(frames 212组 + multi 636组)，
# 分布为subset-1内部84、subset-1↔2 560、subset-1↔3 84、subset-2内部60、
# subset-2↔3 60。已核对: 848组的**原始key完全相同**(不是大小写差异)，
# 但抽查30组的图像md5与编辑指令**全部不同**(847/848组指令不同)，
# 说明是上游对同一视频片段重复采样出的两个不同样本对。
# 按方案确认**涉及重复sample_key的样本全部过滤掉不保留**(1696行整体丢弃)，
# 所以本脚本必须先全量扫一遍所有jsonl统计sample_key出现次数、再做第二遍解析。
#
# 【子集划分】
# 标注里没有任何细分编辑任务类型字段(dataset_task_type恒为image_edit;
# subset-1..9是官方纯分片、不是任务; frame_kind是采样粒度、不是任务)，
# 但README明确整个数据集就是一种任务: 非刚性运动编辑。
# 所以按方案确认只有一个子集 non_rigid_motions(不是mix)。
# ==============================================================================

# 上游015解压脚本按tar另存的汇总标注，每行一个完整样本对
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_NAME_SUFFIX = '.jsonl'

# 标注里的图像路径已经是相对上游数据集根目录的完整相对路径
# (形如unzip_images/subset-1/sampled_frames/batch_0/xxx_reference.png)，
# 不需要再拼images子目录
LOAD_IMAGE_DIR_NAME_LIST = []

# 过滤掉无用信息后unzip_annotations下只应该有这9个子集目录
LOAD_SUBSET_DIR_NAME_LIST = [
    'subset-1',
    'subset-2',
    'subset-3',
    'subset-4',
    'subset-5',
    'subset-6',
    'subset-7',
    'subset-8',
    'subset-9',
]

SAMPLED_FRAMES_DIR_NAME = 'sampled_frames'

SAMPLED_MULTI_FRAMES_DIR_NAME = 'sampled_multi_frames'

# 每个子集目录下有且只有这2个kind目录，两个kind的样本对全部保留
LOAD_FRAME_KIND_DIR_NAME_LIST = [
    SAMPLED_FRAMES_DIR_NAME,
    SAMPLED_MULTI_FRAMES_DIR_NAME,
]

# 样本id，实测格式为<video_id>_<clip>_<sub_clip>_<global>_<extra>_seed<seed>_<pair>
# (frames)或..._frame_<f0>_<f1>(multi)，含大小写(video_id是大小写敏感的youtube id)。
# 全量核对: 转小写后唯一key数5877830 = 5878678 - 848，**转小写没有引入任何新撞名**，
# 所以可以安全地把它全小写后拼进保存图像名
ANNOTATION_SAMPLE_KEY_KEY_NAME = 'sample_key'

# 行内记录的子集名/kind名/tar名，必须与该标注文件所在的目录严格一致，
# 不一致说明上游产物被搬动过，继续跑会把样本对写进错误位置(实测全量0例不符)
ANNOTATION_SUBSET_NAME_KEY_NAME = 'subset_name'

ANNOTATION_FRAME_KIND_NAME_KEY_NAME = 'frame_kind_name'

ANNOTATION_ARCHIVE_NAME_KEY_NAME = 'archive_name'

# 编辑指令(祈使句)，上游取自原始json的edit_rewrite字段。
# README定义它是instruction-based image editing的编辑指令，是本数据集唯一的训练主文本。
# 实测5878678行全非空、全英文、不含任何[Vn*]占位符
ANNOTATION_CAPTION_KEY_NAME = 'edit_instruction'

# 参考图(编辑前帧)路径列表，实测长度恒为1
ANNOTATION_REFERENCE_IMAGE_PATH_LIST_KEY_NAME = 'reference_image_path_list'

ANNOTATION_REFERENCE_IMAGE_NUM_KEY_NAME = 'reference_image_num'

# 编辑后图(编辑后帧)路径
ANNOTATION_EDITED_IMAGE_PATH_KEY_NAME = 'edited_image_path'

# 上游落盘图像名的两个固定后缀，用于反查"这张图是否真的属于这个sample_key"
LOAD_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.png'

LOAD_EDITED_IMAGE_NAME_SUFFIX = '_edited.png'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 唯一的子集名(即图像编辑任务类型): 非刚性运动编辑。
# 取自官方README对整个数据集的定义("instruction-based image editing focused on
# non-rigid motions")，而不是mix: 任务类型是明确可知的，只是没有更细的分类字段。
# subset-1..9只是官方按下载分片切的目录、sampled_frames/sampled_multi_frames只是
# 抽帧粒度，两者都不是编辑任务类型，所以都不用来分子集
SAVE_SET_NAME = 'non_rigid_motions'

SAVE_SET_NAME_LIST = [
    SAVE_SET_NAME,
]

# 新标注固定只存这七个key，多一个少一个都在收尾自校验里报错。
# 上游jsonl里剩下的属性按方案确认全部丢弃、不另存索引(落盘后**永久丢失**,
# 想恢复只能回 huggingface_datasets_unzip/BM-6M 重跑):
#   edit_description        : 陈述句版"发生了什么变化"的描述(全非空，长度19~362),
#                             可做指令改写增强，丢弃
#   reference_image_caption : 参考图(编辑前帧)的详细整图caption(长度0~2484，1条为空),
#                             由014的t2i脚本承载，本脚本丢弃
#   edited_image_caption    : 编辑后图的详细整图caption(长度5~3791),
#                             由014的t2i脚本承载，本脚本丢弃
#   video_caption           : **视频级**运动caption，同一段视频抽出来的4个帧对
#                             共享同一条(实测frames的1条与multi的3条完全相同;44条为空),
#                             粒度比本帧对粗，丢弃
#   video_id / clip_index / sub_clip_index / global_index / extra_index /
#   seed_index / seed_name / start_frame_index / end_frame_index /
#   frame_pair_index        : 溯源与帧序元信息。**丢掉后无法再按视频划分train/val
#                             防同视频泄漏、也无法再做同视频去重**，按方案确认丢弃
#   frame_kind_name         : 大位移(frames) vs 小位移(multi)的抽帧粒度，
#                             两个kind合并进同一个子集后这个信息消失(但仍编码在
#                             保存图像名里，可从名字里反解)
#   subset_name / archive_name / source_archive_relative_path /
#   source_concat_image_member_name / source_annotation_member_name /
#   source_concat_image_width / source_concat_image_height /
#   source_concat_image_layout / image_file_save_flag / annotation_file_path /
#   annotation_file_save_flag / dataset_task_type / edit_instruction_key_name /
#   reference_image_width / reference_image_height / reference_image_num
#                           : 溯源与冗余信息，前几项已内含在保存图像名里
#   "frames对与multi三对同属一段4帧视频"的时序链关系(帧0->帧1->帧2->帧3)
#                           : 落盘后彻底丢失，无法再做多帧/视频式训练
#   数据集授权cc0-1.0        : 记录在上游unzip_check_missing_images.json里，新格式不带
SAVE_ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

# 本数据集每个编辑对只有"编辑前帧"这一张参考图，没有第二张视觉条件图，
# 所以reference_image恒为长度1的list、reference_image_num恒为1
EXPECT_REFERENCE_IMAGE_NUM = 1

# 保存图像名里只允许小写字母/数字/下划线/中划线/点，与001~012完全一致。
# 实测587.7万个保存名100%满足这个模式、0个非法名，最长119字符
# (bm_6m_non_rigid_motions_subset_9_sampled_multi_frames_batch_195_
#  <sample_key>_edited.jpg)，远低于文件系统单文件名255字节的上限
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# sample_key小写后必须只含小写字母/数字/下划线/中划线(实测全量0例不符),
# 不满足的样本没法保证保存名合法，整对丢弃并上报
VALID_SAMPLE_KEY_PATTERN = re.compile(r'^[a-z0-9_\-]+$')

# 只保留RGB三通道图，灰度图/P图/RGBA图/CMYK图等一律过滤掉，
# 编辑后图像和所有参考图都必须是RGB，任意一张不合格则整个图像编辑对丢弃。
# 上游015已硬校验过拼接图colortype恒为2(truecolor RGB)，所以这里一条都不会命中
VALID_IMAGE_MODE_LIST = [
    'RGB',
]

# 每个<subset>/<kind>下的实测jsonl分片数(与上游015的tar数一一对应)，合计2120个。
# 注意subset-4/sampled_multi_frames只有23个(上游huggingface仓库自身就是稀疏编号，
# 不是本地缺失，上游015已对账过)，所以subset-4的132个batch只有frames侧
EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT = {
    'subset-1/sampled_frames': 88,
    'subset-1/sampled_multi_frames': 88,
    'subset-2/sampled_frames': 88,
    'subset-2/sampled_multi_frames': 88,
    'subset-3/sampled_frames': 84,
    'subset-3/sampled_multi_frames': 84,
    'subset-4/sampled_frames': 155,
    'subset-4/sampled_multi_frames': 23,
    'subset-5/sampled_frames': 117,
    'subset-5/sampled_multi_frames': 117,
    'subset-6/sampled_frames': 136,
    'subset-6/sampled_multi_frames': 136,
    'subset-7/sampled_frames': 187,
    'subset-7/sampled_multi_frames': 187,
    'subset-8/sampled_frames': 75,
    'subset-8/sampled_multi_frames': 75,
    'subset-9/sampled_frames': 196,
    'subset-9/sampled_multi_frames': 196,
}

EXPECTED_TOTAL_ANNOTATION_FILE_COUNT = 2120

# 每个<subset>/<kind>下的实测标注行数(全量扫描，非抽样)，合计5878678行,
# 与上游unzip_check_missing_images.json里的total_valid_sample_pair_count一致。
# 少一行都说明上游015没跑完或产物被改动过，这时候继续跑只会得到一个悄悄少样本的
# 新数据集，必须直接报错
EXPECTED_SUBSET_KIND_ANNOTATION_COUNT_DICT = {
    'subset-1/sampled_frames': 126015,
    'subset-1/sampled_multi_frames': 378045,
    'subset-2/sampled_frames': 126094,
    'subset-2/sampled_multi_frames': 378282,
    'subset-3/sampled_frames': 120537,
    'subset-3/sampled_multi_frames': 361611,
    'subset-4/sampled_frames': 222144,
    'subset-4/sampled_multi_frames': 96762,
    'subset-5/sampled_frames': 167204,
    'subset-5/sampled_multi_frames': 501612,
    'subset-6/sampled_frames': 195077,
    'subset-6/sampled_multi_frames': 585231,
    'subset-7/sampled_frames': 266960,
    'subset-7/sampled_multi_frames': 800880,
    'subset-8/sampled_frames': 106954,
    'subset-8/sampled_multi_frames': 320862,
    'subset-9/sampled_frames': 281102,
    'subset-9/sampled_multi_frames': 843306,
}

EXPECTED_TOTAL_ANNOTATION_COUNT = 5878678

# 跨batch/subset重复的sample_key: 实测848个key各出现2次，
# 按方案确认整组丢弃(1696行)，解析阶段逐项硬对账
EXPECTED_DUPLICATE_SAMPLE_KEY_COUNT = 848

EXPECTED_DUPLICATE_SAMPLE_KEY_ANNOTATION_COUNT = 1696

# 文本层各类不合格指令的实测精确条数(已排除上面那1696行)，解析阶段逐项硬对账。
# 只有1条562字符的超长指令会被砍掉，其余全部为0
EXPECTED_INVALID_CAPTION_COUNT_DICT = {
    'empty_caption_count': 0,
    'null_like_caption_count': 0,
    'no_word_char_caption_count': 0,
    'too_short_caption_count': 0,
    'too_long_caption_count': 1,
    'invalid_placeholder_caption_count': 0,
}

# 每个<subset>/<kind>过滤后的实测合格样本对数，合计5876981对，解析阶段逐项硬对账。
# 这一级对账能额外拦住"重复key集合算错"或"某个分片被漏读"这种总数对账看不出来的问题
EXPECTED_SUBSET_KIND_VALID_ANNOTATION_COUNT_DICT = {
    'subset-1/sampled_frames': 125812,
    'subset-1/sampled_multi_frames': 377436,
    'subset-2/sampled_frames': 125909,
    'subset-2/sampled_multi_frames': 377727,
    'subset-3/sampled_frames': 120501,
    'subset-3/sampled_multi_frames': 361503,
    'subset-4/sampled_frames': 222144,
    'subset-4/sampled_multi_frames': 96762,
    'subset-5/sampled_frames': 167204,
    'subset-5/sampled_multi_frames': 501612,
    'subset-6/sampled_frames': 195077,
    'subset-6/sampled_multi_frames': 585231,
    'subset-7/sampled_frames': 266960,
    'subset-7/sampled_multi_frames': 800880,
    'subset-8/sampled_frames': 106954,
    'subset-8/sampled_multi_frames': 320862,
    'subset-9/sampled_frames': 281102,
    'subset-9/sampled_multi_frames': 843305,
}

EXPECTED_VALID_ANNOTATION_COUNT = 5876981

# 唯一子集的实测条数，与EXPECTED_VALID_ANNOTATION_COUNT相同但分开写死:
# 子集级对账能额外拦住"子集名推导被改动"这种总数对账看不出来的问题
EXPECTED_SAVE_SET_ANNOTATION_COUNT_DICT = {
    SAVE_SET_NAME: 5876981,
}

EXPECTED_SAVE_SET_COUNT = 1

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

# 实测指令长度min 10 / p50约103 / p90约153 / p99约202 / max 562，
# 下限10一条都不会砍掉(最短的那条正好是10)，只作兜底
MIN_CAPTION_LENGTH = 10

# 实测超过200的有96734条(1.6%)、超过256的5830条、超过300的572条、
# 超过384的11条、超过512的只有1条。
# 砍200/256会砍掉大量正常的长指令(这个数据集的指令要同时描述物体运动与镜头运动,
# 天然比anyedit那种短指令长)，按方案确认取512，只丢1条
MAX_CAPTION_LENGTH = 512

# 判定"指令里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)，
# 只剩标点的指令没有任何可训练的语义，整对丢弃(实测0条)
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# 判定null字面量之前先剥掉两端的标点和空白，这样"None."与"None"能命中同一条规则
CAPTION_STRIP_CHAR = '.。!！?？,，;；:：、"\'“”‘’()（） \t\r\n'

# 无意义指令黑名单(小写化并剥掉两端标点后做全串精确匹配)，与006/007口径一致。
# 本数据集实测0条命中，只做防御性拦截
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
# 本数据集恒为单参考图，即要求指令里完全没有占位符(实测0条含[Vn*]),
# 这里只做防御性拦截
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数，本数据集用不到，只做防御性拦截
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2

MAX_SAVE_MESSAGE_NUM = 10000


def get_set_name():
    """本数据集唯一的子集名(即图像编辑任务类型): 非刚性运动编辑

    任务类型来自官方README对整个数据集的定义，是明确可知的，所以不用mix兜底。
    写成函数而不是直接用常量，是为了和002/006/007的写法保持一致。
    """
    return SAVE_SET_NAME


def get_expect_reference_image_num(per_set_name):
    """按子集名推导这个子集每个图像编辑对应有的参考图数量

    这个数据集每个编辑对只有"编辑前帧"这一张参考图(上游标注里的
    reference_image_num也恒为1)，所以恒为1。
    保留这个函数是为了和002/006/007的收尾自校验口径保持一致。
    """
    return EXPECT_REFERENCE_IMAGE_NUM


def get_normalized_ti2i_caption(per_ti2i_caption):
    """归一化编辑指令

    这个数据集的指令是上游VLM改写出来的英文祈使句(edit_rewrite)，
    里面没有任何视觉参考图占位符、也没有"the reference image"这类自然语言指代
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
    本数据集实测0条命中，只做防御性拦截。
    """
    per_ti2i_caption = str(per_ti2i_caption).strip().lower().strip(
        CAPTION_STRIP_CHAR)

    return per_ti2i_caption in NULL_LIKE_CAPTION_LIST


def get_all_annotation_file_pair(root_dataset_path):
    """收集上游全部标注文件，返回[标注文件任务列表, 每个<subset>/<kind>的标注文件数]

    上游标注按子集分目录、子集内按kind分目录、kind内按tar分文件
    (实测9个子集 * 2个kind = 18组，合计2120个jsonl)，
    这里按"标注文件"这一粒度出任务，正好能把多进程铺满。
    子集名与kind名由主进程按目录名算好后带给worker，worker会再和行内字段交叉核对。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    subset_kind_annotation_file_count_dict = {}
    for per_subset_name in LOAD_SUBSET_DIR_NAME_LIST:
        for per_frame_kind_name in LOAD_FRAME_KIND_DIR_NAME_LIST:
            per_subset_kind_key = f'{per_subset_name}/{per_frame_kind_name}'
            per_subset_kind_path = os.path.join(root_annotation_path,
                                                per_subset_name,
                                                per_frame_kind_name)
            if not os.path.isdir(per_subset_kind_path):
                print('2222', per_subset_kind_path)
                subset_kind_annotation_file_count_dict[per_subset_kind_key] = 0
                continue

            per_annotation_file_name_list = sorted([
                per_annotation_file_name for per_annotation_file_name in
                os.listdir(per_subset_kind_path)
                if per_annotation_file_name.endswith(
                    LOAD_ANNOTATION_FILE_NAME_SUFFIX)
            ])

            for per_annotation_file_name in per_annotation_file_name_list:
                per_archive_name = per_annotation_file_name[:-len(
                    LOAD_ANNOTATION_FILE_NAME_SUFFIX)]
                annotation_file_pair_list.append([
                    os.path.join(per_subset_kind_path,
                                 per_annotation_file_name),
                    per_subset_name,
                    per_frame_kind_name,
                    per_archive_name,
                ])

            subset_kind_annotation_file_count_dict[per_subset_kind_key] = len(
                per_annotation_file_name_list)

    annotation_file_pair_list = sorted(annotation_file_pair_list,
                                       key=lambda x: x[0])

    return annotation_file_pair_list, subset_kind_annotation_file_count_dict


def process_single_annotation_file_sample_key(annotation_file_pair):
    """第一遍扫描: 只把单个标注文件里的全部sample_key读出来

    必须先全量扫一遍才能知道哪些sample_key跨batch/subset重复:
    实测848个key各出现2次(frames 212组 + multi 636组)，
    它们的图像像素与编辑指令都不同(是上游对同一视频片段重复采样出的两个样本对),
    按方案确认涉及重复key的样本全部丢弃。
    只返回key字符串(约5878678个)，不返回整行，内存与IPC开销都可控。
    """
    per_annotation_path, _, _, _ = annotation_file_pair

    sample_key_list = []
    try:
        with open(per_annotation_path, 'r',
                  encoding='UTF-8') as load_jsonl_file:
            for per_line in load_jsonl_file:
                per_line = per_line.strip()
                if not per_line:
                    continue

                try:
                    per_annotation = json.loads(per_line)
                except Exception:
                    continue

                if not isinstance(per_annotation, dict):
                    continue

                per_sample_key = per_annotation.get(
                    ANNOTATION_SAMPLE_KEY_KEY_NAME, '')
                if not isinstance(per_sample_key, str):
                    continue

                per_sample_key = per_sample_key.strip()
                if not per_sample_key:
                    continue

                sample_key_list.append(per_sample_key)
    except Exception as e:
        print('2222', per_annotation_path, e)

    return sample_key_list


def get_duplicate_sample_key_dict(annotation_file_pair_list):
    """全量统计sample_key出现次数，返回出现超过1次的key集合

    实测2120个jsonl共5878678行、其中848个key各出现2次。
    这个集合只有848个元素，可以直接随partial带给第二遍解析的worker。
    """
    sample_key_count_dict = {}
    total_sample_key_count = 0
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_sample_key_list in tqdm(pool.imap_unordered(
                process_single_annotation_file_sample_key,
                annotation_file_pair_list),
                                        total=len(annotation_file_pair_list)):
            total_sample_key_count += len(per_sample_key_list)
            for per_sample_key in per_sample_key_list:
                sample_key_count_dict[
                    per_sample_key] = sample_key_count_dict.get(
                        per_sample_key, 0) + 1

    duplicate_sample_key_dict = {
        per_sample_key: per_sample_key_count
        for per_sample_key, per_sample_key_count in
        sample_key_count_dict.items() if per_sample_key_count > 1
    }

    duplicate_sample_key_annotation_count = sum(
        duplicate_sample_key_dict.values())

    return [
        duplicate_sample_key_dict,
        total_sample_key_count,
        len(sample_key_count_dict),
        duplicate_sample_key_annotation_count,
    ]


def process_single_annotation_file(annotation_file_pair,
                                   duplicate_sample_key_dict,
                                   root_dataset_path):
    """解析单个标注文件，组装图像编辑对(参考图+编辑后图+编辑指令)的列表

    这一步只做纯文本层面的过滤(json坏行、缺字段、行内字段与目录不自洽、
    sample_key重复、sample_key非法、图像路径与sample_key不对应、图不存在、
    指令为空、指令是null字面量、指令没有任何文字字符、指令过短、指令过长、
    指令是坏占位符指令、保存名非法)，
    图像本身的解码校验和分辨率过滤留到后面多进程里做。

    图像是否存在这里按目录缓存一次os.listdir的结果、之后只做集合查表:
    每个标注文件里的图像全部落在同一个batch目录下
    (unzip_images/<subset>/<kind>/<batch_i>/，实测每个目录2880或8640个文件),
    所以一个worker只需要对那个目录listdir一次。
    如果按样本逐个os.path.exists，光这一步就是约1175万次网络往返。
    """
    per_annotation_path, per_subset_name, per_frame_kind_name, per_archive_name = annotation_file_pair

    per_subset_kind_key = f'{per_subset_name}/{per_frame_kind_name}'

    total_annotation_count, load_annotation_failed_count = 0, 0
    field_not_match_count, duplicate_sample_key_count = 0, 0
    invalid_sample_key_count, missing_image_count = 0, 0
    image_name_not_match_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    # 每个worker只处理一个标注文件，缓存里通常只有一个batch目录，内存开销可忽略
    dir_file_name_cache_dict = {}

    per_set_name = get_set_name()
    per_expect_reference_image_num = get_expect_reference_image_num(
        per_set_name)

    try:
        load_jsonl_file = open(per_annotation_path, 'r', encoding='UTF-8')
    except Exception as e:
        print('2222', per_annotation_path, e)

        return [
            edit_annotation_pair_list,
            per_subset_kind_key,
            total_annotation_count,
            1,
            field_not_match_count,
            duplicate_sample_key_count,
            invalid_sample_key_count,
            missing_image_count,
            image_name_not_match_count,
            empty_caption_count,
            null_like_caption_count,
            no_word_char_caption_count,
            too_short_caption_count,
            too_long_caption_count,
            invalid_placeholder_caption_count,
            invalid_save_image_name_count,
            set_annotation_count_dict,
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
                print('2222', per_annotation_path, e)
                continue

            if not isinstance(per_annotation, dict):
                load_annotation_failed_count += 1
                print('2222', per_annotation_path, 'annotation not a dict')
                continue

            per_sample_key = per_annotation.get(ANNOTATION_SAMPLE_KEY_KEY_NAME,
                                                '')
            if not isinstance(per_sample_key, str):
                per_sample_key = ''
            per_sample_key = per_sample_key.strip()

            if not per_sample_key:
                invalid_sample_key_count += 1
                print('2222', per_annotation_path, 'empty sample key')
                continue

            # 行内记录的子集名/kind名/tar名必须和这个标注文件所在的目录一致:
            # 不一致说明上游产物被搬动过，继续跑会把样本对写进错误位置(实测0例)
            if per_annotation.get(
                    ANNOTATION_SUBSET_NAME_KEY_NAME,
                    '') != per_subset_name or per_annotation.get(
                        ANNOTATION_FRAME_KIND_NAME_KEY_NAME,
                        '') != per_frame_kind_name or per_annotation.get(
                            ANNOTATION_ARCHIVE_NAME_KEY_NAME,
                            '') != per_archive_name:
                field_not_match_count += 1
                print('2222', per_annotation_path, per_sample_key)
                continue

            # 跨batch/subset重复的sample_key整组丢弃(实测848组、1696行):
            # 它们的像素与指令都不同，是两个不同的有效样本对，
            # 但按方案确认这类样本全部不保留
            if per_sample_key in duplicate_sample_key_dict:
                duplicate_sample_key_count += 1
                continue

            per_sample_key_lower = per_sample_key.lower()
            if not VALID_SAMPLE_KEY_PATTERN.match(per_sample_key_lower):
                invalid_sample_key_count += 1
                print('2222', per_annotation_path, per_sample_key)
                continue

            per_edited_image_relative_path = per_annotation.get(
                ANNOTATION_EDITED_IMAGE_PATH_KEY_NAME, '')
            if not isinstance(per_edited_image_relative_path, str):
                per_edited_image_relative_path = ''
            per_edited_image_relative_path = per_edited_image_relative_path.replace(
                '\\', '/').strip().lstrip('/')

            per_reference_image_relative_path_list = per_annotation.get(
                ANNOTATION_REFERENCE_IMAGE_PATH_LIST_KEY_NAME, [])
            if not isinstance(per_reference_image_relative_path_list,
                              (list, tuple)):
                per_reference_image_relative_path_list = []
            per_reference_image_relative_path_list = [
                str(per_reference_image_relative_path).replace(
                    '\\', '/').strip().lstrip('/')
                for per_reference_image_relative_path in
                per_reference_image_relative_path_list
                if isinstance(per_reference_image_relative_path, str)
                and per_reference_image_relative_path.strip()
            ]

            # 参考图数量必须与本子集口径一致(恒为1)，缺图或多图都说明上游规格变了
            if not per_edited_image_relative_path or len(
                    per_reference_image_relative_path_list
            ) != per_expect_reference_image_num:
                missing_image_count += 1
                print('2222', per_annotation_path, per_sample_key)
                continue

            # 上游标注里另存的reference_image_num只做交叉核对，一律以list长度为准
            if per_annotation.get(ANNOTATION_REFERENCE_IMAGE_NUM_KEY_NAME,
                                  per_expect_reference_image_num
                                  ) != per_expect_reference_image_num:
                missing_image_count += 1
                print('2222', per_annotation_path, per_sample_key)
                continue

            # 图像文件名必须与sample_key严格对应，否则说明上游成员错位，
            # 这种样本会让保存名指向另一张图(实测全量0例不符)
            if os.path.basename(
                    per_edited_image_relative_path
            ) != f'{per_sample_key}{LOAD_EDITED_IMAGE_NAME_SUFFIX}':
                image_name_not_match_count += 1
                print('2222', per_annotation_path,
                      per_edited_image_relative_path)
                continue

            if os.path.basename(
                    per_reference_image_relative_path_list[0]
            ) != f'{per_sample_key}{LOAD_REFERENCE_IMAGE_NAME_SUFFIX}':
                image_name_not_match_count += 1
                print('2222', per_annotation_path,
                      per_reference_image_relative_path_list[0])
                continue

            per_edited_image_path = os.path.join(
                root_dataset_path, *LOAD_IMAGE_DIR_NAME_LIST,
                per_edited_image_relative_path)
            per_reference_image_path_list = [
                os.path.join(root_dataset_path, *LOAD_IMAGE_DIR_NAME_LIST,
                             per_reference_image_relative_path)
                for per_reference_image_relative_path in
                per_reference_image_relative_path_list
            ]

            # 编辑后图和参考图缺任意一张，这个编辑对的信息都不完整，整对丢弃
            per_missing_image_flag = not check_image_file_exists(
                per_edited_image_path, dir_file_name_cache_dict)
            for per_reference_image_path in per_reference_image_path_list:
                if not check_image_file_exists(per_reference_image_path,
                                               dir_file_name_cache_dict):
                    per_missing_image_flag = True

            if per_missing_image_flag:
                missing_image_count += 1
                continue

            per_ti2i_caption = per_annotation.get(ANNOTATION_CAPTION_KEY_NAME,
                                                  '')
            # 上游指令固定是str，这里兼容list和str两种形式
            if isinstance(per_ti2i_caption, (list, tuple)):
                per_ti2i_caption = per_ti2i_caption[0] if len(
                    per_ti2i_caption) > 0 else ''
            if not isinstance(per_ti2i_caption, str):
                per_ti2i_caption = ''
            per_ti2i_caption = per_ti2i_caption.strip()

            # 空指令、全空格指令视为不合格图像编辑对(实测0条)
            if not per_ti2i_caption:
                empty_caption_count += 1
                continue

            # null字面量与"不做任何修改"这类无意义指令同样丢弃(实测0条)
            if check_null_like_caption(per_ti2i_caption):
                null_like_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:50])
                continue

            # 只剩标点、没有任何数字/字母/汉字的指令也丢弃(实测0条)
            if not CAPTION_WORD_CHAR_PATTERN.search(per_ti2i_caption):
                no_word_char_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:50])
                continue

            # 过短指令视为不合格图像编辑对(实测最短就是10，0条被砍)
            if len(per_ti2i_caption) < MIN_CAPTION_LENGTH:
                too_short_caption_count += 1
                print('3333', per_edited_image_path, len(per_ti2i_caption))
                continue

            # 本数据集的指令不需要任何占位符改写，这里只做strip，
            # 写进json的一定是归一化后的指令
            per_ti2i_caption = get_normalized_ti2i_caption(per_ti2i_caption)

            # 过长指令同样视为不合格图像编辑对，按归一化后的指令判定，
            # 和写进json的指令口径完全一致(实测只有1条562字符的会被砍掉)
            if len(per_ti2i_caption) > MAX_CAPTION_LENGTH:
                too_long_caption_count += 1
                print('3333', per_edited_image_path, len(per_ti2i_caption))
                continue

            # 占位符编号与参考图数量不自洽的指令也丢弃(本数据集单参考图，
            # 即要求指令里完全没有[Vn*]占位符，实测0条命中)
            if check_invalid_caption(per_ti2i_caption,
                                     per_expect_reference_image_num):
                invalid_placeholder_caption_count += 1
                print('3333', per_edited_image_path, per_ti2i_caption[:100])
                continue

            # 保存图像名必须拼出全局唯一键: sample_key只在同一个batch内唯一,
            # 跨batch/subset有848个重复(那848组已在上面整组丢弃)，
            # 但为了让名字自带溯源信息、也为了防止上游之后新增重复,
            # 这里仍然拼上子集名 + 官方分片名 + kind名 + tar名 + sample_key(全小写)。
            # 实测587.7万个保存名100%唯一、最长119字符
            per_save_image_name_prefix = (
                f'{DATASET_NAME}_{per_set_name}_'
                f'{per_subset_name.replace("-", "_")}_{per_frame_kind_name}_'
                f'{per_archive_name}_{per_sample_key_lower}')
            per_save_edited_image_name = f'{per_save_image_name_prefix}{SAVE_EDITED_IMAGE_NAME_SUFFIX}'
            per_save_reference_image_name_list = [
                f'{per_save_image_name_prefix}{SAVE_REFERENCE_IMAGE_NAME_SUFFIX}',
            ]

            # 保存名里出现路径分隔符或其它异常字符会写坏目录结构，整对丢弃(实测0条)
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
                print('3333', per_edited_image_path,
                      per_save_edited_image_name)
                continue

            # 每个图像编辑对独占一个文件夹，文件夹名就是编辑后图像名去掉.jpg后缀的前缀
            # (即带_edited那一段)，和002/006/007的写法保持一致，
            # 收尾自校验也是按edited_image去掉.jpg来反推这个文件夹名的
            per_save_pair_folder_name = os.path.splitext(
                per_save_edited_image_name)[0]

            set_annotation_count_dict[
                per_set_name] = set_annotation_count_dict.get(per_set_name,
                                                              0) + 1

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
        per_subset_kind_key,
        total_annotation_count,
        load_annotation_failed_count,
        field_not_match_count,
        duplicate_sample_key_count,
        invalid_sample_key_count,
        missing_image_count,
        image_name_not_match_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
        set_annotation_count_dict,
    ]


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上，逐样本打一次os.path.exists就是一次网络往返，
    1175万张图就要打1175万次。
    实测同一个标注文件里的图像全部落在同一个batch目录下，所以这里按目录缓存一次
    os.listdir的结果，之后只做集合查表，网络往返次数从"图像张数"降到"batch目录数"
    (2120次)。listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判，
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


def get_all_edit_annotation_pair(root_dataset_path):
    """两遍扫描上游2120个jsonl，多进程组装全部图像编辑对的列表

    第一遍只读sample_key做全量重复统计(必须先知道848组重复key才能整组丢弃),
    第二遍才真正解析每一行。两遍都按标注文件粒度开多进程，
    最后按保存的编辑后图像名统一排序，保证输出顺序可复现。
    """
    annotation_file_pair_list, subset_kind_annotation_file_count_dict = get_all_annotation_file_pair(
        root_dataset_path)

    print('1111', 'annotation file:', len(annotation_file_pair_list),
          'subset kind:', len(subset_kind_annotation_file_count_dict))

    duplicate_sample_key_dict, first_pass_annotation_count, unique_sample_key_count, duplicate_sample_key_annotation_count = get_duplicate_sample_key_dict(
        annotation_file_pair_list)

    print('1111', 'first pass annotation:', first_pass_annotation_count,
          'unique sample key:',
          unique_sample_key_count, 'duplicate sample key:',
          len(duplicate_sample_key_dict), 'duplicate sample key annotation:',
          duplicate_sample_key_annotation_count)

    total_annotation_count, load_annotation_failed_count = 0, 0
    field_not_match_count, duplicate_sample_key_count = 0, 0
    invalid_sample_key_count, missing_image_count = 0, 0
    image_name_not_match_count = 0
    empty_caption_count, null_like_caption_count = 0, 0
    no_word_char_caption_count, too_short_caption_count = 0, 0
    too_long_caption_count = 0
    invalid_placeholder_caption_count = 0
    invalid_save_image_name_count = 0
    subset_kind_annotation_count_dict = {}
    subset_kind_valid_annotation_count_dict = {}
    set_annotation_count_dict = {}
    edit_annotation_pair_list = []

    process_func = partial(process_single_annotation_file,
                           duplicate_sample_key_dict=duplicate_sample_key_dict,
                           root_dataset_path=root_dataset_path)
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_func, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            edit_annotation_pair_list.extend(per_load_result[0])

            per_subset_kind_key = per_load_result[1]
            subset_kind_annotation_count_dict[
                per_subset_kind_key] = subset_kind_annotation_count_dict.get(
                    per_subset_kind_key, 0) + per_load_result[2]
            subset_kind_valid_annotation_count_dict[
                per_subset_kind_key] = subset_kind_valid_annotation_count_dict.get(
                    per_subset_kind_key, 0) + len(per_load_result[0])

            total_annotation_count += per_load_result[2]
            load_annotation_failed_count += per_load_result[3]
            field_not_match_count += per_load_result[4]
            duplicate_sample_key_count += per_load_result[5]
            invalid_sample_key_count += per_load_result[6]
            missing_image_count += per_load_result[7]
            image_name_not_match_count += per_load_result[8]
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

    edit_annotation_pair_list = sorted(edit_annotation_pair_list,
                                       key=lambda x: x[3])

    return [
        edit_annotation_pair_list,
        len(annotation_file_pair_list),
        subset_kind_annotation_file_count_dict,
        subset_kind_annotation_count_dict,
        subset_kind_valid_annotation_count_dict,
        set_annotation_count_dict,
        len(duplicate_sample_key_dict),
        duplicate_sample_key_annotation_count,
        first_pass_annotation_count,
        total_annotation_count,
        load_annotation_failed_count,
        field_not_match_count,
        duplicate_sample_key_count,
        invalid_sample_key_count,
        missing_image_count,
        image_name_not_match_count,
        empty_caption_count,
        null_like_caption_count,
        no_word_char_caption_count,
        too_short_caption_count,
        too_long_caption_count,
        invalid_placeholder_caption_count,
        invalid_save_image_name_count,
    ]


def check_load_annotation_count(
        total_annotation_file_count, subset_kind_annotation_file_count_dict,
        subset_kind_annotation_count_dict,
        subset_kind_valid_annotation_count_dict, set_annotation_count_dict,
        duplicate_sample_key_count, duplicate_sample_key_annotation_count,
        first_pass_annotation_count, total_annotation_count,
        valid_annotation_count, invalid_caption_count_dict,
        edit_annotation_pair_list):
    """解析完标注后硬对账: 分片数、原始条数、重复key、指令过滤条数、子集条数、保存名唯一

    上游015的解压产物是一次性解出来的确定结果，条数对不上说明上游没跑完或被改动过，
    这时候继续往下跑只会得到一个悄悄少样本的新数据集，必须直接报错。
    <subset>/<kind>两级对账还能额外拦住"某个分片被漏读"或"重复key集合算错"这种
    总数对账看不出来的问题。
    保存名唯一性也必须在落盘前查: 撞名的样本对会在磁盘上互相覆盖、
    在json里互相顶掉key，事后从产物里根本看不出少了多少对。
    """
    check_error_message_list = []

    # 分片数对账: 18组逐一比对，subset-4/sampled_multi_frames本来就只有23个
    for per_subset_kind_key in sorted(
            EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT.keys()):
        per_expect_annotation_file_num = EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT[
            per_subset_kind_key]
        per_annotation_file_num = subset_kind_annotation_file_count_dict.get(
            per_subset_kind_key, 0)
        if per_annotation_file_num != per_expect_annotation_file_num:
            check_error_message_list.append(
                f'{per_subset_kind_key} annotation file num not match '
                f'{per_annotation_file_num} != {per_expect_annotation_file_num}'
            )

    for per_subset_kind_key in sorted(
            subset_kind_annotation_file_count_dict.keys()):
        if per_subset_kind_key not in EXPECTED_SUBSET_KIND_ANNOTATION_FILE_NUM_DICT:
            check_error_message_list.append(
                f'unknown subset kind {per_subset_kind_key}')

    if total_annotation_file_count != EXPECTED_TOTAL_ANNOTATION_FILE_COUNT:
        check_error_message_list.append(
            f'total annotation file count not match '
            f'{total_annotation_file_count} != '
            f'{EXPECTED_TOTAL_ANNOTATION_FILE_COUNT}')

    # 原始条数对账: 18组逐一比对
    for per_subset_kind_key in sorted(
            EXPECTED_SUBSET_KIND_ANNOTATION_COUNT_DICT.keys()):
        per_expect_annotation_count = EXPECTED_SUBSET_KIND_ANNOTATION_COUNT_DICT[
            per_subset_kind_key]
        per_annotation_count = subset_kind_annotation_count_dict.get(
            per_subset_kind_key, 0)
        if per_annotation_count != per_expect_annotation_count:
            check_error_message_list.append(
                f'{per_subset_kind_key} annotation count not match '
                f'{per_annotation_count} != {per_expect_annotation_count}')

    if total_annotation_count != EXPECTED_TOTAL_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ANNOTATION_COUNT}')

    # 两遍扫描读到的行数必须完全一致，不一致说明中途有分片读失败
    if first_pass_annotation_count != total_annotation_count:
        check_error_message_list.append(
            f'first pass annotation count not match '
            f'{first_pass_annotation_count} != {total_annotation_count}')

    # 重复sample_key对账: 组数与丢弃行数都必须与实测值一致
    if duplicate_sample_key_count != EXPECTED_DUPLICATE_SAMPLE_KEY_COUNT:
        check_error_message_list.append(
            f'duplicate sample key count not match '
            f'{duplicate_sample_key_count} != '
            f'{EXPECTED_DUPLICATE_SAMPLE_KEY_COUNT}')

    if duplicate_sample_key_annotation_count != EXPECTED_DUPLICATE_SAMPLE_KEY_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'duplicate sample key annotation count not match '
            f'{duplicate_sample_key_annotation_count} != '
            f'{EXPECTED_DUPLICATE_SAMPLE_KEY_ANNOTATION_COUNT}')

    # 第二遍解析真正丢掉的行数必须等于第一遍统计出来的重复行数
    if invalid_caption_count_dict[
            'duplicate_sample_key_annotation_count'] != duplicate_sample_key_annotation_count:
        check_error_message_list.append(
            f'skip duplicate sample key annotation count not self consistent '
            f'{invalid_caption_count_dict["duplicate_sample_key_annotation_count"]} != '
            f'{duplicate_sample_key_annotation_count}')

    # 文本层各项过滤条数逐项对账
    for per_count_name in sorted(EXPECTED_INVALID_CAPTION_COUNT_DICT.keys()):
        per_expect_count = EXPECTED_INVALID_CAPTION_COUNT_DICT[per_count_name]
        if invalid_caption_count_dict[per_count_name] != per_expect_count:
            check_error_message_list.append(
                f'{per_count_name} not match '
                f'{invalid_caption_count_dict[per_count_name]} != '
                f'{per_expect_count}')

    # 合格条数对账: 18组逐一比对 + 总数
    for per_subset_kind_key in sorted(
            EXPECTED_SUBSET_KIND_VALID_ANNOTATION_COUNT_DICT.keys()):
        per_expect_valid_annotation_count = EXPECTED_SUBSET_KIND_VALID_ANNOTATION_COUNT_DICT[
            per_subset_kind_key]
        per_valid_annotation_count = subset_kind_valid_annotation_count_dict.get(
            per_subset_kind_key, 0)
        if per_valid_annotation_count != per_expect_valid_annotation_count:
            check_error_message_list.append(
                f'{per_subset_kind_key} valid annotation count not match '
                f'{per_valid_annotation_count} != '
                f'{per_expect_valid_annotation_count}')

    if valid_annotation_count != EXPECTED_VALID_ANNOTATION_COUNT:
        check_error_message_list.append(
            f'valid annotation count not match '
            f'{valid_annotation_count} != {EXPECTED_VALID_ANNOTATION_COUNT}')

    # 子集级对账: 本数据集只有一个子集，保留下来的集合必须与白名单严格一一对应
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

    for per_set_name in sorted(set_annotation_count_dict.keys()):
        if per_set_name not in SAVE_SET_NAME_LIST:
            check_error_message_list.append(f'unknown save set {per_set_name}')

    if len(set_annotation_count_dict) != EXPECTED_SAVE_SET_COUNT:
        check_error_message_list.append(
            f'save set count not match '
            f'{len(set_annotation_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

    # 保存的编辑后图像名必须全局唯一，撞名会让两个样本对在磁盘和json里互相覆盖
    save_edited_image_name_dict = {}
    duplicate_save_edited_image_name_list = []
    for per_edit_annotation_pair in edit_annotation_pair_list:
        per_save_edited_image_name = per_edit_annotation_pair[3]
        if per_save_edited_image_name in save_edited_image_name_dict:
            duplicate_save_edited_image_name_list.append(
                per_save_edited_image_name)
            continue
        save_edited_image_name_dict[per_save_edited_image_name] = 1

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

    # 检查图像宽高比，取长短边之比，宽高比大于8和小于1/8这两种极端样本一起判掉
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
    本数据集只有non_rigid_motions这一个子集(约587.7万对)，会切出588个文件夹。
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

    上游015为了不引入二次有损压缩，把拆出来的两张单图存成了无损png，
    这里统一重编码成jpg，只换编码格式不换像素尺寸。
    编码参数用cv2.imencode('.jpg', img)的默认值(质量95 + 色度4:2:0),
    与001~006这几个已产出的数据集口径完全一致。
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
    上游jsonl里剩下的属性(edit_description/两条图像caption/video_caption/
    video_id与全部帧序溯源字段/frame_kind_name/各类source_*)全部丢弃，
    理由见文件开头SAVE_ANNOTATION_KEY_NAME_LIST的注释。
    ti2i_caption写的就是上游edit_instruction字段(即原始json的edit_rewrite)
    strip后的英文原文，不做任何改写。
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


def check_single_save_folder(folder_check_pair, save_dataset_path):
    """校验单个文件夹: 文件夹容量、json与磁盘一一对应、指令与参考图数量

    每个子集除最后一个文件夹外都必须是满10000对，json里的每个key都必须在磁盘上有
    对应的样本对文件夹且文件恰好等于编辑后图像 + 所有参考图像，磁盘上也不允许有
    json没记录的残留样本对文件夹。另外还要复检ti2i_caption: 占位符编号集合必须与
    reference_image这个list的长度自洽、长度必须在阈值区间内、不能是null字面量或
    只剩标点的无意义指令、记录的长度必须与字符串实际长度一致。
    """
    per_set_name, per_folder_name, per_is_set_last_folder = folder_check_pair

    check_error_message_list = []

    per_expect_reference_image_num = get_expect_reference_image_num(
        per_set_name)
    per_exempt_aspect_ratio_align_flag = per_set_name in EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST

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

    # 除每个子集最后一个文件夹外都必须是满10000对
    if not per_is_set_last_folder and len(
            per_folder_annotation_dict) != PER_FOLDER_EDIT_PAIR_NUM:
        check_error_message_list.append(
            f'{per_folder_name} edit pair num not match {len(per_folder_annotation_dict)} != {PER_FOLDER_EDIT_PAIR_NUM}'
        )

    per_folder_path = os.path.join(save_dataset_path, per_set_name,
                                   per_folder_name)
    per_exist_pair_folder_name_list = sorted([
        per_save_pair_folder_name
        for per_save_pair_folder_name in os.listdir(per_folder_path)
        if os.path.isdir(
            os.path.join(per_folder_path, per_save_pair_folder_name))
    ]) if os.path.isdir(per_folder_path) else []
    per_expect_pair_folder_name_list = sorted([
        per_save_edited_image_name.removesuffix('.jpg')
        for per_save_edited_image_name in per_folder_annotation_dict.keys()
    ])
    if per_exist_pair_folder_name_list != per_expect_pair_folder_name_list:
        check_error_message_list.append(
            f'{per_folder_name} pair folder not match {len(per_exist_pair_folder_name_list)} != {len(per_expect_pair_folder_name_list)}'
        )

    for per_save_edited_image_name in sorted(
            per_folder_annotation_dict.keys()):
        per_annotation = per_folder_annotation_dict[per_save_edited_image_name]

        # 每条标注的字段集合必须和约定的七个key严格一致，不能多也不能少
        if sorted(per_annotation.keys()) != sorted(
                SAVE_ANNOTATION_KEY_NAME_LIST):
            check_error_message_list.append(
                f'{per_save_edited_image_name} annotation key not match')
            continue

        if per_save_edited_image_name != per_annotation['edited_image']:
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name not match')
        if not per_save_edited_image_name.endswith(
                SAVE_EDITED_IMAGE_NAME_SUFFIX):
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name suffix not match'
            )
        if not per_save_edited_image_name.startswith(
                f'{DATASET_NAME}_{per_set_name}_'):
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name prefix not match'
            )
        if per_save_edited_image_name != per_save_edited_image_name.lower():
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name not all lower case'
            )
        if not VALID_IMAGE_NAME_PATTERN.match(per_save_edited_image_name):
            check_error_message_list.append(
                f'{per_save_edited_image_name} edited image name not a valid name'
            )
        if not isinstance(per_annotation['reference_image'], list):
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image not a list')
            continue
        if per_annotation['reference_image_num'] != len(
                per_annotation['reference_image']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image num not match')
        # 本数据集所有样本对都必须是单参考图
        if per_annotation[
                'reference_image_num'] != per_expect_reference_image_num:
            check_error_message_list.append(
                f'{per_save_edited_image_name} reference image num not match set {per_annotation["reference_image_num"]} != {per_expect_reference_image_num}'
            )
        # 参考图名的前缀必须和编辑后图像名同一个前缀、后缀必须是_reference.jpg
        for per_save_reference_image_name in per_annotation['reference_image']:
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
        if per_annotation['width'] <= 0 or per_annotation['height'] <= 0:
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
        # ti2i_caption的占位符编号集合必须与reference_image这个list的长度自洽:
        # 单参考图即要求指令里完全没有占位符
        if check_invalid_caption(per_annotation['ti2i_caption'],
                                 len(per_annotation['reference_image'])):
            check_error_message_list.append(
                f'{per_save_edited_image_name} caption placeholder index not match reference image num {len(per_annotation["reference_image"])}'
            )
        # 落盘后的指令里不允许再残留null字面量或"不做任何修改"这类无意义指令
        if check_null_like_caption(per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} still a null like caption')
        # 也不允许残留只剩标点、没有任何数字/字母/汉字的指令
        if not CAPTION_WORD_CHAR_PATTERN.search(
                per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} still a no word char caption')
        # json里存的就是归一化后的指令，长度过滤也是按归一化后判定的，
        # 两者口径一致，这里直接量json里的长度复检
        if len(per_annotation['ti2i_caption'].strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_edited_image_name} still an invalid caption')
        if len(per_annotation['ti2i_caption'].strip()) > MAX_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_edited_image_name} still a too long caption')
        # 记录的指令长度必须和指令字符串的实际长度对得上
        if per_annotation['ti2i_caption_length'] != len(
                per_annotation['ti2i_caption']):
            check_error_message_list.append(
                f'{per_save_edited_image_name} ti2i caption length not match')

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
                f'{per_save_pair_folder_name} pair image file not match')

        # 真解一次落盘后的reference_image[0]，硬校验它的shape严格等于
        # json里的width/height(也就是编辑后图的宽高)。
        # 这一条是"第一张参考图与编辑后图尺寸必须一致"这个核心不变式的最终
        # 验收: 只对账json里的数字是查不出resize有没有真的生效的。
        # 非豁免子集才校验: 豁免子集是完全原样落盘的，尺寸本来就可以不等
        if CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG and not per_exempt_aspect_ratio_align_flag and len(
                per_annotation['reference_image']) > 0:
            per_check_save_reference_image_path = os.path.join(
                per_pair_folder_path, per_annotation['reference_image'][0])
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
                    f'{per_annotation["width"]} {per_annotation["height"]}')

    return [
        per_folder_name,
        len(per_folder_annotation_dict),
        check_error_message_list,
    ]


def check_save_dataset(save_dataset_path, set_folder_count_dict):
    """全部落盘后的收尾自校验: 文件夹容量、json与磁盘一一对应、指令与参考图数量

    校验口径与006/007完全一致，只是把"按文件夹"这一层改成了多进程:
    本数据集有588个文件夹、约587.7万个样本对文件夹，每个样本对文件夹都要listdir
    一次再核对文件名，串行跑在NAS上太久，所以按文件夹粒度开多进程(与003一致)。
    """
    check_error_message_list = []

    folder_check_pair_list = []
    for per_set_name in sorted(set_folder_count_dict.keys()):
        per_set_folder_count = set_folder_count_dict[per_set_name]
        for per_folder_index in range(per_set_folder_count):
            per_folder_name = f'{per_set_name}_{per_folder_index:05d}'
            folder_check_pair_list.append([
                per_set_name,
                per_folder_name,
                per_folder_index == per_set_folder_count - 1,
            ])

    total_edit_pair_count = 0
    check_func = partial(check_single_save_folder,
                         save_dataset_path=save_dataset_path)
    with Pool(processes=min(PROCESS_NUM, max(len(folder_check_pair_list),
                                             1))) as pool:
        for per_check_result in tqdm(pool.imap_unordered(
                check_func, folder_check_pair_list),
                                     total=len(folder_check_pair_list)):
            _, per_folder_edit_pair_count, per_check_error_message_list = per_check_result
            total_edit_pair_count += per_folder_edit_pair_count
            check_error_message_list.extend(per_check_error_message_list)

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

    edit_annotation_pair_list, total_annotation_file_count, subset_kind_annotation_file_count_dict, subset_kind_annotation_count_dict, subset_kind_valid_annotation_count_dict, set_annotation_count_dict, duplicate_sample_key_count, duplicate_sample_key_annotation_count, first_pass_annotation_count, total_annotation_count, load_annotation_failed_count, field_not_match_count, skip_duplicate_sample_key_annotation_count, invalid_sample_key_count, missing_image_count, image_name_not_match_count, empty_caption_count, null_like_caption_count, no_word_char_caption_count, too_short_caption_count, too_long_caption_count, invalid_placeholder_caption_count, invalid_save_image_name_count = get_all_edit_annotation_pair(
        root_dataset_path)

    print('1111', total_annotation_file_count, total_annotation_count,
          load_annotation_failed_count, field_not_match_count,
          duplicate_sample_key_count, duplicate_sample_key_annotation_count,
          skip_duplicate_sample_key_annotation_count, invalid_sample_key_count,
          missing_image_count, image_name_not_match_count, empty_caption_count,
          null_like_caption_count, no_word_char_caption_count,
          too_short_caption_count, too_long_caption_count,
          invalid_placeholder_caption_count, invalid_save_image_name_count,
          len(set_annotation_count_dict), len(edit_annotation_pair_list))

    if len(edit_annotation_pair_list) > 0:
        print('1111', edit_annotation_pair_list[0])

    invalid_caption_count_dict = {
        'empty_caption_count':
        empty_caption_count,
        'null_like_caption_count':
        null_like_caption_count,
        'no_word_char_caption_count':
        no_word_char_caption_count,
        'too_short_caption_count':
        too_short_caption_count,
        'too_long_caption_count':
        too_long_caption_count,
        'invalid_placeholder_caption_count':
        invalid_placeholder_caption_count,
        'duplicate_sample_key_annotation_count':
        skip_duplicate_sample_key_annotation_count,
    }

    # 标注侧硬对账不过直接中断，不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list = check_load_annotation_count(
        total_annotation_file_count, subset_kind_annotation_file_count_dict,
        subset_kind_annotation_count_dict,
        subset_kind_valid_annotation_count_dict, set_annotation_count_dict,
        duplicate_sample_key_count, duplicate_sample_key_annotation_count,
        first_pass_annotation_count, total_annotation_count,
        len(edit_annotation_pair_list), invalid_caption_count_dict,
        edit_annotation_pair_list)

    print('1111', 'load annotation check error',
          load_annotation_check_error_message_list[:20])
    if len(load_annotation_check_error_message_list) > 0:
        raise Exception(
            f'load annotation check failed {load_annotation_check_error_message_list[:20]}'
        )

    if load_annotation_failed_count > 0:
        raise Exception(
            f'load annotation failed count {load_annotation_failed_count}')

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

    print('3333', 'total annotation file:', total_annotation_file_count,
          'total annotation:', total_annotation_count,
          'load annotation failed:', load_annotation_failed_count,
          'field not match:', field_not_match_count, 'duplicate sample key:',
          duplicate_sample_key_count, 'skip duplicate sample key annotation:',
          skip_duplicate_sample_key_annotation_count, 'invalid sample key:',
          invalid_sample_key_count, 'missing image:', missing_image_count,
          'image name not match:', image_name_not_match_count,
          'empty caption:', empty_caption_count, 'null like caption:',
          null_like_caption_count, 'no word char caption:',
          no_word_char_caption_count, 'too short caption:',
          too_short_caption_count, 'too long caption:', too_long_caption_count,
          'invalid placeholder caption:', invalid_placeholder_caption_count,
          'invalid save image name:', invalid_save_image_name_count,
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
        'total_annotation_file_count':
        total_annotation_file_count,
        'total_annotation_count':
        total_annotation_count,
        'first_pass_annotation_count':
        first_pass_annotation_count,
        'load_annotation_failed_count':
        load_annotation_failed_count,
        'field_not_match_count':
        field_not_match_count,
        'duplicate_sample_key_count':
        duplicate_sample_key_count,
        'duplicate_sample_key_annotation_count':
        duplicate_sample_key_annotation_count,
        'skip_duplicate_sample_key_annotation_count':
        skip_duplicate_sample_key_annotation_count,
        'invalid_sample_key_count':
        invalid_sample_key_count,
        'missing_image_count':
        missing_image_count,
        'image_name_not_match_count':
        image_name_not_match_count,
        'empty_caption_count':
        empty_caption_count,
        'null_like_caption_count':
        null_like_caption_count,
        'no_word_char_caption_count':
        no_word_char_caption_count,
        'too_short_caption_count':
        too_short_caption_count,
        'too_long_caption_count':
        too_long_caption_count,
        'invalid_placeholder_caption_count':
        invalid_placeholder_caption_count,
        'invalid_save_image_name_count':
        invalid_save_image_name_count,
        'invalid_image_count':
        invalid_image_count,
        # reference_image[0]与编辑后图长宽比不同而被整对丢弃的条数。
        # 第一轮跑完后可以把实测值回填成EXPECTED_DIFFERENT_ASPECT_RATIO_COUNT
        # 再上硬对账，守住"哪些样本被resize对齐、哪些被丢弃"这条口径
        'different_aspect_ratio_count':
        different_aspect_ratio_count,
        'set_different_aspect_ratio_count_dict':
        set_different_aspect_ratio_count_dict,
        # 落盘口径标记，便于下游一眼看出这份产物是不是"参考图已对齐"的版本
        'resize_reference_image_to_edited_image_shape_flag':
        True,
        'long_side_align_extra_reference_image_flag':
        True,
        'exempt_aspect_ratio_align_set_name_list':
        EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST,
        'check_save_reference_image_shape_flag':
        CHECK_SAVE_REFERENCE_IMAGE_SHAPE_FLAG,
        'save_edit_pair_failed_count':
        save_edit_pair_failed_count,
        'total_save_edit_pair_count':
        len(save_result_list),
        'total_save_reference_image_count':
        total_save_reference_image_count,
        'reference_image_num_mismatch_count':
        reference_image_num_mismatch_count,
        'total_save_set_count':
        len(set_folder_count_dict),
        'total_save_folder_count':
        len(folder_edit_pair_count_dict),
        'check_total_edit_pair_count':
        check_total_edit_pair_count,
        'check_error_count':
        len(check_error_message_list),
        'min_caption_length':
        MIN_CAPTION_LENGTH,
        'max_caption_length':
        MAX_CAPTION_LENGTH,
        'subset_kind_annotation_file_count_dict':
        subset_kind_annotation_file_count_dict,
        'subset_kind_annotation_count_dict':
        subset_kind_annotation_count_dict,
        'subset_kind_valid_annotation_count_dict':
        subset_kind_valid_annotation_count_dict,
        'set_annotation_count_dict':
        set_annotation_count_dict,
        'set_folder_count_dict':
        set_folder_count_dict,
        'folder_edit_pair_count_dict':
        folder_edit_pair_count_dict,
        'check_error_message_list':
        check_error_message_list[:MAX_SAVE_MESSAGE_NUM],
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/BM-6M'
    save_dataset_path = r'/root/autodl-tmp/ti2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
