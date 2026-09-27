import os
import re
import json
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

# 通用图像编辑数据集数据完整性复核脚本
# ti2i_datasets下每个一级子目录都是一个图像编辑数据集,且文件组织结构完全一致:
# ti2i_datasets/<dataset_dir_name>/<set_name>/<folder_name>/<pair_folder_name>/*.jpg
# ti2i_datasets/<dataset_dir_name>/<set_name>/<folder_name>.json
# 每个样本对(1张编辑后图像 + 1~2张参考图像 + 1条编辑指令)独占一个<pair_folder_name>
# 文件夹,该文件夹里只放这个样本对的编辑后图像和全部参考图像;
# 每个<folder_name>.json与同级同名的<folder_name>文件夹一一对应,json内容形如:
# {"<save_edited_image_name>.jpg": {"reference_image": [str, ...],
#                                   "edited_image": str,
#                                   "reference_image_num": int,
#                                   "width": int, "height": int,
#                                   "ti2i_caption": str,
#                                   "ti2i_caption_length": int}}
# 标注key就是编辑后图像名,<pair_folder_name>就是标注key去掉.jpg后缀
# 其中ti2i_caption与ti2i_caption_length这两个key的value类型按数据集分两种规格:
# 单条指令数据集(绝大多数)是str与int;
# 多条指令数据集(见MULTI_CAPTION_KEY_NAME_DICT,目前有上游
# 016.resave_unicedit_10m_ti2i_dataset.py产出的UnicEdit(中英双语两条)与
# 017.resave_conceptedit_12m_ti2i_dataset.py产出的ConceptEdit-12M
# (中英双语 × 粗细两档共四条))这两个key的value都是字典,每条指令各占一条,形如
# {"<save_edited_image_name>.jpg": {...,
#                                   "ti2i_caption": {
#                                       "english_ti2i_caption": str,
#                                       "chinese_ti2i_caption": str},
#                                   "ti2i_caption_length": {
#                                       "english_ti2i_caption_length": int,
#                                       "chinese_ti2i_caption_length": int}}}
# 或
# {"<save_edited_image_name>.jpg": {...,
#                                   "ti2i_caption": {
#                                       "short_english_ti2i_caption": str,
#                                       "short_chinese_ti2i_caption": str,
#                                       "detailed_english_ti2i_caption": str,
#                                       "detailed_chinese_ti2i_caption": str},
#                                   "ti2i_caption_length": {
#                                       "short_english_ti2i_caption_length": int,
#                                       "short_chinese_ti2i_caption_length": int,
#                                       "detailed_english_ti2i_caption_length": int,
#                                       "detailed_chinese_ti2i_caption_length": int}}}
# 本脚本只做只读复核,不对数据集做任何修改(全程只有os.scandir/os.listdir/os.path/
# 只读open/json.load,复核报告一律写到数据集目录之外的save_check_result_path下)
# 每次运行只复核一个数据集,运行前修改__main__里的root_dataset_path即可

# 落盘标注固定只有这7个key,多一个少一个都算标注key-value对不完整
ANNOTATION_KEY_NAME_LIST = [
    'reference_image',
    'edited_image',
    'reference_image_num',
    'width',
    'height',
    'ti2i_caption',
    'ti2i_caption_length',
]

SAVE_IMAGE_NAME_SUFFIX = '.jpg'

SAVE_EDITED_IMAGE_NAME_SUFFIX = '_edited.jpg'

SAVE_REFERENCE_IMAGE_NAME_SUFFIX = '_reference.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点,与上游resave脚本口径一致
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 图像完整性校验级别:
# exists: 只查图像文件存在且字节数大于0,最快,查不出内容损坏。
#         字节数直接取自样本对文件夹那一次os.scandir的结果,不额外打任何IO
# header: 只读文件头拿到真实编码格式与宽高并与标注比对,再按**真实格式**读文件尾
#         校验结束标记(JPEG的FFD9 / PNG的IEND),每张图只有几次小IO,
#         能查出丢失/零字节/文件头损坏/写盘截断,千万级图像量也能全量跑完,
#         默认用这一档
# decode: 完整cv2解码并比对shape,最彻底但最慢(千万级图像量要跑几十小时)。
#         cv2.imdecode按文件内容解码、不看后缀,所以这一档不受下面那条
#         "PNG内容 + .jpg后缀"的规格影响
CHECK_IMAGE_LEVEL = 'header'

VALID_CHECK_IMAGE_LEVEL_LIST = [
    'exists',
    'header',
    'decode',
]

# 各图像格式的文件尾结束标记,写盘被截断的图像尾部一定不是这串字节。
# 必须按PIL从文件魔数解析出的**真实格式**取,绝不能按文件名后缀取:
# 上游015.1(InterEdit-Mask)落盘的mask是**PNG无损内容 + _mask_reference.jpg后缀**
# (mask是二值图,用jpg有损编码会在物体边缘插出0~255的中间值,所以内容走PNG无损;
#  后缀保持.jpg是为了满足"参考图名统一以_reference.jpg结尾"的命名规范),
# 按后缀去查JPEG的FFD9会把全部mask误判成写盘截断。
# 字典里没有的格式只做文件头校验、不校验结束标记,不引入新的误判
IMAGE_FORMAT_END_MARKER_BYTES_DICT = {
    'JPEG': b'\xff\xd9',
    'PNG': b'IEND\xaeB`\x82',
}

# 落盘图像允许的真实编码格式白名单(按文件魔数判定,完全不看文件名后缀)。
# 上游各resave脚本只会产出两种编码: 编辑后图与参考图是jpg、
# InterEdit-Mask的mask是png,出现其它格式说明混进了非本链路产出的文件
VALID_IMAGE_FORMAT_NAME_LIST = [
    'JPEG',
    'PNG',
]

PROCESS_NUM = 32

# 上游resave时每个文件夹放满10000个样本对(每个子集最后一个文件夹允许不满),
# 文件夹不满不算数据损坏,只作为warning上报
PER_FOLDER_EDIT_PAIR_NUM = 10000

MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 标注里的width/height一律是编辑后图像的宽高,参考图像的宽高上游没有单独落盘,
# 所以短边和宽高比只对编辑后图像校验。
#
# 【CHECK_REFERENCE_IMAGE_SIZE的含义已随上游resave改动而变化】
# resave脚本现在保证了一条新的不变式:
#   **reference_image[0](编辑前原图)的尺寸严格等于json里的width/height**
#   (长宽比与编辑后图相同的等比resize过去、不同的整对丢弃)。
# 所以这里可以、也应该把它打开,真解一次reference_image[0]、
# 硬校验它的尺寸等于width/height,作为对上游resize链路的独立复检。
#
# 注意校验范围必须收窄,只校验reference_image[0]:
# 1. reference_image[k>=1](第二张视觉条件图/主体图/物体图)在长宽比与编辑后图
#    不同时走的是"长边对齐"等比resize,尺寸本来就不等于width/height;
# 2. EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_DICT里那几个豁免子集是完全原样落盘的,
#    尺寸也本来就可以不等。
CHECK_REFERENCE_IMAGE_SIZE = True

# 豁免"reference_image[0]尺寸必须等于width/height"这条校验的(数据集, 子集)。
# 必须与上游resave脚本里的EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_LIST保持一致:
# 目前只有anyedit-split的outpaint(向外扩画布)与rotation_change(旋转90度会
# 把宽高互换)这两个子集,它们的任务语义本身就要求编辑后图与原图长宽比不同,
# 所以上游是完全原样落盘、不做任何对齐的
EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_DICT = {
    'anyedit-split': [
        'outpaint',
        'rotation_change',
    ],
}

# 多条指令数据集里ti2i_caption这个字典应有的key(严格一一对应,缺key/多key/错名
# 都算标注key-value对不完整): 这些数据集的ti2i_caption与ti2i_caption_length
# 这两个key的value都是字典、每条指令各占一条,而不是单条指令数据集的str与int
# (见文件头的json格式说明)。
# 必须与上游resave脚本的SAVE_TI2I_CAPTION_KEY_NAME_LIST逐项一致:
# UnicEdit来自016(上游prompt_en与prompt_cn全库非空、是同一条编辑语义的两种语言
# 表述,两条都有训练价值,所以两条都落盘);
# ConceptEdit-12M来自017(中英双语 × 粗细两档共四条,短指令长度只有详细指令的28%、
# 四条互不相同,都有训练价值,所以四条都落盘)。
#
# 【为什么按数据集目录名写死而不是看value的类型现判】
# 看到dict就自动按多条指令校验,等于把"单条指令数据集里混进了一个dict"这种真损坏
# 静默放过; 反过来多条指令数据集里退化成str也一样查不出来。
# 写死之后两种偏差都会被判成invalid_annotation: 声明为多条指令的数据集里出现str、
# 没声明的数据集里出现dict,都是标注规格被改动过的信号
MULTI_CAPTION_KEY_NAME_DICT = {
    'UnicEdit': [
        'english_ti2i_caption',
        'chinese_ti2i_caption',
    ],
    'ConceptEdit-12M': [
        'short_english_ti2i_caption',
        'short_chinese_ti2i_caption',
        'detailed_english_ti2i_caption',
        'detailed_chinese_ti2i_caption',
    ],
}

# 多条指令数据集里ti2i_caption_length这个字典应有的key,顺序必须与上面那个字典
# 里同一个数据集的列表逐项对应(第i条指令的长度就记在第i个key里),
# 与上游resave脚本的SAVE_TI2I_CAPTION_LENGTH_KEY_NAME_LIST完全一致
MULTI_CAPTION_LENGTH_KEY_NAME_DICT = {
    'UnicEdit': [
        'english_ti2i_caption_length',
        'chinese_ti2i_caption_length',
    ],
    'ConceptEdit-12M': [
        'short_english_ti2i_caption_length',
        'short_chinese_ti2i_caption_length',
        'detailed_english_ti2i_caption_length',
        'detailed_chinese_ti2i_caption_length',
    ],
}

# 各数据集上游的指令长度阈值并不一致(anyedit取200,ImgEdit取512,
# VINS-120K和X2Edit的下限也各不相同),所以通用复核脚本只校验指令非空以及
# 记录长度与实际长度自洽,不校验长度上限
MIN_CAPTION_LENGTH = 1

# 每个样本对的参考图像数量: 第0张永远是编辑前原图,
# 双参考图子集(ImgEdit的reference_replace、anyedit的visual_*)再多一张视觉参考图
MIN_REFERENCE_IMAGE_NUM = 1

MAX_REFERENCE_IMAGE_NUM = 2

# 带编号的视觉参考图占位符,编号从"非原图的第1张参考图"起算:
# [V1*]指代reference_image[1]...[VN*]指代reference_image[N],
# 其中N == reference_image_num - 1,编辑前原图reference_image[0]隐式不占编号。
# 这套写法与上游各resave脚本完全一致
CAPTION_VISUAL_PLACEHOLDER_PATTERN = re.compile(r'\[V(\d*)\*\]')

# 同一个编号在一条指令里最多允许重复出现的次数
MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM = 2

# 上游resave时在数据集根目录留下的落盘统计文件,不参与复核,只作为warning放过
SKIP_ROOT_FILE_NAME_LIST = [
    'resave_check_result.json',
]

# 只复核指定子集(留空表示复核该数据集下的全部子集)
CHECK_SET_NAME_LIST = []

# 每一类问题最多在报告里保留多少条样例信息,避免坏样本极多时报告体积爆炸
MAX_ERROR_MESSAGE_NUM = 100

# 复核结果按问题类别分类统计,每一类既统计条数也保留样例信息
CHECK_ERROR_TYPE_NAME_LIST = [
    # 标注json自身无法解析(整个分片的标注都不可用)
    'broken_annotation_json',
    # <folder_name>文件夹与<folder_name>.json没有一一对应
    'folder_json_not_paired',
    # 标注里记录的样本对在磁盘上找不到样本对文件夹,或文件夹里缺编辑后图像/参考图像
    # (既是"每个样本对图像文件是否存在"的失败项,也就是孤立标注)
    'orphan_annotation',
    # 磁盘上的样本对文件夹在标注json里找不到对应标注,
    # 或样本对文件夹里混入了没有被这条标注引用的图像文件
    'orphan_image',
    # 图像文件存在但内容不完整(零字节/文件头损坏/写盘截断/无法解码)
    'broken_image',
    # 编辑后图像的真实宽高与标注里记录的width/height不一致
    'image_size_not_match',
    # 标注的key-value对不完整(缺key/多key/类型非法/取值非法/长度不自洽)
    'invalid_annotation',
    # 标注中涉及数量的值不正确(reference_image_num与参考图像列表长度不一致、
    # 数量越界、参考图像名重复、指令里的占位符编号与参考图像数量不自洽)
    'invalid_reference_image_num',
    # 图像文件名不符合落盘命名规范
    'invalid_image_name',
    # 分片文件夹/样本对文件夹里混入了非预期的残留文件
    'unexpected_file',
]


def check_single_image_file(per_image_path,
                            per_image_file_size,
                            per_annotation_image_w=None,
                            per_annotation_image_h=None,
                            check_image_size_flag=False):
    """按CHECK_IMAGE_LEVEL校验单个图像文件是否存在且完整,并比对真实宽高

    只返回第一个命中的问题,返回[错误类别, 错误信息],完全正常时返回[None, None]。
    per_image_file_size直接取自样本对文件夹那一次os.scandir的结果,避免逐张图像
    再打一次getsize(NAS上每次都是一轮网络往返)。
    上游resave时图像都是cv2重编码写盘的,不带EXIF方向信息,所以PIL读文件头拿到的
    宽高与cv2解码后的shape一定一致,两种校验级别得到的宽高口径相同。
    参考图像的宽高标注里没有记录,调用方传check_image_size_flag=False即可只查完整性。

    header档里的编码格式与结束标记一律按PIL从文件魔数解析出的**真实格式**判定,
    绝不按文件名后缀判定: 上游015.1(InterEdit-Mask)的mask是
    PNG无损内容 + _mask_reference.jpg后缀(见IMAGE_FORMAT_END_MARKER_BYTES_DICT
    处的说明),按后缀去查JPEG的FFD9会把全部mask误判成写盘截断。
    """
    if per_image_file_size is None:
        try:
            per_image_file_size = os.path.getsize(per_image_path)
        except Exception as e:
            return [
                'broken_image', f'{per_image_path} get file size error {e}'
            ]

    if per_image_file_size <= 0:
        return ['broken_image', f'{per_image_path} file size is 0']

    if CHECK_IMAGE_LEVEL == 'exists':
        return [None, None]

    per_image_w, per_image_h = 0, 0
    if CHECK_IMAGE_LEVEL == 'header':
        # PIL的Image.open是惰性的,只读文件头就能拿到真实编码格式与宽高,
        # 不解码整张图像; format是按文件魔数解析出来的,拿它不额外打任何IO
        per_image_format = None
        try:
            with Image.open(per_image_path) as per_image:
                per_image_w, per_image_h = per_image.size
                per_image_format = per_image.format
        except Exception as e:
            return [
                'broken_image', f'{per_image_path} read image header error {e}'
            ]

        # 落盘图像的真实编码格式必须在白名单内,
        # 出现其它格式说明混进了非本链路产出的文件
        if per_image_format not in VALID_IMAGE_FORMAT_NAME_LIST:
            return [
                'broken_image',
                f'{per_image_path} image format illegal {per_image_format}'
            ]

        # 文件头正常但写盘被截断的图像,尾部不会是这个格式的结束标记。
        # 结束标记按真实格式取,取不到的格式只做文件头校验、不校验结束标记
        per_image_end_marker_bytes = IMAGE_FORMAT_END_MARKER_BYTES_DICT.get(
            per_image_format, None)
        if per_image_end_marker_bytes is not None:
            try:
                with open(per_image_path, 'rb') as load_image_file:
                    load_image_file.seek(-len(per_image_end_marker_bytes),
                                         os.SEEK_END)
                    per_image_end_bytes = load_image_file.read(
                        len(per_image_end_marker_bytes))
            except Exception as e:
                return [
                    'broken_image',
                    f'{per_image_path} read image end bytes error {e}'
                ]

            if per_image_end_bytes != per_image_end_marker_bytes:
                return [
                    'broken_image',
                    f'{per_image_path} {per_image_format} end marker not match'
                ]
    else:
        try:
            per_image = cv2.imdecode(
                np.fromfile(per_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return ['broken_image', f'{per_image_path} decode image error {e}']

        if per_image is None or per_image.ndim != 3 or per_image.shape[2] != 3:
            return ['broken_image', f'{per_image_path} decode image failed']

        per_image_h, per_image_w = per_image.shape[0], per_image.shape[1]

    if per_image_w <= 0 or per_image_h <= 0:
        return [
            'broken_image',
            f'{per_image_path} image shape illegal {per_image_w} {per_image_h}'
        ]

    if not check_image_size_flag:
        return [None, None]

    # 标注里的width/height缺失或类型非法时已经在标注校验里报过,这里不再重复比对
    # (bool是int的子类,必须显式排除,否则True会被当成1参与比对)
    if not isinstance(per_annotation_image_w, int) or isinstance(
            per_annotation_image_w,
            bool) or not isinstance(per_annotation_image_h, int) or isinstance(
                per_annotation_image_h, bool):
        return [None, None]

    if per_image_w != per_annotation_image_w or per_image_h != per_annotation_image_h:
        return [
            'image_size_not_match',
            f'{per_image_path} image size not match real {per_image_w} {per_image_h} annotation {per_annotation_image_w} {per_annotation_image_h}'
        ]

    return [None, None]


def get_multi_caption_key_name_pair(per_dataset_dir_name):
    """取这个数据集多条指令规格下的[指令key名列表, 指令长度key名列表]

    单条指令数据集两个列表都为空,调用方据此走str与int那套校验。
    见MULTI_CAPTION_KEY_NAME_DICT的注释: 规格按数据集目录名写死,
    不看标注里value的实际类型现判,这样两个方向的规格偏差都能被判出来。
    """
    return [
        MULTI_CAPTION_KEY_NAME_DICT.get(per_dataset_dir_name, []),
        MULTI_CAPTION_LENGTH_KEY_NAME_DICT.get(per_dataset_dir_name, []),
    ]


def get_annotation_ti2i_caption_pair_list(per_set_name, per_edited_image_name,
                                          per_annotation,
                                          per_multi_caption_key_name_pair):
    """把标注里的指令与指令长度统一取成[指令列表, 错误信息列表]

    指令列表里每一项都是
    [指令key名, 指令, 指令长度key名, 指令长度],
    单条指令数据集只有1项(key名就是ti2i_caption与ti2i_caption_length)、
    多条指令数据集有N项(UnicEdit是英文中文两项、ConceptEdit-12M是简短英文/
    简短中文/详细英文/详细中文四项),之后的指令校验一律按这个列表逐项执行,
    两种规格就能走完全相同的一套校验逻辑。

    这里只校验多条指令字典自身的结构(两个value都必须是dict、两个dict的key集合
    必须严格等于这个数据集约定的两个列表),指令与指令长度本身的类型和取值留给
    调用方校验: 单条指令数据集这两个key的value是不是str与int也是由调用方判的,
    所以"单条指令数据集里混进dict"会被判成指令类型非法、
    "多条指令数据集里退化成str"会被判成字典结构非法,两个方向都不会被静默放过。
    结构不对时返回空的指令列表,调用方不再做后续的指令内容校验,
    避免同一个问题被重复上报。
    """
    check_error_message_list = []
    ti2i_caption_pair_list = []

    per_caption_key_name_list, per_caption_length_key_name_list = per_multi_caption_key_name_pair

    per_ti2i_caption = per_annotation.get('ti2i_caption', None)
    per_ti2i_caption_length = per_annotation.get('ti2i_caption_length', None)

    # 单条指令数据集原样带下去,类型校验由调用方做,
    # 报错文案与只支持单条指令时完全一致
    if len(per_caption_key_name_list) == 0:
        ti2i_caption_pair_list.append([
            'ti2i_caption',
            per_ti2i_caption,
            'ti2i_caption_length',
            per_ti2i_caption_length,
        ])

        return [ti2i_caption_pair_list, check_error_message_list]

    if not isinstance(per_ti2i_caption, dict) or not isinstance(
            per_ti2i_caption_length, dict):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation ti2i_caption ti2i_caption_length type illegal {type(per_ti2i_caption)} {type(per_ti2i_caption_length)}'
        )

        return [ti2i_caption_pair_list, check_error_message_list]

    # 指令字典的key集合必须严格一一对应,缺key/多key/错名都算标注key-value对不完整
    if sorted(per_ti2i_caption.keys()) != sorted(per_caption_key_name_list):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation ti2i_caption key not match {sorted(per_ti2i_caption.keys())}'
        )
    if sorted(per_ti2i_caption_length.keys()) != sorted(
            per_caption_length_key_name_list):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation ti2i_caption_length key not match {sorted(per_ti2i_caption_length.keys())}'
        )

    if len(check_error_message_list) > 0:
        return [ti2i_caption_pair_list, check_error_message_list]

    # 两个列表逐项对应: 第i条指令的长度就记在第i个长度key里
    for per_caption_key_name, per_caption_length_key_name in zip(
            per_caption_key_name_list, per_caption_length_key_name_list):
        ti2i_caption_pair_list.append([
            per_caption_key_name,
            per_ti2i_caption[per_caption_key_name],
            per_caption_length_key_name,
            per_ti2i_caption_length[per_caption_length_key_name],
        ])

    return [ti2i_caption_pair_list, check_error_message_list]


def check_single_annotation(per_set_name, per_edited_image_name,
                            per_annotation, per_multi_caption_key_name_pair):
    """校验单个样本对标注的key-value对是否完整合法,返回错误信息列表

    只做标注自身的完整性和自洽性校验,不涉及磁盘上的图像文件。
    涉及数量的值(reference_image_num、占位符编号)单独放在
    check_single_reference_image_num里校验,便于按问题类别分开统计。

    ti2i_caption与ti2i_caption_length这两个key的value按数据集分单条指令
    (str与int)与多条指令(两个字典)两种规格,统一由
    get_annotation_ti2i_caption_pair_list取成同一种列表后逐条校验,
    多条指令数据集的每条指令走完全相同的一套校验。
    """
    check_error_message_list = []

    if not isinstance(per_annotation, dict):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation is not a dict')

        return check_error_message_list

    # 缺key和多key都算标注key-value对不完整
    for per_key_name in ANNOTATION_KEY_NAME_LIST:
        if per_key_name not in per_annotation:
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation key {per_key_name} not exists'
            )
    for per_key_name in per_annotation.keys():
        if per_key_name not in ANNOTATION_KEY_NAME_LIST:
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation has unexpected key {per_key_name}'
            )

    if len(check_error_message_list) > 0:
        return check_error_message_list

    per_reference_image_name_list = per_annotation['reference_image']
    per_annotation_edited_image_name = per_annotation['edited_image']
    per_image_w = per_annotation['width']
    per_image_h = per_annotation['height']

    # 标注key就是编辑后图像名,edited_image字段必须和它完全一致
    if not isinstance(per_annotation_edited_image_name, str):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation edited_image type illegal {type(per_annotation_edited_image_name)}'
        )
    elif per_annotation_edited_image_name != per_edited_image_name:
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation edited_image not match {per_annotation_edited_image_name}'
        )

    if not isinstance(per_reference_image_name_list, list):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation reference_image type illegal {type(per_reference_image_name_list)}'
        )
    else:
        for per_reference_image_name in per_reference_image_name_list:
            if not isinstance(per_reference_image_name, str):
                check_error_message_list.append(
                    f'{per_set_name}/{per_edited_image_name} annotation reference_image element type illegal {type(per_reference_image_name)}'
                )
            elif len(per_reference_image_name.strip()) == 0:
                check_error_message_list.append(
                    f'{per_set_name}/{per_edited_image_name} annotation reference_image element is empty'
                )

    # bool是int的子类,这里必须显式把bool判为非法类型
    if not isinstance(per_image_w, int) or isinstance(
            per_image_w,
            bool) or not isinstance(per_image_h, int) or isinstance(
                per_image_h, bool):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation width height type illegal {type(per_image_w)} {type(per_image_h)}'
        )
    elif per_image_w <= 0 or per_image_h <= 0:
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation width height illegal {per_image_w} {per_image_h}'
        )
    else:
        if min(per_image_w, per_image_h) < MIN_IMAGE_SHORT_SIDE:
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation image short side not match {per_image_w} {per_image_h}'
            )
        if max(per_image_w / per_image_h,
               per_image_h / per_image_w) > MAX_IMAGE_ASPECT_RATIO:
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation image aspect ratio not match {per_image_w} {per_image_h}'
            )

    per_ti2i_caption_pair_list, per_ti2i_caption_error_message_list = get_annotation_ti2i_caption_pair_list(
        per_set_name, per_edited_image_name, per_annotation,
        per_multi_caption_key_name_pair)
    for per_error_message in per_ti2i_caption_error_message_list:
        check_error_message_list.append(per_error_message)

    # 单条指令数据集这个列表只有1条、多条指令数据集有N条,
    # 两种规格走完全相同的一套校验
    for per_caption_key_name, per_ti2i_caption, per_caption_length_key_name, per_ti2i_caption_length in per_ti2i_caption_pair_list:
        if not isinstance(per_ti2i_caption, str):
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation {per_caption_key_name} type illegal {type(per_ti2i_caption)}'
            )
            continue

        if len(per_ti2i_caption.strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation {per_caption_key_name} is empty'
            )
        # 记录的指令长度必须和指令字符串的实际长度对得上
        if not isinstance(per_ti2i_caption_length, int) or isinstance(
                per_ti2i_caption_length, bool):
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation {per_caption_length_key_name} type illegal {type(per_ti2i_caption_length)}'
            )
        elif per_ti2i_caption_length != len(per_ti2i_caption):
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation {per_caption_length_key_name} not match {per_ti2i_caption_length} != {len(per_ti2i_caption)}'
            )

    return check_error_message_list


def check_single_caption_visual_placeholder(per_set_name,
                                            per_edited_image_name,
                                            per_caption_key_name,
                                            per_ti2i_caption,
                                            per_reference_image_num):
    """校验单条指令里的占位符编号与参考图像数量是否自洽,返回错误信息列表

    指令里的占位符编号必须正好是1...N,其中N = per_reference_image_num - 1,
    编辑前原图reference_image[0]隐式不占编号。N为0时指令里不允许有任何占位符,
    同一个编号最多重复MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM次,不允许跳号,
    也不允许出现无编号的[V*]。
    多条指令数据集的每条指令各调一次这个函数,报错信息里带上指令的key名,
    这样一眼就能看出是哪一条指令不自洽。
    """
    check_error_message_list = []

    per_placeholder_index_list = CAPTION_VISUAL_PLACEHOLDER_PATTERN.findall(
        per_ti2i_caption)

    per_placeholder_index_count_dict = {}
    per_no_index_placeholder_count = 0
    for per_placeholder_index in per_placeholder_index_list:
        if per_placeholder_index == '':
            per_no_index_placeholder_count += 1
            continue
        per_placeholder_index = int(per_placeholder_index)
        per_placeholder_index_count_dict[
            per_placeholder_index] = per_placeholder_index_count_dict.get(
                per_placeholder_index, 0) + 1

    # 落盘标注里不允许出现无编号的[V*]
    if per_no_index_placeholder_count > 0:
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation {per_caption_key_name} has no index visual placeholder'
        )

        return check_error_message_list

    for per_placeholder_index, per_placeholder_count in per_placeholder_index_count_dict.items(
    ):
        if per_placeholder_count > MAX_SAME_VISUAL_PLACEHOLDER_REPEAT_NUM:
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation {per_caption_key_name} visual placeholder [V{per_placeholder_index}*] repeat {per_placeholder_count} times'
            )

    per_expect_placeholder_num = per_reference_image_num - 1
    if per_expect_placeholder_num < 0:
        per_expect_placeholder_num = 0

    per_max_placeholder_index = max(per_placeholder_index_count_dict.keys(
    )) if len(per_placeholder_index_count_dict) > 0 else 0
    if per_max_placeholder_index != per_expect_placeholder_num:
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation {per_caption_key_name} max visual placeholder index not match {per_max_placeholder_index} != {per_expect_placeholder_num}'
        )

    for per_placeholder_index in range(1, per_expect_placeholder_num + 1):
        if per_placeholder_index not in per_placeholder_index_count_dict:
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} annotation {per_caption_key_name} visual placeholder [V{per_placeholder_index}*] not exists'
            )

    return check_error_message_list


def check_single_reference_image_num(per_set_name, per_edited_image_name,
                                     per_annotation,
                                     per_multi_caption_key_name_pair):
    """校验标注中涉及数量的值是否正确,返回错误信息列表

    做四条校验:
    1. reference_image_num必须等于reference_image这个list的实际长度;
    2. reference_image_num必须落在[MIN, MAX]区间内(第0张是编辑前原图,
       双参考图子集再多一张视觉参考图);
    3. 同一个样本对里的参考图像名不允许重复(重复意味着落盘时互相覆盖过);
    4. 指令里的占位符编号必须正好是1...N,其中N = reference_image_num - 1,
       具体判定见check_single_caption_visual_placeholder。
       多条指令数据集的每条指令都要各判一遍: 落盘的json里每条指令都会被训练
       用到,只判其中一条等于放过其余指令里的坏占位符。
    """
    check_error_message_list = []

    if not isinstance(per_annotation, dict):
        return check_error_message_list

    per_reference_image_name_list = per_annotation.get('reference_image', None)
    per_reference_image_num = per_annotation.get('reference_image_num', None)

    if not isinstance(per_reference_image_name_list, list):
        return check_error_message_list

    if not isinstance(per_reference_image_num, int) or isinstance(
            per_reference_image_num, bool):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation reference_image_num type illegal {type(per_reference_image_num)}'
        )

        return check_error_message_list

    # 校验1: 记录的数量必须和参考图像列表的实际长度对得上
    if per_reference_image_num != len(per_reference_image_name_list):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation reference_image_num not match {per_reference_image_num} != {len(per_reference_image_name_list)}'
        )

    # 校验2: 数量必须落在合法区间内
    if per_reference_image_num < MIN_REFERENCE_IMAGE_NUM or per_reference_image_num > MAX_REFERENCE_IMAGE_NUM:
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation reference_image_num illegal {per_reference_image_num}'
        )

    # 校验3: 同一个样本对里的参考图像名不允许重复
    if len(set(per_reference_image_name_list)) != len(
            per_reference_image_name_list):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} annotation reference_image name duplicated {per_reference_image_name_list}'
        )

    # 校验4: 占位符编号与参考图像数量交叉对账,
    # 数量一律用参考图像列表的实际长度现算,不采信记录的reference_image_num。
    # 指令的取法与check_single_annotation完全共用同一个函数,
    # 所以单条指令与多条指令两种规格走的是同一套判定;
    # 指令类型非法或指令字典结构非法都已经在check_single_annotation里报过,
    # 这里只跳过、不重复上报
    per_ti2i_caption_pair_list, _ = get_annotation_ti2i_caption_pair_list(
        per_set_name, per_edited_image_name, per_annotation,
        per_multi_caption_key_name_pair)

    for per_caption_key_name, per_ti2i_caption, _, _ in per_ti2i_caption_pair_list:
        if not isinstance(per_ti2i_caption, str):
            continue

        per_caption_error_message_list = check_single_caption_visual_placeholder(
            per_set_name, per_edited_image_name, per_caption_key_name,
            per_ti2i_caption, len(per_reference_image_name_list))
        for per_error_message in per_caption_error_message_list:
            check_error_message_list.append(per_error_message)

    return check_error_message_list


def check_single_image_name(per_set_name, per_edited_image_name,
                            per_annotation):
    """校验单个样本对的落盘图像名是否符合命名规范,返回错误信息列表

    上游resave时编辑后图像名固定形如<...>_edited.jpg,参考图像名固定形如
    <编辑后图像名去掉_edited.jpg>_<原始名前缀>_reference.jpg,
    所以参考图像名一定以编辑后图像名的基名为前缀,靠这个前缀关系就能查出
    "把别的样本对的图像混进这个样本对文件夹"这类问题。
    """
    check_error_message_list = []

    if not per_edited_image_name.endswith(SAVE_EDITED_IMAGE_NAME_SUFFIX):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} edited image name suffix not match'
        )
    if per_edited_image_name != per_edited_image_name.lower():
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} edited image name not all lower case'
        )
    if not VALID_IMAGE_NAME_PATTERN.match(per_edited_image_name):
        check_error_message_list.append(
            f'{per_set_name}/{per_edited_image_name} edited image name has illegal char'
        )

    per_edit_pair_name_prefix = get_edit_pair_name_prefix(
        per_edited_image_name)

    if not isinstance(per_annotation, dict):
        return check_error_message_list

    per_reference_image_name_list = per_annotation.get('reference_image', None)
    if not isinstance(per_reference_image_name_list, list):
        return check_error_message_list

    for per_reference_image_name in per_reference_image_name_list:
        if not isinstance(per_reference_image_name, str):
            continue

        if not per_reference_image_name.endswith(
                SAVE_REFERENCE_IMAGE_NAME_SUFFIX):
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} reference image name suffix not match {per_reference_image_name}'
            )
        if per_reference_image_name != per_reference_image_name.lower():
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} reference image name not all lower case {per_reference_image_name}'
            )
        if not VALID_IMAGE_NAME_PATTERN.match(per_reference_image_name):
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} reference image name has illegal char {per_reference_image_name}'
            )
        if len(per_edit_pair_name_prefix
               ) > 0 and not per_reference_image_name.startswith(
                   per_edit_pair_name_prefix):
            check_error_message_list.append(
                f'{per_set_name}/{per_edited_image_name} reference image name prefix not match {per_reference_image_name}'
            )

    return check_error_message_list


def get_edit_pair_folder_name(per_edited_image_name):
    """标注key就是编辑后图像名,去掉.jpg后缀就是这个样本对的文件夹名"""
    if per_edited_image_name.endswith(SAVE_IMAGE_NAME_SUFFIX):
        return per_edited_image_name[:-len(SAVE_IMAGE_NAME_SUFFIX)]

    return per_edited_image_name


def get_edit_pair_name_prefix(per_edited_image_name):
    """取编辑后图像名去掉_edited.jpg后的基名,同一个样本对的参考图像名都以它为前缀"""
    if per_edited_image_name.endswith(SAVE_EDITED_IMAGE_NAME_SUFFIX):
        return per_edited_image_name[:-len(SAVE_EDITED_IMAGE_NAME_SUFFIX)]

    return ''


def get_empty_check_result_dict():
    """初始化单次复核的计数与样例信息容器"""
    check_result_dict = {
        'total_annotation_count': 0,
        'total_disk_edit_pair_folder_count': 0,
        'total_disk_image_count': 0,
        'checked_image_count': 0,
        'valid_edit_pair_count': 0,
    }
    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        check_result_dict[f'{per_error_type_name}_count'] = 0
        check_result_dict[f'{per_error_type_name}_message_list'] = []

    return check_result_dict


def add_check_error_message(check_result_dict, per_error_type_name,
                            per_error_message):
    """按问题类别累计条数,样例信息只保留前MAX_ERROR_MESSAGE_NUM条"""
    check_result_dict[f'{per_error_type_name}_count'] += 1
    per_error_message_list = check_result_dict[
        f'{per_error_type_name}_message_list']
    if len(per_error_message_list) < MAX_ERROR_MESSAGE_NUM:
        per_error_message_list.append(per_error_message)

    return


def merge_check_result_dict(check_result_dict, per_folder_check_result_dict):
    """把单个分片的复核结果合并进数据集级别的复核结果"""
    for per_key_name in [
            'total_annotation_count',
            'total_disk_edit_pair_folder_count',
            'total_disk_image_count',
            'checked_image_count',
            'valid_edit_pair_count',
    ]:
        check_result_dict[per_key_name] += per_folder_check_result_dict[
            per_key_name]

    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        check_result_dict[f'{per_error_type_name}_count'] += (
            per_folder_check_result_dict[f'{per_error_type_name}_count'])
        per_error_message_list = check_result_dict[
            f'{per_error_type_name}_message_list']
        for per_error_message in per_folder_check_result_dict[
                f'{per_error_type_name}_message_list']:
            if len(per_error_message_list) >= MAX_ERROR_MESSAGE_NUM:
                break
            per_error_message_list.append(per_error_message)

    return check_result_dict


def get_check_error_count(check_result_dict):
    """统计所有问题类别的问题总条数"""
    check_error_count = 0
    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        check_error_count += check_result_dict[f'{per_error_type_name}_count']

    return check_error_count


def get_all_check_folder_pair(root_dataset_path):
    """扫描数据集目录结构,配出所有<folder_name>文件夹与<folder_name>.json的任务对

    这一步只做目录层面的结构校验(文件夹与json是否一一对应、有没有杂项文件),
    单个分片内部的逐样本对校验留到后面多进程里做。
    任务对里带上数据集目录名,是为了让worker能查
    EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_DICT(它是按数据集分组的)。
    """
    per_dataset_dir_name = os.path.basename(root_dataset_path.rstrip('/'))
    structure_check_result_dict = get_empty_check_result_dict()
    check_warning_message_list = []

    folder_check_pair_list = []
    set_folder_count_dict = {}

    try:
        per_dataset_dir_name_list = sorted(os.listdir(root_dataset_path))
    except Exception as e:
        raise Exception(f'{root_dataset_path} listdir error {e}')

    for per_set_name in per_dataset_dir_name_list:
        per_set_dir_path = os.path.join(root_dataset_path, per_set_name)

        # 数据集根目录下除子集目录外还有resave时留下的落盘统计json,
        # 这个文件不影响样本对完整性,只作为warning上报
        if not os.path.isdir(per_set_dir_path):
            if per_set_name in SKIP_ROOT_FILE_NAME_LIST:
                check_warning_message_list.append(
                    f'{per_set_name} is a skipped root file')
            else:
                check_warning_message_list.append(
                    f'{per_set_name} is not a set dir')
            continue

        if len(CHECK_SET_NAME_LIST
               ) > 0 and per_set_name not in CHECK_SET_NAME_LIST:
            continue

        try:
            per_set_sub_name_list = sorted(os.listdir(per_set_dir_path))
        except Exception as e:
            add_check_error_message(structure_check_result_dict,
                                    'folder_json_not_paired',
                                    f'{per_set_name} listdir error {e}')
            continue

        per_set_folder_name_list = []
        per_set_json_name_prefix_list = []
        for per_sub_name in per_set_sub_name_list:
            per_sub_path = os.path.join(per_set_dir_path, per_sub_name)
            if os.path.isdir(per_sub_path):
                per_set_folder_name_list.append(per_sub_name)
            elif per_sub_name.endswith('.json'):
                per_set_json_name_prefix_list.append(
                    os.path.splitext(per_sub_name)[0])
            else:
                add_check_error_message(
                    structure_check_result_dict, 'unexpected_file',
                    f'{per_set_name}/{per_sub_name} is an unexpected file in set dir'
                )

        if len(per_set_folder_name_list) == 0 and len(
                per_set_json_name_prefix_list) == 0:
            check_warning_message_list.append(f'{per_set_name} set dir empty')
            continue

        per_set_folder_name_set = set(per_set_folder_name_list)
        per_set_json_name_prefix_set = set(per_set_json_name_prefix_list)

        # 有文件夹没json: 这个分片的全部样本对都是孤立图像,无法逐样本对复核
        for per_folder_name in per_set_folder_name_list:
            if per_folder_name not in per_set_json_name_prefix_set:
                add_check_error_message(
                    structure_check_result_dict, 'folder_json_not_paired',
                    f'{per_set_name}/{per_folder_name} annotation json not exists'
                )

        # 有json没文件夹: 这个分片的全部标注都是孤立标注,无法逐样本对复核
        for per_json_name_prefix in per_set_json_name_prefix_list:
            if per_json_name_prefix not in per_set_folder_name_set:
                add_check_error_message(
                    structure_check_result_dict, 'folder_json_not_paired',
                    f'{per_set_name}/{per_json_name_prefix} edit pair folder not exists'
                )

        # 只有文件夹和json都在的分片才进入逐样本对复核
        per_set_paired_folder_name_list = sorted(
            per_set_folder_name_set
            & per_set_json_name_prefix_set)
        set_folder_count_dict[per_set_name] = len(
            per_set_paired_folder_name_list)

        for per_folder_index, per_folder_name in enumerate(
                per_set_paired_folder_name_list):
            # 每个子集最后一个分片允许不满PER_FOLDER_EDIT_PAIR_NUM个样本对
            per_is_last_folder = (
                per_folder_index == len(per_set_paired_folder_name_list) - 1)
            folder_check_pair_list.append([
                per_dataset_dir_name,
                per_set_name,
                per_folder_name,
                os.path.join(per_set_dir_path, per_folder_name),
                os.path.join(per_set_dir_path, f'{per_folder_name}.json'),
                per_is_last_folder,
            ])

    return [
        folder_check_pair_list,
        set_folder_count_dict,
        structure_check_result_dict,
        check_warning_message_list,
    ]


def check_single_edit_pair_folder(per_dataset_dir_name, per_set_name,
                                  per_folder_name, per_edit_pair_folder_path,
                                  per_edited_image_name, per_annotation,
                                  per_folder_check_result_dict):
    """复核单个样本对文件夹: 与标注双向比对文件名,并逐张校验图像完整性

    上游图像都放在NAS上,逐张图像打一次os.path.exists/getsize就是一次网络往返,
    而本数据集是"一个样本对独占一个文件夹、里面只放2~3张图"的结构,
    所以这里每个样本对文件夹只做一次os.scandir,一次遍历同时拿到全部文件名和字节数,
    之后所有存在性判定都退化成集合查表、字节数判定直接复用scandir的结果。
    返回这个样本对命中的问题条数,0条才算一个完好的样本对。
    """
    per_edit_pair_error_count = 0

    per_disk_image_name_dict = {}
    per_disk_unexpected_name_list = []
    try:
        with os.scandir(per_edit_pair_folder_path) as per_scandir_iterator:
            for per_dir_entry in per_scandir_iterator:
                if per_dir_entry.is_dir(
                ) or not per_dir_entry.name.endswith(SAVE_IMAGE_NAME_SUFFIX):
                    per_disk_unexpected_name_list.append(per_dir_entry.name)
                    continue

                try:
                    per_disk_image_name_dict[
                        per_dir_entry.name] = per_dir_entry.stat().st_size
                except Exception:
                    per_disk_image_name_dict[per_dir_entry.name] = None
    except Exception as e:
        # 标注里记录的样本对在磁盘上打不开,等价于这个样本对的图像文件不存在
        add_check_error_message(
            per_folder_check_result_dict, 'orphan_annotation',
            f'{per_set_name}/{per_folder_name}/{per_edited_image_name} edit pair folder scandir error {e}'
        )

        return per_edit_pair_error_count + 1

    per_folder_check_result_dict['total_disk_image_count'] += len(
        per_disk_image_name_dict)

    # 样本对文件夹里只应该有这个样本对的.jpg图像,其他文件/子目录都是残留
    for per_unexpected_name in sorted(per_disk_unexpected_name_list):
        add_check_error_message(
            per_folder_check_result_dict, 'unexpected_file',
            f'{per_set_name}/{per_folder_name}/{per_edited_image_name} has an unexpected file {per_unexpected_name}'
        )
        per_edit_pair_error_count += 1

    # 标注里这个样本对应该有的全部图像: 1张编辑后图像 + 全部参考图像
    per_annotation_image_name_list = [per_edited_image_name]
    if isinstance(per_annotation, dict) and isinstance(
            per_annotation.get('reference_image', None), list):
        for per_reference_image_name in per_annotation['reference_image']:
            if isinstance(
                    per_reference_image_name, str
            ) and per_reference_image_name not in per_annotation_image_name_list:
                per_annotation_image_name_list.append(per_reference_image_name)

    per_annotation_image_name_set = set(per_annotation_image_name_list)

    # 磁盘上有图像但这条标注没有引用它,即孤立图像
    for per_disk_image_name in sorted(
            set(per_disk_image_name_dict.keys()) -
            per_annotation_image_name_set):
        add_check_error_message(
            per_folder_check_result_dict, 'orphan_image',
            f'{per_set_name}/{per_folder_name}/{per_edited_image_name} image has no annotation {per_disk_image_name}'
        )
        per_edit_pair_error_count += 1

    per_annotation_image_w, per_annotation_image_h = None, None
    if isinstance(per_annotation, dict):
        per_annotation_image_w = per_annotation.get('width', None)
        per_annotation_image_h = per_annotation.get('height', None)

    # 只有reference_image[0](编辑前原图)的尺寸严格等于width/height:
    # reference_image[k>=1]走的是长边对齐、尺寸本就不等,
    # 豁免子集是完全原样落盘的、尺寸也本就可以不等
    per_first_reference_image_name = ''
    if isinstance(per_annotation, dict) and isinstance(
            per_annotation.get('reference_image', None), list) and len(
                per_annotation['reference_image']) > 0 and isinstance(
                    per_annotation['reference_image'][0], str):
        per_first_reference_image_name = per_annotation['reference_image'][0]

    per_exempt_aspect_ratio_align_flag = per_set_name in EXEMPT_ASPECT_RATIO_ALIGN_SET_NAME_DICT.get(
        per_dataset_dir_name, [])

    for per_annotation_image_name in per_annotation_image_name_list:
        # 标注里记录的图像在磁盘上找不到,即孤立标注
        if per_annotation_image_name not in per_disk_image_name_dict:
            add_check_error_message(
                per_folder_check_result_dict, 'orphan_annotation',
                f'{per_set_name}/{per_folder_name}/{per_edited_image_name} annotation has no image file {per_annotation_image_name}'
            )
            per_edit_pair_error_count += 1
            continue

        per_is_edited_image = (
            per_annotation_image_name == per_edited_image_name)
        # 上游resave保证reference_image[0]的尺寸严格等于width/height,
        # 所以它和编辑后图像一样要比对宽高;
        # reference_image[k>=1](长边对齐)与豁免子集(原样落盘)只校验存在且完整
        per_is_first_reference_image = (not per_is_edited_image
                                        and per_annotation_image_name
                                        == per_first_reference_image_name)
        per_image_path = os.path.join(per_edit_pair_folder_path,
                                      per_annotation_image_name)
        per_check_image_size_flag = per_is_edited_image or (
            CHECK_REFERENCE_IMAGE_SIZE and per_is_first_reference_image
            and not per_exempt_aspect_ratio_align_flag)
        per_image_error_type_name, per_image_error_message = check_single_image_file(
            per_image_path,
            per_disk_image_name_dict[per_annotation_image_name],
            per_annotation_image_w, per_annotation_image_h,
            per_check_image_size_flag)
        per_folder_check_result_dict['checked_image_count'] += 1

        if per_image_error_type_name is not None:
            add_check_error_message(per_folder_check_result_dict,
                                    per_image_error_type_name,
                                    per_image_error_message)
            per_edit_pair_error_count += 1

    return per_edit_pair_error_count


def check_single_folder(folder_check_pair):
    """复核单个分片: 标注与磁盘样本对文件夹双向比对,并逐样本对校验图像与标注完整性"""
    per_dataset_dir_name, per_set_name, per_folder_name, per_folder_dir_path, per_folder_json_path, per_is_last_folder = folder_check_pair

    per_folder_check_result_dict = get_empty_check_result_dict()
    per_folder_check_warning_message_list = []

    # 标注json无法解析时整个分片的标注都不可用,只上报一条问题后直接返回,
    # 此时磁盘上的样本对既无法判定是否孤立,也无法逐样本对复核
    try:
        with open(per_folder_json_path, 'r',
                  encoding='UTF-8') as load_json_file:
            per_folder_annotation_dict = json.load(load_json_file)
    except Exception as e:
        add_check_error_message(
            per_folder_check_result_dict, 'broken_annotation_json',
            f'{per_set_name}/{per_folder_name}.json load error {e}')

        return [
            per_dataset_dir_name,
            per_set_name,
            per_folder_name,
            per_folder_check_result_dict,
            per_folder_check_warning_message_list,
        ]

    if not isinstance(per_folder_annotation_dict, dict):
        add_check_error_message(
            per_folder_check_result_dict, 'broken_annotation_json',
            f'{per_set_name}/{per_folder_name}.json is not a dict')

        return [
            per_dataset_dir_name,
            per_set_name,
            per_folder_name,
            per_folder_check_result_dict,
            per_folder_check_warning_message_list,
        ]

    # 分片文件夹下每个成员都应该是一个样本对文件夹,一次scandir就能同时拿到
    # 全部样本对文件夹名和混进来的残留文件名
    per_disk_edit_pair_folder_name_list = []
    per_disk_unexpected_name_list = []
    try:
        with os.scandir(per_folder_dir_path) as per_scandir_iterator:
            for per_dir_entry in per_scandir_iterator:
                if per_dir_entry.is_dir():
                    per_disk_edit_pair_folder_name_list.append(
                        per_dir_entry.name)
                else:
                    per_disk_unexpected_name_list.append(per_dir_entry.name)
    except Exception as e:
        add_check_error_message(
            per_folder_check_result_dict, 'folder_json_not_paired',
            f'{per_set_name}/{per_folder_name} scandir error {e}')

        return [
            per_dataset_dir_name,
            per_set_name,
            per_folder_name,
            per_folder_check_result_dict,
            per_folder_check_warning_message_list,
        ]

    for per_unexpected_name in sorted(per_disk_unexpected_name_list):
        add_check_error_message(
            per_folder_check_result_dict, 'unexpected_file',
            f'{per_set_name}/{per_folder_name}/{per_unexpected_name} is an unexpected file in folder dir'
        )

    per_disk_edit_pair_folder_name_set = set(
        per_disk_edit_pair_folder_name_list)
    per_annotation_edited_image_name_set = set(
        per_folder_annotation_dict.keys())
    # 标注key就是编辑后图像名,去掉.jpg后缀才是磁盘上的样本对文件夹名
    per_annotation_edit_pair_folder_name_set = set()
    for per_edited_image_name in per_annotation_edited_image_name_set:
        per_annotation_edit_pair_folder_name_set.add(
            get_edit_pair_folder_name(per_edited_image_name))

    per_folder_check_result_dict['total_annotation_count'] = len(
        per_annotation_edited_image_name_set)
    per_folder_check_result_dict['total_disk_edit_pair_folder_count'] = len(
        per_disk_edit_pair_folder_name_set)

    # 除每个子集最后一个分片外都应该是满PER_FOLDER_EDIT_PAIR_NUM个样本对,
    # 不满不代表样本对损坏,只作为warning上报
    if not per_is_last_folder and len(
            per_annotation_edited_image_name_set) != PER_FOLDER_EDIT_PAIR_NUM:
        per_folder_check_warning_message_list.append(
            f'{per_set_name}/{per_folder_name} annotation num not full {len(per_annotation_edited_image_name_set)} != {PER_FOLDER_EDIT_PAIR_NUM}'
        )

    # 磁盘上有样本对文件夹但标注json里没有对应标注,即孤立图像
    for per_edit_pair_folder_name in sorted(
            per_disk_edit_pair_folder_name_set -
            per_annotation_edit_pair_folder_name_set):

        add_check_error_message(
            per_folder_check_result_dict, 'orphan_image',
            f'{per_set_name}/{per_folder_name}/{per_edit_pair_folder_name} edit pair folder has no annotation'
        )

    # 指令是单条还是多条指令字典是整个数据集统一的规格,这里算一次就够,
    # 不必逐样本对重复查表
    per_multi_caption_key_name_pair = get_multi_caption_key_name_pair(
        per_dataset_dir_name)

    for per_edited_image_name in sorted(per_annotation_edited_image_name_set):
        per_annotation = per_folder_annotation_dict[per_edited_image_name]

        per_annotation_error_message_list = check_single_annotation(
            per_set_name, per_edited_image_name, per_annotation,
            per_multi_caption_key_name_pair)
        for per_error_message in per_annotation_error_message_list:
            add_check_error_message(per_folder_check_result_dict,
                                    'invalid_annotation', per_error_message)

        per_reference_image_num_error_message_list = check_single_reference_image_num(
            per_set_name, per_edited_image_name, per_annotation,
            per_multi_caption_key_name_pair)
        for per_error_message in per_reference_image_num_error_message_list:
            add_check_error_message(per_folder_check_result_dict,
                                    'invalid_reference_image_num',
                                    per_error_message)

        per_image_name_error_message_list = check_single_image_name(
            per_set_name, per_edited_image_name, per_annotation)
        for per_error_message in per_image_name_error_message_list:
            add_check_error_message(per_folder_check_result_dict,
                                    'invalid_image_name', per_error_message)

        per_edit_pair_folder_name = get_edit_pair_folder_name(
            per_edited_image_name)

        # 标注json里有标注但磁盘上没有对应的样本对文件夹,即孤立标注
        if per_edit_pair_folder_name not in per_disk_edit_pair_folder_name_set:
            add_check_error_message(
                per_folder_check_result_dict, 'orphan_annotation',
                f'{per_set_name}/{per_folder_name}/{per_edited_image_name} annotation has no edit pair folder'
            )
            continue

        per_edit_pair_error_count = check_single_edit_pair_folder(
            per_dataset_dir_name, per_set_name, per_folder_name,
            os.path.join(per_folder_dir_path,
                         per_edit_pair_folder_name), per_edited_image_name,
            per_annotation, per_folder_check_result_dict)

        # 编辑后图像和全部参考图像都存在且完整、标注key-value对完整、
        # 数量相关的值正确、图像名规范,才算一个完好的样本对
        if per_edit_pair_error_count == 0 and len(
                per_annotation_error_message_list) == 0 and len(
                    per_reference_image_num_error_message_list) == 0 and len(
                        per_image_name_error_message_list) == 0:
            per_folder_check_result_dict['valid_edit_pair_count'] += 1

    return [
        per_dataset_dir_name,
        per_set_name,
        per_folder_name,
        per_folder_check_result_dict,
        per_folder_check_warning_message_list,
    ]


def check_dataset(root_dataset_path, save_check_result_path):
    dataset_dir_name = os.path.basename(root_dataset_path.rstrip('/'))

    # 校验级别写错时必须直接失败,不能静默退化成某一档校验后给出"复核通过"的结论
    if CHECK_IMAGE_LEVEL not in VALID_CHECK_IMAGE_LEVEL_LIST:
        raise Exception(f'{CHECK_IMAGE_LEVEL} is an illegal check image level')

    if not os.path.isdir(root_dataset_path):
        raise Exception(f'{root_dataset_path} is not a dir')

    # 复核报告一律写到数据集目录之外,保证本脚本不对数据集做任何修改
    if os.path.abspath(save_check_result_path).startswith(
            os.path.abspath(root_dataset_path) + os.sep):
        raise Exception(
            f'{save_check_result_path} must be outside {root_dataset_path}')

    folder_check_pair_list, set_folder_count_dict, check_result_dict, check_warning_message_list = get_all_check_folder_pair(
        root_dataset_path)

    print('1111', dataset_dir_name, 'check set:', len(set_folder_count_dict),
          'check folder:', len(folder_check_pair_list), 'structure error:',
          get_check_error_count(check_result_dict), 'structure warning:',
          len(check_warning_message_list))

    for per_check_warning_message in check_warning_message_list:
        print('2222', per_check_warning_message)

    per_set_check_result_dict = {}
    for per_set_name in set_folder_count_dict.keys():
        per_set_check_result_dict[per_set_name] = get_empty_check_result_dict()

    with Pool(processes=min(PROCESS_NUM, max(len(folder_check_pair_list),
                                             1))) as pool:
        for per_folder_check_result in tqdm(pool.imap_unordered(
                check_single_folder, folder_check_pair_list),
                                            total=len(folder_check_pair_list)):
            _, per_set_name, per_folder_name, per_folder_check_result_dict, per_folder_check_warning_message_list = per_folder_check_result

            merge_check_result_dict(check_result_dict,
                                    per_folder_check_result_dict)
            merge_check_result_dict(per_set_check_result_dict[per_set_name],
                                    per_folder_check_result_dict)

            for per_check_warning_message in per_folder_check_warning_message_list:
                check_warning_message_list.append(per_check_warning_message)
                print('2222', per_check_warning_message)

            per_folder_check_error_count = get_check_error_count(
                per_folder_check_result_dict)
            if per_folder_check_error_count > 0:
                print('3333', f'{per_set_name}/{per_folder_name}',
                      'check error:', per_folder_check_error_count)

    check_error_count = get_check_error_count(check_result_dict)

    print('4444', dataset_dir_name, 'total annotation:',
          check_result_dict['total_annotation_count'],
          'total disk edit pair folder:',
          check_result_dict['total_disk_edit_pair_folder_count'],
          'total disk image:', check_result_dict['total_disk_image_count'],
          'checked image:', check_result_dict['checked_image_count'],
          'valid edit pair:', check_result_dict['valid_edit_pair_count'],
          'check error:', check_error_count, 'check warning:',
          len(check_warning_message_list))

    for per_error_type_name in CHECK_ERROR_TYPE_NAME_LIST:
        print('5555', per_error_type_name,
              check_result_dict[f'{per_error_type_name}_count'])
        for per_error_message in check_result_dict[
                f'{per_error_type_name}_message_list'][:10]:
            print('6666', per_error_message)

    # 复核报告一律写到数据集目录之外,保证本脚本不对数据集做任何修改
    os.makedirs(save_check_result_path, exist_ok=True)
    save_check_result_json_path = os.path.join(
        save_check_result_path,
        f'{dataset_dir_name}_integrity_check_result.json')
    save_check_result_dict = {
        'dataset_dir_name':
        dataset_dir_name,
        'root_dataset_path':
        root_dataset_path,
        'check_image_level':
        CHECK_IMAGE_LEVEL,
        'total_set_count':
        len(set_folder_count_dict),
        'total_folder_count':
        len(folder_check_pair_list),
        'check_error_count':
        check_error_count,
        'check_warning_count':
        len(check_warning_message_list),
        'check_warning_message_list':
        check_warning_message_list[:MAX_ERROR_MESSAGE_NUM],
        'set_folder_count_dict':
        set_folder_count_dict,
        'check_result_dict':
        check_result_dict,
        'per_set_check_result_dict':
        per_set_check_result_dict,
    }
    with open(save_check_result_json_path, 'w',
              encoding='UTF-8') as save_json_file:
        json.dump(save_check_result_dict, save_json_file, ensure_ascii=False)

    print('7777', save_check_result_json_path)

    if check_error_count > 0:
        # 复核不通过必须让上层感知,不能静默留下坏样本对或孤立文件
        raise Exception(
            f'check dataset {dataset_dir_name} error num {check_error_count}')

    return


if __name__ == '__main__':
    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/BM-6M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/ConceptEdit-12M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/CrispEdit-2M'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/FoundIR'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/ImgEdit'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/InterEdit'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/InterEdit-Mask'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/ScaleEdit'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/UnicEdit'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/VINS120K'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)

    root_dataset_path = r'/root/autodl-tmp/ti2i_datasets/X2Edit'
    save_check_result_path = r'/root/code/SimpleGeneration_pytorch_training_examples/SimpleGeneration/universal_generation_edit/generation_edit_dataset_preprocessing3/check_result2'
    check_dataset(root_dataset_path, save_check_result_path)
