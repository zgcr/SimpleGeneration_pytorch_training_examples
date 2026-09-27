import os
import re
import json
import hashlib
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DATASET_NAME = 'faceid_6m'

SAVE_DATASET_DIR_NAME = 'FaceID-6M'

# ==============================================================================
# 【数据集类型判定】FaceID-6M 只处理成纯文生图(text-to-image)数据集,
# 不出图像编辑(ti2i)脚本。所以本目录下只有这一个010的t2i脚本。
#
# 上游010.unzip_faceid_6m_dataset.py把1.4T的单流tar.gz解成
#   images/laion_512/<id//10000:05d>/<id>.png        目标图(生成后图) 6231808张
#   images/laion_512/face/<id//10000:05d>/<id>.png   人脸裁图(参考图) 6231808张
#   unzip_annotations/laion_512_00000~00062.jsonl    63个汇总标注分片
# 每个样本对是"1条LAION网页alt-text + 1张目标图 + 1张同一个人的人脸裁图"。
#
# 【为什么不做图像编辑数据集(实测依据)】
# 唯一能当参考图的只有那张人脸裁图,但实测已确认它**就是目标图自身的一块原像素裁剪**:
# 随机抽36对做全分辨率模板匹配(cv2.matchTemplate + TM_CCOEFF_NORMED),
# 最佳匹配相关系数全部落在0.998~1.000、匹配坐标与标注bbox的偏移≤1像素
# (其中5对匹配到了另一个坐标,是同一张图里有多张人脸,不是不匹配);
# 人脸裁图面积占目标图的比例中位数只有2%(p90 12%/p99 30%/max 69%)。
# 再加上这个数据集**根本没有编辑指令**(caption是网页alt-text而不是指令),
# 所以它构成的不是instruction-based editing对,而是InstantID那种
# "ID保持定制化生成"对。硬做成ti2i会有三处与本仓库既有6个ti2i库冲突:
#   1) reference_image[0]的语义被改变: 其余6个库里它恒为"编辑前原图"
#      (与edited_image同构图、宽高比一致时还会与它逐像素对齐),
#      这里却是一张只占2%面积的人脸小图,既不像素对齐也不是"编辑前"状态;
#   2) 参考图像素被目标图完整包含(自监督泄漏): 训练目标退化成
#      "以人脸小图为条件把周围内容补出来"; X2Edit的personalized_generation
#      之所以成立,是因为它的主体参考图来自**另一张**照片;
#   3) 没有指令: 把alt-text塞进ti2i_caption后,模型看到"1张参考图 + 一段描述"时
#      无法区分"复原/编辑这张图"和"保持这张脸的身份去生成",会和FoundIR/X2Edit打架。
# 按方案确认整个 images/laion_512/face/ 目录(6231808张、约98GB)不使用。
#
# 【上游标注每行固定14个key(全量6231808行key组合唯一、0缺失)】
# sample_key / dataset_task_type / image_path / face_image_path /
# has_face_embedding / caption / caption_key_name / image_file / height /
# width / face_file / bbox / landmarks / insightface_feature_file
#
# 本脚本只读这63个jsonl定位样本对,绝不os.walk图像目录:
# 上游一共解出1246万个小文件,扫一遍目录树在NAS上不可接受。
# ==============================================================================

# 上游010解压脚本按行号合并出的汇总标注目录
LOAD_ANNOTATION_DIR_NAME = 'unzip_annotations'

LOAD_ANNOTATION_FILE_SUFFIX = '.jsonl'

# 上游只有laion_512这一个系列(tar内唯一的顶层目录名),
# 它既是上游的系列名,也是本脚本切分后子集目录名的前缀
LOAD_SERIES_DIR_NAME_LIST = [
    'laion_512',
]

# 实测上游jsonl分片数(前62片各100000行 + 末片31808行),数量不对说明上游没跑完
EXPECTED_SERIES_ANNOTATION_FILE_NUM_DICT = {
    'laion_512': 63,
}

# 实测上游有效样本对数,与上游对账报告unzip_check_missing_images.json里的
# total_valid_sample_pair_count完全一致(该报告同时是
# 0缺图/0 orphan/0重名/0未知成员/reach_gzip_end=True),直接当完整性ground truth
EXPECTED_SERIES_ROW_COUNT_DICT = {
    'laion_512': 6231808,
}

EXPECTED_TOTAL_ROW_COUNT = 6231808

ANNOTATION_IMAGE_KEY_NAME = 'image_path'

# sample_key就是jsonl里的行号,也等于落盘图像的文件名前缀(纯数字),是全局唯一样本id。
# 实测全量6231808行里"图像名前缀是纯数字"与"前缀 == str(sample_key)"各0例不符,
# 所以它可以直接当保存图像名的后半段,保存名天然全局唯一
ANNOTATION_IMAGE_NAME_KEY_NAME = 'sample_key'

# t2i描述。上游已把原始jsonl里命中的文本字段(additional_feature)统一写进caption,
# 实测抽样20万行 caption 与 additional_feature 逐字100%相同,只有这一路文本可用
ANNOTATION_CAPTION_KEY_NAME = 'caption'

# 上游标注里记录的原图宽高。本脚本**不**拿它做最终分辨率过滤(万一它和实际图像不一致
# 就会出偏差),只在解码后顺手比对一次,不一致的条数只上报不丢样本
# (实测抽样320张0例不符)
ANNOTATION_IMAGE_WIDTH_KEY_NAME = 'width'

ANNOTATION_IMAGE_HEIGHT_KEY_NAME = 'height'

# 上游给每行都标了任务类型,实测全量6231808行恒为这个值。
# 一旦出现别的取值说明上游规格变了(比如混进了编辑样本),这种样本不能进t2i数据集,
# 直接丢弃并计数上报
ANNOTATION_TASK_TYPE_KEY_NAME = 'dataset_task_type'

EXPECTED_ANNOTATION_TASK_TYPE = 'text_to_image_faceid_customization'

# 上游标注里另外这些属性,按方案确认全部丢弃,新标注只留
# width/height/t2i_caption/t2i_caption_length:
# bbox                     : 人脸框[x1,y1,x2,y2]浮点,实测与人脸裁图尺寸一致。
#                            新标注只有4个字段、放不下它,按方案确认丢弃;
#                            想训InstantID那种kps条件的话必须在这一步之前另存
# landmarks                : 5点人脸关键点(画kps条件图用),同上丢弃
# has_face_embedding       : 实测全量恒为True。注意ArcFace 512维id特征本身
#                            **上游根本没落盘**(010的EXTRACT_FACE_EMBEDDING_FLAG
#                            默认False),要用必须把那个开关改True重跑整个1.4T解压
# face_image_path          : 人脸裁图路径,按A1不做ti2i、整块不使用
# image_file / face_file / insightface_feature_file :
#                            原作者的gpfs绝对路径(README明确要求忽略它们、
#                            只用行号定位),丢弃
# width / height           : 由真实解码后的shape得到的width/height承载
# sample_key               : 已内含在保存图像名里
# dataset_task_type        : 只用于校验,不落盘
# caption_key_name         : 上游命中的原始文本字段名(恒为additional_feature)
SAVE_IMAGE_NAME_SUFFIX = '.jpg'

# 保存图像名里只允许小写字母/数字/下划线/中划线/点
VALID_IMAGE_NAME_PATTERN = re.compile(r'^[a-z0-9_\-\.]+$')

# 原始图像名前缀必须是纯数字(即sample_key),不符的样本没法保证保存图像名全局唯一
# (可能去覆盖别的样本),直接丢弃并上报。实测全量0例不符
VALID_IMAGE_NAME_PREFIX_PATTERN = re.compile(r'^\d+$')

# 只保留RGB三通道图与灰度图,P图/RGBA图/CMYK图等一律过滤掉。
# 实测抽样3000张目标图100%是RGB且100%是JPEG内容
# (上游成员后缀叫.png但内容其实是JPEG,这是LAION原始数据自带的规格,
#  本脚本一律按实际解码结果处理、不看后缀)
VALID_IMAGE_MODE_LIST = [
    'RGB',
    'L',
]

# 本机128核,这里和001/003/009保持一致取32。本数据集要对623万张图各解码两遍
# (jsonl扫描阶段校验一次并顺手算md5、写盘阶段重编码一次),想跑快可以直接调大这个常量
PROCESS_NUM = 32

PER_FOLDER_IMAGE_NUM = 10000

# 每个子集目录放100个文件夹(即100万张图)。本数据集最终落盘约445万张会切出约446个
# 文件夹,全塞进一个目录下NAS元数据压力太大,所以每100个文件夹再归入一个子集目录,
# 子集目录形如laion_512_000/laion_512_001/...(约5个)
PER_SET_FOLDER_NUM = 100

# 只有一个系列,它也需要按PER_SET_FOLDER_NUM切成带编号的子集目录,
# 保证与003/009的子集目录命名规格一致
SPLIT_SET_DIR_SERIES_NAME_LIST = [
    'laion_512',
]

# 实测全量6231808行的目标图短边 min 513/p50 683/p99 2048/max 10000,
# 所以短边这条过滤实际一条都不会命中(实测0条);
# 宽高比 p50 1.4/p99 2.0/max 26.4,大于8的有205条,会被这条判掉。
# 短边那条即使一条都不命中也照写: 只要有一张图不符合就必须被判掉,
# 不能靠"实测都合格"这个假设去省掉校验
MIN_IMAGE_SHORT_SIDE = 64

MAX_IMAGE_ASPECT_RATIO = 8

# 实测全量caption(strip后)长度 min 4/p1 16/p50 61/p90 181/p99 461/p999 892/
# max 32023。小于10的只有1238条(0.0199%),按001/003的惯例取10
MIN_CAPTION_LENGTH = 10

# 这个数据集的描述是LAION-5B的网页alt-text,长度是**长尾**而不是像003那样按粒度分层,
# 所以阈值只是在砍长尾异常值、不会砍掉某个正常类别:
# 超过200的占8.29%(51.7万条,里面大量是完全正常的长alt-text)、
# 超过512的占0.70%(4.4万条)、超过1024的占0.064%、超过1536的占0.027%。
# 取200会砍掉8%的正常样本,取1024/1536留下的又都是几百字的网页正文段落,
# 按方案确认取512
MAX_CAPTION_LENGTH = 512

# 判定"描述里有没有任何一个实际文字"用的字符集(数字/英文字母/CJK)。
# 实测全量只有2条strip后只剩标点,这类描述没有任何可训练的语义,整条丢弃
CAPTION_WORD_CHAR_PATTERN = re.compile(r'[0-9A-Za-z\u4e00-\u9fff]')

# ------------------------------------------------------------------------------
# 【D2】网页噪声硬过滤: 含URL / 域名 / 图片文件后缀的描述整条丢弃。
# LAION的alt-text直接抓自网页,这三类是最典型的"整条都不是图像描述"的样本,
# 按本脚本的短路顺序实测分别有47659 / 61444 / 68428条(合计约2.85%)。
# 这三条放在Q4改写之前: 它们命中时整条描述基本都是链接或文件名,改写救不回来
# ------------------------------------------------------------------------------
CAPTION_URL_PATTERN = re.compile(r'https?://|www\.', re.I)

CAPTION_DOMAIN_PATTERN = re.compile(
    r'\.(?:com|net|org|edu|gov|info|biz|io|co|ru|de|jp|cn|uk|fr|it|es|nl|pl|br|in|au|ca)(?:/|\b)',
    re.I)

CAPTION_IMAGE_SUFFIX_PATTERN = re.compile(
    r'\.(?:jpg|jpeg|png|gif|bmp|webp|tif|tiff|svg)\b', re.I)

# 【D3】单词数<=1的描述整条丢弃(实测120841条、1.9391%)。
# 这类描述形如"Megan Fox"/"arwen"/"Maryse",只是一个名字或一个词,
# 既没有画面信息也没有句子结构,当文生图文本指令用是纯噪声
MIN_CAPTION_WORD_NUM = 2

# ------------------------------------------------------------------------------
# 【K3】NSFW关键词黑名单: 命中的描述整条丢弃(改写前实测35719条、0.573%,
# 改写后再判一次兜底、实测0条)。
# LAION-5B本身没做成人内容过滤,实测抽样能直接看到
# "The perfect mature wife for a cuck 8"这种描述。
# 只做词边界精确匹配、不做子串匹配: 子串匹配会把"Essex"/"Middlesex"/
# "cocktail"/"Dickens"这类正常词误伤
# ------------------------------------------------------------------------------
CAPTION_NSFW_PATTERN = re.compile(
    r'\b(?:porn|porno|pornstar|xxx|nude|nudes|naked|nsfw|sex|sexo|blowjob|'
    r'handjob|anal|cumshot|creampie|milf|cuck|cuckold|hentai|boobs|tits|'
    r'titties|pussy|vagina|orgasm|masturbat\w*|fetish|bdsm|escort|hooker|'
    r'slut|whore|erotic|erotica|camgirl|onlyfans|deepthroat|threesome|'
    r'gangbang|bukkake|shemale|tranny|incest|upskirt|voyeur|stripper|'
    r'topless|bottomless)\b', re.I)

# ------------------------------------------------------------------------------
# 【Q4】描述改写规则: 把机器产物的网页噪声片段从描述里**剥掉**而不是整条丢弃。
# 与001/003/009"caption原样保存"的口径不同,这是本脚本唯一一处会修改上游文本的地方,
# 按方案确认这么做,理由是这批alt-text里10.57%带结构性垃圾,
# 但垃圾往往只占一小段、整条丢弃会白扔掉62万条其余部分正常的描述。
# 实测全量1987581条(31.89%)被改写过,改写后结构性噪声从10.57%降到0.88%(降低92%)。
# 改写只删噪声片段、不替换任何语义词,所以剩下的文字仍然是上游原文的子串组合
# ------------------------------------------------------------------------------
# HTML标签与实体: 实测有"on guitar.<br /> <br /> Attn ..."这种整段带<br />的描述
CAPTION_REWRITE_HTML_TAG_PATTERN = re.compile(r'<[^>]{0,80}>')

CAPTION_REWRITE_HTML_ENTITY_PATTERN = re.compile(
    r'&(?:amp|nbsp|quot|apos|lt|gt|#\d{1,6}|#x[0-9a-f]{1,5});', re.I)

# 版权声明: "Copyright 2006 John M. Cerra"
CAPTION_REWRITE_COPYRIGHT_PATTERN = re.compile(
    r'(?:\u00a9|\(c\)|\bcopyright\b)\s*\d{0,4}\s*[^\.\|]{0,50}', re.I)

# 摄影师/图库署名: "Photo by Earl J. Ebensteiner, SportsEngine" / "Photo of model xxx"
CAPTION_REWRITE_CREDIT_PATTERN = re.compile(
    r'(?:^|[\.\|\-\u2013\u2014,;:]\s*)'
    r'(?:photo|photos|picture|image|img|credit|photograph|photography|pic)'
    r'\s*(?:by|:|courtesy of|of)\s*[^\.\|]{0,60}', re.I)

# 图库水印文案【K2,含在K3里一起做】: "stock photo" / "— Стоковое фото" / "royalty free"
CAPTION_REWRITE_STOCK_PATTERN = re.compile(
    r'[\-\u2013\u2014,;:\(]?\s*'
    r'(?:stock\s*(?:photo|photos|image|images|picture|footage|illustration)|'
    r'royalty[\s\-]?free|\u0421\u0442\u043e\u043a\u043e\u0432\u043e\u0435\s*\u0444\u043e\u0442\u043e|'
    r'\u0441\u0442\u043e\u043a\u043e\u0432\u043e\u0435\s*\u0444\u043e\u0442\u043e|'
    r'shutterstock|istock|getty images|alamy|depositphotos|dreamstime|123rf|'
    r'adobe stock)\s*\)?', re.I)

# 网页UI文案: "- gallery image 4" / "Load image into Gallery viewer" / "view larger"
CAPTION_REWRITE_UI_PATTERN = re.compile(
    r'\b(?:click here|read more|view larger(?:\s*image)?|see more|view gallery|'
    r'gallery image(?:\s*\d+)?|thumbnail|slide\s*\d+|page\s*\d+|zoom(?:\s*in)?|'
    r'preview|download|share this|next photo|prev(?:ious)?\s*photo|full size|'
    r'enlarge|load image into gallery viewer|tap to expand|hover to zoom|'
    r'roll over image to zoom in|view details|details)\b', re.I)

# 电商促销词: "Free Shipping" / "Cheap Bridesmaid Dresses"里的促销段
CAPTION_REWRITE_ECOMMERCE_PATTERN = re.compile(
    r'\b(?:free shipping|add to cart|order now|buy now|buy online|shop now|'
    r'on sale|for sale|wholesale|in stock|out of stock|coupon code|best price|'
    r'low price|clearance|hot sale|drop\s?shipping|new arrival|limited time|'
    r'shop the look)\b', re.I)

CAPTION_REWRITE_PRICE_PATTERN = re.compile(
    r'[$\u20ac\u00a3\u00a5]\s?\d[\d,\.]*(?:\s?(?:million|billion|k|m|bn))?|'
    r'\b\d[\d,\.]*\s?(?:USD|EUR|GBP|RMB|CNY)\b', re.I)

# 像素尺寸串"1024x768"与文件大小"108kB": 实测22612条带像素串,是网页图片属性而非描述
CAPTION_REWRITE_IMAGE_SIZE_PATTERN = re.compile(
    r'\b\d{2,5}\s?[xX\u00d7]\s?\d{2,5}\b')

CAPTION_REWRITE_FILE_SIZE_PATTERN = re.compile(
    r'\b\d+(?:[\.,]\d+)?\s?(?:kb|mb|gb|kib|mib)\b', re.I)

# 商品编号串: "ID 395402" / "SKU: 1234"
CAPTION_REWRITE_ID_NUM_PATTERN = re.compile(
    r'\b(?:id|no|sku|item|ref|art|model|code)\s*[:#\-]?\s*\d{3,}\b', re.I)

CAPTION_REWRITE_HASHTAG_PATTERN = re.compile(r'(?:^|\s)#\w+')

# SKU编码: "WG370" / "a92c508dae" / "1bT613"。
# 前面的负向断言放掉"4位数字结尾"的形态,避免把"Nikon D3200"这种正常型号
# 以及年份类词误删太多
CAPTION_REWRITE_SKU_PATTERN = re.compile(
    r'\b(?![A-Za-z]{1,4}\d{4}\b)'
    r'(?:[A-Z]{1,5}\d{3,}[A-Z0-9\-]*|[a-f0-9]{8,}|[A-Z]{2,}\d+[A-Z]\d+)\b')

# 竖线面包屑: "Striped linen dress | MANGO"。改写口径是**只保留最长的那一段**,
# 因为面包屑里最长的那段几乎总是真正的图像描述,其余是站点名/栏目名
CAPTION_REWRITE_PIPE_PATTERN = re.compile(r'\s*[\|\u2502\u00a6]\s*')

# 剥离之后残留的空括号"( )"、重复标点".."、多个连续空格
CAPTION_REWRITE_EMPTY_BRACKET_PATTERN = re.compile(
    r'\(\s*[\-,;:\.\s]*\)|\[\s*[\-,;:\.\s]*\]')

CAPTION_REWRITE_DUPLICATE_PUNCT_PATTERN = re.compile(r'([\.,;:!\?\-])\1{1,}')

CAPTION_REWRITE_MULTI_SPACE_PATTERN = re.compile(r'\s{2,}')

# ------------------------------------------------------------------------------
# 【J2】改写截断残留清理: 剥掉首尾的孤立标点和悬空虚词。
# 不加这一步的话,改写会留下"... at Imtech Arena on."(尾部on悬空,
# 因为原文的"on <日期>"里日期段被剥掉了)、"- , 108kB"这种半截句,
# 实测占改写后描述的5.53%。加上这一步后降到约1%
# ------------------------------------------------------------------------------
CAPTION_REWRITE_EDGE_CHAR_PATTERN = re.compile(
    r'^[\s\-\u2013\u2014_\.,;:\|/\\\*\+~\u2018\u2019\u201c\u201d"\'\(\)\[\]]+|'
    r'[\s\-\u2013\u2014_,;:\|/\\\*\+~\u2018\u2019\u201c\u201d"\'\(\[]+$')

CAPTION_REWRITE_TAIL_PUNCT_PATTERN = re.compile(r'[\s\.,;:\-\u2013\u2014]+$')

# 悬空虚词表: 一句话不可能以这些词结尾(结尾一定是被截断了)。
# 句首只剥掉非冠词的那些: "A woman in a red dress"这种以冠词开头是完全正常的
CAPTION_DANGLING_WORD_LIST = [
    'and',
    'or',
    'the',
    'a',
    'an',
    'of',
    'in',
    'on',
    'at',
    'to',
    'with',
    'by',
    'for',
    'from',
    'as',
    'is',
    'are',
    'was',
    'were',
    'be',
    'that',
    'this',
    'his',
    'her',
    'its',
    'their',
    'but',
    'if',
    'than',
    'then',
    'into',
    'over',
    'under',
    'about',
    'after',
    'before',
    'during',
    'while',
    'who',
    'whom',
    'which',
]

CAPTION_DANGLING_KEEP_HEAD_WORD_LIST = [
    'a',
    'an',
    'the',
]

# 悬空清理是"剥一个词之后可能又露出新的悬空词",所以要迭代;
# 实测最多3轮就收敛,这里给6轮上限防止极端情况死循环
MAX_CAPTION_DANGLING_CLEAN_LOOP_NUM = 6

# 一句话至少要留下这么多个词才允许继续剥悬空虚词,
# 否则"Photo of a man"会被一路剥成空串
MIN_CAPTION_DANGLING_KEEP_WORD_NUM = 3

# ------------------------------------------------------------------------------
# 【Q2 + Q3】改写之后仍然残留噪声的描述整条丢弃。
# 这些是"改写救不回来"的情况: 比如整条就是一串逗号关键词
# (dev, dev film, dev trailer, karthi, ...),剥掉哪一段都还是关键词串
# ------------------------------------------------------------------------------
# Q2: 机器产物类残留
CAPTION_INVALID_HTML_TAG_PATTERN = re.compile(r'<[a-z/][^>]{0,60}>', re.I)

CAPTION_INVALID_HTML_ENTITY_PATTERN = re.compile(
    r'&(?:amp|nbsp|quot|lt|gt|#\d+);', re.I)

# 开头就是5段以上逗号短语的关键词串(实测85397条、1.3704%,是Q2里最大的一类)
CAPTION_INVALID_TAG_LIST_PATTERN = re.compile(r'^([^,]{1,40},){4,}')

CAPTION_INVALID_SKU_PATTERN = re.compile(
    r'\b[A-Z]{1,5}\d{3,}[A-Z0-9\-]*\b|\b[a-f0-9]{8,}\b|\b[A-Z]{2,}\d+[A-Z]\d+\b'
)

CAPTION_INVALID_NUM_START_PATTERN = re.compile(r'^\s*\d{6,}')

# Q3: 署名/面包屑/UI/电商类残留
CAPTION_INVALID_CREDIT_PATTERN = re.compile(
    r'\b(?:photo|picture|image|credit|photography)\s*(?:by|:)|'
    r'\u00a9|\bcopyright\b', re.I)

CAPTION_INVALID_PIPE_PATTERN = re.compile(r'[\|\u2502\u00a6]')

CAPTION_INVALID_UI_PATTERN = re.compile(
    r'\b(?:click here|read more|view larger|thumbnail|gallery image|slide \d+|'
    r'page \d+|zoom|preview|download)\b', re.I)

CAPTION_INVALID_ECOMMERCE_PATTERN = re.compile(
    r'\b(?:free shipping|add to cart|order now|buy now|wholesale|in stock|'
    r'coupon|clearance|hot sale)\b', re.I)

CAPTION_INVALID_STOCK_PATTERN = re.compile(
    r'\bstock\s*(?:photo|image|picture|footage)\b|\broyalty[\s\-]?free\b|'
    r'\u0441\u0442\u043e\u043a\u043e\u0432\u043e\u0435', re.I)

# ------------------------------------------------------------------------------
# 【纯文本层过滤的硬对账值】下面每一项都是**全量6231808行实测**(非抽样)的精确条数。
# 它们之所以能硬编码,是因为这些判定只依赖上游jsonl里的caption字符串与
# width/height字段,不需要解码任何图像,所以能在几十秒内跑完全量、拿到确定值。
# 判定按上面的过滤顺序短路执行,所以彼此互斥。
# 条数对不上说明上游010没跑完、产物被改动过、或者本脚本的文本规则被改了,
# 这时候继续往下跑只会得到一个悄悄少样本(或悄悄多留脏样本)的新数据集,必须直接报错
# ------------------------------------------------------------------------------
EXPECTED_TEXT_FILTER_COUNT_DICT = {
    # 文本层(改写之前)
    'too_short_caption_count': 1238,
    'too_long_caption_count': 43926,
    'no_word_char_caption_count': 2,
    'url_caption_count': 47659,
    'domain_caption_count': 61444,
    'image_suffix_caption_count': 68428,
    'one_word_caption_count': 120841,
    'nsfw_caption_count': 35719,
    # 文本层(改写之后重新判定)。
    # rewrite_too_long恒为0是因为改写只删不加、描述不可能变长;
    # rewrite_nsfw恒为0说明改写没有把两个词拼成黑名单词。这两条纯防御
    'rewrite_no_word_char_caption_count': 21649,
    'rewrite_too_short_caption_count': 4393,
    'rewrite_too_long_caption_count': 0,
    'rewrite_one_word_caption_count': 1245,
    'rewrite_nsfw_caption_count': 0,
    # Q2残留。html_tag与html_entity恒为0是因为Q4改写已经把它们全剥干净了
    # (改写前实测有1.77%的描述带<br />这类标签),这两条留着当"改写失效"的哨兵
    'invalid_html_tag_caption_count': 0,
    'invalid_html_entity_caption_count': 0,
    'invalid_tag_list_caption_count': 85397,
    'invalid_sku_caption_count': 17421,
    'invalid_num_start_caption_count': 1126,
    # Q3残留。pipe恒为0同理: 面包屑在改写时已经被"只保留最长一段"处理掉了
    'invalid_credit_caption_count': 84665,
    'invalid_pipe_caption_count': 0,
    'invalid_ui_caption_count': 20,
    'invalid_ecommerce_caption_count': 505,
    'invalid_stock_caption_count': 3,
}

# ------------------------------------------------------------------------------
# 【图像层过滤的期望值(只做软校验、只打印告警,不硬失败)】
# 这一项与上面那些文本层的值有本质区别: 它统计的是"真解码之后"被判掉的图像
# (解码失败 / 非RGB与非L / 短边 < 64 / 宽高比 > 8 四类合一),
# 而本脚本编写时**没有**把623万张图全解码一遍(那要跑好几小时),
# 所以这个期望值是按"上游标注里记录的width/height"跑完整级联推算出来的下界:
#   全量6231808行按标注宽高算,短边 < 64的有0条、宽高比 > 8的有205条,
#   但这205条里有28条在轮到图像层判定之前就已经被前面的文本层规则丢掉了
#   (图像层排在整个级联的最后),所以真正会记到这个key上的是177条;
#   实测抽样320张标注宽高与真实解码结果0例不符,所以这177条是可信的;
#   但真解码还会额外判掉"文件损坏解不开"和"mode不是RGB/L"这两类,
#   抽样3000张目标图里这两类是0张,可全量到底有多少**未实测**。
# 所以这里只做软校验: 实际值大于177时只打印告警(说明有一批图确实坏了或不是RGB),
# 具体条数会写进resave_check_result.json。第一次跑完之后可以把真实数字回填到这里,
# 再把下面的CHECK_EXPECTED_IMAGE_FILTER_COUNT_FLAG改成True变成硬对账
EXPECTED_INVALID_IMAGE_COUNT = 177

# 图像层期望值默认只告警不硬失败,理由见上。第一次跑完拿到真实数字回填之后再改成True
CHECK_EXPECTED_IMAGE_FILTER_COUNT_FLAG = False

# 图像层过滤在计数字典里用的key名。它和上面那些文本层的key放在同一个字典里累加,
# 这样"各项之和 + 存活数 == 总行数"这条闭合校验才能覆盖全部过滤路径。
# 之所以只用一个key而不像文本层那样细分成"解码失败/非RGB/短边/宽高比"四个:
# process_single_image_check只返回None表示不合格(与001/003/007/009完全一致的写法),
# 主调方拿不到具体原因; 四类的区分体现在该函数里的4444/5555/6666/7777四种日志编号上
INVALID_IMAGE_FILTER_COUNT_KEY_NAME = 'invalid_image_count'

# 计数字典的全部key名 = 文本层各项 + 图像层那一项。
# worker和主进程都按这个列表初始化计数字典,保证两边的key集合永远一致
FILTER_COUNT_KEY_NAME_LIST = list(EXPECTED_TEXT_FILTER_COUNT_DICT.keys()) + [
    INVALID_IMAGE_FILTER_COUNT_KEY_NAME,
]

# ------------------------------------------------------------------------------
# 【依赖图像层结果的期望值(同样只做软校验)】
# 下面这几项都建立在"图像层一张都不会被判掉(除了那205条极端宽高比)"这个假设上,
# 所以它们和EXPECTED_INVALID_IMAGE_COUNT一样只打印告警、不硬失败。
# 硬失败的只有"闭合校验"(见check_load_annotation_count): 各类过滤条数之和加上存活数
# 必须正好等于总行数——这条是结构性的、不依赖任何实测值,任何漏处理/重复计数都能兜住
# ------------------------------------------------------------------------------
# 被Q4改写过的条数(实测1987581条、31.89%)。
# 它统计的是"进到改写这一步的样本里有多少条文本被动过",不参与闭合对账
EXPECTED_REWRITE_CAPTION_COUNT = 1987581

# 文本+图像过滤全部通过、进入去重阶段的条数(实测5635950条、90.44%)
EXPECTED_FILTER_PASS_COUNT = 5635950

# 【E-c去重】先按图像内容md5去重(E2),再按描述去重(E3),保证落盘后
# "每张图唯一、每条描述也唯一"。
# 实测E2去重掉的图像其实是E3的子集(同一条描述对应的4874张图字节完全相同,
# 抽查400张md5全一致),所以两步做完的结果 == 只做E3的结果 == 4457233对。
# 注意E2这一项的期望值给0是因为**它没能全量实测**: 算全量md5必须先把623万张图
# (1.4TB)全读一遍,而本脚本是把md5合并进图像校验worker顺手算的,
# 编写阶段只抽样量到"随机20万张里1.26%字节级重复"。
# 所以真实值几乎肯定是几万这个量级、不会是0,这一项必然只能当告警看,
# 第一次跑完从resave_check_result.json里拿真值回填即可。
# E2仍然照做的理由: 它是"落盘后每张图唯一"这个性质的直接证据,且零额外IO
EXPECTED_DUPLICATE_IMAGE_MD5_COUNT = 0

EXPECTED_DUPLICATE_CAPTION_COUNT = 1178717

EXPECTED_VALID_IMAGE_ANNOTATION_COUNT = 4457233

# 最终产出的文件夹数与子集目录数(4457233 / 10000 -> 446个文件夹; 446 / 100 -> 5个子集)
EXPECTED_SAVE_FOLDER_COUNT = 446

EXPECTED_SAVE_SET_COUNT = 5

# 纯文本层过滤的硬对账开关。
# 上面EXPECTED_TEXT_FILTER_COUNT_DICT里的值就是全量实测出来的,所以默认打开;
# 之后如果**故意**调了任何一条文本过滤规则或阈值,先把这个开关改成False跑一遍,
# 拿resave_check_result.json里的真实数字回填那个dict,再改回True
CHECK_EXPECTED_FILTER_COUNT_FLAG = True


def get_set_name(per_series_name, per_set_index):
    """子集目录名: 每个系列按每100个文件夹切成<系列名>_000/_001/...

    子集目录名同时是保存图像名的中间段,所以它必须在切分完成后才能确定,
    这也是后面排序键只能用原图名前缀而不能用保存图像名的原因。
    """
    per_set_name = per_series_name.lower()

    if per_series_name not in SPLIT_SET_DIR_SERIES_NAME_LIST:
        return per_set_name

    return f'{per_set_name}_{per_set_index:03d}'


def check_image_file_exists(per_image_path, dir_file_name_cache_dict):
    """用每个目录只列一次的文件名集合替代逐样本os.path.exists

    上游图像都放在NAS上,逐样本打一次os.path.exists就是一次网络往返,
    623万条标注就要打623万次,这一步本身就能占掉整个扫描阶段的大头。
    上游按 id // 10000 分桶落盘,而汇总标注也是按id顺序写的,所以同一个jsonl分片
    (10万行)里的图像正好落在连续的10个分桶目录下,这里按目录缓存一次os.listdir
    的结果,之后只做集合查表,网络往返次数从"标注条数"降到"分桶目录数"(625次)。
    listdir失败(目录不存在/无权限)时回退到os.path.exists逐个判,
    保证判定结果和改造前完全一致。
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


def get_caption_word_num(per_t2i_caption):
    """按空白切分数词数,用于单词数<=1的判定"""
    return len(str(per_t2i_caption).split())


def check_url_caption(per_t2i_caption):
    """判定描述里有没有URL(实测47659条)"""
    return CAPTION_URL_PATTERN.search(per_t2i_caption) is not None


def check_domain_caption(per_t2i_caption):
    """判定描述里有没有域名(实测61444条)"""
    return CAPTION_DOMAIN_PATTERN.search(per_t2i_caption) is not None


def check_image_suffix_caption(per_t2i_caption):
    """判定描述里有没有图片文件后缀(实测68428条)"""
    return CAPTION_IMAGE_SUFFIX_PATTERN.search(per_t2i_caption) is not None


def check_nsfw_caption(per_t2i_caption):
    """判定描述是不是成人内容(改写前实测35719条)

    只做词边界精确匹配、不做子串匹配: 子串匹配会把Essex/Middlesex/cocktail/
    Dickens这类正常词误伤。
    """
    return CAPTION_NSFW_PATTERN.search(per_t2i_caption) is not None


def clean_dangling_caption(per_t2i_caption):
    """剥掉描述首尾的孤立标点与悬空虚词

    改写剥掉噪声片段之后经常留下半截句(如"... at Imtech Arena on."的尾部on悬空,
    因为原文"on <日期>"里的日期段被剥掉了),这一步把它们清干净。
    剥一个词之后可能又露出新的悬空词,所以要迭代到不再变化为止;
    句首只剥非冠词的虚词("A woman in a red dress"以冠词开头是完全正常的);
    剩下的词数少于MIN_CAPTION_DANGLING_KEEP_WORD_NUM时停手,避免一路剥成空串。
    """
    per_t2i_caption = str(per_t2i_caption)

    for _ in range(MAX_CAPTION_DANGLING_CLEAN_LOOP_NUM):
        per_last_t2i_caption = per_t2i_caption

        per_t2i_caption = CAPTION_REWRITE_TAIL_PUNCT_PATTERN.sub(
            '', per_t2i_caption)
        per_t2i_caption = CAPTION_REWRITE_EDGE_CHAR_PATTERN.sub(
            '', per_t2i_caption).strip()

        per_word_list = per_t2i_caption.split()
        if len(per_word_list
               ) >= MIN_CAPTION_DANGLING_KEEP_WORD_NUM and per_word_list[
                   -1].lower().strip('.,;:') in CAPTION_DANGLING_WORD_LIST:
            per_t2i_caption = ' '.join(per_word_list[:-1])

        per_word_list = per_t2i_caption.split()
        if len(per_word_list) >= MIN_CAPTION_DANGLING_KEEP_WORD_NUM:
            per_head_word = per_word_list[0].lower().strip('.,;:')
            if per_head_word in CAPTION_DANGLING_WORD_LIST and per_head_word not in CAPTION_DANGLING_KEEP_HEAD_WORD_LIST:
                per_t2i_caption = ' '.join(per_word_list[1:])

        if per_t2i_caption == per_last_t2i_caption:
            break

    return per_t2i_caption.strip()


def get_normalized_t2i_caption(per_t2i_caption):
    """把网页噪声片段从描述里剥掉,返回归一化后的描述

    这是本脚本唯一一处会修改上游文本的地方(001/003/009都是原样保存),
    按方案确认这么做: 这批alt-text里10.57%带结构性垃圾,但垃圾往往只占一小段,
    整条丢弃会白扔掉62万条其余部分正常的描述。
    改写只**删**噪声片段、不替换任何语义词,所以剩下的文字仍然是上游原文的子串组合。
    竖线面包屑的口径是只保留最长的那一段(最长那段几乎总是真正的图像描述,
    其余是站点名/栏目名)。
    最后统一做一次悬空虚词清理(J2),把改写留下的半截句收干净。
    """
    per_t2i_caption = str(per_t2i_caption)

    per_t2i_caption = CAPTION_REWRITE_HTML_TAG_PATTERN.sub(
        ' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_HTML_ENTITY_PATTERN.sub(
        ' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_COPYRIGHT_PATTERN.sub(
        ' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_CREDIT_PATTERN.sub(' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_STOCK_PATTERN.sub(' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_UI_PATTERN.sub(' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_ECOMMERCE_PATTERN.sub(
        ' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_PRICE_PATTERN.sub(' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_IMAGE_SIZE_PATTERN.sub(
        ' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_FILE_SIZE_PATTERN.sub(
        ' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_ID_NUM_PATTERN.sub(' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_HASHTAG_PATTERN.sub(' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_SKU_PATTERN.sub(' ', per_t2i_caption)

    # 竖线面包屑只保留最长的那一段
    if CAPTION_REWRITE_PIPE_PATTERN.search(per_t2i_caption):
        per_pipe_part_list = [
            per_pipe_part.strip() for per_pipe_part in
            CAPTION_REWRITE_PIPE_PATTERN.split(per_t2i_caption)
            if per_pipe_part.strip()
        ]
        if len(per_pipe_part_list) > 0:
            per_t2i_caption = max(per_pipe_part_list, key=len)

    per_t2i_caption = CAPTION_REWRITE_EMPTY_BRACKET_PATTERN.sub(
        ' ', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_DUPLICATE_PUNCT_PATTERN.sub(
        r'\1', per_t2i_caption)
    per_t2i_caption = CAPTION_REWRITE_MULTI_SPACE_PATTERN.sub(
        ' ', per_t2i_caption).strip()

    per_t2i_caption = clean_dangling_caption(per_t2i_caption)

    per_t2i_caption = CAPTION_REWRITE_MULTI_SPACE_PATTERN.sub(
        ' ', per_t2i_caption).strip()

    return per_t2i_caption


def get_invalid_rewrite_caption_count_key_name(per_t2i_caption):
    """判定改写之后仍然残留噪声的描述,返回命中的计数key名(空串表示合格)

    这些是"改写救不回来"的情况: 比如整条就是一串逗号关键词
    (dev, dev film, dev trailer, karthi, ...),剥掉哪一段都还是关键词串,
    只能整条丢弃。判定顺序与EXPECTED_TEXT_FILTER_COUNT_DICT里的顺序一致,
    所以每条描述只会被记到第一个命中的类别里、各类别条数互斥。
    """
    # Q2: 机器产物类残留
    if CAPTION_INVALID_HTML_TAG_PATTERN.search(per_t2i_caption):
        return 'invalid_html_tag_caption_count'
    if CAPTION_INVALID_HTML_ENTITY_PATTERN.search(per_t2i_caption):
        return 'invalid_html_entity_caption_count'
    if CAPTION_INVALID_TAG_LIST_PATTERN.search(per_t2i_caption):
        return 'invalid_tag_list_caption_count'
    if CAPTION_INVALID_SKU_PATTERN.search(per_t2i_caption):
        return 'invalid_sku_caption_count'
    if CAPTION_INVALID_NUM_START_PATTERN.search(per_t2i_caption):
        return 'invalid_num_start_caption_count'

    # Q3: 署名/面包屑/UI/电商类残留
    if CAPTION_INVALID_CREDIT_PATTERN.search(per_t2i_caption):
        return 'invalid_credit_caption_count'
    if CAPTION_INVALID_PIPE_PATTERN.search(per_t2i_caption):
        return 'invalid_pipe_caption_count'
    if CAPTION_INVALID_UI_PATTERN.search(per_t2i_caption):
        return 'invalid_ui_caption_count'
    if CAPTION_INVALID_ECOMMERCE_PATTERN.search(per_t2i_caption):
        return 'invalid_ecommerce_caption_count'
    if CAPTION_INVALID_STOCK_PATTERN.search(per_t2i_caption):
        return 'invalid_stock_caption_count'

    return ''


def process_single_image_check(per_image_path):
    """校验单张图像能否正常解码,过滤非RGB图和极端分辨率图,并顺手算内容md5

    返回[宽, 高, 内容md5]。宽高只用于和上游标注比对,最终写进json的宽高一定取自
    实际写盘图像的shape。
    md5是在这里"顺手"算的: 这个函数本来就要把整个文件字节读进内存做解码
    (np.fromfile),所以算md5不产生任何额外的NAS往返。这是相对003唯一的结构性改动,
    为的是支持按图像内容全局去重(E2)——如果单独再开一遍去算623万张图的哈希,
    在NAS上要多花几小时。
    """
    # cv2.IMREAD_COLOR会把灰度图静默复制成3通道、把RGBA图静默丢掉alpha通道、
    # 把CMYK图静默转成3通道,所以必须先用PIL读原始mode才能把这些图判出来
    try:
        per_image_mode = Image.open(per_image_path).mode
    except Exception as e:
        print('4444', per_image_path, e)
        return None

    if per_image_mode not in VALID_IMAGE_MODE_LIST:
        print('5555', per_image_path, per_image_mode)
        return None

    try:
        per_image_bytes = np.fromfile(per_image_path, dtype=np.uint8)
        per_image = cv2.imdecode(per_image_bytes, cv2.IMREAD_COLOR)
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

    # 内容md5取自**原始文件字节**而不是解码后的数组:
    # 上游图像都是同一批LAION原图,字节相同就一定是同一张图,
    # 而按解码后数组算哈希还要多一次几百MB/s的内存序列化
    per_image_md5 = hashlib.md5(per_image_bytes.tobytes()).hexdigest()

    return [
        per_image_w,
        per_image_h,
        per_image_md5,
    ]


def process_single_annotation_file(annotation_file_pair):
    """解析单个上游jsonl标注,组装图像路径和t2i描述的样本对列表

    这个worker把文本层过滤(缺图、图像名与sample_key不一致、图像名非法、
    任务类型不对、描述为空或过短、描述过长、无文字字符、含URL/域名/图片后缀、
    单词数<=1、成人内容、改写后重新判定、改写后仍残留噪声)和图像层过滤
    (能否解码、是否RGB、短边、宽高比)一次做完,并把图像内容md5一起带回来供去重用。
    图像校验没有像001那样单独再开一个Pool,是因为本数据集有623万条标注:
    分两个Pool的话主进程要先攒623万条记录、再逐条发给check worker、再收回623万条,
    光进程间序列化就要来回搬几十GB,而合并进来之后IPC只传存活样本。
    判定逻辑、过滤口径、日志编号和001/003/009完全一致,图像也一样是解码两遍
    (这里校验一遍、写盘时重编码再解一遍),没有为了省时间跳过任何一道校验。
    """
    per_jsonl_path, root_dataset_path, per_series_name = annotation_file_pair

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    annotation_image_size_not_match_count = 0
    rewrite_caption_count = 0
    filter_count_dict = {
        per_filter_count_key_name: 0
        for per_filter_count_key_name in FILTER_COUNT_KEY_NAME_LIST
    }
    image_annotation_pair_list = []

    # 每个worker只处理一个jsonl(10万行),缓存里通常只有10个分桶目录,内存开销可忽略
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
            missing_image_count,
            invalid_image_name_count,
            invalid_task_type_count,
            annotation_image_size_not_match_count,
            rewrite_caption_count,
            filter_count_dict,
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

            per_image_relative_path = per_annotation.get(
                ANNOTATION_IMAGE_KEY_NAME, '')
            if not isinstance(per_image_relative_path, str):
                per_image_relative_path = ''
            if not per_image_relative_path:
                missing_image_count += 1
                continue

            per_image_relative_path = per_image_relative_path.replace(
                '\\', '/').lstrip('/')
            per_image_path = os.path.join(root_dataset_path,
                                          per_image_relative_path)
            if not check_image_file_exists(per_image_path,
                                           dir_file_name_cache_dict):
                missing_image_count += 1
                continue

            # 落盘图像的文件名前缀就是样本id(也就是jsonl里的行号),和标注里的
            # sample_key必须完全一致,不一致说明上游行号映射错位,
            # 这种样本没法保证保存图像名唯一,直接丢弃
            per_image_name_prefix = os.path.splitext(
                os.path.basename(per_image_relative_path))[0].lower()
            per_sample_key = per_annotation.get(ANNOTATION_IMAGE_NAME_KEY_NAME,
                                                '')
            per_sample_key = str(per_sample_key).strip().lower()

            if not per_sample_key or per_sample_key != per_image_name_prefix:
                invalid_image_name_count += 1
                print('2222', per_image_path, per_sample_key)
                continue

            if not VALID_IMAGE_NAME_PREFIX_PATTERN.match(
                    per_image_name_prefix):
                invalid_image_name_count += 1
                print('2222', per_image_path, per_image_name_prefix)
                continue

            # 上游给每行都标了任务类型,实测恒为text_to_image_faceid_customization。
            # 出现别的取值说明上游规格变了(比如混进了编辑样本),不能进t2i数据集
            per_task_type = per_annotation.get(ANNOTATION_TASK_TYPE_KEY_NAME,
                                               '')
            if not isinstance(per_task_type, str):
                per_task_type = ''
            if per_task_type.strip() != EXPECTED_ANNOTATION_TASK_TYPE:
                invalid_task_type_count += 1
                print('2222', per_image_path, per_task_type)
                continue

            per_t2i_caption = per_annotation.get(ANNOTATION_CAPTION_KEY_NAME,
                                                 '')
            # 上游描述固定是str,这里兼容list和str两种形式
            if isinstance(per_t2i_caption, (list, tuple)):
                per_t2i_caption = per_t2i_caption[0] if len(
                    per_t2i_caption) > 0 else ''
            if not isinstance(per_t2i_caption, str):
                per_t2i_caption = ''
            per_t2i_caption = per_t2i_caption.strip()

            # 空描述、全空格描述、过短描述都视为不合格样本对
            # (上游已校验caption全部非空,实测全量有1238条strip后长度小于10)
            if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                filter_count_dict['too_short_caption_count'] += 1
                print('3333', per_image_path, len(per_t2i_caption))
                continue

            # 过长描述同样视为不合格样本对(实测全量43926条超过512)
            if len(per_t2i_caption) > MAX_CAPTION_LENGTH:
                filter_count_dict['too_long_caption_count'] += 1
                print('3333', per_image_path, len(per_t2i_caption))
                continue

            # 只剩标点、没有任何数字/字母/汉字的描述也丢弃(实测全量只有2条)
            if not CAPTION_WORD_CHAR_PATTERN.search(per_t2i_caption):
                filter_count_dict['no_word_char_caption_count'] += 1
                print('3333', per_image_path, per_t2i_caption[:50])
                continue

            # 【D2】含URL/域名/图片后缀的描述整条丢弃: 命中时整条基本都是链接或文件名,
            # 改写救不回来,所以这三条排在Q4改写之前
            if check_url_caption(per_t2i_caption):
                filter_count_dict['url_caption_count'] += 1
                continue
            if check_domain_caption(per_t2i_caption):
                filter_count_dict['domain_caption_count'] += 1
                continue
            if check_image_suffix_caption(per_t2i_caption):
                filter_count_dict['image_suffix_caption_count'] += 1
                continue

            # 【D3】单词数<=1的描述整条丢弃(形如"Megan Fox"/"arwen",实测120841条)
            if get_caption_word_num(per_t2i_caption) < MIN_CAPTION_WORD_NUM:
                filter_count_dict['one_word_caption_count'] += 1
                continue

            # 【K3】成人内容整条丢弃(改写前实测35719条)
            if check_nsfw_caption(per_t2i_caption):
                filter_count_dict['nsfw_caption_count'] += 1
                continue

            # 【Q4 + J2】把网页噪声片段从描述里剥掉,写进json的一定是归一化后的描述
            per_raw_t2i_caption = per_t2i_caption
            per_t2i_caption = get_normalized_t2i_caption(per_t2i_caption)
            if per_t2i_caption != per_raw_t2i_caption:
                rewrite_caption_count += 1

            # 改写会让描述变短甚至变空,所以长度/词数/文字字符/成人内容这几条
            # 必须按**改写后**的文本重新判一遍。这样写进json的描述与判定口径完全一致,
            # 收尾自校验直接量json里的长度就能复检
            if not CAPTION_WORD_CHAR_PATTERN.search(per_t2i_caption):
                filter_count_dict['rewrite_no_word_char_caption_count'] += 1
                continue
            if len(per_t2i_caption) < MIN_CAPTION_LENGTH:
                filter_count_dict['rewrite_too_short_caption_count'] += 1
                continue
            if len(per_t2i_caption) > MAX_CAPTION_LENGTH:
                # 改写只删不加,所以理论上不可能变长,这里只做防御性拦截(实测0条)
                filter_count_dict['rewrite_too_long_caption_count'] += 1
                continue
            if get_caption_word_num(per_t2i_caption) < MIN_CAPTION_WORD_NUM:
                filter_count_dict['rewrite_one_word_caption_count'] += 1
                continue
            if check_nsfw_caption(per_t2i_caption):
                # 改写剥掉中间片段后有可能让两个词贴到一起命中黑名单,
                # 这里再判一次兜底(实测0条)
                filter_count_dict['rewrite_nsfw_caption_count'] += 1
                continue

            # 【Q2 + Q3】改写之后仍然残留噪声的描述整条丢弃
            per_invalid_rewrite_caption_count_key_name = get_invalid_rewrite_caption_count_key_name(
                per_t2i_caption)
            if per_invalid_rewrite_caption_count_key_name:
                filter_count_dict[
                    per_invalid_rewrite_caption_count_key_name] += 1
                continue

            per_check_result = process_single_image_check(per_image_path)
            if per_check_result is None:
                # 图像层的四类不合格(解码失败/非RGB与非L/短边<64/宽高比>8)在
                # process_single_image_check里已经按4444/5555/6666/7777四种日志编号
                # 分别打印过,这里只能合到一个key里计数(该函数只返回None、
                # 拿不到具体原因),这与001/003/007/009的写法完全一致
                filter_count_dict[INVALID_IMAGE_FILTER_COUNT_KEY_NAME] += 1
                continue

            per_image_w, per_image_h, per_image_md5 = per_check_result

            # 分辨率过滤一律用上面真实解码出来的shape,这里只是顺手核对一遍上游标注里
            # 记录的宽高,不一致只计数上报、不丢样本(实测抽样320张0例不符)
            if per_annotation.get(
                    ANNOTATION_IMAGE_WIDTH_KEY_NAME,
                    per_image_w) != per_image_w or per_annotation.get(
                        ANNOTATION_IMAGE_HEIGHT_KEY_NAME,
                        per_image_h) != per_image_h:
                annotation_image_size_not_match_count += 1
                print(
                    '2222', per_image_path, per_image_w, per_image_h,
                    per_annotation.get(ANNOTATION_IMAGE_WIDTH_KEY_NAME, None),
                    per_annotation.get(ANNOTATION_IMAGE_HEIGHT_KEY_NAME, None))

            # 保存图像名要等切完子集目录才能拼出来,这里只带上原图名前缀;
            # 样本id(int)单独带一份,去重时用它当"保留哪一条"的排序键,
            # 保证留下的永远是id最小的那条、结果可复现
            image_annotation_pair_list.append([
                per_series_name,
                per_image_path,
                per_image_name_prefix,
                per_t2i_caption,
                per_image_md5,
                int(per_image_name_prefix),
            ])

    return [
        image_annotation_pair_list,
        per_series_name,
        total_annotation_count,
        load_annotation_failed_count,
        missing_image_count,
        invalid_image_name_count,
        invalid_task_type_count,
        annotation_image_size_not_match_count,
        rewrite_caption_count,
        filter_count_dict,
    ]


def get_all_image_annotation_pair(root_dataset_path):
    """扫描上游解压好的jsonl标注,多进程组装图像路径和t2i描述的样本对列表

    这里只listdir unzip_annotations拿到63个jsonl路径,绝不去os.walk图像目录:
    上游解出1246万个小文件,扫目录树在NAS上不可接受。
    每个jsonl(10万行)里的图像正好落在连续的10个分桶目录下,
    worker只需要对那10个目录各listdir一次。
    最后按[系列名, 原图名前缀]统一排序,保证输出顺序与串行版本完全一致。
    """
    root_annotation_path = os.path.join(root_dataset_path,
                                        LOAD_ANNOTATION_DIR_NAME)

    annotation_file_pair_list = []
    series_annotation_file_count_dict = {}
    for per_series_name in LOAD_SERIES_DIR_NAME_LIST:
        per_series_annotation_file_count = 0
        if not os.path.isdir(root_annotation_path):
            print('2222', root_annotation_path)
            series_annotation_file_count_dict[per_series_name] = 0
            continue

        for per_jsonl_name in sorted(os.listdir(root_annotation_path)):
            if not per_jsonl_name.endswith(LOAD_ANNOTATION_FILE_SUFFIX):
                continue

            # 上游把整个系列的标注切成了 <系列名>_<分片号>.jsonl,
            # 所以这里按文件名前缀归系列(不像003那样每个系列一个子目录)
            if not per_jsonl_name.startswith(f'{per_series_name}_'):
                continue

            # 系列名由主进程按文件名算好后带给worker,
            # worker只认标注文件、数据集根目录、系列名这三个入参
            annotation_file_pair_list.append([
                os.path.join(root_annotation_path, per_jsonl_name),
                root_dataset_path,
                per_series_name,
            ])
            per_series_annotation_file_count += 1

        series_annotation_file_count_dict[
            per_series_name] = per_series_annotation_file_count

    total_annotation_count, load_annotation_failed_count = 0, 0
    missing_image_count, invalid_image_name_count = 0, 0
    invalid_task_type_count = 0
    annotation_image_size_not_match_count = 0
    rewrite_caption_count = 0
    filter_count_dict = {
        per_filter_count_key_name: 0
        for per_filter_count_key_name in FILTER_COUNT_KEY_NAME_LIST
    }
    series_row_count_dict = {}
    image_annotation_pair_list = []
    with Pool(processes=min(PROCESS_NUM, max(len(annotation_file_pair_list),
                                             1))) as pool:
        for per_load_result in tqdm(pool.imap_unordered(
                process_single_annotation_file, annotation_file_pair_list),
                                    total=len(annotation_file_pair_list)):
            image_annotation_pair_list.extend(per_load_result[0])

            per_series_name = per_load_result[1]
            series_row_count_dict[per_series_name] = series_row_count_dict.get(
                per_series_name, 0) + per_load_result[2]

            total_annotation_count += per_load_result[2]
            load_annotation_failed_count += per_load_result[3]
            missing_image_count += per_load_result[4]
            invalid_image_name_count += per_load_result[5]
            invalid_task_type_count += per_load_result[6]
            annotation_image_size_not_match_count += per_load_result[7]
            rewrite_caption_count += per_load_result[8]

            for per_filter_count_key_name, per_filter_count in per_load_result[
                    9].items():
                filter_count_dict[
                    per_filter_count_key_name] = filter_count_dict.get(
                        per_filter_count_key_name, 0) + per_filter_count

    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[0], x[2]])

    return [
        image_annotation_pair_list,
        series_annotation_file_count_dict,
        series_row_count_dict,
        total_annotation_count,
        load_annotation_failed_count,
        missing_image_count,
        invalid_image_name_count,
        invalid_task_type_count,
        annotation_image_size_not_match_count,
        rewrite_caption_count,
        filter_count_dict,
    ]


def check_load_annotation_count(series_annotation_file_count_dict,
                                series_row_count_dict, total_annotation_count,
                                filter_count_dict, rewrite_caption_count,
                                filter_pass_count):
    """解析完标注后按分片数、行数、逐项过滤条数对账

    上游010的解压产物是一次性解出来的确定结果,条数对不上说明上游没跑完、
    产物被改动过、或者本脚本的过滤规则被改了,这时候继续往下跑只会得到一个
    悄悄少样本(或悄悄多留脏样本)的新数据集,必须在跑几十小时的图像重编码之前直接报错。

    分三类校验,严格程度不同:
    1. 硬失败(与任何开关无关): **闭合校验**——各项过滤条数之和加上存活数必须正好
       等于总行数。它是结构性的、不依赖任何实测值: 少了说明有样本既没被判掉也没进
       存活列表(漏处理),多了说明某条样本被重复计数。这是本函数最重要的一条;
    2. 硬失败(受CHECK_EXPECTED_FILTER_COUNT_FLAG控制): 纯文本层各项过滤条数。
       它们只依赖jsonl里的字符串,能全量实测出确定值,所以敢写死;
    3. 只告警: 图像层过滤条数、被改写条数、存活总数。它们依赖真解码结果,
       而本脚本编写时没有把623万张图全解码一遍,期望值是按标注宽高推算的下界,
       详见EXPECTED_INVALID_IMAGE_COUNT的注释。
    """
    check_error_message_list, check_warning_message_list = [], []

    # 上游jsonl分片数量不对说明上游没跑完,继续跑只会静默少样本对
    for per_series_name, per_expected_annotation_file_num in EXPECTED_SERIES_ANNOTATION_FILE_NUM_DICT.items(
    ):
        per_annotation_file_num = series_annotation_file_count_dict.get(
            per_series_name, 0)
        if per_annotation_file_num != per_expected_annotation_file_num:
            check_error_message_list.append(
                f'{per_series_name} annotation file num not match '
                f'{per_annotation_file_num} != {per_expected_annotation_file_num}'
            )

    for per_series_name, per_expected_row_count in EXPECTED_SERIES_ROW_COUNT_DICT.items(
    ):
        per_row_count = series_row_count_dict.get(per_series_name, 0)
        if per_row_count != per_expected_row_count:
            check_error_message_list.append(
                f'{per_series_name} annotation row count not match '
                f'{per_row_count} != {per_expected_row_count}')

    for per_series_name in sorted(series_row_count_dict.keys()):
        if per_series_name not in EXPECTED_SERIES_ROW_COUNT_DICT:
            check_error_message_list.append(
                f'unknown series {per_series_name}')

    if total_annotation_count != EXPECTED_TOTAL_ROW_COUNT:
        check_error_message_list.append(
            f'total annotation count not match '
            f'{total_annotation_count} != {EXPECTED_TOTAL_ROW_COUNT}')

    # 计数字典的key集合必须和约定的一致,多一个少一个都说明代码被改了但没同步
    for per_filter_count_key_name in sorted(filter_count_dict.keys()):
        if per_filter_count_key_name not in FILTER_COUNT_KEY_NAME_LIST:
            check_error_message_list.append(
                f'unknown filter count key {per_filter_count_key_name}')

    for per_filter_count_key_name in sorted(FILTER_COUNT_KEY_NAME_LIST):
        if per_filter_count_key_name not in filter_count_dict:
            check_error_message_list.append(
                f'missing filter count key {per_filter_count_key_name}')

    # 【始终硬失败】闭合校验: 各项过滤条数互斥,所以它们之和加上存活数必须正好等于
    # 总行数。这条不依赖任何实测期望值,纯结构性,所以与任何开关无关。
    # 少了说明有样本既没被判掉也没进存活列表(漏处理),
    # 多了说明某条样本被重复计数
    if sum(filter_count_dict.values()
           ) + filter_pass_count != total_annotation_count:
        check_error_message_list.append(
            f'filter count not self consistent '
            f'{sum(filter_count_dict.values())} + {filter_pass_count} != '
            f'{total_annotation_count}')

    # 【只告警】图像层过滤条数: 期望值是按标注宽高推算的下界,真解码还会额外判掉
    # 损坏图与非RGB/L图,所以实际值 >= 期望值是正常的,只有小于期望值才值得注意
    per_invalid_image_count = filter_count_dict.get(
        INVALID_IMAGE_FILTER_COUNT_KEY_NAME, 0)
    if per_invalid_image_count != EXPECTED_INVALID_IMAGE_COUNT:
        per_invalid_image_message = (
            f'{INVALID_IMAGE_FILTER_COUNT_KEY_NAME} not match '
            f'{per_invalid_image_count} != {EXPECTED_INVALID_IMAGE_COUNT}')
        if CHECK_EXPECTED_IMAGE_FILTER_COUNT_FLAG:
            check_error_message_list.append(per_invalid_image_message)
        else:
            check_warning_message_list.append(per_invalid_image_message)

    # 【只告警】被改写过的条数与存活总数: 前者规则微调时容易偏离、
    # 后者依赖图像层结果,都只打印告警
    if rewrite_caption_count != EXPECTED_REWRITE_CAPTION_COUNT:
        check_warning_message_list.append(
            f'rewrite caption count not match '
            f'{rewrite_caption_count} != {EXPECTED_REWRITE_CAPTION_COUNT}')

    if filter_pass_count != EXPECTED_FILTER_PASS_COUNT:
        check_warning_message_list.append(
            f'filter pass count not match '
            f'{filter_pass_count} != {EXPECTED_FILTER_PASS_COUNT}')

    if not CHECK_EXPECTED_FILTER_COUNT_FLAG:
        # 故意调过文本过滤规则时先关掉这个开关跑一遍,
        # 拿resave_check_result.json里的真实数字回填常量再打开
        check_warning_message_list.append(
            'check expected filter count flag is off')

        return check_error_message_list, check_warning_message_list

    # 【硬失败】纯文本层各项过滤条数逐项对账
    for per_filter_count_key_name in sorted(
            EXPECTED_TEXT_FILTER_COUNT_DICT.keys()):
        per_expected_filter_count = EXPECTED_TEXT_FILTER_COUNT_DICT[
            per_filter_count_key_name]
        per_filter_count = filter_count_dict.get(per_filter_count_key_name, 0)
        if per_filter_count != per_expected_filter_count:
            check_error_message_list.append(
                f'{per_filter_count_key_name} not match '
                f'{per_filter_count} != {per_expected_filter_count}')

    return check_error_message_list, check_warning_message_list


def get_deduplicated_image_annotation_pair(image_annotation_pair_list):
    """先按图像内容md5去重,再按t2i描述去重,保证落盘后图像和描述都全局唯一

    两步的必要性(全量与抽样实测):
    1) 按图像内容md5去重(E2): LAION原始数据里同一张图会被不同网页重复收录,
       实测随机20万张图里有1.26%字节级重复,最极端的一条描述对应4874个样本id、
       抽查其中400个的md5完全相同(即同一张514x514的图被复制了几千份)。
       同一个md5必然是同一张图,留多份只会让这张图在训练时被反复采到。
    2) 按描述去重(E3): 剩下的重复里还有"一文多图"(同一场活动/同一款商品的一组连拍,
       网页alt-text相同但照片不同)。实测随机3000个多成员描述组里80%组内图像互不相同,
       所以E3会砍掉一批图像不同的正常样本; 按方案确认仍然要做,
       目的是让落盘后"每条描述也唯一",避免同一条prompt对应多张图。
    实测E2去重掉的图像是E3的子集(结果都是4457233对),但E2仍然照做并单独计数:
    它是"图像唯一"这个性质的直接证据,而且md5是在图像校验里顺手算的、零额外IO。

    两步的排序键都把样本id(int)放在最后当tie-breaker,保证同一个md5/同一条描述
    留下的永远是id最小的那条、结果完全可复现。
    """
    # E2: 按图像内容md5去重
    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[4], x[5]])

    duplicate_image_md5_dict = {}
    duplicate_image_md5_list = []
    deduplicated_image_annotation_pair_list = []
    for per_image_annotation_pair in image_annotation_pair_list:
        per_series_name, per_image_path, per_image_name_prefix, per_t2i_caption, per_image_md5, per_sample_id = per_image_annotation_pair
        if per_image_md5 in duplicate_image_md5_dict:
            duplicate_image_md5_list.append(per_image_md5)
            print('2222', per_image_path, per_image_md5)
            continue

        duplicate_image_md5_dict[per_image_md5] = 1
        deduplicated_image_annotation_pair_list.append(
            per_image_annotation_pair)

    # E3: 按t2i描述去重
    deduplicated_image_annotation_pair_list = sorted(
        deduplicated_image_annotation_pair_list, key=lambda x: [x[3], x[5]])

    duplicate_caption_dict = {}
    duplicate_caption_count = 0
    duplicate_caption_sample_list = []
    image_annotation_pair_list = []
    for per_image_annotation_pair in deduplicated_image_annotation_pair_list:
        per_series_name, per_image_path, per_image_name_prefix, per_t2i_caption, per_image_md5, per_sample_id = per_image_annotation_pair
        if per_t2i_caption in duplicate_caption_dict:
            duplicate_caption_count += 1
            if len(duplicate_caption_sample_list) < 10000:
                duplicate_caption_sample_list.append(per_t2i_caption[:100])
            continue

        duplicate_caption_dict[per_t2i_caption] = 1
        image_annotation_pair_list.append(per_image_annotation_pair)

    image_annotation_pair_list = sorted(image_annotation_pair_list,
                                        key=lambda x: [x[0], x[2]])

    return [
        image_annotation_pair_list,
        duplicate_image_md5_list,
        duplicate_caption_count,
        duplicate_caption_sample_list,
    ]


def get_all_image_save_folder_pair(image_annotation_pair_list,
                                   save_dataset_path):
    """把过滤后的合格样本按系列分组,排序后每10000张切成一个文件夹、每100个文件夹归一个子集目录

    切分必须在过滤和去重全部完成之后做,且切分前先按原图名前缀排序,这样才能保证
    每个文件夹都是满10000张(只有每个系列全局最后一个文件夹允许不满)。
    排序键用原图名前缀而不是保存图像名: 保存图像名里含子集目录名,而子集目录名恰恰由
    排序后的位置决定,存在循环依赖; 同一个子集目录内所有图像名前缀完全相同,
    所以按原图名前缀排序与按保存图像名排序结果完全等价。
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
                _, per_image_path, per_image_name_prefix, per_t2i_caption, _, _ = per_image_annotation_pair
                # 保存图像名统一全小写,形如faceid_6m_laion_512_000_<原id>.jpg
                per_save_image_name = f'{DATASET_NAME}_{per_set_name}_{per_image_name_prefix}{SAVE_IMAGE_NAME_SUFFIX}'
                per_folder_save_pair_list.append([
                    per_image_path,
                    per_save_image_name,
                    per_t2i_caption,
                ])

            # 一个文件夹就是一个写盘任务,worker写完这10000张后直接写出该文件夹的json,
            # 主进程只收计数,不用把445万条记录再攒一遍
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
    jpg编码参数用cv2.imencode('.jpg', img)的默认值(质量95 + 色度4:2:0),
    与001~006/008/009完全一致(只有007.resave_foundir_dataset.py因为是图像复原任务、
    对GT保真最敏感才单独用了质量97 + 色度4:4:4)。本数据集的源图**本身就是JPEG**
    (上游成员后缀叫.png但内容是JPEG,实测抽样3000张100%是JPEG),
    按007注释里验证过的"量化表幂等性",用同一档质量重编码近乎无损,
    换成更高质量反而可能因为量化表不同而引入新误差,且体积要涨约50%。
    以文件夹为任务粒度而不是以单张图为粒度: 本数据集约445万张图,逐图收结果的话
    主进程要再攒一份445万条的列表,而且中途挂了只能从头再来;按文件夹收之后
    主进程内存只和文件夹数(约446)相关,且json已经写全的文件夹可以直接跳过、支持断点续跑。
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
        # 保证记录的长度和t2i_caption永远自洽
        # (该字符串在过滤阶段已经strip并归一化过)
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
    描述的复检口径和过滤阶段完全一致(长度/词数/文字字符/URL/域名/图片后缀/
    成人内容/改写后残留噪声),因为json里存的就是过滤时判定的那个字符串。
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

        # 每条标注固定只有这四个key,多一个少一个都报错
        if sorted(per_annotation.keys()) != sorted([
                'width',
                'height',
                't2i_caption',
                't2i_caption_length',
        ]):
            check_error_message_list.append(
                f'{per_save_image_name} annotation key not match')
            continue

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

        per_t2i_caption = per_annotation['t2i_caption']

        if len(per_t2i_caption.strip()) < MIN_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_image_name} still an invalid caption')
        if len(per_t2i_caption.strip()) > MAX_CAPTION_LENGTH:
            check_error_message_list.append(
                f'{per_save_image_name} still a too long caption')
        if not CAPTION_WORD_CHAR_PATTERN.search(per_t2i_caption):
            check_error_message_list.append(
                f'{per_save_image_name} still a no word char caption')
        if get_caption_word_num(per_t2i_caption) < MIN_CAPTION_WORD_NUM:
            check_error_message_list.append(
                f'{per_save_image_name} still a one word caption')
        # 落盘后的描述里不允许再残留URL/域名/图片后缀
        if check_url_caption(per_t2i_caption):
            check_error_message_list.append(
                f'{per_save_image_name} still an url caption')
        if check_domain_caption(per_t2i_caption):
            check_error_message_list.append(
                f'{per_save_image_name} still a domain caption')
        if check_image_suffix_caption(per_t2i_caption):
            check_error_message_list.append(
                f'{per_save_image_name} still an image suffix caption')
        # 也不允许残留成人内容
        if check_nsfw_caption(per_t2i_caption):
            check_error_message_list.append(
                f'{per_save_image_name} still a nsfw caption')
        # 改写之后应该被判掉的那些残留噪声也不允许出现
        per_invalid_rewrite_caption_count_key_name = get_invalid_rewrite_caption_count_key_name(
            per_t2i_caption)
        if per_invalid_rewrite_caption_count_key_name:
            check_error_message_list.append(
                f'{per_save_image_name} still a {per_invalid_rewrite_caption_count_key_name} caption'
            )
        # 落盘的描述必须已经是归一化后的形态(再改写一遍不应该再有任何变化),
        # 这条能兜住"改写函数被改了但没重跑"这种不一致
        if get_normalized_t2i_caption(per_t2i_caption) != per_t2i_caption:
            check_error_message_list.append(
                f'{per_save_image_name} caption not normalized')
        # 记录的描述长度必须和描述字符串的实际长度对得上
        if per_annotation['t2i_caption_length'] != len(per_t2i_caption):
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

    口径与003完全一致: 子集目录只是100个文件夹的容器,laion_512被切成约5个子集目录,
    所以只有整个laion_512系列全局最后一个文件夹允许不满,
    laion_512_000..003下的每个文件夹都必须是满10000张;
    同理每个系列只有最后一个子集目录允许不满100个文件夹。
    约446个文件夹每个都要listdir一万个文件再load一份json,串行跑在NAS上太久,
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

    image_annotation_pair_list, series_annotation_file_count_dict, series_row_count_dict, total_annotation_count, load_annotation_failed_count, missing_image_count, invalid_image_name_count, invalid_task_type_count, annotation_image_size_not_match_count, rewrite_caption_count, filter_count_dict = get_all_image_annotation_pair(
        root_dataset_path)

    print('1111', series_annotation_file_count_dict, series_row_count_dict,
          total_annotation_count, load_annotation_failed_count,
          missing_image_count, invalid_image_name_count,
          invalid_task_type_count, annotation_image_size_not_match_count,
          rewrite_caption_count, len(image_annotation_pair_list))
    print('1111', 'filter count:', filter_count_dict)

    if len(image_annotation_pair_list) > 0:
        print('1111', image_annotation_pair_list[0])

    # 缺图/图像名非法/任务类型不对这三类在上游实测都是0条,
    # 出现非0说明上游产物被改动过,它们没有单独的EXPECTED常量,
    # 所以并进filter_count_dict的闭合校验之前先单独硬拦一次
    load_annotation_error_message_list = []
    if load_annotation_failed_count > 0:
        load_annotation_error_message_list.append(
            f'load annotation failed count {load_annotation_failed_count}')
    if missing_image_count > 0:
        load_annotation_error_message_list.append(
            f'missing image count {missing_image_count}')
    if invalid_image_name_count > 0:
        load_annotation_error_message_list.append(
            f'invalid image name count {invalid_image_name_count}')
    if invalid_task_type_count > 0:
        load_annotation_error_message_list.append(
            f'invalid task type count {invalid_task_type_count}')

    if len(load_annotation_error_message_list) > 0:
        raise Exception(
            f'load annotation failed {load_annotation_error_message_list}')

    # 标注侧硬对账不过直接中断,不白跑后面几十小时的图像重编码
    load_annotation_check_error_message_list, load_annotation_check_warning_message_list = check_load_annotation_count(
        series_annotation_file_count_dict, series_row_count_dict,
        total_annotation_count, filter_count_dict, rewrite_caption_count,
        len(image_annotation_pair_list))

    for per_warning_message in load_annotation_check_warning_message_list:
        print('2222', per_warning_message)

    if len(load_annotation_check_error_message_list) > 0:
        raise Exception(
            f'check annotation failed {load_annotation_check_error_message_list[:20]}'
        )

    # 进入去重阶段的条数,后面的去重闭合校验要用它当ground truth
    filter_pass_count = len(image_annotation_pair_list)

    image_annotation_pair_list, duplicate_image_md5_list, duplicate_caption_count, duplicate_caption_sample_list = get_deduplicated_image_annotation_pair(
        image_annotation_pair_list)

    print('1111', len(image_annotation_pair_list),
          len(duplicate_image_md5_list), duplicate_caption_count)
    if len(duplicate_image_md5_list) > 0:
        print('1111', duplicate_image_md5_list[:10])
    if len(duplicate_caption_sample_list) > 0:
        print('1111', duplicate_caption_sample_list[:10])

    dedup_check_error_message_list, dedup_check_warning_message_list = [], []

    # 【始终硬失败】去重闭合校验: 存活数 + 两步各自去掉的条数 必须正好等于
    # 进入去重阶段的条数。这里用**本次实跑出来的**filter_pass_count而不是
    # EXPECTED_FILTER_PASS_COUNT,所以它是纯结构性校验、不依赖任何实测期望值,
    # 能兜住"某条样本既没被去重也没进存活列表"或"被重复计数"这两种bug
    if len(image_annotation_pair_list) + len(
            duplicate_image_md5_list
    ) + duplicate_caption_count != filter_pass_count:
        dedup_check_error_message_list.append(
            f'dedup count not self consistent '
            f'{len(image_annotation_pair_list)} + {len(duplicate_image_md5_list)} + '
            f'{duplicate_caption_count} != {filter_pass_count}')

    # 【只告警】去重条数与最终存活数: 它们都依赖图像层的真解码结果
    # (E2的md5更是编写阶段完全没能全量实测,详见
    #  EXPECTED_DUPLICATE_IMAGE_MD5_COUNT的注释),所以只打印告警,
    # 真实数字会写进resave_check_result.json供回填
    if len(duplicate_image_md5_list) != EXPECTED_DUPLICATE_IMAGE_MD5_COUNT:
        dedup_check_warning_message_list.append(
            f'duplicate image md5 count not match '
            f'{len(duplicate_image_md5_list)} != {EXPECTED_DUPLICATE_IMAGE_MD5_COUNT}'
        )
    if duplicate_caption_count != EXPECTED_DUPLICATE_CAPTION_COUNT:
        dedup_check_warning_message_list.append(
            f'duplicate caption count not match '
            f'{duplicate_caption_count} != {EXPECTED_DUPLICATE_CAPTION_COUNT}')
    if len(image_annotation_pair_list
           ) != EXPECTED_VALID_IMAGE_ANNOTATION_COUNT:
        dedup_check_warning_message_list.append(
            f'valid image annotation count not match '
            f'{len(image_annotation_pair_list)} != {EXPECTED_VALID_IMAGE_ANNOTATION_COUNT}'
        )

    # 一个合格样本都不剩一定是硬错误,与任何开关无关
    if len(image_annotation_pair_list) == 0:
        dedup_check_error_message_list.append(
            'no valid image annotation pair found')

    for per_warning_message in dedup_check_warning_message_list:
        print('2222', per_warning_message)

    if len(dedup_check_error_message_list) > 0:
        raise Exception(
            f'check dedup failed {dedup_check_error_message_list[:20]}')

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
    # 所以这里再兜一道,撞上就直接中止
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

    # 【始终硬失败】切分闭合校验: 切出来的文件夹里的图像总数必须正好等于
    # 去重后的存活数,一张都不能多也不能少。纯结构性校验,不依赖期望值
    if total_save_task_image_count != len(image_annotation_pair_list):
        raise Exception(
            f'save task image count not match '
            f'{total_save_task_image_count} != {len(image_annotation_pair_list)}'
        )

    # 【只告警】文件夹数与子集目录数: 它们由存活数除以10000算出来,
    # 而存活数依赖图像层真解码结果,所以只打印告警
    if len(image_save_folder_pair_list) != EXPECTED_SAVE_FOLDER_COUNT:
        print(
            '2222', f'save folder count not match '
            f'{len(image_save_folder_pair_list)} != {EXPECTED_SAVE_FOLDER_COUNT}'
        )
    if len(set_folder_count_dict) != EXPECTED_SAVE_SET_COUNT:
        print(
            '2222', f'save set count not match '
            f'{len(set_folder_count_dict)} != {EXPECTED_SAVE_SET_COUNT}')

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
          'missing image:', missing_image_count, 'invalid image name:',
          invalid_image_name_count, 'invalid task type:',
          invalid_task_type_count, 'annotation image size not match:',
          annotation_image_size_not_match_count, 'rewrite caption:',
          rewrite_caption_count, 'filter count:',
          filter_count_dict, 'duplicate image md5:',
          len(duplicate_image_md5_list), 'duplicate caption:',
          duplicate_caption_count, 'save image failed:',
          save_image_failed_count, 'total save image:',
          total_save_image_count, 'total save folder:',
          len(folder_image_count_dict), 'total save set:',
          len(set_folder_count_dict),
          'check total image:', check_total_image_count, 'check error:',
          len(check_error_message_list))

    save_check_result_path = os.path.join(save_dataset_path,
                                          'resave_check_result.json')
    save_check_result_dict = {
        'series_annotation_file_count_dict': series_annotation_file_count_dict,
        'series_row_count_dict': series_row_count_dict,
        'total_annotation_count': total_annotation_count,
        'load_annotation_failed_count': load_annotation_failed_count,
        'missing_image_count': missing_image_count,
        'invalid_image_name_count': invalid_image_name_count,
        'invalid_task_type_count': invalid_task_type_count,
        'annotation_image_size_not_match_count':
        annotation_image_size_not_match_count,
        'rewrite_caption_count': rewrite_caption_count,
        'filter_count_dict': filter_count_dict,
        'filter_pass_count': filter_pass_count,
        'duplicate_image_md5_count': len(duplicate_image_md5_list),
        'duplicate_caption_count': duplicate_caption_count,
        'total_save_task_image_count': total_save_task_image_count,
        'save_image_failed_count': save_image_failed_count,
        'total_save_image_count': total_save_image_count,
        'total_save_folder_count': len(folder_image_count_dict),
        'total_save_set_count': len(set_folder_count_dict),
        'check_total_image_count': check_total_image_count,
        'check_error_count': len(check_error_message_list),
        'series_set_name_list_dict': series_set_name_list_dict,
        'set_folder_count_dict': set_folder_count_dict,
        'folder_image_count_dict': folder_image_count_dict,
        'duplicate_image_md5_list': duplicate_image_md5_list[:10000],
        'duplicate_caption_sample_list': duplicate_caption_sample_list[:10000],
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
    root_dataset_path = r'/root/autodl-tmp/huggingface_datasets_unzip/FaceID-6M'
    save_dataset_path = r'/root/autodl-tmp/t2i_datasets'
    preprocess_dataset(root_dataset_path, save_dataset_path)
