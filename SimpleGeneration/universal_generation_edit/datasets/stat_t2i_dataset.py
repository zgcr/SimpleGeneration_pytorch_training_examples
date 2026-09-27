import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import bisect
import json

from collections import Counter
from fractions import Fraction
from multiprocessing import Pool

from tqdm import tqdm

from SimpleGeneration.universal_generation_edit.aspect_ratio_buckets import ASPECT_RATIO_BUCKETS, get_closest_bucket

# 统计t2i_datasets下每个t2i数据集的图像分辨率分布、图像宽高比桶分布和caption
# 长度分布。
#
# 全程只读<folder_name>.json标注,不读任何图像文件: 图像的width/height在标注里
# 已经记录过一遍(上游resave时用真实解码尺寸写入,001.check_t2i_dataset.py也已经
# 逐张比对过标注宽高与真实宽高一致),所以统计分辨率和宽高比完全不需要再解码
# 上亿张图。
#
# 上亿条样本不可能把每条的长边/宽高比/caption长度都存成数组,所以每个worker只
# 回传三个直方图(Counter): 长边取值->条数、宽高比桶号->条数、caption长度->条数。
# 取值空间只有几十到几万种,内存是常数级,而且直方图上算分位数是精确的
# (nearest-rank),不是近似。
#
# 本脚本只读数据集,不做任何修改,结果写到脚本同级的check_result目录下。

ROOT_DATASET_PATH = r'/root/autodl-tmp/t2i_datasets'

# 每个数据集的中间统计结果json落在check_result目录下,中断之后可以续跑
SAVE_STAT_RESULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'check_result')

# 最终汇总的md表格写到脚本同级目录,不和中间结果混在一起
SAVE_STAT_RESULT_MARKDOWN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 't2i_dataset_statistics.md')

PROCESS_NUM = 64

# 图像长边区间统计口径: 下界逐档抬高,上界统一是MAX_IMAGE_LONG_SIDE,
# 两端都是闭区间
IMAGE_LONG_SIDE_THRESHOLD_LIST = [
    64,
    128,
    256,
    512,
    1024,
    2048,
]

MAX_IMAGE_LONG_SIDE = 4096

# caption长度分位数档位
CAPTION_LENGTH_PERCENTILE_LIST = [
    1,
    5,
    10,
    50,
    90,
    95,
    99,
]

# ------------------------------------------------------------------------------
# 图像宽高比桶统计口径
#
# ASPECT_RATIO_BUCKETS是一张纯比例表(与训练阶段的base_resize无关),按比例
# 递增排列: 索引0是最高的0.25(1:4),索引20是正方形1.0,索引40是最宽的4.0(4:1)。
# 所以表的两端天然就是"能被归桶"的比例上下界。
#
# 比例小于MIN_ASPECT_RATIO或大于MAX_ASPECT_RATIO的样本落在桶表覆盖范围之外
# (T2IDataset默认口径min_aspect_ratio=0.25/max_aspect_ratio=4.0也会把它们
# 过滤掉),统计时用两个哨兵桶号单独成档,避免和正常的0~40混在一起。
# ------------------------------------------------------------------------------
MIN_ASPECT_RATIO = ASPECT_RATIO_BUCKETS[0]
MAX_ASPECT_RATIO = ASPECT_RATIO_BUCKETS[-1]

# 哨兵桶号取在0~40两侧,保证"越界小档 -> 桶0~桶40 -> 越界大档"整体仍是按宽高比
# 从小到大排的
LESS_THAN_MIN_ASPECT_RATIO_BUCKET_INDEX = -1
GREATER_THAN_MAX_ASPECT_RATIO_BUCKET_INDEX = len(ASPECT_RATIO_BUCKETS)

# T2IDataset的默认过滤口径,用来额外统计"这个数据集真正能被T2IDataset加载的
# 样本数",与上面的分辨率区间统计、宽高比桶统计相互独立
MIN_IMAGE_LONG_SIDE = 64
MIN_T2I_CAPTION_LENGTH = 4
MAX_T2I_CAPTION_LENGTH = 1024


def get_aspect_ratio_bucket_index(aspect_ratio):
    """求宽高比落在哪个桶,结果与get_closest_bucket的桶号完全一致

    get_closest_bucket是把41个桶逐个扫一遍取最近的,1.4亿条样本就是57亿次浮点
    比较,纯python跑不动。桶表本身已经按宽高比从小到大排好,最近的桶只可能是
    插入点左右两个邻居之一,所以这里先二分定位插入点,再只比这两个邻居,单条
    样本从41次比较降到约6次。

    两个邻居的距离用和get_closest_bucket完全相同的减法表达式算出(左邻居用
    aspect_ratio - bucket,右邻居用bucket - aspect_ratio),所以连浮点舍入行为
    都一致;距离相等时取比例更小的那个,对应get_closest_bucket从小比例端(索引0)
    开始扫描、只在严格更近时才换桶(diff < best_diff)的取法。
    """
    bucket_num = len(ASPECT_RATIO_BUCKETS)

    right_index = bisect.bisect_left(ASPECT_RATIO_BUCKETS, aspect_ratio)

    if right_index <= 0:
        return 0

    if right_index >= bucket_num:
        return bucket_num - 1

    left_diff = aspect_ratio - ASPECT_RATIO_BUCKETS[right_index - 1]
    right_diff = ASPECT_RATIO_BUCKETS[right_index] - aspect_ratio

    return right_index if right_diff < left_diff else right_index - 1


def check_aspect_ratio_bucket_index_consistency():
    """逐档assert二分归桶与get_closest_bucket同解

    统计口径必须和T2IDataset里真正用的get_closest_bucket一模一样,否则这份统计
    没法用来指导训练配置。这里把最容易出分歧的位置全部试一遍: 41个桶比例本身、
    相邻桶的中点及其两侧、以及整个[0.25, 4.0]区间上的密集扫描点。
    """
    check_aspect_ratio_list = []

    for per_bucket_ratio in ASPECT_RATIO_BUCKETS:
        check_aspect_ratio_list.append(per_bucket_ratio)

    for per_index in range(len(ASPECT_RATIO_BUCKETS) - 1):
        per_middle_ratio = (ASPECT_RATIO_BUCKETS[per_index] +
                            ASPECT_RATIO_BUCKETS[per_index + 1]) / 2
        check_aspect_ratio_list.append(per_middle_ratio)
        check_aspect_ratio_list.append(per_middle_ratio - 1e-9)
        check_aspect_ratio_list.append(per_middle_ratio + 1e-9)

    for per_step in range(10001):
        check_aspect_ratio_list.append(MIN_ASPECT_RATIO +
                                       (MAX_ASPECT_RATIO - MIN_ASPECT_RATIO) *
                                       per_step / 10000)

    for per_aspect_ratio in check_aspect_ratio_list:
        _, per_expect_bucket_index = get_closest_bucket(
            per_aspect_ratio, 1.0, ASPECT_RATIO_BUCKETS)
        per_bucket_index = get_aspect_ratio_bucket_index(per_aspect_ratio)

        assert per_bucket_index == per_expect_bucket_index, f'aspect_ratio:{per_aspect_ratio}, bisect bucket:{per_bucket_index} != get_closest_bucket bucket:{per_expect_bucket_index}'

    print('0000', 'aspect ratio bucket index consistency checked:',
          len(check_aspect_ratio_list), 'aspect ratios matched')

    return True


def get_aspect_ratio_bucket_name_list():
    """宽高比桶的统计档位名,首尾两档是桶表覆盖范围之外的越界样本"""
    aspect_ratio_bucket_name_list = ['aspect_ratio_less_than_min_count']

    for per_bucket_index in range(len(ASPECT_RATIO_BUCKETS)):
        aspect_ratio_bucket_name_list.append(
            f'bucket_{per_bucket_index}_count')

    aspect_ratio_bucket_name_list.append('aspect_ratio_greater_than_max_count')

    return aspect_ratio_bucket_name_list


def get_aspect_ratio_bucket_index_list():
    """与get_aspect_ratio_bucket_name_list一一对应的直方图桶号"""
    aspect_ratio_bucket_index_list = [LESS_THAN_MIN_ASPECT_RATIO_BUCKET_INDEX]

    for per_bucket_index in range(len(ASPECT_RATIO_BUCKETS)):
        aspect_ratio_bucket_index_list.append(per_bucket_index)

    aspect_ratio_bucket_index_list.append(
        GREATER_THAN_MAX_ASPECT_RATIO_BUCKET_INDEX)

    return aspect_ratio_bucket_index_list


def get_dataset_name_list(root_dataset_path):
    """t2i_datasets下每个一级子目录都是一个t2i数据集"""
    dataset_name_list = []
    for per_name in sorted(os.listdir(root_dataset_path)):
        if os.path.isdir(os.path.join(root_dataset_path, per_name)):
            dataset_name_list.append(per_name)

    return dataset_name_list


def get_all_annotation_json_path(per_dataset_dir_path):
    """列出一个数据集下全部<set_name>/<folder_name>.json标注文件

    T2IDataset是"先遍历<folder_name>目录、再读同级同名json"。这里反过来直接以
    json为准: 统计的是标注,而且一个分片目录里的图像与它的json是一一对应的
    (001.check_t2i_dataset.py已经复核过孤立标注/孤立图像都为0),所以以json为准
    既等价又能省掉近万次对图像目录的listdir。
    """
    annotation_json_path_list = []

    for per_set_name in sorted(os.listdir(per_dataset_dir_path)):
        per_set_dir_path = os.path.join(per_dataset_dir_path, per_set_name)
        if not os.path.isdir(per_set_dir_path):
            continue

        for per_sub_name in sorted(os.listdir(per_set_dir_path)):
            if not per_sub_name.endswith('.json'):
                continue

            annotation_json_path_list.append(
                os.path.join(per_set_dir_path, per_sub_name))

    return annotation_json_path_list


def stat_single_annotation_json(per_annotation_json_path):
    """统计单个分片json,回传长边/宽高比桶/caption长度三个直方图

    直方图而不是原始值列表: 单个分片1万条尚可,但数据集级别要合并上亿条,
    只有直方图才能让内存保持常数级。
    """
    per_image_long_side_counter = Counter()
    per_aspect_ratio_bucket_counter = Counter()
    per_caption_length_counter = Counter()

    per_stat_result_dict = {
        'total_annotation_count': 0,
        'invalid_annotation_count': 0,
        'load_json_failed_count': 0,
        't2i_dataset_loadable_count': 0,
        'aspect_ratio_min': None,
        'aspect_ratio_max': None,
    }

    try:
        with open(per_annotation_json_path, 'r',
                  encoding='UTF-8') as load_json_file:
            per_annotation_dict = json.load(load_json_file)
    except Exception:
        per_stat_result_dict['load_json_failed_count'] = 1

        return [
            per_annotation_json_path,
            per_stat_result_dict,
            per_image_long_side_counter,
            per_aspect_ratio_bucket_counter,
            per_caption_length_counter,
        ]

    if not isinstance(per_annotation_dict, dict):
        per_stat_result_dict['load_json_failed_count'] = 1

        return [
            per_annotation_json_path,
            per_stat_result_dict,
            per_image_long_side_counter,
            per_aspect_ratio_bucket_counter,
            per_caption_length_counter,
        ]

    per_stat_result_dict['total_annotation_count'] = len(per_annotation_dict)

    for per_image_name, per_annotation in per_annotation_dict.items():
        # T2IDataset对非dict标注是直接跳过的,这里口径保持一致,同时单独计数
        if not isinstance(per_annotation, dict):
            per_stat_result_dict['invalid_annotation_count'] += 1
            continue

        per_image_w = per_annotation.get('width', None)
        per_image_h = per_annotation.get('height', None)
        per_caption_length = per_annotation.get('t2i_caption_length', None)

        # bool是int的子类,必须显式排除,否则True会被当成1参与统计
        if not isinstance(per_image_w, int) or isinstance(
                per_image_w,
                bool) or not isinstance(per_image_h, int) or isinstance(
                    per_image_h, bool) or not isinstance(
                        per_caption_length, int) or isinstance(
                            per_caption_length, bool):
            per_stat_result_dict['invalid_annotation_count'] += 1
            continue

        if per_image_w <= 0 or per_image_h <= 0 or per_caption_length < 0:
            per_stat_result_dict['invalid_annotation_count'] += 1
            continue

        per_image_long_side = max(per_image_w, per_image_h)
        per_image_aspect_ratio = per_image_w / per_image_h

        # 三路互斥: 越界两档 + 桶表内的41档,加起来正好是全部有效样本,
        # 所以各档百分比天然合计100%
        if per_image_aspect_ratio < MIN_ASPECT_RATIO:
            per_aspect_ratio_bucket_index = LESS_THAN_MIN_ASPECT_RATIO_BUCKET_INDEX
        elif per_image_aspect_ratio > MAX_ASPECT_RATIO:
            per_aspect_ratio_bucket_index = GREATER_THAN_MAX_ASPECT_RATIO_BUCKET_INDEX
        else:
            per_aspect_ratio_bucket_index = get_aspect_ratio_bucket_index(
                per_image_aspect_ratio)

        per_image_long_side_counter[per_image_long_side] += 1
        per_aspect_ratio_bucket_counter[per_aspect_ratio_bucket_index] += 1
        per_caption_length_counter[per_caption_length] += 1

        # 宽高比是连续值,不能像长边那样进直方图求极值,只能一路维护min/max
        if per_stat_result_dict['aspect_ratio_min'] is None or (
                per_image_aspect_ratio
                < per_stat_result_dict['aspect_ratio_min']):
            per_stat_result_dict['aspect_ratio_min'] = per_image_aspect_ratio
        if per_stat_result_dict['aspect_ratio_max'] is None or (
                per_image_aspect_ratio
                > per_stat_result_dict['aspect_ratio_max']):
            per_stat_result_dict['aspect_ratio_max'] = per_image_aspect_ratio

        # T2IDataset默认过滤口径下真正会被加载进来的样本
        if (MIN_IMAGE_LONG_SIDE <= per_image_long_side <= MAX_IMAGE_LONG_SIDE
                and
                MIN_ASPECT_RATIO <= per_image_aspect_ratio <= MAX_ASPECT_RATIO
                and MIN_T2I_CAPTION_LENGTH <= per_caption_length <=
                MAX_T2I_CAPTION_LENGTH):
            per_stat_result_dict['t2i_dataset_loadable_count'] += 1

    return [
        per_annotation_json_path,
        per_stat_result_dict,
        per_image_long_side_counter,
        per_aspect_ratio_bucket_counter,
        per_caption_length_counter,
    ]


def get_counter_total_count(per_counter):
    return sum(per_counter.values())


def get_counter_range_count(per_counter, min_value, max_value):
    """直方图上统计落在[min_value, max_value]闭区间内的条数"""
    range_count = 0
    for per_value, per_count in per_counter.items():
        if min_value <= per_value <= max_value:
            range_count += per_count

    return range_count


def get_counter_percentile(per_counter, percentile):
    """直方图上按nearest-rank精确求分位数

    把所有取值从小到大排开,第ceil(p/100 * N)个值(1起点)就是p分位数。直方图里
    每个取值带着自己的条数,累加条数越过这个秩的那个取值即为所求,不需要展开成
    上亿元素的数组,也不做任何插值近似。
    """
    total_count = get_counter_total_count(per_counter)
    if total_count == 0:
        return None

    # ceil(percentile / 100 * total_count),用整数运算避免浮点误差
    target_rank = -((-percentile * total_count) // 100)
    target_rank = max(1, min(total_count, int(target_rank)))

    accumulate_count = 0
    for per_value in sorted(per_counter.keys()):
        accumulate_count += per_counter[per_value]
        if accumulate_count >= target_rank:
            return per_value

    return max(per_counter.keys())


def get_counter_mean(per_counter):
    total_count = get_counter_total_count(per_counter)
    if total_count == 0:
        return None

    total_value = 0
    for per_value, per_count in per_counter.items():
        total_value += per_value * per_count

    return round(total_value / total_count, 2)


def stat_single_dataset(root_dataset_path, per_dataset_name,
                        save_stat_result_path):
    """统计单个t2i数据集,结果落盘成一个json"""
    save_stat_result_json_path = os.path.join(
        save_stat_result_path, f'{per_dataset_name}_t2i_stat_result.json')

    # 每个数据集单独落盘,已经统计过的直接跳过,GPIC这种要跑很久的数据集
    # 中断之后可以续跑
    if os.path.exists(save_stat_result_json_path):
        print('2222', per_dataset_name, 'stat result already exists, skip')
        with open(save_stat_result_json_path, 'r',
                  encoding='UTF-8') as load_json_file:
            return json.load(load_json_file)

    per_dataset_dir_path = os.path.join(root_dataset_path, per_dataset_name)

    annotation_json_path_list = get_all_annotation_json_path(
        per_dataset_dir_path)

    print('3333', per_dataset_name, 'annotation json:',
          len(annotation_json_path_list))

    image_long_side_counter = Counter()
    aspect_ratio_bucket_counter = Counter()
    caption_length_counter = Counter()

    total_annotation_count = 0
    invalid_annotation_count = 0
    load_json_failed_count = 0
    t2i_dataset_loadable_count = 0
    load_json_failed_path_list = []
    aspect_ratio_min = None
    aspect_ratio_max = None

    with Pool(processes=min(PROCESS_NUM, max(len(annotation_json_path_list),
                                             1))) as pool:
        for per_stat_result in tqdm(pool.imap_unordered(
                stat_single_annotation_json, annotation_json_path_list),
                                    total=len(annotation_json_path_list)):
            per_annotation_json_path, per_stat_result_dict, per_image_long_side_counter, per_aspect_ratio_bucket_counter, per_caption_length_counter = per_stat_result

            total_annotation_count += per_stat_result_dict[
                'total_annotation_count']
            invalid_annotation_count += per_stat_result_dict[
                'invalid_annotation_count']
            load_json_failed_count += per_stat_result_dict[
                'load_json_failed_count']
            t2i_dataset_loadable_count += per_stat_result_dict[
                't2i_dataset_loadable_count']

            if per_stat_result_dict['load_json_failed_count'] > 0:
                load_json_failed_path_list.append(per_annotation_json_path)

            if per_stat_result_dict['aspect_ratio_min'] is not None:
                if aspect_ratio_min is None or per_stat_result_dict[
                        'aspect_ratio_min'] < aspect_ratio_min:
                    aspect_ratio_min = per_stat_result_dict['aspect_ratio_min']
            if per_stat_result_dict['aspect_ratio_max'] is not None:
                if aspect_ratio_max is None or per_stat_result_dict[
                        'aspect_ratio_max'] > aspect_ratio_max:
                    aspect_ratio_max = per_stat_result_dict['aspect_ratio_max']

            image_long_side_counter.update(per_image_long_side_counter)
            aspect_ratio_bucket_counter.update(per_aspect_ratio_bucket_counter)
            caption_length_counter.update(per_caption_length_counter)

    valid_annotation_count = get_counter_total_count(image_long_side_counter)

    # 宽高比桶直方图和长边直方图统计的是同一批有效样本,这里顺手复核一遍,
    # 避免以后改动过滤分支时两个直方图悄悄对不上
    assert valid_annotation_count == get_counter_total_count(
        aspect_ratio_bucket_counter)

    # 各档图像长边区间的图像数量,上界统一是MAX_IMAGE_LONG_SIDE,两端闭区间
    image_long_side_range_count_dict = {}
    for per_threshold in IMAGE_LONG_SIDE_THRESHOLD_LIST:
        image_long_side_range_count_dict[
            f'long_side_{per_threshold}_to_{MAX_IMAGE_LONG_SIDE}_count'] = (
                get_counter_range_count(image_long_side_counter, per_threshold,
                                        MAX_IMAGE_LONG_SIDE))

    # 43档宽高比统计: 越界的<0.25、桶表内的0~40、越界的>4.0
    aspect_ratio_bucket_count_dict = {}
    for per_bucket_name, per_bucket_index in zip(
            get_aspect_ratio_bucket_name_list(),
            get_aspect_ratio_bucket_index_list()):
        aspect_ratio_bucket_count_dict[per_bucket_name] = (
            aspect_ratio_bucket_counter[per_bucket_index])

    caption_length_percentile_dict = {}
    for per_percentile in CAPTION_LENGTH_PERCENTILE_LIST:
        caption_length_percentile_dict[f'p{per_percentile}'] = (
            get_counter_percentile(caption_length_counter, per_percentile))

    per_dataset_stat_result_dict = {
        'dataset_name':
        per_dataset_name,
        'annotation_json_count':
        len(annotation_json_path_list),
        'total_annotation_count':
        total_annotation_count,
        'valid_annotation_count':
        valid_annotation_count,
        'invalid_annotation_count':
        invalid_annotation_count,
        'load_json_failed_count':
        load_json_failed_count,
        'load_json_failed_path_list':
        load_json_failed_path_list[:100],
        't2i_dataset_loadable_count':
        t2i_dataset_loadable_count,
        'image_long_side_range_count_dict':
        image_long_side_range_count_dict,
        'image_long_side_min':
        min(image_long_side_counter.keys())
        if valid_annotation_count > 0 else None,
        'image_long_side_max':
        max(image_long_side_counter.keys())
        if valid_annotation_count > 0 else None,
        'image_long_side_mean':
        get_counter_mean(image_long_side_counter),
        f'image_long_side_greater_than_{MAX_IMAGE_LONG_SIDE}_count':
        get_counter_range_count(image_long_side_counter,
                                MAX_IMAGE_LONG_SIDE + 1, float('inf')),
        f'image_long_side_less_than_{MIN_IMAGE_LONG_SIDE}_count':
        get_counter_range_count(image_long_side_counter, 0,
                                MIN_IMAGE_LONG_SIDE - 1),
        'aspect_ratio_bucket_count_dict':
        aspect_ratio_bucket_count_dict,
        'aspect_ratio_min':
        round(aspect_ratio_min, 4) if aspect_ratio_min is not None else None,
        'aspect_ratio_max':
        round(aspect_ratio_max, 4) if aspect_ratio_max is not None else None,
        'caption_length_percentile_dict':
        caption_length_percentile_dict,
        'caption_length_min':
        min(caption_length_counter.keys())
        if valid_annotation_count > 0 else None,
        'caption_length_max':
        max(caption_length_counter.keys())
        if valid_annotation_count > 0 else None,
        'caption_length_mean':
        get_counter_mean(caption_length_counter),
    }

    os.makedirs(save_stat_result_path, exist_ok=True)
    with open(save_stat_result_json_path, 'w',
              encoding='UTF-8') as save_json_file:
        json.dump(per_dataset_stat_result_dict,
                  save_json_file,
                  ensure_ascii=False)

    print('4444', save_stat_result_json_path)

    return per_dataset_stat_result_dict


def get_count_and_ratio_str(per_count, per_total_count):
    """数量后面跟上占总数的百分比,只看绝对数量容易把不同量级的数据集看串"""
    if per_total_count <= 0:
        return f'{per_count:,}'

    return f'{per_count:,} ({per_count / per_total_count * 100:.2f}%)'


def get_aspect_ratio_bucket_row_header_list():
    """宽高比桶表每一行的"档位"和"宽高比"两列

    宽高比同时给4位小数和最简分数写法: 小数便于直接和数据集里的width/height比,
    最简分数便于和aspect_ratio_buckets.py里用16对齐边长写成的分数对上。桶表里
    的比例分母最大只有32,用limit_denominator(64)取到的就是精确的最简分数。
    """
    aspect_ratio_bucket_row_header_list = [
        ['<0.25', f'(0, {MIN_ASPECT_RATIO})'],
    ]

    for per_bucket_index, per_bucket_ratio in enumerate(ASPECT_RATIO_BUCKETS):
        per_bucket_fraction = Fraction(per_bucket_ratio).limit_denominator(64)
        aspect_ratio_bucket_row_header_list.append([
            f'{per_bucket_index}',
            f'{per_bucket_ratio:.4f} ({per_bucket_fraction.numerator}:{per_bucket_fraction.denominator})',
        ])

    aspect_ratio_bucket_row_header_list.append(
        ['>4.0', f'({MAX_ASPECT_RATIO}, +inf)'])

    return aspect_ratio_bucket_row_header_list


def save_stat_result_markdown(all_stat_result_list,
                              save_stat_result_markdown_path):
    """把全部数据集的样本统计汇总成一个md表格"""
    markdown_line_list = []

    markdown_line_list.append('# t2i_datasets数据集统计结果')
    markdown_line_list.append('')
    markdown_line_list.append(f'- 数据集根目录: `{ROOT_DATASET_PATH}`')
    markdown_line_list.append('- 统计脚本: `stat_t2i_dataset.py`')
    markdown_line_list.append(
        '- 统计口径: 只读每个分片的`<folder_name>.json`标注,不读取任何图像文件')
    markdown_line_list.append(
        '- 统计内容: 图像长边分布、图像宽高比桶(`ASPECT_RATIO_BUCKETS`)分布、caption长度分位数')
    markdown_line_list.append('')

    # 表1: 图像总数量与各档图像长边区间的图像数量
    markdown_line_list.append('## 1. 图像数量与图像长边分布')

    markdown_line_list.append('')
    markdown_line_list.append('图像长边 = `max(标注width, 标注height)`,所有区间两端都是闭区间,'
                              '括号里是占该数据集图像总数的百分比。')
    markdown_line_list.append('')
    # 表头跟着IMAGE_LONG_SIDE_THRESHOLD_LIST自动生成,加档位时不用改这里
    per_range_header_str_list = [
        f'长边[{per_threshold}, {MAX_IMAGE_LONG_SIDE}]'
        for per_threshold in IMAGE_LONG_SIDE_THRESHOLD_LIST
    ]
    markdown_line_list.append('| 数据集 | 图像总数量 | ' +
                              ' | '.join(per_range_header_str_list) + ' |')
    markdown_line_list.append('| --- | --- | ' +
                              ' | '.join(['---'] *
                                         len(IMAGE_LONG_SIDE_THRESHOLD_LIST)) +
                              ' |')

    total_image_count = 0
    total_range_count_dict = {
        per_threshold: 0
        for per_threshold in IMAGE_LONG_SIDE_THRESHOLD_LIST
    }

    for per_stat_result_dict in all_stat_result_list:
        per_total_count = per_stat_result_dict['total_annotation_count']
        total_image_count += per_total_count

        per_range_count_str_list = []
        for per_threshold in IMAGE_LONG_SIDE_THRESHOLD_LIST:
            per_range_count = per_stat_result_dict[
                'image_long_side_range_count_dict'][
                    f'long_side_{per_threshold}_to_{MAX_IMAGE_LONG_SIDE}_count']
            total_range_count_dict[per_threshold] += per_range_count
            per_range_count_str_list.append(
                get_count_and_ratio_str(per_range_count, per_total_count))

        markdown_line_list.append(
            f'| {per_stat_result_dict["dataset_name"]} | '
            f'{per_total_count:,} | ' + ' | '.join(per_range_count_str_list) +
            ' |')

    markdown_line_list.append(
        '| **合计** | '
        f'**{total_image_count:,}** | ' + ' | '.join([
            '**' + get_count_and_ratio_str(
                total_range_count_dict[per_threshold], total_image_count) +
            '**' for per_threshold in IMAGE_LONG_SIDE_THRESHOLD_LIST
        ]) + ' |')
    markdown_line_list.append('')

    # 表2: 宽高比桶分布。43档横着排根本没法看,所以这张表转置过来:
    # 行是桶档位,列是数据集
    markdown_line_list.append('## 2. 图像宽高比桶(ASPECT_RATIO_BUCKETS)分布')

    markdown_line_list.append('')
    markdown_line_list.append(
        '宽高比 = `标注width / 标注height`,按`|桶比例 - 宽高比|`最近归桶,口径与'
        '`aspect_ratio_buckets.py`里的`get_closest_bucket`完全一致(脚本启动时会把41个桶比例、'
        '相邻桶中点及其两侧、整个区间上的密集扫描点逐个assert过一遍)。`ASPECT_RATIO_BUCKETS`是一张'
        '纯比例表,与训练阶段的`base_resize`无关,所以本表只列比例、不列具体分辨率。')
    markdown_line_list.append('')
    markdown_line_list.append(
        '比例`<0.25`与`>4.0`的样本落在桶表覆盖范围之外(T2IDataset默认口径'
        '`min_aspect_ratio=0.25`/`max_aspect_ratio=4.0`也会把它们过滤掉),单独列为首尾两档。'
        '括号里是占该数据集有效样本数的百分比,43档互斥且覆盖全部有效样本,所以每列百分比合计为100%。')
    markdown_line_list.append('')
    markdown_line_list.append(
        '表格行序与`ASPECT_RATIO_BUCKETS`的桶号顺序一致,即宽高比从小到大: 越界档`<0.25`'
        '-> 桶0(最高的0.25,1:4) -> 桶20(正方形1.0) -> 桶40(最宽的4.0,4:1) -> 越界档`>4.0`,'
        '所以从上往下看就是图像从竖到方再到横的分布曲线。')
    markdown_line_list.append('')

    aspect_ratio_bucket_name_list = get_aspect_ratio_bucket_name_list()
    aspect_ratio_bucket_row_header_list = (
        get_aspect_ratio_bucket_row_header_list())

    per_dataset_name_list = [
        per_stat_result_dict['dataset_name']
        for per_stat_result_dict in all_stat_result_list
    ]

    markdown_line_list.append('| 桶 | 宽高比(W/H) | ' +
                              ' | '.join(per_dataset_name_list) + ' | 合计 |')
    markdown_line_list.append('| --- | --- | ' +
                              ' | '.join(['---'] *
                                         len(per_dataset_name_list)) +
                              ' | --- |')

    # 每个数据集的有效样本数单独一行: 下面所有百分比都是以它为分母的
    total_valid_annotation_count = sum([
        per_stat_result_dict['valid_annotation_count']
        for per_stat_result_dict in all_stat_result_list
    ])
    markdown_line_list.append('| **有效样本数** | - | ' + ' | '.join([
        f'**{per_stat_result_dict["valid_annotation_count"]:,}**'
        for per_stat_result_dict in all_stat_result_list
    ]) + f' | **{total_valid_annotation_count:,}** |')

    for per_bucket_name, per_bucket_row_header in zip(
            aspect_ratio_bucket_name_list,
            aspect_ratio_bucket_row_header_list):
        per_bucket_count_str_list = []
        per_bucket_total_count = 0

        for per_stat_result_dict in all_stat_result_list:
            per_bucket_count = per_stat_result_dict[
                'aspect_ratio_bucket_count_dict'][per_bucket_name]
            per_bucket_total_count += per_bucket_count
            per_bucket_count_str_list.append(
                get_count_and_ratio_str(
                    per_bucket_count,
                    per_stat_result_dict['valid_annotation_count']))

        markdown_line_list.append(
            f'| {per_bucket_row_header[0]} | {per_bucket_row_header[1]} | ' +
            ' | '.join(per_bucket_count_str_list) + ' | ' +
            get_count_and_ratio_str(per_bucket_total_count,
                                    total_valid_annotation_count) + ' |')

    markdown_line_list.append('')

    # 表3: 图像长边与宽高比的均值/最小值/最大值
    markdown_line_list.append('## 3. 图像长边与宽高比的均值与取值范围')
    markdown_line_list.append('')
    markdown_line_list.append('长边均值是在长边直方图上按`sum(长边 * 条数) / 总条数`精确算出的全量算术'
                              '平均值,不是抽样估计。最小值/最大值是该数据集全部图像的实际取值边界,'
                              '可以用来解释表1、表2里某些档位为什么是0。')
    markdown_line_list.append('')
    markdown_line_list.append(
        '| 数据集 | 标注分片数 | 图像总数量 | 长边均值 | 长边最小值 | 长边最大值 | 宽高比最小值 | 宽高比最大值 |')
    markdown_line_list.append(
        '| --- | --- | --- | --- | --- | --- | --- | --- |')

    for per_stat_result_dict in all_stat_result_list:
        markdown_line_list.append(
            f'| {per_stat_result_dict["dataset_name"]} | '
            f'{per_stat_result_dict["annotation_json_count"]:,} | '
            f'{per_stat_result_dict["total_annotation_count"]:,} | '
            f'{per_stat_result_dict["image_long_side_mean"]} | '
            f'{per_stat_result_dict["image_long_side_min"]} | '
            f'{per_stat_result_dict["image_long_side_max"]} | '
            f'{per_stat_result_dict["aspect_ratio_min"]} | '
            f'{per_stat_result_dict["aspect_ratio_max"]} |')

    markdown_line_list.append('')

    # 表4: caption长度分位数
    markdown_line_list.append('## 4. caption描述长度分位数')

    markdown_line_list.append('')
    markdown_line_list.append(
        'caption长度取标注里的`t2i_caption_length`(字符数)。分位数按'
        'nearest-rank在长度直方图上精确求得(第`ceil(p/100 * N)`个值),不做插值近似。')
    markdown_line_list.append('')
    markdown_line_list.append(
        '| 数据集 | p1 | p5 | p10 | p50 | p90 | p95 | p99 | 最小值 | 最大值 | 均值 |')
    markdown_line_list.append(
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')

    for per_stat_result_dict in all_stat_result_list:
        per_percentile_dict = per_stat_result_dict[
            'caption_length_percentile_dict']
        per_percentile_str_list = [
            str(per_percentile_dict[f'p{per_percentile}'])
            for per_percentile in CAPTION_LENGTH_PERCENTILE_LIST
        ]
        markdown_line_list.append(
            f'| {per_stat_result_dict["dataset_name"]} | ' +
            ' | '.join(per_percentile_str_list) +
            f' | {per_stat_result_dict["caption_length_min"]} | '
            f'{per_stat_result_dict["caption_length_max"]} | '
            f'{per_stat_result_dict["caption_length_mean"]} |')

    markdown_line_list.append('')

    os.makedirs(os.path.dirname(save_stat_result_markdown_path), exist_ok=True)
    with open(save_stat_result_markdown_path, 'w',
              encoding='UTF-8') as save_markdown_file:

        save_markdown_file.write('\n'.join(markdown_line_list))

    print('6666', save_stat_result_markdown_path)

    return save_stat_result_markdown_path


def stat_all_dataset(root_dataset_path, save_stat_result_path,
                     save_stat_result_markdown_path):
    if not os.path.isdir(root_dataset_path):
        raise Exception(f'{root_dataset_path} is not a dir')

    # 统计结果一律写到数据集目录之外,保证本脚本不对数据集做任何修改
    for per_save_path in [
            save_stat_result_path,
            save_stat_result_markdown_path,
    ]:
        if os.path.abspath(per_save_path).startswith(
                os.path.abspath(root_dataset_path) + os.sep):
            raise Exception(
                f'{per_save_path} must be outside {root_dataset_path}')

    # 宽高比归桶用了二分加速,开跑前先确认它和T2IDataset里的get_closest_bucket
    # 同解,否则这份统计没法用来指导训练配置
    check_aspect_ratio_bucket_index_consistency()

    dataset_name_list = get_dataset_name_list(root_dataset_path)

    print('1111', 'total t2i dataset:', len(dataset_name_list),
          dataset_name_list)

    all_stat_result_list = []
    for per_dataset_name in dataset_name_list:
        per_dataset_stat_result_dict = stat_single_dataset(
            root_dataset_path, per_dataset_name, save_stat_result_path)
        all_stat_result_list.append(per_dataset_stat_result_dict)

        print('5555', per_dataset_name, 'total annotation:',
              per_dataset_stat_result_dict['total_annotation_count'],
              'aspect ratio range:',
              per_dataset_stat_result_dict['aspect_ratio_min'], '~',
              per_dataset_stat_result_dict['aspect_ratio_max'],
              'caption length percentile:',
              per_dataset_stat_result_dict['caption_length_percentile_dict'])

    # 全部数据集统计完之后再汇总成一个md表格
    save_stat_result_markdown(all_stat_result_list,
                              save_stat_result_markdown_path)

    return all_stat_result_list


if __name__ == '__main__':
    stat_all_dataset(ROOT_DATASET_PATH, SAVE_STAT_RESULT_PATH,
                     SAVE_STAT_RESULT_MARKDOWN_PATH)
