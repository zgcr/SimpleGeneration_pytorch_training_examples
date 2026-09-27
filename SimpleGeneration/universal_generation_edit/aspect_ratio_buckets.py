import math
'''
Aspect-ratio bucket table for the T2I / TI2I task.

================================================================================
1. THE BUCKET TABLE
================================================================================

41 buckets of pure width/height ratios, ordered by INCREASING ratio and covering

    0.25 (1:4, index 0)  <=  W / H  <=  4.0 (4:1, index 40)

with the square 1:1 at index 20. The table is its own transpose mirror
(ASPECT_RATIO_BUCKETS[i] * ASPECT_RATIO_BUCKETS[40 - i] == 1 exactly), so the 20
tall buckets and the 20 wide buckets are symmetric. Storing ratios instead of
resolutions keeps a cached bucket_index valid across every training stage: going
256 -> 512 -> 1024 only changes base_resize.

================================================================================
2. WHY EVERY BUCKET SIDE IS A MULTIPLE OF 16
================================================================================

The latents pass through two sub-models whose down-sampling factors multiply:

    AE encode, planes_mult=[1, 2, 4, 4] -> 3 transitions of 2x          -> 8
    flux2 AE folds a 2x2 patchify into its channels (128 = 32 * 2 * 2)  -> 2

So the denoise DiT sees a token grid of (H / 16, W / 16), and its img_in is a
plain per-token Linear with no further patchify inside the DiT. Both H and W
must therefore be multiples of 16 for that grid to be an integer; one token
corresponds to exactly 16 pixels.

================================================================================
3. RATIO + base_resize -> RESOLUTION (get_bucket_resolution)
================================================================================

A stage's token budget is the area A = base_resize^2 (A / 16^2 tokens). For a
bucket ratio AR the ideal sides are H = sqrt(A / AR) and W = sqrt(A * AR); each
side is then snapped DOWN to a multiple of 16, so H * W <= A and the token count
never exceeds the budget. Examples at base_resize=512 (A = 262144, <= 1024
tokens), written as [W, H]:

    AR = 0.25    -> [256, 1024]     AR = 240/256 -> [480, 528]
    AR = 1.0     -> [512, 512]      AR = 448/144 -> [896, 288]

The four ASPECT_RATIO_BUCKETS_* tables below are exactly this function applied
to the ratio table at base_resize 256 / 512 / 1024 / 2048; __main__ re-derives
them and asserts they never drift. Note the two different orders in play:
get_bucket_resolution RETURNS (height, width), because that is the order the
resize transforms want, while the four tables STORE [width, height], so that
their first column grows together with the bucket ratio (index 0 is the
narrowest width, index 40 the widest) and a mis-sorted table is visible at a
glance. The 16 grid is coarse at 256, so that stage maps its 41 buckets onto 35
distinct resolutions - harmless, because the batch samplers group by
bucket_index and the resolution inside one bucket is unique.

================================================================================
4. HOW AN IMAGE IS ALIGNED TO ITS BUCKET RESOLUTION (t2i/ti2i_common.py)
================================================================================

Picking the closest ratio and snapping to 16 are both quantisations, so the
landed ratio W_b / H_b usually differs slightly from the image's own ratio, while
the collators need every sample of a bucket to be EXACTLY (H_b, W_b) to stack a
[B, 3, H, W] tensor. TorchAspectRatioBucketResize absorbs that residual with a
plain ANISOTROPIC resize: no crop (cropping would remove different regions from
the edited image and its reference images and destroy their pixel alignment) and
no padding (black bars would be encoded into the latent). The distortion is
small and, since one bucket collects many original ratios, random and unbiased:
0% at 1:1, 1.59% at 16:9, 3.70% at 4:3, ~2% median over the whole range.

Reference images (TI2I) are resized one by one:

  * original ratio equal to the edited image's -> resized to the same
    (H_b, W_b), hence pixel aligned with the noise-target grid (identical h/w
    RoPE coordinates, only the temporal coordinate differs);
  * ratio not equal -> its own ratio is kept, the long side is scaled to
    max(H_b, W_b) and each side is snapped to a multiple of 16.

Ratios outside the table would be CLAMPED onto an end bucket and then squeezed.
'''

# ------------------------------------------------------------------------------
# The bucket table: pure width/height ratios, resolution independent, 41 entries.
# Written as fractions of the 16-aligned side lengths they come from, so that the
# transpose mirror ASPECT_RATIO_BUCKETS[i] * ASPECT_RATIO_BUCKETS[40 - i] == 1
# holds exactly. Tallest (1:4) -> square (1:1, index 20) -> widest (4:1).
# ------------------------------------------------------------------------------
ASPECT_RATIO_BUCKETS = [
    128 / 512, 128 / 496, 128 / 480, 128 / 464, 144 / 448, 144 / 432,
    144 / 416, 160 / 400, 160 / 384, 176 / 368, 176 / 352, 176 / 336,
    192 / 336, 192 / 320, 208 / 304, 208 / 288, 224 / 288, 224 / 272,
    240 / 272, 240 / 256, 256 / 256, 256 / 240, 272 / 240, 272 / 224,
    288 / 224, 288 / 208, 304 / 208, 320 / 192, 336 / 192, 336 / 176,
    352 / 176, 368 / 176, 384 / 160, 400 / 160, 416 / 144, 432 / 144,
    448 / 144, 464 / 128, 480 / 128, 496 / 128, 512 / 128
]


def get_closest_bucket(width, height, buckets):
    aspect_ratio = width / height

    best_index = 0
    best_diff = None
    for index, bucket_ratio in enumerate(buckets):
        diff = abs(bucket_ratio - aspect_ratio)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_index = index

    return buckets[best_index], best_index


def get_bucket_resolution(bucket_ratio, base_resize):
    bucket_height = int(
        math.sqrt(base_resize * base_resize / bucket_ratio) // 16) * 16
    bucket_width = int(
        math.sqrt(base_resize * base_resize * bucket_ratio) // 16) * 16
    bucket_height, bucket_width = max(16, bucket_height), max(16, bucket_width)

    return bucket_height, bucket_width


# ------------------------------------------------------------------------------
# 256 stage (base_resize=256, area <= 65536, denoise tokens <= 256).
# Every entry is [bucket_width, bucket_height], WIDTH FIRST, so the table reads
# in the same direction as ASPECT_RATIO_BUCKETS: index 0 tallest [128, 512]
# (1:4) -> index 20 square [256, 256] -> index 40 widest [512, 128] (4:1).
# Every side is a multiple of 16. Note the 16 grid is coarse at this scale, so a
# few neighbouring ratios share one resolution (41 buckets -> 35 distinct sizes).
# ------------------------------------------------------------------------------
ASPECT_RATIO_BUCKETS_256 = [[128, 512], [128, 496], [128, 480], [128, 480],
                            [144, 448], [144, 432], [144, 432], [160, 400],
                            [160, 384], [176, 368], [176, 352], [176, 352],
                            [192, 336], [192, 320], [208, 304], [208, 288],
                            [224, 288], [224, 272], [240, 272], [240, 256],
                            [256, 256], [256, 240], [272, 240], [272, 224],
                            [288, 224], [288, 208], [304, 208], [320, 192],
                            [336, 192], [352, 176], [352, 176], [368, 176],
                            [384, 160], [400, 160], [432, 144], [432, 144],
                            [448, 144], [480, 128], [480, 128], [496, 128],
                            [512, 128]]

# ------------------------------------------------------------------------------
# 512 stage (base_resize=512, area <= 262144, denoise tokens <= 1024).
# Every entry is [bucket_width, bucket_height], WIDTH FIRST: index 0 tallest
# [256, 1024] (1:4) -> index 20 square [512, 512] -> index 40 widest
# [1024, 256] (4:1). 41 buckets, 41 distinct resolutions, every side a multiple
# of 16.
# ------------------------------------------------------------------------------
ASPECT_RATIO_BUCKETS_512 = [[256, 1024], [256, 992], [256, 976], [256, 960],
                            [288, 896], [288, 880], [288, 864], [320, 800],
                            [320, 784], [352, 736], [352, 720], [368, 704],
                            [384, 672], [384, 656], [416, 608], [432, 592],
                            [448, 576], [464, 560], [480, 544], [480, 528],
                            [512, 512], [528, 480], [544, 480], [560, 464],
                            [576, 448], [592, 432], [608, 416], [656, 384],
                            [672, 384], [704, 368], [720, 352], [736, 352],
                            [784, 320], [800, 320], [864, 288], [880, 288],
                            [896, 288], [960, 256], [976, 256], [992, 256],
                            [1024, 256]]

# ------------------------------------------------------------------------------
# 1024 stage (base_resize=1024, area <= 1048576, denoise tokens <= 4096).
# Every entry is [bucket_width, bucket_height], WIDTH FIRST: index 0 tallest
# [512, 2048] (1:4) -> index 20 square [1024, 1024] -> index 40 widest
# [2048, 512] (4:1). 41 buckets, 41 distinct resolutions, every side a multiple
# of 16.
# ------------------------------------------------------------------------------
ASPECT_RATIO_BUCKETS_1024 = [[512, 2048], [512, 2000], [528,
                                                        1968], [528, 1936],
                             [576, 1792], [576, 1760], [592,
                                                        1728], [640, 1616],
                             [656, 1584], [704, 1472], [720,
                                                        1440], [736, 1408],
                             [768, 1344], [784, 1312], [832,
                                                        1232], [864, 1200],
                             [896, 1152], [928, 1120], [960,
                                                        1088], [976, 1056],
                             [1024, 1024], [1056, 976], [1088,
                                                         960], [1120, 928],
                             [1152, 896], [1200, 864], [1232,
                                                        832], [1312, 784],
                             [1344, 768], [1408, 736], [1440,
                                                        720], [1472, 704],
                             [1584, 656], [1616, 640], [1728,
                                                        592], [1760, 576],
                             [1792, 576], [1936, 528], [1968, 528],
                             [2000, 512], [2048, 512]]

# ------------------------------------------------------------------------------
# 2048 stage (base_resize=2048, area <= 4194304, denoise tokens <= 16384).
# Every entry is [bucket_width, bucket_height], WIDTH FIRST: index 0 tallest
# [1024, 4096] (1:4) -> index 20 square [2048, 2048] -> index 40 widest
# [4096, 1024] (4:1). 41 buckets, 41 distinct resolutions, every side a multiple
# of 16.
# ------------------------------------------------------------------------------
ASPECT_RATIO_BUCKETS_2048 = [[1024, 4096], [1040, 4016], [1056, 3952],
                             [1072, 3888], [1152, 3600], [1168, 3536],
                             [1200, 3472], [1280, 3232], [1312, 3168],
                             [1408, 2960], [1440, 2896], [1472, 2816],
                             [1536, 2704], [1584, 2640], [1680, 2464],
                             [1728, 2400], [1792, 2320], [1856, 2256],
                             [1920, 2176], [1968, 2112], [2048, 2048],
                             [2112, 1968], [2176, 1920], [2256, 1856],
                             [2320, 1792], [2400, 1728], [2464, 1680],
                             [2640, 1584], [2704, 1536], [2816, 1472],
                             [2896, 1440], [2960, 1408], [3168, 1312],
                             [3232, 1280], [3472, 1200], [3536, 1168],
                             [3600, 1152], [3888, 1072], [3952, 1056],
                             [4016, 1040], [4096, 1024]]

if __name__ == '__main__':
    aspect_ratio_buckets_list = [
        (256, ASPECT_RATIO_BUCKETS_256),
        (512, ASPECT_RATIO_BUCKETS_512),
        (1024, ASPECT_RATIO_BUCKETS_1024),
        (2048, ASPECT_RATIO_BUCKETS_2048),
    ]

    for per_base_resize, per_base_resize_buckets in aspect_ratio_buckets_list:
        assert len(per_base_resize_buckets) == len(ASPECT_RATIO_BUCKETS)

        for per_bucket_index, per_bucket_ratio in enumerate(
                ASPECT_RATIO_BUCKETS):
            per_bucket_h, per_bucket_w = get_bucket_resolution(
                per_bucket_ratio, per_base_resize)

            assert [per_bucket_w,
                    per_bucket_h] == per_base_resize_buckets[per_bucket_index]
            assert per_bucket_h % 16 == 0 and per_bucket_w % 16 == 0
            assert per_bucket_h * per_bucket_w <= per_base_resize * per_base_resize

        for per_bucket_index in range(len(per_base_resize_buckets) - 1):
            per_bucket_w, per_bucket_h = per_base_resize_buckets[
                per_bucket_index]
            per_next_bucket_w, per_next_bucket_h = per_base_resize_buckets[
                per_bucket_index + 1]

            assert per_bucket_w * per_next_bucket_h <= per_next_bucket_w * per_bucket_h
        print(
            f'base_resize:{per_base_resize}, bucket nums:{len(per_base_resize_buckets)}, unique resolution nums: {len(set([tuple(x) for x in per_base_resize_buckets]))}, resolution range:{per_base_resize_buckets[0]}~{per_base_resize_buckets[-1]} matched'
        )

    for per_bucket_index in range(len(ASPECT_RATIO_BUCKETS) - 1):
        assert ASPECT_RATIO_BUCKETS[per_bucket_index] < ASPECT_RATIO_BUCKETS[
            per_bucket_index + 1]

    for per_bucket_index, per_bucket_ratio in enumerate(ASPECT_RATIO_BUCKETS):
        per_mirror_bucket_ratio = ASPECT_RATIO_BUCKETS[
            len(ASPECT_RATIO_BUCKETS) - 1 - per_bucket_index]
        assert abs(per_bucket_ratio * per_mirror_bucket_ratio - 1.0) < 1e-12

    print(
        f'ASPECT_RATIO_BUCKETS nums:{len(ASPECT_RATIO_BUCKETS)}, ratio range:{ASPECT_RATIO_BUCKETS[0]}~{ASPECT_RATIO_BUCKETS[-1]}, ascending order and transpose mirror symmetry matched'
    )
