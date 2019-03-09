import os
import numpy as np
os.chdir("utils")
from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
os.chdir("..")

image_size = 1024
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(128, 1, SSDBoxSizes(0, 86), [2]),
    SSDSpec(64, 2, SSDBoxSizes(86, 172), [2, 3]),
    SSDSpec(32, 4, SSDBoxSizes(172, 258), [2, 3]),
    SSDSpec(16, 8, SSDBoxSizes(258, 344), [2, 3]),
    SSDSpec(10, 16, SSDBoxSizes(344, 430), [2, 3]),
    SSDSpec(9, 32, SSDBoxSizes(430, 516), [2]),
    SSDSpec(6, 64, SSDBoxSizes(516, 602), [2, 3]),
    SSDSpec(5, 128, SSDBoxSizes(602, 688), [2]),
    SSDSpec(4, 256, SSDBoxSizes(688, 774), [2]),
    SSDSpec(3, 512, SSDBoxSizes(774, 860), [2]),
    SSDSpec(2, 1024, SSDBoxSizes(860, 946), [2]),
    SSDSpec(1, 2048, SSDBoxSizes(946, 1032), [2])
]

priors = generate_ssd_priors(specs, image_size)

