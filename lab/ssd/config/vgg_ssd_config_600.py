import os
import numpy as np
os.chdir("utils")
from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
os.chdir("..")

image_size = 600
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(76, 8, SSDBoxSizes(30, 130), [2]),
    SSDSpec(35, 16, SSDBoxSizes(130, 230), [2, 3]),
    SSDSpec(21, 32, SSDBoxSizes(230, 330), [2, 3]),
    SSDSpec(10, 64, SSDBoxSizes(330, 430), [2, 3]),
    SSDSpec(6, 100, SSDBoxSizes(430, 530), [2]),
    SSDSpec(3, 200, SSDBoxSizes(530, 630), [2])
]

priors = generate_ssd_priors(specs, image_size)
