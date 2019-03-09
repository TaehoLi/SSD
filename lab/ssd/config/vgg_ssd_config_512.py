import os
import numpy as np
os.chdir("utils")
from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors
os.chdir("..")

image_size = 512
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(64, 4, SSDBoxSizes(0, 60), [2]),
    SSDSpec(32, 8, SSDBoxSizes(60, 120), [2, 3]),
    SSDSpec(16, 16, SSDBoxSizes(120, 180), [2, 3]),
    SSDSpec(8, 32, SSDBoxSizes(180, 240), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(240, 300), [2]),
    SSDSpec(4, 128, SSDBoxSizes(300, 360), [2, 3]),
    SSDSpec(1, 256, SSDBoxSizes(360, 420), [2]),
    SSDSpec(1, 512, SSDBoxSizes(420, 480), [2]),
    SSDSpec(1, 1024, SSDBoxSizes(480, 540), [2])
]

priors = generate_ssd_priors(specs, image_size)
