import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d

import os
import sys
#sys.path.append(".") # Adds higher directory to python modules path.
#print(os.getcwd())

from nn.vgg import vgg

from ssd.ssd import SSD
from ssd.predictor import Predictor
#from ssd.config import vgg_ssd_config as config


def create_vgg_ssd(num_classes, config_name, device=None, is_test=False):
    vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                  512, 512, 512]
    base_net = ModuleList(vgg(vgg_config))

    source_layer_indexes = [
        (23, BatchNorm2d(512)),
        len(base_net),
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])
    
    if config_name == "vgg_ssd_config_300":
        from ssd.config import vgg_ssd_config_300 as config
    elif config_name == "vgg_ssd_config_512":
        from ssd.config import vgg_ssd_config_512 as config
    elif config_name == "vgg_ssd_config_600":
        from ssd.config import vgg_ssd_config_600 as config
    elif config_name == "vgg_ssd_config_1024":
        from ssd.config import vgg_ssd_config_1024 as config
    else:
        return print("There is no config file.")
    
    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers,
               is_test=is_test, config=config, device=device)


def create_vgg_ssd_predictor(net, config_name, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    
    if config_name == "vgg_ssd_config_300":
        from ssd.config import vgg_ssd_config_300 as config
    elif config_name == "vgg_ssd_config_512":
        from ssd.config import vgg_ssd_config_512 as config
    elif config_name == "vgg_ssd_config_600":
        from ssd.config import vgg_ssd_config_600 as config
    elif config_name == "vgg_ssd_config_1024":
        from ssd.config import vgg_ssd_config_1024 as config
    else:
        return print("There is no config file.")
    
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
