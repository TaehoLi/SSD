import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

import os
import sys
#sys.path.append("..") # Adds higher directory to python modules path.
#print(os.getcwd())
os.chdir("utils")
import box_utils
os.chdir("..")
from collections import namedtuple, OrderedDict
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
                
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        
        if device:
            self.device0 = device
            self.device1 = torch.device("cuda:1")
            self.device2 = torch.device("cuda:2")
            self.device3 = torch.device("cuda:3")
        else:
            self.device0 = torch.device("cpu")
            
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device0)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                try:
                    y = added_layer(x.to(self.device0))
                except RuntimeError:
                    try:
                        y = added_layer(x.to(self.device1))
                    except RuntimeError:
                        try:
                            y = added_layer(x.to(self.device2))
                        except RuntimeError:
                            y = added_layer(x.to(self.device3))
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations
        
        
    def compute_header(self, i, x):
        #print(i)
        try:
            confidence = self.classification_headers[i](x.to(self.device0))
            confidence = confidence.permute(0, 2, 3, 1).contiguous()
            confidence = confidence.view(confidence.size(0), -1, self.num_classes)

            location = self.regression_headers[i](x.to(self.device0))
            location = location.permute(0, 2, 3, 1).contiguous()
            location = location.view(location.size(0), -1, 4)
        except RuntimeError:
            try:
                confidence = self.classification_headers[i](x.to(self.device1))
                confidence = confidence.permute(0, 2, 3, 1).contiguous()
                confidence = confidence.view(confidence.size(0), -1, self.num_classes)

                location = self.regression_headers[i](x.to(self.device1))
                location = location.permute(0, 2, 3, 1).contiguous()
                location = location.view(location.size(0), -1, 4)
            except RuntimeError:
                try:
                    confidence = self.classification_headers[i](x.to(self.device2))
                    confidence = confidence.permute(0, 2, 3, 1).contiguous()
                    confidence = confidence.view(confidence.size(0), -1, self.num_classes)

                    location = self.regression_headers[i](x.to(self.device2))
                    location = location.permute(0, 2, 3, 1).contiguous()
                    location = location.view(location.size(0), -1, 4)
                except RuntimeError:
                    confidence = self.classification_headers[i](x.to(self.device3))
                    confidence = confidence.permute(0, 2, 3, 1).contiguous()
                    confidence = confidence.view(confidence.size(0), -1, self.num_classes)

                    location = self.regression_headers[i](x.to(self.device3))
                    location = location.permute(0, 2, 3, 1).contiguous()
                    location = location.view(location.size(0), -1, 4)
        
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
    
    def load_dataparallel_model(self, model):
        # original saved file with DataParallel
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.load_state_dict(new_state_dict)
    
    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
