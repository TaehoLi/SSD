import numpy as np
import pathlib
import cv2
import json
from collections import OrderedDict
from pprint import pprint


class COCODataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """
        Dataset for COCO data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        
        if is_test:
            image_sets_file = self.root / "images/test.txt"
        else:
            image_sets_file = self.root / "images/train.txt"
            
        self.ids = COCODataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_names = ('BACKGROUND', 'person', 'bicycle', 'car', 'motorcycle',
                            'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                            'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                            'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                            'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush'
                           )
        
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"annotations/annotations/instances_val2014.json"
        
        with open(annotation_file, encoding="utf-8") as data_file:
            data = json.load(data_file, object_pairs_hook=OrderedDict)

        #pprint(data) #data는 json 전체를 dictionary 형태로 저장하고 있음
        
        boxes = []
        labels = []
        is_difficult = []
        image_number = int(image_id[-6:])
        
        for index, name in enumerate(data["images"]):
            if name["id"] == image_number:
                image_height = name["height"]
                image_width = name["width"]

        for index, name in enumerate(data["annotations"]):
            category_id = name["category_id"]
            bbox = name["bbox"]

            if name["image_id"] == image_number:
                for index, name in enumerate(data["categories"]):
                    
                    if name["id"] == category_id:
                        #print(name["name"])
                        #print(bbox)
                        
                        xmin = round(bbox[0])
                        if xmin == 0:
                            xmin += 1
                        ymin = round(bbox[1])
                        if ymin == 0:
                            ymin += 1
                        xmax = round(bbox[0] + bbox[2])
                        if xmax == image_width:
                            xmax -= 1
                        ymax = round(bbox[1] + bbox[3])
                        if ymax == image_height:
                            ymax -= 1

                        bbox = [xmin, ymin, xmax, ymax]
                        boxes.append(bbox)
                        
                        #xmin = bbox[0]
                        #ymin = bbox[1]
                        #xmax = bbox[0] + bbox[2]
                        #ymax = bbox[1] + bbox[3]
                        #bbox = [xmin, ymin, xmax, ymax]
                        #boxes.append(bbox)
                        
                        is_difficult.append(0)
                        labels.append(self.class_dict[name["name"]])

        a = np.array(boxes, dtype=np.float32)
        b = np.array(labels, dtype=np.int64)
        c = np.array(is_difficult, dtype=np.uint8)
        if a.size==0 or b.size==0 or c.size==0:
            a = np.array([[0,0,0,0]], dtype=np.float32)
            b = np.array([0], dtype=np.int64)
            c = np.array([0], dtype=np.uint8)
            return (a,b,c)
        else:
            pass
        return (a,b,c)

    def _read_image(self, image_id):
        image_file = self.root / f"images/val2014/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

