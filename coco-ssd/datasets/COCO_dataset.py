import numpy as np
import pathlib
import cv2
import json
from collections import OrderedDict
from pprint import pprint

class COCODataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=True):
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
            image_sets_file = self.root / "images/trainval.txt"
            
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
        

        for index, name in enumerate(data["annotations"]):
            category_id = name["category_id"]
            bbox = name["bbox"]

            if name["image_id"] == image_number:
                for index, name in enumerate(data["categories"]):
                    
                    if name["id"] == category_id:
                        #print(name["name"])
                        #print(bbox)
                        xmin = float(round(bbox[0])) - 1
                        ymin = float(round(bbox[1])) - 1
                        xmax = float(round(bbox[0] + bbox[2])) - 1
                        ymax = float(round(bbox[1] + bbox[3])) - 1
                        bbox = [xmin, ymin, xmax, ymax]

                        boxes.append(bbox)
                        is_difficult.append(0)

                        if category_id <= 11:
                            labels.append(category_id+1)
                        elif category_id <= 25:
                            labels.append(category_id-1+1)
                        elif category_id <= 28:
                            labels.append(category_id-2+1)
                        elif category_id <= 44:
                            labels.append(category_id-4+1)
                        elif category_id <= 65:
                            labels.append(category_id-5+1)
                        elif category_id == 67:
                            labels.append(61+1)
                        elif category_id <= 70:
                            labels.append(62+1)
                        elif category_id <= 82:
                            labels.append(category_id-9+1)
                        else:
                            labels.append(category_id-10+1)


        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"images/val2014/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



