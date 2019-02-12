import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2


class NEUDataset:

    def __init__(self, root, transform=None, target_transform=None, is_validate=False, keep_difficult=True):
        """Dataset for NEU data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        
        if is_validate:
            image_sets_file = self.root / "val+noise.txt"
        else:
            #if is_validate:
            #    image_sets_file = self.root / "ImageSets/Main/val.txt"
            #else:
            #    image_sets_file = self.root / "ImageSets/Main/train.txt"
            image_sets_file = self.root / "train+noise.txt"
        
        self.ids = NEUDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_names = ('BACKGROUND', 'crazing',
            'inclusion', 'patches', 'pitted_surface',
            'rolled-in_scale', 'scratches'
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
        
        try:
            annotation_file = self.root / f"ANNOTATIONS/{image_id}.xml"
            objects = ET.parse(annotation_file).findall("object")
            boxes = []
            labels = []
            is_difficult = []

            for object in objects:
                class_name = object.find('name').text.lower().strip()
                bbox = object.find('bndbox')

                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)

                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

            a = np.array(boxes, dtype=np.float32)
            b = np.array(labels, dtype=np.int64)
            c = np.array(is_difficult, dtype=np.uint8)
            if a.size==0 or b.size==0:
                a = np.array([[0,0,0,0]], dtype=np.float32)
                b = np.array([0], dtype=np.int64)
                c = np.array([0], dtype=np.uint8)
                return (a,b,c)
            else:
                pass
            return (a,b,c)
        except FileNotFoundError:
            a = np.array([[0,0,0,0]], dtype=np.float32)
            b = np.array([0], dtype=np.int64)
            c = np.array([0], dtype=np.uint8)
            return (a,b,c)

    def _read_image(self, image_id):
        try:
            image_file = self.root / f"IMAGES_COLOR/{image_id}.jpg"
            image = cv2.imread(str(image_file), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image_file = self.root / f"IMAGES_distorted/{image_id}.jpg"
            image = cv2.imread(str(image_file), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

