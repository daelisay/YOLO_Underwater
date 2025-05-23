import torch
import xml.etree.ElementTree as ET
import os
import cv2
import copy
import numpy as np
from torchvision import transforms

from pycocotools.coco import COCO

from tqdm import tqdm

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class DUODataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
    "holothurian-seeweed",
    "echinus",
    "holothurian",
    "scallop",
    "starfish"
    )

    def __init__(self, root_dir, annotation_file, image_folder='image_folder',
                 image_size=416, split='train', use_augmentation=False,
                 box_type='yolo', cache=False, preprocessing=False):
        self.root = root_dir
        self.annotation_file = annotation_file
        self.image_folder = image_folder
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.box_type = box_type
        self.cache = cache
        self.preprocessing = preprocessing
        self.split = split

        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.img_ids = self.coco.getImgIds()

        # Map COCO category ids ke label 0..N
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats.sort(key=lambda x: x['id'])
        self.catid2label = {cat['id']: i for i, cat in enumerate(cats)}
        self.label2catid = {v: k for k, v in self.catid2label.items()}

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Augmentation pipeline
        self.transform = self.get_augmentation(split, use_augmentation)

        if self.cache:
            print(f"LOADING {split} dataset into memory...")
            self.cached_images = []
            self.cached_boxes = []
            self.cached_classes = []
            for idx in range(len(self.img_ids)):
                img, boxes, classes = self.load_image_label(idx)
                self.cached_images.append(img)
                self.cached_boxes.append(boxes)
                self.cached_classes.append(classes)
            print(f"INFO=====>DUO {split} init finished!")

    def __len__(self):
        return len(self.img_ids)

    def load_image_label(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root, self.image_folder, self.split, img_info['file_name'])
        img_path_norm = os.path.normpath(img_path)

        print(f"Trying to load image from: '{img_path}'")
        print(f"Normalized path: '{img_path_norm}'")
        print(f"Exists: {os.path.exists(img_path)}; IsFile: {os.path.isfile(img_path)}")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"File {img_path} does not exist or cannot be read!")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"File {img_path} cannot be read!")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        classes = []

        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            classes.append(self.catid2label[ann['category_id']])

        return img, boxes, classes


    def __getitem__(self, index):
        if self.cache:
            img = self.cached_images[index]
            boxes = self.cached_boxes[index]
            classes = self.cached_classes[index]
        else:
            img, boxes, classes = self.load_image_label(index)

        img_aug, boxes_aug = self.transform(
            image=img,
            bounding_boxes=BoundingBoxesOnImage([BoundingBox(*box) for box in boxes], shape=img.shape)
        )
        boxes_aug = boxes_aug.remove_out_of_image().clip_out_of_image()

        boxes = []
        for box in boxes_aug:
            boxes.append([box.x1, box.y1, box.x2, box.y2])

        im_h, im_w, _ = img_aug.shape
        boxes = self.box_type_convert(boxes, im_h, im_w, mode=self.box_type)

        targets = []
        for i, box in enumerate(boxes):
            targets.append([index, classes[i], *box])

        img_tensor = transforms.ToTensor()(img_aug)
        if not self.preprocessing:
            img_tensor = transforms.Normalize(self.mean, self.std)(img_tensor)

        targets = torch.from_numpy(np.array(targets, dtype=np.float32))
        index_tensor = torch.LongTensor([index])

        return img_tensor, targets, index_tensor

    def get_augmentation(self, split, augmentation):
        if split == 'train' and augmentation:
            seq = iaa.Sequential([
                iaa.PadToSquare(),
                iaa.Crop(percent=(0, 0.2)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=(-25, 25)),
                iaa.Resize(self.image_size),
                # Komentar ini dulu jika bermasalah
                # iaa.AddToHueAndSaturation((-60, 60))
            ])
        else:
            seq = iaa.Sequential([
                iaa.PadToSquare(position='center'),
                iaa.Resize(self.image_size),
            ])
        return seq

    def box_type_convert(self, boxes, height, width, mode='yolo'):
        if mode == 'xyxy':
            return boxes
        elif mode == 'coco':
            new_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                new_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            return new_boxes
        elif mode == 'yolo':
            new_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                tx = (xmin + xmax) / (2 * width)
                ty = (ymin + ymax) / (2 * height)
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                new_boxes.append([tx, ty, w, h])
            return new_boxes
        else:
            raise ValueError(f'Invalid box type: {mode}')


def collate_fn(batch):
    images, targets, indexes = list(zip(*batch))

    images = torch.stack([im for im in images], dim=0)
    indexes = torch.stack([id for id in indexes], dim=0)

    targets = [bboxes for bboxes in targets if bboxes is not None]
    for i, bboxes in enumerate(targets):
        if len(bboxes) == 0:
            continue
    targets = torch.cat(targets, 0)

    return images, targets, indexes


if __name__ == '__main__':
    # Test singkat
    dataset = DUODataset(
        root_dir='../data',
        annotation_file='../kaggle/working/YOLO_Underwater/data/train.json',
        image_folder='image_folder',
        split='train',
        use_augmentation=False)
    img, targets, idx = dataset[0]
    print(img.shape)
    print(targets)
    print(idx)

    import cv2
    img_show = (img.numpy() * 255).astype('uint8').transpose(1, 2, 0)
    cv2.imshow('test', img_show)
    cv2.waitKey(0)