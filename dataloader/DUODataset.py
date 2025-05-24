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
        "holothurian-seeweed", "echinus", "holothurian", "scallop", "starfish"
    )

    def __init__(self, root_dir, annotation_file, image_folder='/kaggle/working/YOLO_Underwater/data/image_folder',
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

        self.coco = COCO(annotation_file)
        self.img_ids = self.coco.getImgIds()

        cats = self.coco.loadCats(self.coco.getCatIds())
        cats.sort(key=lambda x: x['id'])
        self.catid2label = {cat['id']: i for i, cat in enumerate(cats)}
        self.label2catid = {v: k for k, v in self.catid2label.items()}

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.transform = self.get_augmentation(split, use_augmentation)

    def __len__(self):
        return len(self.img_ids)

    def load_image_label(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, self.image_folder, self.split, img_info['file_name'])
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, classes = [], []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            classes.append(self.catid2label[ann['category_id']])

        return img, boxes, classes

    def __getitem__(self, index):
        try:
            img, boxes, classes = self.load_image_label(index)
            if len(boxes) == 0:
                raise ValueError("No bounding boxes")

            bbs = BoundingBoxesOnImage([BoundingBox(*box) for box in boxes], shape=img.shape)
            img_aug, boxes_aug = self.transform(image=img, bounding_boxes=bbs)
            boxes_aug = boxes_aug.remove_out_of_image().clip_out_of_image()

            if len(boxes_aug) == 0:
                raise ValueError("All boxes removed by augmentation")

            boxes = [[b.x1, b.y1, b.x2, b.y2] for b in boxes_aug]
            h, w, _ = img_aug.shape
            boxes = self.box_type_convert(boxes, h, w, self.box_type)

            targets = [[index, classes[i], *box] for i, box in enumerate(boxes)]
            img_tensor = transforms.ToTensor()(img_aug)
            if not self.preprocessing:
                img_tensor = transforms.Normalize(self.mean, self.std)(img_tensor)

            return img_tensor, torch.tensor(targets, dtype=torch.float32), torch.tensor(index)

        except Exception as e:
            print(f"Warning: Skipped index {index} due to: {e}")
            return None  # biarkan collate_fn yang skip None

    def get_augmentation(self, split, augmentation):
        if split == 'train' and augmentation:
            return iaa.Sequential([
                iaa.PadToSquare(), iaa.Crop(percent=(0, 0.2)),
                iaa.Fliplr(0.5), iaa.Flipud(0.5),
                iaa.Affine(rotate=(-25, 25)), iaa.Resize(self.image_size)
            ])
        else:
            return iaa.Sequential([
                iaa.PadToSquare(position='center'), iaa.Resize(self.image_size)
            ])

    def box_type_convert(self, boxes, height, width, mode='yolo'):
        if mode == 'xyxy':
            return boxes
        elif mode == 'coco':
            return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        elif mode == 'yolo':
            return [  # [cx, cy, w, h] normalized
                [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                 (x2 - x1) / width, (y2 - y1) / height]
                for x1, y1, x2, y2 in boxes
            ]
        else:
            raise ValueError(f"Invalid box type: {mode}")


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None

    images, targets, indexes = zip(*batch)
    return (
        torch.stack(images),
        torch.cat(targets, dim=0),
        torch.stack(indexes)
    )


# if __name__ == '__main__':
#     # Test singkat
#     dataset = DUODataset(
#         root_dir='../data',
#         annotation_file='../kaggle/working/YOLO_Underwater/data/train.json',
#         image_folder='/kaggle/working/YOLO_Underwater/data/image_folder',
#         split='train',
#         use_augmentation=False)
#     img, targets, idx = dataset[0]
#     print(img.shape)
#     print(targets)
#     print(idx)

#     import cv2
#     img_show = (img.numpy() * 255).astype('uint8').transpose(1, 2, 0)
#     cv2.imshow('test', img_show)
#     cv2.waitKey(0)