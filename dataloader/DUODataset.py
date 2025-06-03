import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from pycocotools.coco import COCO
import copy

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class DUODataset(torch.utils.data.Dataset):
    CLASSES_NAME = ("echinus", "holothurian", "scallop", "starfish")

    def __init__(self, root_dir, annotation_file, split='train',
                 image_folder='image_folder', image_size=416,
                 use_augmentation=False, box_type='yolo', cache=False, preprocessing=False):

        self.root = root_dir
        self.annotation_file = annotation_file
        self.split = split
        self.image_folder = image_folder
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.box_type = box_type
        self.cache = cache
        self.preprocessing = preprocessing

        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.img_ids = self.coco.getImgIds()

        # Mapping category IDs to label indices
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats.sort(key=lambda x: x['id'])
        self.catid2label = {cat['id']: i for i, cat in enumerate(cats)}
        self.label2catid = {v: k for k, v in self.catid2label.items()}

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.transform = self.get_augmentation(self.split, self.use_augmentation)

        if self.cache:
            print(f'LOADING {self.split} dataset...')
            self.cached_images = []
            self.cached_boxes = []
            self.cached_classes = []
            for idx in range(len(self.img_ids)):
                img, boxes, classes = self.load_image_label(idx)
                self.cached_images.append(img)
                self.cached_boxes.append(boxes)
                self.cached_classes.append(classes)
            print(f"CACHE for {self.split} dataset loaded.")

    def __len__(self):
        return len(self.img_ids)

    def _read_img_rgb(self, path):
        img = cv2.imread(path)
        assert img is not None, f"File not found: {path}"
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_image_label(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, self.image_folder, self.split, img_info['file_name'])
        img = self._read_img_rgb(img_path)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        ia_boxes = []
        classes = []

        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = x1 + bbox[2], y1 + bbox[3]
            ia_box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            ia_boxes.append(ia_box)

            cat_id = ann['category_id']
            classes.append(self.catid2label[cat_id])

        ia_boxes = BoundingBoxesOnImage(ia_boxes, shape=img.shape)
        return img, ia_boxes, classes

    def __getitem__(self, index):
        if self.cache:
            img = copy.deepcopy(self.cached_images[index])
            ia_boxes = copy.deepcopy(self.cached_boxes[index])
            classes = copy.deepcopy(self.cached_classes[index])
        else:
            img, ia_boxes, classes = self.load_image_label(index)

        # Skip if no boxes
        if len(ia_boxes) == 0:
            return None

        # Apply augmentation
        img_aug, ia_boxes_aug = self.transform(image=img, bounding_boxes=ia_boxes)
        ia_boxes_aug = ia_boxes_aug.remove_out_of_image().clip_out_of_image()

        if len(ia_boxes_aug) == 0:
            return None

        boxes = [[b.x1, b.y1, b.x2, b.y2] for b in ia_boxes_aug]

        h, w, _ = img_aug.shape
        boxes_converted = self.box_type_convert(boxes, h, w, self.box_type)

        targets = []
        for i, box in enumerate(boxes_converted):
            targets.append([index, classes[i], *box])

        img_tensor = transforms.ToTensor()(img_aug)
        if not self.preprocessing:
            img_tensor = transforms.Normalize(self.mean, self.std)(img_tensor)

        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        index_tensor = torch.tensor([index])

        return img_tensor, targets_tensor, index_tensor

    def get_augmentation(self, split, use_augmentation):
        if split == 'train' and use_augmentation:
            return iaa.Sequential([
                iaa.PadToSquare(),
                iaa.Crop(percent=(0, 0.2)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.AddToHueAndSaturation((-30, 30)),
                iaa.MotionBlur(k=3),
                iaa.Multiply((0.8, 1.2)),
                iaa.Sharpen(alpha=(0.0, 0.3)),
                iaa.Affine(rotate=(-25, 25)),
                iaa.Resize(self.image_size)
            ])
        else:
            return iaa.Sequential([
                iaa.PadToSquare(position='center'),
                iaa.Resize(self.image_size)
            ])

    def box_type_convert(self, boxes, height, width, mode='yolo'):
        if mode == 'xyxy':
            return boxes
        elif mode == 'coco':
            return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        elif mode == 'yolo':
            return [
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
    return torch.stack(images), torch.cat(targets, 0), torch.stack(indexes)

if __name__ == '__main__':
    import os
    # Gunakan default dataset_path sesuai opts.py, sesuaikan jika kamu pakai environment lain
    root_dir = '/kaggle/working/YOLO_Underwater/data'  
    annotation_file = os.path.join(root_dir, 'train_vixed.json')  # Pastikan nama file anotasi benar

    dataset = DUODataset(root_dir=root_dir, annotation_file=annotation_file,
                         split='train', use_augmentation=False)
    img_tensor, targets, index = dataset[0]

    print("Image tensor shape:", img_tensor.shape)
    print("Targets:", targets)
    print("Index:", index)

    img_np = (img_tensor.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    import cv2
    cv2.imshow('test image', img_np)
    cv2.waitKey(0)
