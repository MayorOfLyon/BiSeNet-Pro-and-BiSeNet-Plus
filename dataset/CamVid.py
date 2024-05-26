import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random

class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, scale, loss='dice', mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
 
        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(csv_path)
        
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================

        img = np.array(img)
        # load label
        label = Image.open(self.label_list[index])

        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        label = np.array(label)

        # augment image and label
        if self.mode == 'train':
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            label = seq_det.augment_image(label)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
    data = CamVid(['/data/sqy/CamVid/train', '/data/sqy/CamVid/val'],
                  ['/data/sqy/CamVid/train_labels', '/data/sqy/CamVid/val_labels'], '/data/sqy/CamVid/class_dict.csv',
                  (720, 960), loss='crossentropy', mode='val')
    from model.build_BiSeNet import BiSeNet
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    label_info = get_label_info('/data/sqy/CamVid/class_dict.csv')
    for i, (img, label) in enumerate(data):
        print(label.size())
        print(torch.max(label))

