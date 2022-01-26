from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class A2D2Dataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 52
        self.palette = palette.A2D2_palette
        super(A2D2Dataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["train", "val"]:
            self.image_dir = os.path.join(self.root, 'camera_lidar_semantic', 'images', self.split)
            self.label_dir = os.path.join(self.root, 'camera_lidar_semantic', 'labels', self.split)
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.png')]
        else: raise ValueError(f"Invalid split name {self.split}")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, image_id.replace('camera', 'label') + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int64)
        #label_dict = {[255,0,0],[200,0,0],[150,0,0],[128,0,0],[182,89,6],[150,50,4],[90,30,1],[90,30,30],
        #                [204,153,155],[189,73,155],[239,89,191],[155,128,0],[200,128,0],[150,128,0],[0,255,0],
        #                [0,200,0],[0,150,0],[0,128,255],[30,28,258],[60,28,100],[0,255,255],[30,220,220],[60,157,199],
        #                [255,255,0],[255,255,200],[233,100,0],[110,110,0],[128,128,0],[155,193,37],[64,0,64],
        #                [185,122,87],[0,0,100],[139,99,108],[210,50,115],[255,0,128],[255,246,143],[150,0,150],
        #                [204,255,153],[238,162,173],[33,44,177],[180,50,180],[255,70,185],[238,233,191],
        #                [147,253,194],[150,150,200],[180,150,200],[72,209,204],[200,125,210],[159,121,238],
        #                [128,0,255],[255,0,255],[135,206,255],[241,230,255],[96,69,143],[53,46,82]}
        print(label[0])
        return image, label, image_id

class A2D2(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=8, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.5,0.5, 0.5]
        self.STD = [0.5, 0.5, 0.5]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = A2D2Dataset(**kwargs)
        super(A2D2, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
