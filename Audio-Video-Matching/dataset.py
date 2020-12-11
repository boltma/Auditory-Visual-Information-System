import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import wave
import pickle
labels = {
    '061_foam_brick':0,
    'green_basketball':1,
    'salt_cylinder':2,
    'shiny_toy_gun':3,
    'stanley_screwdriver':4,
    'strawberry':5,
    'toothpaste_box':6,
    'toy_elephant':7,
    'whiteboard_spray':8,
    'yellow_block':9,
}
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, mode = "train", transforms = None):
        self.num_class = 10
        self.imgs = []
        self.img_class = []
        self.mode = mode
        self.transforms = transforms
        if self.mode == "train":
            self.classes = os.listdir(os.path.join("dataset/train"))
            for img_classes in self.classes:
                img_dir = os.listdir(os.path.join("dataset/train", img_classes))
                self.imgs += list(map(lambda x: img_classes + "/" + x, img_dir))
                c = [img_classes] * len(img_dir)
                self.img_class += c
        elif self.mode == "test":
            self.imgs += os.listdir(os.path.join("data/Classification/Data/Test"))

        
    def __getitem__(self, idx):
        if self.mode == "train":
            img_path = os.path.join("dataset/train/", self.imgs[idx])
        elif self.mode == "test":
            img_path = os.path.join("data/Classification/Data/Test/", self.imgs[idx])
        pth = img_path + "/audio_data.pkl"
        #print(pth)
        #img = torch.load(pth)
        img = pickle.load(open(pth, 'rb'))["audio"]
        
        #print(img.shape)
        #img=cPickle.load(open(img_path + "/audio_data.pkl"))
        if self.transforms is not None:
            img = self.transforms(img)
        if self.mode == "train":
            sample = {'image': img, 'label': labels[self.img_class[idx]]}
        elif self.mode == "test":
            sample = {'image': img}
        #print(sample)
        return sample
    def __len__(self):
        return len(self.imgs)