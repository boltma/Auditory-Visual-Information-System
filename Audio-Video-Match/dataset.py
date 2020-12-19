import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import wave
import pickle
from spectrogram import spectrogram
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
        
            pth = img_path + "/audio_data.pkl"
            #print(pth)
            #img = torch.load(pth)
            # TODO

            #a =pickle.load(open('dataset/train/salt_cylinder/0/audio_data.pkl', 'rb'))
            #print(a) 
            # sig = pickle.load(open(pth, 'rb'))["audio"]
            # img = spectrogram(sig)
            # np.save(img_path + "/spec.npy", img)
            img = np.load(img_path + "/spec.npy")
            #print(img)
            img = Image.fromarray(img)
        elif self.mode == "test":
            img_path = os.path.join("dataset/task1/test/", self.imgs[idx])
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
class matching_dataset(torch.utils.data.Dataset):
    def __init__(self, cat = 'yellow_block', mode = "train", transforms = None):
        #self.num_class = 10
        #self.classname = labels[cat]
        self.classname = cat
        self.imgs = []
        self.img_class = []
        self.mode = mode
        self.transforms = transforms
        if self.mode == "train":
            #self.classes = os.listdir(os.path.join("dataset/train"))
            img_dir = os.listdir(os.path.join("dataset/train" , self.classname))
            self.imgs += list(map(lambda x: self.classname + "/" + x, img_dir))
            c = [self.classname] * len(img_dir)
            #self.img_class += c
        elif self.mode == "test":
            self.imgs += os.listdir(os.path.join("dataset/task2/test/"))
        self.imglen = len(self.imgs)
        #a = [lists for lists in os.listdir("dataset/train/") if os.path.isfile(os.path.join("dataset/train", lists))]
        #print(a)
    def __getitem__(self, idx):
        if self.mode == "train":
            idx1 = idx // self.imglen 
            idx2 = idx % self.imglen
            imgpath = os.path.join("dataset/train/", self.imgs[idx1])
            img_pth = imgpath + "/audio_data.pkl"

            img = np.load(imgpath + "/spec.npy")
            img = Image.fromarray(img)

            #img = img.swapaxes(0, 2)
            video_pth = os.path.join("dataset/train/", self.imgs[idx2], 'mask')

            video = np.array([])
            for a in os.listdir(video_pth):
                image = Image.open(os.path.join(video_pth, a))
                
                image = np.array(image)
                if video.shape == (0,):
                    video = image
                else:
                    video = np.dstack((video, image))
            i = 0
            while video.shape[2] < 30:
                video = np.dstack((video, video[:,:,i]))
                i = i + 1
            video = np.reshape(video, (1, 30, 440, 440))
            #print(video.shape)
        if self.transforms is not None:
            img = self.transforms(img)
        label = (idx1 == idx2)
        sample = {'raw': (img, video), 'label': label} 
        return sample



    def __len__(self):
        return len(self.imgs) * len(self.imgs)