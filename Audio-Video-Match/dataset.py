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
            self.imgs += os.listdir(os.path.join("dataset/task1/test/"))

        
    def __getitem__(self, idx):
        if self.mode == "train":
            img_path = os.path.join("dataset/train/", self.imgs[idx])
        
            pth = img_path + "/audio_data.pkl"
            #print(pth)
            #img = torch.load(pth)
            # TODO

            #a =pickle.load(open('dataset/train/salt_cylinder/0/audio_data.pkl', 'rb'))
            #print(a) 
            sig = pickle.load(open(pth, 'rb'))["audio"]
            img = spectrogram(sig)
            np.save(img_path + "/spec.npy", img)
            #img = np.load(img_path + "/spec.npy")
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
    def __init__(self, cat = 0, mode = "train", transforms = None):
        #self.num_class = 10
        self.classname = labels[cat]
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

    
    def __getitem__(self, idx1, idx2):
            if self.mode == "train":
                imgpath = os.path.join("dataset/train/", self.imgs[idx1])
                img_pth = imgpath + "/audio_data.pkl"
                #print(pth)
                #img = torch.load(pth)
                # TODO
                #a =pickle.load(open('dataset/train/salt_cylinder/0/audio_data.pkl', 'rb'))
                #print(a) 
                sig = pickle.load(open(img_pth, 'rb'))["audio"]
                img = spectrogram(sig)
                np.save(img_path + "/spec.npy", img)
                #img = np.load(img_path + "/spec.npy")
                img = Image.fromarray(img)

                videopath = os.path.join("dataset/train/", self.imgs[idx2]) 
                video_pth = videopath + "mask"

                video = np.array([])
                for a in video_pth:
                    image = Image.open(os.path.join(video_pth, a))
                    image = np.array(image)
                    if video.shape == (0,):
                        video = image
                    else:
                        video = np.dstack((video, image))
            label = (idx1 == idx2)
            sample = {'raw': (img, video), 'label': label} 
            return sample