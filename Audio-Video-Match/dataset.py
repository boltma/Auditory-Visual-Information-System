import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import wave
import pickle
import random
from spectrogram import spectrogram
from centroid import centroid
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
    def __init__(self, mode = "train", transforms = None, test_path = None):
        self.num_class = 10
        self.imgs = []
        self.img_class = []
        self.mode = mode
        self.transforms = transforms
        self.test_path = test_path
        if self.mode == "train":
            self.classes = os.listdir(os.path.join("dataset/train"))
            for img_classes in self.classes:
                img_dir = os.listdir(os.path.join("dataset/train", img_classes))
                self.imgs += list(map(lambda x: img_classes + "/" + x, img_dir))
                c = [img_classes] * len(img_dir)
                self.img_class += c
        elif self.mode == "test":
            self.imgs += [f for f in os.listdir(self.test_path) if f.endswith('pkl')]
   
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
            img_path = os.path.join(self.test_path, self.imgs[idx])
            #img = pickle.load(open(img_path, 'rb'))["audio"]
            filename, file_extension = os.path.splitext(img_path)
            #print(img_path)
            sig = pickle.load(open(img_path, 'rb'))['audio']
            #print(sig['audio'])
            
            img = spectrogram(sig)
            #np.save(filename + "/spec.npy", img)
            img = Image.fromarray(img)
        
        #print(img.shape)
        #img=Pickle.load(open(img_path + "/audio_data.pkl"))
        if self.transforms is not None:
            img = self.transforms(img)
        if self.mode == "train":
            sample = {'image': img, 'label': labels[self.img_class[idx]]}
        elif self.mode == "test":
            sample = {'image': img, "name": img_path}
        #print(sample)
        return sample
    def __len__(self):
        return len(self.imgs)

class matching_dataset(torch.utils.data.Dataset):
    def __init__(self, cat = "yellow_block", mode = "train", transforms = None, test_path = None, test_audio = None, test_video = None):
        #self.num_class = 10
        #self.classname = labels[cat]
        self.classname = cat
        self.classes = os.listdir(os.path.join("dataset/train"))
        self.imgs = []
        self.img_class = []
        self.mode = mode
        self.transforms = transforms
        self.test_path = test_path
        self.audio = []
        self.video = []
        if self.mode == "train":
            #self.classes = os.listdir(os.path.join("dataset/train"))
            if self.classname is None:
                for img_classes in self.classes:
                    img_dir = os.listdir(os.path.join("dataset/train" , img_classes))
                    self.imgs += list(map(lambda x: img_classes + "/" + x, img_dir))
                    c = [self.classname] * len(img_dir)
                    #self.img_class += c
            else:
                img_dir = os.listdir(os.path.join("dataset/train" , self.classname))

                self.imgs += list(map(lambda x: self.classname + "/" + x, img_dir))
                c = [self.classname] * len(img_dir)
            self.imglen = len(self.imgs)
        elif self.mode == "test":
            audio_pth = [f + ".pkl" for f in test_audio]
            for v in test_video:
                self.video += [v] * len(audio_pth)
                self.audio += audio_pth
            self.imglen = len(self.video)
                    
        #a = [lists for lists in os.listdir("dataset/train/") if os.path.isfile(os.path.join("dataset/train", lists))]
        #print(a)
    def __getitem__(self, idx):
        label = ''
        if self.mode == "train":
            # idx1 = idx // 2
            # idx2 = idx1
            # if idx % 2:
            #     idx2 = idx2 - 1
            #     if idx2 < 0:
            #         idx2 = idx2 + 2
            idx1 = idx // 2
            if idx % 2 == 0:
                idx2 = idx1
            else:
                idx2 =  random.randint(idx1 + 1, self.imglen - 1) if idx1 < self.imglen / 2 else random.randint(0, idx1 - 1) 

            
            #print(idx2)
            # idx1 = idx // self.imglen 
            # idx2 = idx % self.imglen
            imgpath = os.path.join("dataset/train/", self.imgs[idx1])
            img_pth = imgpath + "/audio_data.pkl"

            #sig = pickle.load(open(img_pth, 'rb'))["audio"]
            #img = spectrogram(sig)
            # np.save(imgpath + "/spec.npy", img)
            img = np.load(imgpath + "/spec.npy")
            img = Image.fromarray(img)

            #img = img.swapaxes(0, 2)
            video_pth = os.path.join("dataset/train/", self.imgs[idx2], 'mask')
            pth_list = os.listdir(video_pth)
            pth_list.sort()
            # video = np.array([])
            # for a in pth_list:
            #     filename, file_extension = os.path.splitext(os.path.join(video_pth, a))
            #     if file_extension != '.png':
            #         continue
            #     image = Image.open(os.path.join(video_pth, a))
            #     image = image.resize((60, 60))
            #     image = np.array(image)
            #     if video.shape == (0,):
            #         video = image
            #     else:
            #         video = np.dstack((video, image))
            # i = 0
            # while video.shape[2] < 35:
            #     video = np.dstack((video, video[:,:,i]))
            #     i = i + 1
            
            # video = np.swapaxes(video, 0, 2)
            
            # video = np.reshape(video, (1, 35, 60, 60))

            cen = np.array([])
            for a in pth_list:
                
                filename, file_extension = os.path.splitext(os.path.join(video_pth, a))
                if file_extension != '.png':
                    continue
                image = Image.open(os.path.join(video_pth, a))
                image = np.array(image)
                x_center, y_center = centroid(image)
                cen = np.append(cen, np.array([x_center, y_center]))
            if len(cen) < 70:
                cen = np.append(cen, cen[0:70-len(cen)])
            cen = np.float32(cen)
            np.save(video_pth + "/centroid.npy", cen)

            #cen = np.load(video_pth + "/centroid.npy")
            #print(len(cen))
                
            # for i in range(35):
            #     x_center, y_center = centroid(video[:, :, i])
            #     cen = np.append(cen, np.array([x_center, y_center]))
            # video = np.swapaxes(video, 0, 2)
            # video = np.reshape(video, (1, 35, 60, 60))
            # print(cen.shape)
            
            label = np.float32(idx1 == idx2)
        
        elif self.mode == "test":
            img_pth = os.path.join(self.test_path, self.audio[idx])

            sig = pickle.load(open(img_pth, 'rb'))["audio"]
            img = spectrogram(sig)
            img = Image.fromarray(img)

            video_pth = os.path.join(self.test_path, self.video[idx], 'mask')
            pth_list = os.listdir(video_pth)
            pth_list.sort()

            cen = np.array([])
            for a in pth_list:
                filename, file_extension = os.path.splitext(os.path.join(video_pth, a))
                if file_extension != '.png':
                    continue
                image = Image.open(os.path.join(video_pth, a))
                image = np.array(image)
                x_center, y_center = centroid(image)
                cen = np.append(cen, np.array([x_center, y_center]))
            if len(cen) < 70:
                cen = np.append(cen, cen[0:70-len(cen)])
            cen = np.float32(cen)

        if self.transforms is not None:
            img = self.transforms(img)
        if self.mode == "train":
            sample = {'raw': (img, cen), 'label': label} 
        elif self.mode == "test":
            sample = {'raw': (img, cen), 'name': (self.audio[idx][:-4], self.video[idx])} 
        return sample 

    def __len__(self):
        if self.mode == "train":
            return self.imglen * 2
        elif self.mode == "test":
            return self.imglen
        return 0

class video_dataset(torch.utils.data.Dataset):
    def __init__(self, mode = "train", transforms = None, test_path = None):
        self.num_class = 10
        self.imgs = []
        self.img_class = []
        self.mode = mode
        self.transforms = transforms
        self.test_path = test_path
        if self.mode == "train":
            self.classes = os.listdir(os.path.join("dataset/train"))
            for img_classes in self.classes:
                img_dir = os.listdir(os.path.join("dataset/train", img_classes))
                img_samples = list(map(lambda x: img_classes + "/" + x, img_dir))
                for sample_path in img_samples:
                    sample_dir = [f for f in os.listdir(os.path.join("dataset/train", sample_path, "mask")) if f.endswith('png')]
                    self.imgs += list(map(lambda x: sample_path + "/mask/" + x, sample_dir))
                    c = [img_classes] * len(sample_dir)
                    self.img_class += c
        elif self.mode == "test":
            img_dir = [d for d in os.listdir(self.test_path) if os.path.isdir(os.path.join(self.test_path, d))]
            for sample_path in img_dir:
                sample_dir = [f for f in os.listdir(os.path.join(self.test_path, sample_path, "mask")) if f.endswith('png')]
                self.imgs += list(map(lambda x: sample_path + "/mask/" + x, sample_dir))
        
    def __getitem__(self, idx):
        if self.mode == "train":
            img_path = os.path.join("dataset/train/", self.imgs[idx])
            img = Image.open(img_path)
            img = img.resize((60, 60))
        elif self.mode == "test":
            img_path = os.path.join(self.test_path, self.imgs[idx])
            img = Image.open(img_path)
            img = img.resize((60, 60))
        if self.transforms is not None:
            img = self.transforms(img)
        if self.mode == "train":
            sample = {'image': img, 'label': labels[self.img_class[idx]]}
        elif self.mode == "test":
            name = self.imgs[idx].split('/')[0]
            sample = {'image': img, 'name': name}
        #print(sample)
        return sample

    def __len__(self):
        return len(self.imgs)