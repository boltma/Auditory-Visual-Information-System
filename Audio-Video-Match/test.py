import models
from dataset import my_dataset
from dataset import matching_dataset
from dataset import video_dataset
import numpy as np
import torch
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import pickle

transform = transforms.Compose([
        transforms.ToTensor()
    ])

classes = [
    '061_foam_brick',
    'green_basketball',
    'salt_cylinder',
    'shiny_toy_gun',
    'stanley_screwdriver',
    'strawberry',
    'toothpaste_box',
    'toy_elephant',
    'whiteboard_spray',
    'yellow_block',
]

def test_task1(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task1/test/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 1, ‘audio_0001’: 3, ...}
    class number:
        ‘061_foam_brick’: 0
        'green_basketball': 1
        'salt_cylinder': 2
        'shiny_toy_gun': 3
        'stanley_screwdriver': 4
        'strawberry': 5
        'toothpaste_box': 6
        'toy_elephant': 7
        'whiteboard_spray': 8
        'yellow_block': 9
    '''
    preds = []
    names = []
    ds = my_dataset("test", transform, root_path)
    loader = torch.utils.data.DataLoader(ds, 64, False, num_workers = 20)
    model = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load("task1resnet18.pkl"))
    model.eval()
    for data in loader:
        image, name = data['image'], data['name']
        image = image.cuda()
        image = torch.autograd.Variable(image)
        output = model(image)
        pred = torch.argmax(output, 1)
        names.extend(name)
        preds.extend(pred)
    preds = [int(i) for i in preds]
    names = [i[-14:] for i in names]
    #dirs = os.listdir(os.path.join(root_path))
    results = dict(zip(names, preds)) 
    np.save("task1result.npy", results)
    return results

def test_task2(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task2/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 23, ‘audio_0001’: 11, ...}
    This means audio 'audio_0000.pkl' is matched to video 'video_0023' and ‘audio_0001’ is matched to 'video_0011'.
    '''
    results = {}
    
    audio_preds = []
    audio_names = []
    audio_ds = my_dataset("test", transform, root_path)
    audio_loader = torch.utils.data.DataLoader(audio_ds, 64, False, num_workers = 20)
    audio_model = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
    audio_model = nn.DataParallel(audio_model).cuda()
    audio_model.load_state_dict(torch.load("task1resnet18.pkl"))
    audio_model.eval()
    for data in audio_loader:
        image, name = data['image'], data['name']
        image = image.cuda()
        image = torch.autograd.Variable(image)
        output = audio_model(image)
        pred = torch.argmax(output, 1)
        audio_names.extend(name)
        audio_preds.extend(pred)
    audio_preds = [int(i) for i in audio_preds]
    audio_names = [i.split('/')[-1][:-4] for i in audio_names]
    audio_results = dict(zip(audio_names, audio_preds))
    
    video_preds = []
    video_names = []
    video_ds = video_dataset("test", transform, root_path)
    video_loader = torch.utils.data.DataLoader(video_ds, 64, False, num_workers = 20)
    video_model = models.ResNet(in_ch=1, in_stride=(1,1), fc_size=64, block=models.BasicBlock, num_blocks=[3,3,3])
    video_model = nn.DataParallel(video_model).cuda()
    video_model.load_state_dict(torch.load("new_video_resnet.pkl"))
    video_model.eval()
    for data in video_loader:
        image, name = data['image'], data['name']
        image = image.cuda()
        image = torch.autograd.Variable(image)
        output = video_model(image)
        pred = torch.argmax(output, 1)
        video_names.extend(name)
        video_preds.extend(pred)
    video_preds = [int(i) for i in video_preds]
    video_names_unique = list(set(video_names))
    video_preds_max = []
    for name in video_names_unique:
        indices = [i for i, x in enumerate(video_names) if x == name]
        pred = [video_preds[i] for i in indices]
        pred = max(pred, key=pred.count)
        video_preds_max.append(pred)
    video_results = dict(zip(video_names_unique, video_preds_max))
    
    audio_num = len(audio_names)
    for i in range(10):
        class_name = classes[i]
        matching_resnet_model = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
        matching_resnet_model = nn.DataParallel(matching_resnet_model).cuda()
        matching_resnet_model.load_state_dict(torch.load(class_name + "_resnet.pkl"))
        matching_resnet_model.eval()
        matching_mlp_model = models.MLP()
        matching_mlp_model = nn.DataParallel(matching_mlp_model).cuda()
        matching_mlp_model.load_state_dict(torch.load(class_name + "_mlp.pkl"))
        matching_mlp_model.eval()
        
        audio_i = [k for k, v in audio_results.items() if v == i]
        video_i = [k for k, v in video_results.items() if v == i]
        print(audio_i)
        print(video_i)
        
        matching_ds = matching_dataset(mode="test", transforms=transform, test_path=root_path, test_audio=audio_i, test_video=video_i)
        matching_loader = torch.utils.data.DataLoader(matching_ds, 64, False, num_workers = 20)
        
        distance_matrix = np.zeros((len(audio_i), len(video_i)))
        
        for data in matching_loader:
            raw, name = data['raw'], data['name']
            image = raw[0]
            video = raw[1]
            image = torch.autograd.Variable(image.cuda())
            video = torch.autograd.Variable(video.cuda()) 
            video_output = matching_mlp_model(video)
            image_output = matching_resnet_model(image)
            dist = F.pairwise_distance(video_output, image_output)
            for j in range(len(dist)):
                audio_num = audio_i.index(name[0][j])
                video_num = video_i.index(name[1][j])
                distance_matrix[audio_num][video_num] = dist[j]
        
        print(distance_matrix)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        print(row_ind)
        print(col_ind)
        for j in range(len(row_ind)):
            audio_name = audio_i[row_ind[j]]
            video_name = video_i[col_ind[j]]
            results[audio_name] = video_name
    
    audio_set = list(set(audio_names) - set([k for k, v in results.items()]))
    video_set = list(set(video_names) - set([v for k, v in results.items()]))
    perm = np.random.permutation(len(audio_set))
    for j in perm:
        audio_name = audio_set[j]
        video_name = video_set[j]
        results[audio_name] = video_name
    for k, v in results.items():
        results[k] = int(v[-4:])
    print(results)
    return results

def test_task3(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task3/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': -1, ‘audio_0001’: 12, ...}
    This means audio 'audio_0000.pkl' is not matched to any video and ‘audio_0001’ is matched to 'video_0012'.
    '''
    results = {}
    
    audio_preds = []
    audio_names = []
    audio_ds = my_dataset("test", transform, root_path)
    audio_loader = torch.utils.data.DataLoader(audio_ds, 64, False, num_workers = 20)
    audio_model = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
    audio_model = nn.DataParallel(audio_model).cuda()
    audio_model.load_state_dict(torch.load("task1resnet18.pkl"))
    audio_model.eval()
    for data in audio_loader:
        image, name = data['image'], data['name']
        image = image.cuda()
        image = torch.autograd.Variable(image)
        output = audio_model(image)
        pred = torch.argmax(output, 1)
        audio_names.extend(name)
        audio_preds.extend(pred)
    audio_preds = [int(i) for i in audio_preds]
    audio_names = [i.split('/')[-1][:-4] for i in audio_names]
    audio_results = dict(zip(audio_names, audio_preds))
    
    video_preds = []
    video_names = []
    video_ds = video_dataset("test", transform, root_path)
    video_loader = torch.utils.data.DataLoader(video_ds, 64, False, num_workers = 20)
    video_model = models.ResNet(in_ch=1, in_stride=(1,1), fc_size=64, block=models.BasicBlock, num_blocks=[3,3,3])
    video_model = nn.DataParallel(video_model).cuda()
    video_model.load_state_dict(torch.load("new_video_resnet.pkl"))
    video_model.eval()
    for data in video_loader:
        image, name = data['image'], data['name']
        image = image.cuda()
        image = torch.autograd.Variable(image)
        output = video_model(image)
        pred = torch.argmax(output, 1)
        video_names.extend(name)
        video_preds.extend(pred)
    video_preds = [int(i) for i in video_preds]
    video_names_unique = list(set(video_names))
    video_preds_max = []
    for name in video_names_unique:
        indices = [i for i, x in enumerate(video_names) if x == name]
        pred = [video_preds[i] for i in indices]
        pred = max(pred, key=pred.count)
        video_preds_max.append(pred)
    video_results = dict(zip(video_names_unique, video_preds_max))
    
    audio_num = len(audio_names)
    for i in range(10):
        class_name = classes[i]
        matching_resnet_model = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
        matching_resnet_model = nn.DataParallel(matching_resnet_model).cuda()
        matching_resnet_model.load_state_dict(torch.load(class_name + "_resnet.pkl"))
        matching_resnet_model.eval()
        matching_mlp_model = models.MLP()
        matching_mlp_model = nn.DataParallel(matching_mlp_model).cuda()
        matching_mlp_model.load_state_dict(torch.load(class_name + "_mlp.pkl"))
        matching_mlp_model.eval()
        
        audio_i = [k for k, v in audio_results.items() if v == i]
        video_i = [k for k, v in video_results.items() if v == i]
        print(audio_i)
        print(video_i)
        
        matching_ds = matching_dataset(mode="test", transforms=transform, test_path=root_path, test_audio=audio_i, test_video=video_i)
        matching_loader = torch.utils.data.DataLoader(matching_ds, 64, False, num_workers = 20)
        
        distance_matrix = np.zeros((len(audio_i), len(video_i)))
        
        for data in matching_loader:
            raw, name = data['raw'], data['name']
            image = raw[0]
            video = raw[1]
            image = torch.autograd.Variable(image.cuda())
            video = torch.autograd.Variable(video.cuda()) 
            video_output = matching_mlp_model(video)
            image_output = matching_resnet_model(image)
            dist = F.pairwise_distance(video_output, image_output)
            for j in range(len(dist)):
                audio_num = audio_i.index(name[0][j])
                video_num = video_i.index(name[1][j])
                distance_matrix[audio_num][video_num] = dist[j]
        
        print(distance_matrix)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        print(row_ind)
        print(col_ind)
        for j in range(len(row_ind)):
            audio_name = audio_i[row_ind[j]]
            video_name = video_i[col_ind[j]]
            if distance_matrix[row_ind[j]][col_ind[j]] <= 10:
                results[audio_name] = video_name
            else:
                results[audio_name] = -1
    
    audio_set = list(set(audio_names) - set([k for k, v in results.items()]))
    for audio_name in audio_set:
        results[audio_name] = -1
    for k, v in results.items():
        if results[k] != -1:
            results[k] = int(v[-4:])
    print(results)
    print(len(set([k for k, v in results.items()])))
    return results

# test_task1("dataset/task1/test")
test_task2("dataset/task2/test/0/")
