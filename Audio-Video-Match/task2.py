from dataset import matching_dataset
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import models
from contrastive_loss import ContrastiveLoss
import time
import numpy as np

lr = 0.1
device = torch.device("cuda")
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
])


def train(trainloader, testloader, img_model, video_model, criterion, optimizer, epoch, use_cuda, size=None,
          save=False):
    # switch to train mode
    img_model = nn.DataParallel(img_model).cuda()
    video_model = nn.DataParallel(video_model).cuda()
    img_model.train()
    video_model.train()
    acc_list = []
    loss_list = []
    if use_cuda:
        img_model = img_model.to(device)
        video_model = video_model.to(device)
    best_acc = 0.0
    for i in range(epoch):
        for param_group in optimizer.param_groups:
            if i in [200, 400]:
                param_group['lr'] *= 0.1
        loss = 0.0
        j = 0

        print("%d | %d: " % (i, epoch))
        for data in trainloader:
            # print(j)
            j = j + 1
            img_model.train(True)
            video_model.train(True)
            raw, label = data['raw'], data['label']
            image = raw[0]
            video = raw[1]
            if use_cuda:
                image = torch.autograd.Variable(image.cuda())
                video = torch.autograd.Variable(video.cuda())
                label = torch.autograd.Variable(label.cuda())

            optimizer.zero_grad()
            video_output = video_model(video)
            image_output = img_model(image)

            # dist = torch.max(dist) - dist
            # if j == 1:
            #     print(video_output)
            #     print(image_output)
            output = ContrastiveLoss()(video_output, image_output, label)
            output.backward()
            optimizer.step()
            loss += output.item()
        print("loss: %.04f" % (loss))

        # test_loss, test_acc = test(testloader, model, criterion, 1, use_cuda = True)
        # if test_acc > best_acc:
        #    best_acc = test_acc
        # print('the accuracy is %.03f, the best accuracy is %.03f'%(test_acc, best_acc))
        loss_list.append(loss)
        acc = test(video_model, img_model, testloader, use_cuda)
        acc_list.append(acc)
        if save == True and best_acc < acc:
            torch.save(video_model.state_dict(), "whiteboard_spray_mlp.pkl")
            torch.save(img_model.state_dict(), "whiteboard_spray_resnet.pkl")
        best_acc = max(best_acc, acc)
        print("accuracy : %.03f, best : %.03f" % (acc, best_acc))
    np.save("acc_8.npy", acc_list)
    np.save("loss_8.npy", loss_list)


def test(video_model, img_model, loader, use_cuda):
    video_model.eval()
    img_model.eval()
    correct = 0
    total = 0
    for data in loader:
        raw, label = data['raw'], data['label']
        image = raw[0]
        video = raw[1]
        if use_cuda:
            image = torch.autograd.Variable(image.cuda())
            video = torch.autograd.Variable(video.cuda())
            label = torch.autograd.Variable(label.cuda())
        video_output = video_model(video)
        image_output = img_model(image)
        dist = F.pairwise_distance(video_output, image_output)
        print(dist)
        pred = (dist < 10.0)
        print(label)
        correct += (pred == label).sum().float()
        total += len(label)
    acc = (100 * correct * 1.0 / total)

    return acc


resnet = models.ResNet(block=models.BasicBlock, num_blocks=[3, 3, 3])
cnn3d = models.CNN3D()
cnn = models.CNN()
mlp = models.MLP()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(list(mlp.parameters()) + list(resnet.parameters()), lr=lr)
# optimizer = torch.optim.SGD(list(cnn3d.parameters()) + list(resnet.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)

task2_ds = matching_dataset(cat="whiteboard_spray", transforms=transform_train)
val_len = len(task2_ds) // 10
train_len = len(task2_ds) - val_len
train_ds, val_ds = torch.utils.data.random_split(task2_ds, [train_len, val_len])
train_loader = torch.utils.data.DataLoader(train_ds, 10, False, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_ds, 10, False, num_workers=10)

# print(len(task2_ds))

# dataset2 = matching_dataset(mode="train")
# x = dataset2.__getitem__(5, 6)
# print(x['raw'][1].shape)
# task2_ds.__getitem__(0)
train(train_loader, val_loader, video_model=mlp, img_model=resnet, criterion=criterion, optimizer=optimizer, epoch=500,
      use_cuda=True, save=True)
