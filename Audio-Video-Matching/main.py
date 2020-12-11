from dataset import my_dataset
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import models

lr = 1e-3
device = torch.device("cuda")

def train(trainloader, model, criterion, optimizer, epoch , use_cuda, size = None):
    # switch to train mode
    model.train()
    if use_cuda:
        model = model.to(device)
    for data in trainloader:
        model.train(True)
        image, label = data['image'], data['label']
        if use_cuda:
            image = torch.autograd.Variable(image.cuda())
            label = torch.autograd.Variable(label.cuda())

        optimizer.zero_grad()

        output = criterion(model(image), label)
        output.backward()
        optimizer.step()

resnet = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(resnet.parameters(), lr = lr)

train_ds = my_dataset("train")
train_loader = torch.utils.data.DataLoader(train_ds, 128, False, num_workers = 8)
print(len(train_ds))


train_ds.__getitem__(500)
train(train_loader, resnet, criterion, optimizer, 0, use_cuda=True)