from dataset import my_dataset
from dataset import matching_dataset
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import time
lr = 0.02
device = torch.device("cuda")
transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
    ])
def train(trainloader, testloader, model, criterion, optimizer, epoch , use_cuda, size = None):
    # switch to train mode
    model = nn.DataParallel(model).cuda()
    model.train()
    if use_cuda:
        model = model.to(device)
    best_acc = 0.0
    for i in range(epoch):
        loss = 0.0
        print("%d | %d: "%(i, epoch))
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
            loss += output.item()
        print("loss: %.04f"%(loss))
        
        #test_loss, test_acc = test(testloader, model, criterion, 1, use_cuda = True)
        #if test_acc > best_acc:
        #    best_acc = test_acc
        #print('the accuracy is %.03f, the best accuracy is %.03f'%(test_acc, best_acc))
        test1(model, testloader)
        
def test1(model, loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        image, label = data['image'], data['label']
        image, label = image.cuda(), label.cuda()
        image, label = torch.autograd.Variable(image), torch.autograd.Variable(label)
        output = model(image)
        pred = torch.argmax(output, 1)
        correct += (pred == label).sum().float()
        total += len(label)
    acc = (100 * correct * 1.0 / total) 
    print("accuracy : %.03f"%(acc))
    return acc


def test(testloader, model, criterion, epoch, use_cuda, save_pth = None, save = False, best_acc = 0.0):
    #global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    #bar = Bar('Processing', max=len(testloader))
    for data in testloader:
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = data['image'], data['label'] 
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        #print(outputs.data.shape)
        #print(targets.data.shape) 
        prec1 = accuracy(outputs.data, targets.data)
        losses.update(loss.data, inputs.size(0))
        #print(prec1)
        top1.update(prec1[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    if save:
        if top1.avg > best_acc:
           torch.save(model.state_dict(), save_pth) 

    return (losses.avg, top1.avg)

resnet = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
mlp = models.MLP()
criterion = nn.CrossEntropyLoss().cuda()
#optimizer = torch.optim.Adam(resnet.parameters(), lr = lr)
optimizer = torch.optim.SGD(resnet.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
ds = my_dataset("train", transform_train)

train_ds, val_ds = torch.utils.data.random_split(ds, [2000, 249])
train_loader = torch.utils.data.DataLoader(train_ds, 64, False, num_workers = 20)
val_loader = torch.utils.data.DataLoader(val_ds, 64, False, num_workers = 20)

print(len(train_ds))
ds.__getitem__(4)

dataset2 = matching_dataset("train")
dataset2.__getitem__(5)

#train(train_loader, val_loader, resnet, criterion, optimizer, 200, use_cuda=True)