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
def train(trainloader, img_model, video_model, criterion, optimizer, epoch, use_cuda, testloader = None, size = None):
    # switch to train mode
    img_model = nn.DataParallel(img_model).cuda()
    video_model = nn.DataParallel(video_model).cuda()
    img_model.train()
    video_model.train()
    if use_cuda:
        img_model = img_model.to(device)
        video_model = video_model.to(device)
    best_acc = 0.0
    for i in range(epoch):
        loss = 0.0
        print("%d | %d: "%(i, epoch))
        for data in trainloader:
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
            dist = torch.sqrt(torch.sum((image_output - video_output) ** 2, dim=1)) / 10
            output = nn.BCELoss()(dist, label)
            output.backward()
            optimizer.step()
            loss += output.item()
        print("loss: %.04f"%(loss))
        
        #test_loss, test_acc = test(testloader, model, criterion, 1, use_cuda = True)
        #if test_acc > best_acc:
        #    best_acc = test_acc
        #print('the accuracy is %.03f, the best accuracy is %.03f'%(test_acc, best_acc))
        #test(model, testloader)
        
def test(model, loader):
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


resnet = models.ResNet(block=models.BasicBlock, num_blocks=[3,3,3])
cnn = models.CNN3D()
criterion = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(resnet.parameters(), lr = lr)
optimizer = torch.optim.SGD(list(resnet.parameters()) + list(cnn.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)

task2_ds = matching_dataset(cat='yellow_block', transforms = transform_train)
task2_loader = torch.utils.data.DataLoader(task2_ds, 16, False, num_workers = 0)


# print(len(task2_ds))

#dataset2 = matching_dataset(mode="train")
#x = dataset2.__getitem__(5, 6)
#print(x['raw'][1].shape)

train(task2_loader, video_model = cnn, img_model = resnet, criterion= criterion, optimizer= optimizer, epoch= 200, use_cuda=True)
