import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.init as init
import torch.nn as nn
import argparse
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=20.0):
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin

    def forward(self, output1, output2, label):
        l = 0.5
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(l * (label) * torch.pow(euclidean_distance, 2) +
                                      (1 - l) * (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive