"""Explicit 1D model definitions; all models emit logits for five labels."""
from __future__ import annotations
import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self,inputs,outputs,kernel,stride=1):
        super().__init__(nn.Conv1d(inputs,outputs,kernel,stride,padding=kernel//2,bias=False),
                         nn.BatchNorm1d(outputs),nn.ReLU())


class CompactCNN(nn.Module):
    def __init__(self,dropout=0.2,width=1):
        super().__init__()
        channels=[int(c*width) for c in [64,128,192]]
        self.features=nn.Sequential(ConvBlock(12,channels[0],15,2),ConvBlock(channels[0],channels[1],9,2),
                                   ConvBlock(channels[1],channels[2],5,2),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(channels[2],64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,5))
    def forward(self,x):return self.head(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self,inputs,outputs,stride=1):
        super().__init__()
        self.main=nn.Sequential(ConvBlock(inputs,outputs,3,stride),nn.Conv1d(outputs,outputs,3,padding=1,bias=False),nn.BatchNorm1d(outputs))
        self.skip=nn.Identity() if inputs==outputs and stride==1 else nn.Sequential(nn.Conv1d(inputs,outputs,1,stride,bias=False),nn.BatchNorm1d(outputs))
        self.relu=nn.ReLU()
    def forward(self,x):return self.relu(self.main(x)+self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self,dropout=0.2,base_width=16):
        super().__init__()
        layers=[ConvBlock(12,base_width,7,2),nn.MaxPool1d(3,2,padding=1)]
        inputs=base_width
        for stage,outputs in enumerate([base_width,base_width*2,base_width*4,base_width*8]):
            layers.extend([ResidualBlock(inputs,outputs,1 if stage==0 else 2),ResidualBlock(outputs,outputs)])
            inputs=outputs
        self.features=nn.Sequential(*layers,nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(inputs,5))
    def forward(self,x):return self.head(self.features(x))


class InceptionModule(nn.Module):
    def __init__(self,inputs,filters=16):
        super().__init__()
        self.bottleneck=nn.Conv1d(inputs,filters,1,bias=False)
        self.branches=nn.ModuleList([nn.Conv1d(filters,filters,k,padding=k//2,bias=False) for k in [9,19,39]])
        self.pool=nn.Sequential(nn.MaxPool1d(3,1,padding=1),nn.Conv1d(inputs,filters,1,bias=False))
        self.activation=nn.Sequential(nn.BatchNorm1d(filters*4),nn.ReLU())
    def forward(self,x):
        z=self.bottleneck(x)
        return self.activation(torch.cat([branch(z) for branch in self.branches]+[self.pool(x)],dim=1))


class InceptionTime(nn.Module):
    def __init__(self,dropout=0.2,filters=16,stem_stride=4):
        super().__init__()
        width=filters*4
        self.stem=nn.Sequential(nn.Conv1d(12,12,15,stride=stem_stride,padding=7,groups=12,bias=False),nn.BatchNorm1d(12),nn.ReLU())
        self.group1=nn.Sequential(InceptionModule(12,filters),InceptionModule(width,filters),InceptionModule(width,filters))
        self.project=nn.Sequential(nn.Conv1d(12,width,1,bias=False),nn.BatchNorm1d(width))
        self.group2=nn.Sequential(*[InceptionModule(width,filters) for _ in range(3)])
        self.relu=nn.ReLU()
        self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(dropout),nn.Linear(width,5))
    def forward(self,x):
        x=self.stem(x)
        z=self.relu(self.group1(x)+self.project(x))
        return self.head(self.relu(self.group2(z)+z))


def build_model(name,dropout=0.2,profile='cpu'):
    if name=='compact_cnn':return CompactCNN(dropout)
    if name=='resnet1d':return ResNet1D(dropout,base_width=16 if profile=='cpu' else 64)
    if name=='inception_time':return InceptionTime(dropout,filters=16 if profile=='cpu' else 32,stem_stride=4 if profile=='cpu' else 1)
    raise ValueError('Unknown model '+name)


def parameter_count(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)
