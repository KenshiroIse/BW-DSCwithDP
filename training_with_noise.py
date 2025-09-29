import argparse
import os
import random
import shutil
import time
import warnings
import datetime
import sys
import json
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pprint import pprint
import matplotlib.ticker as ticker

os.chdir('./')
# EdMIPS/models ディレクトリへのパスを追加
sys.path.append('./models')
from models.quant_efficientnet import BasicCNNBlock
import models as models

# カレントディレクトリを 'EdMIPS' に変更
os.chdir('.')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--model_name',type=str)
parser.add_argument('--mixed_arch',type=str,help='量子化精度決定済みモデルのパス,U8の時は空白でOK',default="")
parser.add_argument('-b','--batch_size',type=int)
parser.add_argument('-w','--workers',type=int)
parser.add_argument('-i','--image_size',type=int,default=256)
parser.add_argument('--ep_clean',type=int)
parser.add_argument('--ep_noise',type=int)
parser.add_argument('-n','--noise_scale',type=int)
parser.add_argument('--lr',default=0.1,type=float)
parser.add_argument('-m','--momentum',default=0.9,type=float)
parser.add_argument('--weight_decay',default=1e-4,type=float)
parser.add_argument('--save_dir',type=str)
parser.add_argument('--save_name',type=str,default='')

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
IMAGE_SIZE = args.image_size

def main():
    args = parser.parse_args()
    print(args)
    
    # data loading
    # train_loader,val_loader = load_imagenet100(DATASETDIR,config)
    train_loader,val_loader = load_cifar10(batch=args.batch_size)
    # check
    # print(len(train_loader),len(val_loader))
    
    # create models　最初から学習を始めるため
    model = models.__dict__[args.model_name](args.mixed_arch)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)


    model.to(device)
    model.change_learning_method(False)
    
    acc1_progress = []
    acc5_progress = []

    acc1,acc5= validate(val_loader, model, criterion)
    print(f'----- No Training accuracy ------')
    print(f'acc1 : {acc1}  acc5 : {acc5}')

    acc1_progress.append(acc1)
    acc5_progress.append(acc5)

    best_epoch = 0
    best_acc1 =acc1
    # Clean Training
    for epoch in range(args.ep_clean):
        adjust_learning_rate(optimizer, epoch, args.lr)
        # train for one epoch
        train(train_loader, model, criterion, optimizer)
            
        # evaluate on validation set
        acc1,acc5= validate(val_loader, model, criterion)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch
        print(f'----- Clean Training epoch:{epoch+1} accuracy ------')
        print(f'acc1 : {acc1}  acc5 : {acc5}')
        
        acc1_progress.append(acc1)
        acc5_progress.append(acc5)
        
        input_norm_list = []
        
    # boundingの探索
    # for i, (images, target) in enumerate(train_loader):
    #     input_norm = torch.max(torch.flatten(images,1,-1),dim=1).values
    #     # print(input_norm)
    #     if i==0:
    #         input_norm_list = input_norm
    #     else:
    #         input_norm_list = torch.cat([input_norm_list,input_norm],dim = 0)
            
    # B_input = torch.median(input_norm_list)
    
    # # B_input:0.9529(cifar-10)
    # # B_input:0.9686(cifar-100)
    B_input = 0.9529
    print(f"\nBounding threshold of input : {B_input}\n")
    
    for epoch in range(args.ep_noise):
        adjust_learning_rate(optimizer, args.ep_clean+epoch, args.lr)
        # train for one epoch
        noisy_train(train_loader, model, criterion, optimizer, epoch, B_input , args.noise_scale)
            
        # evaluate on validation set
        acc1,acc5= validate(val_loader, model, criterion)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch
        print(f'----- Only Noise Data Training epoch:{epoch+1} accuracy ------')
        print(f'acc1 : {acc1}  acc5 : {acc5}')
        
        acc1_progress.append(acc1)
        acc5_progress.append(acc5)
    
    # save model
    save_model({
    'epoch': epoch + 1,
    'arch': args.model_name,
    'state_dict': model.state_dict(),
    'best_acc1': best_acc1,
    'optimizer': optimizer.state_dict(),
    }, epoch, filename=f"{args.save_name}.pth.tar",save_dir=args.save_dir)
    
    # visualization
    epochs = range(0,args.ep_clean+args.ep_noise+1)

    plt.plot(epochs[:args.ep_clean],acc1_progress[:args.ep_clean],color = 'b',label = 'Clean Training')
    plt.plot(epochs[args.ep_clean:],acc1_progress[args.ep_clean:],color = 'r',label = 'Noise Trainig')

    # plt.ylim(60,80)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Top1-Accuracy(%)')
    plt.title('Top1-Accuracy Progress')
    plt.legend()
    plt.savefig(f"./privacy_aware_dsc/{args.save_dir}/acc1_progress.png",bbox_inches='tight')
    plt.close()
        
    
    
    
def validate(val_loader, model, criterion):
    
    acc1_avg = 0
    acc5_avg = 0
    # switch to evaluate mode
    model.eval()
    model.to(device)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # cifar-10,100のみ
            images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_avg += float(acc1)
            acc5_avg += float(acc5)
            # check
            # print(f'{i}  acc1:{int(acc1)}  acc5:{int(acc5)}')
            
    acc1_avg = acc1_avg / len(val_loader)
    acc5_avg = acc5_avg / len(val_loader)
    return acc1_avg , acc5_avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
    step_epoch = 10
    lr = start_lr * (0.1 ** (epoch // step_epoch))
    # print(f'Learning rate : {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(train_loader, model, criterion, optimizer):

    # switch to train mode
    model.train()
    model.to(device)

    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):


        images = images.to(device)
        target = target.to(device)
        
        # cifar-10,100のみ
        images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
        
        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute gradient and do SGD step
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 入力にノイズを加えてトレーニング
def noisy_train(train_loader, model, criterion, optimizer, epoch, B, noise_scale):


    # switch to train mode
    model.train()
    model.to(device)

    
    for i, (images, target) in enumerate(train_loader):
        
        images = images.to(device)
        target = target.to(device)
        
        # cifar-10,100のみ
        images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')

        laplace_dist =torch.distributions.Laplace(loc = 0, scale = B/noise_scale)
        noisy_images = images + laplace_dist.sample(images.size()).to(device)
        noisy_images = noisy_images.to(device)
        
        # compute output
        noisy_output = model(noisy_images)
        loss = criterion(noisy_output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(noisy_output, target, topk=(1, 5))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# bounding の探索
def find_B(train_loader,model,splitting_points):
    model.to(device)
    model.change_learning_method(False)
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            # cifar-10,100のみ
            images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
            model.eval()
            if i==0:
                features_list = torch.stack(model.feature_extractor(images,splitting_points))
            else:
                features = torch.stack(model.feature_extractor(images,splitting_points))
                features_list = torch.cat([features_list,features],dim = 1)
    # check
    # print([len(v) for v in features_list])
    
    
    bound_threshold_medi = torch.median(features_list,dim = 1).values
    bound_threshold_mean = torch.mean(features_list,dim = 1)
    
    # check 
    # print(f"bound threshold median : {bound_threshold_medi}")
    # print(f"bound threshold mean : {bound_threshold_mean}")

    return bound_threshold_medi

def load_cifar10(batch):
    train_loader = DataLoader(
        datasets.CIFAR10('./data',
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                [0.5, 0.5, 0.5],  # RGB 平均
                                [0.5, 0.5, 0.5]   # RGB 標準偏差
                                )
                         ])),
        batch_size=batch,
        shuffle=True
    )

    val_loader = DataLoader(
        datasets.CIFAR10('./data',
                         train=False,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.5, 0.5, 0.5],  # RGB 平均
                                 [0.5, 0.5, 0.5]  # RGB 標準偏差
                             )
                         ])),
        batch_size=batch,
        shuffle=True
    )

    return train_loader,val_loader

def save_model(state, epoch, filename='checkpoint.pth.tar',save_dir=datetime.datetime.now().strftime('%y-%m%d_%H%M%S')):
    save = f'./privacy_aware_dsc/{save_dir}'
    os.makedirs(save, exist_ok=True)
    torch.save(state, f"{save}/{filename}")

if __name__ == '__main__':
    
    args = parser.parse_args()
    save_path = f'./privacy_aware_dsc/{args.save_dir}'
    os.makedirs(save_path, exist_ok=True)
    
    print(args.save_name)
    print("Program training_with_noise.py execution has begun.", end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))
    
    original_stdout = sys.stdout
    with open(f'{save_path}/traning_output.txt', 'w') as f:
        sys.stdout = f
        print(f"beginning time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}")
        main()
        print(f"end time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}")
    sys.stdout = original_stdout
    
    print(f"Program training_with_noise.py execution has done.",end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))
    