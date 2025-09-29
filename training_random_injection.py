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

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training:中間出力にノイズ追加')
parser.add_argument('--model_name',type=str)
parser.add_argument('--mixed_arch',type=str,help='量子化精度決定済みモデルのパス,U8の時は空白でOK',default="")
parser.add_argument('-b','--batch_size',type=int)
parser.add_argument('-w','--workers',type=int)
parser.add_argument('-i','--image_size',type=int,default=256)
parser.add_argument('--ep_clean',type=int)
parser.add_argument('--ep_noise',type=int)
parser.add_argument('-n','--noise_scale',type=int)
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('-m','--momentum',default=0.9,type=float)
parser.add_argument('--weight_decay',default=1e-4,type=float)
parser.add_argument('--save_dir',type=str)
parser.add_argument('--save_name',type=str,default='')

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
IMAGE_SIZE = args.image_size
DATASETDIR = '~/datasets/imagenet-100'
def main():
    args = parser.parse_args()
    print(args)
    
    # data loading
    train_loader,val_loader = load_imagenet100(args)
    # train_loader,val_loader = load_cifar10(batch=args.batch_size)
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
    print(f'----- No Training accuracy------')
    print(f'acc1 : {acc1} , acc5 : {acc5}')

    acc1_progress.append(acc1)
    acc5_progress.append(acc5)

    best_epoch = 0
    best_acc1 =acc1
    # Clean Training
    print('----- Clean Trainig -----')
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
        print(f'--- epoch:{epoch+1} ---')
        print(f'acc1 : {acc1} , acc5 : {acc5}')
        
        acc1_progress.append(acc1)
        acc5_progress.append(acc5)
    print('\n')
    natural_bottlenecks = get_natural_bottlenecks(model,IMAGE_SIZE,model.archas)
    split_points = [bottleneck['block_number'] for bottleneck in natural_bottlenecks]
    print(f'split points(block) : {split_points}')
    B_list = find_B(train_loader,model,split_points)
    print(f'Bound threshold median list : {B_list}',end='\n\n')
    # 中間出力にノイズを入れて学習
    # Random Block Noise Injection Training
    print('----- Random Block Noise Injection Training ------')
    model.change_learning_method(noisy_flag = True)
    for epoch in range(args.ep_noise):
        adjust_learning_rate(optimizer, args.ep_clean+epoch, args.lr)
        # B_listを10エポック毎に更新
        if epoch % 10 ==0 and epoch!=0:
            model.change_learning_method(noisy_flag = False)
            B_list = find_B(train_loader,model,split_points)
            print(f'Update Bound threshold median list : {B_list}',end='\n\n')
            model.change_learning_method(noisy_flag = True)
        # train for one epoch
        random_block_noisy_train(train_loader, model, criterion, optimizer, split_points, B_list , args.noise_scale)
            
        # evaluate on validation set
        acc1,acc5= validate(val_loader, model, criterion)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch
        print(f'--- epoch:{epoch+1} ---')
        print(f'acc1 : {acc1} , acc5 : {acc5}')
        
        acc1_progress.append(acc1)
        acc5_progress.append(acc5)
        
    # 念のため
    model.change_learning_method(noisy_flag = False)
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

    plt.plot(epochs[:args.ep_clean+1],acc1_progress[:args.ep_clean+1],color = 'b',label = 'Clean Training')
    plt.plot(epochs[args.ep_clean:],acc1_progress[args.ep_clean:],color = 'r',label = 'Random Block Noise Injection Trainig')

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
            # images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
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

    for i, (images, target) in enumerate(train_loader):


        images = images.to(device)
        target = target.to(device)
        
        # cifar-10,100のみ
        # images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
        
        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 中間出力にノイズを加えてトレーニング
def random_block_noisy_train(train_loader, model, criterion, optimizer, split_points, B_list, noise_scale):


    # switch to train mode
    model.train()
    model.to(device)

    model.change_learning_method(noisy_flag = True)
    for i, (images, target) in enumerate(train_loader):
        
        images = images.to(device)
        target = target.to(device)
        
        # cifar-10,100のみ
        # images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic').to(device)
        # select noise injection block number (random)
        injection_block_idx= torch.randint(low=0, high=torch.numel(B_list), size=(1,))
        model.set_splitting_point(split_points[injection_block_idx])
        model.set_B(B_list[injection_block_idx])
        model.set_noise_scale(noise_scale)
        
        # check
        # print(f"Split Point : {split_points[injection_block_idx]}, Clipping Bound Threshold: {B_list[injection_block_idx]}")
        
        # compute output
        noisy_output = model(images)
        loss = criterion(noisy_output, target)
        
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(noisy_output, target, topk=(1, 5))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# bounding の探索
def find_B(train_loader,model,split_points):
    model.to(device)
    model.change_learning_method(False)
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            # cifar-10,100のみ
            # images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
            model.eval()
            if i==0:
                features_list = torch.stack(model.feature_extractor(images,split_points))
            else:
                features = torch.stack(model.feature_extractor(images,split_points))
                features_list = torch.cat([features_list,features],dim = 1)
    # check
    # print([len(v) for v in features_list])
    
    
    bound_threshold_medi = torch.median(features_list,dim = 1).values
    # bound_threshold_mean = torch.mean(features_list,dim = 1)
    
    # check 
    # print(f"bound threshold median : {bound_threshold_medi}")
    # print(f"bound threshold mean : {bound_threshold_mean}")

    return bound_threshold_medi

def get_natural_bottlenecks(model, input_size, act_bits, compressive_only=True):
    # 各層のinputサイズを計算して、圧縮率が最も高い層を探す
    natural_bottlenecks = []
    best_compression = 1.0
    cnn_count = 0  # CNNレイヤーのカウント
    input_bit = 8 # 入力のbit数
    min_bit = 8  # 探索する最小のbit数←使って無くない？
    bit_compression = [act_bit / input_bit for act_bit in act_bits]

    device = next(model.parameters()).device
    
    mock_input = torch.randn(1, 3, input_size, input_size).to(device)
    previous_size = torch.prod(torch.tensor(mock_input.shape[1:])).item()

    for i, module in enumerate(model.features):
        # print(i, module)
        block_number = i-1 # 0はfeaturesの最初のBasicCNNBlockなので、1から始める
        if isinstance(module, BasicCNNBlock):
            print(f"Encountered BasicBlock at features.{i}")
            output = module(mock_input)
            mock_input = output.detach()
            continue
        
        input_size_layer = torch.prod(torch.tensor(mock_input.shape[1:])).item()
        if input_size_layer * act_bits[cnn_count] < input_size * input_size * 3 * input_bit:
            compression = float(input_size_layer) / (input_size * input_size * 3)
            compression *= bit_compression[cnn_count]
            if not compressive_only or compression < best_compression:
                natural_bottlenecks.append({
                    'layer_name': "blocks_{}".format(block_number),
                    'compression': compression,
                    'cnn_layer_number': cnn_count,  # ここでCNNレイヤーの番号を記録
                    'block_number': block_number,  
                })
                best_compression = compression
        output = module(mock_input)
        mock_input = output.detach()
        
        cnn_count += count_conv2d_layers(module)
    return natural_bottlenecks

def count_conv2d_layers(model):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count += 1
        elif isinstance(module, nn.Sequential):
            # Sequentialブロック内でさらにConv2dを探す
            for sub_module in module:
                if isinstance(sub_module, nn.Conv2d):
                    count += 1
    return count
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

def load_imagenet100(args):
    # Data loading code
    traindir = os.path.join(DATASETDIR, 'train')
    valdir = os.path.join(DATASETDIR , 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    crop_size, short_size = IMAGE_SIZE,IMAGE_SIZE


    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    # print(f'crop size:{crop_size},short_size:{short_size}')
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, 
        sampler=train_sampler)

    val_dataset =datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(short_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # print(len(train_loader))
    # print(len(val_loader))
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
    print("Program training_random_injection.py execution has begun.", end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))
    
    original_stdout = sys.stdout
    with open(f'{save_path}/traning_random_output.txt', 'w') as f:
        sys.stdout = f
        print(f"beginning time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}\n")
        main()
        print(f"\nend time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}")
    sys.stdout = original_stdout
    
    print(f"Program training_random_injection.py execution has done.",end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))
    