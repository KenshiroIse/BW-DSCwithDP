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

# cudnn.benchmark = True
torch.backends.cudnn.deterministic = True  # 再現性確保のための設定
torch.backends.cudnn.benchmark = False     # パフォーマンス向上を無効化

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--model_name',type=str)
parser.add_argument('--mixed_arch',type=str,help='量子化精度決定済み未学習モデルのパス,U8の時は空白でOK',default="")
parser.add_argument('--quant_arch',type=str,help='学習モデルのパス')
parser.add_argument('-b','--batch_size',type=int)
parser.add_argument('-w','--workers',type=int)
parser.add_argument('-i','--image_size',type=int,default=256)
parser.add_argument('--save_dir',type=str)

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
IMAGE_SIZE = args.image_size
DATASETDIR = '~/datasets/imagenet-100'

def main():
    args = parser.parse_args()
    print(args)
    # data loading
    # train_loader,val_loader = load_cifar10(batch=args.batch_size)
    train_loader,val_loader = load_imagenet100(args)
    # model loading
    model = models.__dict__[args.model_name](args.mixed_arch)
    checkpoint = torch.load(args.quant_arch)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    print('')
    # baseline
    model.change_learning_method(noisy_flag = False)
    baseline_acc1,baseline_acc5=validate(val_loader,model,criterion)
    print(f'baseline_acc1 : {baseline_acc1}, baseline_acc5 : {baseline_acc5}')
    
    # settings
    natural_bottlenecks = get_natural_bottlenecks(model,IMAGE_SIZE,model.archas)
    split_points = [bottleneck['block_number'] for bottleneck in natural_bottlenecks]
    noise_scales = [round(num,1) for num in range(1,21)]
    print(f'split points(block) : {split_points}')
    B_list = find_B(train_loader,model,split_points)
    print(f'Bound threshold median list : {B_list}')
    
    # evaluation　ノイズを加えた時のaccuracy
    acc1_list = {}
    acc5_list = {}
    model.change_learning_method(noisy_flag = True)
    for split_point in split_points:
        print(f'Split_Point: {split_point}')
        model.set_splitting_point(split_point)
        model.set_B(B_list[split_points.index(split_point)])
        acc1_list[split_point] = []
        acc5_list[split_point] = []
        for noise_scale in noise_scales:
            # print(f'Noise_Scale: {noise_scale}')
            model.set_noise_scale(noise_scale)
            
            acc1,acc5 = validate(val_loader,model,criterion)
            acc1_list[split_point].append(acc1)
            acc5_list[split_point].append(acc5)
        print(f' acc1 : {acc1_list[split_point]}')
        print(f' acc5 : {acc5_list[split_point]}')
    
    # visualization
    save_path = f'./privacy_aware_dsc/{args.save_dir}'
    os.makedirs(save_path, exist_ok=True)
    # Top1 Accuracy
    plt.figure(figsize=[16,6])
    for split_point in split_points:
        plt.plot(noise_scales,acc1_list[split_point])
    plt.axhline(baseline_acc1,color='midnightblue')

    plt.ylim(top=100)
    plt.legend(split_points,title="split point" ,fontsize="xx-large",title_fontsize="xx-large",loc='lower right')
    plt.xticks(noise_scales)
    plt.xlabel("σ:noise scale",fontsize=24)
    plt.ylabel("Top1-Accuracy[%]",fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22)
    plt.grid()
    plt.savefig(f"{save_path}/noise_added_acc1_list.png",bbox_inches='tight')
    plt.close()
    
    # Top5 Accuracy
    plt.figure(figsize=[16,6])
    for split_point in split_points:
        plt.plot(noise_scales,acc5_list[split_point])
    plt.axhline(baseline_acc5,color='midnightblue')

    plt.ylim(top=100)

    plt.legend(split_points,title="split point" ,fontsize="xx-large",title_fontsize="xx-large",loc='lower right')
    plt.xticks(noise_scales)
    plt.xlabel("σ:noise scale",fontsize=24)
    plt.ylabel("Top5-Accuracy[%]",fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22)
    plt.grid()
    plt.savefig(f"{save_path}/noise_added_acc5_list.png",bbox_inches='tight')
    plt.close()
    
def set_seed(seed):
    random.seed(seed)                  # Pythonの乱数
    np.random.seed(seed)              # NumPyの乱数
    torch.manual_seed(seed)           # PyTorchのCPU乱数
    torch.cuda.manual_seed(seed)      # PyTorchのGPU乱数
    torch.cuda.manual_seed_all(seed)  # マルチGPUのための乱数


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
    bound_threshold_mean = torch.mean(features_list,dim = 1)
    
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
            # print(f"Encountered BasicBlock at features.{i}")
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
        shuffle=False
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

if __name__ == '__main__':
    
    print("Program evaluation_with_noise.py execution has begun.", end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))
    
    original_stdout = sys.stdout
    args = parser.parse_args()
    save_path = f'./privacy_aware_dsc/{args.save_dir}'
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/evaluation_output.txt', 'w') as f:
        sys.stdout = f
        main()
    sys.stdout = original_stdout
    
    print(f"Program evaluation_with_noise.py execution has done.",end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))