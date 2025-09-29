from __future__ import print_function

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
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from utils import ssim
from Ginver.Model import InversionModelEffnet, InversionModelEffnet_test
os.chdir('./')
# EdMIPS/models ディレクトリへのパスを追加
sys.path.append('./models')
# from models.quant_efficientnet import BasicCNNBlock
import models as models

# カレントディレクトリを 'EdMIPS' に変更
os.chdir('.')

# cudnn.benchmark = True
torch.backends.cudnn.deterministic = True  # 再現性確保のための設定
torch.backends.cudnn.benchmark = False     # パフォーマンス向上を無効化

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--model_name',type=str)
parser.add_argument('--mixed_arch',type=str,help='量子化精度決定済み未学習モデルのパス,U8の時は空白でOK',default="")
parser.add_argument('--quant_arch',type=str,help='学習モデルのパス')
parser.add_argument('--inv_arch',type=str)
parser.add_argument('-b','--batch_size',type=int)
# parser.add_argument('-w','--workers',type=int)
parser.add_argument('-i','--image_size',type=int,default=256)
parser.add_argument('--split_point',type = int)
parser.add_argument('--noise_flag',type=bool,default = False)
parser.add_argument('--noise_scale',type=int,default=0)
parser.add_argument('--env',type=str,default='blackbox',help='blackbox or whitebox')
parser.add_argument('--save_dir',type=str)

torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.cuda.manual_seed_all(2)
np.random.seed(2)
random.seed(2)

args = parser.parse_args()
IMAGE_SIZE = args.image_size
assert args.env =='blackbox' or args.env == 'whitebox' ,'You have to chose blackbox or whitebox'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'Using device : {device}')
def main():
    args = parser.parse_args()
    print(args)
    os.makedirs('./MIA/'+ args.env +'/' + args.save_dir, exist_ok=True)
    # os.makedirs('./MIA/'+ args.env +'/' + args.save_dir, exist_ok=True)

    train_loader,val_loader = load_cifar10(batch=args.batch_size)

    model = models.__dict__[args.model_name](args.mixed_arch)
    checkpoint = torch.load(args.quant_arch)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.change_learning_method(noisy_flag = False)
    model.set_splitting_point(args.split_point)
    if args.noise_flag:
        print('=> adding noise inference')
        B = find_B(train_loader,model,[args.split_point])
        print(f'Bound threshold median list : {B}')
        model.set_B(B[0])
        model.change_learning_method(noisy_flag = True)
        model.set_noise_scale(args.noise_scale)
    # EfficientNet-B0のみ対応
    if args.split_point == 2:
        input_channels = 24
    elif args.split_point == 4:
        input_channels = 40
    elif args.split_point == 6:
        input_channels = 80
    elif args.split_point == 1:
        input_channels = 16
    
    inversion = InversionModelEffnet(input_channels=input_channels,output_channels=3,image_size=IMAGE_SIZE).to(device)
    # inversion = InversionModelEffnet_test(input_channels=input_channels,nc=1, ngf=128, nz=128,ig=IMAGE_SIZE).to(device)

    # Load inversion
    try:
        checkpoint = torch.load(args.inv_arch)
        inversion.load_state_dict(checkpoint['model'])
        begin_epoch = checkpoint['epoch']
        best_mse_loss = checkpoint['best_mse_loss']
        print("=> loaded inversion checkpoint '{}' (epoch {}, best_mse_loss {:.4f})".format(args.inv_arch, begin_epoch, best_mse_loss))
    except:
        print("=> load inversion checkpoint '{}' failed".format(args.inv_arch))
        begin_epoch = 0
        # sys.exit()

    mse_loss,ssim_loss = test(model, inversion, device, val_loader)
    print(' MSE : {} '.format(mse_loss))
    print(' SSIM : {}'.format(ssim_loss))
    record(model, inversion, args, device, val_loader, begin_epoch, args.save_dir  + "_same", 8, mse_loss)
    # record(model, inversion, args, device, test_loader2, epoch, flag+"_differ", 32, mse_loss)

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
                         ])
                         ),
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
                         ])
                         ),
        batch_size=batch,
        shuffle=False
    )

    return train_loader,val_loader

    
# test
def test(model, inversion, device, data_loader):
    model.eval()
    inversion.eval()
    mse_loss = 0
    ssim_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # cifar-10,100のみ
            data = F.interpolate(data,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
            prediction = model.forward_for_MIA(data)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='mean').item()
            ssim_loss+= ssim(reconstruction,data)
    mse_loss /= len(data_loader)
    ssim_loss /=  len(data_loader)
    # print('\nTest inversion model on test set: Average MSE loss: {:.4f}\n'.format(mse_loss))
    return mse_loss,ssim_loss

# record
def record(model, inversion, args, device, data_loader, epoch, msg, num, loss):
    model.eval()
    inversion.eval()

    plot = True
    if args.noise_flag:
        strnoise=f'noise{args.noise_scale}'
    else:
        strnoise=''

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # cifar-10,100のみ
            data = F.interpolate(data,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
            prediction = model.forward_for_MIA(data)
            reconstruction = inversion(prediction)

            truth = data[0:num]
            inverse = reconstruction[0:num]
            out = torch.cat((inverse, truth))

            vutils.save_image(out, './MIA/' + args.env+'/' + args.save_dir +'/{}_{}_{:.4f}_{}.png'.format(msg.replace(" ", ""), epoch, loss,strnoise), normalize=False)
            vutils.save_image(inverse, './MIA/' + args.env + '/' + args.save_dir + f'/inverse_{strnoise}.png', normalize=False)
            vutils.save_image(truth, './MIA/' + args.env + '/' + args.save_dir + '/origin.png',normalize=False)
            break

def find_B(train_loader,model,split_points):
    model.to(device)
    # print(model.noisy_flag)
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            # cifar-10,100のみ
            images = F.interpolate(images,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
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


if __name__ == '__main__':
    
    print("Program model_inversion_attack.py execution has begun.", end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))
    
    original_stdout = sys.stdout
    args = parser.parse_args()
    if args.noise_flag:
        strnoise=f'_noise{args.noise_scale}'
    else:
        strnoise=''
    save_path = './MIA/' + args.env + '/' + args.save_dir
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/MIA{strnoise}_output.txt', 'w') as f:
        sys.stdout = f
        print(f"beginning time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}\n")
        main()
        print(f"\nend time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}")
    sys.stdout = original_stdout
    
    print(f"Program model_inversion_attack.py execution has done.",end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))