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

import utils
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
parser.add_argument('-b','--batch_size',type=int)
# parser.add_argument('-w','--workers',type=int)
parser.add_argument('-i','--image_size',type=int,default=256)
parser.add_argument('--split_point',type = int)
parser.add_argument('--noise_flag',type=bool,default = False)
parser.add_argument('--noise_scale',type=int,default=0)
parser.add_argument('-e','--epoch',type = int)
parser.add_argument('-s','--step',type=float,default = 0.001)
parser.add_argument('-l','--lamb',type=float,default= 1.0 )
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
    # os.makedirs('./MIA/'+ args.env +'/' + args.save_dir, exist_ok=True)
    os.makedirs('./MIA/'+ args.env +'/' + args.save_dir, exist_ok=True)

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
    optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    # Load inversion
    # path = './MIA/'+args.env + '/' + args.save_dir +'/inversion.pth'
    path = './MIA/'+args.env + '/' + args.save_dir
    best_mse_loss = 0.0500
    begin_epoch = 0

    # try:
    #     checkpoint = torch.load(path)
    #     inversion.load_state_dict(checkpoint['model'])
    #     begin_epoch = checkpoint['epoch']
    #     best_mse_loss = checkpoint['best_mse_loss']
    #     print("=> loaded inversion checkpoint '{}' (epoch {}, best_mse_loss {:.4f})".format(path, begin_epoch, best_mse_loss))
    # except:
    #     print("=> load inversion checkpoint '{}' failed".format(path))
        
    target_mse_loss = best_mse_loss - 0.0005
    mse_loss,TV_loss = test(model, inversion, device, val_loader)
    print('Before Train')
    print(' Total Loss {} '.format(mse_loss+TV_loss))
    print('  MSE Loss {} '.format(mse_loss))
    print('  TV Loss {} '.format(TV_loss))
    for epoch in range(begin_epoch+1, args.epoch+1):
        adjust_learning_rate(optimizer,epoch-1,0.0002)
        train(model, inversion, args, device, train_loader, optimizer, epoch)
        mse_loss,TV_loss = test(model, inversion, device, val_loader)

        print(' Total Loss {} '.format(mse_loss+TV_loss))
        print('  MSE Loss {} '.format(mse_loss))
        print('  TV Loss {} '.format(TV_loss))

        if mse_loss < target_mse_loss:
            target_mse_loss = mse_loss - 0.0005
            best_mse_loss = mse_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            torch.save(state, path + '/best_inversion.pth')
            print('-Test inversion model on test set: Average MSE loss: {}_{:.4f}\n'.format(epoch, mse_loss))
            # record(model, inversion, args, device, val_loader, epoch, args.save_dir +"_same", 32, mse_loss)
            # record(model, inversion, args,  device, test_loader2, epoch, flag+"_differ", 32, mse_loss)
        if epoch == args.epoch :
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            torch.save(state, path + '/final_inversion.pth')
            # record(model, inversion, args, device, val_loader, epoch, args.save_dir  + "_same", 32, mse_loss)
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
                         ])),
        batch_size=batch,
        shuffle=False
    )

    return train_loader,val_loader

def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
    step_epoch = 10
    lr = start_lr * (0.1 ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def train(model, inversion, args, device, data_loader, optimizer, epoch):
    model.eval()
    inversion.train()
    
    print('Train epoch {} '.format(epoch))
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        # cifar-10,100のみ
        data = F.interpolate(data,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = model.forward_for_MIA(data).detach()
        reconstruction = inversion(prediction)
        # blackbox
        if args.env == 'blackbox':
            with torch.no_grad():
                grad = torch.zeros_like(reconstruction)
                num = 0
                for j in range(50):
                    random_direction = torch.randn_like(reconstruction)
            
                    new_pic1 = reconstruction + args.step * random_direction
                    new_pic2 = reconstruction - args.step * random_direction
            
                    target1 = model.forward_for_MIA(new_pic1)
                    target2 = model.forward_for_MIA(new_pic2)
            
                    loss1 = F.mse_loss(target1, prediction)
                    loss2 = F.mse_loss(target2, prediction)
            
                    num = num + 2
                    grad = loss1 * random_direction + grad
                    grad = loss2 * -random_direction + grad
            
                grad = grad / (num * args.step)
                # grad = grad.squeeze(dim=0)
            loss_TV = args.lamb * utils.TV(reconstruction)
            loss_TV.backward(retain_graph=True)
            reconstruction.backward(grad)
            optimizer.step()
        
        # whitebox
        elif args.env == 'whitebox':
            reconstruction_prediction = model.forward_for_MIA(reconstruction)
            loss_TV = args.lamb * utils.TV(reconstruction)
            loss_mse = F.mse_loss(reconstruction_prediction, prediction)
            loss = loss_mse + loss_TV
            loss.backward()
            optimizer.step()

# test
def test(model, inversion, device, data_loader):
    model.eval()
    inversion.eval()
    mse_loss = 0
    TV_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # cifar-10,100のみ
            data = F.interpolate(data,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
            prediction = model.forward_for_MIA(data)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='mean').item()
            TV_loss += utils.TV(reconstruction)

    mse_loss /= len(data_loader)
    TV_loss  /= len(data_loader)
    # print('\nTest inversion model on test set: Average MSE loss: {:.4f}\n'.format(mse_loss))
    return mse_loss,TV_loss

# record
def record(model, inversion, args, device, data_loader, epoch, msg, num, loss):
    model.eval()
    inversion.eval()

    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # cifar-10,100のみ
            data = F.interpolate(data,size=(IMAGE_SIZE,IMAGE_SIZE),mode='bicubic')
            prediction = model.forward_for_MIA(data)
            reconstruction = inversion(prediction)

            # truth = data[0:num]
            # inverse = reconstruction[0:num]
            # out = torch.cat((inverse, truth))
            # vutils.save_image(out, './MIA/' + args.env+'/' + args.save_dir +'/{}_{}_{:.4f}.png'.format(msg.replace(" ", ""), epoch, loss), normalize=False)
            if epoch != args.epoch-1:
                vutils.save_image(reconstruction[0], './MIA/' + args.env + '/' + args.save_dir + '/final_inverse.png', normalize=False)
                vutils.save_image(data[0], './MIA/' + args.env + '/' + args.save_dir + '/origin.png',normalize=False)
            if epoch == args.epoch-1:
                vutils.save_image(reconstruction[0], './MIA/' + args.env + '/' + args.save_dir + + '/final_epoch.png',
                                normalize=False)
                vutils.save_image(data[0], './MIA/' + args.env + '/' + args.save_dir + + '/origin.png',normalize=False)
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
    bound_threshold_mean = torch.mean(features_list,dim = 1)

    # check 
    # print(f"bound threshold median : {bound_threshold_medi}")
    # print(f"bound threshold mean : {bound_threshold_mean}")

    return bound_threshold_medi

if __name__ == '__main__':
    
    print("Program model_inversion_train.py execution has begun.", end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))
    
    original_stdout = sys.stdout
    args = parser.parse_args()
    save_path = './MIA/' + args.env + '/' + args.save_dir
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/inv_train_output.txt', 'w') as f:
        sys.stdout = f
        print(f"beginning time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}\n")
        main()
        print(f"\nend time : {datetime.datetime.now().strftime('%Y-%m%d_%H:%M')}")
    sys.stdout = original_stdout
    
    print(f"Program model_inversion_train.py execution has done.",end=" ")
    print(datetime.datetime.now().strftime('%Y-%m%d_%H:%M'))