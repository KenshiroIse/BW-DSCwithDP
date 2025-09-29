# https://pystyle.info/pytorch-vgg/

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
from . import quant_module as qm
# import quant_module as qm

__all__ = [
      'mixvgg11_w1234a234', 'mixvgg13_w1234a234', 'mixvgg16_w1234a234', 
      'mixvgg19_w1234a234', 'mixvgg16_w1234a234_100',
]

def conv3x3(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "3x3 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, **kwargs)

class VGG(nn.Module):
    def __init__(self, conv_func, num_classes=1000, **kwargs):
        super(VGG, self).__init__()
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64, affine=True)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)          
        self.conv_func = conv_func
        self.features = self._make_layers(**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.bn0(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, **kwargs):
        layers = []
        in_channels = 3
        for idx, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.BatchNorm2d(in_channels, affine=True)]
            elif idx == 0:
                in_channels = x
                pass
            else:
                layers += [#nn.BatchNorm2d(in_channels, affine=True),  #inceptionのようなquantconvモジュールを作るのでもあり（batchが入ったやつ）
                           conv3x3(self.conv_func, in_channels, x, **kwargs),
                           nn.BatchNorm2d(x, affine=True),]
                in_channels = x
        return nn.Sequential(*layers)

    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                complexity_loss, _ = m.complexity_loss()
                loss += complexity_loss
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss
    
    def split_complexity_loss(self):
        loss = 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                _, split_complexity_loss = m.complexity_loss()
                if layer_idx == 6:
                    loss += split_complexity_loss
                layer_idx += 1
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw



class myVGG(nn.Module):

    def __init__(self, conv_func, num_classes=100, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.conv_func = conv_func
        super(myVGG, self).__init__()

        self.conv01 = nn.Conv2d(3, 64, 3, bias=False)
        self.conv02 = conv_func(64, 64, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv03 = conv_func(64, 128, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.conv04 = conv_func(128, 128, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv05 = conv_func(128, 256, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.conv06 = conv_func(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.conv07 = conv_func(256, 256, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv08 = conv_func(256, 512, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.conv09 = conv_func(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.conv10 = conv_func(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = conv_func(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.conv12 = conv_func(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.conv13 = conv_func(512, 512, kernel_size=3, stride=1,
                     padding=1, bias=False, **kwargs)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.avepool1 = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = self.pool1(x)

        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = self.pool2(x)

        x = F.relu(self.conv05(x))
        x = F.relu(self.conv06(x))
        x = F.relu(self.conv07(x))
        x = self.pool3(x)

        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool5(x)

        x = self.avepool1(x)

        # 行列をベクトルに変換
        x = x.view(-1, 512 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                complexity_loss, _ = m.complexity_loss()
                loss += complexity_loss
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss
    
    def split_complexity_loss(self):
        loss = 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                _, split_complexity_loss = m.complexity_loss()
                if layer_idx == 6:
                    loss += split_complexity_loss
                layer_idx += 1
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw


cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def mixvgg11_w1234a234(**kwargs):
    cfg = cfgs['VGG11']
    return VGG(qm.MixActivConv2d, cfg=cfg, wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)

def mixvgg13_w1234a234(**kwargs):
    cfg = cfgs['VGG13']
    return VGG(qm.MixActivConv2d, cfg=cfg, wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)

def mixvgg16_w1234a234(**kwargs):
    cfg = cfgs['VGG16']
    return VGG(qm.MixActivConv2d, cfg=cfg, wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)

def mixvgg19_w1234a234(**kwargs):
    cfg = cfgs['VGG19']
    return VGG(qm.MixActivConv2d, cfg=cfg, wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)

def mixvgg16_w1234a234_100(**kwargs):
    return myVGG(qm.MixActivConv2d, wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)


# if __name__ == '__main__':
#     model = mixvgg11_w1234a234()
#     summary(model, input_size=(1, 3, 224, 224))

