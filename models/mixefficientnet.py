# https://www.youtube.com/watch?v=fR_0o25kigM&list=WL&index=6&t=2070s&ab_channel=AladdinPersson

import torch
import torch.nn as nn
from math import ceil
from torchinfo import summary
from . import quant_module as qm
# import quant_module as qm

__all__ = ['mixeffnet_b0_w1234a234', 'mixeffnet_b0_w1234a234_100', 'mixeffnet_b0_w248a248_chan',"mixeffnet_b0_w2468a2468_100",
           'mixeffnet_b0_w2468a2468_cifar10', 'mixeffnet_b3_w2468a2468_100','mixeffnet_b3_w2468a2468_cifar10' ]

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class BasicCNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicCNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.silu = nn.SiLU()  # SiLU <-> Swish
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.silu(x)

class CNNBlock(nn.Module):
    def __init__(
        self, conv_func, in_channels, out_channels, 
        kernel_size, stride, padding, groups=1, **kwargs
    ):
        super(CNNBlock, self).__init__()
        self.cnn = conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            **kwargs,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        out = self.cnn(x)
        out = self.silu(self.bn(out))
        return out
    # def forward(self, x):
    #     return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, conv_func, in_channels, reduced_dim, **kwargs):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            conv_func(
                in_channels, reduced_dim, kernel_size=1, bias=False, **kwargs
            ),  # C x 1 x 1 -> C_reduced x 1 x 1
            nn.SiLU(),  # SiLU <-> Swish
            conv_func(
                reduced_dim, in_channels, kernel_size=1, bias=False, **kwargs
            ),  # C_reduced x 1 x 1 -> C x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            conv_func,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4,  # squeeze excitation
            survival_prob=0.8,  # for stochastic depth
             **kwargs,
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                conv_func, in_channels, hidden_dim, kernel_size=3, stride=1, padding=1, **kwargs,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                conv_func, hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, **kwargs,
            ),
            SqueezeExcitation(conv_func, hidden_dim, reduced_dim, **kwargs,),
            conv_func(hidden_dim, out_channels, kernel_size=1, bias=False, **kwargs,),
            nn.BatchNorm2d(out_channels,),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        
        if self.use_residual:
            # return self.stochastic_depth(self.conv(x)) + inputs
            if self.expand:
                return self.stochastic_depth(self.conv(x)) + self.expand_conv.cnn.quant_skip
            else:
                return self.stochastic_depth(self.conv(x)) + self.conv[0].cnn.quant_skip
        else:
            return self.conv(x)
        



class EfficientNet(nn.Module):

    def __init__(self, conv_func, version, num_classes=1000, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.conv_func = conv_func
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate, res = self.calculate_factors(version)
        self.res = res
        last_channel = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(conv_func, width_factor, depth_factor, last_channel, **kwargs)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate, res
    
    def create_features(self, conv_func, width_factor, depth_factor, last_channel, **kwargs):
        channels = int(32 * width_factor)
        features = [BasicCNNBlock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        conv_func,
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                         **kwargs,
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(conv_func, in_channels, last_channel, kernel_size=1, stride=1, padding=0,  **kwargs)
        )
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
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
    
    def split_complexity_loss(self, bottlenecks_list):
        loss = 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                # どうしてm.complexity_loss()の返り値が2つある？→quant_module.pyにある
                
                _, split_complexity_loss = m.complexity_loss()
                if layer_idx in bottlenecks_list: 
                    loss += split_complexity_loss
                layer_idx += 1
        normalizer = len(bottlenecks_list) * 5 #2,4,6,8ビットの平均
        loss /= normalizer
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



def mixeffnet_b0_w1234a234(**kwargs):
    version = "b0"
    return EfficientNet(qm.MixActivConv2d, version, num_classes=1000,
                     wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)

def mixeffnet_b0_w1234a234_100(**kwargs):
    version = "b0"
    return EfficientNet(qm.MixActivConv2d, version, num_classes=100,
                     wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)
    
def mixeffnet_b0_w2468a2468_100(**kwargs):
    version = "b0"
    return EfficientNet(qm.MixActivConv2d, version, num_classes=100,
                     wbits=[2, 4, 6, 8], abits=[2, 4, 6, 8], share_weight=True, **kwargs)

def mixeffnet_b0_w248a248_chan(**kwargs):
    version = "b0"
    return EfficientNet(qm.MixActivChanConv2d, version, num_classes=100,
                     wbits=[1, 2, 3, 4], abits=[2, 3, 4], share_weight=True, **kwargs)


def mixeffnet_b3_w2468a2468_100(**kwargs):
    version = "b3"
    return EfficientNet(qm.MixActivConv2d, version, num_classes=100,
                     wbits=[2, 4, 6, 8], abits=[2, 4, 6, 8], share_weight=True, **kwargs)
    
def mixeffnet_b0_w2468a2468_cifar10(**kwargs):
    version = "b0"
    return EfficientNet(qm.MixActivConv2d, version, num_classes=10,
                        wbits=[2, 4, 6, 8], abits=[2, 4, 6, 8], share_weight=True, **kwargs)
def mixeffnet_b3_w2468a2468_cifar10(**kwargs):
    version = "b3"
    return EfficientNet(qm.MixActivConv2d, version, num_classes=10,
                        wbits=[2, 4, 6, 8], abits=[2, 4, 6, 8], share_weight=True, **kwargs)
        
    
    
    # @classmethod
    # def get_image_size(cls, model_name):
    #     """Get the input image size for a given efficientnet model.

    #     Args:
    #         model_name (str): Name for efficientnet.

    #     Returns:
    #         Input image size (resolution).
    #     """
    #     cls._check_model_name_is_valid(model_name)
    #     _, _, res, _ = efficientnet_params(model_name)
    #     return res
    
    #image_size = EfficientNet.get_image_size(args.arch)