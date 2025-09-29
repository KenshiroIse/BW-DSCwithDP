import torch
import torch.nn as nn
from math import ceil
from torchinfo import summary
from . import quant_module as qm
# import quant_module as qm

__all__ = ['quanteffnet_w8a8_chan', 'quanteffnet_cfg', "quanteffnet_w32a32_chan", 
           "quanteffnet_w2a2_chan", "quanteffnet_w4a4_chan", "quanteffnet_w8a8", 
           "quanteffnet_cfg_2468", "quanteffnet_w4a4", "quanteffnet_w3a3", "quanteffnet_w2a2",
           "quanteffnet_cfg_2468_b3", "quanteffnet_w8a8_b3","quanteffnet_cfg_2468_forlossynet",
           "quanteffnet_cfg_2468_with_DP","quanteffnet_w8a8_with_DP","quanteffnet_cfg_2468_b3_with_DP",
           "quanteffnet_w8a8_b3_with_DP"]

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
        self, conv_func, in_channels, out_channels, wbit, abit,
        kernel_size, stride, padding, groups=1, **kwargs
    ):
        super(CNNBlock, self).__init__()
        self.cnn = conv_func(
            in_channels,
            out_channels,
            wbit,
            abit,
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
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, conv_func, archws, archas, in_channels, reduced_dim, **kwargs):
        super(SqueezeExcitation, self).__init__()
        assert len(archas) == 2
        assert len(archws) == 2
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            conv_func(
                in_channels, reduced_dim, archws[0], archas[0], kernel_size=1, bias=False, **kwargs
            ),  # C x 1 x 1 -> C_reduced x 1 x 1
            nn.SiLU(),  # SiLU <-> Swish
            conv_func(
                reduced_dim, in_channels, archws[1], archas[1], kernel_size=1, bias=False, **kwargs
            ),  # C_reduced x 1 x 1 -> C x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            conv_func,
            archws, 
            archas,
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

        i = 0
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)
        if self.expand:
            assert len(archas) == 5
            assert len(archws) == 5
        else:
            assert len(archas) == 4
            assert len(archws) == 4
        if self.expand:
            self.expand_conv = CNNBlock(
                conv_func, in_channels, hidden_dim, archws[i], archas[i], kernel_size=3, stride=1, padding=1, **kwargs,
            )
            i += 1

        self.conv = nn.Sequential(
            CNNBlock(
                conv_func, hidden_dim, hidden_dim, archws[i], archas[i], kernel_size, stride, padding, groups=hidden_dim, **kwargs,
            ),
            SqueezeExcitation(conv_func, archws[i+1:i+3], archas[i+1:i+3], hidden_dim, reduced_dim, **kwargs,),
            conv_func(hidden_dim, out_channels, archws[i+3], archas[i+3], kernel_size=1, bias=False, **kwargs,),
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

    def __init__(self, conv_func, version, archws, archas, num_classes=1000, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        self.archws = archws
        self.archas = archas
        # assert len(archas) == 80
        # assert len(archws) == 80
        self.conv_func = conv_func
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate, res = self.calculate_factors(version)
        self.res = res
        last_channel = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(conv_func, width_factor, depth_factor, last_channel, archws, archas, **kwargs)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate, res
    
    def create_features(self, conv_func, width_factor, depth_factor, last_channel, archws, archas, **kwargs):
        channels = int(32 * width_factor)
        features = [BasicCNNBlock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels
        i = 0
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            if expand_ratio == 1:
                j = 4
            else:
                j = 5
            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        conv_func,
                        archws[i:i+j],
                        archas[i:i+j],
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
                i += j
        # assert i == 79
        features.append(
            CNNBlock(conv_func, in_channels, last_channel, archws[j], archas[j], kernel_size=1, stride=1, padding=0,  **kwargs)
        )
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))

    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                bitops = size_product * m.abit * m.wbit
                bita = m.memory_size.item() * m.abit
                bitw = m.param_size * m.wbit
                # weight_shape = list(m.conv.weight.shape)
                # print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                #       'param: {:.3f}M * {}'.format(layer_idx, weight_shape, size_product, m.abit,
                #                                    m.wbit, memory_size, m.abit, m.param_size, m.wbit))
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_bitops, sum_bita, sum_bitw

import torch.nn.functional as F
class LossyNet(EfficientNet):
    def __init__(self, conv_func, version, archws, archas, dropout, num_classes=1000,  **kwargs):
        super().__init__(conv_func, version, archws, archas, num_classes=num_classes, **kwargs)
        self.dropout = dropout
    def forward(self,x):
        p=self.dropout
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i == 6:
                x = F.dropout(x,p=p,training=True)*(1-p)
        x = self.pool(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x
    
class EfficientNet_with_DP(EfficientNet):
    def __init__(self, conv_func, version, archws, archas, num_classes,  **kwargs):
        super().__init__(conv_func, version, archws, archas, num_classes=num_classes, **kwargs)
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.noise_scale = 0
        self.splitting_point = -1
        self.noisy_flag = False
        self.B = 0
    def forward(self,x):
        n = self.noise_scale
        for i in range(len(self.features)):
            
            x = self.features[i](x)

            if self.noisy_flag == True and i == self.splitting_point:
                # x = torch.where(x > self.B,self.B,x)
                if torch.max(x) > self.B:
                    x = x * (self.B/torch.max(x))
                laplace_dist = torch.distributions.Laplace(loc = 0, scale = self.B.item()/n)
                noise = laplace_dist.sample(sample_shape=x.size()).to(self.device)
                
                x = x + noise
            
        x = self.pool(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x
    def forward_for_MIA(self,x):
        n = self.noise_scale
        for i in range(len(self.features)):
            
            x = self.features[i](x)

            if  i == self.splitting_point:
                if self.noisy_flag:
                    if torch.max(x) > self.B:
                        x = x * (self.B/torch.max(x))
                    laplace_dist = torch.distributions.Laplace(loc = 0, scale = self.B.item()/n)
                    noise = laplace_dist.sample(sample_shape=x.size()).to(self.device)
                    x = x + noise
                    return x
                else:
                    return x
            
        x = self.pool(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x
    def feature_extractor(self,x,split_points):
        feature_list = []
        for i in range(len(self.features)):
            
            x = self.features[i](x)
            if i in split_points:
                feature_list.append(torch.max(torch.flatten(x,1,-1),dim=1).values)
            
        # x = self.pool(x)
        # x = self.classifier(x.view(x.shape[0], -1))
        return feature_list
    def change_learning_method(self,noisy_flag):
        # methodがTrueならNoisy,FalseならClean
        self.noisy_flag = noisy_flag
    def set_splitting_point(self,splitting_point):
        self.splitting_point = splitting_point
    def set_B(self,bound_threshold):
        self.B = bound_threshold.to(self.device)  #BだけGPUに移さないと動かなかったため(torch.where(x > self.B,self.B,x)が原因)
    def set_noise_scale(self,noise_scale):
        self.noise_scale = noise_scale


def _load_arch(arch_path, names_nbits):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    for name in names_nbits.keys():
        best_arch[name], worst_arch[name] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name in names_nbits.keys():
            alpha = params.cpu().numpy()
            assert names_nbits[name] == alpha.shape[0]
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())

    return best_arch, worst_arch
    

def quanteffnet_w8a8(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [8] *80
    archws = [8] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)

def quanteffnet_w4a4(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [4] *80
    archws = [4] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)

def quanteffnet_w3a3(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [3] *80
    archws = [3] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)

def quanteffnet_w2a2(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [2] *80
    archws = [2] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)


def quanteffnet_w8a8_chan(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [8] *80
    archws = [8] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantMixActivChanConv2d, version, archws, archas, num_classes=100, **kwargs)

def quanteffnet_cfg(arch_cfg_path, **kwargs):
    wbits, abits = [1, 2, 3, 4], [2, 3, 4]
    version = "b0"
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)


def quanteffnet_cfg_2468(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 6, 8],  [2, 4, 6, 8]
    version = "b0"
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 80
    assert len(archws) == 80
    # return LossyNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)


def quanteffnet_cfg_2468_b3(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 6, 8],  [2, 4, 6, 8]
    version = "b3"
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    # assert len(archas) == 80
    # assert len(archws) == 80
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)

 
def quanteffnet_w8a8_b3(arch_cfg_path, **kwargs):
    version = "b3"
    archas = [8] *129
    archws = [8] *129
    assert len(archas) == 129
    assert len(archws) == 129
    return EfficientNet(qm.QuantActivConv2d, version, archws, archas, num_classes=100, **kwargs)

 
def quanteffnet_w32a32_chan(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [32] *80
    archws = [32] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantMixActivChanConv2d, version, archws, archas, num_classes=100, **kwargs)

def quanteffnet_w2a2_chan(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [2] *80
    archws = [2] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantMixActivChanConv2d, version, archws, archas, num_classes=100, **kwargs)

def quanteffnet_w4a4_chan(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [4] *80
    archws = [4] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet(qm.QuantMixActivChanConv2d, version, archws, archas, num_classes=100, **kwargs)

def quanteffnet_cfg_2468_forlossynet(arch_cfg_path, dropout,**kwargs):
    wbits, abits = [2, 4, 6, 8],  [2, 4, 6, 8]
    version = "b0"
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 80
    assert len(archws) == 80
    return LossyNet(qm.QuantActivConv2d, version, archws, archas, dropout, num_classes=100 , **kwargs)


# ours
def quanteffnet_cfg_2468_with_DP(arch_cfg_path,**kwargs):
    wbits, abits = [2, 4, 6, 8],  [2, 4, 6, 8]
    version = "b0"
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet_with_DP(qm.QuantActivConv2d, version, archws, archas, num_classes=10 , **kwargs)

def quanteffnet_w8a8_with_DP(arch_cfg_path, **kwargs):
    version = "b0"
    archas = [8] *80
    archws = [8] *80
    assert len(archas) == 80
    assert len(archws) == 80
    return EfficientNet_with_DP(qm.QuantActivConv2d, version, archws, archas, num_classes=10, **kwargs)

def quanteffnet_cfg_2468_b3_with_DP(arch_cfg_path, **kwargs):
    wbits, abits = [2, 4, 6, 8],  [2, 4, 6, 8]
    version = "b3"
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]

    return EfficientNet_with_DP(qm.QuantActivConv2d, version, archws, archas, num_classes=10, **kwargs)

def quanteffnet_w8a8_b3_with_DP(arch_cfg_path, **kwargs):
    version = "b3"
    archas = [8] *129
    archws = [8] *129
    assert len(archas) == 129
    assert len(archws) == 129
    return EfficientNet_with_DP(qm.QuantActivConv2d, version, archws, archas, num_classes=10, **kwargs)