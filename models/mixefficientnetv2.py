#https://www.kaggle.com/code/vikramsandu/efficientnetv2-from-scratch


# Import useful Modules
import torch
from torch import nn
from math import ceil
from torchinfo import summary
from . import quant_module as qm


__all__ = ["mixeffnet_v2_w2468a2468_100"]

Eff_V2_SETTINGS = {
    # expansion factor, k, stride, n_in, n_out, num_layers, use_fusedMBCONV
    's' : [
        [1, 3, 1, 24, 24, 2, True],
        [4, 3, 2, 24, 48, 4, True],
        [4, 3, 2, 48, 64, 4, True],
        [4, 3, 2, 64, 128, 6, False],
        [6, 3, 1, 128, 160, 9, False],
        [6, 3, 2, 160, 256, 15, False]
    ],
    
    'm' : [
        [1, 3, 1, 24, 24, 3, True],
        [4, 3, 2, 24, 48, 5, True],
        [4, 3, 2, 48, 80, 5, True],
        [4, 3, 2, 80, 160, 7, False],
        [6, 3, 1, 160, 176, 14, False],
        [6, 3, 2, 176, 304, 18, False],
        [6, 3, 1, 304, 512, 5, False]
    ],
    
    'l' : [
        [1, 3, 1, 32, 32, 4, True],
        [4, 3, 2, 32, 64, 7, True],
        [4, 3, 2, 64, 96, 7, True],
        [4, 3, 2, 96, 192, 10, False],
        [6, 3, 1, 192, 224, 19, False],
        [6, 3, 2, 224, 384, 25, False],
        [6, 3, 1, 384, 640, 7, False]
    ]
}

'''A simple Convolution + Batch Normalization + Activation Class'''


class BasicConvBnAct(nn.Module):
    
    def __init__(
        self,
        n_in, # in_channels
        n_out, # out_channels
        k_size = 3, # Kernel Size
        stride = 1, 
        padding = 0,
        groups = 1, 
        act = True, 
        bn = True, 
        bias = False,
        **kwargs
    ):
        super(BasicConvBnAct, self).__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = k_size, stride = stride,
                              padding = padding, groups = groups,bias = bias, **kwargs
                             )
        self.batch_norm = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        return x
    
class ConvBnAct(nn.Module):
    
    def __init__(
        self,
        conv_func,
        n_in, # in_channels
        n_out, # out_channels
        k_size = 3, # Kernel Size
        stride = 1, 
        padding = 0,
        groups = 1, 
        act = True, 
        bn = True, 
        bias = False,
        **kwargs
    ):
        super(ConvBnAct, self).__init__()
        
        self.conv = conv_func(n_in, 
                              n_out, 
                              kernel_size = k_size, 
                              stride = stride,
                              padding = padding,
                              groups = groups,
                              bias = bias,
                              **kwargs
                             )
        self.batch_norm = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        return x
    

#--------------------------------------------------------------------------------------------

'''Squeeze and Excitation Class'''

class SqueezeExcitation(nn.Module):
    
    def __init__(
        self,
        conv_func,
        n_in, # In_channels
        reduced_dim,
        **kwargs
    ):
        super(SqueezeExcitation, self).__init__()
      
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(conv_func(n_in, reduced_dim, kernel_size=1, bias=False, **kwargs),
                                   nn.SiLU(),
                                   conv_func(reduced_dim, n_in, kernel_size=1, bias=False, **kwargs),
                                   nn.Sigmoid()
                                   )
        
    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
            
        return x * y
    

#--------------------------------------------------------------------------------------

''' Stochastic Depth Class'''

class StochasticDepth(nn.Module):
    
    def __init__(
        self,
        survival_prob = 0.8
    ):
        super(StochasticDepth, self).__init__()
        
        self.p =  survival_prob
        
    def forward(self, x):
        
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
        
        return torch.div(x, self.p) * binary_tensor
    

#-------------------------------------------------------------------------------

'''MBCONV Class'''

class MBConvN(nn.Module):
    
    def __init__(
        self,
        conv_func,
        n_in, # In_channels
        n_out, # out_channels
        k_size = 3, # kernel_size
        stride = 1,
        expansion_factor = 4,
        reduction_factor = 4, # SqueezeExcitation Block
        survival_prob = 0.8, # StochasticDepth Block
        **kwargs
    ):
        super(MBConvN, self).__init__()
        reduced_dim = int(n_in//4)
        expanded_dim = int(expansion_factor * n_in)
        padding = (k_size - 1)//2
        
        self.use_residual = (n_in == n_out) and (stride == 1)
        self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(conv_func, n_in, expanded_dim, k_size = 1, **kwargs)
        self.depthwise_conv = ConvBnAct(conv_func, expanded_dim, expanded_dim,
                                        k_size, stride = stride,
                                        padding = padding, groups = expanded_dim, **kwargs
                                       )
        self.se = SqueezeExcitation(conv_func, expanded_dim, reduced_dim, **kwargs)
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = ConvBnAct(conv_func, expanded_dim, n_out, k_size = 1, act = False, **kwargs)
        
    def forward(self, x):
        
        residual = x.clone()
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = self.drop_layers(x)
            # x += residual
            x += self.expand.conv.quant_skip
            
        return x
    


#--------------------------------------------------------------------------------------

'''Fused-MBCONV Class'''

class FusedMBConvN(nn.Module):
    
    def __init__(
        self,
        conv_func,
        n_in, # In_channels
        n_out, # out_channels
        k_size = 3, # kernel_size
        stride = 1,
        expansion_factor = 4,
        reduction_factor = 4, # SqueezeExcitation Block
        survival_prob = 0.8, # StochasticDepth Block
        **kwargs
    ):
        super(FusedMBConvN, self).__init__()
        
        reduced_dim = int(n_in//4)
        expanded_dim = int(expansion_factor * n_in)
        padding = (k_size - 1)//2
        
        self.use_residual = (n_in == n_out) and (stride == 1)
        #self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded_dim, k_size = 1)
        self.conv = ConvBnAct(conv_func, n_in, expanded_dim,
                              k_size, stride = stride,
                              padding = padding, groups = 1, **kwargs
                             )
        #self.se = SqueezeExcitation(expanded_dim, reduced_dim)
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = nn.Identity() if (expansion_factor == 1) else ConvBnAct(conv_func, expanded_dim, n_out, k_size = 1, act = False, **kwargs)
        
    def forward(self, x):
        
        residual = x.clone()
        #x = self.conv(x)
        x = self.conv(x)
        #x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = self.drop_layers(x)
            # x += residual
            x += self.conv.conv.quant_skip
            
        return x

#-----------------------------------------------------------------------------------------------

class EfficientNetV2(nn.Module):
    
    def __init__(
    self,
    conv_func,
    version = 's',
    dropout_rate = 0.2,
    num_classes = 1000,
    **kwargs
    ):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.conv_func = conv_func
        super(EfficientNetV2, self).__init__()
        last_channel = 1280
        self.features = self._feature_extractor(conv_func, version, last_channel, **kwargs)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate, inplace = True),
            nn.Linear(last_channel, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        return x
        
    def _feature_extractor(self, conv_func, version, last_channel, **kwargs):
        
        # Extract the Config
        config = Eff_V2_SETTINGS[version]
        
        layers = []
        layers.append(BasicConvBnAct(3, config[0][3], k_size = 3, stride = 2, padding = 1))
        #in_channel = config[0][3]
        
        for (expansion_factor, k, stride, n_in, n_out, num_layers, use_fused) in config:
            
            if use_fused:
                layers += [FusedMBConvN(conv_func,
                                        n_in if repeat==0 else n_out, 
                                        n_out,
                                        k_size=k,
                                        stride = stride if repeat==0 else 1,
                                        expansion_factor=expansion_factor,
                                        **kwargs
                                       ) for repeat in range(num_layers)
                          ]
            else:
                
                layers += [MBConvN(conv_func,
                                   n_in if repeat==0 else n_out, 
                                   n_out,
                                   k_size=k,
                                   stride = stride if repeat==0 else 1,
                                   expansion_factor=expansion_factor,
                                   **kwargs
                                   ) for repeat in range(num_layers)
                      ]
                
        layers.append(ConvBnAct(conv_func, config[-1][4], last_channel, k_size = 1))   
            
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
                if layer_idx in [0, 4, 9, 19, 44, 59]:
                    loss += split_complexity_loss
                layer_idx += 1
        normalizer = 6 * 5
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


def mixeffnet_v2_w2468a2468_100(**kwargs):
    version = "s"
    return EfficientNetV2(qm.MixActivConv2d, version, num_classes=100,
                     wbits=[2, 4, 6, 8], abits=[2, 4, 6, 8], share_weight=True, **kwargs)
    
    
    
