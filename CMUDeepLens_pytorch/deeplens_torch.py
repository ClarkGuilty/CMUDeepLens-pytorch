"""
Created on Mon Feb 13 00:38:51 2023

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import torch
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.nn.functional as F

from math import ceil

class Conv2dSame(nn.Conv2d):
    "https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351"
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
def identity_func(x):
    return x

def get_output_shape(model, image_dim):
    "https://stackoverflow.com/a/62197038"
    return model(torch.rand(*(image_dim))).data.shape

class Bottleneck(nn.Module): #Inspired by the official implementation.
    def __init__(self, infeatures, outfeatures, downsample=False,
                 dilation=1, norm_layer=None, prevfeatures = 1,
                 preactivated=False, non_linearity=F.elu,verbose=False):
        super(Bottleneck, self).__init__()
        
        self.verbose = verbose
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        stride = 2 if downsample else 1
        self.increase_dim = outfeatures != prevfeatures
        
        self.non_linearity = non_linearity
        self.preactivated = preactivated
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if not preactivated:
            self.bn0 = norm_layer(prevfeatures,
                              momentum=0.01,
                              eps=1e-3,
                              )
        
        self.conv1 = Conv2dSame(prevfeatures, infeatures, kernel_size=1,
                         stride=stride, 
                         bias=False)
        
        self.bn1 = norm_layer(infeatures,
                              momentum=0.01,
                              eps=1e-3)
        
        self.conv2 = nn.Conv2d(infeatures, infeatures, kernel_size=3, stride=1,
                         bias=False, padding='same')
        self.bn2 = norm_layer(infeatures,
                              momentum=0.01,
                              eps=1e-3)
        
        self.conv3 = nn.Conv2d(infeatures, outfeatures, kernel_size=1,
                               padding='same',stride=1, bias=True)
        
        self.downsample = downsample
        
        if self.downsample or self.increase_dim:
            self.conv_skip = Conv2dSame(infeatures, outfeatures, kernel_size=1,
                              stride=stride, bias=True)
        
    def forward(self, x_input):
        if self.preactivated:
            x_in = x_input
        else:
            x_in = self.bn0(x_input)
            x_in = self.non_linearity(x_in)
            
        print('x_input.shape=',x_input.shape) if self.verbose else False
        out = self.conv1(x_input)
        out = self.bn1(out)
        out = self.non_linearity(out)
        print('conv1.shape=',out.shape) if self.verbose else False
    
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.non_linearity(out)
        print('conv2.shape=',out.shape) if self.verbose else False
    
        out = self.conv3(out)
        print('conv3.shape=',out.shape) if self.verbose else False
        
        if self.downsample or self.increase_dim:
            x_shortcut = self.conv_skip(x_in)
        else:
            x_shortcut = x_input
            
        out += x_shortcut
    
        return out

class CMUDeepLens(nn.Module):
    def __init__(self, nchannels = 1, image_size=66,
                 non_linearity=F.elu,verbose=True):
        super(CMUDeepLens, self).__init__()
        self.verbose=verbose
        self.non_linearity = non_linearity
        
        self.conv0 = nn.Conv2d(nchannels, 32, kernel_size=7, stride=1,
                         groups=1, dilation=1,padding='same',bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        
        self.nn1a = Bottleneck(16,32,preactivated=True,prevfeatures=32)
        self.nn1b = Bottleneck(16,32,prevfeatures=32)
        self.nn1c = Bottleneck(16,32,prevfeatures=32)
        
        self.nn2a = Bottleneck(32,64,downsample=True,prevfeatures=32)
        self.nn2b = Bottleneck(32,64,prevfeatures=64)
        self.nn2c = Bottleneck(32,64,prevfeatures=64)
        
        self.nn3a = Bottleneck(64,128,downsample=True,prevfeatures=64)
        self.nn3b = Bottleneck(64,128,prevfeatures=128)
        self.nn3c = Bottleneck(64,128,prevfeatures=128)
        
        self.nn4a = Bottleneck(128,256,downsample=True,prevfeatures=128)
        self.nn4b = Bottleneck(128,256,prevfeatures=256)
        self.nn4c = Bottleneck(128,256,prevfeatures=256)
        
        self.nn5a = Bottleneck(256,512,downsample=True,prevfeatures=256)
        self.nn5b = Bottleneck(256,512,prevfeatures=512)
        self.nn5c = Bottleneck(256,512,prevfeatures=512)
        nnna = nn.Sequential(
          self.conv0,
          self.nn1a,
          self.nn2a,
          self.nn3a,
          self.nn4a,
          self.nn5a,
          self.nn5c
        )
        
        
        input_size = (1,nchannels,image_size,image_size)
        size_after_bottlenecks = get_output_shape(nnna,input_size)
        self.pool1 = nn.AvgPool2d(size_after_bottlenecks[-1],padding='same')
        size_after_pool = get_output_shape(self.pool1, size_after_bottlenecks)
        self.flatten = nn.Flatten()
        size_after_flatten=np.prod(size_after_pool[1:])
        
        self.dense = nn.Linear(size_after_flatten,1)

        self.features = nn.Sequential(
                      nn.Conv2d(nchannels, 32, kernel_size=7, stride=1,
                                       groups=1, dilation=1,padding='same',bias=False),
                      
                      nn.BatchNorm2d(32),
                      nn.ELU(),
                      
                      Bottleneck(16,32,preactivated=True,prevfeatures=32),
                      Bottleneck(16,32,prevfeatures=32),
                      Bottleneck(16,32,prevfeatures=32),
                      
                      Bottleneck(32,64,downsample=True,prevfeatures=32),
                      Bottleneck(32,64,prevfeatures=64),
                      Bottleneck(32,64,prevfeatures=64),
                      
                      Bottleneck(64,128,downsample=True,prevfeatures=64),
                      Bottleneck(64,128,prevfeatures=128),
                      Bottleneck(64,128,prevfeatures=128),
                      
                      Bottleneck(128,256,downsample=True,prevfeatures=128),
                      Bottleneck(128,256,prevfeatures=256),
                      Bottleneck(128,256,prevfeatures=256),
                      
                      Bottleneck(256,512,downsample=True,prevfeatures=256),
                      Bottleneck(256,512,prevfeatures=512),
                      Bottleneck(256,512,prevfeatures=512),
                    )
        
        self.classifier = nn.Sequential(
                      nn.Linear(size_after_flatten,1),
                      nn.Sigmoid()
                )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class CMUDeepLens_2(nn.Module):
    def __init__(self, nchannels = 1, image_size=66,
                 non_linearity=F.elu,verbose=True):
        super(CMUDeepLens_2, self).__init__()
        self.verbose=verbose
        
        input_size = (1,nchannels,image_size,image_size)

        self.features = nn.Sequential(
                      nn.Conv2d(nchannels, 32, kernel_size=7, stride=1,
                                       dilation=1,padding='same',bias=False),
                      
                      nn.BatchNorm2d(32,
                                     momentum=0.01,
                                     eps=1e-3),
                      nn.ELU(),
                      
                      Bottleneck(16,32,preactivated=True,prevfeatures=32),
                      Bottleneck(16,32,prevfeatures=32),
                      Bottleneck(16,32,prevfeatures=32),
                      
                      Bottleneck(32,64,downsample=True,prevfeatures=32),
                      Bottleneck(32,64,prevfeatures=64),
                      Bottleneck(32,64,prevfeatures=64),
                      
                      Bottleneck(64,128,downsample=True,prevfeatures=64),
                      Bottleneck(64,128,prevfeatures=128),
                      Bottleneck(64,128,prevfeatures=128),
                      
                      Bottleneck(128,256,downsample=True,prevfeatures=128),
                      Bottleneck(128,256,prevfeatures=256),
                      Bottleneck(128,256,prevfeatures=256),
                      
                      Bottleneck(256,512,downsample=True,prevfeatures=256),
                      Bottleneck(256,512,prevfeatures=512),
                      Bottleneck(256,512,prevfeatures=512),
                    )
        
        size_after_bottlenecks = get_output_shape(self.features,input_size)
        print('size_after_bottlenecks:',size_after_bottlenecks)
        self.pool = nn.AvgPool2d(size_after_bottlenecks[-1],padding=0)
        size_after_pool = get_output_shape(self.pool, size_after_bottlenecks)
        size_after_flatten = np.prod(size_after_pool[1:])
        print(f'{size_after_flatten = }')
        self.classifier = nn.Sequential(
                      nn.Linear(size_after_flatten,1),
                      nn.Sigmoid()
                )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x

class CMUDeepLens_features(nn.Module):
    def __init__(self, nchannels = 1, image_size=66,
                 non_linearity=F.elu,
                 verbose=True):
        super(CMUDeepLens_features, self).__init__()
        self.verbose=verbose
        
        input_size = (1,nchannels,image_size,image_size)

        self.features = nn.Sequential(
                      nn.Conv2d(nchannels, 32, kernel_size=7, stride=1,
                                       groups=1, dilation=1,padding='same',bias=False),
                      
                      nn.BatchNorm2d(32),
                      nn.ELU(),
                      
                      Bottleneck(16,32,preactivated=True,prevfeatures=32),
                      Bottleneck(16,32,prevfeatures=32),
                      Bottleneck(16,32,prevfeatures=32),
                      
                      Bottleneck(32,64,downsample=True,prevfeatures=32),
                      Bottleneck(32,64,prevfeatures=64),
                      Bottleneck(32,64,prevfeatures=64),
                      
                      Bottleneck(64,128,downsample=True,prevfeatures=64),
                      Bottleneck(64,128,prevfeatures=128),
                      Bottleneck(64,128,prevfeatures=128),
                      
                      Bottleneck(128,256,downsample=True,prevfeatures=128),
                      Bottleneck(128,256,prevfeatures=256),
                      Bottleneck(128,256,prevfeatures=256),
                      
                      Bottleneck(256,512,downsample=True,prevfeatures=256),
                      Bottleneck(256,512,prevfeatures=512),
                      Bottleneck(256,512,prevfeatures=512),
                    )
        
        size_after_bottlenecks = get_output_shape(self.features,input_size)
        # print('size_after_bottlenecks:',size_after_bottlenecks)
        self.pool = nn.AvgPool2d(size_after_bottlenecks[-1],padding=0)
        size_after_pool = get_output_shape(self.pool, size_after_bottlenecks)
        size_after_flatten = np.prod(size_after_pool[1:])
        

    def forward(self, x: torch.float) -> torch.float:
        x = self.features(x)
        x = self.pool(x)
        x = nn.Flatten()(x)
        return x    


#%%
def count_nb_of_parameters(net):
    """Compute the number of parameters in the intup network
    
    Args:
      net : Input nn.Module
      
    """
    # Count the number of parameters in `net`
    #
    nb_parameters = np.sum([layer.numel() for layer in net.parameters()])
    return nb_parameters
