from sklearn.utils import shuffle
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm

from typing import Optional, Union
#### positional encoding ####

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0) former
        pe = pe.unsqueeze(0).transpose(0,1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):

        return x+Variable(self.pe[:x.size(0), :], requires_grad=False)


class regular_PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, period: Optional[int] = 24):
        super(regular_PositionalEncoding, self).__init__()
        self.d_model = d_model
        length = 128
        PE = torch.zeros((length, self.d_model))

        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        PE = torch.sin(pos * 2 * np.pi / period)
        PE = PE.repeat((1, self.d_model))
        PE = PE.unsqueeze(0).transpose(0,1)

        self.register_buffer('PE', PE)
    def forward(self, x):
        return x+Variable(self.PE, requires_grad=False)

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.5, 0.5)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + Variable(self.pe[:x.size(0), :], 
                         requires_grad=False)
        return self.dropout(x)
    


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,embedding_size2=16,k=5):
        super(context_embedding,self).__init__()
        self.causal_convolution = weight_norm(CausalConv1d(in_channels,embedding_size,kernel_size=k))
        # self.causal_convolution2 = weight_norm(CausalConv1d(in_channels,embedding_size,kernel_size=k))
        self.init_weights()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_weights(self):
        self.causal_convolution.weight.data.normal_(0, 0.01)


    def forward(self,x):

        return F.tanh(self.causal_convolution(x))


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=2, #3
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], 1)
    
class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)