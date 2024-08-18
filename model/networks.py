# from posixpath import split
# from time import time
# from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
# import functools
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
from model.positional_encoding import *
import math
# from torch.autograd import Function
# from typing import Any, Optional, Tuple
# from collections import OrderedDict
# from scipy.spatial.distance import cdist
# import ot
import torch.nn.functional as F
OFF_SLOPE=1e-3

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('LSTM')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_network(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class crossEntropy_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(crossEntropy_loss, self).__init__()
        self.reduction = reduction
    def __call__(self, logits, targets):
        log_probs = -1.0 * torch.log(logits)
        targets_probs = F.softmax(targets, dim=1)
        loss = torch.mul(log_probs, targets_probs).sum(dim=1)

        return loss.mean()

class crossEntropy_compare_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(crossEntropy_loss, self).__init__()
        self.reduction = reduction
    def __call__(self, logits, targets):
        log_probs = -1.0 * torch.log(logits)
        # targets_probs = F.softmax(targets, dim=1)
        targets_probs = targets
        loss = torch.mul(log_probs, targets_probs).sum(dim=1)

        return loss.mean()


def define_G(netG, batch_size, window_len, gpu_ids=[], init_type='normal', init_gain=0.02):
    if netG == 'Trans_wo_decoder':
        net = Transformer_wo_decoder(batch_size=batch_size, feature_dim = 4, hidden_dim = 24)
    elif netG == 'Trans_wo_decoder_vid':
        net = Transformer_wo_decoder_vid(batch_size=batch_size, feature_dim = 1, hidden_dim = 12)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(net, init_type, init_gain, gpu_ids)

def define_D(netD, batch_size, dims, gpu_ids=[], init_type='normal', init_gain=0.02):
    if netD == 'cls_attention':
        net = cls_attention(dims = dims)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_network(net, init_type, init_gain, gpu_ids)

class Transformer_wo_decoder(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim = 32, num_layers=3, dropout=0.05,max_len=100,position='fixed',adding_module='default',batch_size=32, dict_dim=32):
        super(Transformer_wo_decoder, self).__init__()
        print("using Transformer without decoder in generator for extracting video features")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.bs=batch_size
        self.feature_num=feature_dim
        
        self.d_model=hidden_dim  # origin: self.d_model = feature_size
        self.dropout=dropout
        self.max_len=max_len
        self.input_project=nn.Linear(feature_dim,self.d_model)
        self.t2c = SineActivation(feature_dim, self.d_model)

        #---------whether use 1D conv-------------#
        self.local=context_embedding(self.d_model,self.d_model,1)
        
        #----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)
        # self.pos_encoder = regular_PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.d_model, 32) 

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, eeg, gsr, ppg):
        """
        src:torch.Size([600, 32])->torch.Size([150, 32, 4]) [seq_len,batch_size,feature_dim]
        src_key_padding_mask:torch.Size([32, 150]) [batch_size,seq_len]
        src after linear projection:torch.Size([150, 32, 256]) [seq_len,batch_size,d_model]
        static_seq: torch.size([32,7])
        Encoder input:[seq_len,batch_size,d_model]
        """
        src = torch.cat([eeg, gsr, ppg], dim = 1) # concat eeg, gsr, ppg signal on chennel dimension

        src=src.permute(1,0,2)


        src=self.input_project(src) * math.sqrt(self.d_model) # for input with feature dim=4

        src = self.pos_encoder(src) # torch.Size([150, 32, 256])
        encoder = self.transformer_encoder(src)
        decoder = self.decoder(encoder[-1,:,:])

        return decoder


class Transformer_ALL_decoder(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32, num_layers=3, dropout=0.05, max_len=100, position='fixed',
                 adding_module='default', batch_size=32, dict_dim=32):
        super(Transformer_ALL_decoder, self).__init__()
        print("using Transformer without decoder in generator for extracting video features")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.bs = batch_size
        self.feature_num = feature_dim

        self.d_model = hidden_dim  # origin: self.d_model = feature_size
        self.dropout = dropout
        self.max_len = max_len
        self.input_project = nn.Linear(feature_dim, self.d_model)
        self.t2c = SineActivation(feature_dim, self.d_model)

        # ---------whether use 1D conv-------------#
        self.local = context_embedding(self.d_model, self.d_model, 1)

        # ----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)
        # self.pos_encoder = regular_PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.d_model, dict_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        src:torch.Size([600, 32])->torch.Size([150, 32, 4]) [seq_len,batch_size,feature_dim]
        src_key_padding_mask:torch.Size([32, 150]) [batch_size,seq_len]
        src after linear projection:torch.Size([150, 32, 256]) [seq_len,batch_size,d_model]
        static_seq: torch.size([32,7])
        Encoder input:[seq_len,batch_size,d_model]
        """


        src = src.permute(1, 0, 2)

        src = self.input_project(src) * math.sqrt(self.d_model)  # for input with feature dim=4

        src = self.pos_encoder(src)  # torch.Size([150, 32, 256])
        encoder = self.transformer_encoder(src)
        decoder = self.decoder(encoder[-1, :, :])

        return decoder

class Transformer_wo_decoder_vid(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim = 32, num_layers=3, dropout=0.05,max_len=100,position='fixed',adding_module='default',batch_size=32, dict_dim=32):
        super(Transformer_wo_decoder_vid, self).__init__()
        print("using Transformer without decoder in Video")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.bs=batch_size
        self.feature_num=feature_dim
        
        self.d_model=hidden_dim  # origin: self.d_model = feature_size
        self.dropout=dropout
        self.max_len=max_len
        self.input_project=nn.Linear(feature_dim,self.d_model)
        self.t2c = SineActivation(feature_dim, self.d_model)

        #---------whether use 1D conv-------------#
        self.local=context_embedding(self.d_model,self.d_model,1)
        
        #----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)
        # self.pos_encoder = regular_PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.d_model, dict_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, video):
        """
        src:torch.Size([600, 32])->torch.Size([150, 32, 4]) [seq_len,batch_size,feature_dim]
        src_key_padding_mask:torch.Size([32, 150]) [batch_size,seq_len]
        src after linear projection:torch.Size([150, 32, 256]) [seq_len,batch_size,d_model]
        static_seq: torch.size([32,7])
        Encoder input:[seq_len,batch_size,d_model]
        """

        src = video.reshape(-1,32,1) # concat eeg, gsr, ppg signal on chennel dimension
        src=src.permute(1,0,2)

        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        src=self.input_project(src) * math.sqrt(self.d_model) # for input with feature dim=4
        src = self.pos_encoder(src) 
        encoder = self.transformer_encoder(src)
      
        decoder = self.decoder(encoder[-1,:,:])

        return decoder


class Transformer_wo_decoder_Signal(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32, num_layers=3, dropout=0.05, max_len=100, batch_size=32, signal_dim=18,dict_dim=32):
        super(Transformer_wo_decoder_Signal, self).__init__()
        print("using Transformer without decoder in Signal")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.bs = batch_size
        self.feature_num = feature_dim
        self.signal_dim=signal_dim

        self.d_model = hidden_dim  # origin: self.d_model = feature_size
        self.dropout = dropout
        self.max_len = max_len
        self.input_project = nn.Linear(feature_dim, self.d_model)
        self.t2c = SineActivation(feature_dim, self.d_model)

        # ---------whether use 1D conv-------------#
        self.local = context_embedding(self.d_model, self.d_model, 1)

        # ----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)
        # self.pos_encoder = regular_PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.d_model, dict_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, singal):
        """
        src:torch.Size([600, 32])->torch.Size([150, 32, 4]) [seq_len,batch_size,feature_dim]
        src_key_padding_mask:torch.Size([32, 150]) [batch_size,seq_len]
        src after linear projection:torch.Size([150, 32, 256]) [seq_len,batch_size,d_model]
        static_seq: torch.size([32,7])
        Encoder input:[seq_len,batch_size,d_model]
        """

        src = singal.reshape(-1, 2, self.signal_dim)  # concat eeg, gsr, ppg signal on chennel dimension
        src = src.permute(1, 0, 2)

        src = self.input_project(src) * math.sqrt(self.d_model)  # for input with feature dim=4
        src = self.pos_encoder(src)  # torch.Size([150, 32, 256])
        encoder = self.transformer_encoder(src)
        decoder = self.decoder(encoder[-1, :, :])
        return decoder

class Transformer_wo_decoder_EEG(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32, num_layers=3, dropout=0.05, max_len=100, position='fixed',
                 adding_module='default', batch_size=32,  dict_dim=32):
        super(Transformer_wo_decoder_EEG, self).__init__()
        print("using Transformer without decoder in EEG")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.bs = batch_size
        self.feature_num = feature_dim

        self.d_model = hidden_dim  # origin: self.d_model = feature_size
        self.dropout = dropout
        self.max_len = max_len
        self.input_project = nn.Linear(feature_dim, self.d_model)
        self.t2c = SineActivation(feature_dim, self.d_model)

        # ---------whether use 1D conv-------------#
        self.local = context_embedding(self.d_model, self.d_model, 1)

        # ----------whether use learnable position encoding---------#
        self.pos_encoder = PositionalEncoding(self.d_model)
        # self.pos_encoder = regular_PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.d_model, dict_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, eeg):
        src = eeg.permute(1, 0, 2)

        src = self.input_project(src) * math.sqrt(self.d_model)  # for input with feature dim=4
        src = self.pos_encoder(src)
        encoder = self.transformer_encoder(src)
        decoder = self.decoder(encoder[-1, :, :])
        return decoder


class cls_attention(nn.Module):
    def __init__(self,  dims,cls=10):
        super(cls_attention, self).__init__()
        print("using discriminator cls_attention")

        self.dims = dims

        layer_1 = [
            nn.Linear(dims, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        ]
        layer_2 = [
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Sigmoid()
        ]
        self.model_1 = nn.Sequential(*layer_1)
        self.model_2 = nn.Sequential(*layer_2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128,cls)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input):
        x = input
        x = self.model_1(x)

        att = self.model_2(x)
        x = torch.multiply(x, att)
        x = self.fc(x)

        return self.softmax(x)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).cuda())
            # support.append(torch.eye(A.shape[1]))
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out).cuda())
        # self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            # self.bias = nn.Parameter(torch.FloatTensor(num_out))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):

        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Chebynet(nn.Module):
    def __init__(self, edge_num, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim, num_out))
        self.A = nn.Parameter(torch.FloatTensor(edge_num, edge_num))
        nn.init.xavier_normal_(self.A)
        self.g_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_out * edge_num, num_out)
        self.bn1=nn.LayerNorm(num_out)

    def forward(self, x):

        adj = generate_cheby_adj(self.A, self.K)
        # print("adj",adj[0].shape)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = torch.flatten(result, start_dim=1)
        result = self.fc1(result)
        result = self.bn1(result)
        return F.softmax(result)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, out_size, hidden_dim=512, dropout_prob=0.5, num_classes=82):
        super().__init__()
        self.dense1 = nn.Linear(out_size, 128)
        self.norm_0 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.7)
        self.out_proj = nn.Linear(128, num_classes)


    def forward(self, features):
        # x = torch.cat(features, -1)
        # x = self.dropout(features)
        x = self.dense1(features)
        x = torch.relu(self.norm_0(x))
        x = self.dropout(x)
        x = self.out_proj(x)

        return F.softmax(x)

class EmotionDict(nn.Module):
    def __init__(self, motion_dim=32):
        super(EmotionDict, self).__init__()



        self.weight = nn.Parameter(torch.randn(128, motion_dim))
    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):


        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, dim=1024, num_clusters=64, lamb=64, groups=4, max_frames=300, num_classes=10):
        super(NeXtVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.K = num_clusters
        self.G = groups
        self.group_size = int((lamb * dim) // self.G)
        print("group_size:",self.group_size,lamb,dim,self.G)
        # expansion FC

        self.fc0 = nn.Linear(dim, lamb * dim)
        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * dim, self.G * self.K)
        # attention over groups FC
        self.fc_g = nn.Linear(lamb * dim, self.G)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.group_size, self.K))

        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)
        self.classifer = ClassificationHead(out_size=self.K * self.group_size, num_classes=num_classes)
        self.softmax = LabelSmoothingCrossEntropy()

    def forward(self, x, mask=None):

        _, M, N = x.shape
        # expansion FC: B x M x N -> B x M x λN
        x_dot = self.fc0(x)
        # x_dot = x
        # print("fc0", x_dot.shape)
        # reshape into groups: B x M x λN -> B x M x G x (λN/G)
        x_tilde = x_dot.reshape(-1, M, self.G, self.group_size)

        # residuals across groups and clusters: B x M x λN -> B x M x (G*K)
        WgkX = self.fc_gk(x_dot)
        WgkX = self.bn0(WgkX)

        # residuals reshape across clusters: B x M x (G*K) -> B x (M*G) x K
        WgkX = WgkX.reshape(-1, M * self.G, self.K)

        # softmax over assignment: B x (M*G) x K -> B x (M*G) x K
        alpha_gk = F.softmax(WgkX, dim=-1)

        # attention across groups: B x M x λN -> B x M x G
        # alpha_g = torch.sigmoid(self.fc_g(x_dot))
        alpha_g = self.fc_g(x_dot)
        alpha_g = .5 * (1 + torch.tanh(.5 * alpha_g))
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))

        # reshape across time: B x M x G -> B x (M*G) x 1
        alpha_g = alpha_g.reshape(-1, M * self.G, 1)

        # apply attention: B x (M*G) x K (X) B x (M*G) x 1 -> B x (M*G) x K
        activation = torch.mul(alpha_gk, alpha_g)

        # sum over time and group: B x (M*G) x K -> B x 1 x K
        a_sum = torch.sum(activation, -2, keepdim=True)

        # calculate group centers: B x 1 x K (X) 1 x (λN/G) x K -> B x (λN/G) x K
        a = torch.mul(a_sum, self.cluster_weights2)

        # permute: B x (M*G) x K -> B x K x (M*G)
        activation = activation.permute(0, 2, 1)

        # reshape: B x M x G x (λN/G) -> B x (M*G) x (λN/G)
        reshaped_x_tilde = x_tilde.reshape(-1, M * self.G, self.group_size)

        # cluster activation: B x K x (M*G) (X) B x (M*G) x (λN/G) -> B x K x (λN/G)
        vlad = torch.matmul(activation, reshaped_x_tilde)
        # print(f"vlad: {vlad.shape}")

        # permute: B x K x (λN/G) (X) B x (λN/G) x K
        vlad = vlad.permute(0, 2, 1)
        # distance to centers: B x (λN/G) x K (-) B x (λN/G) x K
        vlad = torch.sub(vlad, a)
        # normalize: B x (λN/G) x K
        vlad = F.normalize(vlad, 1)
        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)
        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)
        out = self.classifer(vlad)
        return out

    def loss(self, pred, y):
        return self.softmax(pred, y)

class NeXtDict(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, dim=1024, num_clusters=64, lamb=64, groups=4, max_frames=300, num_classes=10):
        super(NeXtDict, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.K = num_clusters
        self.G = groups
        self.group_size = int((lamb * dim) // self.G)
        print("group_size:",self.group_size,lamb,dim,self.G)
        # expansion FC

        self.fc0 = nn.Linear(dim, lamb * dim)
        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * dim, self.G * self.K)
        # attention over groups FC
        # attention over groups FC
        self.fc_g = nn.Linear(lamb * dim, self.G)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.group_size, self.K))

        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)
        dict_dim=self.K * self.group_size
        fc = []
        fc.append(EqualLinear(dict_dim, dict_dim//4))
        fc.append(EqualLinear(dict_dim//4, dict_dim // 16))
        fc.append(EqualLinear(dict_dim // 16, dict_dim // 64))
        fc.append(EqualLinear(dict_dim//64, 16))
        self.fc = nn.Sequential(*fc)




    def forward(self, x, mask=None):

        _, M, N = x.shape
        # expansion FC: B x M x N -> B x M x λN
        x_dot = self.fc0(x)
        # x_dot = x
        # print("fc0", x_dot.shape)
        # reshape into groups: B x M x λN -> B x M x G x (λN/G)
        x_tilde = x_dot.reshape(-1, M, self.G, self.group_size)

        # residuals across groups and clusters: B x M x λN -> B x M x (G*K)
        WgkX = self.fc_gk(x_dot)
        WgkX = self.bn0(WgkX)

        # residuals reshape across clusters: B x M x (G*K) -> B x (M*G) x K
        WgkX = WgkX.reshape(-1, M * self.G, self.K)

        # softmax over assignment: B x (M*G) x K -> B x (M*G) x K
        alpha_gk = F.softmax(WgkX, dim=-1)

        # attention across groups: B x M x λN -> B x M x G
        # alpha_g = torch.sigmoid(self.fc_g(x_dot))
        alpha_g = self.fc_g(x_dot)
        alpha_g = .5 * (1 + torch.tanh(.5 * alpha_g))
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))

        # reshape across time: B x M x G -> B x (M*G) x 1
        alpha_g = alpha_g.reshape(-1, M * self.G, 1)

        # apply attention: B x (M*G) x K (X) B x (M*G) x 1 -> B x (M*G) x K
        activation = torch.mul(alpha_gk, alpha_g)

        # sum over time and group: B x (M*G) x K -> B x 1 x K
        a_sum = torch.sum(activation, -2, keepdim=True)

        # calculate group centers: B x 1 x K (X) 1 x (λN/G) x K -> B x (λN/G) x K
        a = torch.mul(a_sum, self.cluster_weights2)

        # permute: B x (M*G) x K -> B x K x (M*G)
        activation = activation.permute(0, 2, 1)

        # reshape: B x M x G x (λN/G) -> B x (M*G) x (λN/G)
        reshaped_x_tilde = x_tilde.reshape(-1, M * self.G, self.group_size)

        # cluster activation: B x K x (M*G) (X) B x (M*G) x (λN/G) -> B x K x (λN/G)
        vlad = torch.matmul(activation, reshaped_x_tilde)
        # print(f"vlad: {vlad.shape}")

        # permute: B x K x (λN/G) (X) B x (λN/G) x K
        vlad = vlad.permute(0, 2, 1)
        # distance to centers: B x (λN/G) x K (-) B x (λN/G) x K
        vlad = torch.sub(vlad, a)
        # normalize: B x (λN/G) x K
        vlad = F.normalize(vlad, 1)
        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)
        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)

        embedding=self.fc(vlad)

        return embedding

    def loss(self, pred, y):
        return self.softmax(pred, y)