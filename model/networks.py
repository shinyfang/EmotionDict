import torch
import torch.nn as nn
from torch.nn import init
from models.positional_encoding import *
import math
import torch.nn.functional as F

OFF_SLOPE = 1e-3


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
        if hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('LSTM') != -1):
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
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
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
        assert (torch.cuda.is_available())
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


class Transformer_ALL_decoder(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32, num_layers=3, dropout=0.05, max_len=100, position='fixed',
                 adding_module='default', batch_size=32, local=None):
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
        self.decoder = nn.Linear(self.d_model, 32)

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
        # video = video.reshape(-1, 8, 4)
        # src = torch.cat([eeg, gsr, ppg,video], dim=1)  # concat eeg, gsr, ppg signal on chennel dimension

        src = src.permute(1, 0, 2)

        src = self.input_project(src) * math.sqrt(self.d_model)  # for input with feature dim=4

        src = self.pos_encoder(src)  # torch.Size([150, 32, 256])
        encoder = self.transformer_encoder(src)
        # print(encoder.size())
        decoder = self.decoder(encoder[-1, :, :])

        return decoder


class Transformer_wo_decoder_vid(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32, num_layers=3, dropout=0.05, max_len=100, position='fixed',
                 adding_module='default', batch_size=32, local=None):
        super(Transformer_wo_decoder_vid, self).__init__()
        print("using Transformer without decoder in Video")
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
        self.decoder = nn.Linear(self.d_model, 32)

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

        src = video.reshape(-1, 32, 1)  # concat eeg, gsr, ppg signal on chennel dimension
        src = src.permute(1, 0, 2)
        # print("vid input shape", video.shape,src.shape)
        # vid input shape torch.Size([16, 32]) torch.Size([32, 16, 1])
        # tgt = src[-1:, :, :]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(device)

        src = self.input_project(src) * math.sqrt(self.d_model)  # for input with feature dim=4
        src = self.pos_encoder(src)  # torch.Size([150, 32, 256])
        encoder = self.transformer_encoder(src)

        decoder = self.decoder(encoder[-1, :, :])
        # print("vid decoder shape", decoder.shape)
        # vid decoder shape torch.Size([16, 32])
        return decoder


class Transformer_wo_decoder_Signal(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32, num_layers=3, dropout=0.05, max_len=100, batch_size=32,
                 signal_dim=4):
        super(Transformer_wo_decoder_Signal, self).__init__()
        print("using Transformer without decoder in Signal")
        self.model_type = 'Transformer'
        self.src_mask = None
        self.bs = batch_size
        self.feature_num = feature_dim
        self.signal_dim = signal_dim

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
        self.decoder = nn.Linear(self.d_model, 32)

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

        src = singal.reshape(-1, self.signal_dim, 1)  # concat eeg, gsr, ppg signal on chennel dimension
        src = src.permute(1, 0, 2)

        src = self.input_project(src) * math.sqrt(self.d_model)  # for input with feature dim=4
        src = self.pos_encoder(src)  # torch.Size([150, 32, 256])
        encoder = self.transformer_encoder(src)
        decoder = self.decoder(encoder[-1, :, :])
        return decoder


class Transformer_wo_decoder_EEG(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32, num_layers=3, dropout=0.05, max_len=100, position='fixed',
                 adding_module='default', batch_size=32, local=None):
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
        self.decoder = nn.Linear(self.d_model, 32)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, eeg):
        """
        src:torch.Size([600, 32])->torch.Size([150, 32, 4]) [seq_len,batch_size,feature_dim]
        src_key_padding_mask:torch.Size([32, 150]) [batch_size,seq_len]
        src after linear projection:torch.Size([150, 32, 256]) [seq_len,batch_size,d_model]
        static_seq: torch.size([32,7])
        Encoder input:[seq_len,batch_size,d_model]
        """

        src = eeg.reshape(-1, 18, 4)  # concat eeg, gsr, ppg signal on chennel dimension
        src = src.permute(1, 0, 2)

        src = self.input_project(src) * math.sqrt(self.d_model)  # for input with feature dim=4
        src = self.pos_encoder(src)  # torch.Size([150, 32, 256])
        encoder = self.transformer_encoder(src)
        decoder = self.decoder(encoder[-1, :, :])
        return decoder


class cls_attention(nn.Module):
    def __init__(self, dims, cls=10):
        super(cls_attention, self).__init__()
        print("using discriminator cls_attention")

        self.dims = dims

        layer_1 = [
            nn.Linear(dims, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        ]
        layer_2 = [
            nn.Linear(128, 64),
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
        self.fc = nn.Linear(128, cls)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # input = torch.cat([feature_phy, feature_vid], dim=1)

        x = input
        # print("cls attention input shape", feature_phy.shape,feature_vid.shape,input.shape, x.shape)
        # cls attention input shape torch.Size([16, 32]) torch.Size([16, 32]) torch.Size([16, 64]) torch.Size([16, 64])
        x = self.model_1(x)
        # print(x)
        att = self.model_2(x)
        x = torch.multiply(x, att)
        x = self.fc(x)
        # print(x)
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
