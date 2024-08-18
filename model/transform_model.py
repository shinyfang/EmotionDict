from .base_model import BaseModel
from . import networks
import torch
import itertools


class TransformModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        # build the framework of our basic model
        print("Using Transform Model Now!")

        self.feature_extract = networks.Transformer_ALL_decoder(batch_size=opt.batch_size, feature_dim=4,
                                                                hidden_dim=24).cuda()
        self.fe_eeg = networks.Transformer_wo_decoder_EEG(batch_size=opt.batch_size, feature_dim=4,
                                                          hidden_dim=12).cuda()
        self.fe_gsr = networks.Transformer_wo_decoder_Signal(batch_size=opt.batch_size, feature_dim=1,
                                                             hidden_dim=12).cuda()
        self.fe_ppg = networks.Transformer_wo_decoder_Signal(batch_size=opt.batch_size, feature_dim=1,
                                                             hidden_dim=12).cuda()
        self.fe_video = networks.Transformer_wo_decoder_vid(batch_size=opt.batch_size, feature_dim=1,
                                                            hidden_dim=12).cuda()
        self.e_dict = networks.EmotionDict(32).cuda()

        self.classifer = networks.cls_attention(dims=128).cuda()
        self.softmax = torch.nn.BCEWithLogitsLoss()

        self.criterionClS = networks.crossEntropy_loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.criterionKL = torch.nn.KLDivLoss()

        self.optimizers = torch.optim.Adam(
            itertools.chain(self.feature_extract.parameters(), self.fe_eeg.parameters(), self.fe_gsr.parameters(),
                            self.fe_ppg.parameters(), self.fe_video.parameters(), self.e_dict.parameters(),
                            self.classifer.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['train_eeg'].float().to(self.device), \
                                                               data['train_gsr'].float().to(self.device), \
                                                               data['train_ppg'].float().to(self.device), \
                                                               data['train_video'].float().to(self.device), \
                                                               data['train_label'].float().to(self.device)
        self.video = self.video.reshape(-1, 8, 4)
        self.src = torch.cat([self.eeg, self.gsr, self.ppg, self.video],
                             dim=1).cuda()

    def set_test_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['test_eeg'].float().to(self.device), \
                                                               data['test_gsr'].float().to(self.device), \
                                                               data['test_ppg'].float().to(self.device), \
                                                               data['test_video'].float().to(self.device), \
                                                               data['test_label'].float().to(self.device)
        self.video = self.video.reshape(-1, 8, 4)
        self.src = torch.cat([self.eeg, self.gsr, self.ppg, self.video],
                             dim=1).cuda()

    def forward(self):
        attention = self.feature_extract(self.src)
        embedding = self.e_dict(attention)
        self.logits = self.classifer(embedding)
        self.attention = embedding

        self.eeg_attention = self.e_dict(self.fe_eeg(self.eeg))
        self.eeg_logits = self.classifer(self.eeg_attention)
        self.gsr_attention = self.e_dict(self.fe_gsr(self.gsr))
        self.gsr_logits = self.classifer(self.gsr_attention)
        self.ppg_attention = self.e_dict(self.fe_ppg(self.ppg))
        self.ppg_logits = self.classifer(self.ppg_attention)
        self.video_attention = self.e_dict(self.fe_video(self.video))
        self.video_logits = self.classifer(self.video_attention)

    def backward(self):
        self.kl_loss = 0
        max_label, _ = torch.max(self.label, dim=1, keepdim=True)
        self.multi_hot_label = (self.label == max_label) * 1.0
        self.cls_loss = self.criterionClS(self.logits, self.multi_hot_label)
        self.mne_loss = self.criterionKL(self.logits, self.label)

        self.three_attention = (self.gsr_attention + self.ppg_attention + self.video_attention) / 3
        self.kl_loss = self.criterionKL(self.three_attention, self.attention.detach()) + self.criterionKL(
            self.eeg_attention, self.attention.detach())
        self.loss = self.cls_loss + self.mne_loss + self.kl_loss * 0.1
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizers.zero_grad()
        self.backward()
        self.optimizers.step()

    def current_loss(self):
        return float(self.loss), float(self.cls_loss), float(self.mne_loss), float(self.kl_loss)

    def save(self, epoch):
        self.save_network(self.feature_extract, 'Transform', epoch)
        self.save_network(self.e_dict, 'EDict', epoch)
        self.save_network(self.classifer, 'classifer', epoch)

    def load(self, epoch):
        self.load_network(self.feature_extract, 'Transform', epoch)
        self.load_network(self.e_dict, 'EDict', epoch)
        self.load_network(self.classifer, 'classifer', epoch)

        print("load pretrained network successfully")

    def test(self):
        with torch.no_grad():
            self.forward()
        return self.logits.cpu().numpy(), self.label.cpu().numpy()

    def eval(self):
        self.feature_extract.eval()
        self.e_dict.eval()
        self.classifer.eval()

    def train(self):
        self.feature_extract.train(mode=True)
        self.e_dict.train(mode=True)
        self.classifer.train(mode=True)
