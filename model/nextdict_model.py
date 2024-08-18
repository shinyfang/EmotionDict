from .base_model import BaseModel
from . import networks
import torch
import itertools
from info_nce import InfoNCE
class NextdictModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        # build the framework of our basic model

        self.feature_extract = networks.NeXtDict(dim=4,max_frames=28).cuda()
        self.fe_eeg = networks.NeXtDict(dim=4, max_frames=18).cuda()
        self.fe_gsr = networks.NeXtDict(dim=4, max_frames=1).cuda()
        self.fe_ppg = networks.NeXtDict(dim=4, max_frames=1).cuda()
        self.fe_video = networks.NeXtDict(dim=4, max_frames=8).cuda()
        self.e_dict = networks.EmotionDict(16).cuda()

        self.classifer = networks.cls_attention(dims=128).cuda()
        self.softmax = torch.nn.BCEWithLogitsLoss()

        self.criterionClS = networks.crossEntropy_loss()
        # self.criterionClS = torch.nn.NLLLoss()
        self.criterionMSE = torch.nn.MSELoss()  # regression loss
        self.criterionNCE=InfoNCE()
        self.optimizers = torch.optim.Adam(
            itertools.chain(self.feature_extract.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['train_eeg'].float().to(self.device), \
                                                               data['train_gsr'].float().to(self.device), \
                                                               data['train_ppg'].float().to(self.device), \
                                                               data['train_video'].float().to(self.device), \
                                                               data['train_label'].float().to(self.device)
        self.video = self.video.reshape(-1,8,4) # concat eeg, gsr, ppg signal on chennel dimension
        self.src = torch.cat([self.eeg, self.gsr, self.ppg,self.video ], dim=1).cuda()  # concat eeg, gsr, ppg signal on chennel dimension

    def set_test_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['test_eeg'].float().to(self.device), \
                                                               data['test_gsr'].float().to(self.device), \
                                                               data['test_ppg'].float().to(self.device), \
                                                               data['test_video'].float().to(self.device), \
                                                               data['test_label'].float().to(self.device)
        self.video = self.video.reshape(-1,8,4) # concat eeg, gsr, ppg signal on chennel dimension
        self.src = torch.cat([self.eeg, self.gsr, self.ppg,self.video ], dim=1).cuda()  # concat eeg, gsr, ppg signal on chennel dimension

    def forward(self):
        print("src",self.src.shape)
        attention = self.feature_extract.forward(self.src)
        # print(v.shape)
        embedding = self.e_dict(attention)
        print("embedding",embedding.shape)
        self.logits = self.classifer(embedding)
        self.attention=attention

        self.eeg_attention=self.fe_eeg(self.eeg)
        self.gsr_attention = self.fe_gsr(self.gsr)
        self.ppg_attention = self.fe_ppg(self.ppg)
        self.video_attention = self.fe_video(self.video)



    def backward(self):
        # print(self.logits.size(), self.label.size())attention
        # compute multi-target label
        max_label, _ = torch.max(self.label, dim=1, keepdim=True)
        self.multi_hot_label = (self.label == max_label) * 1.0
        # print(multi_hot_label)
        self.cls_loss = self.criterionClS(self.logits, self.multi_hot_label)  # use corss entropy loss in classification
        self.mne_loss = self.criterionMSE(self.logits, self.label)
        self.nce_loss=self.criterionNCE(self.eeg_attention,self.attention)+self.criterionNCE(self.gsr_attention,self.attention)+self.criterionNCE(self.ppg_attention,self.attention)+self.criterionNCE(self.video_attention,self.attention)
        self.loss = self.cls_loss + self.mne_loss
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizers.zero_grad()
        self.backward()
        self.optimizers.step()

    def current_loss(self):
        # print(self.logits, self.label)
        return float(self.loss), float(self.cls_loss), float(self.mne_loss)

    def save(self, epoch):
        self.save_network(self.feature_extract, 'Next', epoch)


    def load(self, epoch):
        self.load_network(self.feature_extract, 'Next', epoch)

        print("load pretrained network successfully")

    def test(self):  # test trained model
        with torch.no_grad():
            self.forward()

        return self.logits.cpu().numpy(), self.label.cpu().numpy()

    def eval(self):
        self.feature_extract.eval()


    def train(self):
        self.feature_extract.train(mode=True)
