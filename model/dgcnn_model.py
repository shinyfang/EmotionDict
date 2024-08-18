from .base_model import BaseModel
from . import networks
import torch
import itertools

class DgcnnModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        # build the framework of our basic model

        self.feature_extract = networks.Chebynet(edge_num=28, xdim=4, K=2, num_out=10).cuda()

        self.criterionClS = networks.crossEntropy_loss()
        self.criterionMSE = torch.nn.MSELoss()  # regression loss

        self.optimizers = torch.optim.Adam(
            itertools.chain(self.feature_extract.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['train_eeg'].float().to(self.device), \
                                                               data['train_gsr'].float().to(self.device), \
                                                               data['train_ppg'].float().to(self.device), \
                                                               data['train_video'].float().to(self.device), \
                                                               data['train_label'].float().to(self.device)
        self.video = self.video.reshape(-1,8,4) 
        self.src = torch.cat([self.eeg, self.gsr, self.ppg,self.video ], dim=1).cuda()  

    def set_test_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['test_eeg'].float().to(self.device), \
                                                               data['test_gsr'].float().to(self.device), \
                                                               data['test_ppg'].float().to(self.device), \
                                                               data['test_video'].float().to(self.device), \
                                                               data['test_label'].float().to(self.device)
        self.video = self.video.reshape(-1,8,4) 
        self.src = torch.cat([self.eeg, self.gsr, self.ppg,self.video ], dim=1).cuda()  

    def forward(self):
        self.logits = self.feature_extract.forward(self.src)

    def backward(self):

        max_label, _ = torch.max(self.label, dim=1, keepdim=True)
        self.multi_hot_label = (self.label == max_label) * 1.0

        self.cls_loss = self.criterionClS(self.logits, self.multi_hot_label) 
        self.mne_loss = self.criterionMSE(self.logits, self.label)

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
