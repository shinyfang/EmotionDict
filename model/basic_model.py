from .base_model import BaseModel
from . import networks
import torch
import itertools
class BasicModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        # build the framework of our basic model    

        self.feature_extract_phy = networks.define_G(netG = self.netG, batch_size = self.batch_size, window_len = 1, gpu_ids=self.gpu_ids)
        self.feature_extract_vid = networks.define_G(netG = self.netG+"_vid", batch_size = self.batch_size, window_len = 1, gpu_ids=self.gpu_ids)
        
        self.cls = networks.define_D(netD = self.netD, batch_size = self.batch_size, dims= 32 * 2, gpu_ids=self.gpu_ids)

        self.criterionClS = networks.crossEntropy_loss()
        # self.criterionClS = torch.nn.NLLLoss()
        self.criterionMSE = torch.nn.MSELoss()  # regression loss

        self.optimizers = torch.optim.Adam(itertools.chain(self.feature_extract_phy.parameters(), self.feature_extract_vid.parameters(),\
                             self.cls.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['train_eeg'].float().to(self.device), \
                                                                data['train_gsr'].float().to(self.device), \
                                                                data['train_ppg'].float().to(self.device),\
                                                                data['train_video'].float().to(self.device),\
                                                                data['train_label'].float().to(self.device)

    def set_test_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['test_eeg'].float().to(self.device), \
                                                                data['test_gsr'].float().to(self.device), \
                                                                data['test_ppg'].float().to(self.device),\
                                                                data['test_video'].float().to(self.device),\
                                                                data['test_label'].float().to(self.device)
   
    def forward(self):
        self.feature_phy = self.feature_extract_phy(self.eeg, self.gsr, self.ppg)
        self.feature_vid = self.feature_extract_vid(self.video)
        input = torch.cat([self.feature_phy, self.feature_vid], dim=1)
        self.logits = self.cls(input)

    def backward(self):
        # print(self.logits.size(), self.label.size())
        # compute multi-target label 
        max_label, _ = torch.max(self.label,dim=1, keepdim=True)
        self.multi_hot_label = (self.label == max_label) * 1.0
        # print(multi_hot_label)
        self.cls_loss = self.criterionClS(self.logits, self.multi_hot_label) # use corss entropy loss in classification
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
        self.save_network(self.feature_extract_vid, 'G_V', epoch)
        self.save_network(self.feature_extract_phy, 'G_P', epoch)
        self.save_network(self.cls, 'D', epoch)

   
    def load(self, epoch):
        self.load_network(self.feature_extract_vid, 'G_V', epoch)
        self.load_network(self.feature_extract_phy, 'G_P', epoch)
        self.load_network(self.cls, 'D', epoch)

        print("load pretrained network successfully")
        
    
    def test(self): # test trained model
        with torch.no_grad():
            self.forward()

        return self.logits.cpu().numpy(), self.label.cpu().numpy()

    def eval(self):
        self.feature_extract_phy.eval()
        self.feature_extract_vid.eval()
        self.cls.eval()

    def train(self):
        self.feature_extract_phy.train(mode=True)
        self.feature_extract_vid.train(mode=True)
        self.cls.train(mode=True)


class NextModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        # build the framework of our basic model

        self.feature_extract = networks.NeXtVLAD(dim=4,max_frames=28)

        self.criterionClS = networks.crossEntropy_loss()
        # self.criterionClS = torch.nn.NLLLoss()
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
        self.video = self.video.reshape(-1,8,4) # concat eeg, gsr, ppg signal on chennel dimension
        self.src = torch.cat([self.eeg, self.gsr, self.ppg,self.video ], dim=1)  # concat eeg, gsr, ppg signal on chennel dimension

    def set_test_input(self, data):
        super().set_input(data)
        self.eeg, self.gsr, self.ppg, self.video, self.label = data['test_eeg'].float().to(self.device), \
                                                               data['test_gsr'].float().to(self.device), \
                                                               data['test_ppg'].float().to(self.device), \
                                                               data['test_video'].float().to(self.device), \
                                                               data['test_label'].float().to(self.device)
        self.video = self.video.reshape(-1,8,4) # concat eeg, gsr, ppg signal on chennel dimension
        self.src = torch.cat([self.eeg, self.gsr, self.ppg,self.video ], dim=1)  # concat eeg, gsr, ppg signal on chennel dimension

    def forward(self):
        self.logits = self.feature_extract.forward(self.src)

    def backward(self):
        # print(self.logits.size(), self.label.size())
        # compute multi-target label
        max_label, _ = torch.max(self.label, dim=1, keepdim=True)
        self.multi_hot_label = (self.label == max_label) * 1.0
        # print(multi_hot_label)
        self.cls_loss = self.criterionClS(self.logits, self.multi_hot_label)  # use corss entropy loss in classification
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
