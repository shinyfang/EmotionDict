from .base_model import BaseModel
from . import networks
import torch
import itertools
class CompareModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        # build the framework of our basic model
        self.feature_extract = networks.define_G(netG = self.netG, batch_size = self.batch_size, window_len = 1, gpu_ids=self.gpu_ids)
        self.criterionClS = networks.crossEntropy_compare_loss()
        self.optimizers = torch.optim.SGD(self.feature_extract.parameters(), lr=0.01,  momentum=0.9, weight_decay=0.0005)

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
        self.logits = self.feature_extract(self.eeg, self.gsr, self.ppg, self.video)

    def backward(self):
        # print(multi_hot_label)
        self.loss = self.criterionClS(self.logits, self.label) # use corss entropy loss in classification
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizers.zero_grad()
        self.backward()
        self.optimizers.step()
        
    
    def current_loss(self):
        # print(self.logits, self.label)
        return float(self.loss), 0, 0

    
    def save(self, epoch):
        self.save_network(self.feature_extract, 'G', epoch)
   
    def load(self, epoch):
        self.load()

    
    def test(self): # test trained model
        with torch.no_grad():
            self.forward()

        return self.logits.cpu().numpy(), self.label.cpu().numpy()

    def eval(self):
        self.feature_extract.eval()

    def train(self):
        self.feature_extract.train(mode=True)
       

    