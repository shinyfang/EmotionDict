from abc import ABC, abstractmethod
import torch
import os
class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.lr = opt.lr
        self.batch_size = opt.batch_size
        self.netG = opt.netG
        self.netD = opt.netD

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
        self.save_dir = os.path.join(opt.checkpoints, opt.model_name)

        if (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
    
    @abstractmethod
    def set_input(self, data):
        
        return 0

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    @abstractmethod
    def save(self, epoch):
        pass

    @abstractmethod
    def load(self, epoch):
        pass

    @abstractmethod
    def test(self): # test trained model
        pass

    def save_network(self, network, network_label, epoch):
        save_filename = '%d_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        
        network.cuda()

    def load_network(self, network, network_label, epoch):
        save_filename = '%d_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))