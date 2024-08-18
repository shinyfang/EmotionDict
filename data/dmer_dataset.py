from data.base_dataset import *
import random
import numpy as np

class DMERdataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        sub_path = os.path.join(self.root_path, str(self.sub_id))
        self.train_eeg, self.train_gsr, self.train_ppg, self.train_video, self.train_label,\
            self.test_eeg, self.test_gsr, self.test_ppg, self.test_video, self.test_label  = load_signal(sub_path, self.sub_id) # load data of certain subject
        self.last_eeg, self.last_gsr, self.last_ppg, self.last_video, self.last_label = load_last_signal(sub_path,
                                                                                                    self.sub_id)
        self.train_size = self.train_eeg.shape[0]
        self.test_size = self.test_eeg.shape[0]
        
    
    def __len__(self):
        if self.phase == "test":
            return self.test_size
        return self.train_size


    def __getitem__(self, index):
        if self.phase == 'train':
            return {
                'train_eeg': self.train_eeg[index],\
                'train_gsr': self.train_gsr[index],\
                'train_ppg': self.train_ppg[index],\
                'train_video': self.train_video[index],\
                'train_label': self.train_label[index]
            }
        else: # self.phase == 'test'
            return {
                'test_eeg': self.test_eeg[index],\
                'test_gsr': self.test_gsr[index],\
                'test_ppg': self.test_ppg[index],\
                'test_video': self.test_video[index],\
                'test_label': self.test_label[index],
                'last_eeg': self.last_eeg, \
                'last_gsr': self.last_gsr, \
                'last_ppg': self.last_ppg, \
                'last_video': self.last_video, \
                'last_label': self.last_label
            }
        