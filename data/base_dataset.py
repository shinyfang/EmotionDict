import torch.utils.data as data
import os
from abc import ABC, abstractmethod
from scipy.io import loadmat
import numpy as np


class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.phase = self.opt.phase
        self.root_path = self.opt.root_path
        self.sub_id = self.opt.subject
        self.win_len = self.opt.win_len

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


# origin load data function
def load_signal(path, sub):
    # load data
    data = loadmat(os.path.join(path, "datas.mat"))
    label = data['dis_label']

    video_fet = loadmat(os.path.join(path, "videoFea.mat"))
    labels = np.array([label[int(x)] for x in video_fet["vids"][0]])

    shuffle_ix = np.random.permutation(np.arange(len(labels)))
    video_feats = video_fet["lbps_all"]
    video_padding = np.zeros((len(labels), 36))
    video_padding[:, :32] = video_feats
    video_padding = video_padding.reshape(len(labels), -1, 18)

    eeg_fet = loadmat(os.path.join(path, "eegfea.mat"))
    eeg_feats = eeg_fet["feas"]

    peri_fet = loadmat(os.path.join(path, "perifea.mat"))
    gsr_feats = peri_fet["feas_gsr"]
    gsr_padding = np.zeros((len(labels), 36))
    gsr_padding[:, :28] = gsr_feats
    gsr_padding = gsr_padding.reshape(len(labels), -1, 18)

    ppg_feats = peri_fet["feas_ppg"]
    ppg_padding = np.zeros((len(labels), 36))
    ppg_padding[:, :27] = ppg_feats
    ppg_padding = ppg_padding.reshape(len(labels), -1, 18)
    split_samples = video_feats.shape[0] // 10 * 8

    eeg = eeg_feats[shuffle_ix]
    gsr = gsr_padding[shuffle_ix]
    ppg = ppg_padding[shuffle_ix]
    video = video_padding[shuffle_ix]

    mixture_label = labels[shuffle_ix]
    return eeg[:split_samples], gsr[:split_samples], ppg[:split_samples], video[:split_samples], mixture_label[
                                                                                                 :split_samples], \
           eeg[split_samples:], gsr[split_samples:], ppg[split_samples:], video[split_samples:], mixture_label[
                                                                                                 split_samples:]

def load_last_signal(path, sub):
    # load data
    data = loadmat(os.path.join(path, "datas.mat"))
    label = data['dis_label']

    video_fet = loadmat(os.path.join(path, "videoFea.mat"))
    labels = np.array([label[int(x)] for x in video_fet["vids"][0]])
    last_ix=[]
    for x in range(len(video_fet["vids"][0])):
        if x==len(video_fet["vids"][0])-1 or video_fet["vids"][0][x]!=video_fet["vids"][0][x+1] :
            for i in range(5):
                last_ix.append(x-i)
    video_feats = video_fet["lbps_all"]
    video_padding = np.zeros((len(labels), 36))
    video_padding[:, :32] = video_feats
    video_padding = video_padding.reshape(len(labels), -1, 18)

    eeg_fet = loadmat(os.path.join(path, "eegfea.mat"))
    eeg_feats = eeg_fet["feas"]

    peri_fet = loadmat(os.path.join(path, "perifea.mat"))
    gsr_feats = peri_fet["feas_gsr"]
    gsr_padding = np.zeros((len(labels), 36))
    gsr_padding[:, :28] = gsr_feats
    gsr_padding = gsr_padding.reshape(len(labels), -1, 18)

    ppg_feats = peri_fet["feas_ppg"]
    ppg_padding = np.zeros((len(labels), 36))
    ppg_padding[:, :27] = ppg_feats
    ppg_padding = ppg_padding.reshape(len(labels), -1, 18)


    eeg = eeg_feats[last_ix]
    gsr = gsr_padding[last_ix]
    ppg = ppg_padding[last_ix]
    video = video_padding[last_ix]

    mixture_label = labels[last_ix]
    return eeg, gsr, ppg, video, mixture_label
   


if __name__ == '__main__':
    load_last_signal("./MixedEmoR/28", 1)
