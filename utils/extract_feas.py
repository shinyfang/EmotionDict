from scipy.io import loadmat
import os
from scipy import signal
from scipy.fft import fft, fftshift
import numpy as np
from math import log2
from functools import reduce
import scipy.io as sio
import torch
import torchvision.models as models
import cv2
import torch.cuda
import torchvision.transforms as transforms
from torch.autograd import Variable


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = signal.butter(order, [low, high], 'bandpass')
    return b, a

def process_de(data, window_length, window_type, time_sample_rate, frequency_sample_rate, bands):
    """
        compute de feature in the first 18 channels
        record corresponding video number
        input: [18channel + video list, down-sampled timesteps]
        ourput: eeg de features, video list
    """
    channels = data.shape[0] - 1
    timesteps = data.shape[1]
    windows = int(timesteps / time_sample_rate / window_length)
    bands_num = len(bands) - 1
    rate = int(frequency_sample_rate / time_sample_rate)  # 频时采样比率
    frequency_window_points = int(window_length * frequency_sample_rate)
    # Declare DE tensor
    DE = np.zeros((channels, windows, bands_num))
    # get window function
    time_window_points = int(window_length * time_sample_rate)
    if window_type == 'hanning':
        window_function = signal.windows.hann(time_window_points)
    else:
        return "undefined window type yet0"
    # compute DE of a sampled data
    for i in range(channels):
        for j in range(windows):
            # Apply window function
            data_mul_window = data[i, j * time_window_points:(j + 1) * time_window_points] * window_function
            # Apply DFT
            fft_data = abs(fft(data_mul_window, frequency_window_points)[:int(frequency_window_points / 2)])
            # compute DE
            for k in range(bands_num):
                bands_list = fft_data[int(bands[k] * rate):int(bands[k + 1] * rate - 1)]
                DE[i][j][k] = log2(100 * reduce(lambda x, y: x + y, map(lambda x: x * x, bands_list)) / len(bands_list))

    # select movie induced data and record corresponding video number
    video_list = np.zeros(windows)
    video_tag = np.array(data[-1]).astype(int)
    for j in range(windows):
        video_list[j] = max(set(video_tag[j*time_window_points:(j+1)*time_window_points].tolist()), key=video_tag[j*time_window_points:(j+1)*time_window_points].tolist().count)
    
    # print(DE.shape)
    # print(video_list.shape)
    return DE, video_list


def process_physiological(path):
    """ process eeg, ppg, gsr signal """
    data = loadmat(os.path.join(path, 'datas.mat'))
    ori_eeg = data['eeg_datas']
    ori_gsr = data['gsr_datas']
    ori_ppg = data['ppg_datas']
    label = data['dis_label']
    # print(ori_eeg.shape, ori_gsr.shape, ori_ppg.shape, label.shape)

    # process eeg first
    sample_eeg = ori_eeg[:,::3] # downsample 300hz to 100hz
    eeg_bands = [4,8,14,31,45]
    eeg, video_list_eeg = process_de(sample_eeg,window_length=1, window_type='hanning',\
            time_sample_rate=100, frequency_sample_rate=1024,bands=eeg_bands)

    # process gsr 
    # gsr does not need downsample
    b, a = butter_bandpass(0.01, 1.9, 4.0, 2)
    filt_gsr = signal.filtfilt(b, a, ori_gsr[0:1,:], axis=1)   # filt the gsr signal using 2-th butter filter 
    ori_gsr[0:1,:] = filt_gsr
    gsr_bands = [0,0.6,1.2,1.8,2.0]
    gsr, video_list_gsr = process_de(ori_gsr,window_length=1, window_type='hanning',\
        time_sample_rate=4,frequency_sample_rate=64,bands=gsr_bands)

    # process ppg
    b, a = butter_bandpass(0.01, 49, 100, 2)
    filt_ppg = signal.filtfilt(b, a, ori_ppg[0:1,:], axis=1) 
    ori_ppg[0:1,:] = filt_ppg
    ppg_bands = [4,8,14,31,45]
    ppg, video_list_ppg = process_de(ori_ppg,window_length =1, window_type='hanning',\
        time_sample_rate=100,frequency_sample_rate=1024,bands=ppg_bands)

    return eeg, gsr, ppg, video_list_eeg, video_list_gsr, video_list_ppg, label

def process_rest_sate(signal, video_list):
    # split all sample into traing set and testing set, 8 : 2
    signal = signal.transpose(1,0,2)
    video_list = video_list.reshape(-1,1).astype("int")

    # process resting state eeg
    trial_eeg = np.where(video_list< 40)[0] # tag 40 represents resting state among expriment
    signal = signal[trial_eeg]
    video_list = video_list[trial_eeg]

    return signal, video_list

def make_model():
    model=models.vgg16(pretrained=True)
    model=model.eval()	
    model.cuda()	
    return model

def process_vid(model, path):
    video_feas = []
    model.eval()
    img_to_tensor = transforms.ToTensor()
    pooling = torch.nn.AvgPool2d(kernel_size=(4, 4))
    video_list = []
    for video in range(32):
        # print(video)
        video_path = os.path.join(path, str(video))
        frames = os.listdir(video_path)
        frames.sort(key=lambda l: int(l[:-4]))

        for frame in frames:
            frame_path = os.path.join(video_path, frame)
            img = cv2.imread(frame_path)
            img = cv2.resize(img,(128,128))
            img = img.transpose(2,0,1)
            tensor=img_to_tensor(img)
            tensor = tensor.reshape(1,3,128,128)
            tensor=tensor.cuda()

            result=model.features(Variable(tensor))
            result = pooling(result) # [1,512,4,4] -> [1,512,1,1]
            result_npy=result.data.cpu().numpy() # cuda.Tensor -> numpy.array
            
            result_npy = result_npy.reshape((32,16)) # [1,512,1,1] -> [32,16]
            result_mean = np.mean(result_npy,axis=1).reshape(1,-1) # [32,16] -> [32,1] -> [1,32]

            video_feas.append(result_mean)
            video_list.append(video)

    video_feas = np.vstack(video_feas)
    video_list = np.vstack(video_list)
    return np.array(video_feas), np.array(video_list)

def align_time(eeg, video_list_eeg, gsr, video_list_gsr, ppg, video_list_ppg, video, video_list_video):
    # check the length of eeg, gsr, ppg, video in each trial
    eeg_align = np.array([])
    gsr_align = np.array([])
    ppg_align = np.array([])
    video_align = np.array([])
    video_list = np.array([])

    for trial in range(32):
        eeg_list = np.where(video_list_eeg == trial)[0]
        gsr_list = np.where(video_list_gsr == trial)[0]
        ppg_list = np.where(video_list_ppg == trial)[0]
        vid_list = np.where(video_list_video == trial)[0]

        # print(eeg_list)
        if not (len(eeg_list) == len(gsr_list) and len(eeg_list) == len(ppg_list) and len(eeg_list) == len(vid_list)):
            print("the length does not match:",trial, len(eeg_list), len(gsr_list), len(ppg_list), len(vid_list))
            len_min = np.min([len(eeg_list), len(gsr_list), len(ppg_list), len(vid_list)])
            eeg_list = eeg_list[-len_min:]
            # print(eeg_list)
        if trial == 0:
            eeg_align = np.array(eeg[eeg_list])
            gsr_align = np.array(gsr[eeg_list])
            ppg_align = np.array(ppg[eeg_list])
            video_align = np.array(video[eeg_list])
            video_list = np.array(video_list_eeg[eeg_list])
        else:
            eeg_align = np.concatenate([eeg_align, eeg[eeg_list]], axis=0)
            gsr_align = np.concatenate([gsr_align, gsr[eeg_list]], axis=0)
            ppg_align = np.concatenate([ppg_align, ppg[eeg_list]], axis=0)
            video_align = np.concatenate([video_align, video[eeg_list]], axis=0)
            video_list = np.concatenate([video_list, video_list_eeg[eeg_list]], axis=0)

    return eeg_align, gsr_align, ppg_align, video_align, video_list

def split_sample(eeg, gsr, ppg, video, video_list, label):
    # shuffle the data and label
    shuffle_ix = np.random.permutation(np.arange(eeg.shape[0]))
    eeg = eeg[shuffle_ix]
    gsr = gsr[shuffle_ix]
    ppg = ppg[shuffle_ix]
    video = video[shuffle_ix]
    video_list = video_list[shuffle_ix]
    mixture_label = np.zeros((eeg.shape[0], label.shape[1]))
    for i in range(video_list.shape[0]):  # generate label
        mixture_label[i] = label[video_list[i,0]]

    split_samples = eeg.shape[0] // 10 * 8
    return eeg[:split_samples], gsr[:split_samples], ppg[:split_samples], video[:split_samples],mixture_label[:split_samples],\
        eeg[split_samples:], gsr[split_samples:], ppg[split_samples:], video[split_samples:],mixture_label[split_samples:]

if __name__ == '__main__':
    root_path = './DMER/MixedEmoR/'

    for subject in range(35):
        print(subject + 1)
        if (subject + 1) in [3, 4, 13, 16, 17, 27, 31]:
            continue
        else:
            sub_path = os.path.join(root_path, str(subject+1))

            # process eeg gsr ppg
            process_eeg, process_gsr, process_ppg, video_list_eeg, video_list_gsr, video_list_ppg, label = process_physiological(sub_path)
            print("the shape of process feature", process_eeg.shape, video_list_eeg.shape, process_gsr.shape, video_list_gsr.shape,\
                                process_ppg.shape, video_list_ppg.shape, label.shape)
            # remove  resting state
            process_eeg, video_list_eeg = process_rest_sate(process_eeg, video_list_eeg)
            process_ppg, video_list_ppg = process_rest_sate(process_ppg, video_list_ppg)
            process_gsr, video_list_gsr = process_rest_sate(process_gsr, video_list_gsr)
            print("after process resting state:", process_eeg.shape, video_list_eeg.shape, process_gsr.shape, video_list_gsr.shape,\
                                process_ppg.shape, video_list_ppg.shape,label.shape)

            # process video
            model=make_model()
            process_video, video_list_video = process_vid(model, sub_path)
            print("the shape of video process feature: ", process_video.shape, video_list_video.shape)

            # align the length of eeg, gsr, ppg and video
            process_eeg, process_ppg, process_gsr, process_video, video_list = align_time(process_eeg, video_list_eeg, \
                                                            process_gsr, video_list_gsr, process_ppg, video_list_gsr, process_video, video_list_video)
            print("the shape of align feature: ", process_eeg.shape, process_ppg.shape, process_gsr.shape, process_video.shape, video_list.shape)
            
            # split training set and testing set
            train_eeg, train_gsr, train_ppg, train_video, train_label, \
                test_eeg, test_gsr, test_ppg, test_video, test_label = split_sample(process_eeg,process_gsr, process_ppg, process_video,video_list, label)

            print("after split train/test dataset:", train_eeg.shape, test_eeg.shape)

            # save data into mat
            result_path = os.path.join(sub_path, 'datas_all_fea_{}.mat'.format(subject+1))
            sio.savemat(result_path, {'train_eeg': train_eeg, 'train_gsr': train_gsr, 'train_ppg': train_ppg, 'train_video':train_video, \
                'train_label': train_label, 'test_eeg':test_eeg, 'test_gsr':test_gsr,'test_ppg':test_ppg, 'test_video':test_video, \
                'test_label': test_label})

            print("==========  successfully generate feature mat of subject: ", str(subject+1), "==========")
            
