import argparse
from data import create_dataset
import torch
import numpy as np
from model import create_model
from utils.utils import *

def parse_args():
    desc = 'mixture emotion recognition implemented by Pytorch'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test / domain]')
    # parameters for loading data
    parser.add_argument('--root_path', type=str, default='MixedEmoR', help='the root path of dataset')
    parser.add_argument('--subject', type=int, default=10, help='the number of training subject')
    parser.add_argument('--win_len', type=int, default=1, help='the length of slidding window')
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--dataset_mode', type=str, default='dmer') # mixture dataset
    # parameters for training setting
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dict_dim', type=int, default=32, help='learning rate')
    parser.add_argument('--epochs', type=int, default=301, help='train epochs')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model', type=str, default='transform',help='different model framework')
    parser.add_argument('--model_name', type=str, default='test',help='different model names')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',help='different model names')
    parser.add_argument('--netG', type=str, default='basic',help='different model names')
    parser.add_argument('--netD', type=str, default='basic',help='different model names')
    parser.add_argument('--num_class', type=int, default=2, help='the number of calss categories')
    

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    str_ids = opt.gpu_ids.split(',')  # split the GPU number
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    if opt.phase == 'train':
        dataset = create_dataset(opt)
        dataset_size = len(dataset)
        print('The total samples of training dataset = %d' % dataset_size)

        test_dataset = create_dataset(opt,phase = "test")
        test_dataset_size = len(test_dataset)
        print('The total samples of test dataset = %d' % test_dataset_size)

        # build the model
        model = create_model(opt)

        # log file
        log_train=open('log_{}_{}.txt'.format(opt.model, "train"),mode='a+')
        log_val = open('log_{}_{}.txt'.format(opt.model, "val"), mode='a+')
        log_train.write("================== subject: {} ===================\n".format(opt.subject))
        log_val.write("================== subject: {} ===================\n".format(opt.subject))

        # train loop
        for epoch in range(opt.epochs):
            model.train()
            for i, data in enumerate(dataset):

                model.set_input(data)
                model.optimize_parameters()

                if i % 100 == 0:
                    loss, loss_cls, loss_mne,loss_kl = model.current_loss()
                    print("epoch %d, iter %d, loss %f, loss_cls %f, loss_mne %f, loss_kl %f" %(epoch, i, loss, loss_cls, loss_mne,loss_kl))
                    log_train.write("epoch %d, iter %d, loss %f, loss_cls %f, loss_mne %f, loss_kl %f" %(epoch, i, loss, loss_cls, loss_mne,loss_kl))
            acc = np.array([0,0,0,0,0,0]).astype(float)

            model.eval()
            for i, data in enumerate(test_dataset):

                model.set_test_input(data)
                predict, label = model.test()
                acc = acc+ compute_distance(predict[0], label[0])

            print("================== validation acc:", acc / test_dataset_size, "=============")
            log_val.write("epoch: {}, acuracy: {} \n".format(epoch, acc / test_dataset_size) )


            # save network
            if epoch % 50 == 0:
                print('saving the latest model (epoch %d)' % (epoch))
                model.save(epoch)
        log_train.close()
        log_val.close()
    else: # opt.phase = 'test'
        print(opt.phase)
        test_dataset = create_dataset(opt,phase = "test")
        test_dataset_size = len(test_dataset)
        print('The total samples of test dataset = %d' % test_dataset_size)

        model = create_model(opt)
        model.load(epoch=50) # set pretrian model here

        acc = np.array([0,0,0,0,0,0]).astype(float)
        for i, data in enumerate(test_dataset):
            model.eval()
            model.set_test_input(data)
            predict, label = model.test()
            log=open('log_{}_{}.txt'.format(opt.model, opt.phase),mode='a+')
            log.write("predict: {}, label: {} \n".format(predict, label))
            log.close()
            acc = acc+ compute_distance(predict[0], label[0])
        print("================== validation acc:", acc / test_dataset_size, "=============")
        # write into log file
        log=open('log_{}_{}.txt'.format(opt.model, opt.phase),mode='a+')
        log.write("acuracy: {} \n".format(acc / test_dataset_size) )
        log.close()


    
            


