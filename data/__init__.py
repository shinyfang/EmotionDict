import importlib
from data.base_dataset import BaseDataset
import torch

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

class CustomDataset():
    def __init__(self, opt, phase):
        self.opt = opt
        self.shuffle = True

        if phase == "test":
            dataset_class = find_dataset_using_name(opt.dataset_mode)
            test_opt = self.opt
            test_opt.phase = phase
            self.dataset = dataset_class(test_opt)
            self.dataset.opt = phase
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                drop_last =True,
                num_workers=0)

        else:
            self.opt.batch_size = opt.batch_size
            self.shuffle = True

            dataset_class = find_dataset_using_name(opt.dataset_mode)
            self.dataset = dataset_class(self.opt)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.opt.batch_size,
                shuffle=self.shuffle,
                drop_last =True,
                num_workers=int(opt.num_threads))
        
    def load_data(self):
        return self.dataloader

def create_dataset(opt, phase="train"):
    dataset = CustomDataset(opt, phase)
    data_loader = dataset.load_data()

    return data_loader