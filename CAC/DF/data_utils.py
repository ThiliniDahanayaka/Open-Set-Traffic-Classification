import torch
# import torchvision
# import torchvision.transforms as tf
import json
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import random
random.seed(1000)

class customDataset(Dataset):
    def __init__(self, data, target, shuffle=False):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.shuffle = shuffle

        if self.shuffle:
        	ind=np.arange(self.data.shape[0])
        	np.random.shuffle(ind)
        	self.data = self.data[ind]
        	self.target = self.target[ind]

        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def get_train_loaders(datasetName, trial_num, cfg):
    """
        Create training dataloaders.

        datasetName: name of dataset
        trial_num: trial number dictating known/unknown class split
        cfg: config file

        returns trainloader, evalloader, testloader, mapping - changes labels from original to known class label
    """
    trainSet, valSet, testSet, _ = load_datasets(datasetName, cfg, trial_num)
    
    # print(type(trainSet))
    batch_size = cfg['batch_size']

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers = 0)
    valloader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers = 0)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers = 0)

    mapping=None

    # print(type(trainloader))

    return trainloader, valloader, testloader, mapping

def get_eval_loaders(datasetName, trial_num, cfg):
    # print('*****************************************************************************************\n get_eval_loaders To be implemented\n ****************************************************************************************************')
    _, _, testSet, unknownSet = load_datasets(datasetName, cfg, trial_num)

    batch_size = cfg['batch_size']

    knownloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False)
    unknownloader = torch.utils.data.DataLoader(unknownSet, batch_size=batch_size, shuffle=False)

    mapping=None

    return knownloader, unknownloader, mapping



def get_data_stats(dataset, known_classes):
    print('*****************************************************************************************\n get_data_stats To be implemented\n ****************************************************************************************************')


def load_datasets(datasetName, cfg, trial_num):
    if datasetName == "DF":
        datatpath = '/media/SATA_1/thilini_open_extra/final_datasets/DF/'
    else:
        print('unknown dataset')
        return
        
    x = np.load(datatpath+'X_train.npy').astype('float32')
    y = np.load(datatpath+'y_train.npy').astype('float32')
    # print(y.max())

    x = x[:, 0:1500, np.newaxis]
    x = np.swapaxes(x, 2, 1)
    trainSet = x, y

    trainSet = customDataset(data=x, target=y, shuffle=True)

    x = np.load(datatpath+'X_valid.npy').astype('float32')
    y = np.load(datatpath+'y_valid.npy').astype('float32')

    x = x[:, 0:1500, np.newaxis]
    x = np.swapaxes(x, 2, 1)
    valSet = customDataset(data=x, target=y)

    x = np.load(datatpath+'X_test.npy').astype('float32')
    y = np.load(datatpath+'y_test.npy').astype('float32')

    x = x[:, 0:1500, np.newaxis]
    x = np.swapaxes(x, 2, 1)
    testSet = customDataset(data=x, target=y)


    open_l = cfg['num_known_classes']
    x = np.load(datatpath+'X_open.npy').astype('float32')
    y = (np.ones((x.shape[0],))*open_l).astype('float32')

    x = x[:, 0:1500, np.newaxis]
    x = np.swapaxes(x, 2, 1)
    unknownSet = customDataset(data=x, target=y)

    # print(type(trainSet))
    # print(type(valSet))
    # print(type(testSet))
    # print(type(unknownSet))

    return trainSet, valSet, testSet, unknownSet

def get_anchor_loaders(datasetName, trial_num, cfg):
    # print('*****************************************************************************************\n get_anchor_loaders To be implemented\n ****************************************************************************************************')
    trainSet = load_anchor_datasets(datasetName, cfg, trial_num)
    
    # print(type(trainSet))
    batch_size = cfg['batch_size']

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers = 0)
    # valloader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers = 0)
    # testloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers = 0)

    # mapping=None

    # print(type(trainloader))

    return trainloader

def load_anchor_datasets(datasetName, cfg, trial_num):
    # print('*****************************************************************************************\n load_anchor_datasets To be implemented\n ****************************************************************************************************')
    if datasetName == "DF":
        datatpath = '/media/SATA_1/thilini_open_extra/final_datasets/DF/'
    else:
        print('unknown dataset')
        return
    x = np.load(datatpath+'X_train.npy').astype('float32')
    y = np.load(datatpath+'y_train.npy').astype('float32')
    # print(y.max())

    x = x[:, 0:1500, np.newaxis]
    x = np.swapaxes(x, 2, 1)
    trainSet = x, y

    trainSet = customDataset(data=x, target=y, shuffle=True)

    return trainSet

def create_dataSubsets(dataset, classes_to_use, idxs_to_use = None):
    print('*****************************************************************************************\n create_dataSubsets To be implemented\n ****************************************************************************************************')


def create_target_map(known_classes, num_classes):
    print('*****************************************************************************************\n create_target_map To be implemented\n ****************************************************************************************************')
