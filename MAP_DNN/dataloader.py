import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import DataStructs
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem as Chem
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)
from sklearn.model_selection import train_test_split

class ECFPs_From_SMILES(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X):
        N = len(X)
        vector = np.zeros((N, 1024), dtype=np.int8)
        for i in range(N):
            bit = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(X[i]), 3, 1024)
            fp = np.zeros((1, 1024), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bit, fp)
            vector[i] = fp
        return vector


transformer_ = ECFPs_From_SMILES()


def load_data(file, target_str, pre_str, size=1024):
    train_set, valid_test_set = train_test_split(file, test_size=0.2, random_state=0)
    valid_set, test_set = train_test_split(valid_test_set, test_size=0.5, random_state=0)
    train_file = train_set.copy()
    valid_file = valid_set.copy()
    test_file = test_set.copy()

    train_list = list(train_file["smiles"])
    train_post = (train_file[target_str].values).reshape(-1, 1)
    train_pre = (train_file[pre_str].values).reshape(-1, 1)

    valid_list = list(valid_file["smiles"])
    valid_post = (valid_file[target_str].values).reshape(-1, 1)
    valid_pre = (valid_file[pre_str].values).reshape(-1, 1)

    test_list = list(test_file["smiles"])
    test_post = (test_file[target_str].values).reshape(-1, 1)
    test_pre = (test_file[pre_str].values).reshape(-1, 1)

    class train_dataset(Dataset):
        def __init__(self):
            self.Data = np.asarray(transformer_.transform(train_list))
            self.Post = np.asarray(train_post)
            self.Pre = np.asarray(train_pre)

        def __getitem__(self, index):
            ECFP = torch.from_numpy(self.Data[index]).float()
            post = torch.tensor(self.Post[index]).float()
            pre = torch.tensor(self.Pre[index]).float()
            return ECFP, post, pre

        def __len__(self):
            return len(self.Data)

    class valid_dataset(Dataset):
        def __init__(self):
            self.Data = np.asarray(transformer_.transform(valid_list))
            self.Post = np.asarray(valid_post)
            self.Pre = np.asarray(valid_pre)

        def __getitem__(self, index):
            ECFP = torch.from_numpy(self.Data[index]).float()
            post = torch.tensor(self.Post[index]).float()
            pre = torch.tensor(self.Pre[index]).float()
            return ECFP, post, pre

        def __len__(self):
            return len(self.Data)

    class test_dataset(Dataset):
        def __init__(self):
            self.Data = np.asarray(transformer_.transform(test_list))
            self.Post = np.asarray(test_post)
            self.Pre = np.asarray(test_pre)

        def __getitem__(self, index):
            ECFP = torch.from_numpy(self.Data[index]).float()
            post = torch.tensor(self.Post[index]).float()
            pre = torch.tensor(self.Pre[index]).float()
            return ECFP, post, pre

        def __len__(self):
            return len(self.Data)

    torch.cuda.is_available()
    train_data = train_dataset()
    valid_data = valid_dataset()
    train_data_loader = DataLoader(train_data,
                                   batch_size=size,
                                   shuffle=True, )
    valid_data_loader = DataLoader(valid_data,
                                   batch_size=size,
                                   shuffle=False, )
    test_data = test_dataset()
    test_data_loader = DataLoader(test_data, batch_size=size, shuffle=False)
    return train_data_loader, valid_data_loader, test_data_loader
