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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(1024, 256), nn.BatchNorm1d(256))
        self.fc2 = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(256, 1), )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def init_weights(m):
    torch.nn.init.kaiming_uniform_(m.fc1[0].weight)
    m.fc1[0].bias.data.fill_(0.)
    torch.nn.init.kaiming_uniform_(m.fc2[1].weight)
    m.fc2[1].bias.data.fill_(0.1)
