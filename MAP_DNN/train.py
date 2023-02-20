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

def train_and_test(model,n_epochs,train_data_loader, valid_data_loader,test_data_loader,ratio,loss_fn,learning_rate=0.001):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, )

    for epoch in range(n_epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.to(device)
        model.train()
        for batch in train_data_loader:

            optimizer.zero_grad()
            input, post, pre = batch
            input, post, pre = input.to(device), post.to(device), pre.to(device)
            output = model(input)
            loss1 = loss_fn(output, post, pre,ratio)
            torch_sum = torch.sum(loss1, dim=0)
            (torch_sum).backward()
            optimizer.step()
            training_loss += torch_sum
        training_loss /= len(train_data_loader.dataset)
        model.eval()
        for batch in valid_data_loader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                input, post, pre = batch
                input, post, pre = input.to(device), post.to(device), pre.to(device)
                output = model(input)
                loss1 = loss_fn(output, post, pre,ratio)
                torch_sum = torch.sum(loss1, dim=0)
                valid_loss += torch_sum
        valid_loss /= len(valid_data_loader.dataset)
        print(f"epoch={epoch};train_loss={training_loss.detach().item()};valid_loss={valid_loss.item()}")

    test_loss = 0
    model.eval()
    model.to(device)
    for batch in test_data_loader:
        input, post, pre = batch
        input, post, pre = input.to(device), post.to(device), pre.to(device)
        output = model(input)
        loss1 = loss_fn(output, post, pre,ratio)
        torch_sum = torch.sum(loss1, dim=0)
        test_loss += torch_sum.detach()
    test_loss /= len(test_data_loader.dataset)
    print(test_loss.item())
    return model
