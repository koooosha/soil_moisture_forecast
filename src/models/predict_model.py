import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_sequence
from src.models.dl_model import LSTMModel
import numpy as np
import yaml

def lstm_predict(model:LSTMModel,x_new:np.array,config:dict):
    '''This function predict values for a new sample'''
    if len(x_new.shape) == 2:
        X_new_tensor = torch.tensor(x_new, dtype=torch.float).unsqueeze(0)
    else:
        X_new_tensor = torch.tensor(x_new, dtype=torch.float)
    batch_size = config['Training']['batch_size']
    new_dataset = TensorDataset(X_new_tensor)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)
    # Make predictions on the new data
    model.eval()
    with torch.no_grad():
        predictions = []
        for inputs in new_loader:
            outputs = model(inputs[0])
            predictions.extend(outputs.numpy())
    return predictions