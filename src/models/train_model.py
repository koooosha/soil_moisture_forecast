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

def lstm_train(x_train:np.array,y_train:np.array,config:dict):
    '''
        This function creates and trains a lstm model for the soil moisture forecast problem.
    '''
    X_tensor = torch.tensor(x_train, dtype=torch.float32)
    Y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, \
        test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    batch_size = config['Training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_size = x_train.shape[2]
    hidden_size = config['Training']['hidden_size']
    output_size = config['Training']['output_size']
    model = LSTMModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["Training"]['lr'])
    # Train the model
    num_epochs = config['Training']['num_epochs']
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss / len(val_loader))

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

    return model,train_losses,val_losses

def lstm_eval(model:LSTMModel,x_test:np.array,y_test:np.array,config:dict):
    '''This function predict values for the test data sets'''
    X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    batch_size = config['Training']['batch_size']
    # Create DataLoader for the test set
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Evaluate the model on the test set
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        predictions = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predictions.extend(outputs.numpy())
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    return predictions