# Peiyi Leng; edsml-pl1023
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from ..utils import set_seed
from livelossplot import PlotLosses
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


def train(model, optimizer, criterion, data_loader, device,
          classification=False):
    model.train()
    train_loss, train_accuracy = 0, 0
    for X, y in tqdm(data_loader):
        # do y-1 because my data's label starts from 1
        if classification:
            X, y = X.to(device), (y - 1).to(device)
        else:
            X, y = X.to(device), y.to(device).float()
        optimizer.zero_grad()
        if X.shape[0] == 1:
            continue  # Skip this batch if it only contains one sample
        output = model(X.view(-1, 1, 64, 64, 64))
        if not classification:
            output = output.squeeze()
        loss = criterion(output, y)
        loss.backward()
        train_loss += loss*X.size(0)
        if classification:
            y_pred = F.log_softmax(output, dim=1).max(1)[1]
            train_accuracy += accuracy_score(
                y.cpu().numpy(),
                y_pred.detach().cpu().numpy())*X.size(0)
        else:
            y_pred = torch.round(output).long()
            train_accuracy += accuracy_score(
               y.cpu().numpy(),
               y_pred.detach().cpu().numpy().reshape(-1)) * X.size(0)
        optimizer.step()

    return (train_loss/len(data_loader.dataset),
            train_accuracy/len(data_loader.dataset))


def validate(model, criterion, data_loader, device, classification=False):
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in tqdm(data_loader):
        with torch.no_grad():
            if classification:
                X, y = X.to(device), (y - 1).to(device)
            else:
                X, y = X.to(device), y.to(device).float()
            if X.shape[0] == 1:
                continue  # Skip this batch if it only contains one sample
            output = model(X.view(-1, 1, 64, 64, 64))
            if not classification:
                output = output.squeeze()

            loss = criterion(output, y)
            validation_loss += loss*X.size(0)
            if classification:
                y_pred = F.log_softmax(output, dim=1).max(1)[1]
                validation_accuracy += accuracy_score(
                   y.cpu().numpy(), y_pred.cpu().numpy())*X.size(0)
            else:
                y_pred = torch.round(output).long()
                validation_accuracy += accuracy_score(
                    y.cpu().numpy(),
                    y_pred.detach().cpu().numpy().reshape(-1)) * X.size(0)

    return (validation_loss/len(data_loader.dataset),
            validation_accuracy/len(data_loader.dataset))


def evaluate(model, data_loader, device, classification=False):
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            output = model(X.view(-1, 1, 64, 64, 64))
            # do +1 to match original data label
            if classification:
                y_pred = F.log_softmax(output, dim=1).max(1)[1] + 1
            else:
                y_pred = torch.round(output).long().squeeze()
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, 0),  np.concatenate(ys, 0)


def train_model(model, lr, momentum, epoch, train_loader,
                val_loader, device, classification=False):
    set_seed(42)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=0.05)
    if classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    liveloss = PlotLosses()
    try:
        for epoch in range(epoch):
            logs = {}
            train_loss, train_accuracy = train(
                model, optimizer, criterion, train_loader,
                device, classification=classification)

            logs['' + 'log loss'] = train_loss.item()
            logs['' + 'accuracy'] = train_accuracy.item()

            validation_loss, validation_accuracy = validate(
                model, criterion, val_loader,
                device, classification)

            logs['val_' + 'log loss'] = validation_loss.item()
            logs['val_' + 'accuracy'] = validation_accuracy.item()

            liveloss.update(logs)
            liveloss.draw()

    except KeyboardInterrupt:
        print("Training interrupted, returning the latest model...")
        return model

    return model
