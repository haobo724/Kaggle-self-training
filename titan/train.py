import torch
import numpy as np
from torch import nn
EPOCHS = 100

def caculate_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc*100

def start_train(model,train_dataloader,val_dataloader,device):
    loss_fn = nn.BCEWithLogitsLoss()
    LR = 0.005
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch + 1}\n{'-' * 10}")
        train_step(model, train_dataloader, loss_fn, optimizer, device,epoch)
        if epoch % 10 == 0:
            LR = LR * (0.5)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            validation_step(model, val_dataloader, loss_fn, device,epoch)
    return model
        
def train_step(model, data_loader, loss_fn, optimizer, device,epoch):
    train_loss, train_acc =0, 0
    model.train()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        targets = targets.unsqueeze(1)
        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)
        train_loss += loss.item()
        train_acc += caculate_accuracy(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent or adam step
        optimizer.step()
        # print(f"loss: {loss.item()}")
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Train loss: {train_loss:.5f}, Train accuracy: {train_acc:.2f}%")

def validation_step(model, data_loader, loss_fn, device,epoch):
    val_loss, val_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for x_val, y_val in data_loader:
            x_val, y_val = x_val.to(device).float(), y_val.to(device).float().unsqueeze(1)
            pred = model(x_val)
            loss = loss_fn(pred, y_val)
            val_loss += loss.item()
            val_acc += caculate_accuracy(pred,y_val)
        val_loss /= len(data_loader)
        val_acc /= len(data_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Validation loss: {val_loss:.5f}, Validation accuracy: {val_acc:.2f}%")