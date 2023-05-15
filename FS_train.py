import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from data import *
from FS_model import Model


def train():
    # Configuration settings
    cfg = Config()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([cfg.adam1, cfg.adam2],lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                steps_per_epoch=int(len(DL_train)),
                                                epochs=cfg.epochs,
                                                anneal_strategy='linear')
     # Initialize model
    model = Model(cfg)
    model.train()
    model = model.to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Training loop
    for epoch in range(cfg.epochs):
        running_loss = 0
        correct_pred = 0
        total_pred = 0

        for i, data in enumerate(DL_train):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs,labels)
            # Backprop
            loss.backward()
            # Take a step with the optimizer
            optimizer.step()
            # Take a step with the learning rate schedule
            scheduler.step()

            #Update loss
            running_loss += loss.item()
            _, pred = torch.max(outputs,1)

            correct_pred += (pred == labels).sum().item()
            total_pred += pred.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(DL_train)
        avg_loss = running_loss / num_batches
        acc = correct_pred/total_pred
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('Finished Training')
    save_path = 'model.pth'
    torch.save(model.state_dict(), save_path)
    print("Saved trained model as {}.".format(save_path))
  
if __name__ == "__main__":
  train()
