import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from data import *
from FS_model import *

# the following lines of code are for the sake of using the debugger
if os.path.split(os.getcwd())[-1] != '5aua0-2022-group-18':
    os.chdir('5aua0-2022-group-18')
print(os.getcwd())


trainset = LibriDataset('train')
testset = LibriDataset('test')
cfg = Config()
DL_train = DataLoader(trainset, batch_size=cfg.batch_size_train, shuffle=True)
DL_val = DataLoader(testset, batch_size=cfg.batch_size_test, shuffle=False)

def train(model, DL_train):
    # Configuration settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                steps_per_epoch=int(len(DL_train)),
                                                epochs=cfg.epochs,
                                                anneal_strategy='linear')
    # Initialize model
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  

    #Training loop
    for epoch in range(cfg.epochs):
        running_loss = 0
        correct_pred = 0
        total_pred = 0

        for i, data in enumerate(DL_train):
            inputs, labels = data[0].to(device), data[1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
            print(i)

        # Print stats at the end of the epoch
        num_batches = len(DL_train)
        avg_loss = running_loss / num_batches
        acc = correct_pred/total_pred
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('Finished Training')
    save_path = 'model.pth'
    torch.save(model.state_dict(), save_path)
    print("Saved trained model as {}.".format(save_path))


myModel = Model()
if __name__ == "__main__":
  train(myModel, DL_train)
