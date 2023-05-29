import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import timedelta

from utils.save_functions import save_checkpoint, visualize_losses
from utils.config import Config
from utils.data_CPC import LibriDataset
from utils.model import CPC_model
from utils.infoNCE import InfoNCELoss

start_time = time.time()

# the following lines of code are for the sake of using the debugger
if os.path.split(os.getcwd())[-1] != '5aua0-2022-group-18':
    os.chdir('5aua0-2022-group-18')
# print(os.getcwd())

cfg = Config()
trainset = LibriDataset('train')
testset = LibriDataset('test')
DL_train = DataLoader(trainset, batch_size=cfg.batch_size_train, shuffle=True, drop_last=True)
DL_val = DataLoader(testset, batch_size=cfg.batch_size_test, shuffle=False, drop_last=True)

def make_save_dir():
    save_dir = os.path.join(os.getcwd(), f'trained_models/{cfg.output_name}')
    if os.path.exists(save_dir):
        answer = input(f'Directory {save_dir} exists. Overwrite current files in the directory? [yes/no] ')
        if answer == 'no': 
            print("Please rename variable 'output_name' in your config file")
            quit()
    else: os.mkdir(save_dir)
    return save_dir


def train(model, DL_train):
    # Configuration settings
    criterion = InfoNCELoss(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                steps_per_epoch=int(len(DL_train)),
                                                epochs=cfg.epochs,
                                                anneal_strategy='linear')
    # Initialize model
    model.train()
    print(f'using device {device}')
    model = model.to(device)

    # Initialize empty lists for training metrics
    train_metrics = {'Epoch': [], 'Loss': [], 'Accuracy': []}
    val_metrics = {'Epoch': [], 'Loss': [], 'Accuracy': []}

    #Training loop
    for epoch in range(cfg.epochs):
        start_epoch_time = time.time()
        epoch+=1
        running_loss = 0
        correct_pred = 0
        total_pred = 0
        for i, data in enumerate(DL_train):
            i += 1 # start at iteration 1, not 0
            current_input = data[0].to(device)
            future_inputs = data[1:]
            
            # pass input through encoder and get predictions of future latent representations as dict:
            latent_predictions = model(current_input, generate_predictions=True)
            
            # get the positive samples
            positive_samples = {}
            for future_step, future_input in enumerate(future_inputs):
                future_input = future_input.to(device)
                positive_samples["k+"+str(future_step+1)] = model(future_input, generate_predictions=False) 
            
            loss, correct_pred_batch = criterion(latent_predictions, positive_samples)
            
            # Backprop
            loss.backward()
            # Take a step with the optimizer
            optimizer.step()
            # Take a step with the learning rate schedule
            scheduler.step()

            #Update loss
            running_loss += loss.item()
            correct_pred += correct_pred_batch
            total_pred += current_input.shape[0]
            
            if i % cfg.log_iterations==0:
                acc = np.mean(correct_pred/total_pred)
                avg_loss = running_loss / i
                print(f'Epoch: {epoch}, Iter in epoch: {i}, Loss: {avg_loss:.2f}, Accuracies: {acc:.2f}')

        # Print stats at the end of the epoch
        num_batches = len(DL_train)
        avg_loss = running_loss / num_batches
        acc = correct_pred/total_pred * 100 # as percentage
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        
        train_metrics['Epoch'].append(epoch)
        train_metrics['Loss'].append(avg_loss)
        train_metrics['Accuracy'].append(acc)

        val_loss, val_acc = validation(model, DL_val, device, criterion)
        val_metrics['Epoch'].append(epoch)
        val_metrics['Loss'].append(val_loss)
        val_metrics['Accuracy'].append(val_acc)
        
        if epoch ==1: save_dir = make_save_dir()
        visualize_losses(save_dir, train_metrics, val_metrics)
        save_checkpoint(save_dir, model, epoch)

        print(f'Epoch {epoch} took a total time of {str(timedelta(seconds=(time.time() - start_time)))}')
        
    print('Finished Training')


def validation(model, DL_val, device, criterion):
    running_loss = 0
    correct_pred = 0
    total_pred = 0
    
    model.eval()

    for data in DL_train:
        current_input = data[0].to(device)
        future_inputs = data[1:]
        
        # pass input through encoder and get predictions of future latent representations as dict:
        latent_predictions = model(current_input, generate_predictions=True)
        
        # get the positive samples
        positive_samples = {}
        for future_step, future_input in enumerate(future_inputs):
            future_input = future_input.to(device)
            positive_samples["k+"+str(future_step+1)] = model(future_input, generate_predictions=False) 
        
        loss, correct_pred_batch = criterion(latent_predictions, positive_samples)

        #Update loss
        running_loss += loss.item()
        correct_pred += correct_pred_batch
        total_pred += current_input.shape[0]

    loss = running_loss / len(DL_val)
    acc = correct_pred/total_pred * 100 # as percentage

    model.train() # not sure we need this, best ask supervisor

    return loss, acc
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = CPC_model(device)
    
if __name__ == "__main__":
  train(myModel, DL_train)
  print(f'Program ran for a total time of {str(timedelta(seconds=(time.time() - start_time)))}')
