import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import timedelta

from utils.save_functions import *
from utils.config import CONFIG
from utils.data import LibriDataset
from utils.model import CPC_model, Classifier
from utils.infoNCE import InfoNCELoss

# the following lines of code are for the sake of using the debugger
if os.path.split(os.getcwd())[-1] != '5aua0-2022-group-18':
    os.chdir('5aua0-2022-group-18')
# print(os.getcwd())

cfg = CONFIG
print(cfg)

def make_save_dir():
    save_dir = os.path.join(os.getcwd(), f'trained_models/{cfg.output_name}')
    if os.path.exists(save_dir):
        answer = input(f'Directory {save_dir} exists. Overwrite current files in the directory? [yes/no] ')
        if answer == 'no': 
            print("Please rename variable 'output_name' in your config file")
            quit()
    else: os.mkdir(save_dir)
    return save_dir


def get_CPC_loss(data, criterion, model, device, batch_size):
    data = data.to(device) # tensor of size (batch_size, n_past_latents+n_predictions+1, channels, height, width)
    data = torch.transpose(data, 0, 1) # convert to size (n_past_latents+n_predictions+1, batch_size, channels, height, width)
    past_present_inputs = data[0:cfg.n_past_latents+1] 
    future_inputs = data[cfg.n_past_latents+1:]
    
    # pass input through encoder and get past latent embeddings & predictions of future latent representations as dict:
    latent_predictions, past_latents = model(past_present_inputs, generate_predictions=True)

    # get the positive samples
    positive_samples = {}
    for future_step, future_input in enumerate(future_inputs):
        future_input = future_input
        positive_samples["k+"+str(future_step+1)] = model(future_input, generate_predictions=False)  
    
    loss, correct_pred_batch = criterion(latent_predictions, positive_samples, past_latents, batch_size)
    
    return loss, correct_pred_batch


def get_supervised_loss(data, criterion, model, device, batch_size):
    inputs, labels = data[0].to(device), data[1].to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    _, pred = torch.max(outputs,1)
    correct_pred_batch = (pred == labels).sum().item()
    return loss, correct_pred_batch


def train(model, DL_train, DL_val, loss_function, optimizer, criterion, device):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                steps_per_epoch=int(len(DL_train)),
                                                epochs=cfg.epochs,
                                                anneal_strategy='linear')
    # Initialize model
    model.train()
    model = model.to(device)

    # Initialize empty lists for training metrics
    train_metrics = {'Epoch': [], 'Loss': [], 'Accuracy': []}
    val_metrics = {'Epoch': [], 'Loss': [], 'Accuracy': []}
    
    save_dir = make_save_dir() # make save directory if it doesn't exist yet
    save_config(cfg, save_dir)
    
    #Training loop
    for epoch in range(cfg.epochs):
        start_epoch_time = time.time()
        epoch+=1
        running_loss = 0
        correct_pred = 0
        total_pred = 0
        for i, data in enumerate(DL_train):
            i += 1 # start at iteration 1, not 0
            
            loss, correct_pred_batch = loss_function(data, criterion, model, device, cfg.batch_size_train)
            
            optimizer.zero_grad()
            loss.backward() # Backprop
            optimizer.step() # Take a step with the optimizer
            scheduler.step() # Take a step with the learning rate scheduler

            #Update loss
            running_loss += loss.item()
            correct_pred += correct_pred_batch
            total_pred += cfg.batch_size_train

            if i % cfg.log_iterations==0:
                acc = np.mean(correct_pred/total_pred) * 100 # as percentage
                avg_loss = running_loss / i
                print(f'Epoch: {epoch}, Iter in epoch: {i}, Loss: {avg_loss:.2f}, Accuracies: {acc:.2f}')

        # Print stats at the end of the epoch
        num_batches = len(DL_train)
        avg_loss = running_loss / num_batches
        acc = np.mean(correct_pred/total_pred) * 100 # as percentage
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        
        train_metrics['Epoch'].append(epoch)
        train_metrics['Loss'].append(avg_loss)
        train_metrics['Accuracy'].append(acc)

        val_loss, val_acc = validation(model, DL_val, device, criterion)
        val_metrics['Epoch'].append(epoch)
        val_metrics['Loss'].append(val_loss)
        val_metrics['Accuracy'].append(val_acc)

        visualize_losses(save_dir, train_metrics, val_metrics)
        save_checkpoint(save_dir, model, epoch)

        print(f'Epoch {epoch} took a total time of {str(timedelta(seconds=(time.time() - start_epoch_time)))}')
    print('Finished Training')


def validation(model, DL_val, device, criterion):
    running_loss = 0
    correct_pred = 0
    total_pred = 0
    
    model.eval()

    for i, data in enumerate(DL_val):
        i += 1
        loss, correct_pred_batch = loss_function(data, criterion, model, device, cfg.batch_size_test)

        #Update loss
        running_loss += loss.item()
        correct_pred += correct_pred_batch
        total_pred += cfg.batch_size_test
        
    loss = running_loss / len(DL_val)
    acc = np.mean(correct_pred/total_pred) * 100 # as percentage

    model.train() # not sure we need this, best ask supervisor

    return loss, acc

    
if __name__ == "__main__":
    start_time = time.time()
    
    # Load data
    trainset = LibriDataset('train')
    testset = LibriDataset('test')  
    DL_train = DataLoader(trainset, batch_size=cfg.batch_size_train, shuffle=True, drop_last=True)
    DL_val = DataLoader(testset, batch_size=cfg.batch_size_test, shuffle=False, drop_last=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.CPC:
        model = CPC_model()
        loss_function = get_CPC_loss
        criterion = InfoNCELoss(device)
        print('Training with CPC using device {device}')
    else:
        model = Classifier()
        loss_function = get_supervised_loss
        criterion = nn.CrossEntropyLoss()
        if cfg.load_checkpoint != "":
            model.load_state_dict(torch.load(cfg.load_checkpoint, map_location='cpu'), strict = False)
            print(f'\nresuming training from following checkpoint: \n{cfg.load_checkpoint}')
        if cfg.freeze_encoder:
            model.convencoder.requires_grad_(False) # freeze encoder weights
        print('Training in a fully supervised manner using device {device}')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    
    # Train model
    train(model, DL_train, DL_val, loss_function, optimizer, criterion, device)
    
    print(f'Program ran for a total time of {str(timedelta(seconds=(time.time() - start_time)))}')
