import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import os
import shutil
import pandas as pd
import json

def visualize_losses(save_dir, train_metrics, val_metrics):
    
    pd.DataFrame(train_metrics).to_csv(os.path.join(save_dir,'train_metrics.csv'), index=False)
    pd.DataFrame(val_metrics).to_csv(os.path.join(save_dir,'val_metrics.csv'), index=False)
    
    _, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(train_metrics['Epoch'], train_metrics['Accuracy'], linestyle='-', color='blue', label='train')
    ax[0].plot(val_metrics['Epoch'], val_metrics['Accuracy'], linestyle='-', color='orange', label='val')
    ax[0].set_title('Accuracy [%]')
    ax[0].legend()

    ax[1].plot(train_metrics['Epoch'], train_metrics['Loss'], linestyle='-', color='blue', label='train')
    ax[1].plot(val_metrics['Epoch'], val_metrics['Loss'], linestyle='-', color='orange', label='val')
    ax[1].set_title('Loss value')
    ax[1].legend()

    ax[1].set_xlabel('Training epoch')    
    plt.savefig(os.path.join(save_dir, 'loss_plots.png'))

    print('Plots containing losses and mIoU have been saved.')
    return


def save_checkpoint(save_dir, model, train_epochs):
    save_name = 'ckpt_' + str(train_epochs) + 'epochs.pth'
    save_path = os.path.join(save_dir, save_name)
    
    # remove checkpoint from previous epoch
    for file in os.listdir(save_dir): 
        if file.endswith('epochs.pth'): os.remove(os.path.join(save_dir, file))

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

    print('Model checkpoint saved.')
    return

# function to save my config class to txt file
# def save_config(save_dir):
#     with open(os.path.join('utils/','config.py'), 'r') as f:
#         lines = f.readlines()
#     with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
#         f.writelines(lines)
#     return

def save_config(config, save_dir):
    with open(os.path.join(save_dir, 'config.txt'), 'w') as file:
        json.dump(config.__dict__, file, indent=4)