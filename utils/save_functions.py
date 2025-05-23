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


def save_checkpoint(save_dir, model, train_epochs, val_metrics):
    save_name = 'ckpt_' + str(train_epochs) + 'epochs.pth'
    save_path = os.path.join(save_dir, save_name)
    
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

    print('Model checkpoint saved.')
    return

def save_config(config, save_dir, dataPercentageTrain, dataPercentageTest):
    with open(os.path.join(save_dir, 'config.txt'), 'w') as file:
        json.dump(config.__dict__, file, indent=4)
        file.close()

    with open(os.path.join(save_dir, 'config.txt'), 'a') as file:
        file.writelines(["TRAIN: Percentage of amount samples used is ", str(dataPercentageTrain), "% \n"])
        file.writelines(["TEST: Percentage of amount samples used is ", str(dataPercentageTrain), "% \n"])
        file.close()