import torch
import torch.nn as nn
import numpy as np
from random import choice

from utils.config import CONFIG
cfg = CONFIG

class InfoNCELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.k_future = ["k+"+str(i+1) for i in range(cfg.n_predictions)]
        self.k_past_present = ["k-"+str(i+1) for i in range(cfg.n_past_latents)] + ["k"]
        self.k_all = self.k_past_present + self.k_future
        self.device = device

    def forward(self, latent_predictions, positive_samples, past_latents, batch_size):
        '''
        Inputs:
            'latent_predictions': dict with keys 'k' to 'k+n_predictions', each item of size batch_sizex512
            'positive_samples': dict with keys 'k+1' to 'k+n_predictions', each item of size batch_sizex512
        Output:
            'loss': scalar loss
            'correct predictions': mean of correct predictions for each future latent space
        '''
        loss = 0
        negative_samples = {}
        correct_predictions = np.zeros((cfg.n_predictions))
        
        for future_step, k in enumerate(self.k_future):

            negative_samples[k] = torch.zeros((batch_size, cfg.n_negatives, 512)).to(self.device) # initialize empty tensor
            for batch_idx in range(batch_size): 
                negatives = torch.Tensor([]).to(self.device)
                
                for _ in range(cfg.n_negatives):
                    # choose random time index 'k-n_past_latents' to 'k+n_predictions'
                    time_index_choice = choice(self.k_all) 
                    # choose a sample which is not positive, i.e. a batch index different from 'batch_idx'
                    batch_index_choice = choice(list(range(0, batch_idx))+list(range(batch_idx+1, batch_size)))
                    # Take a latent from the list of positive samples of other batch indeces => negative sample for current batch_idx
                    if time_index_choice in self.k_future:
                        negative_sample = positive_samples[time_index_choice][batch_index_choice]
                    elif time_index_choice in self.k_past_present:
                        negative_sample = past_latents[time_index_choice][batch_index_choice]
                    else:
                        raise ValueError(f"Time index {time_index_choice} not in k_past_present or k_future")
                    
                    negatives = torch.cat((negatives, negative_sample.unsqueeze(0)), dim=0) # tensor of shape n_negatives x 512
                
                # negative_samples is a dict with elements of size (batchsize x n_negatives x 512)
                negative_samples[k][batch_idx] = negatives
                
            fk_positives = torch.sum(torch.exp(latent_predictions[k] * positive_samples[k]), dim=-1)
            fk_negatives = torch.sum(torch.exp(latent_predictions[k].unsqueeze(1) * negative_samples[k]), dim=-1)
            
            loss += 1/(cfg.n_negatives+1) * torch.log(fk_positives/(fk_positives+torch.sum(fk_negatives, dim=-1)))
            
            np_fk_positives = fk_positives.detach().cpu().numpy()
            np_fk_negatives = fk_negatives.detach().cpu().numpy()
            
            for batch_idx in range(batch_size): 
                correct_predictions[future_step] += all(np_fk_positives[batch_idx] > np_fk_negatives[batch_idx])
        
        loss /= cfg.n_predictions # average the number of predictions
        
        loss = -torch.sum(loss)/batch_size # average loss across batch
        
        correct_predictions = np.mean(correct_predictions)
             
        return loss, correct_predictions