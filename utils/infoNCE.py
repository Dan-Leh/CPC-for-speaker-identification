import torch
import torch.nn as nn
import numpy as np
from random import choice

from utils.config import Config
cfg = Config()

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys_no_k = ["k+"+str(i+1) for i in range(cfg.n_predictions)]
        self.keys = self.keys_no_k + ['k']
        

    def forward(self, latent_predictions, positive_samples):
        '''
        Inputs:
            'latent_predictions': dict with keys 'k' to 'k+n_predictions', each item of size batch_sizex512
            'positive_samples': dict with keys 'k+1' to 'k+n_predictions', each item of size batch_sizex512
        Output:
            'loss': scalar loss
        '''
        loss = 0
        negative_samples = {}
        correct_predictions = np.zeros((cfg.n_predictions))
        
        for future_step, k in enumerate(self.keys_no_k):

            negative_samples[k] = torch.zeros((cfg.batch_size_train, cfg.n_negatives, 512)) # initialize empty tensor
            for batch_idx in range(cfg.batch_size_train): 
                negatives = torch.Tensor([])
                
                for _ in range(cfg.n_negatives):
                    # choose random time index 'k' to 'k+n_predictions'
                    time_index_choice = choice(self.keys) 
                    # choose a sample which is not positive, i.e. a batch index different from 'batch_idx'
                    batch_index_choice = choice(list(range(0, batch_idx))+list(range(batch_idx+1, cfg.batch_size_train)))
                    # Take a latent from the list of positive samples of other batch indeces => negative sample for current batch_idx
                    if time_index_choice == 'k':
                        negative_sample = latent_predictions[time_index_choice][batch_index_choice]
                    else:
                        negative_sample = positive_samples[time_index_choice][batch_index_choice]
                    
                    negatives = torch.cat((negatives, negative_sample.unsqueeze(0)), dim=0) # tensor of shape n_negatives x 512
                
                # negative_samples is a dict with elements of size (batchsize x n_negatives x 512)
                negative_samples[k][batch_idx] = negatives
                
            fk_positives = torch.sum(torch.exp(latent_predictions[k] * positive_samples[k]), dim=-1)
            fk_negatives = torch.sum(torch.exp(latent_predictions[k].unsqueeze(1) * negative_samples[k]), dim=-1)
            
            loss += 1/(cfg.n_negatives+1) * torch.log(fk_positives/(fk_positives+torch.sum(fk_negatives, dim=-1)))
            
            np_fk_positives = fk_positives.detach().numpy()
            np_fk_negatives = fk_negatives.detach().numpy()
            
            for batch_idx in range(cfg.batch_size_train): 
                correct_predictions[future_step] += all(np_fk_positives[batch_idx] > np_fk_negatives[batch_idx])
        
        loss /= cfg.n_predictions # average the number of predictions
        
        loss = torch.sum(loss)/cfg.batch_size_train # average loss across batch
                
        return loss, correct_predictions