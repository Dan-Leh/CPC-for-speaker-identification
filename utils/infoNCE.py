import torch
import torch.nn as nn
import random

from utils.config import Config
cfg = Config()

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = ["k"+str(i+1) for i in range(cfg.n_predictions)]
        self.keys.append('k')
        
    def softmax(self):
        pass

    def forward(self, latent_predictions, positive_samples):
        # latent_predictions: dict with keys 'k' to 'k+n_predictions', each item of size batch_sizex512
        # positive_samples: dict with keys 'k+1' to 'k+n_predictions', each item of size batch_sizex512
        
        negative_samples = {}
        for future_step in range(cfg.n_predictions):
            negative_samples["k+"+str(future_step+1)][0] = torch.Tensor([])
            
            for index_neg_sample in cfg.n_negatives:
                time_index = random.choice(self.keys) #choose random time index 'k' to 'k+n_predictions'
                 # choose a sample which is not positive
                
                negative_samples["k+"+str(future_step+1)][0].append()
                
        
        
        return 