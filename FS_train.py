import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from config import Config
from data import LibriDataset

from FS_model import Model


def train():
    # Configuration settings
    cfg = Config()

     # Load dataset
    dataset = LibriDataset(cfg.gt_dir, split='train')
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size_train, shuffle=True, num_workers=4)
     # Initialize network
    model = Model(cfg)
    model.train()
    
    if cfg.enable_cuda:
        model = model.cuda()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.lr_momentum, weight_decay=cfg.weight_decay)
        optim.Adam([cfg.adam1, cfg.adam2], lr=cfg.lr)