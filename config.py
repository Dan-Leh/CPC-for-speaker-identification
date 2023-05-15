from dataclasses import dataclass


@dataclass
class Config:
    batch_size_train = 32
    batch_size_test = 25
    lr = 0.005
    max_lr = 0.005
    num_classes = 251
    gt_dir = "./data/YOURDATASET/"
    num_iterations = 10000
    log_iterations = 100
    enable_cuda = True
    adam1 = 0.9
    adam2 = 0.999
    epochs = 20
