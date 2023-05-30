from dataclasses import dataclass


@dataclass
class Config:
    output_name = 'CPC_much_lower_lr'
    batch_size_train = 64
    batch_size_test = 64
    lr = 0.00001
    max_lr = 0.0001
    num_classes = 251
    gt_dir = "./data/LibriSpeech/"
    log_iterations = 100
    enable_cuda = True
    adam1 = 0.9
    adam2 = 0.999
    epochs = 50
    CPC = True
    if CPC:
        patch_size = 1.28
        n_predictions = 5
        n_negatives = 10
