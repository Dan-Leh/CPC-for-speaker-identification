from dataclasses import dataclass


@dataclass
class Config:
    output_name = 'trial_for_debug'
    batch_size_train = 32
    batch_size_test = 32
    lr = 0.001
    max_lr = 0.01
    num_classes = 251
    gt_dir = "./data/LibriSpeech/"
    log_iterations = 100
    enable_cuda = True
    adam1 = 0.9
    adam2 = 0.999
    epochs = 20
    CPC = True
    if CPC:
        patch_size = 1.28
        n_predictions = 5
