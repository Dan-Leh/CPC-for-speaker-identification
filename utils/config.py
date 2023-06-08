from dataclasses import dataclass


@dataclass
class Config:
    output_name = 'Test_Twan'
    batch_size_train = 4
    batch_size_test = 4
    lr = 0.00001
    max_lr = 0.0001
    num_classes = 251
    gt_dir = "./data/LibriSpeech/"
    log_iterations = 100
    enable_cuda = True
    adam1 = 0.9
    adam2 = 0.999
    epochs = 2
    data_percentage = 50
    CPC = True
    if CPC:
        patch_size = 1.28
        n_predictions = 2
        n_negatives = 10
        GRU_layers = 3
        GRU_dropout = 0.5
        n_ARmemory = 3
