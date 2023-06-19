import argparse
from dataclasses import dataclass
import numpy as np
from datetime import datetime

current_time = datetime.now()

@dataclass
class Config:
    batch_size_train: int
    batch_size_test: int
    lr: float
    max_lr: float
    num_classes: int
    epochs: int
    data_percentage: int
    patch_size: float
    
    output_name: str
    gt_dir: str
    log_iterations: int
    
    CPC: bool
    replicate_CPC_params: bool
    n_predictions: int
    n_negatives: int
    n_past_latents: int
    
    freeze_encoder: bool
    load_checkpoint: str
    
    random_search : bool

def parse_args():
    parser = argparse.ArgumentParser(description='Config parameters')
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training and testing')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.0001, help='Maximum learning rate')
    parser.add_argument('--num_classes', type=int, default=251, help='Number of classes') # currently not used
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--data_percentage', type=int, default=100, help='Percentage of data to use when training')
    parser.add_argument('--patch_size', type=float, default=1.28, help='Patch size') # shouldn't need to be changed
    
    # Organizational/data parameters
    parser.add_argument('--output_name', type=str, default=current_time.strftime("%m/%d/%Y, %H:%M:%S").replace('/','-').replace(', ','_'), help='Output name')
    parser.add_argument('--gt_dir', type=str, default='./data/LibriSpeech/', help='Ground truth directory')
    parser.add_argument('--log_iterations', type=int, default=100, help='Number of iterations to log')
    
    # CPC hyperparameters
    parser.add_argument('--CPC', action='store_true', help='Enable CPC. If false, trains fully supervised')
    parser.add_argument('--replicate_CPC_params', action='store_true', help='Only used if CPC is false to make fully supervised model \
                        replicate CPC parameters by deleting small samples')
    parser.add_argument('--n_predictions', type=int, default=2, help='Number of latent predictions (& positive samples)')
    parser.add_argument('--n_negatives', type=int, default=10, help='Number of negative latent embeddings')
    parser.add_argument('--n_past_latents', type=int, default=3, help='Number of past latent embeddings to use for context vector')
    
    parser.add_argument('--freeze_encoder', action='store_true', help='whether to freeze encoder when training fully supervised')
    parser.add_argument('--load_checkpoint', type=str, default="", help='Path to checkpoint to load')

    parser.add_argument('--random_search', action='store_true', help='set to true if running random search')
    
    args = parser.parse_args()
    return args

args = parse_args()
assert (args.CPC and args.replicate_CPC_params) == False, "replicate CPC parameters only relevant for supervised training"

CONFIG = Config(
num_classes=args.num_classes,
gt_dir=args.gt_dir,
log_iterations=args.log_iterations,
data_percentage=args.data_percentage, 
CPC=args.CPC,
replicate_CPC_params=args.replicate_CPC_params, 
freeze_encoder=args.freeze_encoder,
load_checkpoint=args.load_checkpoint,
random_search=args.random_search,
output_name=args.output_name,
batch_size_train=args.batch_size,
batch_size_test=args.batch_size,
lr=args.lr,
max_lr=args.max_lr,
epochs=args.epochs,
patch_size=args.patch_size,
n_predictions=args.n_predictions,
n_negatives=args.n_negatives,
n_past_latents=args.n_past_latents)

if args.random_search:
    CONFIG.batch_size_train=np.random.randint(10, 64)
    CONFIG.batch_size_test=CONFIG.batch_size_train
    CONFIG.lr=np.random.uniform(0.000001, 0.01)
    CONFIG.max_lr=np.random.uniform(CONFIG.lr, 1)
    CONFIG.n_predictions=np.random.randint(1, 10)
    CONFIG.n_negatives=np.random.randint(2, CONFIG.batch_size_train)
    CONFIG.n_past_latents=np.random.randint(1, 12-CONFIG.n_predictions)