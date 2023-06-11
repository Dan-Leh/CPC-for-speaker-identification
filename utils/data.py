import os
import torch
import time
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from utils.config import CONFIG
cfg = CONFIG

class LibriDataset(Dataset):
    def __init__(self, split='train', patch_size = cfg.patch_size, n_predictions = cfg.n_predictions, n_past_latents = cfg.n_past_latents):
        
        assert split in ['train', 'test'], 'Only train and test splits are implemented.'
        
        data_dir = './data' # root directory for data
        self.patch_size = int(np.rint(patch_size*16000))    
        
        # getting filenames
        txt_file_split = os.path.join(data_dir, split+'_split.txt') # path to txt file containing train/test sample names
        with open(txt_file_split) as fp: # extracting sample names from txt file
            self.filename_list = fp.read().splitlines() 
            
        # creating list of filepaths from filenames
        Libri_dir = os.path.join(data_dir, 'LibriSpeech', 'train-clean-100/')
        self.filepath_list = [Libri_dir] * len(self.filename_list)
        for i, filename in enumerate(self.filename_list):
            split_name = filename.split('-')
            self.filepath_list[i] += os.path.join(split_name[0], split_name[1], filename+'.flac')
    
        if cfg.CPC or cfg.replicate_CPC_params: # CPC training or training fully supervised, but with same amount of data as CPC training:
            self.n_predictions = n_predictions
            self.n_past_latents = n_past_latents
            
            # delete small samples in dataset:
            txt_sample_lengths = os.path.join(data_dir, split+'_sample_lengths.txt') # path to txt file containing train/test sample lengths
            with open(txt_sample_lengths) as fp: # extracting sample lengths from txt file
                self.sample_lengths = fp.read().splitlines() 
            if split == 'train' or cfg.CPC:
                self.delete_items() # only use data that is long to make enough future predictions

        if not cfg.CPC: # fully supervised training, get labels:
            self.speakerID_list = sorted(np.array(os.listdir(Libri_dir), dtype=np.uint32)) #list of speaker IDs in ascending order
            if cfg.data_percentage < 100 and split == 'train':
                self.delete_items(cfg.data_percentage) # delete percentage of data selected randomly
            
        self.num_samples = len(self.filename_list)
        

    # Returns the length of the dataset
    def __len__(self):
        return self.num_samples

    def test_dataset(self):
        for path in self.filepath_list:
            if not os.path.isfile(path): print(f'not a file: {path}')

    def get_all_items(self, save_path):
        with open(save_path, 'w') as file:
            for idx in range(self.__len__()):
                audio_fp = self.filepath_list[idx]
                waveform, _ = torchaudio.load(audio_fp, normalize=True)
                # Write each element from the list to a new line
                file.write(str(len(waveform[0])) + '\n')
    
    def get_all_speakers(self):
        IDs = []
        for idx in range(self.__len__()):
            IDs.append(self.filename_list[idx].split('-')[0])
        return IDs
    
    def delete_items(self, data_percentage = None):
        if data_percentage != None: # delete percentage of data selected randomly
            mask = np.random.choice(len(self.filepath_list), int(len(self.filepath_list)*data_percentage/100), replace=False)
            self.filepath_list = self.filepath_list[mask]
            self.filename_list = self.filename_list[mask]
        else: # delete small samples in dataset (called when training in CPC mode)
            mask = np.array(self.sample_lengths, dtype=np.int32) > self.patch_size*(self.n_predictions+self.n_past_latents+1)
            self.filepath_list = np.array(self.filepath_list)[mask]
            self.filename_list = np.array(self.filename_list)[mask]
        print(f'Deleted {len(mask)-np.count_nonzero(mask)} samples from dataset. {len(self.filepath_list)} samples remaining.')
        return self

    def crop_audio(self, waveform):
        waveform = waveform.squeeze() # get rid of channel dimension
        
        if cfg.CPC:
            crop_length = self.patch_size*(self.n_past_latents+self.n_predictions+1)
        else:
            crop_length = self.patch_size
            
        viable_start = len(waveform) - self.patch_size*(crop_length)
        start_idx = np.random.randint(viable_start) if viable_start > 0 else 0
        end_idx = start_idx+crop_length
        return waveform[start_idx:end_idx].unsqueeze(0) # cropped_waveform, with channel dimension
    
    def split_patches(self, waveform):
        patches = []
        for i in range(self.n_past_latents + self.n_predictions + 1):
            patches.append(waveform[:, i*self.patch_size:(i+1)*self.patch_size])
        return patches
    
    def normalize(self, spectrogram):
        '''
        Input: 'spectrogram' Tensor of dimensions 1xHxW
        Output: Column-wise normalized 'spectrogram' Tensor of dimensions 1xHxW
        '''
        stds, means = torch.std_mean(spectrogram, dim = 1)
        
        return (spectrogram - means) / stds

    # Returns a dataset sample given an idx [0, len(dataset))
    def __getitem__(self, idx):
        audio_fp = self.filepath_list[idx]
        waveform, sample_rate = torchaudio.load(audio_fp, normalize = True)
        waveform = self.crop_audio(waveform) #take random crop of sample
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate, n_mfcc=64, melkwargs={"n_fft": 429}) # n_mfcc and n_fft chosen so that images are of resolution 64x96
        
        if cfg.CPC:
            patches = self.split_patches(waveform) # split into patches
            mfcc_spectrograms = map(lambda patch: mfcc_transform(patch), patches) # apply mfcc transform to each patch
            mfcc_spectrograms = map(lambda patch: self.normalize(patch), mfcc_spectrograms) #  normalize each patch
            mfcc_spectrograms = list(mfcc_spectrograms) # convert map object to list
            mfcc_spectrograms = torch.stack(mfcc_spectrograms) # stack patches along first dimension
            
            return mfcc_spectrograms #return list (tensor) of patches with n_past_latents + 1 + n_predictions patches
        
        else:
            mfcc_spectrogram = mfcc_transform(waveform) # apply mfcc transform to waveform
            speaker_ID = int(self.filename_list[idx].split('-')[0])
            label = self.speakerID_list.index(speaker_ID) # get label from speaker_ID
            
            return mfcc_spectrogram, label 
            
        
    