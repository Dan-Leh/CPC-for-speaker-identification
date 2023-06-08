import os
import torch
import time
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from utils.config import Config
cfg = Config()

class LibriDataset(Dataset):
    def __init__(self, split='train', patch_size = cfg.patch_size, n_predictions = cfg.n_predictions, n_ARmemory = cfg.n_ARmemory):
        assert split in ['train', 'test'], 'Only train and test splits are implemented.'
        self.patch_size = int(np.rint(patch_size*16000))
        self.n_predictions = n_predictions
        self.n_ARmemory = n_ARmemory
        data_dir = './data' # root directory for data

        txt_file_split = os.path.join(data_dir, split+'_split.txt') # path to txt file containing train/test sample names
        with open(txt_file_split) as fp: # extracting sample names from txt file
            self.filename_list = fp.read().splitlines() 
            
        txt_sample_lengths = os.path.join(data_dir, split+'_sample_lengths.txt') # path to txt file containing train/test sample lengths
        with open(txt_sample_lengths) as fp: # extracting sample lengths from txt file
            self.sample_lengths = fp.read().splitlines() 
        
        Libri_dir = os.path.join(data_dir, 'LibriSpeech', 'train-clean-100/')
        # creating list of filepaths from filenames
        self.filepath_list = [Libri_dir] * len(self.filename_list)
        for i, filename in enumerate(self.filename_list):
            split_name = filename.split('-')
            self.filepath_list[i] += os.path.join(split_name[0], split_name[1], filename+'.flac')
            
        self.delete_items() # only use data that is long to make enough future predictions
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
    
    def delete_items(self, data_percentage = cfg.data_percentage):
        mask = np.array(self.sample_lengths, dtype=np.int32) > self.patch_size*(self.n_predictions+self.n_ARmemory+1)
        self.filepath_list = np.array(self.filepath_list)[mask]
        self.filename_list = np.array(self.filename_list)[mask]
        # delete percentage of data selected randomly
        if data_percentage < 100:
            mask = np.random.choice(len(self.filepath_list), int(len(self.filepath_list)*data_percentage/100), replace=False)
            self.filepath_list = self.filepath_list[mask]
            self.filename_list = self.filename_list[mask]
        return self

    def crop_audio(self, waveform):
        waveform = waveform.squeeze() # get rid of channel dimension
        crop_length = self.patch_size*(self.n_ARmemory+self.n_predictions+1)
        viable_start = len(waveform) - self.patch_size*(crop_length)
        start_idx = np.random.randint(viable_start) if viable_start > 0 else 0
        end_idx = start_idx+self.patch_size*(crop_length)
        return waveform[start_idx:end_idx].unsqueeze(0) # cropped_waveform, with channel dimension
    
    def split_patches(self, waveform):
        patches = []
        for i in range(self.n_ARmemory + self.n_predictions + 1):
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
        patches = self.split_patches(waveform) # split into patches
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate, n_mfcc=64, melkwargs={"n_fft": 429}) # n_mfcc and n_fft chosen so that images are of resolution 64x96
        mfcc_spectrograms = map(lambda patch: mfcc_transform(patch), patches) # apply mfcc transform to each patch
        mfcc_spectrograms = map(lambda patch: self.normalize(patch), mfcc_spectrograms) #  normalize each patch
        mfcc_spectrograms = list(mfcc_spectrograms) # convert map object to list

        return mfcc_spectrograms #return list of patches with n_ARmemory + 1 patches as input and n_predictions patches as target
    