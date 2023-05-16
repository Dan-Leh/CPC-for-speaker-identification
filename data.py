import os
import torch
import time
import numpy as np
import torchaudio
from torch.utils.data import Dataset



class LibriDataset(Dataset):
    def __init__(self, split='train'):
        assert split in ['train', 'test'], 'Only train and test splits are implemented.'
        data_dir = './data' # root directory for data

        self.split = split

        txt_file_split = os.path.join(data_dir, split+'_split.txt') # path to txt file containing train/test sample names
        with open(txt_file_split) as fp: # extracting sample names from txt file
            self.filename_list = fp.read().splitlines() 

        self.num_samples = len(self.filename_list)
        
        Libri_dir = os.path.join(data_dir, 'LibriSpeech', 'train-clean-100/')
        # creating list of filepaths from filenames
        self.filepath_list = [Libri_dir] * self.num_samples
        for i, filename in enumerate(self.filename_list):
            split_name = filename.split('-')
            self.filepath_list[i] += os.path.join(split_name[0], split_name[1], filename+'.flac')
        
        self.labelID_list = os.listdir(Libri_dir)

    # Returns the length of the dataset
    def __len__(self):
        return self.num_samples

    def test_dataset(self):
        for path in self.filepath_list:
            if not os.path.isfile(path): print(f'not a file: {path}')

    def get_all_items(self):
        lengths = np.zeros(self.__len__())
        for idx in range(self.__len__()):
            audio_fp = self.filepath_list[idx]
            waveform, sample_rate = torchaudio.load(audio_fp, normalize=True)
            waveform = self.crop_audio(waveform)
            lengths[idx] = len(waveform[0])
        return lengths
    
    def crop_audio(self, waveform):
        #TODO: change cropping to make it random
        cropped_waveform = waveform[0][0:22580]
        return cropped_waveform.unsqueeze(0) # to make dimension 1 x len

    # Returns a dataset sample given an idx [0, len(dataset))
    def __getitem__(self, idx):
        audio_fp = self.filepath_list[idx]
        speaker_ID = self.filename_list[idx].split('-')[0]

        waveform, sample_rate = torchaudio.load(audio_fp, normalize=True)
        waveform = self.crop_audio(waveform)

        transform = torchaudio.transforms.MelSpectrogram(sample_rate)
        mel_spectrogram = transform(waveform)

        return mel_spectrogram, speaker_ID
