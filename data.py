import os
import torch
import time
import sklearn.preprocessing as preprocess
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
        
        self.speakerID_list = sorted(np.array(os.listdir(Libri_dir), dtype=np.uint32)) #list of speaker IDs in ascending order

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
            # waveform = self.crop_audio(waveform)
            lengths[idx] = len(waveform[0])
        return lengths
    
    def get_all_speakers(self):
        IDs = []
        for idx in range(self.__len__()):
            IDs.append(self.filename_list[idx].split('-')[0])
        return IDs
    
    def crop_audio(self, waveform):
        waveform = waveform.squeeze() # get rid of channel dimension
        crop_length = 22560 # crop length equivalent to length of smallest sample
        start_idx = np.random.randint(len(waveform)-crop_length)
        end_idx = start_idx+crop_length
        return waveform[start_idx:end_idx].unsqueeze(0) # cropped_waveform, with channel dimension
    
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
        speaker_ID = int(self.filename_list[idx].split('-')[0])

        waveform, sample_rate = torchaudio.load(audio_fp, normalize = True)
        waveform = self.crop_audio(waveform) #take random crop of sample

        mfcc_transform = torchaudio.transforms.MFCC(sample_rate, n_mfcc=64, melkwargs={"n_fft": 474}) # n_mfcc and n_fft chosen so that images are of resolution 64x96
        mfcc_spectrogram = mfcc_transform(waveform)

        mfcc_spectrogram = self.normalize(mfcc_spectrogram) # normalize column values

        # waveform, sample_rate = librosa.load(audio_fp, sr=None)
        # waveform = self.crop_audio(waveform) #take random crop of sample

        # mfcc_spectrogram = librosa.feature.mfcc(y = waveform, sr = sample_rate)
        # mfcc_spectrogram = sklearn.preprocessing.scale(mfcc_spectrogram) # normalize to zero mean, std 1

        label = self.speakerID_list.index(speaker_ID) # get label from speaker_ID

        return mfcc_spectrogram, label

