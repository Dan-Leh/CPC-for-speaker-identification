from utils.data_CPC import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

if os.path.split(os.getcwd())[-1] != '5aua0-2022-group-18':
    os.chdir('5aua0-2022-group-18')
print(os.getcwd())


def plot_histogram(data):
    plt.hist(data, bins='auto')
    plt.xlabel('Seconds of Audio')
    plt.ylabel('Frequency')
    plt.title('Histogram of audio sample lengths')
    plt.show()
    plt.savefig('/home/dlehman/5aua0-2022-group-18/histogram_training_sample_durations.png')
    print(f"Histogram saved at: '/home/dlehman/5aua0-2022-group-18/histogram_training_sample_durations.png'")

trainset = LibriDataset('train')
trainset = trainset.__delete_small_items__()
testset = LibriDataset('test')
testset = testset.__delete_small_items__()



# lengths_train = trainset.get_all_items()
# lengths_test = testset.get_all_items()

# lengths_train_seconds = lengths_train/16000
# plot_histogram(lengths_train_seconds)

# print(f'Training set:\naverage length: {np.mean(lengths_train_seconds)}\nmin length: {np.min(lengths_train_seconds)}\nmax length: {np.max(lengths_train_seconds)}')
# print(f'Test set:\naverage length: {np.mean(lengths_test)}\nmin length: {np.min(lengths_test)}\nmax length: {np.max(lengths_test)}')






label = [0]*10
for i in range(10):
    in_patch, future_patches = trainset.__getitem__(i)
    in_patch = in_patch.numpy().squeeze()*255
    in_patch = Image.fromarray(in_patch).convert('L')
    in_patch.save('spectrogram'+str(i)+'.png')

# print(label)

# print(len(trainset.labelID_list))
# print(len(testset.labelID_list))
