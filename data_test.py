from data import *
from PIL import Image
import numpy as np
import os

if os.path.split(os.getcwd())[-1] != '5aua0-2022-group-18':
    os.chdir('5aua0-2022-group-18')
print(os.getcwd())

trainset = LibriDataset('train')
testset = LibriDataset('test')

# lengths_train = trainset.get_all_items()
# lengths_test = testset.get_all_items()
# print(f'Training set:\naverage length: {np.mean(lengths_train)}\nmin length: {np.min(lengths_train)}\nmax length: {np.max(lengths_train)}')
# print(f'Test set:\naverage length: {np.mean(lengths_test)}\nmin length: {np.min(lengths_test)}\nmax length: {np.max(lengths_test)}')

label = [0]*10
for i in range(10):
    img, label[i] = trainset.__getitem__(i)
    img = img.numpy().squeeze()
    img = Image.fromarray(img).convert('L')
    img.save('spectrogram'+str(i)+'.png')

# print(label)

# print(len(trainset.labelID_list))
# print(len(testset.labelID_list))
