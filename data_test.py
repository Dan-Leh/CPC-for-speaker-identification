from data import *
from PIL import Image
import numpy as np
import os

if os.path.split(os.getcwd())[-1] != '5aua0-2022-group-18':
    os.chdir('5aua0-2022-group-18')
print(os.getcwd())

trainset = LibriDataset('train')
testset = LibriDataset('test')

label = [0]*10
for i in range(10):
    img, label[i] = trainset.__getitem__(i)
    img = img.numpy().squeeze()
    img = Image.fromarray(img).convert('L')
    img.save('spectrogram'+str(i)+'.png')




print(label)

print(len(trainset.labelID_list))
print(len(testset.labelID_list))
