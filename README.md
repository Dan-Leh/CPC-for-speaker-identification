Student 1: Dan Lehman - d.lehman@student.tue.nl - 1495739

Student 2: Twan Leloup - t.r.leloup@student.tue.nl - 1009272

![image](https://user-images.githubusercontent.com/112875332/232781631-bb8b5604-8e4f-4d55-bf67-d452007e17ea.png)

Hello, and welcome to Dan and Twan's project for the course 5AUA0 Advanced sensing using deep learning. Here you will find all the information you need for running our code. 

### Requirements:
You need these packages:
- python 3.10.9
- torch 1.12.1
- numpy 1.23.5
- torchaudio 0.12.1
- matplotlib 3.7.1
- pandas 1.5.3

# How to train
First clone the repository to a folder on your pc. Then download the correct dataset and place it in the 'data' directory, using the following steps:
- Create a directory called "LibriSpeech"
- In that directory, download the LibriSpeech 'train-clean-100' dataset from http://www.openslr.org/12/.
- Unzip the download such that the audio data can be found under the following path: ``data/LibriSpeech/train-clean-100``


There are three models to be trained to replicate the results reported in our paper. The instructions to train them are individually described below. Before training, please create a directory called ``trained_models`` in the root directory. The training and validation losses and model checkpoints for each model you train will be saved in this folder.

## Training the CLE (Classifier with CPC-pretrained Locked Encoder)
1. First train the encoder using CPC:
```
sbatch Train_P-CPC_EXAMPLE.sh
```
2. Within Train_CLE_EXAMPLE.sh, specify the path to the default saved checkpoint. That's the best .pth file created in the previous step.

3. Train the classyfying fully connected layer:
```
sbatch Train_CLE_EXAMPLE.sh
```

## Training the CUE (Classifier with CPC-pretrained Unlocked Encoder)
1. First train the encoder using CPC (if not done already):
```
sbatch Train_P-CPC_EXAMPLE.sh
```
2. Within Train_CUE_EXAMPLE.sh, specify the path to the default saved checkpoint. That's the best .pth file created in the previous step.

3. Train the classyfying fully connected layer:
```
sbatch Train_CUE_EXAMPLE.sh
```

## Training the FS (Fully supervised model)
Simply run the following command:
```
sbatch Train_FS.sh
```

Note that within the bash files, a number of flags can be set to change some hyperparameters. Please check the ``utils/config.py`` to see all the hyperparameter options as well as their default values.
