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
First clone the repository to a folder on your pc. Then create a folder in the main directory: ``data/<your_dataset>`` where ``your_dataset``. How to get the right dataset:
- We used the LibriSpeech 'train-clean-100' dataset from http://www.openslr.org/12/.
- Unpack the file such you get the path ``data/LibriSpeech``


There are three ways of training (within the .sh files you can change the hyperparameters, please check the ``utils/config.py`` to see all options):

## Training the CLE (CPC with Locked Encoder)
1. First train the encoder using CPC:
```
sbatch Train_P-CPC_EXAMPLE.sh
```
2. Within Train_CLE_EXAMPLE.sh specify the path to the checkpoint. That's the best .pth file created in the previous step.

3. Train the classyfying fully connected layer:
```
sbatch Train_CLE_EXAMPLE.sh
```

## Training the CUE (CPC with Unlocked Encoder)
1. First train the encoder using CPC (if not done already):
```
sbatch Train_P-CPC_EXAMPLE.sh
```
2. Within Train_CUE_EXAMPLE.sh specify the path to the checkpoint. That's the best .pth file created in the previous step.

3. Train the classyfying fully connected layer:
```
sbatch Train_CUE_EXAMPLE.sh
```

## Training the FS (Fully supervised)
Simply run the following command:
```
sbatch Train_FS.sh
```