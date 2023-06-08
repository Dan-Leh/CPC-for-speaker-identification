import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import init

from utils.config import Config

cfg = Config()

# Architecture based on https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5 # 

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=512, out_features=251)
        self.convencoder = ConvEncoder()

    def forward(self, x):
        # Run the convolutional blocks
        x = self.convencoder(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
    

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []

        # First Convolution Block 
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu = nn.ReLU()
        self.max = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        layers += [self.conv1, self.relu, self.max, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        layers += [self.conv2, self.relu, self.max, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        layers += [self.conv3, self.relu, self.max, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        layers += [self.conv4, self.relu, self.max, self.bn4]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        layers += [self.conv5, self.relu, self.max, self.bn5]

        # Sixth Convolution Block
        self.conv6 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(512)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        layers += [self.conv6, self.relu, self.max, self.bn6]

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class LatentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 512
        self.out_features = 512
        self.FC = nn.Linear(in_features=self.in_features, 
                            out_features=self.out_features, 
                            bias = False)

    def forward(self, x):
        x = self.FC(x)
        return x
    
class CPCAR(nn.Module):
    def __init__(self, device, GRU_layers=cfg.GRU_layers, drop_prop = cfg.GRU_dropout):
        super().__init__()
        self.n_layers = cfg.GRU_layers
        self.encoder = ConvEncoder()
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True, dropout=drop_prop)
        self.fc = nn.Linear(in_features=256, out_features=512)
        self.relu = nn.ReLU()
        self.device = device
        
    def forward(self, x_prevs, x):
        h_prev = torch.zeros(1, 256).to(self.device)
        for x in x_prevs:
            x.to(self.device)
            x = self.encoder(x)
            x = x.view(x.shape[0], -1) # flatten 
            _, h_prev = self.gru(x, h_prev)
        x, h = self.gru(x, h_prev)
        x = self.fc(x)
        #x = self.fc(self.relu(x))
        return x, h
    
class CPC_model(nn.Module):
    def __init__(self, device, n_predictions=cfg.n_predictions):
        super().__init__()
        
        self.n_predictions = n_predictions
        self.latentpredictors = [LatentPredictor().to(device) for _ in range(n_predictions)]
        self.encoder = ConvEncoder()
        self.ARmodel = CPCAR(device).to(device)
        
    def forward(self, x, x_prevs):
        
        if x_prevs != None: # if the input consists of past and current samples, generate context vector and future predictions
            x , _ = self.ARmodel(x_prevs, x)
            # make multiple predictions and output them in dict
            output = {}
            output["k"] = x
            for i in range(self.n_predictions):
                output["k+"+str(i+1)] = self.latentpredictors[i](x)
                if output["k+"+str(i+1)].isnan().any():
                    print('nan detected')

        else: # if the input consists of only future samples, just generate their latent encodings
            x = self.encoder(x)
            output = x.view(x.shape[0], -1) # flatten 
        
        return output

        