import torch.nn.functional as F
from torch.nn import init

# Code bases on https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5 # 


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []

        # First Convolution Block 
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.max1 = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        layers += [self.conv1, self.max1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max2 = nn.MaxPool2d(2,2)
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        layers += [self.conv2, self.max2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max3 = nn.MaxPool2d(2,2)
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        layers += [self.conv3, self.max3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
        self.max4 = nn.MaxPool2d(2,2)
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        layers += [self.conv4, self.max4, self.bn4]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
        self.max5 = nn.MaxPool2d(2,2)
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv5.bias.data.zero_()
        layers += [self.conv5, self.max5, self.bn5]

        # Sixth Convolution Block
        self.conv6 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
        self.max6 = nn.MaxPool2d(2,2)
        self.bn6 = nn.BatchNorm2d(512)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv6.bias.data.zero_()
        layers += [self.conv6, self.max6, self.bn6]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=512, out_features=251)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


myModel = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device