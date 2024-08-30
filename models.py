import torch.nn as nn
from einops import rearrange

class VerySmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        """
        Initializes the VerySmallCNN model.
        
        Args: 
            n_classes (int): The number of classes in our output layer. 
            Because of the dataset, our number of classes will be 10. 
        """
        super(VerySmallCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Dropout(p=0.4, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Dropout(p=0.4, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.dropout=nn.Dropout(p=0.4, inplace=True)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.layers(x)
        out = self.dropout(out)
        out = self.pool(out)
        out = rearrange(out, 'b c h w -> b (c h w)') # Flattening the output from 4D to 2D tensor.
        out = self.relu(self.fc2(self.fc1(out)))
        return out


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        """
        Initializes the larger SmallCNN model for the CIFAR-10 dataset.

        Args:
            n_classes (int): The number of classes in our output layer. 
            Because of the dataset, our number of classes will be 10. 
        """
        super(SmallCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Dropout(p=0.4, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Dropout(p=0.4, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.Dropout(p=0.4, inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.Dropout(p=0.4, inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.Dropout(p=0.4, inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.act = nn.Softmax(dim=1)
        self.dropout=nn.Dropout(p=0.4, inplace=True)
        self.fc1 = nn.Linear(128 * 16 * 16, 4096)
        self.fc2 = nn.Linear(4096, n_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out = self.dropout(out)
        out = self.pool(out)
        out = rearrange(out, 'b c h w -> b (c h w)') # Flattening the output from 4D to 2D tensor.
        out = self.relu(self.fc2(self.fc1(out)))
        return out