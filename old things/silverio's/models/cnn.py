import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as pl
from pytorch_lightning import LightningModule

class CNN_baseline(nn.Module):
    def __init__(self, input_channels, fc_input_size, output_channels=3, kernel_size=3, stride=1, padding='same'):
        super(CNN_baseline, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(output_channels, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, )
        self.fc1 = nn.Linear(fc_input_size, fc_input_size)
        self.fc2 = nn.Linear(fc_input_size, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.squeeze(1)
    
    
class CNN_baseline_PL(LightningModule):
    def __init__(self, input_channels, fc_input_size, output_channels=3, kernel_size=3, stride=1, padding='same'):
        super(CNN_baseline_PL, self).__init__()
        self.model = CNN_baseline(input_channels, fc_input_size)
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        x = batch['spectrograms'].float()
        y = batch['labels'].float()
        out = self.model(x)
        loss = nn.functional.binary_cross_entropy(out, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['spectrograms'].float()
        y = batch['labels'].float()
        out = self.model(x)
        loss = nn.functional.binary_cross_entropy(out, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['spectrograms'].float()
        y = batch['labels'].float()
        out = self.model(x)
        loss = nn.functional.binary_cross_entropy(out, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        
    def forward(self, x):
        return self.model(x)