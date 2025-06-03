import torch
from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, encoder):
        super(BinaryClassifier, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(self.encoder(x)).squeeze(1)


class BinaryClassifierPL(LightningModule):
    def __init__(self, input_size, encoder):
        super().__init__()
        self.model = BinaryClassifier(input_size, encoder)

        
    def training_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = torch.tensor([1. if label == 'EPILEPTIC' else 0. for label in batch['labels']['eeg_label']]).to(self.device)
        preds = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = torch.tensor([1. if label == 'EPILEPTIC' else 0. for label in batch['labels']['eeg_label']]).to(self.device)
        preds = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        self.log('val_loss', loss)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x = batch['data'].float()
        y = torch.tensor([1. if label == 'EPILEPTIC' else 0. for label in batch['labels']['eeg_label']]).to(self.device)
        preds = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return self.model(x)