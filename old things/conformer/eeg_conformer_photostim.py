from braindecode.models import EEGConformer
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def train_one_epoch(dataloader: DataLoader,model,loss_fn,optimizer,scheduler,epoch: int,device,print_batch_stats=True):  
    model.train()  # Set the model to training mode
    train_loss, correct = 0.0, 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats)
    for batch_idx, (X, y) in progress_bar:        
        X, y = X.to(device), y.to(device)   

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (y_pred.argmax(1)== y).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}/{n_epochs}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}"
            )

    # Update the learning rate
    scheduler.step()

    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct

def evaluate(dataloader: DataLoader, model,loss_fn,device,print_batch_stats=True):
        
        model.eval()  # Set the model to evaluation mode
        val_loss, correct = 0.0, 0.0
    
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats)
    
        with torch.no_grad():
            for batch_idx, (X, y) in progress_bar:        
                X, y = X.to(device), y.to(device)  
                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                    
                val_loss += loss.item()
                correct += (y_pred.argmax(1).float() == y).sum().item()
    
                if print_batch_stats:
                    progress_bar.set_description(f"Batch {batch_idx + 1}/{len(dataloader)}")
    
        correct /= len(dataloader.dataset)
        return val_loss / len(dataloader), correct

N_RUNS = 20
N_EPOCHS = 50

X=torch.load('/space/gzanardini/X.pt',weights_only=True)
y=torch.load('/space/gzanardini/Y.pt',weights_only=True).squeeze()
device = "cuda:3" if torch.cuda.is_available() else "cpu"

wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

for i in range(N_RUNS):
    model=EEGConformer(
        n_outputs=2,
        n_chans=19,
        n_filters_time=20,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length="auto",
        return_features=False,
        n_times=2500,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        add_log_softmax=False
    )

    model.to(device)
    n_epochs = N_EPOCHS
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_loader = DataLoader(Dataset(x_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(Dataset(x_test, y_test), batch_size=16, shuffle=False)

    wandb.init(project="EEGConformer",            
                    config={"n_epochs": n_epochs,
                    "batch_size": 64,
                    "lr": 0.001,
                    "optimizer": "Adam",
                    "scheduler": "StepLR",
                    "scheduler_step_size": 20,
                    "scheduler_gamma": 0.5,
                    "loss_fn": "CrossEntropyLoss",
                    "model": "EEGConformer",
                    "device": device})

    best_val_loss = float("inf")
    best_model = None

    train_counts = np.unique(y_train.cpu(), return_counts=True)[1]
    test_counts = np.unique(y_test.cpu(), return_counts=True)[1]

    print(train_counts)
    print(test_counts)

    wandb.log({"train_healthy": train_counts[0], "train_epileptic": train_counts[1], "test_healthy": test_counts[0], "test_epileptic": test_counts[1]})
    for epoch in range(n_epochs):

        train_loss, train_acc = train_one_epoch(train_loader,model,loss_fn,optimizer,scheduler,epoch,device)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}")

        val_loss, val_acc = evaluate(test_loader,model,loss_fn,device)
        print(f"Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}")

        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
    
    wandb.finish()
    torch.save(best_model, "EEGConformer_photostim_best.pth")