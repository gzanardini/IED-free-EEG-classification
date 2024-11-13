from utils.TUEP import TUHEpilepsy
from braindecode.datautil import load_concat_dataset
from braindecode.models import EEGConformer
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import wandb

wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

datapath='/space/gzanardini/tuh_eeg/preprocessed/windows10s/'
data=load_concat_dataset(path=datapath, preload=True)

def create_data_loaders(data):
    subjects = np.unique(data.description["subject"])
    subj_train, subj_test = train_test_split(subjects, test_size=0.2, shuffle=True)
    subj_valid, subj_test = train_test_split(subj_test, test_size=0.5, shuffle=True)

    idx_train = np.where(np.isin(data.description["subject"], subj_train))[0]
    idx_valid = np.where(np.isin(data.description["subject"], subj_valid))[0]
    idx_test = np.where(np.isin(data.description["subject"], subj_test))[0]

    split_ids = {
    "train": idx_train,
    "valid": idx_valid,
    "test": idx_test
}

    print(f"Number of training samples: {len(idx_train)}")
    print(f"Number of validation samples: {len(idx_valid)}")
    print(f"Number of test samples: {len(idx_test)}")

    splitted = {}
    for name, values in split_ids.items():
        splitted[name] = data.split(split_ids=values)

    train_loader = DataLoader(splitted["train"]['0'], batch_size=64, shuffle=True, num_workers=3)
    valid_loader = DataLoader(splitted["valid"]['0'], batch_size=64, shuffle=False, num_workers=3)
    test_loader = DataLoader(splitted["test"]['0'], batch_size=64, shuffle=False, num_workers=3)
    return train_loader,valid_loader,test_loader

train_loader, valid_loader, test_loader = create_data_loaders(data)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(dataloader: DataLoader,model,loss_fn,optimizer,scheduler,epoch: int,device,print_batch_stats=True):
    
    model.train()  # Set the model to training mode
    train_loss, correct = 0.0, 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats)

    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()  # update the model weights
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

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

def evaluate(dataloader: DataLoader,model,loss_fn,device,print_batch_stats=True):
        
        model.eval()  # Set the model to evaluation mode
        val_loss, correct = 0.0, 0.0
    
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats)
    
        with torch.no_grad():
            for batch_idx, (X, y, _) in progress_bar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
    
                val_loss += loss.item()
                correct += (pred.argmax(1) == y).sum().item()
    
                if print_batch_stats:
                    progress_bar.set_description(f"Batch {batch_idx + 1}/{len(dataloader)}")
    
        correct /= len(dataloader.dataset)
        return val_loss / len(dataloader), correct

for i in range(10):

    model=EEGConformer(
        n_outputs=2,
        n_chans=20,
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
    n_epochs = 50
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_loader, valid_loader, test_loader = create_data_loaders(data)

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

    for epoch in range(n_epochs):
        
        train_loss, train_acc = train_one_epoch(train_loader,model,loss_fn,optimizer,scheduler,epoch,device)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}")
        #every 10 epochs evaluate the model
        val_loss, val_acc = evaluate(valid_loader,model,loss_fn,device)
        print(f"Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}")

        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    test_loss, test_acc = evaluate(test_loader,model,loss_fn,device)

    print(f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}")
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    wandb.finish()
    torch.save(best_model, "EEGConformer_best.pth")