import argparse
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from model import LitFeedForward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print("Using device:", device)

    args_parser = argparse.ArgumentParser()
    
    args_parser.add_argument("--n_layers", type=int, default=1)
    args_parser.add_argument("--hidden_dim", type=int, default=100)
    args_parser.add_argument("--max_epochs", type=int, default=100)
    args_parser.add_argument("--batch_size", type=int, default=32)
    args_parser.add_argument("--lr", type=float, default=1e-3)

    args = args_parser.parse_args()

    print(args)


    print(args)
    print("Loading data...")

    X_train = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/X_nn_tr.pt")
    X_test = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/X_nn_test.pt")
    y_train = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/y_nn_tr.pt")
    y_test = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/y_nn_test.pt")

    print("Data loaded.")

    print("data shape:", X_train.shape)
    print("label shape:", y_train.shape)

    print("making dataloaders...")

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)  

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("dataloaders made.")

    # Define model parameters
    input_dim = X_train.shape[1] 
    output_dim = y_train.shape[1]

    print(f"creating model with {args.n_layers} layers \n hidden_dim: {args.hidden_dim} \n input_dim: {input_dim} \n output_dim: {output_dim}")

    model = LitFeedForward(input_dim, args.hidden_dim, output_dim, n_layers=args.n_layers)

    print(f"model created... now training for {args.max_epochs} epochs")
    trainer = pl.Trainer(max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, val_loader)

    print("training complete.")
    print("saving model...")
    # Save trained model
    # if there is no model directory, create it
    if not os.path.exists("model"):
        os.makedirs("model")
        # if model.pt exists, make a new model.pt named model_{i}.pt where i is the smallest integer such that model_{i}.pt does not exist
        if os.path.exists("model/model.pt"):
            i = 1
            while os.path.exists(f"model/model_{i}.pt"):
                i += 1
            torch.save(model.state_dict(), f"model/model_{i}.pt")
        else:
            torch.save(model.state_dict(), "model/model_0.pt")
    else:
        if os.path.exists("model/model.pt"):
            i = 1
            while os.path.exists(f"model/model_{i}.pt"):
                i += 1
            torch.save(model.state_dict(), f"model/model_{i}.pt")
        else:
            torch.save(model.state_dict(), "model/model_0.pt")
