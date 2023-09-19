import argparse
import os
import torch
import lightning.pytorch as pl
from modelv2 import LitFeedForward
import torch.nn.functional as F

if __name__ == "__main__":

    print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args_parser = argparse.ArgumentParser()
    
    args_parser.add_argument("--n_layers", type=int, default=5)
    args_parser.add_argument("--hidden_dim", type=int, default=100)
    args_parser.add_argument("--max_epochs", type=int, default=100)
    args_parser.add_argument("--batch_size", type=int, default=32)
    args_parser.add_argument("--lr", type=float, default=1e-3)
    args_parser.add_argument("--activation", type=object, default=F.mish)

    args = args_parser.parse_args()

    print("Arguments:", args)

    # Initialize the model
    model = LitFeedForward(hidden_dim=args.hidden_dim, n_layers=args.n_layers, lr=args.lr, batchsize=args.batch_size, activation=args.activation)

    # Load and prepare data
    model.prepare_data()

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=args.max_epochs)

    print(f"Training model with {args.n_layers} layers and hidden_dim of {args.hidden_dim} for {args.max_epochs} epochs")

    # Fit the model
    trainer.fit(model)

    print("Training complete.")

    # Save the trained model
    if not os.path.exists("model"):
        os.makedirs("model")

    i = 0
    while os.path.exists(f"model/model_{i}.pt"):
        i += 1
    torch.save(model.state_dict(), f"model/model_{i}.pt")

    print(f"Model saved as model/model_{i}.pt")
