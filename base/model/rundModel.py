import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from model import LitFeedForward

torch.cuda.is_available()

if __name__ == "__main__":
    # Load data
    X_train = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/X_nn_tr.pt")
    X_test = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/X_nn_test.pt")
    y_train = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/y_nn_tr.pt")
    y_test = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/train_data/y_nn_test.pt")

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)  

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Define model parameters
    input_dim = X_train.shape[1] 
    output_dim = 80

    # Create and train model
    model = LitFeedForward(input_dim, 100, output_dim, n_layers=2)
    trainer = pl.Trainer(max_epochs=100) 
    trainer.fit(model, train_loader, val_loader)

    # Save trained model
    torch.save(model.state_dict(), "model/model.pt")

    # Use trained model to make predictions on test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    # Save predictions along with actual labels
    torch.save({ 
        "predictions": predictions,
        "actuals": y_test,
        "X_test": X_test,
    }, "model/model_logs/predictions.pt")
