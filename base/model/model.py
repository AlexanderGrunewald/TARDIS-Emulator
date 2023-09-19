import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
import joblib
torch.cuda.is_available()

def inverse_transform(X, y, scalers, index):
    X[:,3] = X[:,3] *-1
    X_unscaled = np.exp(scalers[0].inverse_transform(X))
    y_unscaled = np.exp(scalers[1].inverse_transform(y))
    y_unscaled[index,:40] = np.cumsum(y_unscaled[index,:40])
    return X_unscaled, y_unscaled

def plot_predictions(predictions, actuals, X_test, scalers, save_path=None, index=0):
    plt.figure()
    X_unscaled, y_unscaled = inverse_transform(X_test, actuals, scalers, index=index)
    y_hat_unscaled = np.exp(scalers[1].inverse_transform(predictions))
    y_hat_unscaled[index,:40] = np.cumsum(y_hat_unscaled[index,:40])
    
    y_hat_unscaled = y_hat_unscaled[index,:]
    y_unscaled = y_unscaled[index,:]
    
    plt.plot(y_hat_unscaled[:40], y_hat_unscaled[40:], label="Predictions")
    plt.plot(y_unscaled[:40], y_unscaled[40:], label="Actuals")
    plt.xlabel("Velovity")
    plt.ylabel("Dilution Factor (w)")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    return plt.gcf() 

class LitFeedForward(pl.LightningModule):
    def __init__(
            self, input_dim, 
            hidden_dim, 
            output_dim, 
            n_layers=5, 
            activation=F.mish, 
            name="FeedForward", 
            lr=1e-3, 
            batchsize=32
            ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(n_layers - 2):  
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        self.scalers = [joblib.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/statistics/scalar_X_test.pkl"),
                        joblib.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/statistics/scalar_y_test.pkl")]
        
        self.val_predictions = []
        self.val_actuals = []
        self.val_x = []
        self.index = 29
        # hyperparameters
        self.lr = lr
        self.activation = activation
        self.name = name
        self.batchsize = batchsize

        self.save_hyperparameters()

    def forward(self, x):
        for layer in self.layers[:-1]:  
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.sqrt(F.mse_loss(y_hat, y))
        self.log('val_loss', loss)

        # Return the values for further processing in `validation_epoch_end`
        return {"predictions": y_hat.detach(), "actuals": y.detach(), "x": x.detach()}  

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_predictions.append(outputs["predictions"])
        self.val_actuals.append(outputs["actuals"])
        self.val_x.append(outputs["x"])


    def on_validation_epoch_end(self):

        if self.current_epoch % 10 == 0:
            predictions = self.val_predictions[0].cpu().numpy()
            actuals = self.val_actuals[0].cpu().numpy()
            x = self.val_x[0].cpu().numpy()


            fig = plot_predictions(predictions, actuals, x, self.scalers)
            fig2 = plot_predictions(predictions, actuals, x, self.scalers, index=self.index)

            self.logger.experiment.add_figure('predictions vs. actuals at index 0', fig, global_step=self.global_step)
            plt.close(fig)
            self.logger.experiment.add_figure(f"predictions vs. actuals at index random", fig2, global_step=self.global_step)
            plt.close(fig2)

        self.val_predictions = []
        self.val_actuals = []
        self.val_x = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
   
