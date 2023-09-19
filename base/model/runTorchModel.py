from torchmodel import MLP
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
import csv
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    X_train = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/trainingData/X_nn_tr.pt")[:,4:]
    X_test = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/trainingData/X_nn_test.pt")[:,4:]
    y_train = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/trainingData/y_nn_tr.pt")
    y_test = torch.load("/mnt/home/grunew14/Documents/tardis/emulator/Data/trainingData/y_nn_test.pt")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)  

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = MLP(input_dim, output_dim).to(device)

    lr = 1e-4
    epochs = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    with open('../logs/training_log.csv', 'w', newline='') as csvfile:

        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])

        for epoch in range(epochs):

            model.train()
            train_loss = 0.0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = model(X_batch)
                    loss = loss_fn(predictions, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)

            log_writer.writerow([epoch + 1, train_loss, val_loss])

            print(f"Epoch: {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")


