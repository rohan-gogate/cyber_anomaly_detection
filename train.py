import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.model import LSTMVAE
from src.losses import vae_loss
import joblib
import os

data = np.loadtxt('artifacts/preprocessed_training_data.csv', delimiter=',', skiprows=1)

def create_windows(data, window_size):
    return np.stack([data[i:i+window_size] for i in range(len(data) - window_size + 1)])
window_size = 30
X = create_windows(data, window_size)

tensor_X = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(tensor_X)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = X.shape[2]
hidden_dim = 64
latent_dim = 16

model = LSTMVAE(input_dim, hidden_dim, latent_dim, window_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for batch in loader:
        x = batch[0].to(device)

        x_recon, mu, logvar = model(x)
        loss, recon_loss, kl_div = vae_loss(x_recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()

    avg_loss = total_loss / len(loader)
    avg_recon = total_recon / len(loader)
    avg_kl = total_kl / len(loader)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

os.makedirs("artifacts", exist_ok=True)
torch.save(model.state_dict(), "artifacts/lstm_vae.pth")

