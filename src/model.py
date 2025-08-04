import torch
import torch.nn as nn

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, window_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = window_size

        self.encoder_LSTM = nn.LSTM(input_dim, hidden_dim, batch_first= True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_LSTM = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def encode(self, x):
        _, (h_n, _) = self.encoder_LSTM(x)
        h_n = h_n.squeeze(0)
        mu = self.to_mu(h_n)
        logvar = self.to_logvar(h_n)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z_repeated = z.unsqueeze(1).repeat(1, self.window_size, 1)
        h_0 = self.decoder_input(z_repeated)
        out, _ = self.decoder_LSTM(h_0)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
