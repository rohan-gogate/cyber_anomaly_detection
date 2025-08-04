import numpy as np
import torch
import torch.functional as F 

def vae_loss(x_recon, x, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction = 'mean')
    kl_div = -0.5*torch.mean(1+logvar+mu.pow(2)-logvar.exp())
    return recon_loss + kl_div, recon_loss, kl_div
