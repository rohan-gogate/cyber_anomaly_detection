import pandas as pd
import numpy as np
import torch
import joblib
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from src.model import LSTMVAE


window_size = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'artifacts/lstm_vae.pth'
encoder_path = 'artifacts/encoder.joblib'
scaler_path = 'artifacts/scaler.joblib'
test_path = 'data/raw/UNSW_NB15_testing-set.csv'

CAT_FEATURES = ['proto', 'state', 'service']
DROP_FEATURES = ['id', 'attack_cat']


df = pd.read_csv(test_path)


labels = df['label'].values.copy()
df.drop(columns=['label'], inplace=True)

df.drop(columns=[col for col in DROP_FEATURES if col in df.columns], inplace=True)

encoder = joblib.load(encoder_path)
CAT_FEATURES = list(encoder.feature_names_in_)
assert all(col in df.columns for col in CAT_FEATURES), "missing expected categorical columns"

encoded = encoder.transform(df[CAT_FEATURES])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(),  
    index=df.index
)

df.drop(columns=CAT_FEATURES, inplace=True)
df = pd.concat([df, encoded_df], axis=1)

scaler = joblib.load(scaler_path)
df[df.columns] = scaler.transform(df[df.columns])
def create_windows(data, window_size):
    return np.stack([data[i:i+window_size] for i in range(len(data) - window_size + 1)])

X_test = create_windows(df.to_numpy(), window_size)
label_windows = labels[window_size - 1:]

X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

input_dim = X_tensor.shape[2]
model = LSTMVAE(input_dim, hidden_dim=64, latent_dim=16, window_size=window_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    x_recon, _, _ = model(X_tensor)
    recon_error = F.mse_loss(x_recon, X_tensor, reduction='none')
    scores = recon_error.reshape(X_tensor.shape[0], -1).mean(dim=1).cpu().numpy()
threshold = np.percentile(scores, 65)
predicted_anomalies = scores > threshold

precision = precision_score(label_windows, predicted_anomalies)
recall = recall_score(label_windows, predicted_anomalies)
f1 = f1_score(label_windows, predicted_anomalies)

print(f"Threshold: {threshold:.6f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
