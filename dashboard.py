import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import time
import matplotlib.pyplot as plt
from src.model import LSTMVAE
import torch.nn.functional as F

window_size = 30
input_dim = 65
st.set_page_config(layout="wide")


st.title("Live Cybersecurity Anomaly Detection")

@st.cache_resource
def load_artifacts():
    model = LSTMVAE(input_dim=input_dim, hidden_dim=64, latent_dim=16, window_size=window_size)
    model.load_state_dict(torch.load("artifacts/lstm_vae.pth", map_location='cpu'))
    model.eval()

    encoder = joblib.load("artifacts/encoder.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    columns = joblib.load("artifacts/feature_columns.joblib")
    return model, encoder, scaler, columns

model, encoder, scaler, expected_columns = load_artifacts()


@st.cache_data
def load_test_data():
    df = pd.read_csv("data/raw/UNSW_NB15_testing-set.csv")
    labels = df['label'].values
    df.drop(columns=['id', 'attack_cat', 'label'], inplace=True)

    CAT_FEATURES = list(encoder.feature_names_in_)
    encoded = encoder.transform(df[CAT_FEATURES])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)

    df.drop(columns=CAT_FEATURES, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    df[df.columns] = scaler.transform(df[df.columns])

    return df, labels

df_full, label_full = load_test_data()


if 'start_idx' not in st.session_state:
    st.session_state.start_idx = 0
if 'history_scores' not in st.session_state:
    st.session_state.history_scores = []
    st.session_state.history_preds = []
    st.session_state.history_labels = []

score_placeholder = st.empty()
plot_placeholder = st.empty()

threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 50.0, 2.5, step=0.1)


st.sidebar.button("Reset Stream", on_click=lambda: st.session_state.update(start_idx=0, history_scores=[], history_preds=[], history_labels=[]))

for i in range(st.session_state.start_idx, len(df_full) - window_size + 1):
    current_window = df_full.iloc[i:i+window_size].to_numpy()
    X_tensor = torch.tensor(current_window[None, :, :], dtype=torch.float32)

    with torch.no_grad():
        x_recon, _, _ = model(X_tensor)
        recon_error = F.mse_loss(x_recon, X_tensor, reduction='none')
        score = recon_error.reshape(1, -1).mean(dim=1).item()

    label = label_full[i + window_size - 1]  
    predicted = score > threshold

    st.session_state.history_scores.append(score)
    st.session_state.history_preds.append(predicted)
    st.session_state.history_labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(0, score, color='red' if predicted else 'blue')
    ax.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
    ax.set_ylim([0, 10])
    ax.set_title(f"Live Score: {score:.4f} | Predicted: {int(predicted)} | True: {label}")
    ax.set_xticks([])
    ax.legend()
    score_placeholder.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(st.session_state.history_scores, label='Anomaly Score')
    ax2.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
    ax2.fill_between(
        range(len(st.session_state.history_scores)),
        0, max(st.session_state.history_scores),
        where=st.session_state.history_preds,
        color='red', alpha=0.3, label='Anomaly'
    )
    ax2.set_title("Score History")
    ax2.set_xlabel("Window Index")
    ax2.set_ylabel("Score")
    ax2.legend()
    plot_placeholder.pyplot(fig2)

    st.session_state.start_idx = i + 1
    time.sleep(1.0)

    if st.session_state.start_idx >= len(df_full) - window_size:
        break
