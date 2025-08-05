# 🚨 Real-Time Cybersecurity Anomaly Detection (LSTM-VAE)

This project implements a **real-time anomaly detection system** for cybersecurity using a **Variational Autoencoder (VAE)** with an **LSTM architecture**. It simulates live network traffic, scores each window with reconstruction error, and visualizes the results in an interactive Streamlit dashboard.

🌐 **Live App**: [demo link](https://cyberanomalydetection.streamlit.app/)

---

## 📌 Key Features

- ✅ Live simulation of network traffic from the UNSW-NB15 dataset  
- ✅ LSTM-based VAE trained on benign traffic only  
- ✅ Streaming anomaly detection via reconstruction error  
- ✅ Scrolling plot of scores with real-time anomaly highlights  
- ✅ Adjustable threshold and reset button in sidebar  
- ✅ Deployed on Streamlit Cloud (no Docker needed)

---

## 🧠 Model Overview

- Input: 65 features (39 numeric + 26 one-hot encoded categorical)
- Architecture: LSTM encoder → latent space → LSTM decoder
- Loss: MSE + KL divergence
- Detection: Windows with high reconstruction error are flagged as anomalies
- Training: Model was trained only on benign samples (unsupervised)

---

## 📈 How It Works

- Data from the UNSW-NB15 dataset is streamed window-by-window
- Each window is encoded, sampled, decoded, and reconstructed
- Mean squared reconstruction error is computed in real time
- High scores are flagged and shown with red overlays on a scrolling plot

---

## 🛠 Run Locally

```bash
git clone https://github.com/rohan-gogate/cyber-anomaly-dashboard.git
cd cyber-anomaly-dashboard
pip install -r requirements.txt
streamlit run dashboard.py
