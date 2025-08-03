import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

df = pd.read_csv(r'C:\Users\Rohan\Code\cyber_anomaly_detection\cyber_anomaly_detection\data\raw\UNSW_NB15_training-set.csv')
DROP_FEATURES = ['id']
CAT_FEATURES = ['proto', 'state', 'service']
LABEL_FEATURES = ['label', 'attack_cat']
NUM_FEATURES = [col for col in df.columns if col not in DROP_FEATURES + LABEL_FEATURES + CAT_FEATURES] #list comprehensions are awesome
df = df.drop(columns=DROP_FEATURES + LABEL_FEATURES)

encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
encoder.fit(df[CAT_FEATURES])
joblib.dump(encoder, 'artifacts/encoder.joblib')
encoded = encoder.transform(df[CAT_FEATURES])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CAT_FEATURES))
df = df.drop(columns=CAT_FEATURES, inplace=True)
df = pd.concat([df, encoded_df], axis=1)

scaler = StandardScaler()
df[NUM_FEATURES] = scaler.fit_transform(df[NUM_FEATURES])
joblib.dump(scaler, 'artifacts/scaler.joblib')
print(df.head())
