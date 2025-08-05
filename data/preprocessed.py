import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

os.makedirs('artifacts', exist_ok=True)

df = pd.read_csv(r'C:\Users\Rohan\Code\cyber_anomaly_detection\cyber_anomaly_detection\data\raw\UNSW_NB15_training-set.csv')
DROP_FEATURES = ['id']
CAT_FEATURES = ['proto', 'state', 'service']
LABEL_FEATURES = ['label', 'attack_cat']
df = df[df['label'] == 0]  
df.drop(columns=[col for col in DROP_FEATURES + LABEL_FEATURES if col in df.columns], inplace=True)
NUM_FEATURES = [col for col in df.columns if col not in CAT_FEATURES]

encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
encoder.fit(df[CAT_FEATURES])
joblib.dump(encoder, 'artifacts/encoder.joblib')
encoded = encoder.transform(df[CAT_FEATURES])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CAT_FEATURES), index=df.index)
df.drop(columns=CAT_FEATURES, inplace=True)
df = pd.concat([df, encoded_df], axis=1)

scaler = StandardScaler()
scaler.fit(df)
joblib.dump(scaler, 'artifacts/scaler.joblib')

df.to_csv('artifacts/preprocessed_training_data.csv', index=False)
joblib.dump(df.columns.tolist(), 'artifacts/feature_columns.joblib')