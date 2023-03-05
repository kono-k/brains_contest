from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from build_model import MyNetWork
import os
import pandas as pd
import joblib
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

dataset_df = pd.read_csv("datasets/dataset.csv")
print(dataset_df)
for col in dataset_df.columns:
    if col == "SMILES":
        continue
    dataset_df[col] = dataset_df[col].fillna(0)

y = dataset_df["λmax"].to_numpy()

fp_df = pd.read_csv("datasets\RDK_fingerprint.csv")
fp_df = fp_df.set_index("Unnamed: 0")
fp = fp_df.to_numpy()
pca = PCA(n_components=100)
pca.fit(fp)
fp = pca.transform(fp)
joblib.dump(pca, open(os.path.dirname(__file__) +  "/pca.pkl", "wb"))

dataset_df = dataset_df.drop(["SMILES", "λmax"],  axis=1)

X = dataset_df.to_numpy()
X = np.concatenate([X, fp], axis=1)
print(len(X[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_scaled = scaler.transform(X)
joblib.dump(scaler, open(os.path.dirname(__file__) +  "/scaler_X.pkl", "wb"))

scaler.fit(y.reshape(-1, 1))
y_train_scaled = scaler.transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))
#y_scaled = scaler.transform(y.reshape(-1, 1))
joblib.dump(scaler, open(os.path.dirname(__file__) +  "/scaler_y.pkl", "wb"))

model = MyNetWork()
model.fit(X_train_scaled, y_train_scaled)
#model.fit(X_scaled, y_scaled)
model.plot_history()

train_pred_scaled = model.predict(X_train_scaled)
test_pred_scaled = model.predict(X_test_scaled)

r2_train = r2_score(y_train_scaled, train_pred_scaled)
r2_test = r2_score(y_test_scaled, test_pred_scaled)
print(f"R2 train:{r2_train}")
print(f"R2 test:{r2_test}")

model.save(f"{os.path.dirname(__file__)}\model.h5")