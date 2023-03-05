import sys
import os
import joblib
import pandas as pd
from tensorflow import keras
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

input_data = []
for line in sys.stdin:
    input_data.append(line.strip().split(","))

input_df = pd.DataFrame(data=input_data[1:], columns=input_data[0])
smile_list = input_df["SMILES"].to_numpy()
#mol_list = [Chem.MolFromSmiles(i) for i in (smile_list)]
#fp = np.array([np.array(AllChem.GetMACCSKeysFingerprint(i)) for i in (mol_list)])
#tsvd = joblib.load(open(os.path.dirname(__file__) + "/tsvd.pkl", "rb"))
#fp = tsvd.transform(fp)

input_df = input_df.drop("SMILES", axis=1)
input_df = input_df.fillna(0)

X = input_df.to_numpy()
#X = np.concatenate([X, fp], axis=1)
scaler_X = joblib.load(open(os.path.dirname(__file__) + "/scaler_X.pkl", "rb"))
X_scaled = scaler_X.transform(X)

model = keras.models.load_model(f"{os.path.dirname(__file__)}/model.h5")
y_pred_scaled = model.predict(X_scaled, verbose=0)

scaler_y = joblib.load(open(os.path.dirname(__file__) + "/scaler_y.pkl", "rb"))
y_pred = scaler_y.inverse_transform(y_pred_scaled)

for val in y_pred:
    print(val[0])
