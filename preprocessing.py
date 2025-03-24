import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import torch
import torch.nn as nn
from formulaencoding import encode_formula

def preprocess(filepath):
    df = pd.read_csv(filepath)
    df['formula_charge_encoded'] = df['formula_charge'].apply(lambda x: encode_formula(x).tolist())
    df['formula_discharge_encoded'] = df['formula_discharge'].apply(lambda x: encode_formula(x).tolist())

    df_charge = pd.DataFrame(df['formula_charge_encoded'].tolist(), columns=[f'charge_{i+1}' for i in range(118)])
    df_discharge = pd.DataFrame(df['formula_discharge_encoded'].tolist(), columns=[f'discharge_{i+1}' for i in range(118)])
    df = pd.concat([df, df_charge, df_discharge], axis=1)

    num_classes, emb_dim = 230, 10
    embedding_layer = nn.Embedding(num_classes, emb_dim)
    charge_tensor = torch.tensor(df['charge_space_group_number'].values - 1)
    discharge_tensor = torch.tensor(df['discharge_space_group_number'].values - 1)

    df_emb_charge = pd.DataFrame(embedding_layer(charge_tensor).detach().numpy(),
                                 columns=[f'charge_emb_{i}' for i in range(emb_dim)])
    df_emb_discharge = pd.DataFrame(embedding_layer(discharge_tensor).detach().numpy(),
                                    columns=[f'discharge_emb_{i}' for i in range(emb_dim)])
    df = pd.concat([df, df_emb_charge, df_emb_discharge], axis=1)

   
    df_working_ion = pd.get_dummies(df['working_ion'], prefix='ion')
    df_steps = pd.get_dummies(df['num_steps'], prefix='steps')
    df = pd.concat([df, df_working_ion, df_steps], axis=1)

    drop_cols = ['battery_id', 'battery_formula', 'framework_formula', 'adj_pairs', 'capacity_vol', 
                 'energy_vol', 'formula_charge', 'formula_discharge', 'formula_charge_encoded', 
                 'formula_discharge_encoded', 'id_charge', 'id_discharge', 'working_ion', 'num_steps',
                 'stability_charge', 'stability_discharge', 'charge_crystal_system', 'charge_energy_per_atom',
                 'charge_formation_energy_per_atom', 'charge_band_gap', 'charge_efermi',
                 'discharge_crystal_system', 'discharge_energy_per_atom', 'discharge_formation_energy_per_atom',
                 'discharge_band_gap', 'discharge_efermi']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    x_df = df.drop(columns=['average_voltage'])
    y_df = df[['average_voltage']]

    split = int(len(df) * 0.9)
    x_train, x_test = x_df.iloc[:split], x_df.iloc[split:]
    y_train, y_test = y_df.iloc[:split], y_df.iloc[split:]

    pca = PCA(0.99)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    scaler = preprocessing.MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_pca)
    x_test_scaled = scaler.transform(x_test_pca)

    return x_train_scaled, y_train.values, x_test_scaled, y_test.values
