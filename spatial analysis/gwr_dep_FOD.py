import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
# mpl.rcParams['figure.figsize']=(10,10)

# features = [
#     "population",
#     "indice_CNR",
#     "rate_USD_to_EUR",
#     "cours_baril_en_USD",
#     "tmoy",
# ]

df = pd.read_pickle("../data/processed/processed_data_FOD.pkl").drop_duplicates()

###########################
# Régression géographiquement pondérée (GWR)
###########################
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

X = df[["population"]].dropna()
y = df.loc[X.index, "prix"]
y = y.astype("float64")
coords = df.loc[X.index][["longitude", "latitude"]]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
coords = coords.reset_index(drop=True)


# bw = Sel_BW(coords, y, X).search(pool=None)
# model = GWR(coords, y, X, bw, n_jobs=1)
# results = model.fit()

sample_size = 100  # ou un nombre plus petit si nécessaire
np.random.seed(42)  # pour la reproductibilité
indices = np.random.choice(X.shape[0], sample_size, replace=False)

X_sample = X.iloc[indices]
y_sample = y.iloc[indices]
coords_sample = coords.iloc[indices]

X_sample = X_sample.astype(float)
y_sample = y_sample.astype(float)
coords_sample = coords_sample.astype(float)

X_sample_array = np.array(X_sample)
y_sample_array = np.array(y_sample)
coords_sample_array = np.array(coords_sample)

bw = Sel_BW(coords_sample, y_sample, X_sample).search(pool=None)
model = GWR(coords_sample, y_sample, X_sample, bw)
results = model.fit()


# Visualisez les coefficients variables pour chaque feature
for i, feature in enumerate(X.columns):
    plt.figure(figsize=(12, 8))
    df[f"{feature}_coef"] = results.params[:, i]
    df.plot(column=f"{feature}_coef", cmap="RdBu", legend=True)
    plt.title(f"Influence spatiale de {feature}")
    plt.show()
