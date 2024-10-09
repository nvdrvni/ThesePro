import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# read data file
df_source = pd.read_pickle("../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")


# Effectuer les agrégations
df = (
    df_source.groupby(["date", "region"])
    .agg(
        {
            "latitude": "mean",
            "longitude": "mean",
            "population": "sum",
            "surface": "sum",
            "tranche_population": lambda x: x.mode()[0] if not x.mode().empty else None,
            "densité_population": "mean",
            "sap": lambda x: x.mode()[0] if not x.mode().empty else None,
            "depot": lambda x: x.mode()[0] if not x.mode().empty else None,
            "region du dépôt": lambda x: x.mode()[0] if not x.mode().empty else None,
            "indice_CNR": "mean",
            "USD_to_EUR": "mean",
            "EUR_to_USD": "mean",
            "cours_baril_en_USD": "mean",
            "tmoy": "mean",
            "nb_entreprise_ensemble": "sum",
            "nb_entreprise_industrie": "sum",
            "nb_entreprise_construction": "sum",
            "nb_entreprise_commerce": "sum",
            "nb_entreprise_info_communication": "sum",
            "nb_entreprise_autres_services": "sum",
            "nb_entreprise_sciences_techno": "sum",
            "nb_entreprise_immobilier": "sum",
            "nb_entreprise_finance_assurance": "sum",
            "nb_entreprise_santé_enseign_adminis": "sum",
            "nb_exploit_agricole": "sum",
            "prix": "mean",
        }
    )
    .reset_index()
)

X = df.drop("prix", axis=1)
y = df.prix
# Label Encoding of Categorical Data (Missing Values will be encoded as -1)
for col in X.select_dtypes(include=["datetime64[ns]", "period[M]", "object"]).columns:
    X[col], _ = X[col].factorize()

X = X.dropna()
y = y.iloc[X.index]

discrete_features = X.dtypes == "int64"


mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
mi_scores = pd.Series(mi_scores, index=X.columns)

scores = mi_scores.sort_values(ascending=True)  # Ascending order

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15, 15))
width = np.arange(len(mi_scores))
ticks = scores.index
plt.barh(width, scores)
plt.yticks(width, ticks)
plt.savefig(
    "../report/mutual info/most_important_features_GNR_prixMoyenParJourParRegion_07_10.png"
)
plt.show()
