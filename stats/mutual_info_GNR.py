import pandas as pd
from sklearn.feature_selection import mutual_info_regression

df = pd.read_pickle("../data/processed/processed_data_GNR.pkl")
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
plt.savefig("../report/mutual info/most_important_features_GNR.png")
plt.show()
