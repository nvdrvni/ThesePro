import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

# Supposons que df est votre DataFrame avec une colonne 'date' et les autres variables
df = pd.read_pickle(
    "../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl"
).dropna()

# 1. Préparation des données
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")


# 2. Fonction pour calculer les corrélations glissantes
def rolling_correlations(data, window_size=30):
    correlations = {}
    for col in data.columns[1:]:  # Ignorer la première colonne pour la boucle
        corr_series = data[data.columns[0]].rolling(window=window_size).corr(data[col])
        # Remplacer les valeurs infinies par NaN et supprimer les valeurs manquantes
        corr_series = corr_series.replace([np.inf, -np.inf], np.nan).dropna()
        correlations[col] = corr_series
    return pd.DataFrame(correlations, index=data.index)


# 3. Calcul des corrélations glissantes
variables = ["prix", "indice_CNR", "USD_to_EUR", "cours_baril_en_USD"]
rolling_corr = rolling_correlations(df[variables], window_size=30)

# 4. Visualisation des corrélations glissantes
plt.figure(figsize=(12, 8))
for var in variables[1:]:
    plt.plot(df["date"][30:], rolling_corr[var][30:], label=f"Prix vs {var}")
plt.title("Évolution des corrélations glissantes")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Corrélation")
plt.show()

# 5. Analyse saisonnière
df["season"] = df["date"].dt.quarter
variables_no_prix = [var for var in variables if var != "prix"]
seasonal_corr = df.groupby("season")[variables].corr()
seasonal_corr_prix = seasonal_corr.xs("prix", level=1)
# Filtrer pour ne garder que les variables souhaitées, en excluant 'prix' lui-même
seasonal_corr_filtered = seasonal_corr_prix[variables_no_prix]

# Visualisation des corrélations saisonnières
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(seasonal_corr_filtered, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(cax)

for (i, j), val in np.ndenumerate(seasonal_corr_filtered):
    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

plt.xticks(
    range(len(seasonal_corr_filtered.columns)),
    seasonal_corr_filtered.columns,
    rotation=45,
)
plt.yticks(range(len(seasonal_corr_filtered.index)), seasonal_corr_filtered.index)
plt.show()


# 6. Test de stationnarité des corrélations
def test_stationarity(timeseries):
    result = adfuller(timeseries.dropna(), autolag="AIC")
    return result[1]  # Retourner la p-value


stationarity_results = {
    var: test_stationarity(rolling_corr[var][30:]) for var in variables[1:]
}
print("P-values du test de stationnarité des corrélations:")
print(stationarity_results)

# 7. Corrélations conditionnelles (exemple avec deux variables)
returns = df[["prix", "indice_CNR"]].pct_change().dropna()
model = arch_model(
    returns["prix"],
    vol="GARCH",
    p=1,
    o=0,
    q=1,
    dist="normal",
    mean="Zero",
    rescale=False,
)
res = model.fit(update_freq=5)


# 8. Heatmap des corrélations pour différentes périodes
def plot_correlation_heatmap(data, title):
    corr = data[variables].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()


# Diviser les données en deux périodes
mid_point = len(df) // 2
plot_correlation_heatmap(
    df.iloc[:mid_point], "Corrélations - Première moitié de la période"
)
plot_correlation_heatmap(
    df.iloc[mid_point:], "Corrélations - Seconde moitié de la période"
)

# 9. Analyse des événements (exemple)
event_date = "2024-07-01"  # Date hypothétique d'un événement important
before_event = df[df["date"] < event_date]
after_event = df[df["date"] >= event_date]

plot_correlation_heatmap(before_event, "Corrélations avant l'événement")
plot_correlation_heatmap(after_event, "Corrélations après l'événement")
