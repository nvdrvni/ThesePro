import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import geopandas as gpd

mpl.rcParams["figure.dpi"] = 300
# Configuration du style
sns.set_style("whitegrid")
sns.set_palette("deep")
sns.set_context("paper")


df = pd.read_pickle("../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")

# Stats descriptives globales Prix GRN
display(df.prix.describe())

# Stats descriptives par région
for region in df.region.unique():
    df_reg = df[df["region"] == region]
    print("Région ", region)
    print(df_reg.prix.describe())

df.prix.hist()


# Fonction pour sauvegarder les figures
def save_fig(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches="tight")


# 1. Histogramme Global Prix
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x="prix", kde=True, color="blue", ax=ax)
ax.set_title("Distribution des prix du GNR", fontsize=16)
ax.set_xlabel("Prix", fontsize=12)
ax.set_ylabel("Fréquence", fontsize=12)
save_fig(fig, "../report/figures/descriptive stats/histogramme_prix_gnr.png")

# 2. Box plot Prix par Région
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x="region", y="prix", ax=ax)
ax.set_title("Distribution des prix du GNR par région", fontsize=16)
ax.set_xlabel("Région", fontsize=12)
ax.set_ylabel("Prix", fontsize=12)
plt.xticks(rotation=45, ha="right")
save_fig(fig, "../report/figures/descriptive stats/boxplot_prix_gnr_par_region.png")

# 3. Violin plot (une alternative élégante au box plot)
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=df, x="region", y="prix", ax=ax)
ax.set_title("Distribution des prix du GNR par région", fontsize=16)
ax.set_xlabel("Région", fontsize=12)
ax.set_ylabel("Prix", fontsize=12)
plt.xticks(rotation=45, ha="right")
save_fig(fig, "../report/figures/descriptive stats/violinplot_prix_gnr_par_region.png")

# 4. Évolution temporelle des prix
fig, ax = plt.subplots(figsize=(12, 6))
df.groupby("date")["prix"].mean().plot(ax=ax)
ax.set_title("Évolution du prix moyen du GNR", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Prix moyen", fontsize=12)
save_fig(fig, "../report/figures/descriptive stats/evolution_temporelle_prix_gnr.png")

# 5. Heatmap des corrélations
corr_vars = df.select_dtypes(include=["Int64", "float"]).columns
corr_matrix = df[corr_vars].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Matrice de corrélation des variables", fontsize=16)
save_fig(fig, "../report/figures/descriptive stats/heatmap_correlations.png")

# 6. Scatter plot: Prix vs. Indice CNR
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="indice_CNR", y="prix", alpha=0.5, ax=ax)
ax.set_title("Prix du GNR en fonction de l'indice CNR", fontsize=16)
ax.set_xlabel("Indice CNR", fontsize=12)
ax.set_ylabel("Prix du GNR", fontsize=12)
save_fig(fig, "../report/figures/descriptive stats/scatter_prix_vs_indice_cnr.png")


# 7. Calculez le prix moyen par département
france = gpd.read_file("../resources/contour-des-departements.geojson")
# Joindre les données de prix moyens avec les régions
france_departements = france.rename(
    columns={"code": "département", "nom": "nom_département"}
)
france_departements = france_departements[
    (france_departements["nom_département"] != "Haute-Corse")
    & (france_departements["nom_département"] != "Corse-du-Sud")
]

prix_moyen_dep = df.groupby("département")["prix"].mean().reset_index()
france = france_departements.merge(prix_moyen_dep, on="département", how="left")

fig, ax = plt.subplots(figsize=(15, 10))
france.plot(
    column="prix",
    cmap="YlOrRd",
    linewidth=0.8,
    edgecolor="0.8",
    ax=ax,
    legend=True,
    legend_kwds={"label": "Prix moyen du GNR"},
)
ax.set_title("Distribution géographique des prix moyens du GNR", fontsize=16)
ax.axis("off")
save_fig(fig, "../report/figures/descriptive stats/carte_chaleur_prix_gnr.png")

# 9. Graphique en ligne: Évolution temporelle par région
fig, ax = plt.subplots(figsize=(12, 6))
for region in df["region"].unique():
    df[df["region"] == region].groupby("date")["prix"].mean().plot(ax=ax, label=region)
ax.set_title("Évolution du prix du GNR par région", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Prix du GNR", fontsize=12)
ax.legend(title="Région", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
save_fig(fig, "../report/figures/descriptive stats/evolution_prix_par_region.png")
