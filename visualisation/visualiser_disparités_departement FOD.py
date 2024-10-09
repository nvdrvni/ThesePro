import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
# mpl.rcParams['figure.figsize']=(10,10)


df = pd.read_pickle("../data/processed/processed_data_FOD.pkl")[
    ["date", "année-mois", "code", "département", "region", "prix"]
]
df_agg = df.groupby(["département"]).agg({"prix": "mean"}).reset_index()


####
# Visalusation des Disparités Régionales
####

# Charger le fichier shape de la carte des régions
france_departements = gpd.read_file("../resources/contour-des-departements.geojson")
# Joindre les données de prix moyens avec les régions
france_departements = france_departements.rename(
    columns={"code": "département", "nom": "nom_département"}
)
france_departements = france_departements[
    (france_departements["nom_département"] != "Haute-Corse")
    & (france_departements["nom_département"] != "Corse-du-Sud")
]


df_joined = france_departements.merge(df_agg, on="département", how="left")
df_joined["prix"] = df_joined["prix"].astype("float64")

import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter

vmin = df_joined["prix"].min()
vmax = df_joined["prix"].max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("OrRd")

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

df_joined.plot(
    ax=ax,
    color=df_joined["prix"].map(lambda x: cmap(norm(x))),
    edgecolor="black",
    linewidth=0.1,
)

# Ajouter les codes des départements
for idx, row in df_joined.iterrows():
    ax.annotate(
        row.département,
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=8,
    )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Formater les étiquettes de la barre de couleur pour n'afficher que 2 décimales
formatter = FuncFormatter(lambda x, p: format(x, ".2f"))
cbar = plt.colorbar(sm, ax=ax, format=formatter)
cbar.set_label("Prix moyen du FOD (€/L)", rotation=270, labelpad=15)

plt.title("Prix moyen du FOD par département", fontsize=16)
ax.axis("off")

plt.tight_layout()
plt.savefig(
    "../report/figures/cartes choroplèthes/disparité_prix_moyen_par_dép_FOD.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# #######################
# Analyse
# #######################
# Calculer des statistiques descriptives
stats = df_joined["prix"].describe()
print(stats)

# Calculer le coefficient de variation
cv = df_joined["prix"].std() / df_joined["prix"].mean()
print(f"Coefficient de variation: {cv}")

# Top 5 des départements les plus chers et les moins chers
top_5_cher = df_joined.nlargest(5, "prix")[["département", "prix"]]
top_5_moins_cher = df_joined.nsmallest(5, "prix")[["département", "prix"]]
print("Top 5 des départements les plus chers:")
print(top_5_cher)
print("\nTop 5 des départements les moins chers:")
print(top_5_moins_cher)

"""
Voici les stats :



Voici le coefficient de variation : 
Coefficient de variation: 

Voici les top 5 chers
Top 5 des départements les plus chers:



Voici les top 5 moins cher
Top 5 des départements les moins chers:

"""

# #######################
# Analyse de l'autocorrélation spatiale (Indice de Moran)
# #######################
from pysal.explore import esda
from pysal.lib import weights

# Créer une matrice de poids spatiaux
w = weights.Queen.from_dataframe(df_joined.dropna(subset="prix"))

# Calculer l'indice de Moran
moran = esda.Moran(df_joined.dropna(subset="prix")["prix"], w)
print(f"Indice de Moran: {moran.I}")
print(f"p-value: {moran.p_sim}")


# #######################
# Visualisation de la distribution des prix :
# #######################
import numpy as np
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df_joined["prix"], kde=True)
plt.title("Distribution des prix moyens du GNR par département")
plt.xlabel("Prix")
plt.ylabel("Fréquence")
plt.show()
