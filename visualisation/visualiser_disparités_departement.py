import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
# mpl.rcParams['figure.figsize']=(10,10)


df = pd.read_pickle("../data/processed/processed_data.pkl")[
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

vmin = df_joined["prix"].min()
vmax = df_joined["prix"].max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("OrRd")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

df_joined.plot(
    ax=ax,
    color=df_joined["prix"].map(lambda x: cmap(norm(x))),
    edgecolor="black",
    linewidth=0.1,
)


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)
plt.title("Prix moyen du GNR par région")
ax.axis("off")

plt.savefig(
    "../report/figures/disparité_prix_moyen_GNR_par_dép.png",
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
count      62.000000
mean     1193.316939
std        26.587530
min      1139.266964
25%      1182.663690
50%      1192.522726
75%      1203.992280
max      1316.323336
Name: prix, dtype: float64

Voici le coefficient de variation : 
Coefficient de variation: 0.022280358939240057

Voici les top 5 chers
Top 5 des départements les plus chers:

   département         prix
51          53  1316.323336
70          72  1259.053900
59          61  1243.894534
10          11  1236.280604
61          63  1222.723829


Voici les top 5 moins cher
Top 5 des départements les moins chers:
   département         prix
16          17  1139.266964
82          84  1152.537120
12          13  1153.126507
28          30  1158.561699
26          28  1159.767327
"""

# #######################
# Analyse de l'autocorrélation spatiale (Indice de Moran)
# #######################
from pysal.explore import esda
from pysal.lib import weights

# Créer une matrice de poids spatiaux
w = weights.Queen.from_dataframe(df_joined)

# Calculer l'indice de Moran
moran = esda.Moran(df_joined["prix"], w)
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
