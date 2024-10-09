import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
# mpl.rcParams['figure.figsize']=(10,10)


df = pd.read_pickle("../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")[
    ["code", "département", "region", "prix"]
]

df_agg = df.groupby(["region"]).agg({"prix": "mean"}).reset_index()


####
# Visalusation des Disparités Régionales
####

# Charger le fichier shape de la carte des régions
france_regions = gpd.read_file("../resources/geo_data_shp/gadm41_FRA_1.shx")[
    ["NAME_1", "geometry"]
]
# Joindre les données de prix moyens avec les régions
france_regions = france_regions.rename(columns={"NAME_1": "region"})
france_regions = france_regions[france_regions["region"] != "Corse"]


df_joined = france_regions.merge(df_agg, on="region", how="left")
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
plt.colorbar(sm, ax=ax)
plt.title("Prix moyen du GNR par région")
ax.axis("off")

plt.savefig(
    "../report/figures/cartes choroplèthes/disparité_prix_GNR_moyenParRégion_FioulReduc_07_10.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# ####
# # Visalusation Interactive dans le temps
# ####
# import plotly.express as px

# # Créer un graphique choroplèthe avec une animation sur la période
# fig = px.choropleth(
#     df_agg,
#     geojson=france_regions.__geo_interface__,
#     locations="region",
#     color="prix",
#     animation_frame="date",
#     featureidkey="properties.region",
#     projection="mercator",
#     title="Évolution des prix du GNR par région dans le temps",
# )
# fig.update_geos(fitbounds="locations", visible=False)
# fig.show()
