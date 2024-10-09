import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
# mpl.rcParams['figure.figsize']=(10,10)


df = pd.read_pickle("../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")[
    ["année-mois", "code", "département", "region", "prix"]
]

df_agg = df.groupby(["département"]).agg({"prix": "mean"}).reset_index()


####
# Visalusation des Disparités par Départements
####

# Charger le fichier shape de la carte des départements
france_departements = gpd.read_file("../resources/contour-des-departements.geojson")
# Joindre les données de prix moyens avec les dép
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

# Ajouter les étiquettes des dép
for idx, row in df_joined.iterrows():
    plt.annotate(
        text=row["nom_département"],
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        xytext=(-40, 5),
        textcoords="offset points",
        fontsize=5,
    )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax)
plt.title("Prix moyen du GNR par région")
ax.axis("off")

plt.savefig(
    "../report/figures/cartes choroplèthes/disparité_prix_moyen_GNR_par_dép_FioulReduc_07_10.png",
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
