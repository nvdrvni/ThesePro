import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Charger les données
df = pd.read_pickle("../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")
df_agg = df.groupby(["region", "date"]).agg({"prix": "mean"}).reset_index()

# Charger le fichier shape de la carte des régions
france_regions = gpd.read_file("../resources/geo_data_shp/gadm41_FRA_1.shx")[
    ["NAME_1", "geometry"]
]
france_regions = france_regions.rename(columns={"NAME_1": "region"})

# Supprimer la Corse si elle n'est pas dans df_agg
france_regions = france_regions[france_regions["region"] != "Corse"]

# Joindre les données de prix moyens avec les régions
df_joined = france_regions.merge(df_agg, on="region", how="left")

# Gérer les valeurs manquantes si nécessaire
df_joined["prix"] = df_joined["prix"].fillna(df_joined["prix"].mean())

# Préparer la colormap
vmin = df_joined["prix"].min()
vmax = df_joined["prix"].max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("OrRd")

# Créer la figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Tracer la carte
df_joined.plot(
    ax=ax,
    edgecolor="black",
    linewidth=0.1,
    color=df_joined["prix"].map(lambda x: cmap(norm(x))),
)

# Ajouter les étiquettes des régions
for idx, row in df_joined.iterrows():
    plt.annotate(
        text=row["region"],
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        xytext=(-40, 5),
        textcoords="offset points",
        fontsize=8,
    )

# Configurer la barre de couleur
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)

# Configurer le graphique
plt.title("Prix moyen du GNR par région")
ax.axis("off")

# Sauvegarder et afficher
plt.savefig(
    "../report/figures/cartes choroplèthes/disparité_prix_moyen_GNR_labels_FioulReduc_07_10.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
