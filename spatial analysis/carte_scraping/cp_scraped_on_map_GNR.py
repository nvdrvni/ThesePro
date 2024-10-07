import pandas as pd
import folium
import requests

# lire la dataframe
df = pd.read_pickle("../../data/processed/processed_data_GNR.pkl")[
    ["longitude", "latitude"]
].drop_duplicates()

# Créer une carte centrée sur la France
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

# Ajouter des points depuis la DataFrame
for index, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=0.2,  # Taille plus petite des cercles
        color="#8B0000",
        fill=True,
        fill_color="#8B0000",
        fill_opacity=0,  # Transparence des points
    ).add_to(m)


# Charger le fichier GeoJSON des départements français
url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
response = requests.get(url)
geojson_data = response.json()

# Ajouter les frontières des départements sur la carte
folium.GeoJson(
    geojson_data,
    name="Départements",
    style_function=lambda x: {"color": "#00008B", "weight": 1, "fillOpacity": 0},
).add_to(m)

# # Ajouter un contrôle pour activer/désactiver les frontières
# folium.LayerControl().add_to(m)

# Charger le fichier GeoJSON des régions françaises
url_regions = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
response_reg = requests.get(url_regions)
geojson_regions = response_reg.json()

# Ajouter les frontières des régions sur la carte
folium.GeoJson(
    geojson_regions,
    name="Régions",
    style_function=lambda x: {"color": "#00008B", "weight": 3, "fillOpacity": 0},
).add_to(m)

# Enregistrer la carte dans un fichier HTML
m.save(
    "../../report/figures/carte_scraping_state/carte_france_transparence_frontieres_region_GNR.html"
)
