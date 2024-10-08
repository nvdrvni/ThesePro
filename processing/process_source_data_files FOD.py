import pandas as pd
import numpy as np
from glob import glob


def read_process_pv_df():
    df_pv = pd.read_pickle("../data/source/data_pv_fod.pkl")[
        ["date", "depot", "sap", "site", "prix"]
    ]
    # df_pv['date'] =   df_pv['date'].astype(str)
    df_pv["sap"] = df_pv["sap"].apply(str.strip)  # strip SAP column

    df_pv["site"] = "Carrefour"

    df_pv = df_pv.merge(df_depot[["code", "sap"]], on="sap", how="inner")
    df_pv = df_pv[["date", "code", "sap", "depot", "site", "prix"]]
    df_pv["depot"] = df_pv["depot"].apply(str.strip)
    df_pv["date"] = pd.to_datetime(df_pv["date"])
    return df_pv


def read_process_concu_df():
    df_concu = pd.read_pickle("../data/source/data_concu_fod.pkl")
    df_concu.loc[df_concu["sap"] == "NICE", "sap"] = "CNIC"

    concu_columns = [
        "Prix_aubinsfioul",
        "Prix_bretagne_multienergies",
        "Prix_campusidf",
        "Prix_clicandfioul",
        "Prix_dyneff",
        "Prix_eldenergie",
        "Prix_fioulaumeilleurprix",
        "Prix_fiouleleclerc",
        "Prix_fioulexpress",
        "Prix_fioulmarket",
        "Prix_fioulmoinscher",
        "Prix_fioulreduc",
        "Prix_hellofioul",
        "Prix_leclerc_chalon",
        "Prix_leclerc_normandie",
        "Prix_leclerc_serviceouest",
        "Prix_ohfioul",
        "Prix_rossicarburant",
    ]

    for concu in concu_columns:
        df_concu.rename(columns={concu: concu[5].upper() + concu[6:]}, inplace=True)

    df_concu_melted = (
        pd.melt(
            df_concu,
            id_vars=df_concu.columns[:3],
            var_name="site",  # Nom de la nouvelle colonne pour les anciens noms de colonnes
            value_name="prix",
        )
        .dropna(subset="prix")
        .reset_index(drop=True)
    )

    df_concu_melted = df_concu_melted.merge(
        df_depot[["code", "sap", "depot"]], on=["sap", "code"], how="left"
    )

    df_concu_melted = df_concu_melted[["date", "code", "sap", "depot", "site", "prix"]]
    df_concu_melted["depot"] = df_concu_melted["depot"].apply(str.strip)
    df_concu_melted["date"] = pd.to_datetime(df_concu_melted["date"])
    return df_concu_melted


def read_process_communes_df():
    df_communes = pd.read_csv(
        "../data/source/communes2024.csv",
        usecols=["code_postal", "latitude", "longitude", "population", "surface"],
    )
    df_communes.rename(columns={"code_postal": "code"}, inplace=True)

    df_communes["code"] = (
        df_communes["code"].astype(str).apply(lambda x: "0" + x if len(x) < 5 else x)
    )

    df_communes["surface"] = df_communes["surface"] / 100  # Convertir surface en Km²

    # Group BY code (duplicated codes)
    apply_map = {
        "latitude": "mean",
        "longitude": "mean",
        "population": "sum",
        "surface": "sum",
    }

    df_communes = df_communes.groupby(["code"]).agg(apply_map).reset_index()

    df_communes["tranche_population"] = (
        df_communes["population"].apply(assign_population_index).astype(str)
    )
    return df_communes


def read_process_regions_df():
    df_regions = pd.read_pickle("../data/source/data_region.pkl")
    return df_regions


def read_process_depot_df():
    df_depot = pd.read_pickle("../data/source/data_depot_fod.pkl").drop(
        ["ZT", "BASSIN"], axis=1
    )

    df_depot.loc[df_depot["depot"] == "ST PIERRE DES CORPS", "depot"] = (
        "SAINT PIERRE DES CORPS"
    )

    df_depot["code"] = (
        df_depot["code"].astype(str).apply(lambda x: "0" + x if len(x) < 5 else x)
    )
    return df_depot


def read_process_indiceCNR_df():
    df_indice_cnr = pd.read_csv(
        "../data/source/Indice CNR gazole professionnel_2022_2024.csv", sep=";"
    )

    df_indice_cnr.rename(
        columns={"Date": "année-mois", "Indice CNR gazole professionnel": "indice_CNR"},
        inplace=True,
    )
    df_indice_cnr["année-mois"] = pd.to_datetime(df_indice_cnr["année-mois"])
    df_indice_cnr["année-mois"] = df_indice_cnr["année-mois"].dt.to_period("M")

    df_indice_cnr.indice_CNR = df_indice_cnr.indice_CNR.astype("float64")
    return df_indice_cnr


def read_process_changeRate_df():
    df_taux_change = pd.read_excel("../data/source/ExchangeRate.xlsx")

    df_taux_change["date"] = pd.to_datetime(df_taux_change["date"], format="%b %d, %Y")
    # df_taux_change["date"] = df_taux_change["date"].dt.strftime("%Y-%m-%d")

    df_taux_change["EUR_to_USD"] = (1 / df_taux_change["USD_to_EUR"]).apply(
        lambda x: round(x, 2)
    )
    return df_taux_change


def read_process_temperature_moyenne_df():
    df_temperature = pd.read_csv(
        "../data/source/temperature-quotidienne-departementale.csv",
        sep=";",
        usecols=["date_obs", "code_insee_departement", "tmoy"],
    )

    df_temperature = df_temperature.rename(
        columns={"date_obs": "date", "code_insee_departement": "département"}
    )

    # df_temperature["date"] = df_temperature["date"].apply(
    #     lambda x: x[-4:] + "-" + x[3:5] + "-" + x[:2]
    # )

    df_temperature["tmoy"] = df_temperature["tmoy"].astype("float64")
    df_temperature["date"] = pd.to_datetime(df_temperature["date"])

    return df_temperature


def read_entreprise_secteur_activité():
    df_entreprises_secteur = pd.read_csv(
        "../data/source/entreprise_par_secteur_dactivité_par_cp.csv",
        sep=";",
        encoding="latin1",
    )
    df_entreprises_secteur = df_entreprises_secteur.drop("Libellé", axis=1)

    df_entreprises_secteur.rename(columns={"Code": "code"}, inplace=True)

    for col in df_entreprises_secteur.columns[1:]:
        df_entreprises_secteur[col] = df_entreprises_secteur[col].astype("float64")

    return df_entreprises_secteur


def read_exploitations_agricole():
    df_exploit_agricole = pd.read_excel("../data/source/agri.xlsx", header=4).drop(
        ["libgeo", "an"], axis=1
    )
    df_exploit_agricole.rename(
        columns={"exp2020": "nb_exploit_agricole", "codgeo": "code"}, inplace=True
    )
    return df_exploit_agricole


def read_process_cours_baril_df():
    df_baril = pd.read_csv("../data/source/cours_baril_Brent.csv")
    df_baril.rename(
        columns={"DATE": "date", "DCOILBRENTEU": "cours_baril_en_USD"}, inplace=True
    )

    # df_baril.drop(
    #     df_baril[df_baril["cours_baril_en_USD"] == "."].index, axis=0, inplace=True
    # )

    serie_num = pd.to_numeric(df_baril.cours_baril_en_USD.replace(".", np.nan))

    # Créer un masque pour les valeurs NaN
    masque = serie_num.isna()

    # Calculer les indices des valeurs non-NaN les plus proches
    indices_valides = np.where(~masque)[0]
    indices_plus_proches_avant = pd.Series(masque.index).map(
        lambda x: (
            indices_valides[indices_valides <= x][-1]
            if any(indices_valides <= x)
            else indices_valides[0]
        )
    )
    indices_plus_proches_apres = pd.Series(masque.index).map(
        lambda x: (
            indices_valides[indices_valides >= x][0]
            if any(indices_valides >= x)
            else indices_valides[-1]
        )
    )

    # Calculer la moyenne des valeurs avant et après
    valeurs_avant = serie_num.iloc[indices_plus_proches_avant].values
    valeurs_apres = serie_num.iloc[indices_plus_proches_apres].values
    moyenne = (valeurs_avant + valeurs_apres) / 2

    # Remplacer uniquement les valeurs NaN par la moyenne calculée
    serie_num.loc[masque] = moyenne[masque]

    df_baril["cours_baril_en_USD"] = serie_num
    df_baril["date"] = pd.to_datetime(df_baril["date"])
    return df_baril


# ===============================
# Intermediate functions for editting columns
# ===============================


# Assigner un indice Population à chaque commuen en fonction du nombre d'habitants
def assign_population_index(population):
    if population < 5000:
        return 1
    elif population < 10000:
        return 2
    elif population < 20000:
        return 3
    elif population < 50000:
        return 4
    elif population < 100000:
        return 5
    elif population < 200000:
        return 6
    elif population < 2000000:
        return 7
    else:
        return 8


# Convertir date format pour la df taux de change
def convert_date_ChangeRate(date_str):
    date = date_str.split()[0]
    m, d, y = date.split("/")
    if len(m) == 1:
        m = "0" + m
    if len(d) == 1:
        d = "0" + d
    return y + "-" + m + "-" + d


# Remplir manuellement les régions manquantes
def fill_missing_regions(depot):
    if depot == "ST BAUSSANT":
        return "Grand Est"
    elif depot == "STOCBREST":
        return "Bretagne"
    elif depot == "COURNON":
        return "Auvergne Rhones Alpes"
    elif depot == "COIGNIERES":
        return "Ile de France"
    elif depot == "DONGES":
        return "Pays de la Loire"
    elif depot == "FOS":
        return "Provence Alpes Côte d'Azur"
    elif depot == "ST JEAN R" or depot == "GDH":
        return "Occitanie"
    elif depot == "LE MANS ":
        return "Pays de la Loire"
    elif depot == "MITRY":
        return "Ile de France"
    elif depot == "LA ROCHELLE":
        return "Nouvelle Aquitaine"
    elif depot == "ST POL" or depot == "SOGEPP":
        return "Hauts de France"
    elif depot == "VERN":
        return "Bretagne"
    elif depot == "VALENCIENNES":
        return "Hauts de France"


# Fonction pour extraire le département d'un code postal et retourner la région correspondante
def get_region_from_code_postal(code_postal):
    departement = str(code_postal)[:2]  # Extraire les deux premiers chiffres
    return departement_to_region.get(departement, np.nan)  # Retourne la région NaN


# ===============================
# Recup df et fusion
# ===============================

df_depot = read_process_depot_df()
df_pv = read_process_pv_df()
df_concu = read_process_concu_df()
df_regions = read_process_regions_df()
df_communes = read_process_communes_df()
df_indice_cnr = read_process_indiceCNR_df()
df_taux_change = read_process_changeRate_df()
df_temperature_moy = read_process_temperature_moyenne_df()
df_baril = read_process_cours_baril_df()
df_entreprises_secteur = read_entreprise_secteur_activité()
df_exploit_agricole = read_exploitations_agricole()

# ===============================
# Recup df et fusion
# ===============================
df_prices = pd.concat([df_concu, df_pv], axis=0)
df_prices = df_prices.merge(df_regions[["sap", "region"]], on="sap", how="left")

###################### GARDER QUE LES DATES ENTRE 2023/09/01 et 2024/09/01
df_prices = df_prices[
    (df_prices["date"] >= pd.to_datetime("2023-09-01"))
    & (df_prices["date"] < pd.to_datetime("2024-09-01"))
].sort_values(by="date")

# df_prices = df_prices[df_prices["date"] >= "2023-09-01"]

# ===============================
# Traiter les depots avec région manquante
# ===============================

list_depots_with_mssing_regions = df_prices[df_prices["region"].isna()][
    "depot"
].unique()


df_prices_filled = df_prices.copy()
df_prices_filled.loc[df_prices_filled["region"].isna(), "region"] = df_prices_filled[
    "depot"
].apply(
    lambda x: fill_missing_regions(x) if x in list_depots_with_mssing_regions else None
)


# ===============================
# Créer une colonne Région (du code postal)
# ===============================
# Rename region column (these regions are for depots)
df_prices_filled.rename(columns={"region": "region du dépôt"}, inplace=True)

# Dictionnaire des départements métropolitains et régions
departement_to_region = {
    "01": "Auvergne-Rhône-Alpes",
    "02": "Hauts-de-France",
    "03": "Auvergne-Rhône-Alpes",
    "04": "Provence-Alpes-Côte d'Azur",
    "05": "Provence-Alpes-Côte d'Azur",
    "06": "Provence-Alpes-Côte d'Azur",
    "07": "Auvergne-Rhône-Alpes",
    "08": "Grand Est",
    "09": "Occitanie",
    "10": "Grand Est",
    "11": "Occitanie",
    "12": "Occitanie",
    "13": "Provence-Alpes-Côte d'Azur",
    "14": "Normandie",
    "15": "Auvergne-Rhône-Alpes",
    "16": "Nouvelle-Aquitaine",
    "17": "Nouvelle-Aquitaine",
    "18": "Centre-Val de Loire",
    "19": "Nouvelle-Aquitaine",
    "21": "Bourgogne-Franche-Comté",
    "22": "Bretagne",
    "23": "Nouvelle-Aquitaine",
    "24": "Nouvelle-Aquitaine",
    "25": "Bourgogne-Franche-Comté",
    "26": "Auvergne-Rhône-Alpes",
    "27": "Normandie",
    "28": "Centre-Val de Loire",
    "29": "Bretagne",
    "30": "Occitanie",
    "31": "Occitanie",
    "32": "Occitanie",
    "33": "Nouvelle-Aquitaine",
    "34": "Occitanie",
    "35": "Bretagne",
    "36": "Centre-Val de Loire",
    "37": "Centre-Val de Loire",
    "38": "Auvergne-Rhône-Alpes",
    "39": "Bourgogne-Franche-Comté",
    "40": "Nouvelle-Aquitaine",
    "41": "Centre-Val de Loire",
    "42": "Auvergne-Rhône-Alpes",
    "43": "Auvergne-Rhône-Alpes",
    "44": "Pays de la Loire",
    "45": "Centre-Val de Loire",
    "46": "Occitanie",
    "47": "Nouvelle-Aquitaine",
    "48": "Occitanie",
    "49": "Pays de la Loire",
    "50": "Normandie",
    "51": "Grand Est",
    "52": "Grand Est",
    "53": "Pays de la Loire",
    "54": "Grand Est",
    "55": "Grand Est",
    "56": "Bretagne",
    "57": "Grand Est",
    "58": "Bourgogne-Franche-Comté",
    "59": "Hauts-de-France",
    "60": "Hauts-de-France",
    "61": "Normandie",
    "62": "Hauts-de-France",
    "63": "Auvergne-Rhône-Alpes",
    "64": "Nouvelle-Aquitaine",
    "65": "Occitanie",
    "66": "Occitanie",
    "67": "Grand Est",
    "68": "Grand Est",
    "69": "Auvergne-Rhône-Alpes",
    "70": "Bourgogne-Franche-Comté",
    "71": "Bourgogne-Franche-Comté",
    "72": "Pays de la Loire",
    "73": "Auvergne-Rhône-Alpes",
    "74": "Auvergne-Rhône-Alpes",
    "75": "Île-de-France",
    "76": "Normandie",
    "77": "Île-de-France",
    "78": "Île-de-France",
    "79": "Nouvelle-Aquitaine",
    "80": "Hauts-de-France",
    "81": "Occitanie",
    "82": "Occitanie",
    "83": "Provence-Alpes-Côte d'Azur",
    "84": "Provence-Alpes-Côte d'Azur",
    "85": "Pays de la Loire",
    "86": "Nouvelle-Aquitaine",
    "87": "Nouvelle-Aquitaine",
    "88": "Grand Est",
    "89": "Bourgogne-Franche-Comté",
    "90": "Bourgogne-Franche-Comté",
    "91": "Île-de-France",
    "92": "Île-de-France",
    "93": "Île-de-France",
    "94": "Île-de-France",
    "95": "Île-de-France",
}
df_prices_filled["region"] = df_prices_filled["code"].apply(get_region_from_code_postal)


# ===============================
# Fusionner avec données Communes (Code, Lat, Long, Population)
# ===============================
df_all_prices_filled = df_prices_filled.merge(df_communes, on="code", how="left")


# ===============================
# Créer un colonne Année-Mois
# ===============================
df_all_prices_filled["date"] = pd.to_datetime(df_all_prices_filled["date"])
df_all_prices_filled["année-mois"] = df_all_prices_filled["date"].dt.to_period("M")

# ===============================
# Inclure la variable Indice CNR (gazole professionnel)
# ===============================
df_all_prices_filled = df_all_prices_filled.merge(
    df_indice_cnr, on="année-mois", how="left"
)

# ===============================
# Inclure la variable Taux de Change
# ===============================
df_all_prices_filled = df_all_prices_filled.merge(df_taux_change, on="date", how="left")


# ===============================
# Inclure une colonne Département
# ===============================
df_all_prices_filled["département"] = df_all_prices_filled["code"].apply(
    lambda x: x[:2]
)


# ===============================
# Inclure la variable Température
# ===============================
df_all_prices_filled = df_all_prices_filled.merge(
    df_temperature_moy, on=["date", "département"], how="left"
)


# ===============================
# Créer une varibale Densité de Population
# ===============================
df_all_prices_filled["densité_population"] = np.round(
    df_all_prices_filled["population"] / df_all_prices_filled["surface"], 2
)


# ===============================
# Inclure la variable Cours Baril Brent
# ===============================
df_all_prices_filled = df_all_prices_filled.merge(
    df_baril,
    on=[
        "date",
    ],
    how="left",
)

# ===============================
# Inclure les données nombres entreprises par secteur d'activité
# ===============================
df_all_prices_filled = df_all_prices_filled.merge(
    df_entreprises_secteur, on="code", how="left"
)


# ===============================
# Inclure les données nombres exploitations agricoles
# ===============================
df_all_prices_filled = df_all_prices_filled.merge(
    df_exploit_agricole, on="code", how="left"
)


df_all_prices_filled = (
    df_all_prices_filled[
        [
            "date",
            "année-mois",
            "code",
            "département",
            "region",
            "latitude",
            "longitude",
            "population",
            "surface",
            "tranche_population",
            "densité_population",
            "sap",
            "depot",
            "region du dépôt",
            "site",
            "indice_CNR",
            "USD_to_EUR",
            "EUR_to_USD",
            "cours_baril_en_USD",
            "tmoy",
            "nb_entreprise_ensemble",
            "nb_entreprise_industrie",
            "nb_entreprise_construction",
            "nb_entreprise_commerce",
            "nb_entreprise_info_communication",
            "nb_entreprise_autres_services",
            "nb_entreprise_sciences_techno",
            "nb_entreprise_immobilier",
            "nb_entreprise_finance_assurance",
            "nb_entreprise_santé_enseign_adminis",
            "nb_exploit_agricole",
            "prix",
        ]
    ]
    .sort_values(by=["date", "code"])
    .reset_index(drop=True)
)


display(df_all_prices_filled)

df_all_prices_filled.to_pickle("../data/processed/processed_data_FOD_07_10.pkl")
