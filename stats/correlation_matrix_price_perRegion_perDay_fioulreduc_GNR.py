import pandas as pd
import matplotlib.pyplot as plt

# read data file
df = pd.read_pickle("../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")


# Effectuer les agrégations
aggregated_df = (
    df.groupby(["date", "region"])
    .agg(
        {
            "latitude": "mean",
            "longitude": "mean",
            "population": "sum",
            "surface": "sum",
            "tranche_population": lambda x: x.mode()[0] if not x.mode().empty else None,
            "densité_population": "mean",
            "sap": lambda x: x.mode()[0] if not x.mode().empty else None,
            "depot": lambda x: x.mode()[0] if not x.mode().empty else None,
            "region du dépôt": lambda x: x.mode()[0] if not x.mode().empty else None,
            "indice_CNR": "mean",
            "USD_to_EUR": "mean",
            "EUR_to_USD": "mean",
            "cours_baril_en_USD": "mean",
            "tmoy": "mean",
            "nb_entreprise_ensemble": "sum",
            "nb_entreprise_industrie": "sum",
            "nb_entreprise_construction": "sum",
            "nb_entreprise_commerce": "sum",
            "nb_entreprise_info_communication": "sum",
            "nb_entreprise_autres_services": "sum",
            "nb_entreprise_sciences_techno": "sum",
            "nb_entreprise_immobilier": "sum",
            "nb_entreprise_finance_assurance": "sum",
            "nb_entreprise_santé_enseign_adminis": "sum",
            "nb_exploit_agricole": "sum",
            "prix": "mean",
        }
    )
    .reset_index()
)

# aggregated_df.to_csv('fichier_aggregate.csv', index=False)


df_num = aggregated_df.select_dtypes(include=["int", "float"])
display(df_num.corr())

df_num.corr().to_excel(
    f"../report/correlation matrix/Correlation Matrix FioulReduc_GNR_07_10_moyenneparregion.xlsx"
)

df_num.corr().to_csv(
    f"../report/correlation matrix/Correlation Matrix FioulReduc_GNR_07_10_moyenneparregion.csv"
)
