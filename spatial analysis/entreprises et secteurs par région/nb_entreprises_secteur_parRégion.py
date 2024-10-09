import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("../../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")[
    [
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
        "region",
    ]
]

df_agg = df.groupby(["region"]).sum()


# Création du graphique en barres empilées
ax = df_agg.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="tab20")
plt.title("Nombre d'entreprises par secteur pour chaque région")
plt.xlabel("Région")
plt.ylabel("Nombre d'entreprises")
plt.legend(title="Secteurs", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(
    "../../report/entreprises et secteurs par région/entreprises_par_région.png",
    dpi=300,
)
plt.show()
