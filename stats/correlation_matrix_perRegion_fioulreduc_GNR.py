import pandas as pd
import matplotlib.pyplot as plt

# read data file
df = pd.read_pickle("../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl")

for region in df.region.unique():
    df_reg = df[df["region"] == region]
    df_num = df_reg.select_dtypes(include=["int", "float"])
    display(df_num.corr())

    df_num.corr().to_excel(
        f"../report/correlation matrix/Correlation Matrix FioulReduc_GNR_07_10_{region}.xlsx"
    )
