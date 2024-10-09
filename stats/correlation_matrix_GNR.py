import pandas as pd
import matplotlib.pyplot as plt

# read data file
df = pd.read_pickle("../data/processed/processed_data_GNR_07_10.pkl")
df_num = df.select_dtypes(include=["int", "float"])
display(df_num.corr())

df_num.corr().to_excel("../report/correlation matrix/Correlation Matrix GNR_07_10.xlsx")

df_fioulreduc = pd.read_pickle(
    "../data/processed/processed_data_Fioulreduc_GNR_07_10.pkl"
)
df_num = df_fioulreduc.select_dtypes(include=["int", "float"])
df_num.corr().to_excel(
    "../report/correlation matrix/Correlation Matrix FioulReduc_GNR_07_10.xlsx"
)
