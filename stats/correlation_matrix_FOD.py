import pandas as pd
import matplotlib.pyplot as plt

# read data file
df = pd.read_pickle("../data/processed/processed_data_FOD_07_10.pkl")
df_num = df.select_dtypes(include=["int", "float"])
display(df_num.corr())

df_num.corr().to_excel("../report/correlation matrix/Correlation Matrix FOD_07_10.xlsx")
