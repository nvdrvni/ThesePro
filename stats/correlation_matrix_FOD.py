import pandas as pd
import matplotlib.pyplot as plt

# read data file
df = pd.read_pickle("../data/processed/processed_data_FOD.pkl")
df_num = df.select_dtypes(include=["int", "float"])
display(df_num.corr())
