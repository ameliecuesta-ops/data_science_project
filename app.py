import streamlit as st
import pandas as pd

st.title("Detection RS vs RP")

st.write("Dashboard consommation électrique Enedis")

df = pd.read_csv("data/RES2-6-9kVA.csv")

st.subheader("Aperçu du dataset")

st.dataframe(df.head())
