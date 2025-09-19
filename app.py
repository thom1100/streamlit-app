import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data

def load_data(train_path, test_path):
    train = pd.read_parquet("/Users/thom/Desktop/DSB_year2/Streamlit_app/train.parquet")
    test = pd.read_parquet("/Users/thom/Desktop/DSB_year2/Streamlit_app/final_test.parquet")
    return train, test

def main():
    st.title("🛠️ Bike Counters - Dashboard")
    st.write("Exploration du projet *bike_counters_*")

    st.header("1. Données d'entraînement")
    train, test = load_data("data/train.parquet", "data/final_test.parquet")
    st.write("Aperçu des données d'entraînement :", train.head())
    st.write("Dimension des données:", train.shape)

    st.header("2. Distribution des compteurs de vélos")
    if 'bike_log_count' in train.columns:
        fig = px.histogram(train, x="bike_log_count", nbins=50, title="Distribution du nombre de vélos (log count)")
        st.plotly_chart(fig)
    else:
        st.write("La colonne **bike_log_count** n'a pas été trouvée dans le jeu d'entraînement.")

    st.header("3. Aperçu des prédictions / test final")
    st.write("Aperçu test final :", test.head())
    st.write("Dimension du test final:", test.shape)

    st.header("4. Quelques statistiques rapides")
    st.write(train.describe())

if __name__ == "__main__":
    main()
