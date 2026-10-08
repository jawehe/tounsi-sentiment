import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Tounsi Sentiment", page_icon="🇹🇳")
st.title("🇹🇳 Analyse de sentiment en tounsi")
st.write("Écris un commentaire en arabizi / dialecte tunisien et découvre son sentiment.")

text = st.text_area("Ton texte :", placeholder="nhebbou barcha hedhi application")

if st.button("Analyser"):
    if not text.strip():
        st.warning("Écris un texte d'abord.")
    else:
        try:
            r = requests.post(API_URL, json={"text": text}, timeout=10)
            r.raise_for_status()
            data = r.json()
            sentiment = data["sentiment"]
            conf = data["confidence"]

            if sentiment == "Positif":
                st.success(f"😊 Positif ({conf:.0%})")
            elif sentiment == "Négatif":
                st.error(f"😠 Négatif ({conf:.0%})")
            else:
                st.info(f"🤔 Incertain ({conf:.0%}), le modèle n'est pas sûr")
            st.progress(conf)
        except requests.exceptions.RequestException:
            st.error("Impossible de joindre l'API. Est-ce que uvicorn tourne ?")