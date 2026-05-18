import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('DATABASE FORAMINIFERA.csv', sep=';')

# rapihin kolom
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['spesies'] = df['spesies'].str.strip().str.replace(" ", "_")

# =========================
# IMAGE
# =========================
image_urls = {
    "Globigerina_bulloides": "https://www.mikrotax.org/images/pf_cenozoic/Globigerinidae/Globigerina/Globigerina%20bulloides/K_S%201983%2006-4.JPG",
    "Globorotalia_menardii": "https://www.mikrotax.org/images/pf_cenozoic/Globorotaliidae/Globorotalia/menardii%20lineage/Globorotalia%20menardii/K_S%201983%2029-1.JPG",
    "Orbulina_universa": "https://www.mikrotax.org/images/pf_cenozoic/Globigerinidae/Orbulina/Orbulina%20universa/K_S%201983%2020-4.JPG",
    "Nummulites_sp": "https://www.mindat.org/imagecache/85/ed/00960130017362090929908.jpg",
    "Operculina_sp": "https://images.marinespecies.org/thumbs/119138_operculina-complanata-defrance-in-de-blainville-1822.jpg"
}

df['image'] = df['spesies'].map(image_urls)

# =========================
# FITUR
# =========================
features = [
    'jumlah_kamar','ukuran','bentuk_cangkang','spiral',
    'tekstur','ostia','oskula','spongocoel','body_wall'
]

df_features = df[features]

# encoding
df_encoded = pd.get_dummies(df_features)

# normalisasi
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_encoded)

# =========================
# UI STREAMLIT
# =========================
st.title("Identifikasi Foraminifera")

jumlah_kamar = st.number_input("Jumlah kamar", min_value=0, key="jk")
ukuran = st.number_input("Ukuran (µm)", min_value=0, key="uk")

bentuk_cangkang = st.selectbox("Bentuk", ['bulat','oval','memanjang'], key="bc")
spiral = st.selectbox("Spiral", ['trochoid','planispiral','serial','spherical'], key="sp")
tekstur = st.selectbox("Tekstur", ['halus','kasar'], key="tx")
ostia = st.selectbox("Ostia", ['kecil','besar'], key="os")
oskula = st.selectbox("Oskula", ['ada','tidak_ada'], key="ok")
spongocoel = st.selectbox("Spongocoel", ['ada','tidak_ada'], key="spg")
body_wall = st.selectbox("Body wall", ['tipis','sedang','tebal','agregat'], key="bw")

# =========================
# BUTTON (CUMA 1!)
# =========================
if st.button("Proses", key="btn_proses"):

    input_user = {
        'jumlah_kamar': jumlah_kamar,
        'ukuran': ukuran,
        'bentuk_cangkang': bentuk_cangkang,
        'spiral': spiral,
        'tekstur': tekstur,
        'ostia': ostia,
        'oskula': oskula,
        'spongocoel': spongocoel,
        'body_wall': body_wall
    }

    df_input = pd.DataFrame([input_user])

    # encoding input
    df_input_encoded = pd.get_dummies(df_input)
    df_input_encoded = df_input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

    # scaling
    df_input_scaled = scaler.transform(df_input_encoded)

    # similarity
    similarity = cosine_similarity(df_input_scaled, df_scaled)[0]
    df['similarity'] = similarity * 100

    # ambil top 5
    top5 = df.sort_values('similarity', ascending=False).head(5)

    st.subheader("Hasil Identifikasi")

    for i, row in top5.iterrows():
        st.write(f"**Spesies:** {row['spesies']}")
        st.write(f"Similarity: {row['similarity']:.2f}%")

        if isinstance(row['image'], str):
            st.image(row['image'], width=200)
        else:
            st.write("Gambar tidak ada")

        st.write("---")
