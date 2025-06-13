import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from foodspot.recommendation import get_recommendations_tf
from foodspot.feature_engineering import prepare_features

if __name__ == '__main__':
    data = pd.read_csv('model/clean_resto_data.csv')
    model = tf.keras.models.load_model('model/tf_model.keras')
    scaler = joblib.load('model/scaler.pkl')

    data.set_index('Nama Restoran', inplace=True)
    fitur_scaled, _, fitur = prepare_features(data.reset_index())

    query = input("Masukkan nama restoran / jenis makanan / lokasi: ").strip().lower()

    nama_cocok = [nama for nama in data.index if query in nama.lower()]
    jenis_cocok = data[data['Jenis Makanan'].str.lower().str.contains(query)]
    lokasi_cocok = data[data['Kota'].str.lower().str.contains(query)]

    if nama_cocok:
        input_vector = scaler.transform(fitur.loc[[nama_cocok[0]]])
        rekom = get_recommendations_tf(input_vector[0], model, fitur_scaled, fitur.index, data, top_n=5)
        print(rekom)
    elif not jenis_cocok.empty:
        print(f"\nMenampilkan restoran dengan jenis makanan: {query.title()}")
        print(jenis_cocok[['Rating', 'Harga', 'Alamat']].sort_values(by='Rating', ascending=False).head(10))
    elif not lokasi_cocok.empty:
        print(f"\nMenampilkan restoran di lokasi: {query.title()}")
        print(lokasi_cocok[['Rating', 'Jenis Makanan', 'Alamat']].sort_values(by='Rating', ascending=False).head(10))
    else:
        print("Tidak ditemukan hasil yang sesuai.")