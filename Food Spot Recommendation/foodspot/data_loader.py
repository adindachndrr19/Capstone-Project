import pandas as pd
from foodspot.utils import clean_text, deteksi_kota

def load_data(path):
    data = pd.read_csv(path)
    data = data.dropna().drop_duplicates()
    data['Nama Restoran'] = data['Nama Restoran'].apply(clean_text)
    data['Kota'] = data['Alamat'].apply(deteksi_kota)
    return data