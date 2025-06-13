import pandas as pd
from sklearn.preprocessing import StandardScaler
from foodspot.utils import custom_tokenizer

def prepare_features(data):
    data['combined_features'] = data['Jenis Makanan'] + ' ' + data['Jenis Makanan'] + ' ' + data['Kota']
    fitur = pd.get_dummies(data.set_index('Nama Restoran')[['Jenis Makanan', 'Rating', 'Harga']], prefix='Jenis')
    scaler = StandardScaler()
    fitur_scaled = scaler.fit_transform(fitur)
    return fitur_scaled, scaler, fitur