import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from foodspot.data_loader import load_data
from foodspot.feature_engineering import prepare_features

if __name__ == '__main__':
    data = load_data('data/restoran_data.csv')
    fitur_scaled, scaler, fitur = prepare_features(data)

    def build_model(input_dim, embedding_dim=64):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(embedding_dim, activation='relu'),  # encoding layer
            layers.Dense(input_dim)  # decode back to original dim
        ])
        return model


    model = build_model(fitur_scaled.shape[1])
    model.compile(optimizer='adam', loss='mse')
    model.fit(fitur_scaled, fitur_scaled, epochs=30, batch_size=16)

    model.save('model/tf_model.keras')
    joblib.dump(scaler, 'model/scaler.pkl')
    data.to_csv('model/clean_resto_data.csv', index=False)
    print("Model dan data berhasil disimpan ke folder 'model/'")
