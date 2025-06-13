import numpy as np
import pandas as pd

def get_recommendations_tf(input_vector, model, data_matrix, data_index, data_df, top_n=5):
    input_embed = model.predict(np.array([input_vector]), verbose=0)[0]
    data_embeds = model.predict(data_matrix, verbose=0)
    similarity_scores = np.dot(data_embeds, input_embed)
    top_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    hasil = []
    for idx in top_indices:
        nama = data_index[idx]
        hasil.append({
            'Nama Restoran': nama,
            'Skor Similaritas': similarity_scores[idx],
            'Alamat': data_df.iloc[idx]['Alamat']
        })
    return pd.DataFrame(hasil)
