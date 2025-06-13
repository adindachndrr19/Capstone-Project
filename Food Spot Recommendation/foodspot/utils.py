import re

def custom_tokenizer(text):
    return text.split(',')

def deteksi_kota(alamat):
    daftar_kota = [
        'Jakarta Barat', 'Jakarta Timur',
        'Jakarta Pusat', 'Jakarta Selatan',
        'Jakarta Utara'
    ]
    for kota in daftar_kota:
        if kota.lower() in str(alamat).lower():
            return kota
    return 'Lainnya'

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    return text