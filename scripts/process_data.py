import pandas as pd
import os

# Dosya yolunu belirle
file_path = '../data/turkish_instructions.csv'

# CSV dosyasını okumaya çalış
try:
    # Dosyanın mevcut olup olmadığını kontrol et
    if not os.path.isfile(file_path):
        print(f"Hata: Dosya bulunamadı. Aranan yol: {file_path}")
    else:
        df = pd.read_csv(file_path)  # Dosya yolu
        print("CSV dosyası başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")

# Eğer dosya başarıyla yüklendiyse, sütun isimlerini kontrol et
if 'df' in locals():
    print("Sütun İsimleri:", df.columns)
