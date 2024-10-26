import pandas as pd

# CSV dosyasını okumaya çalış
try:
    df = pd.read_csv('../data/turkish_instructions.csv')  # Dosya yolu
    print("CSV dosyası başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")

# Eğer dosya başarıyla yüklendiyse, sütun isimlerini kontrol et
if 'df' in locals():
    print("Sütun İsimleri:", df.columns)
