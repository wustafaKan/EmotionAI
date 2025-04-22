import pandas as pd

# CSV dosya adı
csv_file = "yorumlar.csv"

# CSV'yi oku
df = pd.read_csv(csv_file, encoding="utf-8-sig")

# Tekrar eden yorumları kaldır (Yorum sütununa göre)
df_cleaned = df.drop_duplicates(subset=["Yorum"], keep="first")

# Etiket sayılarını hesapla
etiket_sayilari = df_cleaned["Etiket"].value_counts()

# 1 ve 0 sayısını al
mutlu_sayisi = etiket_sayilari.get(1, 0)  # Eğer 1 yoksa, 0 döndür
mutsuz_sayisi = etiket_sayilari.get(0, 0)  # Eğer 0 yoksa, 0 döndür

# Temizlenmiş veriyi tekrar CSV'ye yaz
df_cleaned.to_csv(csv_file, index=False, encoding="utf-8-sig")

# Sonuçları ekrana yazdır
print(f"Tekrar eden yorumlar silindi. Yeni dosya {csv_file} olarak kaydedildi.")
print(f"Mutlu (1) yorum sayısı: {mutlu_sayisi}")
print(f"Mutsuz (0) yorum sayısı: {mutsuz_sayisi}")
print(f"Toplam yorum sayısı: {mutlu_sayisi + mutsuz_sayisi}")
