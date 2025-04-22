import time
import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import signal
import sys

# ChromeDriver yolu (kendi yolunu eklemelisin)
CHROMEDRIVER_PATH = "chromedriver.exe"

# Trendyol yorum sayfası
URL = "https://www.trendyol.com/fantom/pratic-s-p1200-torbasiz-dikey-supurge-antrasit-p-35509789/yorumlar"

# Başlangıçta yorumları saklamak için bir liste
comments_data = []

# Kullanıcıdan filtre seçimi al
user_input = input("Mutlu yorumları çekmek için 1, mutsuz yorumları çekmek için 0 girin: ")

# Geçerli giriş yapıldı mı kontrol et
if user_input not in ["0", "1"]:
    print("Geçersiz giriş. 0 veya 1 girmeniz gerekiyor.")
    sys.exit()

# Seçim türünü belirle
comment_label = int(user_input)

# Selenium başlat
options = Options()
options.add_argument("--start-maximized")  # Tarayıcıyı tam ekran aç
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# Sayfayı aç
driver.get(URL)
time.sleep(5)  # Sayfanın yüklenmesini bekle

try:
    # Filtreleme butonuna bas
    filter_button = driver.find_element(By.XPATH, '//*[@id="rating-and-review-app"]/div/div/div/div[1]/div[2]/div[3]/div[2]/div/button')
    filter_button.click()
    time.sleep(2)  # Filtrelerin açılmasını bekle

    # Filtre seçenekleri içeren div'i bul
    filter_container = driver.find_element(By.CLASS_NAME, "items-container")
    checkboxes = filter_container.find_elements(By.CLASS_NAME, "checkbox")

    # Kullanıcının seçimine göre mutlu/mutsuz filtreyi uygula
    if comment_label == 1:
        checkboxes[0].click()  # Mutlu yorumlar
    else:
        checkboxes[4].click()  # Mutsuz yorumlar

    time.sleep(2)  # Filtrenin uygulanmasını bekle

    # "Uygula" butonuna bas
    apply_button = driver.find_element(By.CLASS_NAME, "btn.btn-apply")
    apply_button.click()
    time.sleep(3)  # Filtrenin yüklenmesini bekle

    # Yorumları kaydeden fonksiyon
    def save_comments():
        """Çekilen yorumları mevcut CSV dosyasına ekler."""
        df = pd.DataFrame(comments_data, columns=["Yorum", "Etiket"])
        
        # Eğer dosya zaten varsa, yeni veriyi ekle
        try:
            existing_df = pd.read_csv("yorumlar.csv", encoding="utf-8-sig")
            df = pd.concat([existing_df, df], ignore_index=True)  # Eski ve yeni veriyi birleştir
        except FileNotFoundError:
            pass  # Dosya yoksa direkt yeni veriyi kaydet
        
        df.to_csv("yorumlar.csv", index=False, encoding="utf-8-sig")
        print("Yorumlar kaydedildi.")


    # Ctrl+C sinyalini yakala
    def signal_handler(sig, frame):
        print("\nCTRL+C algılandı, yorumlar kaydediliyor...")
        save_comments()
        driver.quit()
        sys.exit(0)

    # CTRL+C'yi dinle
    signal.signal(signal.SIGINT, signal_handler)

    # Yorumları çekme işlemi
    while True:
        # "reviews" div'ini bul
        reviews_container = driver.find_element(By.CLASS_NAME, "reviews")
        comments = reviews_container.find_elements(By.CLASS_NAME, "comment")

        for comment in comments:
            try:
                text = comment.find_element(By.CLASS_NAME, "comment-text").find_element(By.TAG_NAME, "p").text.strip()
                comments_data.append((text, comment_label))
            except:
                pass  # Eğer yorum okunamazsa hata almamak için geç

        print(f"{len(comments_data)} yorum çekildi...")

        # Sayfayı aşağı kaydır ve yeni yorumları yükle
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        time.sleep(2)  # Yorumların yüklenmesini bekle

except Exception as e:
    print(f"Hata oluştu: {e}")
    save_comments()
    driver.quit()
