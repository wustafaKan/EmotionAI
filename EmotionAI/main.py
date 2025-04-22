# -*- coding: utf-8 -*-

import torch.nn.functional as F
import pandas as pd
import torch
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator
from google.colab import drive

# 🚀 1️⃣ GOOGLE DRIVE'ı BAĞLAMA
drive.mount('/content/drive')

# 📂 CSV DOSYANIN YOLUNU BELİRLE (Drive'daki konumuna göre düzenle!)
csv_path = "/drive-path/yorumlar.csv"

# 📌 CSV dosyasını oku ve sütun isimlerini düzenle
df = pd.read_csv(csv_path)
df.columns = ["text", "label"]

print("\n📊 Veri kümesi (İlk 5 satır):")
print(df.head())

# 🚀 2️⃣ VERİ ARTIRMA (DATA AUGMENTATION) - İSTEĞE BAĞLI
ENABLE_DATA_AUGMENTATION = False  # Hızlı eğitim için False kullanabilirsiniz.

def augment_dataset(df, num_augments=3):
    translator = Translator()
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(df['text'], df['label']):
        augmented_texts.append(text)
        augmented_labels.append(label)

        for _ in range(num_augments):
            try:
                translated = translator.translate(text, src='tr', dest='en').text
                back_translated = translator.translate(translated, src='en', dest='tr').text
                if back_translated != text:
                    augmented_texts.append(back_translated)
                    augmented_labels.append(label)
            except Exception:
                continue

    return pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})

if ENABLE_DATA_AUGMENTATION:
    print("\n🔄 Veri artırma işlemi başladı... (Bu işlem uzun sürebilir)")
    df = augment_dataset(df)

# 🚀 3️⃣ VERİYİ EĞİTİM VE DOĞRULAMA KÜMELERİNE AYIRMA
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# 🚀 4️⃣ MODEL VE TOKENIZER TANIMLAMA
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "MUTSUZ", 1: "MUTLU"},
    label2id={"MUTSUZ": 0, "MUTLU": 1}
)

# 🚀 5️⃣ GPU AYARI (Eğer varsa GPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🚀 6️⃣ VERİLERİ TOKENİZE ETME
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
)

val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128,
)

# 🚀 7️⃣ DATASET SINIFI TANIMLAMA
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels.tolist())
val_dataset = SentimentDataset(val_encodings, val_labels.tolist())

# 🚀 8️⃣ EĞİTİM AYARLARI
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),  # GPU varsa FP16 kullanılır
    gradient_accumulation_steps=2
)

# 🚀 9️⃣ METRİKLERİ TANIMLAMA
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}

# 🚀 1️⃣0️⃣ TRAINER OLUŞTURMA
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 🚀 🔥 1️⃣1️⃣ MODELİ EĞİTME
print("\nEğitim başlıyor...")
trainer.train()

# 🚀 1️⃣2️⃣ MODELİ KAYDETME
# Modeli eğittikten sonra kaydetme
model_save_path = "/content/drive/My Drive/saved_model"
trainer.save_model(model_save_path)

# Tokenizer'ü de kaydediyoruz
tokenizer.save_pretrained(model_save_path)
print(f"\nModel ve tokenizer kaydedildi: {model_save_path}")

# Daha sonra tahmin yapmak için kaydedilen model ve tokenizer'ü yükleyin:
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
model.to(device)


# Test için kullanılacak cümleler
test_cases = [
    "Ürün kalitesi mükemmel.",
    "Hizmet harikaydı kesinlikle tekrar gelirim",
    "Bu kadar kötü bir deneyim beklemiyordum",
    "Yeni gelen güncellemeden sonra uygulama düzgün çalışmamaya başladı.",
    "Ürün bok gibiydi.",
    "10/10 ürün çok güzel",
    "Harika cok beyendim.",
    "Ürün bozuk geldi iade edeceğim.",
    "Bir daha alıcammm",
    "işimi gördü.",
    "Yunus önermişti aldım memnunum."
]

print("\nTest Sonuçları:")
for text in test_cases:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=-1)
    label = "MUTLU" if predicted_class.item() == 1 else "MUTSUZ"
    print(f"\nMetin: {text}")
    print(f"Tahmin: {label} (%{confidence.item() * 100:.1f})")
