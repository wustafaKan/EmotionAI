Languages
-
**Python**

Screenshots
-
![image](https://github.com/user-attachments/assets/da75acbb-06d1-4743-9f38-ddd744875374)

<br/>

![image](https://github.com/user-attachments/assets/d7dfbd55-e231-4bd4-b643-fa956f16edbb)

<br />


Emotion AI
-
Using Hugging Face's “dbmdz/bert-base-turkish-cased” ready-made model, I made an artificial intelligence that separates sentences as Happy or unhappy. First of all, I would like to say that I ran the code in “Google Colab”. If we examine the code from start to finish, we first add the necessary libraries to the project.
```
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

```

We connect your Drive account to “Google Colab”.

```
drive.mount('/content/drive')

```

Reading the file “yorumlar.csv”

```
csv_path = "/content/drive/My Drive/veriler/yorumlar.csv"
df = pd.read_csv(csv_path)
df.columns = ["text", "label"]
print(df.head())

```

I need to make a small note in this section, I couldn't find a suitable database for my project at the beginning. So I decided to create my own database. There is an online shopping site called “trendyol.com”, I wrote a code (trendyol.py) that will save the product reviews on this site in the form of happy sentences with 5 stars and unhappy sentences with 1 star in the “yorumlar.csv” file. This code also labels sentences as 1 if happy and 0 if unhappy. However, it can add some comments to the dataset more than once, so I created the clearData.py code for this. Actually, I could have added this feature in the trendyol.py file, but this made more sense while doing it.

<br />

The function translates each text into English and back-translates it back into Turkish (“back-translation”) to create unique new examples.

```
ENABLE_DATA_AUGMENTATION = False

def augment_dataset(df, num_augments=3):
    translator = Translator()
    augmented_texts = []
    augmented_labels = []
    …
    return pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})

if ENABLE_DATA_AUGMENTATION:
    df = augment_dataset(df)

```

We divide the data into 80% training and 20% testing.

```
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

```

We configure it to output two classes (happy/unhappy)

```
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "MUTSUZ", 1: "MUTLU"},
    label2id={"MUTSUZ": 0, "MUTLU": 1}
)
```

If Colab has a GPU, it will use it, if not, we set it to run on the CPU.

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

We specify the details of the training cycle: number of epochs, batch size, learning speed, weight regulation, recording and evaluation strategies, use of FP16 (semi-precision).

```
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
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2
)

```

Metric Definition and Trainer Creation

```
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

```

Training and Saving the Model

```
trainer.train()

model_save_path = "/content/drive/My Drive/saved_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

```

Test with Registered Model

```
# Kaydedilen model/tokenizer’ı yükleme
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
model.to(device)

# Test cümleleri
test_cases = [ … ]

for text in test_cases:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=-1)
    label = "MUTLU" if predicted_class.item() == 1 else "MUTSUZ"
    print(f"Metin: {text}")
    print(f"Tahmin: {label} (%{confidence.item()*100:.1f})")


```
