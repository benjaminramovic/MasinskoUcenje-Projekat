import pandas as pd 

import re
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
MODEL_NAME = "bert-base-uncased"

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from dataset_class import ReviewsDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


import numpy as np

import pandas as pd 

df2 = pd.read_csv('reviews.csv')


# Priprema podataka
df2["cleanliness"] = df2["cleanliness"].fillna(0).astype(int)
df2["location"] = df2["location"].fillna(0).astype(int)
df2["luxury"] = df2["luxury"].fillna(0).astype(int)
df2["family_friendly"] = df2["family_friendly"].fillna(0).astype(int)
print(df2.info())

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


df2["comments"] = df2["comments"].apply(clean_text)


labels = ['cleanliness', 'location', 'luxury', 'family_friendly']

# raspodela klasa
df2[labels].sum().plot(kind="bar")
plt.title("Raspodela pozitivnih oznaka po klasama")
plt.ylabel("Broj pozitivnih primera")
#plt.show()

counts = df2[labels].sum()

# heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df2[labels].corr(), annot=True, cmap="coolwarm")
plt.title("Korelacija između labela")
plt.show()


# vizuelizacija
# corr = df2[labels].corr()

# plt.figure(figsize=(6,5))
# sns.heatmap(corr, annot=True)
# plt.title("Korelacija oznaka")
# plt.show()

# plt.figure(figsize=(8,5))
# counts.plot(kind='bar')
# plt.title("Raspodela klasa")
# plt.ylabel("Broj pozitivnih primera")
# plt.show()


# df2['word_count'] = df2['comments'].astype(str).apply(
#     lambda x: len(x.split())
# )

# plt.figure(figsize=(8,5))
# plt.hist(df2['word_count'], bins=30)
# plt.title("Dužina recenzija")
# plt.xlabel("Broj reči")
# plt.ylabel("Broj recenzija")
# plt.show()


# print(df2)
# print(pd.options.display.max_rows)

# print(df2.head())
# print(df2.tail())

# print(df2.info())


## Brisemo redove sa praznim komentarom
# df2.dropna(inplace=True)

# print(df2.info())

# # Ciscenje
# new_df = df2[["comments"]]
# print(new_df) 

# reduced = new_df.sample(n=9000, random_state=45)
# print(reduced)

#  priprema 

df_train = df2.dropna(subset=[
    "cleanliness",
    "location",
    "luxury",
    "family_friendly"
])

train_df, temp_df = train_test_split(
    df_train,
    test_size=0.3,
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df, 
    test_size=1/3, 
    random_state=42
)


#  kreiranje labela
labels = train_df[
    ["cleanliness", "location", "luxury", "family_friendly"]
].values.tolist()

test_labels = test_df[
    ["cleanliness", "location", "luxury", "family_friendly"]
].values.tolist()

val_labels = val_df[
    ["cleanliness", "location", "luxury", "family_friendly"]
].values.tolist()


# tokenizacija 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    train_df["comments"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_df["comments"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_df["comments"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

train_dataset = ReviewsDataset(
    train_encodings,
    labels
)

test_dataset = ReviewsDataset(
    test_encodings,
    test_labels
)

val_dataset = ReviewsDataset(
    val_encodings,
    val_labels
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = 1 / (1 + np.exp(-logits))  # stable sigmoid
    preds = (probs > 0.5).astype(int)

    return {
        "precision_micro": precision_score(labels, preds, average="micro"),
        "recall_micro": recall_score(labels, preds, average="micro"),
        "f1_micro": f1_score(labels, preds, average="micro"),

        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "f1_macro": f1_score(labels, preds, average="macro"),

        "accuracy": accuracy_score(labels, preds)
        # "f1_micro": f1_score(labels, preds, average="micro"),
        # "f1_macro": f1_score(labels, preds, average="macro"),
        # "accuracy": accuracy_score(labels, preds)
    }


model = AutoModelForSequenceClassification.from_pretrained(
    #MODEL_NAME,
    "./results/checkpoint-5250",
    num_labels=4,
    problem_type="multi_label_classification"
)
# train

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

RUN_TRAINING = False

if RUN_TRAINING:
    trainer.train()

test_results = trainer.predict(test_dataset)

# print("TEST METRICS:")
# print(test_results.metrics)

print("\nTEST RESULTS:")
for k, v in test_results.metrics.items():
    print(f"{k}: {v:.4f}")

# testiranje

text = """
The apartment was very clean and close to the city center.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.sigmoid(outputs.logits)


print("Predictions:")
print(predictions)