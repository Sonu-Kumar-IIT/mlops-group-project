import pandas as pd
import json
import re
import os

# Load the SMS Spam dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# --- CLEANING STEPS ---
print("Original size:", len(df))

# 1. Remove duplicates
df = df.drop_duplicates()
print("After removing duplicates:", len(df))

# 2. Remove missing values
df = df.dropna()
print("After removing nulls:", len(df))

# 3. Lowercase and strip punctuation
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# 4. Encode labels
label2id = {"ham": 0, "spam": 1}
id2label = {0: "ham", 1: "spam"}
df['label_id'] = df['label'].map(label2id)

# Print class distribution
print("\nClass distribution:")
print(df['label'].value_counts())

# Save id2label mapping
with open('id2label.json', 'w') as f:
    json.dump(id2label, f)
print("\nSaved id2label.json")

# Save cleaned dataset
df.to_csv('cleaned_data.csv', index=False)
print("Saved cleaned_data.csv")
