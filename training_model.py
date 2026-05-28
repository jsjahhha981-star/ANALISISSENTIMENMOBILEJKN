# ============================================
# TRAINING MODEL SENTIMENT ANALYSIS
# SVM + TF-IDF
# ============================================

import pandas as pd
import joblib
import ast

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# LOAD DATASET CSV
# ============================================

data = pd.read_csv(
    "DATASETMOBILJKNSUDAHPREPROCESSINGDAN PELABELAN.csv",
    sep=';',
    encoding='utf-8',
    engine='python',
    on_bad_lines='skip'
)

print("Dataset berhasil dibaca!")

# ============================================
# BERSIHKAN NAMA KOLOM
# ============================================

data.columns = data.columns.str.strip()

# ============================================
# CEK NAMA KOLOM
# ============================================

print("\nNama Kolom:")
print(data.columns.tolist())

# ============================================
# HAPUS DATA KOSONG
# ============================================

data = data.dropna(subset=['Stemming', 'Label'])

# ============================================
# KONVERSI LIST STRING -> TEXT
# ============================================

def convert_list_to_string(text):

    try:

        if isinstance(text, str):

            text_list = ast.literal_eval(text)

            return ' '.join(text_list)

        else:
            return ''

    except:
        return ''

# Terapkan preprocessing
data['text_clean'] = data['Stemming'].apply(convert_list_to_string)

# ============================================
# FITUR DAN LABEL
# ============================================

X = data['text_clean']

y = data['Label']

# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# PIPELINE MODEL
# ============================================

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# ============================================
# TRAINING MODEL
# ============================================

model.fit(X_train, y_train)

print("\nModel berhasil ditraining!")

# ============================================
# EVALUASI MODEL
# ============================================

prediksi = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, prediksi))

print("\nClassification Report:")
print(classification_report(y_test, prediksi))

# ============================================
# TEST PREDIKSI MANUAL
# ============================================

contoh = [
    "aplikasi sangat bagus dan membantu",
    "aplikasi error dan lambat"
]

hasil = model.predict(contoh)

print("\nHasil Prediksi:")

for text, label in zip(contoh, hasil):

    print("Text :", text)
    print("Sentimen :", label)
    print()

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(model, 'model_sentimen.pkl')

print("Model berhasil disimpan!")