"""
Train and save the Fake News Detection models.
Mirrors the notebook pipeline: merge True.csv + Fake.csv, clean text,
TF-IDF vectorize, train Logistic Regression / Decision Tree / Random Forest.

Hyperparameters are deliberately constrained (limited TF-IDF vocabulary,
capped tree depth/leaf size, fewer Random Forest trees) so the saved
model files stay compact, while keeping accuracy close to the
unconstrained notebook version.

Run this to regenerate:
  fake_news_models.pkl   (dict of the 3 trained models)
  tfidf_vectorizer.pkl
  model_metadata.pkl
"""

import re
import string
import pickle
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING (compact)")
print("=" * 60)

# 1. Load & label datasets
print("\n1. Loading datasets...")
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')
print(f"   Fake: {df_fake.shape}   True: {df_true.shape}")

df_fake["class"] = 0
df_true["class"] = 1

# 2. Merge
print("\n2. Merging datasets...")
df = pd.concat([df_fake, df_true], axis=0)
df = df.drop(["title", "subject", "date"], axis=1)
print(f"   Merged shape: {df.shape}")

# 3. Shuffle
print("\n3. Shuffling...")
df = df.sample(frac=1, random_state=42)
df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)

# 4. Text cleaning (same as notebook's wordopt)
print("\n4. Cleaning text...")


def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


df["text"] = df["text"].apply(wordopt)
print("   Done.")

# 5. Features / target
x = df["text"]
y = df["class"]

# 6. Train/test split (75/25, as in the notebook)
print("\n5. Splitting dataset (75/25)...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print(f"   Train: {x_train.shape[0]}   Test: {x_test.shape[0]}")

# 7. TF-IDF vectorization (vocabulary capped to keep the vectorizer + models small)
print("\n6. Vectorizing text (TF-IDF, max_features=3000)...")
vectorization = TfidfVectorizer(max_features=3000, min_df=3, max_df=0.9)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print(f"   Vocabulary size: {len(vectorization.vocabulary_):,}")

# 8. Train the three models (depth/leaf-capped so the trees stay compact)
results = {}
models = {}

print("\n7. Training Logistic Regression...")
LR = LogisticRegression(max_iter=1000)
LR.fit(xv_train, y_train)
models['Logistic Regression'] = LR

print("\n8. Training Decision Tree (max_depth=20, min_samples_leaf=10)...")
DT = DecisionTreeClassifier(max_depth=20, min_samples_leaf=10, random_state=0)
DT.fit(xv_train, y_train)
models['Decision Tree'] = DT

print("\n9. Training Random Forest (n_estimators=40, max_depth=15, min_samples_leaf=10)...")
RFC = RandomForestClassifier(n_estimators=40, max_depth=15, min_samples_leaf=10, random_state=0)
RFC.fit(xv_train, y_train)
models['Random Forest'] = RFC

# 9. Evaluate all three
print("\n10. Evaluating models...")
for name, clf in models.items():
    pred = clf.predict(xv_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    results[name] = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }
    print(f"   {name}: accuracy = {acc:.4f}")

# 10. Save artifacts
print("\n11. Saving artifacts...")
with open('fake_news_models.pkl', 'wb') as f:
    pickle.dump(models, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorization, f)

metadata = {
    'results': results,
    'labels': {0: 'Fake News', 1: 'Not A Fake News'},
    'n_train': int(x_train.shape[0]),
    'n_test': int(x_test.shape[0]),
    'n_features': len(vectorization.vocabulary_),
    'class_counts': df['class'].value_counts().to_dict(),
}
with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("    Saved fake_news_models.pkl, tfidf_vectorizer.pkl, model_metadata.pkl")
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE — run: streamlit run app.py")
print("=" * 60)
