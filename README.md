# 📰 FAKE NEWS DETECTION USING MACHINE LEARNING

> **Classify news articles as Real or Fake using TF-IDF + Logistic Regression / Decision Tree / Random Forest**

A machine learning project that detects fake news articles using natural language processing (TF-IDF vectorization) and three classification models, wrapped in an interactive Streamlit web app.

---

## 📑 TABLE OF CONTENTS

- [🚀 Live Demo](#-live-demo)
- [✨ Features](#-features)
- [🎯 Quick Start](#-quick-start)
- [📊 Dataset](#-dataset)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📈 Model Performance](#-model-performance)
- [📁 Project Structure](#-project-structure)
- [🛠️ Installation](#-installation)
- [🌐 Streamlit App Guide](#-streamlit-app-guide)
- [🔧 Troubleshooting](#-troubleshooting)
- [🚀 Future Enhancements](#-future-enhancements)

---

## 🚀 LIVE DEMO

### ⭐ Try the Application Now — No Installation Required!

#### **[🌐 OPEN LIVE APPLICATION →](https://fake-news-detection-using-machine-learning-xifavde3nriuaawtfbd.streamlit.app/)**

**Live Application Details:**
- 🔗 **URL**: https://fake-news-detection-using-machine-learning-xifavde3nriuaawtfbd.streamlit.app/
- ✅ **Status**: Live and fully functional
- 📱 **Devices**: Works on desktop, tablet, mobile
- 🆓 **Cost**: Completely free — no signup needed

**In the Live Demo, You Can:**
- 📊 **Dataset Explorer** — browse the built-in news dataset or upload your own CSV
- 📈 **Data Analysis** — label distribution, article count by subject, article length distribution
- 🔮 **Predictions** — paste any article text, pick a model, and classify it as real or fake
- ℹ️ **Model Info** — confusion matrix, classification report, and accuracy comparison across all 3 models

**Sample Prediction in 3 Steps:**
1. Go to the **🔮 Predictions** page
2. Paste in a news headline or article text
3. Choose a model and click **🚀 Predict** → get an instant result!

---

## ✨ FEATURES

| Feature | Description |
|---|---|
| 🔮 **Real/Fake Prediction** | Classifies any pasted article text |
| ⚖️ **Model Comparison** | Compare Logistic Regression, Decision Tree, and Random Forest side by side |
| 📊 **Dataset Explorer** | Upload and browse the news dataset |
| 📈 **Interactive EDA** | Label distribution, subject breakdown, article length analysis |
| ℹ️ **Model Info Dashboard** | Accuracy, confusion matrix, full classification report per model |
| 📱 **Responsive UI** | Works on desktop and mobile browsers |

---

## 🎯 QUICK START

### Fastest way to get started (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web app
streamlit run app.py

# 3. Open in browser
# http://localhost:8501
```

### Try the live demo (30 seconds)
🌐 **[Click here →](https://fake-news-detection-using-machine-learning-xifavde3nriuaawtfbd.streamlit.app/)** — no installation required.

---

## 📊 DATASET

Built from the classic **True.csv / Fake.csv** news dataset, merged and labeled.

| Property | Value |
|---|---|
| **Records in this app's dataset.csv** | 31,500 (sampled from 44,898 total) |
| **Columns** | `title`, `text`, `subject`, `date`, `class`, `label` |
| **Missing Values** | 0 |
| **Format** | CSV |
| **Note** | The `text` field in `dataset.csv` is truncated to ~680 characters per article to keep file size compact for distribution. The models were trained on the **full, untruncated** article text — truncation only affects the dataset preview/EDA pages, not the models themselves. |

**Target Variable:** `class` (0 = Fake, 1 = Real) / `label` (`Fake News` / `Not A Fake News`)

**Class Distribution (in dataset.csv):**
```
Fake News:          16,474 records (52.3%)
Not A Fake News:     15,026 records (47.7%)
```

---

## 🤖 MACHINE LEARNING MODELS

Three models are trained and compared, all on the same TF-IDF features:

| Model | Key Settings |
|---|---|
| **Logistic Regression** | `max_iter=1000` |
| **Decision Tree** | `max_depth=20`, `min_samples_leaf=10` |
| **Random Forest** | `n_estimators=40`, `max_depth=15`, `min_samples_leaf=10` |

**Text Preprocessing (`wordopt`):**
- Lowercase conversion
- Strip bracketed text, URLs, HTML tags
- Remove punctuation and digits
- Collapse whitespace

**Feature Extraction:** TF-IDF with a 3,000-term vocabulary (`min_df=3`, `max_df=0.9`)

**Why these settings?** Tree depth and vocabulary size are intentionally capped — this keeps the saved model files small (a few MB instead of tens of MB) with only a marginal effect on accuracy.

---

## 📈 MODEL PERFORMANCE

```
Training Set: 33,673 records
Testing Set:  11,225 records
```

| Model | Accuracy |
|---|---|
| **Decision Tree** | 99.44% |
| **Random Forest** | 99.14% |
| **Logistic Regression** | 98.45% |

**Key Insight**: All three models perform strongly on this dataset — the real vs. fake articles differ enough in vocabulary and style (source lines, formatting, phrasing) that even a shallow, size-constrained Decision Tree separates them almost perfectly.

---

## 📁 PROJECT STRUCTURE

```
Fake_News_Detection/
│
├── app.py                     Streamlit web application
│   ├─ Home (project overview, model comparison)
│   ├─ Dataset Explorer (upload & browse)
│   ├─ Data Analysis (EDA visualizations)
│   ├─ Predictions (real-time inference, all 3 models)
│   └─ Model Info (accuracy, confusion matrix, classification report)
│
├── train_model.py             Training script — retrain on True.csv/Fake.csv
├── requirements.txt           Python dependencies
├── fake_news_models.pkl       Dict of 3 trained models (LR, DT, RFC)
├── tfidf_vectorizer.pkl       Fitted TF-IDF vectorizer
├── model_metadata.pkl         Per-model accuracy, confusion matrix, report
├── dataset.csv                31,500-row sampled news dataset
├── Fake_News_Detection.ipynb  Full ML pipeline notebook
└── README.md                  This file
```

---

## 🛠️ INSTALLATION

**Requirements:** Python 3.8+

```bash
# 1. Make sure all project files are in one folder
# 2. Open a terminal in that folder
# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`. If it doesn't, copy that URL manually.

**Dependencies** (`requirements.txt`):
```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
streamlit==1.28.1
```

---

## 🌐 STREAMLIT APP GUIDE

| Page | What it does |
|---|---|
| 🏠 **Home** | Project overview, best model, side-by-side accuracy comparison |
| 📊 **Dataset Explorer** | Upload your own CSV or browse the default dataset; view stats and preview rows |
| 📈 **Data Analysis** | Label distribution, article count by subject, article length distribution |
| 🔮 **Predictions** | Paste article text, pick a model → instant prediction + comparison across all 3 models |
| ℹ️ **Model Info** | Confusion matrix, classification report, and accuracy chart per model |

**Retraining the model** (optional — needed if you want the full untruncated dataset or different hyperparameters):
```bash
python train_model.py
```
This requires `True.csv` and `Fake.csv` in the same folder, and regenerates `fake_news_models.pkl`, `tfidf_vectorizer.pkl`, and `model_metadata.pkl`.

---

## 🔧 TROUBLESHOOTING

**`streamlit: command not found`**
```bash
pip install streamlit==1.28.1
```

**Port 8501 already in use**
```bash
streamlit run app.py --server.port=8502
```

**`FileNotFoundError` for `.pkl` or `.csv` files**
Make sure `fake_news_models.pkl`, `tfidf_vectorizer.pkl`, `model_metadata.pkl`, and `dataset.csv` are all in the **same folder** as `app.py` — not in a subfolder.

**`pip: command not found`**
```bash
python -m pip install -r requirements.txt
```

---

## 🚀 FUTURE ENHANCEMENTS

- 📰 Retrain on the full, untruncated 44,898-row dataset for production use
- 🌍 Multi-language fake news detection
- 🧠 Upgrade to transformer-based models (e.g. BERT) for higher accuracy on nuanced cases
- 🔌 Browser extension for real-time fact-checking while reading
- 📊 Source credibility scoring alongside article-level predictions

---

## 🎯 QUICK LINKS

- 🌐 **[Live Demo App →](https://fake-news-detection-using-machine-learning-xifavde3nriuaawtfbd.streamlit.app/)**
- 📓 **[Jupyter Notebook](Fake_News_Detection.ipynb)**

---

**Happy Predicting!** 🚀
