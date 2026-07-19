"""
Fake News Detection using Machine Learning
Streamlit web application: dataset explorer, EDA, live prediction, model comparison.
Mirrors the notebook pipeline: TF-IDF + Logistic Regression / Decision Tree / Random Forest.
"""

import re
import string
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def wordopt(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


@st.cache_resource
def load_model_and_metadata():
    with open('fake_news_models.pkl', 'rb') as f:
        models = pickle.load(f)   # dict: name -> fitted classifier
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return models, vectorizer, metadata


@st.cache_data
def load_default_dataset():
    return pd.read_csv('dataset.csv')


models, vectorizer, metadata = load_model_and_metadata()
LABELS = metadata['labels']  # {0: 'Fake News', 1: 'Not A Fake News'}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📰 Fake News AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dataset Explorer", "📈 Data Analysis", "🔮 Predictions", "ℹ️ Model Info"],
)

st.sidebar.markdown("---")
st.sidebar.caption("TF-IDF + Logistic Regression / Decision Tree / Random Forest")

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("📰 Fake News Detection using Machine Learning")
    st.markdown("### Classify news articles as real or fake using three ML models")

    best_model = max(metadata['results'], key=lambda k: metadata['results'][k]['accuracy'])
    best_acc = metadata['results'][best_model]['accuracy']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", best_model)
    col2.metric("Best Accuracy", f"{best_acc*100:.2f}%")
    col3.metric("Training Records", f"{metadata['n_train']:,}")
    col4.metric("TF-IDF Vocabulary", f"{metadata['n_features']:,}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What this app does")
        st.markdown("""
        - 📊 **Dataset Explorer** — browse the merged True/Fake news dataset
        - 📈 **Data Analysis** — class balance, article length, source breakdown
        - 🔮 **Predictions** — paste any article text and classify it with any of the 3 models
        - ℹ️ **Model Info** — accuracy, confusion matrix, classification report per model
        """)
    with c2:
        st.subheader("How it works")
        st.markdown("""
        ```
        Raw Article Text
              ↓
        Text Cleaning (lowercase, strip URLs/HTML/punctuation/digits)
              ↓
        TF-IDF Vectorization
              ↓
        Logistic Regression / Decision Tree / Random Forest
              ↓
        Real / Fake Prediction
        ```
        """)

    st.subheader("Model Comparison")
    comp_df = pd.DataFrame({
        'Model': list(metadata['results'].keys()),
        'Accuracy': [f"{v['accuracy']*100:.2f}%" for v in metadata['results'].values()],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.info("👈 Use the sidebar to explore the dataset, view analysis, or try a live prediction.")

# ---------------------------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------------------------
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")

    uploaded = st.file_uploader("Upload your own CSV (optional)", type="csv")
    df = pd.read_csv(uploaded) if uploaded is not None else load_default_dataset()

    st.success(f"Loaded dataset with **{df.shape[0]:,}** rows and **{df.shape[1]}** columns.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Preview")
    n_rows = st.slider("Rows to display", 5, 100, 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    if 'label' in df.columns:
        st.subheader("Label Distribution")
        st.bar_chart(df['label'].value_counts())

# ---------------------------------------------------------------------------
# DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
elif page == "📈 Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    df = load_default_dataset()

    tab1, tab2, tab3 = st.tabs(["Label Distribution", "By Subject", "Article Length"])

    with tab1:
        st.subheader("Real vs Fake Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = df['label'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{int(h)}', ha='center', va='bottom', fontweight='bold')
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Fake vs Real News")
        st.pyplot(fig)

    with tab2:
        st.subheader("Article Count by Subject")
        st.bar_chart(df['subject'].value_counts())

    with tab3:
        st.subheader("Article Length Distribution (words)")
        sample = df.sample(min(5000, len(df)), random_state=42).copy()
        sample['word_count'] = sample['text'].astype(str).str.split().apply(len)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=sample, x='word_count', hue='label', bins=40, kde=True, ax=ax)
        ax.set_xlim(0, sample['word_count'].quantile(0.98))
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------
elif page == "🔮 Predictions":
    st.title("🔮 Real-Time Fake News Prediction")
    st.markdown("Paste a news article's text below, choose a model, and click **Predict**.")

    text = st.text_area(
        "Article text",
        "WASHINGTON (Reuters) - The U.S. Senate voted on Tuesday to advance a bipartisan "
        "infrastructure bill after months of negotiations between lawmakers.",
        height=200,
    )
    model_choice = st.selectbox("Model", list(models.keys()))

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        cleaned = wordopt(text)
        vec = vectorizer.transform([cleaned])
        clf = models[model_choice]
        pred = clf.predict(vec)[0]
        label = LABELS[pred]

        st.markdown("---")
        if pred == 0:
            st.error(f"⚠️ **Predicted: {label}** (via {model_choice})")
        else:
            st.success(f"✅ **Predicted: {label}** (via {model_choice})")

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(vec)[0]
            prob_df = pd.DataFrame({
                'Label': [LABELS[0], LABELS[1]],
                'Probability': proba,
            }).set_index('Label')
            st.bar_chart(prob_df)

        st.subheader("Compare across all models")
        rows = []
        for name, m in models.items():
            p = m.predict(vec)[0]
            rows.append({'Model': name, 'Prediction': LABELS[p]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# MODEL INFO
# ---------------------------------------------------------------------------
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")

    model_choice = st.selectbox("Choose a model to inspect", list(metadata['results'].keys()))
    res = metadata['results'][model_choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model_choice)
    c2.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
    c3.metric("Test Records", f"{metadata['n_test']:,}")

    st.subheader("Confusion Matrix")
    cm = np.array(res['confusion_matrix'])
    label_names = [LABELS[0], LABELS[1]]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(res['classification_report']).T
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.subheader("All Models Compared")
    comp_df = pd.DataFrame({
        'Model': list(metadata['results'].keys()),
        'Accuracy': [v['accuracy'] for v in metadata['results'].values()],
    }).sort_values('Accuracy', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(comp_df['Model'], comp_df['Accuracy'] * 100, color="steelblue")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    for i, v in enumerate(comp_df['Accuracy'] * 100):
        ax.text(v + 0.5, i, f"{v:.2f}%", va='center')
    st.pyplot(fig)

    st.caption(f"Trained on {metadata['n_train']:,} records · Tested on {metadata['n_test']:,} records")
