# AI in Cybersecurity: Social Media Disinformation & Threat Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow.svg)](https://scikit-learn.org/)
[![Domain](https://img.shields.io/badge/Domain-Cybersecurity%20%26%20OSINT-red.svg)]()

## 📌 Project Overview & "The Why"
In modern cyber warfare, information operations (IO), social engineering, and coordinated disinformation campaigns represent critical threats to public trust, democratic institutions, and organizational security. Malicious actors, state-sponsored troll farms, and automated bot networks deploy deceptive narratives engineered to bypass traditional security controls.

This project investigates automated disinformation detection using machine learning and deep learning methodologies applied to the **TruthSeeker benchmark dataset** (~134,000 samples). We evaluate two distinct threat detection paradigms:
1. **Behavioral & Metadata Analysis**: Can we detect fake news purely from account credibility, bot scores, and stylometric patterns without reading private message contents?
2. **Semantic & Temporal Analysis**: How do threat campaigns evolve over time, and how effectively do NLP and Deep Neural Networks discriminate disinformation directly from raw text feeds?

---

## 🗂️ Repository Structure

```text
├── main.ipynb                               # Notebook 1: Tabular ML (Random Forest) vs. Deep Learning (MLP)
├── timestamps_analysis.ipynb                # Notebook 2: Temporal Threat Analysis & NLP ML vs. Deep Text NN
├── main.py                                  # Modular dataset cleaning and preprocessing script
├── Features_For_Traditional_ML_Techniques.csv # Feature-engineered dataset (behavioral, bot, linguistic metrics)
├── cleaned_features_ml.csv                  # Cleaned tabular dataset ready for machine learning
├── Truth_Seeker_Model_Dataset_With_TimeStamps 1.xlsx # Raw text statements, tweets, and timestamps
├── Truth_Seeker_Model_Dataset_With_TimeStamps.csv   # Fast-loading CSV version of the timestamps dataset
└── README.md                                # Project documentation, threat intelligence insights & setup guide
```

---

## 📊 Datasets Information

### 1. Traditional ML Features (`Features_For_Traditional_ML_Techniques.csv`)
* **Size**: 134,198 rows × 64 columns
* **Target (`BinaryNumTarget`)**: Binary label where `1 = Real / Truth` and `0 = Fake / Disinformation` (balanced ~51.4% vs ~48.6%).
* **Feature Categories**:
  - **Account Credibility & Bot Signals**: `BotScoreBinary` (Bot vs Human), `cred` (credibility score), `normalize_influence`, `followers_count`, `friends_count`, `statuses_count`.
  - **Engagement Metrics**: `retweets`, `favourites`, `replies`, `quotes`.
  - **Stylometric & Affective Signals**: `Word_count`, `capitals`, `digits`, `exclamation` (`!`), `questions` (`?`), `dots` (`.`).
  - **Named Entity Percentages (SpaCy NER)**: `PERSON_percentage`, `ORG_percentage`, `GPE_percentage`, `MONEY_percentage`, `CARDINAL_percentage`.
  - **Part-of-Speech (POS) Tags**: `present_verbs`, `past_verbs`, `adjectives`, `adverbs`, `pronouns`, `determiners`.

### 2. Timestamps & Raw Text Dataset (`Truth_Seeker_Model_Dataset_With_TimeStamps.csv`)
* **Size**: 134,193 rows × 10 columns
* **Target (`BinaryNumTarget`)**: Ground truth truthfulness classification.
* **Fields**:
  - `statement`: The headline or fact-checked claim from benchmarks (PolitiFact, Snopes, etc.).
  - `tweet`: Raw social media post containing handles (`@user`), URLs, hashtags, emojis, and uncleaned conversational text.
  - `timestamp`: Temporal metadata (`Thu Sep 09 23:58:53 +0000 2021`) enabling time-series threat wave tracking.
  - `3_label_majority_answer` & `5_label_majority_answer`: Human annotator consensus labels (`Agree`, `Disagree`, `Mostly Agree`, `NO MAJORITY`).

---

## 🔬 Experimental Workflows & Notebooks

### 🧪 Notebook 1: `main.ipynb` (Tabular Threat Modeling)
* **Goal**: Evaluate privacy-preserving threat filtering using account and stylistic metadata alone.
* **Models Compared**:
  * **ML Algorithm**: **Random Forest Classifier** (`sklearn`) — 100 trees, parallel execution.
  * **DL Algorithm**: **Multi-Layer Perceptron (MLP)** (`PyTorch`) — 3-layer deep architecture with Batch Normalization and Dropout.
* **Key Results**:
  * **Random Forest**: ~68.5% Accuracy, F1-Score: 0.687, Train Time: ~1.7s.
  * **Deep MLP**: ~66.1% Accuracy, F1-Score: 0.621, Train Time: ~3.0s.
  * **Top Threat Predictors**: User credibility score (`cred`), follower counts, `BotScoreBinary`, and affective punctuation (`exclamation`).

### 🧪 Notebook 2: `timestamps_analysis.ipynb` (Temporal & Semantic Text Modeling)
* **Goal**: Analyze the temporal surge of cyber disinformation waves and classify threat content from raw text feeds.
* **Temporal Insights**:
  * Visualizes monthly attack surges and day-of-week posting volumes to trace coordinated botnet operations.
* **Text Preprocessing**:
  * Strips URLs, handles (`@`), hashtags (`#`), and special characters for natural language processing.
* **Models Compared**:
  * **NLP ML Algorithm**: **TF-IDF Vectorizer + Logistic Regression** (`sklearn`) — n-grams (1,2), 5,000 vocab.
  * **NLP DL Algorithm**: **Deep Text Neural Network** (`PyTorch`) — Dense layers (128 -> 64 -> 1), BatchNorm, Dropout (0.4 & 0.3).
* **Key Results**:
  * **TF-IDF + Logistic Regression**: **96.7% Accuracy**, F1-Score: 0.968, Train Time: ~0.8s.
  * **Deep Text Neural Net**: **97.8% Accuracy**, F1-Score: 0.978, Train Time: ~3.5s.

---

## 🛡️ Cybersecurity Analysis & Takeaways

| Threat Dimension | Behavioral Model (`main.ipynb`) | Semantic NLP Model (`timestamps_analysis.ipynb`) |
| :--- | :--- | :--- |
| **Data Utilized** | Account statistics, Bot scores, Punctuation | Raw tweet text content & n-gram vocabulary |
| **Accuracy Ceiling** | **~69%** | **~97.8%** |
| **Computational Footprint** | Extremely lightweight (microseconds per user) | Moderate (text tokenization & vectorization) |
| **Privacy Impact** | **High Privacy**: Inspects zero message content | **Lower Privacy**: Requires reading full messages |
| **SOC Architecture Role** | **Level-1 Pre-Filter**: Flags suspicious accounts and anomalous activity | **Level-2 Deep Verification**: Confirms semantic truthfulness of high-risk claims |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed with standard data science packages:
```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn openpyxl
```

### 2. Running the Code
* Run the cleaning script:
  ```bash
  python main.py
  ```
* Open either notebook in Jupyter or VS Code:
  ```bash
  jupyter notebook main.ipynb
  # or
  jupyter notebook timestamps_analysis.ipynb
  ```

---

## 👤 Author
* **Sai Nayan Mamilla**
* Subject: **AI in Cybersecurity**
