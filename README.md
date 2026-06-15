# SpikeML 🎯

A machine learning project that scrapes Valorant Champions Tour (VCT) pro match data and combines tabular ML with NLP sentiment analysis to predict match outcomes and explore whether community sentiment correlates with performance.

> ⏸️ **Phase 3 on hold** — awaiting Reddit API developer access approval.

---

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection & Scraping | ✅ Complete |
| 2 | Feature Engineering & Tabular ML | ✅ Complete |
| 3 | NLP Sentiment Analysis | ⏸️ On Hold — pending Reddit API access |
| 4 | Analysis & Insights | ⏳ Upcoming |

---

## Key Findings
- Rolling average features (window=5) eliminate data leakage from same-match stats
- Map-specific win rates initially showed 87% accuracy — traced to data leakage where win rates were calculated using the current match result
- After fixing leakage with rolling historical win rates, honest accuracy sits at ~58–61%, representing genuine predictive power
- Agent role composition (Duelist/Initiator/Controller/Sentinel) added as features in Phase 2 — extracted from per-player agent picks
- Logistic Regression outperforms Random Forest and XGBoost on this dataset at scale
- SHAP analysis shows map win rates and momentum are the strongest predictors

---

## Dashboard
A Streamlit dashboard (`app.py`) allows users to:
- **Predict a match** — select two teams and get win probabilities from the trained model
- **Browse history** — explore past matches with scores and results

```bash
streamlit run app.py
```

---

## Data Sources
- **VLR.gg** — Pro match stats, player performance, map results
- **Reddit (r/ValorantCompetitive)** — Community sentiment (Phase 3, pending)

---

## Tech Stack
`Python` `pandas` `scikit-learn` `XGBoost` `BeautifulSoup` `SHAP` `Streamlit` `PyTorch` `HuggingFace Transformers`

---

## Setup

```bash
git clone https://github.com/Chenul-Gomes/SpikeML.git
cd SpikeML
pip install -r requirements.txt
python main.py
```

---

## Author
Chenul Gomes