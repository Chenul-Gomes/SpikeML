# SpikeML 🎯

A machine learning project that scrapes Valorant Champions Tour (VCT) pro match data and combines tabular ML with NLP sentiment analysis to predict match outcomes and explore whether community sentiment correlates with performance.

> 🚧 **Work in progress** — Phase 3 currently in development.

---

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection & Scraping | ✅ Complete |
| 2 | Feature Engineering & Tabular ML | ✅ Complete |
| 3 | NLP Sentiment Analysis | 🔄 In Progress |
| 4 | Analysis & Insights | ⏳ Upcoming |

---

## Key Findings
- Rolling average features (window=5) eliminate data leakage from same-match stats
- Map-specific win rates initially showed 87% accuracy — traced to data leakage where win rates were calculated using the current match result
- After fixing leakage with rolling historical win rates, honest accuracy sits at ~61%, representing genuine predictive power
- Random Forest outperformed Logistic Regression and XGBoost on this dataset with ~62% accuracy
- SHAP analysis shows kd diff and assists are the strongest predictors

---

## Data Sources
- **VLR.gg** — Pro match stats, player performance, map results
- **Reddit (r/ValorantCompetitive)** — Community sentiment (Phase 3)

---

## Tech Stack
`Python` `pandas` `scikit-learn` `XGBoost` `PyTorch` `HuggingFace Transformers` `BeautifulSoup` `SHAP`

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