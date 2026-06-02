# SpikeML 🎯

A machine learning project that scrapes Valorant Champions Tour (VCT) pro match data and combines tabular ML with NLP sentiment analysis to predict match outcomes and explore whether community sentiment correlates with performance.

> 🚧 **Work in progress** — Phase 2 currently in development.

---

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection & Scraping | ✅ Complete |
| 2 | Feature Engineering & Tabular ML | 🔄 In Progress |
| 3 | NLP Sentiment Analysis | ⏳ Upcoming |
| 4 | Analysis & Insights | ⏳ Upcoming |

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