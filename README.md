# 🤖 AI Data Science Agent

An intelligent ML pipeline agent built with LangGraph that automatically analyzes datasets, selects models based on data characteristics, trains and evaluates multiple models, and generates reports — with decision-making at every step.

**Live Demo**: https://ai-ds-agent-kesya.streamlit.app

---

## 🧠 What Makes This Different From AutoML

Most AutoML tools just run all models and pick the best score. This agent **reasons** about your data:

- Detects imbalance → adjusts metric and applies class_weight
- Checks dataset size → avoids overly complex models on small data
- Detects high cardinality → uses Label Encoding instead of One-Hot
- Evaluates results → decides whether to iterate or finalize
- Explains every decision in plain language

---

## ⚙️ Agent Flow
Upload Dataset
↓
Data Understanding (EDA + outlier detection)
↓
Select Candidate Models (rule-based heuristic)
↓
Preprocessing (adaptive: impute, encode, scale)
↓
Train Multiple Models
↓
Evaluate & Compare
↓
[If score low] → Optimize (RandomizedSearchCV) → Retrain
↓
Generate Report

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Preprocessing | pandas, numpy |
| Optimization | RandomizedSearchCV |
| Explainability | SHAP |
| Frontend | Streamlit |
| LLM (optional) | OpenAI GPT-3.5 |

---

## 🚀 Run Locally

```bash
git clone https://github.com/Kesyaw/ai-ds-agent.git
cd ai-ds-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run frontend/app.py
```

---

## 📁 Project Structure
ai-ds-agent/
├── agent/
│   ├── nodes/
│   │   ├── data_understanding.py  # EDA + outlier detection
│   │   ├── decision.py            # Model selection logic
│   │   ├── preprocessing.py       # Adaptive preprocessing
│   │   ├── modeling.py            # Multi-model training
│   │   ├── evaluation.py          # Metrics + best model selection
│   │   └── optimization.py        # Hyperparameter tuning
│   ├── graph.py                   # LangGraph flow
│   └── state.py                   # Shared agent state
├── tools/
│   └── report_tools.py            # LLM report generator
└── frontend/
└── app.py                     # Streamlit UI

---

## 💡 Example Use Cases

- Upload `fraud.csv` → agent detects imbalance → applies SMOTE logic → prioritizes recall
- Upload `house_prices.csv` → agent switches to regression metrics → optimizes RMSE
- Upload any CSV → agent handles missing values, encoding, scaling automatically