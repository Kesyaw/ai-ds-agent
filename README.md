<div align="center">

# ✦ AI Data Science Agent

**An intelligent ML pipeline agent that reasons through your data**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-a78bfa?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-ds-agent-kesya.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Kesyaw-f472b6?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Kesyaw/ai-ds-agent)
[![Python](https://img.shields.io/badge/Python-3.11-fb923c?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-34d399?style=for-the-badge)](https://langchain-ai.github.io/langgraph)

</div>

---

## 🧠 What Makes This Different From AutoML

Most AutoML tools just run all models and pick the best score.
This agent **reasons** about your data at every step:

| Condition | Agent Decision |
|---|---|
| Dataset imbalanced | Prioritize F1 over accuracy, apply class_weight |
| Dataset < 5000 rows | Avoid complex models, use LR + RF only |
| High cardinality column | Label Encoding instead of One-Hot |
| F1 score < 0.65 | Trigger optimization loop (RandomizedSearchCV) |
| Missing > 50% | Drop column entirely |

---

## ⚡ Agent Flow
Upload CSV
↓
Data Understanding     → EDA, outlier detection, imbalance check
↓
Model Selection        → Rule-based heuristic from data characteristics
↓
Preprocessing          → Adaptive: impute, encode, scale
↓
Train Multiple Models  → LR, RF, GBM, XGBoost
↓
Evaluate & Compare     → F1/Accuracy/R2 based on problem type
↓
[Score low?] ──────────→ Optimize (RandomizedSearchCV) → Retrain
↓
Generate Report        → LLM-powered analysis report

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph (stateful graph) |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Preprocessing | pandas, numpy, scikit-learn |
| Optimization | RandomizedSearchCV |
| Explainability | SHAP (feature importance) |
| API | FastAPI + Swagger UI |
| Frontend | Streamlit |
| Experiment Tracking | JSON-based logging |
| LLM Report | OpenAI GPT-3.5 (optional) |

---

## 🚀 Quick Start

### Run Locally

```bash
git clone https://github.com/Kesyaw/ai-ds-agent.git
cd ai-ds-agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run frontend/app.py
```

### Run API

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### API Usage

```bash
# 1. Train
curl -X POST "http://localhost:8000/train" \
  -F "file=@dataset.csv" \
  -F "target_column=Survived" \
  -F "problem_type=classification"

# Response: { "session_id": "abc123", "best_model": "random_forest", ... }

# 2. Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "data": [{"Pclass": 1, "Age": 25, "Fare": 50.0}]}'

# Response: { "predictions": [1], "probabilities": [[0.43, 0.57]] }
```

---

## 📁 Project Structure
ai-ds-agent/
├── agent/
│   ├── nodes/
│   │   ├── data_understanding.py  # EDA + outlier detection
│   │   ├── decision.py            # Model selection logic + routing
│   │   ├── preprocessing.py       # Adaptive preprocessing
│   │   ├── modeling.py            # Multi-model training
│   │   ├── evaluation.py          # Metrics + best model selection
│   │   └── optimization.py        # Hyperparameter tuning
│   ├── graph.py                   # LangGraph flow definition
│   └── state.py                   # Shared agent state schema
├── tools/
│   ├── report_tools.py            # HTML report generator
│   ├── shap_tools.py              # SHAP explainability
│   └── export_tools.py            # Model export (.pkl + .json)
├── api/
│   └── main.py                    # FastAPI endpoints
└── frontend/
└── app.py                     # Streamlit UI

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/train` | Upload CSV → run agent → return session_id |
| GET | `/status/{id}` | Get training results & feature columns |
| POST | `/predict` | Send data → get predictions + probabilities |
| DELETE | `/session/{id}` | Clear session from memory |

---

## 💡 Example Use Cases

```python
# Fraud Detection
# Agent detects imbalance → applies class_weight → prioritizes recall

# House Price Prediction  
# Agent switches to regression metrics → optimizes RMSE → reports R2

# Customer Churn
# Agent detects categorical features → uses tree-based models → explains top features via SHAP
```

---

## 📊 Sample Results (Titanic Dataset)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7989 | 0.7143 | 0.8394 |
| **Random Forest** ✦ | **0.8045** | **0.7244** | **0.8441** |
| Gradient Boosting | 0.7989 | 0.7143 | 0.8187 |

> Agent selected **Random Forest** — highest accuracy and ROC-AUC on balanced dataset

---

<div align="center">

Built by [Kesya Wangsa](https://github.com/Kesyaw) · Informatics Graduate

</div>
