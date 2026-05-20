# Sample Datasets

Ready-to-use datasets for testing the AI Data Science Agent.

| File | Type | Target | Rows | Description |
|---|---|---|---|---|
| `titanic_classification.csv` | Classification | `Survived` | 891 | Predict passenger survival — has missing values & categorical features |
| `house_prices_regression.csv` | Regression | `price` | 500 | Predict house prices — mixed numeric & categorical |
| `customer_churn_classification.csv` | Classification | `churn` | 800 | Predict customer churn — balanced classes |

## How to Use

1. Go to [Live Demo](https://ai-ds-agent-kesya.streamlit.app)
2. Upload any CSV from this folder
3. Select the target column (see table above)
4. Select problem type
5. Click **Run Agent**

## Via API

```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@samples/titanic_classification.csv" \
  -F "target_column=Survived" \
  -F "problem_type=classification"
```
