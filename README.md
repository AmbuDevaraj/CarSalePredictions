# Used Car Price Prediction — Practical Application II

**Course:** UC Berkeley Professional Certificate in ML & AI  
**Framework:** CRISP-DM  
**Dataset:** 426,880 Craigslist used car listings  

---

## Project Overview

This project applies the CRISP-DM data science process to help used car dealerships understand **what factors drive used car prices**. Regression models were built and compared to predict vehicle resale value from attributes like age, mileage, manufacturer, model, fuel type, and title status.

---

## Summary of Findings

### Model Performance

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| Linear Regression (baseline) | $7,331 | 0.7251 |
| **Ridge Regression (alpha=1.0)** | **$7,331** | **0.7251** |
| Lasso Regression | $7,418 | 0.7180 |
| Ridge + Log-Transformed Target | $7,861 | 0.6839 |

**Ridge Regression** is the recommended model — it explains ~72.5% of price variance with an average prediction error of ~$7,331.

> **Note on log-transform:** Applying log(price) as the target did **not** improve R² on this dataset. The tighter price filter ($500–$100K) and `log_odometer` feature already address price skewness, making the log-transform counterproductive here.

---

### What Drove R² from 0.655 → 0.725

Each improvement was isolated and tested independently:

| Change | R² | Gain |
|--------|-----|------|
| Original baseline | 0.6553 | — |
| Tighter price ($500–$100K) & odometer (100–300K) filters | 0.6940 | **+0.0387** |
| + `log_odometer` and `age_sq` features | 0.7058 | **+0.0118** |
| + `model` column (top 50 models) | 0.7251 | **+0.0193** |

The biggest single gain came from **removing noisy outliers** (extreme prices and odometer readings). The `model` column and engineered features each added meaningful signal on top of that.

---

### Feature Engineering

| Feature | Description | Impact |
|---------|-------------|--------|
| `age` | `2026 - year` | Strong negative correlation with price (-0.59) |
| `age_sq` | Age squared | Captures non-linear depreciation curve |
| `log_odometer` | `log(odometer + 1)` | Linearises curved mileage-price relationship |
| `model` (top 50) | Specific model name, rare models → `'other'` | +0.019 R² gain |

> **Tested but not added:** `mileage_per_year` (correlation -0.33, weaker signal) and `age × odometer` interaction (correlation -0.57, highly collinear with existing features at 0.91).

---

### Key Price Drivers

**Factors that increase price:**
- Ferrari, Tesla, Porsche, Aston-Martin manufacturer badge
- 10- or 12-cylinder engines
- Diesel or electric fuel type
- 4WD/AWD drivetrain
- Low mileage (under 60K miles)
- Newer model year

**Factors that decrease price:**
- Salvage or rebuilt title status
- High mileage (150K+ miles)
- Older vehicle age
- Fair or poor condition rating

---

### Recommendations for Dealers

1. **Prioritize 0–5 year old vehicles** — newer cars command the highest resale margins
2. **Avoid high-mileage stock** — vehicles over 150K miles lose value sharply
3. **Clean titles are non-negotiable** — salvage/rebuilt titles reduce value by thousands
4. **Invest in refurbishment** — upgrading condition rating yields measurable price gains
5. **Stock diesel and EV inventory** — both command a significant market premium
6. **Focus on 4WD/AWD** — especially in regions with winter weather demand

---

### Model Limitations

- ~27.5% of price variance remains unexplained (local market conditions, trim levels, accident history)
- `model` column capped at top 50 — rare models bucketed as `'other'`; finer granularity could improve accuracy
- Tree-based models (Gradient Boosting) are the recommended next step for R² gains beyond ~0.73

---

## Notebook

[prompt_II.ipynb](./prompt_II.ipynb)

---

## Project Structure

```
CarSalePredictions/
├── README.md              # This file — summary of findings
├── prompt_II.ipynb        # Full analysis notebook
├── images/                # Supporting visuals
└── data/
    └── vehicles.csv       # Raw dataset (426K rows, not tracked in git)
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

# Launch notebook
jupyter notebook prompt_II.ipynb
```
