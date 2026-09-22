# Hourly Temperature Trend Predictor 🌡️

Polynomial regression model for forecasting hourly temperature cycles from 5 days of sensor data. Built with scikit-learn, designed to prototype embedded forecasting before deployment on the MCXN947.

---

## Architecture

This project chains two stages:

1. **Feature Engineering** (`PolynomialFeatures(degree=3)`)
   - Raw input: single numerical hour value (0–23)
   - Output: 4 parallel feature streams: `[1, hour, hour², hour³]`
   - Purpose: Capture non-linear daily temperature rhythms without hand-crafted features

2. **Regression Model** (`LinearRegression`)
   - Learns optimal weights for each polynomial term
   - Fits using ordinary least squares (OLS) on 120 hourly samples
   - Output: continuous temperature prediction

---

## Model Equation

The trained model computes predictions as:

$$\text{T}_{\text{pred}} = 23.64 + (-0.53 \cdot h) + (0.15 \cdot h^2) + (-0.01 \cdot h^3)$$

Or in general form:

$$\text{T}_{\text{pred}} = b_0 + (w_1 \cdot h) + (w_2 \cdot h^2) + (w_3 \cdot h^3)$$

Where:
- $h$ = hour of day (0–23)
- $b_0$ = 23.64 (baseline temperature at hour 0)
- $w_1$ = -0.53 (linear cooling trend)
- $w_2$ = 0.15 (quadratic curvature)
- $w_3$ = -0.01 (cubic adjustment)

---

## Usage

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load training data
df = pd.read_csv('hourly_temperature.csv')

# Build and train pipeline
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(df[['date_hour']], df['temperature'])

# Predict temperature for any hour
predicted_temp = model.predict([[15]])  # Hour 15 (3 PM)
print(f"Predicted temperature at 15:00 — {predicted_temp[0]:.1f}°C")
```

---

## Results

**Dataset:**
- 120 samples (5 consecutive days, hourly readings)
- Range: 22.8°C to 28.7°C
- Feature: hour of day (0–23)

**Model Performance:**
- Degree: 3 (cubic polynomial)
- Fitting method: Ordinary Least Squares (OLS)
- Training samples: 120

**Learned Parameters:**
```
Intercept (b₀):           23.64
Linear coeff (w₁):        -0.53
Quadratic coeff (w₂):      0.15
Cubic coeff (w₃):         -0.01
```

**Validation Score:**
```
R² = 0.9547 (95.5%)
```
*95.5% of variance in hourly temperature is explained by the cubic polynomial fit.*

---

## Why Polynomial Regression?

Daily temperature cycles are **smooth, continuous, and non-linear**—they warm through the day and cool at night. A cubic polynomial can approximate this shape with minimal parameters, making it ideal for:

- **Embedded deployment**: Small memory footprint (4 float coefficients)
- **Fast inference**: Single matrix multiply + add (no iteration)
- **Interpretability**: Weights directly show feature importance
- **Prototype-to-production**: Quick validation before TensorFlow Lite export

---

<img width="1210" height="505" alt="image" src="https://github.com/user-attachments/assets/cb906341-9e62-4830-b94e-5d4baf2df293" />

---
