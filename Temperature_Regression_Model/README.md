# Hourly Temperature Trend Predictor 🌡️

This sub-project implements a **Polynomial Regression Pipeline** using `scikit-learn` to analyze and forecast hourly temperature cycles over a 5-day span.

## 🧠 The Architecture: How the Model Works

While standard linear models predict along a flat, straight line, this project wraps `LinearRegression` inside a `PolynomialFeatures` pipeline. Under the hood, this configuration transforms our algorithm to behave like a **single artificial neuron (Perceptron)** with multi-channel inputs and linear processing.


### ⚙️ Component Breakdown

1. **The Inputs (Feature Engineering)**
   The pipeline reads a clean, single numerical hour input column (`date_hour`) from the CSV dataset. The `PolynomialFeatures(degree=3)` engine then expands this **1 raw input** dimension into **4 parallel input streams** (`1`, `x`, `x²`, `x³`) so the neuron can evaluate non-linear curved cycles.

2. **The Weights (`coef_`)**
   The neuron analyzed our dataset and mathematically calculated these optimized weights to determine feature importance:
   *   **w₀ (Bias Line):** `0.00`
   *   **w₁ (Linear Hour Trend):** `-0.93`
   *   **w₂ (Squared Curvature Scaling):** `0.20`
   *   **w₃ (Cubed Wave Adjustment):** `-0.01`

3. **The Bias Instruction (`intercept_`)**
   *   **Value:** `24.21`
   *   **Role:** This acts as the neuron's base threshold. It instructs the system: *"Start calculating from a foundational baseline temperature of 24.21°C (the predicted temperature at hour 0), then scale adjustments using the input weights."*

4. **The Activation Function**
   *   **Type:** **Linear Activation** (`_decision_function`)
   *   **Role:** Unlike deep learning neural networks which warp outputs using non-linear thresholds (like ReLU or Sigmoid), this regression pipeline utilizes a pure linear transfer strategy. It processes the raw dot-product matrix calculation and delivers the final numerical sum directly as the temperature output.


### 📊 The Final Mathematical Equation Running Inside Your Neuron

The model computes predictions by taking the raw feature inputs, multiplying them by their learned line weights, and adding the baseline offset:

$$ \text{Predicted Temperature} = \text{Bias} + (x_{1} \times w_{1}) + (x_{2} \times w_{2}) + (x_{3} \times w_{3}) $$

Plugging in our exact calculated metrics from our cleaned single-day dataset, the math formula converts to:

$$ \text{Predicted Temperature} = 24.21 + (\text{hour} \times -0.93) + (\text{hour}^{2} \times 0.20) + (\text{hour}^{3} \times -0.01) $$


<img width="1210" height="505" alt="image" src="https://github.com/user-attachments/assets/cb906341-9e62-4830-b94e-5d4baf2df293" />

---
