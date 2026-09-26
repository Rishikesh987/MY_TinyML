# MY_TinyML: Learning Guide for Battery State of Charge (SoC) Regression Models

Welcome to **MY_TinyML**! This repository serves as an educational and practical guide for using classical machine learning regression models to predict a battery's **State of Charge (SoC %)** based on **Cell Voltage (V)**. 

Instead of deploying a heavy neural network framework, this project demonstrates how to compress algorithmic knowledge into simple, lightweight math formulas tailored for resource-constrained microcontrollers (e.g., Arduino, ESP32, STM32).

---

## 📂 Project Structure

```text
MY_TinyML/
├── .venv/                         # Local Python Virtual Environment
├── voltage_soc_linear.csv         # Baseline training data (Linear)
├── voltage_soc_exponential.csv    # Baseline training data (Curved Exponential)
├── voltage_soc_logarithmic.csv    # Baseline training data (Curved Logarithmic)
├── battery_regression_model.ipynb # Jupyter Notebook containing training code
└── battery_soc.h                  # Production-ready C/C++ deployment header
```

---

## 🧠 Educational Core: Understanding the Curve Profiles

Lithium-ion batteries do not discharge linearly. Their voltage characteristics bend heavily at the fully charged state (4.2V) and drop off steeply near the fully discharged state (<3.0V), leaving a relatively flat plateau in the middle. 

Choosing the right regression curve depends entirely on your battery dataset profile and your hardware processing boundaries.

### 📊 Direct Model Comparison Table

| Model Type | Governing Equation | Core Advantage | Main Drawback / Hazard | TinyML Hardware Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Linear** | \(SoC = m \cdot V + b\) | Extremely fast; requires almost no processing overhead. | Fails completely at the extreme edges (flat plates and steep drops). | **Lowest load.** Simple floating-point multiplication and addition. |
| **Exponential** | \(SoC = m \cdot e^V + b\) | Excellent for mapping rapid changes or chemical drop-offs at low voltages. | Can easily overshoot or explode toward infinity if the input voltage spikes. | **Moderate load.** Microcontroller must invoke `expf()`, which simulates power series expansion. |
| **Logarithmic** | \(SoC = m \cdot \ln(V) + b\) | Ideal for modeling rapid early charging or voltage saturation at the top end. | Passing zero or a negative voltage to \(\ln(x)\) causes a catastrophic hardware **NaN** fault. | **Moderate load.** Microcontroller must invoke `logf()`, requiring careful input safety boundaries. |

---

## 📈 Your Real-World Model Performance Data

Based on our training notebook, here are the real coefficients, baseline intercepts, and fitness scores calculated across the respective datasets:

### 【 1. LINEAR MODEL 】
* **Formula:** \(SoC = (62.6211 \times Voltage) - 163.0166\)
* **R-squared Score:** `0.9998 (100.0%)`
* **Learning Interpretation:** For this specific baseline dataset, a linear fit provides a near-perfect translation. However, it treats the voltage change as fixed across all boundaries.

### 【 2. EXPONENTIAL MODEL 】
* **Formula:** \(SoC = (1.7204 \times e^{Voltage}) - 33.2878\)
* **R-squared Score:** `0.9307 (93.1%)`
* **Learning Interpretation:** Captures accelerating voltage-to-capacity metrics smoothly, but sacrifices some fit on strict linear plateaus.

### 【 3. LOGARITHMIC MODEL 】
* **Formula:** \(SoC = (175.9327 \times \ln(Voltage)) - 142.9518\)
* **R-squared Score:** `0.8953 (89.5%)`
* **Learning Interpretation:** Perfectly scales diminishing capacity returns as voltage reaches its chemical peak saturation point.

---

## 💻 Microcontroller (C/C++) Deployment Header

Save this optimized, single-precision code block as `battery_soc.h` in your project folder. Note how we apply safety bounds to handle the specific operational vulnerabilities of each mathematical curve:

```cpp
#ifndef BATTERY_SOC_H
#define BATTERY_SOC_H

#include <math.h> // Provides hardware single-precision expf() and logf()

/**
 * LINEAR MODEL
 * Mathematical Execution: Pure addition and scalar multiplication.
 * R² Accuracy: 99.98%
 */
float get_soc_linear(float voltage) {
    float soc = (62.6211f * voltage) + (-163.0166f);
    
    // Negative-Marking Defense: Bound output explicitly between 0% and 100%
    if (soc > 100.0f) return 100.0f;
    if (soc < 0.0f)   return 0.0f;
    return soc;
}

/**
 * EXPONENTIAL MODEL
 * Mathematical Execution: Uses expf() to evaluate natural base e.
 * R² Accuracy: 93.07%
 */
float get_soc_exponential(float voltage) {
    float soc = (1.7204f * expf(voltage)) + (-33.2878f);
    
    if (soc > 100.0f) return 100.0f;
    if (soc < 0.0f)   return 0.0f;
    return soc;
}

/**
 * LOGARITHMIC MODEL
 * Mathematical Execution: Uses logf() to evaluate natural logarithm (ln).
 * R² Accuracy: 89.53%
 */
float get_soc_logarithmic(float voltage) {
    // CRITICAL CRASH PROTECTION: Rejects <= 0.0f inputs to shield the MCU 
    // from generating an undefined 'NaN' (Not a Number) system crash.
    if (voltage <= 0.0f) return 0.0f; 
    
    float soc = (175.9327f * logf(voltage)) + (-142.9518f);
    
    if (soc > 100.0f) return 100.0f;
    if (soc < 0.0f)   return 0.0f;
    return soc;
}

#endif // BATTERY_SOC_H
```

### 📋 Key TinyML Best Practices Demonstrated Here:
1. **FPU Enforcement:** The constants explicitly utilize the `f` suffix (e.g., `175.9327f`) paired with single-precision functions (`logf`, `expf`). This routes calculations straight to the hardware **Floating Point Unit (FPU)** found in chips like the ESP32, avoiding slow double-precision emulation.
2. **Zero Dynamic Allocation:** The operations require absolute baseline memory footprint layout. It executes inside local hardware registers without requiring a single byte of permanent **RAM heap allocation**.
