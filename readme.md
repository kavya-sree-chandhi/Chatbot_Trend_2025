# 📈 Chatbot Usage Trend Analysis (2025)

This repository contains a **Supervised Learning** implementation to analyze and forecast chatbot usage growth. It demonstrates the fundamental "Pattern" of Machine Learning: transitioning from raw data to predictive insights using **Regression** and **Classification**.

## 🚀 The Pattern: Machine Learning Foundations
To crack roles in GenAI, understanding the underlying math of these two models is essential. 

### 1. Regression (Predictive Analysis)
- **Goal**: Predict the continuous value of `Usage_Volume_Millions`.
- **Hypothesis**: $h_\theta(x) = \theta_0 + \theta_1x$ (Linear Trend).
- **Loss Function**: **Mean Squared Error (MSE)** - measures the average squared distance between actual and predicted values.
- **Optimizer**: **Ordinary Least Squares** - minimizes the MSE to find the best-fit line.

### 2. Classification (Categorical Logic)
- **Goal**: Determine if a month belongs to a "High Usage" category (> 600M).
- **Hypothesis**: $h_\theta(x) = \sigma(\theta^T x)$ (Sigmoid Function).
- **Loss Function**: **Binary Cross-Entropy (Log Loss)** - measures the performance of a classification model where the output is a probability between 0 and 1.
- **Optimizer**: **L-BFGS** - a gradient-based optimization algorithm.

---

## 📂 Dataset
The analysis is based on `chatbot_usage_2025.csv`, which tracks monthly usage volume:
- **Month**: January - December 2025.
- **Usage_Volume_Millions**: The target variable for regression.
- **Month_Index**: (Feature Engineered) Numerical representation (1-12) used for training.

---

## 🛠️ How to Run
1. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
