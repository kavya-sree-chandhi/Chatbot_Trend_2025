# 📈 Chatbot Usage Trend Analysis (2025)

This repository contains a **Supervised Learning** implementation to analyze and forecast chatbot usage growth. It demonstrates the fundamental "Pattern" of Machine Learning: transitioning from raw data to predictive insights using **Regression** and **Classification**.

---

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
1. **Execute the Script:**:
   ```bash
   python supervised_learning.py

## 📊 Visual Outputs

1. **Model Visualization (Graph)**:  
   The following plot shows the Linear Regression trend line versus the Logistic Regression sigmoid curve.
   <img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/71d99dab-a086-407a-98ef-ba7ff5e52c84" />


3. **Execution Output (Terminal Screenshot)**:  
   This screenshot shows the successful execution of the script, including the data preview and calculated losses.
   <img width="1847" height="874" alt="image" src="https://github.com/user-attachments/assets/01f8e859-3c92-4974-a3e9-3651f4373718" />

---

## 🧠 Why this matters for GenAI

In Generative AI, these foundations are used for:
- **Resource Forecasting**: Using Regression to predict GPU needs.
- **Probability Logic**: Classification is the core of how LLMs select the most likely "next token."
- **Guardrails**: Detecting and blocking toxic prompts via classification.

---

## 🏁 Future Work
- Implement Unsupervised Learning (K-Means) to cluster user behavior patterns.
- Transition from scikit-learn to PyTorch for deeper neural network implementation.
