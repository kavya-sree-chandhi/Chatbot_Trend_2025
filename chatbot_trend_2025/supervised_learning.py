import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss

# ==========================================
# 1. DATA PREPARATION (The Foundation)
# ==========================================
# Load the dataset
df = pd.read_csv('chatbot_usage_2025.csv')

# Feature Engineering: Convert Months to Numbers (1-12)
df['Month_Index'] = np.arange(1, 13)

# Define 'High Usage' for Classification (Threshold: 600 Million)
df['Is_High_Usage'] = (df['Usage_Volume_Millions'] > 600).astype(int)

# Use .iloc to separate Features (X) and Targets (y)
X = df.iloc[:, 2:3].values  # Month_Index (2D for sklearn)
y_reg = df.iloc[:, 1].values # Usage_Volume (1D for Regression)
y_clf = df.iloc[:, 3].values # Is_High_Usage (1D for Classification)

print("--- DATASET PREVIEW ---")
print(df)

# ==========================================
# 2. REGRESSION (Predicting Quantity)
# ==========================================
# Hypothesis: h(x) = Wx + b (Linear)
# Loss: Mean Squared Error (MSE)
# Optimizer: Ordinary Least Squares / Gradient Descent
reg_model = LinearRegression()
reg_model.fit(X, y_reg)

# Predictions & Loss
y_reg_pred = reg_model.predict(X)
mse_loss = mean_squared_error(y_reg, y_reg_pred)

# Future Prediction (Jan 2026 = Month 13)
next_month_pred = reg_model.predict([[13]])

# ==========================================
# 3. CLASSIFICATION (Predicting Category)
# ==========================================
# Hypothesis: h(x) = Sigmoid(Wx + b)
# Loss: Binary Cross-Entropy (Log Loss)
# Optimizer: L-BFGS (Gradient-based)
clf_model = LogisticRegression()
clf_model.fit(X, y_clf)

# Probabilities & Loss
y_clf_prob = clf_model.predict_proba(X)[:, 1]
bce_loss = log_loss(y_clf, y_clf_prob)

# ==========================================
# 4. OUTPUTS & VISUALIZATION
# ==========================================
print("\n" + "="*40)
print("SUPERVISED LEARNING RESULTS")
print("="*40)
print(f"REGRESSION Metrics:")
print(f" - Hypothesis: Linear Trend Line")
print(f" - Loss (MSE): {mse_loss:.2f}")
print(f" - Jan 2026 Forecast: {next_month_pred[0]:.2f} Million")

print(f"\nCLASSIFICATION Metrics:")
print(f" - Hypothesis: Sigmoid Curve")
print(f" - Loss (Log Loss): {bce_loss:.4f}")
print("="*40)

# Generate Plot
plt.figure(figsize=(12, 5))

# Plot 1: Regression
plt.subplot(1, 2, 1)
plt.scatter(X, y_reg, color='blue', label='Actual Usage')
plt.plot(X, y_reg_pred, color='red', label='Regression Line')
plt.title('Regression: Usage Trend')
plt.xlabel('Month')
plt.ylabel('Millions')
plt.legend()

# Plot 2: Classification
plt.subplot(1, 2, 2)
plt.scatter(X, y_clf, color='green', label='Class (0=Low, 1=High)')
X_smooth = np.linspace(1, 13, 100).reshape(-1, 1)
y_smooth = clf_model.predict_proba(X_smooth)[:, 1]
plt.plot(X_smooth, y_smooth, color='orange', label='Sigmoid Probability')
plt.axhline(0.5, color='black', linestyle='--', label='Boundary (0.5)')
plt.title('Classification: High Usage Probability')
plt.xlabel('Month')
plt.legend()

plt.tight_layout()
plt.savefig('supervised_learning_output.png')
print("\nSuccess! Graph saved as 'supervised_learning_output.png'")
plt.show()