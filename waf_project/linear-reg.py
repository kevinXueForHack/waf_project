#1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats.mstats import winsorize
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score

#2. Load Data
data_file = "data/processed_loan_data.csv"
full_ld_df = pd.read_csv(data_file, low_memory=True)
print(full_ld_df.dtypes)

#3. Filter Data for Default Loans
default_df = full_ld_df.loc[~full_ld_df['LOAN_STATUS']]
default_df.reset_index(inplace=True, drop=True)
print(default_df.head())

#4. Compute VIF and Plot Correlation Matrix
# Prepare variables for VIF calculation
X = default_df[[
    'FEDFUNDS_RATE',
    'TERM',
    'EMPLOYEE_COUNT',
    'DISBURSEMENT_AMOUNT',
    'PERCENTAGE_EXPOSURE'
]].copy()
X['IS_URBAN'] = default_df['IS_URBAN'].astype(int)
X['IS_LOW_DOC'] = default_df['IS_LOW_DOC'].astype(int)

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Plot the correlation matrix heatmap
correlation_matrix = X.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Economic Variables")
plt.show()

#5. OLS Regression Analysis
# Prepare the independent variables (X) and dependent variable (y)
multi_analysis_df = default_df[[
    'FEDFUNDS_RATE',
    'TERM',
    'EMPLOYEE_COUNT',
    'DISBURSEMENT_AMOUNT',
    'PERCENTAGE_EXPOSURE'
]].copy()
multi_analysis_df['IS_URBAN'] = default_df['IS_URBAN'].astype(int)
multi_analysis_df['IS_LOW_DOC'] = default_df['IS_LOW_DOC'].astype(int)

# Target: Log-transformed CHARGE_OFF_AMOUNT with winsorization
y = default_df['CHARGE_OFF_AMOUNT']
y = np.log(y + 1)
y = winsorize(y, limits=[0.01, 0.01])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    multi_analysis_df, y, test_size=0.3, random_state=5
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add a constant term for OLS regression
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

# Fit the OLS model
model = sm.OLS(y_train, X_train_scaled).fit()

# Predict on the test set and evaluate
y_pred = model.predict(X_test_scaled)
mse = np.mean((y_test - y_pred) ** 2)
r2 = model.rsquared

print("OLS Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print(model.summary())

#6. Residual Analysis for the OLS Model
residuals = model.resid
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution (OLS Model)")
plt.show()

#7. XGBoost Regression Analysis
# Train the XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f"R-squared (R²): {r2_xgb:.2f}")

#8. XGBoost Model Visualizations and Residual Plots
# Plot Actual vs. Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.6, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (XGBoost)")
plt.legend()
plt.show()

# Plot Feature Importance
plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=10)
plt.title("Feature Importance - XGBoost")
plt.show()

# Residual Plots for the XGBoost Model
residuals_xgb = y_test - y_pred_xgb

# Scatter plot of residuals
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_xgb, residuals_xgb, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot: XGBoost")
plt.show()

# Histogram of residuals
plt.figure(figsize=(6, 4))
plt.hist(residuals_xgb, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals (XGBoost)")
plt.show()

# QQ Plot of residuals
plt.figure(figsize=(6, 4))
stats.probplot(residuals_xgb, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals (XGBoost)")
plt.show()
