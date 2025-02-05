# 1. Import Libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 2. Load Data
data_file = "data/processed_loan_data.csv"
df = pd.read_csv(data_file, low_memory=True)

# 3. Data Preprocessing for Extended Logistic Regression
features_ext = [
    "LOAN_AMOUNT",
    "PERCENTAGE_EXPOSURE",
    "IS_NEW",
    "IS_LOW_DOC",
    "UNRATE",
    "EMPLOYEE_COUNT",
    "GDP"
]
X_ext = df[features_ext].copy()
X_ext["IS_NEW"] = X_ext["IS_NEW"].astype(int)
X_ext["IS_LOW_DOC"] = X_ext["IS_LOW_DOC"].astype(int)
X_ext = sm.add_constant(X_ext)  # Adds the intercept term

y = df["LOAN_STATUS"]

# 4. Fit the Extended Logistic Regression Model
logit_model_ext = sm.Logit(y, X_ext)
result_ext = logit_model_ext.fit()
print(result_ext.summary())

df['predicted_prob'] = result_ext.predict(X_ext)

plt.figure(figsize=(8,6))
plt.hist(df[df['LOAN_STATUS'] == 1]['predicted_prob'], bins=50, alpha=0.5, label='LOAN_STATUS = 1')
plt.hist(df[df['LOAN_STATUS'] == 0]['predicted_prob'], bins=50, alpha=0.5, label='LOAN_STATUS = 0')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Distribution of Predicted Probabilities')
plt.legend(loc='upper center')
plt.show()

fpr, tpr, thresholds = roc_curve(y, df['predicted_prob'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()