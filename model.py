# ---------------------- Supervised Fault Classification ----------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# ---------------------- Load Data ----------------------

vehicle_type = input("🔍 Enter vehicle type to analyze (EV / IC / ALL): ").strip().upper()

if vehicle_type == "EV":
    df = pd.read_csv("smartdiag_dynamic_writer_EV.csv")
    print("\n📊 Running analysis on EV data...")
elif vehicle_type == "IC":
    df = pd.read_csv("smartdiag_dynamic_writer_IC.csv")
    print("\n📊 Running analysis on IC data...")
elif vehicle_type == "ALL":
    df_ev = pd.read_csv("smartdiag_dynamic_writer_EV.csv")
    df_ic = pd.read_csv("smartdiag_dynamic_writer_IC.csv")
    df = pd.concat([df_ev, df_ic], ignore_index=True)
    print("\n📊 Running analysis on ALL vehicle data...")
else:
    print("❌ Invalid input. Exiting.")
    exit()

# ---------------------- Preprocessing ----------------------

df['Fault_Code'] = df['Fault_Label'].astype('category').cat.codes
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if 'Fault_Code' in numeric_cols:
    numeric_cols.remove('Fault_Code')

X = df[numeric_cols]
y = df['Fault_Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------- Suppress Warnings ----------------------
warnings.filterwarnings("ignore")

# ---------------------- Model 1: Random Forest ----------------------

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("🎯 Random Forest Report:")
print(classification_report(y_test, rf_preds, zero_division=0))
joblib.dump(rf_model, 'trained_rf_model.pkl')

# ---------------------- Model 2: Logistic Regression ----------------------

lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("🎯 Logistic Regression Report:")
print(classification_report(y_test, lr_preds, zero_division=0))

# ---------------------- Model 3: XGBoost Classifier ----------------------

xgb_model = XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("🎯 XGBoost Classifier Report:")
print(classification_report(y_test, xgb_preds, zero_division=0))

# ---------------------- Feature Importance ----------------------

importances = pd.Series(rf_model.feature_importances_, index=numeric_cols).sort_values(ascending=False)
print("\n🔥 Feature Importances:")
print(importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title(f"Feature Importance - {vehicle_type} Data")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ---------------------- Anomaly Detection ----------------------

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X)
df['IF_Score'] = iso_forest.decision_function(X)

top_anomalies = df.sort_values('IF_Score').head(10)
print("\n🚨 Top 10 Anomalous Records:")
print(top_anomalies[['Timestamp', 'IF_Score', 'Fault_Label', 'Battery_T', 'Coolant_T', 'Motor_Current']])

# ---------------------- Save Final Output ----------------------

df.to_csv("full_analyzed_output.csv", index=False)
