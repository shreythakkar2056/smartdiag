# ---------------------- Supervised Fault Classification ----------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss, multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import joblib
import os
import numpy as np

# ---------------------- Mode Selection ----------------------

print("🤖 SmartDiag Multi-Fault Classification System")
print("=" * 50)
mode = input("Select mode:\n1. Train Model\n2. Analyze New Data\nEnter choice (1/2): ").strip()

# ---------------------- Training Mode ----------------------

if mode == "1":
    print("\n🎯 TRAINING MODE")
    print("=" * 30)
    
    vehicle_type = input("🔍 Enter vehicle type to analyze (EV / IC / ALL): ").strip().upper()

    if vehicle_type == "EV":
        df = pd.read_csv("smartdiag_dynamic_writer_EV.csv")
        print("\n📊 Training on EV data...")
    elif vehicle_type == "IC":
        df = pd.read_csv("smartdiag_dynamic_writer_IC.csv")
        print("\n📊 Training on IC data...")
    elif vehicle_type == "ALL":
        df_ev = pd.read_csv("smartdiag_dynamic_writer_EV.csv")
        df_ic = pd.read_csv("smartdiag_dynamic_writer_IC.csv")
        df = pd.concat([df_ev, df_ic], ignore_index=True)
        print("\n📊 Training on ALL vehicle data...")
    else:
        print("❌ Invalid input. Exiting.")
        exit()

    # ---------------------- Multi-Label Preprocessing ----------------------

    df['Fault_Code'] = df['Fault_Label'].astype('category').cat.codes
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if 'Fault_Code' in numeric_cols:
        numeric_cols.remove('Fault_Code')

    X = df[numeric_cols]
    
    # Create multi-label target matrix
    unique_faults = df['Fault_Label'].unique()
    fault_to_code = {fault: idx for idx, fault in enumerate(unique_faults)}
    
    # Initialize multi-label matrix
    y_multilabel = np.zeros((len(df), len(unique_faults)))
    
    # Fill the matrix (1 if fault present, 0 if not)
    for idx, fault_label in enumerate(df['Fault_Label']):
        fault_idx = fault_to_code[fault_label]
        y_multilabel[idx, fault_idx] = 1
    
    # Save fault mapping for later use
    fault_mapping = {idx: fault for fault, idx in fault_to_code.items()}
    joblib.dump(fault_mapping, 'fault_mapping.pkl')
    
    print(f"🔍 Detected {len(unique_faults)} unique fault types:")
    for code, fault in fault_mapping.items():
        print(f"   {code}: {fault}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_multilabel, test_size=0.2, random_state=42)

    # ---------------------- Suppress Warnings ----------------------
    warnings.filterwarnings("ignore")

    # ---------------------- Model 1: Multi-Label Random Forest ----------------------

    rf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("\n🎯 Multi-Label Random Forest Report:")
    print(f"Hamming Loss: {hamming_loss(y_test, rf_preds):.4f}")
    
    # Detailed classification report for each fault type
    for i, fault_name in fault_mapping.items():
        print(f"\n📊 {fault_name}:")
        print(classification_report(y_test[:, i], rf_preds[:, i], zero_division=0))
    
    joblib.dump(rf_model, 'trained_multilabel_rf_model.pkl')

    # ---------------------- Model 2: Multi-Label Logistic Regression ----------------------

    lr_model = MultiOutputClassifier(LogisticRegression(max_iter=2000))
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    print("\n🎯 Multi-Label Logistic Regression Report:")
    print(f"Hamming Loss: {hamming_loss(y_test, lr_preds):.4f}")

    # ---------------------- Model 3: Multi-Label XGBoost Classifier ----------------------

    xgb_model = MultiOutputClassifier(XGBClassifier(eval_metric='mlogloss'))
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    print("\n🎯 Multi-Label XGBoost Classifier Report:")
    print(f"Hamming Loss: {hamming_loss(y_test, xgb_preds):.4f}")

    # ---------------------- Feature Importance (using Random Forest) ----------------------

    # Get feature importance from the first classifier (they should be similar)
    first_classifier = rf_model.estimators_[0]
    importances = pd.Series(first_classifier.feature_importances_, index=numeric_cols).sort_values(ascending=False)
    print("\n🔥 Feature Importances:")
    print(importances)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title(f"Feature Importance - {vehicle_type} Data (Multi-Label)")
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
    print("\n✅ Multi-label training completed! Model saved as 'trained_multilabel_rf_model.pkl'")

# ---------------------- Analysis Mode ----------------------

elif mode == "2":
    print("\n🔍 ANALYSIS MODE")
    print("=" * 30)
    
    # Check if trained model exists
    if not os.path.exists('trained_multilabel_rf_model.pkl'):
        print("❌ No trained multi-label model found! Please train a model first (Mode 1)")
        exit()
    
    # Load trained model and fault mapping
    print("📥 Loading trained multi-label model...")
    rf_model = joblib.load('trained_multilabel_rf_model.pkl')
    fault_mapping = joblib.load('fault_mapping.pkl')
    
    # Get file to analyze
    file_path = input("📁 Enter the CSV file path to analyze: ").strip()
    
    try:
        df_new = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df_new)} records from {file_path}")
        
        # Get numeric columns (same as training data)
        numeric_cols = df_new.select_dtypes(include='number').columns.tolist()
        
        # Make multi-label predictions
        X_new = df_new[numeric_cols]
        predictions = rf_model.predict(X_new)
        prediction_probs = rf_model.predict_proba(X_new)
        
        # Add predictions to dataframe
        df_new['Predicted_Fault_Codes'] = [list(np.where(pred == 1)[0]) for pred in predictions]
        df_new['Predicted_Fault_Labels'] = [
            [fault_mapping[code] for code in np.where(pred == 1)[0]] 
            for pred in predictions
        ]
        
        # Add individual fault columns
        for fault_code, fault_name in fault_mapping.items():
            df_new[f'Fault_{fault_code}_{fault_name}'] = predictions[:, fault_code]
        
        # Count multiple faults per record
        df_new['Fault_Count'] = [len(faults) for faults in df_new['Predicted_Fault_Labels']]
        
        # Save analyzed file
        output_filename = f"multilabel_analyzed_{os.path.basename(file_path)}"
        df_new.to_csv(output_filename, index=False)
        
        print(f"\n✅ Multi-label analysis completed!")
        print(f"📊 Results saved to: {output_filename}")
        
        # Show fault distribution
        print(f"\n🔍 Fault distribution:")
        for fault_code, fault_name in fault_mapping.items():
            fault_count = predictions[:, fault_code].sum()
            print(f"   {fault_name}: {fault_count} occurrences")
        
        # Show multiple fault statistics
        print(f"\n📈 Multiple fault statistics:")
        fault_counts = df_new['Fault_Count'].value_counts().sort_index()
        for count, num_records in fault_counts.items():
            print(f"   {count} fault(s): {num_records} records")
        
        # Show sample of predictions
        print(f"\n📋 Sample predictions (first 10 rows):")
        sample_cols = ['Timestamp'] + numeric_cols[:3] + ['Predicted_Fault_Labels', 'Fault_Count']
        print(df_new[sample_cols].head(10))
        
        # Show records with multiple faults
        multi_fault_records = df_new[df_new['Fault_Count'] > 1]
        if len(multi_fault_records) > 0:
            print(f"\n🚨 Records with multiple faults:")
            multi_sample_cols = ['Timestamp'] + numeric_cols[:3] + ['Predicted_Fault_Labels', 'Fault_Count']
            print(multi_fault_records[multi_sample_cols].head(5))
        
    except FileNotFoundError:
        print(f"❌ File {file_path} not found!")
    except Exception as e:
        print(f"❌ Error: {e}")

else:
    print("❌ Invalid mode selection. Please enter 1 or 2.")
