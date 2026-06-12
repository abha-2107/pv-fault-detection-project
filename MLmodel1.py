import os
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

folder_path = r"C:\Users\HP\Desktop\PV_PROJECT\PV_Data\PV_Data"
all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
df_list = []

def clean_and_float(val):
    """Removes units and non-numeric junk."""
    if pd.isna(val) or str(val).lower() == 'nan': return np.nan
    cleaned = re.sub(r'[^0-9.\-]', '', str(val))
    return float(cleaned) if cleaned != '' else np.nan

print(f"Building Classification Model from {len(all_files)} files...")

for file in all_files:
    path = os.path.join(folder_path, file)
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            skip_idx = -1
            for i, line in enumerate(lines[:50]):
                if 'Time' in line or 'Irridance' in line:
                    skip_idx = i
                    break
        
        if skip_idx != -1:
            temp_df = pd.read_csv(path, skiprows=skip_idx, on_bad_lines='skip')
            temp_df.columns = temp_df.columns.str.strip()
            
            cm = {c.lower().replace('_', ' '): c for c in temp_df.columns}
            target_col = cm.get('v1') 
            feat_keys = ['irridance', 'humidity', 'ta', 't1', 'v2']
            valid_feats = [cm[k] for k in feat_keys if k in cm]

            if target_col and valid_feats:
                df_list.append(temp_df[valid_feats + [target_col]].astype(str))
    except:
        continue

if not df_list:
    print("Error: No valid headers found in CSVs.")
    exit()

df = pd.concat(df_list, ignore_index=True)
for col in df.columns:
    df[col] = df[col].apply(clean_and_float)

actual_target = df.columns[-1]
df_clean = df.dropna().copy()

df_clean['Performance_Level'] = pd.qcut(
    df_clean[actual_target].rank(method='first'), 
    q=3, 
    labels=['Low', 'Medium', 'High']
)

X = df_clean.drop(columns=[actual_target, 'Performance_Level'])
y = df_clean['Performance_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n" + "="*50)
print("      RANDOM FOREST CLASSIFICATION RESULTS")
print("="*50)
print(f"Accuracy Score:    {accuracy_score(y_test, y_pred):.4f}")
print(f"Weighted Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Weighted F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("-" * 50)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))