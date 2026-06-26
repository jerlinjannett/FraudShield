import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

print("=" * 50)
print("   FraudShield — Model Training")
print("=" * 50)

# ── LOAD DATA ──────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv("creditcard.csv")
print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")

# ── PREPROCESS ─────────────────────────────────────────
print("\n[2/6] Preprocessing...")

if "Time" in df.columns:
    df.drop("Time", axis=1, inplace=True)

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"      Normal transactions : {(y==0).sum():,}")
print(f"      Fraud  transactions : {(y==1).sum():,}")

# ── SCALE AMOUNT ───────────────────────────────────────
print("\n[3/6] Scaling Amount column...")
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

# ── TRAIN TEST SPLIT ───────────────────────────────────
print("\n[4/6] Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train size : {len(X_train):,}")
print(f"      Test  size : {len(X_test):,}")

# ── SMOTE ──────────────────────────────────────────────
print("\n[5/6] Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"      After SMOTE — Total rows : {len(X_train_res):,}")
print(f"      Normal : {(y_train_res==0).sum():,}")
print(f"      Fraud  : {(y_train_res==1).sum():,}")

# ── TRAIN MODEL ────────────────────────────────────────
print("\n[6/6] Training Random Forest model...")
print("      This may take 2-5 minutes...")

model = RandomForestClassifier(
    n_estimators = 100,
    max_depth    = 15,
    random_state = 42,
    n_jobs       = -1
)

model.fit(X_train_res, y_train_res)
print("      Training complete!")

# ── EVALUATE ───────────────────────────────────────────
print("\n" + "=" * 50)
print("   Model Evaluation Results")
print("=" * 50)

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal","Fraud"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True  Normal : {cm[0][0]:,}")
print(f"  False Fraud  : {cm[0][1]:,}")
print(f"  False Normal : {cm[1][0]:,}")
print(f"  True  Fraud  : {cm[1][1]:,}")

# ── SAVE MODEL ─────────────────────────────────────────
print("\n" + "=" * 50)
print("   Saving Model Files")
print("=" * 50)

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n  fraud_model.pkl saved")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  scaler.pkl saved")

print("\n" + "=" * 50)
print("   Training Complete!")
print("   Run:  py app.py")
print("=" * 50)