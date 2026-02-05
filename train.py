import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#1. Load Cleaned Dataset
df = pd.read_csv("data/telco.csv")
df.columns = df.columns.str.strip()
# Drop leakage columns
leakage_cols = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason",
    "Churn Score"
]
df = df.drop(columns=leakage_cols)
# Drop missing values
df = df.dropna()

#2. Separate Features & Target
X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]

#3. Encode Categorical Features
label_encoders = {}
for col in X.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

#4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#5. Train Models
# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:",
      accuracy_score(y_test, y_pred_rf))

#6. Evaluation
print("\nLogistic Regression Report\n")
print(classification_report(y_test, y_pred_lr))

print("\nRandom Forest Report\n")
print(classification_report(y_test, y_pred_rf))

#7. Save Best Model
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(lr, f)

with open("model/encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
