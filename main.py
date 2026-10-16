
print("Customer Churn Prediction System started!")

import pandas as pd

print("Loading dataset...")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())
print("\nDataset Information:")
df.info()
print("\nMissing Values")
print(df.isnull().sum())
print("\nTotalcharge datatype:")
print(df["TotalCharges"].dtype)
blank_charges=df["TotalCharges"].str.strip().eq("").sum()
print("\nBlank TotalCharges values:",blank_charges)
df["TotalCharges"]=df["TotalCharges"].str.strip()
df["TotalCharges"]=df["TotalCharges"].replace("",pd.NA)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())
print("\nTotalCharges data type after conversion:")
print(df["TotalCharges"].dtype)
print("\nMissing TotalCharges values after conversion:")
print(df["TotalCharges"].isnull().sum())
df["totalCharges"]=df["TotalCharges"].fillna(0)
print("\nMissing values after cleaning TotalCharges:")
print(df["TotalCharges"].isnull().sum())
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)
# Remove the accidental duplicate column
if "totalCharges" in df.columns:
    df = df.drop(columns=["totalCharges"])

print("\nColumns after removing duplicate:")
print(df.columns.tolist())
print("Dataset shape:", df.shape)
print("\nCustomer Churn Counts:")
print(df["Churn"].value_counts())

print("\nCustomer Churn Percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

print("\nInput Features Shape:", X.shape)
print("Target Shape:", y.shape)
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nEncoded Features Shape:", X.shape)
print(X.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

model.fit(X_train, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["No", "Yes"]))
import joblib
joblib.dump(model, "churn_model.pkl")

print("\nModel saved successfully!")

# Compare Logistic Regression and Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Use the same training and testing data
models = {
    "Logistic Regression": model,
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

results = []

for name, classifier in models.items():
    if name == "Logistic Regression":
        predictions = classifier.predict(X_test)
    else:
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(
            precision_score(y_test, predictions, pos_label="Yes"), 4
        ),
        "Recall": round(
            recall_score(y_test, predictions, pos_label="Yes"), 4
        ),
        "F1 Score": round(
            f1_score(y_test, predictions, pos_label="Yes"), 4
        ),
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

from sklearn.metrics import roc_auc_score

print("\nROC-AUC COMPARISON")

for name, classifier in models.items():
    probabilities = classifier.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(
        (y_test == "Yes").astype(int),
        probabilities
    )
    print(f"{name}: {auc:.4f}")
    
import matplotlib.pyplot as plt

# Visualize model performance
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

comparison = results_df.set_index("Model")[metrics]

comparison.plot(kind="bar", figsize=(10, 6))

plt.title("Customer Churn Model Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("\nModel comparison chart saved as model_comparison.png")
