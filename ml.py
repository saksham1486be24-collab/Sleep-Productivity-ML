
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -------------------------
# Config / Paths
# -------------------------
# Prefer dataset present in current working area (uploaded by you),
# fallback to the Windows path from original script if present.
FALLBACK_PATH = r"C:\\Users\\patha\\Desktop\\sleep_productivity_dataset_1000.csv"
UPLOADED_PATH = "/mnt/data/sleep_productivity_dataset_1000.csv"  # <- your uploaded CSV
LOCAL_CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "") if "__file__" in globals() else os.getcwd()

candidates = [
    os.path.join(LOCAL_CWD, "C:\\Users\\patha\\Desktop\\sleep_productivity_dataset_1000.csv"),
    UPLOADED_PATH,
    FALLBACK_PATH
]

DATA_PATH = next((p for p in candidates if p and os.path.exists(p)), None)

if DATA_PATH is None:
    logging.error("Dataset not found in usual locations. Tried:\n%s", "\n".join(candidates))
    raise FileNotFoundError("Dataset not found. Please provide correct path to csv.")

logging.info("Loading dataset from: %s", DATA_PATH)
df = pd.read_csv(DATA_PATH)
logging.info("Initial shape: %s", df.shape)
logging.info("Columns: %s", list(df.columns))

# -------------------------
# Basic cleaning
# -------------------------
drop_cols = ["Unnamed: 0", "id", "geolocation", "location"]
to_drop = [c for c in drop_cols if c in df.columns]
if to_drop:
    df = df.drop(columns=to_drop)
    logging.info("Dropped columns: %s", to_drop)

# -------------------------
# Target detection / creation
# -------------------------
target_col = None
for col in df.columns:
    if col.lower() in ["disastertype", "disaster_type", "severity", "impact_level", "target"]:
        target_col = col
        break

if target_col is None:
    logging.info("No target found. Creating SyntheticTarget (binary) based on numeric features' mean.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available to build a synthetic target.")
    df["SyntheticTarget"] = (df[numeric_cols].mean(axis=1) > df[numeric_cols].mean().mean()).astype(int)
    target_col = "SyntheticTarget"

logging.info("Using target column: %s", target_col)

# -------------------------
# Preprocessing
# -------------------------
# Drop rows with NA (simple); you could use imputation if you prefer
df = df.dropna().reset_index(drop=True)
logging.info("Shape after dropna: %s", df.shape)

# Separate X/y
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical encoding -- use get_dummies for safety (avoids LabelEncoder pitfalls)
obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if obj_cols:
    logging.info("One-hot encoding object columns: %s", obj_cols)
    X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# -------------------------
# Train/Test split (stratify to keep class balance)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

# -------------------------
# Optional: handle class imbalance by upsampling minority on train set (simple demo)
# -------------------------
if y_train.nunique() == 2:
    # compute class counts
    counts = y_train.value_counts()
    if counts.min() / counts.max() < 0.6:
        logging.info("Detected class imbalance (train). Performing simple upsampling of minority class.")
        train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        majority = train_df[train_df[target_col] == counts.idxmax()]
        minority = train_df[train_df[target_col] == counts.idxmin()]
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        train_balanced = pd.concat([majority, minority_upsampled])
        y_train = train_balanced[target_col]
        X_train = train_balanced.drop(columns=[target_col])
        # convert to numeric index
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

# -------------------------
# Model training
# -------------------------
logging.info("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
acc = accuracy_score(y_test, y_pred)
logging.info("Model accuracy: %.4f", acc)
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Save evaluation artifacts directory
ART_DIR = "/mnt/data/model_artifacts"
os.makedirs(ART_DIR, exist_ok=True)

# -------------------------
# Feature importance (plot + save)
# -------------------------
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(12)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 12 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "feature_importance_top12.png"))
plt.show()

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "confusion_matrix.png"))
plt.show()

# -------------------------
# Correlation Heatmap (works on original X numeric columns)
# -------------------------
corr = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "correlation_heatmap.png"))
plt.show()

# -------------------------
# Target distribution
# -------------------------
plt.figure(figsize=(6, 4))
y.value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title(f"Distribution of target: {target_col}")
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "target_distribution.png"))
plt.show()

# -------------------------
# SHAP explainability (robust)
# -------------------------
logging.info("Generating SHAP plots (this may take a moment)...")
try:
    # Try TreeExplainer path (fast for tree models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # shap.summary_plot supports both multi-class and binary (list) outputs
    plt.figure(figsize=(10, 6))
    # For binary classification, shap_values is usually a list [class0, class1]
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, feature_names=X.columns, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(ART_DIR, "shap_summary.png"))
    plt.show()

    # Local force plot for the first test sample
    # For notebook usage shap.force_plot works inline, but we save an HTML export here as fallback
    try:
        if isinstance(shap_values, list):
            ev = explainer.expected_value[1]
            sv1 = shap_values[1][0, :]
        else:
            ev = explainer.expected_value
            sv1 = shap_values[0, :]
        force_html = shap.force_plot(ev, sv1, X_test.iloc[0, :], matplotlib=False, show=False)
        shap.save_html(os.path.join(ART_DIR, "shap_force_sample0.html"), force_html)
    except Exception:
        logging.warning("Could not create interactive SHAP force plot. This is optional.")
except Exception as e:
    logging.exception("SHAP plotting failed: %s", e)

# -------------------------
# Accuracy visualization (saved)
# -------------------------
plt.figure(figsize=(4, 3))
plt.bar(["Accuracy"], [acc], color="#f4a261")
plt.ylim(0, 1)
plt.title("Overall Model Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "accuracy.png"))
plt.show()

# -------------------------
# Summary printout
# -------------------------
print("\nSummary:")
print(" - Dataset shape (after cleaning):", df.shape)
print(f" - Target used: {target_col}")
print(f" - Model accuracy: {acc:.4f}")
print(" - Top features:", list(top_features.index[:8]))
print(f" - Artifacts saved to: {ART_DIR}")

corr = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "correlation_heatmap.png"))
plt.show()
