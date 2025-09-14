import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

DATA_PATH = Path("Mental Health Dataset.csv")
MODEL_PATH = Path("model.pkl")
META_PATH = Path("model_meta.json")
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True, parents=True)

LIKELY_TARGET_KEYS = [
    "target", "label", "status", "diagnosis", "condition",
    "mental", "health", "outcome", "result", "class", "y"
]

NEGATIVE_NORMAL_KEYS = ["normal", "healthy", "no", "none", "absent", "negative", "low"]
POSITIVE_STRESS_KEYS = ["stress", "depress", "anxiety", "ill", "disorder", "unhealthy", "positive", "high"]

def guess_target_column(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.lower() not in ["id", "user", "username"]]
    scored = []
    for c in cols:
        lc = c.lower()
        score = sum(int(k in lc) for k in LIKELY_TARGET_KEYS)
        scored.append((score, c))
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return cols[-1]

def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    nunique = X.nunique(dropna=False)
    id_like = nunique[nunique == len(X)].index.tolist()
    X = X.drop(columns=id_like, errors='ignore')
    return X, y

def infer_types(X: pd.DataFrame):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def pick_positive_label(classes):
    lower = [str(c).lower() for c in classes]
    for i, name in enumerate(lower):
        if any(k in name for k in POSITIVE_STRESS_KEYS):
            return classes[i]
    for i, name in enumerate(lower):
        if any(k in name for k in NEGATIVE_NORMAL_KEYS) and len(classes) == 2:
            return classes[1 - i]
    return classes[0]

# ---------- Load data ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")
df = pd.read_csv(DATA_PATH)

df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

target_col = guess_target_column(df)
print(f"[i] Using target column: {target_col}")

if pd.api.types.is_numeric_dtype(df[target_col]):
    df[target_col] = df[target_col].astype(int).astype(str)
else:
    df[target_col] = df[target_col].astype(str)

X, y = split_features_target(df, target_col)
num_cols, cat_cols = infer_types(X)

numeric_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer([
    ("num", numeric_tf, num_cols),
    ("cat", categorical_tf, cat_cols),
])

clf = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=200, solver="liblinear"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if y.nunique() < 50 else None
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
acc = accuracy_score(y_test, pred)
print(f"[i] Validation accuracy: {acc:.3f}")

# Save accuracy bar plot
plt.figure()
plt.bar(["Accuracy"], [acc])
plt.ylim(0, 1)
plt.title("Validation Accuracy")
plt.ylabel("Accuracy")
plt.savefig(STATIC_DIR / "accuracy.png", bbox_inches='tight')
plt.close()

# Save confusion matrix
plt.figure()
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Confusion Matrix")
plt.savefig(STATIC_DIR / "confusion_matrix.png", bbox_inches='tight')
plt.close()

joblib.dump(clf, MODEL_PATH)

meta = {
    "target": target_col,
    "numeric_features": num_cols,
    "categorical_features": cat_cols,
    "categorical_choices": {},
    "classes_": clf.named_steps["clf"].classes_.tolist(),
    "positive_label": str(pick_positive_label(clf.named_steps["clf"].classes_)),
}

for c in cat_cols:
    counts = (
        X[c].astype(str)
        .fillna("Unknown")
        .value_counts(dropna=False)
        .head(20)
        .index.tolist()
    )
    meta["categorical_choices"][c] = [str(v) for v in counts]

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("[✓] Model, metadata, and plots saved.")
