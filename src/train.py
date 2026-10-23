
import math
import os
import sys
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ACCEPTANCE_THRESHOLD = 0.85
ARTIFACT_PATH = "artifacts/model.joblib"


def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score : {f1:.4f}")

    return model, accuracy, f1


def validate_and_save(model, f1):
    if not math.isfinite(f1) or not (0.0 <= f1 <= 1.0):
        print("ERROR: F1-score is not a valid value.")
        sys.exit(1)

    if f1 < ACCEPTANCE_THRESHOLD:
        print(
            f"FAIL: F1 {f1:.4f} below "
            f"threshold {ACCEPTANCE_THRESHOLD}"
        )
        sys.exit(1)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    print(f"Model saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    model, accuracy, f1 = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    validate_and_save(model, f1)
