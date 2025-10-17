# train.py
import os
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Crear carpetas necesarias
Path("Model").mkdir(exist_ok=True)
Path("Results").mkdir(exist_ok=True)


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    joblib.dump(pipeline, "Model/model.pkl")

    with open("Results/metrics.txt", "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"f1_score: {f1:.4f}\n")

    print("✅ Entrenamiento completado y modelo guardado.")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
