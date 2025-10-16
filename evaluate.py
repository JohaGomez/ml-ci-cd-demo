# evaluate.py
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

def main():
    model_path = Path("Model/model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("⚠️ No se encontró Model/model.pkl. Ejecuta primero train.py")

    model = joblib.load(model_path)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["maligno", "benigno"])
    disp.plot(values_format="d")
    plt.tight_layout()
    plt.savefig("Results/model_results.png", dpi=180)
    plt.close()

    print("✅ Evaluación completada.")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("📊 Matriz de confusión guardada en Results/model_results.png")

if __name__ == "__main__":
    main()
