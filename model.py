# model.py
import joblib
from pathlib import Path


def load_model():
    model_path = Path("Model/model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Ejecuta train.py primero."
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    model = load_model()
    print("✅ Modelo cargado correctamente:", type(model))
