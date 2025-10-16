import gradio as gr
import pandas as pd

# --- Función principal ---
def analizar_archivo(archivo):
    if archivo is None:
        return "⚠️ Por favor sube un archivo CSV", None
    try:
        # Leer el CSV
        df = pd.read_csv(archivo.name)
        vista_previa = df.head()
        return "✅ Archivo cargado correctamente", vista_previa
    except Exception as e:
        return f"❌ Error al leer el archivo: {e}", None


# --- Interfaz de Gradio ---
demo = gr.Interface(
    fn=analizar_archivo,
    inputs=gr.File(label="Sube un archivo CSV para analizar", file_count="single", file_types=[".csv"]),
    outputs=[
        gr.Textbox(label="Estado del archivo"),
        gr.Dataframe(label="Vista previa del archivo"),
    ],
    title="💊 Drug Classification App",
    description="Demostración de despliegue CI/CD con Gradio y Hugging Face 🚀",
    theme="soft",
    allow_flagging="never"
)

# --- Ejecutar la app ---
if __name__ == "__main__":
    demo.launch()
