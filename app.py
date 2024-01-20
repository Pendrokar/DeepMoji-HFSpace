import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="emoji-prediction", model="Uberduck/torchmoji")

def predict(deepmoji_analysis):
    predictions = pipeline(deepmoji_analysis)
    return deepmoji_analysis, {p["label"]: p["score"] for p in predictions}

gradio_app = gr.Interface(fn=predict, inputs="text", outputs="text")

if __name__ == "__main__":
    gradio_app.launch()