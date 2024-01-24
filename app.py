import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, DistilBertForSequenceClassification

modelName = "Pendrokar/TorchMoji"

distil_tokenizer = AutoTokenizer.from_pretrained(modelName)
distil_model = DistilBertForSequenceClassification.from_pretrained(modelName, problem_type="multi_label_classification")

pipeline = pipeline(task="text-classification", model=distil_model, tokenizer=distil_tokenizer)

def predict(deepmoji_analysis):
    predictions = pipeline(deepmoji_analysis)
    return deepmoji_analysis, {p["label"]: p["score"] for p in predictions}

gradio_app = gr.Interface(fn=predict, inputs="text", outputs="text")

if __name__ == "__main__":
    gradio_app.launch()