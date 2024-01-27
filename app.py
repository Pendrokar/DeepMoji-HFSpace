import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification

modelName = "Pendrokar/TorchMoji"

distil_tokenizer = AutoTokenizer.from_pretrained(modelName)
distil_model = AutoModelForSequenceClassification.from_pretrained(modelName, problem_type="multi_label_classification")

pipeline = pipeline(task="text-classification", model=distil_model, tokenizer=distil_tokenizer)

def predict(deepmoji_analysis):
    predictions = pipeline(deepmoji_analysis)
    return deepmoji_analysis, {p["label"]: p["score"] for p in predictions}

gradio_app = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    verbose=True,
    examples=[
        "This GOT show just remember LOTR times!",
        "Man, can't believe that my 30 days of training just got a NaN loss",
        "I couldn't see 3 Tom Hollands coming...",
        "There is nothing better than a soul-warming coffee in the morning",
        "I fear the vanishing gradient", "deberta"
    ]
)

if __name__ == "__main__":
    gradio_app.launch()