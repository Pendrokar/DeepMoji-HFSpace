import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, DistilBertForSequenceClassification

modelName = "Pendrokar/TorchMoji"

distil_tokenizer = AutoTokenizer.from_pretrained(modelName)
#distil_tokenizer.save_pretrained("./model/")
#distil_model = DistilBertForSequenceClassification.from_pretrained(modelName, problem_type="multi_label_classification")
distil_model = DistilBertForMultilabelSequenceClassification.from_pretrained(modelName)
#num_labels = len(model.config.id2label)

#pipeline = pipeline(task="text-classification", model=distil_model, tokenizer=distil_tokenizer)
pipeline = pipeline(task="text-classification", model=distil_model, tokenizer=distil_tokenizer)
#pipeline = pipeline(task="text-classification", model=modelName)

def predict(deepmoji_analysis):
    predictions = pipeline(deepmoji_analysis)
    return deepmoji_analysis, {p["label"]: p["score"] for p in predictions}

gradio_app = gr.Interface(fn=predict, inputs="text", outputs="text")

if __name__ == "__main__":
    gradio_app.launch()