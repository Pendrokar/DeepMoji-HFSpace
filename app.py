import gradio as gr
from transformers import pipeline, AutoTokenizer, DistilBertForSequenceClassification

modelName = "colinryan/hf-deepmoji"

distilTokenizer = AutoTokenizer.from_pretrained(modelName)
distilTokenizer.save_pretrained("./model/")
distilModel = DistilBertForSequenceClassification.from_pretrained(modelName, problem_type="multi_label_classification", num_labels=64)
#distilModel = DistilBertForMultilabelSequenceClassification.from_pretrained("colinryan/hf-deepmoji")

pipeline = pipeline(task="text-classification", model=distilModel, tokenizer=tokenizer)

def predict(deepmoji_analysis):
    predictions = pipeline(deepmoji_analysis)
    return deepmoji_analysis, {p["label"]: p["score"] for p in predictions}

gradio_app = gr.Interface(fn=predict, inputs="text", outputs="text")

if __name__ == "__main__":
    gradio_app.launch()