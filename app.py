import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification

modelName = "Pendrokar/TorchMoji"

distil_tokenizer = AutoTokenizer.from_pretrained(modelName)
distil_model = AutoModelForSequenceClassification.from_pretrained(modelName, problem_type="multi_label_classification")

pipeline = pipeline(task="text-classification", model=distil_model, tokenizer=distil_tokenizer)

def predict(deepmoji_analysis):
    predictions = pipeline(deepmoji_analysis)

    output_text = "\n"
    for p in predictions:
        output_text += p['label'] + ' (' + str(p['score']) + ")\n"
    return str(distil_tokenizer(deepmoji_analysis)["input_ids"]) + output_text

gradio_app = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
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