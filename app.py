from __future__ import print_function, division, unicode_literals

import gradio as gr

import sys
from os.path import abspath, dirname

import json
import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis

from huggingface_hub import hf_hub_download

model_name = "Pendrokar/TorchMoji"
model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
vocab_path = hf_hub_download(repo_id=model_name, filename="vocabulary.json")

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30

print('Tokenizing using dictionary from {}'.format(vocab_path))
with open(vocab_path, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

model = torchmoji_emojis(model_path)

def predict(deepmoji_analysis):
    output_text = "\n"
    tokenized, _, _ = st.tokenize_sentences([deepmoji_analysis])
    prob = model(tokenized)

    for prob in [prob]:
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the torchMoji repo.
        scores = []
        for i, t in enumerate([deepmoji_analysis]):
            t_tokens = tokenized[i]
            t_score = [t]
            t_prob = prob[i]
            ind_top = top_elements(t_prob, 5)
            t_score.append(sum(t_prob[ind_top]))
            t_score.extend(ind_top)
            t_score.extend([t_prob[ind] for ind in ind_top])
            scores.append(t_score)
            output_text += t_score

    return str(tokenized) + output_text

gradio_app = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    examples=[
        "You love hurting me, huh?",
        "I know good movies, this ain't one",
        "It was fun, but I'm not going to miss you",
        "My flight is delayed.. amazing.",
        "What is happening to me??",
        "This is the shit!",
        "This is shit!",
    ]
)

if __name__ == "__main__":
    gradio_app.launch()