from __future__ import print_function, division, unicode_literals

import gradio as gr

import sys
from os.path import abspath, dirname

import json
import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis

from huggingface_hub import hf_hub_download

model_name = "Uberduck/torchmoji"
model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
vocab_path = hf_hub_download(repo_id=model_name, filename="vocabulary.json")

emoji_codes = []
with open('./data/emoji_codes.json', 'r') as f:
    emoji_codes = json.load(f)

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30

with open(vocab_path, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

model = torchmoji_emojis(model_path)

def predict(deepmoji_analysis, emoji_count):
    return_label = {}
    tokenized, _, _ = st.tokenize_sentences([deepmoji_analysis])
    prob = model(tokenized)

    for prob in [prob]:
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the torchMoji repo.
        scores = []
        for i, t in enumerate([deepmoji_analysis]):
            t_prob = prob[i]
            # sort top
            ind_top_ids = top_elements(t_prob, emoji_count)

            for ind in ind_top_ids:
                return_label[emoji_codes[ind]] = t_prob[ind]

    return return_label

input_textbox = gr.Textbox(
    label="English Text",
    lines=1,
    value=""
)
slider = gr.Slider(1, 64, value=5, step=1, label="Top # Emoji", info="Choose between 1 and 64 top emojis to show")

gradio_app = gr.Interface(
    predict,
    [
        input_textbox,
        slider,
    ],
    outputs="label",
    examples=[
        ["You love hurting me, huh?", 5],
        ["I know good movies, this ain't one", 5],
        ["It was fun, but I'm not going to miss you", 5],
        ["My flight is delayed.. amazing.", 5],
        ["What is happening to me??", 5],
        ["This is the shit!", 5],
        ["This is shit!", 5],
    ],
    live=True
)

if __name__ == "__main__":
    gradio_app.launch()