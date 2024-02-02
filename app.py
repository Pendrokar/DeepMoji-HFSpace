from __future__ import print_function, division, unicode_literals

import gradio as gr

import sys
from os.path import abspath, dirname

import json
import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from transformers import AutoModel, AutoTokenizer
model_name = "Pendrokar/TorchMoji"
model = AutoModel.from_pretrained(model_name, cache_dir="~/.cache/huggingface/hub/")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "~/.cache/huggingface/hub/{model_name}/pytorch_model.bin"
vocab_path = './' + model_name + "/vocabulary.json"

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30

st = SentenceTokenizer(tokenizer.get_added_vocab(), maxlen)

model = torchmoji_emojis(model_path)

def predict(deepmoji_analysis):
    output_text = "\n"
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    prob = model(tokenized)

    for prob in [prob]:
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the torchMoji repo.
        scores = []
        for i, t in enumerate(TEST_SENTENCES):
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