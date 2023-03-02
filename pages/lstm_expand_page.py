import gradio as gr
from lib.text import make_twitter_string
from lib.utils import *
from extensions.tag_expansion.lstm_expand import inference_lstm_gradio, load_model
from typing import *



def do_expansion(tag_str, tag_count):

    cfg = load_configs()
    vocab_path = cfg["inference"]['lstm_vocab_path']
    model_path = cfg["inference"]['lstm_model_path']

    tag_to_index, loaded_model = load_model(vocab_path, model_path)
    input_tags = [i.strip() for i in tag_str.split(",")]

    tags = inference_lstm_gradio(tag_to_index, loaded_model, input_tags, tag_count)
    return_str = ", ".join([i for i in input_tags + tags if i ])

    return return_str, tags

def get_lstm_expand_page():

    # cfg = load_configs()

    tag_str = gr.TextArea(placeholder="prompt")
    tag_count = gr.Slider(1, 100, label="tag count", value=10, step=1)

    page = gr.Interface(
        do_expansion,
        inputs=[
            tag_str,
            tag_count
        ],
        outputs=[
            gr.TextArea(placeholder="new prompt"),
            gr.TextArea(placeholder="stats"),
        ],
    )
    return page
