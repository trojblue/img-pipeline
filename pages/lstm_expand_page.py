import gradio as gr
from lib.text import make_twitter_string
from lib.utils import *
from extensions.tag_expansion.lstm_expand import inference_lstm_gradio, load_model
from typing import *

embedding_dim = 64
hidden_dim = 128

def do_expansion(tag_str, tag_count, line_count, vital_tag_str, bad_tag_str):

    bad_tags = [i.strip() for i in bad_tag_str.split(",")]
    cfg = load_configs()
    vocab_path = cfg["inference"]['lstm_vocab_path']
    model_path = cfg["inference"]['lstm_model_path']

    try:
        tag_to_index, loaded_model = load_model(vocab_path, model_path, embedding_dim, hidden_dim)
    except RuntimeError:
        try:
            tag_to_index, loaded_model = load_model(vocab_path, model_path, embedding_dim*2, hidden_dim*2)
        except RuntimeError:
            try:
                tag_to_index, loaded_model = load_model(vocab_path, model_path, embedding_dim*4, hidden_dim*4)
            except RuntimeError:
                return "ERROR", "dimension expand fail"

    input_tags = [i.strip() for i in tag_str.split(",")]

    return_strs = []
    for i in range (line_count):
        tags = inference_lstm_gradio(tag_to_index, loaded_model, input_tags, tag_count)
        tags = [i for i in tags if i not in bad_tags]   # remove bad_tags
        return_str = ", ".join([i for i in input_tags + tags if i ])
        return_strs.append(vital_tag_str + return_str)  # add vital_tags

    console_str = f"{line_count} line(s) of {tag_count} tags generated"
    return "\n".join(return_strs), console_str

def get_lstm_expand_page():

    # cfg = load_configs()

    tag_str = gr.TextArea(placeholder="prompt", lines=4)
    tag_count = gr.Slider(1, 100, label="tag count", value=10, step=1)
    line_count = gr.Number(label="lines count", value=1, precision=0)
    vital_tag_str = gr.TextArea(label= "vital_tag_str",
                                  value="<lora:LoconLoraOffsetNoise_locon0501:1>",
                                lines=2
                                  )
    bad_tag_str = gr.TextArea(placeholder="bad_tags", label= "bad tags",
                              value="1boy, string bikini, english text, making of available, frilled sleeves",
                              lines=2
                              )

    page = gr.Interface(
        do_expansion,
        inputs=[
            tag_str,
            tag_count,
            line_count,
            vital_tag_str,
            bad_tag_str
        ],
        outputs=[
            gr.TextArea(placeholder="new prompt"),
            gr.TextArea(placeholder="stats"),
        ],
    )
    return page
