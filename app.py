import gradio as gr
from gradio import TabbedInterface
from gradio.components import *

from pages.img_collage_page import get_img_collage_page
from pages.prompt_clean_page import get_prompt_clean_page
from pages.prompt_shuffle_page import get_prompt_shuffle_page
from pages.img_clean_page import get_img_clean_page
from pages.prompt_gen_page import get_prompt_gen_page
from pages.lstm_expand_page import get_lstm_expand_page

def calculate_text_length(text):
    return len(text)


import pyperclip


def copy_to_clipboard(text):
    pyperclip.copy(text)


def get_interface() -> TabbedInterface:
    img_page = get_img_clean_page()
    prompt_clean_page = get_prompt_clean_page()
    img_collage_page = get_img_collage_page()
    gen_page = get_prompt_gen_page()
    shuffle_page = get_prompt_shuffle_page()
    lstm_page = get_lstm_expand_page()


    demo = gr.TabbedInterface(
        [img_page, gen_page, img_collage_page, prompt_clean_page, shuffle_page, lstm_page],
        ["upscale", "gen prompt", "gen collage", "clean prompt", "shuffle tags", "lstm"],
    )
    return demo


if __name__ == "__main__":
    interface = get_interface()
    interface.launch(
        server_port=7867,
        debug=True,
        server_name="0.0.0.0",
        favicon_path="./bin/favicon.ico",
    )
