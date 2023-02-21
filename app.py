import gradio as gr
from gradio import TabbedInterface
from gradio.components import *

from pages.prompt_clean_page import get_prompt_clean_page
from pages.prompt_shuffle_page import get_prompt_shuffle_page
from pages.img_clean_page import get_img_clean_page


def calculate_text_length(text):
    return len(text)


import pyperclip


def copy_to_clipboard(text):
    pyperclip.copy(text)


def get_interface() -> TabbedInterface:
    prompt_clean_page = get_prompt_clean_page()
    shuffle_page = get_prompt_shuffle_page()
    img_page = get_img_clean_page()

    demo = gr.TabbedInterface(
        [img_page, prompt_clean_page, shuffle_page],
        ["image page", "clean prompt", "shuffle tags"],
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
