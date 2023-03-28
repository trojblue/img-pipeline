import gradio as gr
from gradio import TabbedInterface
import pyperclip
from gradio.components import *
from pages.img_collage_page import get_img_collage_page
from pages.prompt_clean_page import get_prompt_clean_page
from pages.prompt_shuffle_page import get_prompt_shuffle_page
from pages.img_clean_page import get_img_clean_page
from pages.prompt_gen_page import get_prompt_gen_page
from pages.lstm_expand_page import get_lstm_expand_page
from pages.img_lookup_page import get_img_lookup_page
from pages.prompt_sr_page import get_prompt_sr_page
from pages.img_compress_page import get_img_compress_page



def get_interface() -> TabbedInterface:
    # PROMPT
    prompt_inner_pages = [
        (get_prompt_gen_page(), "gen prompt"),
        (get_prompt_sr_page(), "promptSR"),
        (get_prompt_clean_page(), "clean prompt"),
        (get_prompt_shuffle_page(), "shuffle tags"),
        (get_lstm_expand_page(), "lstm"),
    ]
    Prompt_tabbed_interface = gr.TabbedInterface(
        [page[0] for page in prompt_inner_pages], [page[1] for page in prompt_inner_pages]
    )

    # TRAIN
    train_inner_pages = [
        (get_img_compress_page(), "compress"),
        (get_img_lookup_page(), "img_lookup_page"),
    ]
    Train_tabbed_interface = gr.TabbedInterface(
        [page[0] for page in train_inner_pages], [page[1] for page in train_inner_pages]
    )


    # Outer tabbed interface
    outer_pages = [
        (get_img_clean_page(), "upscale"),
        (get_img_collage_page(), "gen collage"),
        (Prompt_tabbed_interface, "PROMPT"),
        (Train_tabbed_interface, "TRAIN"),
    ]

    demo = gr.TabbedInterface(
        [page[0] for page in outer_pages], [page[1] for page in outer_pages]
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
