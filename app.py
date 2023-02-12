import gradio as gr
from gradio import TabbedInterface
from gradio.components import *

from lib.twitter_alt import make_twitter_string
from lib.shuffle_tags import shuffle_string

def calculate_text_length(text):
    return len(text)


import pyperclip
def copy_to_clipboard(text):
    pyperclip.copy(text)

def get_interface() -> TabbedInterface:

    txt_interface = gr.Interface(
        make_twitter_string,
        inputs=[
            gr.TextArea(placeholder="prompt"),
            gr.Checkbox(label="No model name", value=True),
            gr.Checkbox(label="No artists", value=True),
            gr.Checkbox(label="No seed"),
            gr.Checkbox(label="SHORTER"),
            gr.Checkbox(label="EVEN SHORTER"),
        ],
        outputs=[
            gr.TextArea(placeholder="new prompt"),
            gr.TextArea(placeholder="stats"),
        ],

    )

    shuffle_interface = gr.Interface(
        shuffle_string,
        inputs=[
            gr.TextArea(placeholder="prompt"),
        ],
        outputs=[
            gr.TextArea(placeholder="new prompt")
        ],
    )

    demo = gr.TabbedInterface(
        [txt_interface, shuffle_interface], ["tag_string_clean", "Text-to-speech"]
    )
    return demo


if __name__ == "__main__":
    interface = get_interface()
    interface.launch(
        server_port=7866,
        debug=True,
        server_name="0.0.0.0",
        favicon_path="./bin/favicon.ico"
    )
