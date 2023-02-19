import gradio as gr
from lib.shuffle_tags import shuffle_string


def get_prompt_shuffle_page():
    page = gr.Interface(
        shuffle_string,
        inputs=[
            gr.TextArea(placeholder="prompt"),
        ],
        outputs=[gr.TextArea(placeholder="new prompt")],
    )

    return page
