import gradio as gr
from lib.text import make_twitter_string


def get_prompt_clean_page():
    page = gr.Interface(
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
    return page
