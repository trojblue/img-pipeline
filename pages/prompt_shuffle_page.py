import gradio as gr
import random


def shuffle_string(prompt: str):
    p_list = [i.strip() for i in prompt.split(",")]

    random.shuffle(p_list)
    return ", ".join(p_list)


def get_prompt_shuffle_page():
    page = gr.Interface(
        shuffle_string,
        inputs=[
            gr.TextArea(placeholder="prompt"),
        ],
        outputs=[gr.TextArea(placeholder="new prompt")],
    )

    return page
