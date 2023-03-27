import gradio as gr
import random


def shuffle_string(prompt: str, count, dropoff):
    new_strings = []
    p_list = [i.strip() for i in prompt.split(",")]
    p_list = [i for i in p_list if i]

    for i in range(int(count)):
        random.shuffle(p_list)
        new_strings.append(", ".join(p_list))

    return "\n".join(new_strings)


def get_prompt_shuffle_page():

    page = gr.Interface(
        shuffle_string,
        inputs=[
            gr.TextArea(placeholder="prompt"),
            gr.Slider(1, 100, label="shuffle count", value=1, step=1),
            gr.Slider(0, 1, label="drop off", value=0.05, step=1)
        ],
        outputs=[gr.TextArea(placeholder="new prompt")],
    )

    return page
