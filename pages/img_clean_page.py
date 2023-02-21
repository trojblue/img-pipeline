import gradio as gr
from lib.img import do_before_upload
from lib.utils import *


# from lib.text import make_twitter_string

def get_img_clean_page():
    cfg = load_configs()

    img_in = cfg['img_in_dir']
    img_out = cfg['img_out_dir']

    html = gr.HTML(
        f"<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running."
        + f"<br>Use an empty output directory to save pictures normally instead of writing to the output directory."
        + f"<br>Add inpaint batch mask directory to enable inpaint batch processing."
          f"</p>"
        )

    file = gr.File(
                label="Batch Process",
                file_count="multiple",
                file_types=['image'],
                interactive=True,
                type="file",
                elem_id="extras_image_batch",
            )
    page = gr.Interface(
        do_before_upload,

        inputs=[
            html,
            gr.Textbox(label="Input directory", elem_id="img2img_batch_input_dir", value=img_in),
            gr.Textbox(label="Output directory", elem_id="img2img_batch_input_dir", value=img_out),
            file
        ],
        outputs=[
            gr.Gallery(
            ).style(grid=4, height="auto"),

            gr.TextArea(placeholder="stats")

        ],
    )

    return page
