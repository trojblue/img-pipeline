import PIL.Image
import gradio as gr

from extensions.grid_maker.grid_maker import GridMaker
from lib.utils import *
from typing import *


def process_images(images) -> List:
    """convert gradio image inputs to a list of PIl images
    """
    pil_images = []
    for image in images:
        pil_image = PIL.Image.open(image.name)
        pil_images.append(pil_image)
    return pil_images


def do_collage(files, x_count, x_tags, y_tags, caption, save_img):
    pil_images = process_images(files)
    get_console_msg("INFO", "task started")
    grid_maker = GridMaker()
    first_iamge_grid = grid_maker.concat_images(
        binary_imgs=pil_images,
        x_count=x_count, x_tags_list=x_tags, y_tags_list=y_tags, caption=caption, save_image=save_img)

    return [first_iamge_grid]


def get_img_collage_page():
    cfg = load_configs()

    files_in = gr.File(file_count="multiple", file_types=["image"], label="images")
    x_count = gr.Slider(-1, 30, value=-1, label="x width")
    x_tags = gr.Text(label="x labels", placeholder="Tag1, Tag2, Tag3, Tag4")
    y_tags = gr.Text(label="y labels", placeholder="Tag1, Tag2, Tag3, Tag4")
    caption = gr.Text(label="captions", placeholder="some text to add at bottom of the page")
    save_img = gr.Checkbox(label="save image to dir", value=True)

    page = gr.Interface(
        do_collage,
        inputs=[
            files_in, x_count, x_tags, y_tags, caption, save_img
        ],
        outputs=[
            gr.Gallery().style(grid=2, height="auto0"),
        ],
    )
    return page
