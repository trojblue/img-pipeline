import random
import gradio as gr
import imagehash

from lib.text import make_twitter_string
from lib.utils import *
from typing import *
import lib.img
import sdtools.fileops as fops
import sdtools.txtops as tops
from PIL import Image, ImageOps


def concat_images2(images, x_count, y_count):

    # Open images and resize them
    pil_images = [Image.open(i) for i in images]
    # pil_images = images
    first_img = pil_images[0]
    print(first_img, type(first_img))
    width, height = first_img.size

    pil_images = [ImageOps.fit(image, first_img.size, Image.ANTIALIAS)
              for image in pil_images]
    # images = [image for image in images]

    # Create canvas for the final image with total size
    shape = (1, len(pil_images))
    image_size = (width * shape[1], height * shape[0])
    grid_image = Image.new('RGB', image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            grid_image.paste(pil_images[idx], offset)

    # save img
    hash = imagehash.average_hash(grid_image)
    grid_name = f"grid_{tops.get_date_str('file')}_{hash}.png"
    cfg = load_configs()
    prompt_in_dir = cfg['img_out_dir']
    filepath = os.path.join(prompt_in_dir, grid_name)
    grid_image.save(filepath)

    # generate message
    debug_string = tops.get_console_msg("INFO", f"grid created at {filepath}")

    # return response
    return filepath, debug_string


def get_img_collage_page():
    cfg = load_configs()


    file = gr.File(
                label="Batch Process",
                file_count="multiple",
                file_types=['image'],
                interactive=True,
                type="file",
            )


    xy = gr.Text(label="xy", value="x")
    xy_len = gr.Slider(-1, 8, label="xy len", value=-1, step=1)
    tag_str = gr.Text(label="tags")


    page = gr.Interface(
        lib.img.concat_images,

        inputs=[
            file, xy, xy_len, tag_str
        ],
        outputs=[
            gr.Image(interactive=True),
            gr.TextArea(placeholder="stats", interactive=True)
        ],
    )
    return page