import os
import imagehash
import webuiapi
from PIL import Image
from lib.utils import load_configs
from sdtools import fileops as fops, txtops as tops
import math
from typing import List
from PIL import Image, ImageDraw, ImageFont


def do_before_upload(html, dir_in, dir_out, img_files):
    """
    :param html: 占位
    :param dir_in: 见config.toml
    :param dir_out: 同
    :param file: List[Byte]
    :return:
    """
    if not img_files:
        # read from dir_in
        pass

    config = load_configs()
    save_dir = config['img_save_dir']

    fops.mkdir_if_not_exist(dir_out)
    fops.mkdir_if_not_exist(save_dir)

    output_images = []
    img_to_upscale = []

    for file in img_files:
        img = Image.open(file)
        img_to_upscale.append(img)
        hash = imagehash.average_hash(img)
        date_str = tops.get_date_str(mode='file')
        filename = os.path.join(save_dir, f"{date_str}_{hash}.png")
        img.save(filename)

        # Append the file path to the output list
        output_images.append(filename)

    uped_files = upscale_and_save(dir_out, img_to_upscale)

    debug_str = f"{len(img_files)} images processed: \n" \
                f"{uped_files}"

    return uped_files, debug_str


def upscale_and_save(dir_out, img_files):
    """

    :param dir_out:
    :param img_files: a list of PIL.Image files
    :return:
    :rtype:
    """
    # https://github.com/mix1009/sdwebuiapi
    api = webuiapi.WebUIApi()
    upscale_results = api.extra_batch_images(images=img_files,
                                             upscaler_1="SwinIR_4x",
                                             upscaler_2="SwinIR_4x",
                                             extras_upscaler_2_visibility=0.65,
                                             gfpgan_visibility=0.15,
                                             upscaling_resize=2)
    upscaled_files = []
    for img in upscale_results.images:
        date_str = tops.get_date_str(mode='file')
        hash = imagehash.average_hash(img)
        filename = os.path.join(dir_out, f"{date_str}_{hash}.png")
        upscaled_files.append(filename)
        # print(img)
        img.save(filename)

    return upscaled_files


def concat_images4(input_imgs: List,
                  xy: str, xy_len: int, tag_str: str) -> (Image.Image, str):

    binary_imgs = [Image.open(i) for i in input_imgs]   # List[Image]
    tags = [i.strip() for i in tag_str.split(",")]
    # Calculate the number of rows and columns based on the xy and xy_len parameters
    if xy == 'x':
        cols = xy_len
        rows = int(math.ceil(len(binary_imgs) / cols))
    elif xy == 'y':
        rows = xy_len
        cols = int(math.ceil(len(binary_imgs) / rows))
    else:  # auto
        rows = int(math.ceil(math.sqrt(len(binary_imgs))))
        cols = int(math.ceil(len(binary_imgs) / rows))

    # Create a new blank image to hold the grid of images
    width, height = binary_imgs[0].size
    grid_width = cols * width
    grid_height = rows * height
    grid_img = Image.new(binary_imgs[0].mode, (grid_width, grid_height))

    # Draw each image onto the grid image at the appropriate position
    draw = ImageDraw.Draw(grid_img)
    for i, binary_img in enumerate(binary_imgs):
        col = i % cols
        row = i // cols
        x = col * width
        y = row * height
        grid_img.paste(binary_img, (x, y))
        if tags:
            draw.text((x, y), tags[row] if xy == 'x' else tags[col], fill=(255, 255, 255))

    return grid_img, "done"



def concat_images(input_imgs: List,
                  xy: str, xy_len: int, tag_str: str) -> (Image.Image, str):

    binary_imgs = [Image.open(i) for i in input_imgs]   # List[Image]
    tags = [i.strip() for i in tag_str.split(",")]

    # Calculate the number of rows and columns based on the xy and xy_len parameters
    if xy == 'x':
        cols = xy_len
        rows = int(math.ceil(len(binary_imgs) / cols))
    elif xy == 'y':
        rows = xy_len
        cols = int(math.ceil(len(binary_imgs) / rows))
    else:  # auto
        rows = int(math.ceil(math.sqrt(len(binary_imgs))))
        cols = int(math.ceil(len(binary_imgs) / rows))

    # Create a new blank image to hold the grid of images
    img_width, img_height = binary_imgs[0].size
    tag_width = max([len(tag) for tag in tags]) * 10
    tag_height = 50
    grid_width = cols * img_width
    grid_height = rows * img_height
    if xy == 'x':
        total_width = grid_width
        total_height = grid_height + tag_height
    else:  # y or auto
        total_width = grid_width + tag_width
        total_height = grid_height
    grid_img = Image.new('RGB', (total_width, total_height), (33, 33, 33))

    # Draw the tags onto the image
    font = ImageFont.truetype('bin/Arial-Rounded.ttf', 24)
    tag_draw = ImageDraw.Draw(grid_img)
    for i, tag in enumerate(tags):
        if xy == 'x':
            x = i * img_width
            y = grid_height
        else:  # y or auto
            x = grid_width
            y = i * img_height
        tag_draw.text((x, y), tag, fill=(200, 200, 200), font=font)

    # Draw each image onto the grid image at the appropriate position
    img_draw = ImageDraw.Draw(grid_img)
    for i, binary_img in enumerate(binary_imgs):
        col = i % cols
        row = i // cols
        x = col * img_width
        y = row * img_height
        grid_img.paste(binary_img, (x, y))

    return grid_img, "done"