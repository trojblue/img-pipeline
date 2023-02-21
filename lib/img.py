import os

import imagehash
import webuiapi
from PIL import Image

from lib.utils import load_configs
from sdtools import fileops as fops, txtops as tops


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
                                             upscaler_1=webuiapi.Upscaler.SwinIR_4x,
                                             upscaler_2=webuiapi.Upscaler.SwinIR_4x,
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
