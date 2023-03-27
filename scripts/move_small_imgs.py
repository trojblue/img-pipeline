import argparse
from pathlib import Path
from sdtools.globals import *
import sdtools.fileops as fops
import sdtools.txtops as tops


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument(
        "-l",
        "--length",
        required=True,
        type=int,
        help="image's short side's minimum resolution to keep",
    )
    # parser.add_argument("-mkb" "--minimum_kbytes", type=int,required=True, help="minimum size of a image to keep")
    args = parser.parse_args()
    return args


def do_move_smaller_imgs():
    """
    移动<src_dir>里短边小于<minimum_short_side>的图片到<dst_dir>,
    如果存在同名txt也会一起移动
    :return:
    """
    # args = parse_args()

    src = "E:\==TEMP_BEFORE_MOVE\webui2\outputs\\txt2img-images\comparisons\\2023-02-26"
    output = src

    real_output = os.path.join(output, Path(output).parts[-1] + "_smaller")
    minimum_short_side = 1023
    # minimum_size = args.mkb
    minimum_size = 40  # kb

    if not minimum_size:
        minimum_size = -1

    small_imgs = fops.get_smaller_imgs(src, minimum_short_side, minimum_size)
    fops.move_files(src, real_output, small_imgs, copy=False)
    tops.get_console_msg(
        "INFO",
        f"{len(small_imgs)} images < {minimum_size} OR < {minimum_size}kb moved to {real_output}",
    )


if __name__ == "__main__":
    do_move_smaller_imgs()