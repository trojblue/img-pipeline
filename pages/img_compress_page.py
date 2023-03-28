import gradio as gr
from lib.img import do_before_upload
from lib.utils import *
import sdtools.threadops as trops

# from lib.text import make_twitter_string

def do_resize(html, src_dir, dst_dir, target_res, inplace):

    real_dst_dir = src_dir if inplace else dst_dir
    return_str = trops.do_resize_images_tr(src_dir, real_dst_dir, target_res)
    return return_str


def get_img_compress_page():
    cfg = load_configs()

    html = gr.HTML(
        f"<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running."
        + f"<br>输入图片目录, 把目录内所有图片压缩到指定大小, 保存到输出目录."
        + f"<br>inplace: 压缩后的图片替换原图. (忽略输出目录)"
          f"</p>"
        )

    page = gr.Interface(
        do_resize,
        inputs=[
            html,
            gr.Textbox(label="Input directory", placeholder="D:/train/in"),
            gr.Textbox(label="Output directory", ),
            gr.Slider(0, 2000, value=1024, label="target_res"),
            gr.Checkbox(label="inplace", value=False),
        ],
        outputs=[
            gr.TextArea(placeholder="stats")
        ],
    )

    return page
