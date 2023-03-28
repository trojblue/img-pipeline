import gradio as gr
from lib.img import do_before_upload
from lib.utils import *
import sdtools.threadops as trops
from extensions.img_downloader.gelbooru_downloader import GelbooruDownloader

# from lib.text import make_twitter_string


def do_gelbooru_download(html, tags, out_dir, max_dl, worker_count, save_tags):
    if max_dl == -1:
        max_dl = 10**6

    downloader = GelbooruDownloader(tags, out_dir, max_dl, worker_count, save_tags)
    return_str = downloader.download_all()
    return return_str

def get_img_download_page():
    cfg = load_configs()

    html = gr.HTML(
        f"<p style='padding-bottom: 1em;' class=\"text-gray-500\">从gelbooru下载含有指定tag的图片, 也可以是artist_name:"
          f"</p>"
        )

    page = gr.Interface(
        do_gelbooru_download,
        inputs=[
            html,
            gr.Textbox(label="Tags", placeholder="mittye97, uppi, happybiirthd"),
            gr.Textbox(label="Output directory", value="D:\Andrew\Pictures\Grabber\gradio_saver"),
            gr.Number(label="Max downloads per tag (-1 for infinite)", value=2000, precision=0),
            gr.Slider(1, 20, label="Worker count", value=4),
            gr.Checkbox(label="Save tags (not working)", value=False),
        ],
        outputs=[
            gr.TextArea(placeholder="stats")
        ],
    )

    return page
