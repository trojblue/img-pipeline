import gradio as gr
from lib.text import make_twitter_string
from lib.utils import *
from typing import *
from extensions.img_lookup.fetch_tags import handle_gradio_req


def do_lookup(img_dir, pixiv_token, danbooru_token, danbooru_username):

    get_console_msg("INFO", "task started")
    handle_gradio_req(img_dir, pixiv_token, danbooru_token, danbooru_username)

    return get_console_msg("INFO", f"task done: {img_dir}")



def get_img_lookup_page():

    cfg = load_configs()
    credentials = load_credentials()
    px_token = credentials['pixiv_token']
    danbooru_token = credentials['danbooru_token']
    danbooru_username = credentials['danbooru_username']

    page = gr.Interface(
        do_lookup,
        inputs=[
            gr.Text(placeholder="D:\\", label="image dir"),
            gr.Text(placeholder="EnKyQ-5m2BhPVD-nM12i6aIZLfWw1CzC2xwULOYNic4",
                    label="pixiv refresh token", value=px_token),
            gr.Text(placeholder="GQsV8623feUVa1ftbg76vfT8",
                    label="danbooru token", value=danbooru_token),
            gr.Text(placeholder="some_name",
                    label="danbooru username", value=danbooru_username),

        ],
        outputs=[
            gr.TextArea(placeholder="stats")
        ],
    )
    return page
