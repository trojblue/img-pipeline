import random
import gradio as gr
from lib.text import make_twitter_string
from lib.utils import *
from typing import *
import sdtools.fileops as fops
import sdtools.txtops as tops


word_freq_dict, txt_lines = None, None
curr_src_dir = None
def gen_prompt(src_dir,
               start_tags: str, end_tags: str, vital_tags:str,
               tag_count: int, line_count:int, underline: bool, original_prompt:bool,  **kargs):

    global curr_src_dir, word_freq_dict, txt_lines  # 缓存

    if curr_src_dir != src_dir: #目录改变时才重新读取
        tops.get_console_msg("INFO", f"reading prompt: {src_dir}")
        word_freq_dict, txt_lines = fops.read_txt_files(src_dir)
        curr_src_dir = src_dir

    if original_prompt: # 训练集原文复制黏贴
        lines = random.choices(txt_lines, k=line_count)
        lines = [', '.join(i) for i in lines]
        # print(lines)
        info = f"{line_count} tags generated\n" \
               f"average char count: {sum([len(i) for i in lines]) / line_count}"

        return "\n".join(lines), info

    config = {
        "start_tags": tag_str_to_list(start_tags),    # 一定出现, 保证开头
        "end_tags": tag_str_to_list(end_tags),    # 一定出现, 保证结尾
        "tag_count": tag_count,
        "vital_tags": tag_str_to_list(vital_tags),   # 一定出现, 随机顺序
        "taboo_tags": [],          # 一定不出现
    }

    prompts = []
    for i in range(line_count):
        prompt_list = tops.gen_prompt_by_config(word_freq_dict, config)

        if underline:
            prompt_list = [i.replace(" ", "_") for i in prompt_list]

        prompt_str = ", ".join(prompt_list)
        prompts.append(prompt_str)

    info = f"{line_count} tags generated\n" \
           f"average char count: {sum([len(i) for i in prompts])/line_count}"

    return "\n".join(prompts), info


def tag_str_to_list(tag_str:str) -> List[str]:
    return [i.strip() for i in tag_str.split(",")]


def get_prompt_gen_page():
    cfg = load_configs()
    prompt_in_dir = cfg['prompt_in_dir']
    p_cfg = cfg['prompt']


    src_dir = gr.Textbox(label="Input directory", elem_id="img2img_batch_input_dir", value=prompt_in_dir)
    start_tags = gr.Textbox(label="Start tags", elem_id="img2img_batch_input_dir", value=p_cfg['start_tags'])
    end_tags = gr.Textbox(label="End tags", elem_id="img2img_batch_input_dir", value=p_cfg['end_tags'])
    vital_tags = gr.Textbox(label="Vital tags", elem_id="img2img_batch_input_dir", value=p_cfg['vital_tags'])

    tag_count = gr.Slider(10, 100, label="tag count", value=25, step=1)
    line_count = gr.Number(label="lines count", value=5, precision=0)

    underline = gr.Checkbox(label="Use undeline", value=True)
    original_prompt = gr.Checkbox(label="Use original prompts", value=False)


    page = gr.Interface(
        gen_prompt,

        inputs=[
            src_dir, start_tags, end_tags, vital_tags,
            tag_count, line_count,
            underline,original_prompt
        ],
        outputs=[
            gr.TextArea(label="new prompt", interactive=True),
            gr.TextArea(label="info")
        ],
    )
    return page
