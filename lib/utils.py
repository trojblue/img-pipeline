from PIL import Image
import tomli
import os
def load_configs():
    with open ("config.toml", "rb") as t:
        cfg = tomli.load(t)

    return cfg



def save_img_to_config_dir(filename, img:Image):
    cfg = load_configs()
    prompt_in_dir = cfg['img_out_dir']
    filepath = os.path.join(filename, prompt_in_dir)
    img.save(filepath)

