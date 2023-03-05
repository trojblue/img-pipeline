from PIL import Image
import tomli
import os
def load_configs():
    with open ("config.toml", "rb") as t:
        cfg = tomli.load(t)

    return cfg

def load_credentials():
    # 获取保存的pixiv refresh token, 位于bin/px_token.txt

    with open ("./bin/credentials.toml", "rb") as t:
        cfg = tomli.load(t)
    return cfg

def save_img_to_config_dir(filename, img:Image):
    cfg = load_configs()
    prompt_in_dir = cfg['img_out_dir']
    filepath = os.path.join(filename, prompt_in_dir)
    img.save(filepath)

import inspect
from datetime import datetime

def get_date_str(mode: str = "file") -> str:
    """返回时间string

    :param mode:  'default' | 'file'

    mode default:
      冒号分隔, 用于debug (14:23:28)

    mode file:
      使用下划线分隔, 带年份 (23.02.09_043028)


    """
    curr_time = datetime.now()
    assert mode in ["date", "file"]

    if mode == "file":
        time_str = curr_time.strftime("%y.%m.%d_%H%M%S")

    else:  # 'default'
        time_str = curr_time.strftime("%H:%M:%S")

    return time_str

def get_console_msg(log_level:str, message:str) -> str:
    """
    打印并返回str格式的信息; 自动包含当前method名称和时间

    :param log_level: INFO | WARNING | ERROR
    :param message:  eg. empty filter list
    :return: eg. 14:23:28 [WARNING] parse_filters: empty filter list
    """
    caller_frame = inspect.currentframe().f_back
    caller_method = caller_frame.f_code.co_name
    date_str = get_date_str(mode="date")
    formatted_msg = f"{date_str} [{log_level}] {caller_method}: {message}"
    print(formatted_msg)
    return formatted_msg