from typing import *
from pixivpy3 import AppPixivAPI
import copy
import toml

dummy_illust_json = {"id": -1, "type": "illust", "user": {}, "tags": {}}


def get_dummy_pixiv_dict(j: Dict):
    """输入danbooru字典, 返回dummy pixiv字典
    新字典的id = danbooru id +
    """
    return_dict = copy.deepcopy(dummy_illust_json)
    dummy_danbooru_prefix = "247800500"  # :DANBOOROO + danbooru id
    new_id = int(dummy_danbooru_prefix + str(j["id"]))
    return_dict["id"] = new_id
    return_dict["danbooru"] = j
    return return_dict


def build_pixiv_dict_from_j(j: Dict, api):
    """查找对应pixiv illust页面; 返回px数据集格式的illust json:
    {
        'id': int
        .....
        'danbooru':{
            ....
        }
    }
    """
    pixiv_id = j.get("pixiv_id", None)

    if pixiv_id:
        illust_json = api.illust_detail(str(pixiv_id)).get("illust", None)
        if illust_json:
            illust_json["danbooru"] = j
            return illust_json

    # no pixiv id or pixiv info not found
    return get_dummy_pixiv_dict(j)
