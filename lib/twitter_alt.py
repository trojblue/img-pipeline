try:
    import lib.bin_loader as loader
except ModuleNotFoundError:
    import bin_loader as loader

from itertools import chain


def make_twitter_string(
    prompt: str, nomodel, noartist, noseed, shorter, even_shorter
) -> tuple[str, str]:
    """输入sd png info的prompt, (尝试)输出1000字以内的解法"""
    str_lines = prompt.splitlines()
    c_str = str_lines[0]
    uc_str = str_lines[1][len("Negative prompt: ") :]
    param_str = str_lines[2]

    params = param_str.split(", ")
    if nomodel:
        params = [p for p in params if not p.startswith("Model: ")]
    if noseed:
        params = [p for p in params if not p.startswith("Seed: ")]

    cs = c_str.split(", ")
    if noartist:
        artists = loader.load_artists()
        artists_str_lists = [i[:2] for i in artists]  # danbooru and token name
        flatten_list = list(chain.from_iterable(artists_str_lists))
        cs = [i for i in cs if i not in flatten_list]

    try_assemble = (
        f"{', '.join(cs)}\n\n" f"Negative prompt: {uc_str}\n\n" f"{', '.join(params)}"
    )

    stats = f"original length: {len(prompt)}\n" f"new length: {len(try_assemble)}"
    return try_assemble, stats
    # print("D")


def get_sample_txt():
    with open("sample_text.txt", "r") as f:
        text = f.readlines()
        return "".join(text)


if __name__ == "__main__":
    text_str = get_sample_txt()

    # make_twitter_string(text_str)
