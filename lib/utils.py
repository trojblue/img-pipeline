import tomli
def load_configs():
    with open ("config.toml", "rb") as t:
        cfg = tomli.load(t)

    return cfg