import yaml


def load_config(config_file: str = "config.yaml"):
    """
    加载配置文件函数。

    参数:
    config_file (str): 配置文件的路径。默认值为'config.yaml'。

    返回:
    dict: 从配置文件中加载的配置信息，以字典形式返回。
    """
    # [THAY DOI] them encoding=utf-8: config.yaml co ky tu tieng Trung,
    # tren Windows neu khong chi dinh se loi UnicodeDecodeError (cp1252).
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
