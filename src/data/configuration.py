import random
from pathlib import Path

import webshop_minimal.utils as webshop_utils
from webshop_minimal import init_basedir


def configure_data(data_dir: Path | None, seed: int) -> None:
    # The price cap in each goal is drawn while goals are built, so seed it for reproducibility.
    random.seed(seed)
    if data_dir is None:
        init_basedir("small")
        return

    data_dir = Path(data_dir)
    if (data_dir / "items_shuffle.json").is_file():
        init_basedir("full")
        webshop_utils.DEFAULT_FILE_PATH = str(data_dir / "items_shuffle.json")
        webshop_utils.DEFAULT_ATTR_PATH = str(data_dir / "items_ins_v2.json")
    else:
        init_basedir("small")
        webshop_utils.DEFAULT_FILE_PATH = str(data_dir / "items_shuffle_1000.json")
        webshop_utils.DEFAULT_ATTR_PATH = str(data_dir / "items_ins_v2_1000.json")
    webshop_utils.HUMAN_ATTR_PATH = str(data_dir / "items_human_ins.json")

