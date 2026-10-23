import random
from pathlib import Path

import webshop_minimal.utils as webshop_utils
from webshop_minimal import init_basedir


def configure_data(data_dir: Path, seed: int) -> None:
    # the price cap in each goal is drawn at random while the goals are built, so seed it to get the same goals every time
    random.seed(seed)
    init_basedir("full")
    # init_basedir only knows paths inside the submodule, so point the env at the HF data directly
    webshop_utils.DEFAULT_FILE_PATH = str(data_dir / "items_shuffle.json")
    webshop_utils.DEFAULT_ATTR_PATH = str(data_dir / "items_ins_v2.json")
    webshop_utils.HUMAN_ATTR_PATH = str(data_dir / "items_human_ins.json")


