"""Lets pyserini be imported without internet, as on the HPC compute nodes.

Importing pyserini downloads catalog files of its prebuilt indexes from GitHub (pyserini/prebuilt_index_info.py).
We never use those indexes, but the import fails without the files. This serves them from a local copy instead,
saving each one the first time it is fetched with internet access.

Import this module before anything that imports pyserini.
"""
import io
import urllib.request
from pathlib import Path

CATALOG_URL_PREFIX = "https://raw.githubusercontent.com/castorini/prebuilt-indexes/"
CATALOG_DIR = Path(__file__).parent / "data" / "pyserini-catalog"

_urlopen = urllib.request.urlopen


def _cached_urlopen(url, *args, **kwargs):
    if isinstance(url, str) and url.startswith(CATALOG_URL_PREFIX):
        path = CATALOG_DIR / url.removeprefix(CATALOG_URL_PREFIX)
        if not path.exists():
            with _urlopen(url, *args, **kwargs) as response:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.read())
        return io.BytesIO(path.read_bytes())
    return _urlopen(url, *args, **kwargs)


urllib.request.urlopen = _cached_urlopen  # pyserini binds urlopen when it is imported, so this must run first
