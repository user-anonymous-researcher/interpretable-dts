"""
    Utilities to manage files
"""
from pathlib import Path


def create_dir(filename):
    """
    Create dir if not exists
    """
    dir_index = filename.rfind("/")
    dir_name = filename[:dir_index]
    Path(dir_name).mkdir(parents=True, exist_ok=True)
