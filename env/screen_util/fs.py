# env/screen_util/fs.py
import os


def safe_mkdir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass
