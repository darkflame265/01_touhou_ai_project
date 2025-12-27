# env/screen_util/assets.py
import os
import json
import cv2


def load_ui_config(env_dir: str):
    """
    env/ui_config.json 로드해서 필요한 값만 꺼내줌.
    return: dict(score_roi, debug_dump_on_start, debug_dump_dir, debug_dump_annotated)
    """
    out = {
        "score_roi": None,
        "capture_debug_dump_on_start": False,
        "capture_debug_dump_dir": None,
        "capture_debug_dump_annotated": True,
    }

    try:
        cfg_path = os.path.join(env_dir, "ui_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            if "score_roi" in cfg:
                r = cfg["score_roi"]
                out["score_roi"] = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))

            if "capture_debug_dump_on_start" in cfg:
                out["capture_debug_dump_on_start"] = bool(cfg["capture_debug_dump_on_start"])

            if "capture_debug_dump_dir" in cfg and isinstance(cfg["capture_debug_dump_dir"], str):
                out["capture_debug_dump_dir"] = os.path.normpath(cfg["capture_debug_dump_dir"])

            if "capture_debug_dump_annotated" in cfg:
                out["capture_debug_dump_annotated"] = bool(cfg["capture_debug_dump_annotated"])

    except Exception as e:
        print("[DEBUG] ui_config load failed:", repr(e))

    return out


def load_score_template(env_dir: str):
    """
    assets/score_template.png (optional) 로드.
    return: gray template ndarray or None
    """
    try:
        tmpl_path = os.path.normpath(os.path.join(env_dir, "..", "assets", "score_template.png"))
        if os.path.exists(tmpl_path):
            g = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            if g is not None and g.size > 0:
                print("[DEBUG] score template loaded:", tmpl_path, "shape=", g.shape)
                return g
            print("[DEBUG] score template exists but failed to read:", tmpl_path)
        else:
            print("[DEBUG] score template not found (optional):", tmpl_path)
    except Exception as e:
        print("[DEBUG] score template load failed:", repr(e))
    return None
