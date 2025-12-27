# env/screen_util/debug_dump.py
import os
import time
import cv2
import numpy as np

from .fs import safe_mkdir


def dump_capture_debug(
    *,
    img_bgr: np.ndarray,
    cap_rect,
    debug_dump_dir: str,
    tag: str,
    debug_dump_annotated: bool,
    playfield_right_ratio: float,
    playfield_crops,  # (L, R, T, B) ratio
    score_roi=None,
):
    """
    현재 캡쳐 영역 확인용 덤프 저장.
    """
    safe_mkdir(debug_dump_dir)

    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(debug_dump_dir, f"capture_debug_raw_{tag}_{ts}.png")
    cv2.imwrite(raw_path, img_bgr)

    ann_path = None
    if debug_dump_annotated:
        ann = img_bgr.copy()
        H, W = ann.shape[:2]

        x_pf = int(W * playfield_right_ratio)
        cv2.line(ann, (x_pf, 0), (x_pf, H - 1), (0, 255, 255), 2)

        # playfield crop rect
        L, R, T, B = playfield_crops
        pw = x_pf
        ph = H
        x1 = int(pw * L)
        x2 = int(pw * R)
        y1 = int(ph * T)
        y2 = int(ph * B)
        cv2.rectangle(ann, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)

        if score_roi is not None:
            sx, sy, sw, sh = score_roi
            cv2.rectangle(ann, (sx, sy), (sx + sw - 1, sy + sh - 1), (255, 0, 0), 2)

        l, t, r, b = cap_rect
        txt = f"CAP_RECT client(screen) L{l} T{t} R{r} B{b} | size={W}x{H}"
        cv2.putText(ann, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        ann_path = os.path.join(debug_dump_dir, f"capture_debug_annotated_{tag}_{ts}.png")
        cv2.imwrite(ann_path, ann)

    print(f"[SCREEN][DUMP] saved: {raw_path}")
    if ann_path is not None:
        print(f"[SCREEN][DUMP] saved: {ann_path}")
