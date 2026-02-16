# env/bullet_tracker_cv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import math
import os

Pt = Tuple[float, float]  # (x,y) in ROI pixels
BBox = Tuple[int, int, int, int]  # (x,y,w,h) in ROI pixels


def ensure_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        mx = float(np.nanmax(f)) if f.size else 0.0
        if mx <= 1.5:
            f *= 255.0
        frame = np.clip(f, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


def _clip_box(box: BBox, w: int, h: int) -> Optional[BBox]:
    x, y, bw, bh = map(int, box)
    if bw <= 0 or bh <= 0:
        return None
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _expand_box(box: BBox, pad: int, w: int, h: int) -> Optional[BBox]:
    x, y, bw, bh = map(int, box)
    return _clip_box((x - pad, y - pad, bw + 2 * pad, bh + 2 * pad), w, h)


def _box_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    ua = float(max(1, aw * ah))
    ub = float(max(1, bw * bh))
    return inter / max(1e-6, (ua + ub - inter))


def _nms_boxes(boxes: List[BBox], max_iou: float = 0.35, max_keep: int = 64) -> List[BBox]:
    if not boxes:
        return []
    out: List[BBox] = []
    for b in boxes:
        keep = True
        for ob in out:
            if _box_iou(b, ob) > float(max_iou):
                keep = False
                break
        if keep:
            out.append(b)
            if len(out) >= int(max_keep):
                break
    return out


@dataclass
class BulletTrackerConfig:
    # 1) pre-mask
    use_hsv: bool = True
    hsv_v_min: int = 165
    hsv_s_min: int = 10
    hsv_s_max: int = 255
    hsv_h_min: int = 0
    hsv_h_max: int = 179

    use_white: bool = True
    white_min: int = 205

    # 2) morph
    open_ks: int = 3
    open_iter: int = 0
    dilate_iter: int = 0

    # 3) candidate filter
    area_min: int = 4
    area_max: int = 520
    w_min: int = 2
    w_max: int = 30
    h_min: int = 2
    h_max: int = 30
    split_elongated_blobs: bool = True
    elongated_aspect_min: float = 1.8
    elongated_len_min: int = 12
    elongated_step_px: float = 8.0
    elongated_max_splits: int = 8
    elongated_area_max: int = 2200

    # 4) outputs
    max_candidates: int = 256
    topk: int = 32
    debug_max_draw: int = 120

    # 5) player suppression (hard prior)
    player_margin_px: int = 4
    player_keep_ring_in: int = 3
    player_keep_ring_out: int = 9
    player_fallback_cut_rx: int = 9
    player_fallback_cut_ry: int = 13

    # 6) item suppression (template + color + ttl)
    use_item_reject: bool = True
    item_template_paths: Tuple[str, ...] = ("assets/item_black_1.png", "assets/item_blue_1.png")
    item_template_scales: Tuple[float, ...] = (0.85, 1.0, 1.15, 1.3)
    item_template_thr: float = 0.84
    item_box_expand_px: int = 2
    item_ttl_frames: int = 2
    item_max_boxes: int = 64

    # color-aided item box proposals
    item_color_use: bool = False
    item_color_min_side: int = 5
    item_color_max_aspect: float = 1.8
    item_color_min_fill: float = 0.45
    item_blue_h_min: int = 88
    item_blue_h_max: int = 140
    item_red_h1_min: int = 0
    item_red_h1_max: int = 12
    item_red_h2_min: int = 165
    item_red_h2_max: int = 179
    item_s_min: int = 45
    item_v_min: int = 35
    item_color_ratio_min: float = 0.16

    # 7) tracking-based suppression
    use_track_filter: bool = True
    track_match_max_dist: float = 14.0
    track_ttl_frames: int = 6
    track_slow_speed_thr: float = 0.20
    track_min_age_for_slow_reject: int = 5
    reject_upward_motion: bool = True
    track_upward_vy_thr: float = -0.40
    track_min_age_for_upward_reject: int = 2


class BulletTrackerCV:
    """
    Negative-first bullet tracker:
      1) raw candidate mask
      2) remove player prior region
      3) remove item regions (template+color, ttl)
      4) contour candidates
      5) track-based slow/static suppression
    """

    def __init__(self, cfg: Optional[BulletTrackerConfig] = None):
        self.cfg = cfg or BulletTrackerConfig()
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(self.cfg.open_ks), int(self.cfg.open_ks))
        )
        self.last_mask_u8: Optional[np.ndarray] = None
        self.last_points_roi: List[Pt] = []
        self.last_points_topk_roi: List[Pt] = []
        self._dbg: Dict[str, Any] = {}

        self._item_ttl: List[Dict[str, Any]] = []
        self._templates_gray: List[np.ndarray] = self._load_item_templates()
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_tid: int = 1

    def reset(self) -> None:
        self.last_mask_u8 = None
        self.last_points_roi = []
        self.last_points_topk_roi = []
        self._dbg = {}
        self._item_ttl = []
        self._tracks = {}
        self._next_tid = 1

    def _load_item_templates(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for p in self.cfg.item_template_paths:
            try:
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                if im.ndim == 3 and im.shape[2] == 4:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
                if im.ndim == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if im.dtype != np.uint8:
                    im = np.clip(im, 0, 255).astype(np.uint8)
                if im.size > 0:
                    out.append(im)
            except Exception:
                continue
        return out

    def _build_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        roi_bgr = ensure_uint8_bgr(roi_bgr)
        h, w = roi_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((1, 1), np.uint8)

        masks = []
        if self.cfg.use_hsv:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            hh, ss, vv = cv2.split(hsv)
            m_v = (vv >= int(self.cfg.hsv_v_min))
            m_s = (ss >= int(self.cfg.hsv_s_min)) & (ss <= int(self.cfg.hsv_s_max))
            m_h = (hh >= int(self.cfg.hsv_h_min)) & (hh <= int(self.cfg.hsv_h_max))
            masks.append((m_v & m_s & m_h).astype(np.uint8) * 255)

        if self.cfg.use_white:
            g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            masks.append((g >= int(self.cfg.white_min)).astype(np.uint8) * 255)

        if not masks:
            return np.zeros((h, w), np.uint8)

        mask = masks[0]
        for mm in masks[1:]:
            mask = cv2.bitwise_or(mask, mm)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel, iterations=int(self.cfg.open_iter))
        if int(self.cfg.dilate_iter) > 0:
            mask = cv2.dilate(mask, None, iterations=int(self.cfg.dilate_iter))
        return mask

    def _player_suppress_mask(
        self,
        roi_shape: Tuple[int, int],
        player_center_roi: Optional[Tuple[int, int]],
        player_bbox_roi: Optional[BBox],
    ) -> np.ndarray:
        h, w = roi_shape
        sup = np.zeros((h, w), np.uint8)
        keep = np.zeros((h, w), np.uint8)

        if player_center_roi is None:
            return sup

        px, py = map(int, player_center_roi)
        if not (0 <= px < w and 0 <= py < h):
            return sup

        if player_bbox_roi is not None:
            eb = _expand_box(player_bbox_roi, int(self.cfg.player_margin_px), w, h)
            if eb is not None:
                x, y, bw, bh = eb
                sup[y:y + bh, x:x + bw] = 255
        else:
            cv2.ellipse(
                sup,
                (px, py),
                (int(max(1, self.cfg.player_fallback_cut_rx)), int(max(1, self.cfg.player_fallback_cut_ry))),
                0.0, 0.0, 360.0, 255, thickness=-1,
            )

        rin = int(max(0, self.cfg.player_keep_ring_in))
        rout = int(max(rin + 1, self.cfg.player_keep_ring_out))
        cv2.circle(keep, (px, py), rout, 255, thickness=-1)
        cv2.circle(keep, (px, py), rin, 0, thickness=-1)

        return cv2.bitwise_and(sup, cv2.bitwise_not(keep))

    def _detect_item_boxes_template(self, roi_bgr: np.ndarray) -> List[BBox]:
        if (not self.cfg.use_item_reject) or (len(self._templates_gray) == 0):
            return []
        h, w = roi_bgr.shape[:2]
        if h <= 4 or w <= 4:
            return []

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        boxes: List[BBox] = []
        thr = float(self.cfg.item_template_thr)

        for t in self._templates_gray:
            th0, tw0 = t.shape[:2]
            if th0 < 2 or tw0 < 2:
                continue

            for sc in self.cfg.item_template_scales:
                tw = int(max(2, round(tw0 * float(sc))))
                th = int(max(2, round(th0 * float(sc))))
                if tw >= w or th >= h:
                    continue

                tt = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(gray, tt, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= thr)
                for yy, xx in zip(ys.tolist(), xs.tolist()):
                    b = _clip_box((int(xx), int(yy), int(tw), int(th)), w, h)
                    if b is not None:
                        boxes.append(b)
                        if len(boxes) >= int(self.cfg.item_max_boxes):
                            return _nms_boxes(boxes, max_iou=0.35, max_keep=int(self.cfg.item_max_boxes))

        return _nms_boxes(boxes, max_iou=0.35, max_keep=int(self.cfg.item_max_boxes))

    def _detect_item_boxes_color(self, roi_bgr: np.ndarray) -> List[BBox]:
        if (not self.cfg.use_item_reject) or (not self.cfg.item_color_use):
            return []
        h, w = roi_bgr.shape[:2]
        if h <= 4 or w <= 4:
            return []

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        sat_v = (ss >= int(self.cfg.item_s_min)) & (vv >= int(self.cfg.item_v_min))

        blue = (
            (hh >= int(self.cfg.item_blue_h_min))
            & (hh <= int(self.cfg.item_blue_h_max))
            & sat_v
        )
        red = (
            (
                (hh >= int(self.cfg.item_red_h1_min))
                & (hh <= int(self.cfg.item_red_h1_max))
            )
            | (
                (hh >= int(self.cfg.item_red_h2_min))
                & (hh <= int(self.cfg.item_red_h2_max))
            )
        ) & sat_v

        cmask = ((blue | red).astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[BBox] = []

        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if min(bw, bh) < int(self.cfg.item_color_min_side):
                continue
            asp = float(max(bw, bh)) / float(max(1, min(bw, bh)))
            if asp > float(self.cfg.item_color_max_aspect):
                continue

            c_area = float(max(0.0, cv2.contourArea(c)))
            fill = c_area / float(max(1, bw * bh))
            if fill < float(self.cfg.item_color_min_fill):
                continue

            patch_blue = blue[y:y + bh, x:x + bw]
            patch_red = red[y:y + bh, x:x + bw]
            color_ratio = max(float(np.mean(patch_blue)), float(np.mean(patch_red)))
            if color_ratio < float(self.cfg.item_color_ratio_min):
                continue

            b = _clip_box((x, y, bw, bh), w, h)
            if b is not None:
                boxes.append(b)
                if len(boxes) >= int(self.cfg.item_max_boxes):
                    break

        return _nms_boxes(boxes, max_iou=0.35, max_keep=int(self.cfg.item_max_boxes))

    def _update_item_ttl(self, new_boxes: List[BBox]) -> List[BBox]:
        ttl_max = int(max(1, self.cfg.item_ttl_frames))
        alive = []
        for r in self._item_ttl:
            r["ttl"] = int(r.get("ttl", 0)) - 1
            if int(r["ttl"]) > 0:
                alive.append(r)
        self._item_ttl = alive

        for b in new_boxes:
            hit = None
            best = 0.0
            for i, r in enumerate(self._item_ttl):
                iou = _box_iou(tuple(r["box"]), b)
                if iou > 0.30 and iou > best:
                    best = iou
                    hit = i
            if hit is None:
                self._item_ttl.append({"box": tuple(b), "ttl": ttl_max})
            else:
                self._item_ttl[hit]["box"] = tuple(b)
                self._item_ttl[hit]["ttl"] = ttl_max

        if len(self._item_ttl) > int(self.cfg.item_max_boxes):
            self._item_ttl = self._item_ttl[: int(self.cfg.item_max_boxes)]

        return [tuple(r["box"]) for r in self._item_ttl]

    def _suppress_boxes(self, mask_u8: np.ndarray, boxes: List[BBox], expand_px: int = 0) -> np.ndarray:
        if mask_u8 is None or mask_u8.size == 0 or (not boxes):
            return mask_u8
        h, w = mask_u8.shape[:2]
        out = mask_u8.copy()
        for b in boxes:
            eb = _expand_box(b, int(expand_px), w, h)
            if eb is None:
                continue
            x, y, bw, bh = eb
            out[y:y + bh, x:x + bw] = 0
        return out

    def _extract_points(self, mask_u8: np.ndarray) -> List[Pt]:
        if mask_u8 is None or mask_u8.size == 0:
            return []

        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts: List[Pt] = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area = float(bw * bh)
            is_normal_size = (
                self.cfg.w_min <= bw <= self.cfg.w_max
                and self.cfg.h_min <= bh <= self.cfg.h_max
                and float(self.cfg.area_min) <= area <= float(self.cfg.area_max)
            )
            if is_normal_size:
                pts.append((float(x + 0.5 * bw), float(y + 0.5 * bh)))
                if len(pts) >= int(self.cfg.max_candidates):
                    break
                continue

            if not bool(self.cfg.split_elongated_blobs):
                continue
            if area < float(self.cfg.area_min) or area > float(self.cfg.elongated_area_max):
                continue
            if len(c) < 5:
                continue

            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            long_side = float(max(rw, rh))
            short_side = float(max(1e-6, min(rw, rh)))
            aspect = long_side / short_side
            if aspect < float(self.cfg.elongated_aspect_min):
                continue
            if long_side < float(self.cfg.elongated_len_min):
                continue

            # Split one long blob into multiple pseudo-centers along major axis.
            # This helps when bullets are merged/connected in the binary mask.
            theta = math.radians(float(ang))
            if rw < rh:
                theta += math.pi * 0.5
            ux = math.cos(theta)
            uy = math.sin(theta)
            half = 0.5 * long_side
            step = float(max(2.0, self.cfg.elongated_step_px))
            n = int(max(2, round(long_side / step)))
            n = int(min(n, max(2, int(self.cfg.elongated_max_splits))))
            if n <= 1:
                n = 2
            for j in range(n):
                t = (j + 0.5) / float(n)
                off = -half + t * long_side
                px = float(cx + ux * off)
                py = float(cy + uy * off)
                pts.append((px, py))
                if len(pts) >= int(self.cfg.max_candidates):
                    break
            if len(pts) >= int(self.cfg.max_candidates):
                break
        return pts

    def _update_tracks(self, pts: List[Pt]) -> Dict[int, int]:
        # decay existing tracks
        dead = []
        for tid, tr in self._tracks.items():
            tr["ttl"] = int(tr.get("ttl", 0)) - 1
            if int(tr["ttl"]) <= 0:
                dead.append(tid)
        for tid in dead:
            self._tracks.pop(tid, None)

        assigned: Dict[int, int] = {}  # point idx -> track id
        used_tracks: set[int] = set()
        maxd = float(max(1.0, self.cfg.track_match_max_dist))

        for i, (x, y) in enumerate(pts):
            best_tid = None
            best_d = 1e9
            for tid, tr in self._tracks.items():
                if tid in used_tracks:
                    continue
                dx = float(x) - float(tr["x"])
                dy = float(y) - float(tr["y"])
                d = float(math.hypot(dx, dy))
                if d < best_d and d <= maxd:
                    best_d = d
                    best_tid = tid

            if best_tid is None:
                tid = int(self._next_tid)
                self._next_tid += 1
                self._tracks[tid] = {
                    "x": float(x),
                    "y": float(y),
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed": 0.0,
                    "age": 1,
                    "ttl": int(self.cfg.track_ttl_frames),
                }
                assigned[i] = tid
                used_tracks.add(tid)
            else:
                tr = self._tracks[best_tid]
                dx = float(x) - float(tr["x"])
                dy = float(y) - float(tr["y"])
                speed = float(math.hypot(dx, dy))
                tr["vx"] = 0.5 * float(tr.get("vx", 0.0)) + 0.5 * dx
                tr["vy"] = 0.5 * float(tr.get("vy", 0.0)) + 0.5 * dy
                tr["speed"] = 0.5 * float(tr.get("speed", 0.0)) + 0.5 * speed
                tr["x"] = float(x)
                tr["y"] = float(y)
                tr["age"] = int(tr.get("age", 1)) + 1
                tr["ttl"] = int(self.cfg.track_ttl_frames)
                assigned[i] = best_tid
                used_tracks.add(best_tid)

        return assigned

    def _filter_by_tracks(self, pts: List[Pt]) -> tuple[List[Pt], int, int]:
        if (not self.cfg.use_track_filter) or (not pts):
            return pts, 0, 0
        map_idx_tid = self._update_tracks(pts)
        out: List[Pt] = []
        rej_slow = 0
        rej_up = 0
        sp_thr = float(self.cfg.track_slow_speed_thr)
        age_thr = int(self.cfg.track_min_age_for_slow_reject)
        use_up = bool(self.cfg.reject_upward_motion)
        up_vy_thr = float(self.cfg.track_upward_vy_thr)
        up_age_thr = int(self.cfg.track_min_age_for_upward_reject)

        for i, p in enumerate(pts):
            tid = map_idx_tid.get(i, None)
            if tid is None:
                out.append(p)
                continue
            tr = self._tracks.get(tid, None)
            if tr is None:
                out.append(p)
                continue
            age = int(tr.get("age", 1))
            speed = float(tr.get("speed", 0.0))
            vy = float(tr.get("vy", 0.0))
            if use_up and age >= up_age_thr and vy <= up_vy_thr:
                rej_up += 1
                continue
            if age >= age_thr and speed < sp_thr:
                rej_slow += 1
                continue
            out.append(p)
        return out, int(rej_slow), int(rej_up)

    def step(
        self,
        roi_bgr: np.ndarray,
        player_center_roi: Optional[Tuple[int, int]] = None,
        player_bbox_roi: Optional[BBox] = None,
    ) -> List[Pt]:
        roi_bgr = ensure_uint8_bgr(roi_bgr)
        h, w = roi_bgr.shape[:2]

        # 1) raw mask
        raw_mask = self._build_mask(roi_bgr)
        mask = raw_mask.copy()

        # 2) subtract player prior
        p_sup = self._player_suppress_mask((h, w), player_center_roi, player_bbox_roi)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(p_sup))

        # 3) subtract item regions (template + color + ttl)
        item_boxes_det: List[BBox] = []
        if self.cfg.use_item_reject:
            item_boxes_det.extend(self._detect_item_boxes_template(roi_bgr))
            item_boxes_det.extend(self._detect_item_boxes_color(roi_bgr))
            item_boxes_det = _nms_boxes(item_boxes_det, max_iou=0.35, max_keep=int(self.cfg.item_max_boxes))
        item_boxes_alive = self._update_item_ttl(item_boxes_det)
        mask = self._suppress_boxes(mask, item_boxes_alive, expand_px=int(self.cfg.item_box_expand_px))

        # 4) contour candidates
        pts = self._extract_points(mask)

        # 5) track-based suppression (slow/static artifacts)
        pts, rej_track_slow, rej_track_up = self._filter_by_tracks(pts)

        # 6) top-k nearest to player (fallback: first K)
        K = int(max(1, self.cfg.topk))
        if player_center_roi is not None and pts:
            px, py = map(float, player_center_roi)
            scored = []
            for (x, y) in pts:
                dx = x - px
                dy = y - py
                d2 = dx * dx + dy * dy
                ang = math.atan2(dy, dx)
                scored.append((d2, ang, (x, y)))
            scored.sort(key=lambda t: (t[0], t[1]))
            topk = [p for _, _, p in scored[:K]]
        else:
            topk = pts[:K]

        self.last_mask_u8 = mask
        self.last_points_roi = pts
        self.last_points_topk_roi = topk
        self._dbg = {
            "n": int(len(pts)),
            "topk": int(len(topk)),
            "points": pts[: int(self.cfg.debug_max_draw)],
            "points_topk": topk,
            "item_boxes": [tuple(map(int, b)) for b in item_boxes_alive[: int(self.cfg.debug_max_draw)]],
            "n_item_boxes": int(len(item_boxes_alive)),
            "reject_track_slow": int(rej_track_slow),
            "reject_track_upward": int(rej_track_up),
            "reject_track_total": int(rej_track_slow + rej_track_up),
            "player_center_roi": player_center_roi,
            "player_bbox_roi": player_bbox_roi,
            "roi_shape": tuple(map(int, roi_bgr.shape[:2])),
            "K": K,
        }
        return topk

    def get_debug(self) -> Dict[str, Any]:
        return self._dbg or {}
