# env/bullet_tracker_cv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import math

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
    hsv_v_min: int = 150
    hsv_s_min: int = 10
    hsv_s_max: int = 255
    hsv_h_min: int = 0
    hsv_h_max: int = 179

    use_white: bool = True
    white_min: int = 195

    # 2) morph
    open_ks: int = 3
    open_iter: int = 0
    dilate_iter: int = 0

    # 3) candidate filter
    area_min: int = 4
    area_max: int = 900
    w_min: int = 2
    w_max: int = 44
    h_min: int = 2
    h_max: int = 44
    split_merged_blobs: bool = True
    merged_blob_area_min: int = 180
    merged_blob_area_max: int = 3200
    merged_peak_min_dt: float = 1.4
    merged_peak_min_dist: int = 5
    merged_peak_max_points: int = 10
    adaptive_pointcloud: bool = True
    pointcloud_area_per_point: float = 120.0
    pointcloud_min_points_complex: int = 2
    pointcloud_max_points_per_blob: int = 12
    pointcloud_aspect_complex: float = 1.65
    pointcloud_contour_area_min: float = 6.0

    # 4) outputs
    max_candidates: int = 384
    topk: int = 32
    debug_max_draw: int = 120
    debug_grid_size: int = 64
    debug_grid_gain: float = 0.8
    debug_grid_use_final_points: bool = True
    debug_grid_point_radius_cells: int = 1
    debug_grid_suppress_player: bool = False
    debug_grid_player_rx_scale: float = 0.48
    debug_grid_player_ry_scale: float = 0.56
    debug_grid_player_fallback_rx: int = 8
    debug_grid_player_fallback_ry: int = 10
    debug_grid_player_donut_enable: bool = False
    debug_grid_player_donut_mode: str = "clear"   # "clear" | "gaussian"
    debug_grid_player_donut_r_inner: float = 1.2
    debug_grid_player_donut_r_outer: float = 3.2
    debug_grid_player_donut_sigma: float = 0.9
    debug_grid_player_donut_strength: float = 1.0
    # new: reject player sprite from hitbox center (independent of old clear/gaussian)
    player_sprite_reject_enable: bool = False
    player_sprite_rx: int = 14
    player_sprite_ry: int = 20
    player_sprite_center_y_offset: int = -3
    player_sprite_extra_top: int = 6
    player_sprite_extra_bottom: int = 6

    # 5) item suppression (template + color + ttl)
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

    # 6) tracking-based suppression
    use_track_filter: bool = True
    track_match_max_dist: float = 14.0
    track_ttl_frames: int = 6
    track_slow_speed_thr: float = 0.20
    track_min_age_for_slow_reject: int = 5
    reject_upward_motion: bool = False
    track_upward_vy_thr: float = -0.40
    track_min_age_for_upward_reject: int = 2
    use_near_player_shape_filter: bool = True
    near_player_radius_px: float = 24.0
    near_player_min_circularity: float = 0.56
    near_player_max_aspect: float = 2.10
    near_player_min_area: float = 4.0
    near_player_max_area: float = 200.0
    # simple near-player new-track rejection
    reject_new_near_player: bool = True
    new_near_player_radius_px: float = 40.0
    new_near_player_max_age: int = 6
    new_near_player_enter_margin_px: float = 3.0
    # short near-player hold memory (no vector prediction)
    hold_near_player_on_miss: bool = True
    hold_near_player_radius_px: float = 40.0
    hold_near_player_frames: int = 3
    hold_match_dist_px: float = 10.0
    hold_dedup_dist_px: float = 6.0



class BulletTrackerCV:
    """
    Negative-first bullet tracker:
      1) raw candidate mask
      2) remove item regions (template+color, ttl)
      3) contour candidates
      4) track-based slow/static suppression
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
        self._near_hold: List[Dict[str, Any]] = []
        g = int(max(2, self.cfg.debug_grid_size))
        self._prev_dbg_occ = np.zeros((g, g), dtype=np.float32)

    def reset(self) -> None:
        self.last_mask_u8 = None
        self.last_points_roi = []
        self.last_points_topk_roi = []
        self._dbg = {}
        self._item_ttl = []
        self._tracks = {}
        self._next_tid = 1
        self._near_hold = []
        g = int(max(2, self.cfg.debug_grid_size))
        self._prev_dbg_occ = np.zeros((g, g), dtype=np.float32)

    def _load_gray_templates(self, paths: Tuple[str, ...]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for p in paths:
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

    def _load_item_templates(self) -> List[np.ndarray]:
        return self._load_gray_templates(self.cfg.item_template_paths)

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

    def _extract_points(
        self,
        mask_u8: np.ndarray,
        player_center_roi: Optional[Tuple[int, int]] = None,
    ) -> List[Pt]:
        if mask_u8 is None or mask_u8.size == 0:
            return []

        def _is_near_player_non_bullet(
            center_xy: Pt,
            area_wh: float,
            bw: int,
            bh: int,
            contour: np.ndarray,
        ) -> bool:
            if (not self.cfg.use_near_player_shape_filter) or (player_center_roi is None):
                return False
            px, py = map(float, player_center_roi)
            x, y = center_xy
            dx = float(x) - px
            dy = float(y) - py
            rr = float(max(1.0, self.cfg.near_player_radius_px))
            if dx * dx + dy * dy > rr * rr:
                return False

            asp = float(max(bw, bh)) / float(max(1, min(bw, bh)))
            peri = float(cv2.arcLength(contour, True))
            c_area = float(max(0.0, cv2.contourArea(contour)))
            circ = 0.0 if peri <= 1e-6 else float((4.0 * math.pi * c_area) / (peri * peri))
            amin = float(self.cfg.near_player_min_area)
            amax = float(self.cfg.near_player_max_area)
            return (
                asp > float(self.cfg.near_player_max_aspect)
                or circ < float(self.cfg.near_player_min_circularity)
                or area_wh < amin
                or area_wh > amax
            )

        def _peaks_from_big_component(comp_mask_u8: np.ndarray, ox: int, oy: int) -> List[Pt]:
            # Distance-transform peaks approximate individual bullet centers in merged blobs.
            dt = cv2.distanceTransform((comp_mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 3)
            if dt is None or dt.size == 0:
                return []
            thr = float(max(0.5, self.cfg.merged_peak_min_dt))
            peak_mask = (dt >= thr)
            local_max = (dt >= cv2.dilate(dt, np.ones((3, 3), np.float32)))
            ys, xs = np.where(peak_mask & local_max)
            if len(xs) == 0:
                return []

            cand = sorted(
                [(float(dt[int(y), int(x)]), int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())],
                key=lambda t: t[0],
                reverse=True,
            )
            outp: List[Pt] = []
            min_d2 = float(max(1, int(self.cfg.merged_peak_min_dist)) ** 2)
            max_pts = int(max(1, self.cfg.merged_peak_max_points))
            for _, x, y in cand:
                keep = True
                for px, py in outp:
                    dx = (float(ox + x) - px)
                    dy = (float(oy + y) - py)
                    if dx * dx + dy * dy < min_d2:
                        keep = False
                        break
                if keep:
                    outp.append((float(ox + x), float(oy + y)))
                    if len(outp) >= max_pts:
                        break
            return outp

        def _sample_points_from_contour(c: np.ndarray, x: int, y: int, bw: int, bh: int) -> List[Pt]:
            # Shape-agnostic point cloud:
            # complex/elongated blobs -> multiple points, compact blobs -> single center.
            c_area = float(max(0.0, cv2.contourArea(c)))
            if c_area <= 0.0:
                return []

            asp = float(max(bw, bh)) / float(max(1, min(bw, bh)))
            is_complex = (
                bool(self.cfg.adaptive_pointcloud)
                and (
                    asp >= float(self.cfg.pointcloud_aspect_complex)
                    or c_area >= float(self.cfg.merged_blob_area_min)
                )
            )
            if not is_complex:
                return [(float(x + 0.5 * bw), float(y + 0.5 * bh))]

            n_area = int(max(1, round(c_area / float(max(1.0, self.cfg.pointcloud_area_per_point)))))
            n = int(max(int(self.cfg.pointcloud_min_points_complex), n_area))
            n = int(min(n, max(1, int(self.cfg.pointcloud_max_points_per_blob))))

            comp = np.zeros((bh, bw), np.uint8)
            c_local = c - np.array([[[x, y]]], dtype=c.dtype)
            cv2.drawContours(comp, [c_local], -1, 255, thickness=-1)
            peaks = _peaks_from_big_component(comp, x, y)

            if len(peaks) >= n:
                return peaks[:n]
            if len(peaks) > 0:
                return peaks

            # Fallback: evenly sample along major axis.
            if len(c) >= 5:
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
                theta = math.radians(float(ang))
                if rw < rh:
                    theta += math.pi * 0.5
                ux = math.cos(theta)
                uy = math.sin(theta)
                long_side = float(max(rw, rh))
                half = 0.5 * long_side
                outp: List[Pt] = []
                for j in range(max(1, n)):
                    t = (j + 0.5) / float(max(1, n))
                    off = -half + t * long_side
                    outp.append((float(cx + ux * off), float(cy + uy * off)))
                return outp

            return [(float(x + 0.5 * bw), float(y + 0.5 * bh))]

        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts: List[Pt] = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            c_area = float(max(0.0, cv2.contourArea(c)))
            area = float(bw * bh)
            if c_area < float(self.cfg.pointcloud_contour_area_min):
                continue
            is_normal = (
                self.cfg.w_min <= bw <= self.cfg.w_max
                and self.cfg.h_min <= bh <= self.cfg.h_max
                and float(self.cfg.area_min) <= area <= float(self.cfg.area_max)
            )
            if is_normal:
                sampled = _sample_points_from_contour(c, x, y, bw, bh)
                for sp in sampled:
                    if _is_near_player_non_bullet(sp, area, bw, bh, c):
                        continue
                    pts.append(sp)
                    if len(pts) >= int(self.cfg.max_candidates):
                        break
                if len(pts) >= int(self.cfg.max_candidates):
                    break
                continue

            if not bool(self.cfg.split_merged_blobs):
                continue
            if not (float(self.cfg.merged_blob_area_min) <= area <= float(self.cfg.merged_blob_area_max)):
                continue
            if bw < int(self.cfg.w_min) or bh < int(self.cfg.h_min):
                continue

            split_pts = _sample_points_from_contour(c, x, y, bw, bh)
            if not split_pts:
                continue
            if player_center_roi is not None and self.cfg.use_near_player_shape_filter:
                kept_split: List[Pt] = []
                for sp in split_pts:
                    if not _is_near_player_non_bullet(sp, area, bw, bh, c):
                        kept_split.append(sp)
                split_pts = kept_split
                if not split_pts:
                    continue
            pts.extend(split_pts)
            if len(pts) >= int(self.cfg.max_candidates):
                pts = pts[: int(self.cfg.max_candidates)]
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
                    "prev_x": None,
                    "prev_y": None,
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
                px_prev = float(tr.get("x", x))
                py_prev = float(tr.get("y", y))
                dx = float(x) - float(tr["x"])
                dy = float(y) - float(tr["y"])
                speed = float(math.hypot(dx, dy))
                tr["prev_x"] = px_prev
                tr["prev_y"] = py_prev
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

    def _filter_by_tracks(
        self,
        pts: List[Pt],
        player_center_roi: Optional[Tuple[int, int]] = None,
    ) -> tuple[List[Pt], int, int, int]:
        if not self.cfg.use_track_filter:
            return pts, 0, 0, 0
        map_idx_tid = self._update_tracks(pts)
        out: List[Pt] = []
        rej_slow = 0
        rej_up = 0
        rej_new_near = 0
        sp_thr = float(self.cfg.track_slow_speed_thr)
        age_thr = int(self.cfg.track_min_age_for_slow_reject)
        use_up = bool(self.cfg.reject_upward_motion)
        up_vy_thr = float(self.cfg.track_upward_vy_thr)
        up_age_thr = int(self.cfg.track_min_age_for_upward_reject)
        use_new_near = bool(self.cfg.reject_new_near_player)
        near_r = float(max(1.0, self.cfg.new_near_player_radius_px))
        near_r2 = near_r * near_r
        enter_margin = float(max(0.0, self.cfg.new_near_player_enter_margin_px))
        near_outer2 = (near_r + enter_margin) * (near_r + enter_margin)
        near_age = int(max(1, self.cfg.new_near_player_max_age))
        pcx = pcy = None
        if player_center_roi is not None:
            pcx, pcy = map(float, player_center_roi)

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
            if pcx is not None and pcy is not None and use_new_near:
                dxp = float(p[0]) - pcx
                dyp = float(p[1]) - pcy
                d2p = dxp * dxp + dyp * dyp
                if age <= near_age and d2p <= near_r2:
                    entered_from_outside = False
                    px0 = tr.get("prev_x", None)
                    py0 = tr.get("prev_y", None)
                    if px0 is not None and py0 is not None:
                        ddx = float(px0) - pcx
                        ddy = float(py0) - pcy
                        prev_d2 = ddx * ddx + ddy * ddy
                        if prev_d2 > near_outer2:
                            entered_from_outside = True
                    if entered_from_outside:
                        out.append(p)
                        continue
                    rej_new_near += 1
                    continue
            if use_up and age >= up_age_thr and vy <= up_vy_thr:
                rej_up += 1
                continue
            if age >= age_thr and speed < sp_thr:
                rej_slow += 1
                continue
            out.append(p)
        return out, int(rej_slow), int(rej_up), int(rej_new_near)

    def _update_near_hold(
        self,
        pts: List[Pt],
        player_center_roi: Optional[Tuple[int, int]],
    ) -> List[Pt]:
        if not bool(self.cfg.hold_near_player_on_miss):
            self._near_hold = []
            return []
        if player_center_roi is None:
            # No anchor point; decay existing entries only.
            alive = []
            for r in self._near_hold:
                r["ttl"] = int(r.get("ttl", 0)) - 1
                if int(r["ttl"]) > 0:
                    alive.append(r)
            self._near_hold = alive
            return [(
                float(r.get("x", 0.0)),
                float(r.get("y", 0.0)),
            ) for r in self._near_hold]

        px, py = map(float, player_center_roi)
        near_r = float(max(1.0, self.cfg.hold_near_player_radius_px))
        near_r2 = near_r * near_r
        match_d = float(max(1.0, self.cfg.hold_match_dist_px))
        match_d2 = match_d * match_d
        ttl_full = int(max(1, self.cfg.hold_near_player_frames))

        # decay existing entries
        alive = []
        for r in self._near_hold:
            r["ttl"] = int(r.get("ttl", 0)) - 1
            if int(r["ttl"]) > 0:
                alive.append(r)
        self._near_hold = alive

        near_pts: List[Pt] = []
        for (x, y) in pts:
            dx = float(x) - px
            dy = float(y) - py
            if dx * dx + dy * dy <= near_r2:
                near_pts.append((float(x), float(y)))

        # match near detected points to hold entries
        used = set()
        for (x, y) in near_pts:
            best_i = -1
            best_d = 1e18
            for i, r in enumerate(self._near_hold):
                if i in used:
                    continue
                dx = float(x) - float(r.get("x", 0.0))
                dy = float(y) - float(r.get("y", 0.0))
                d2 = dx * dx + dy * dy
                if d2 <= match_d2 and d2 < best_d:
                    best_d = d2
                    best_i = i
            if best_i >= 0:
                rr = self._near_hold[best_i]
                rr["x"] = float(x)
                rr["y"] = float(y)
                rr["ttl"] = ttl_full
                used.add(best_i)
            else:
                self._near_hold.append({"x": float(x), "y": float(y), "ttl": ttl_full})

        return [(
            float(r.get("x", 0.0)),
            float(r.get("y", 0.0)),
        ) for r in self._near_hold]

    def _player_core_mask_for_grid(
        self,
        roi_shape: Tuple[int, int],
        player_center_roi: Optional[Tuple[int, int]],
        player_bbox_roi: Optional[BBox],
    ) -> np.ndarray:
        h, w = roi_shape
        m = np.zeros((h, w), np.uint8)
        if (not bool(self.cfg.debug_grid_suppress_player)) or (player_center_roi is None):
            return m

        px, py = map(int, player_center_roi)
        if not (0 <= px < w and 0 <= py < h):
            return m

        if player_bbox_roi is not None:
            bx, by, bw, bh = map(int, player_bbox_roi)
            cx = int(bx + 0.5 * bw)
            cy = int(by + 0.5 * bh)
            rx = int(max(2, round(float(bw) * float(self.cfg.debug_grid_player_rx_scale))))
            ry = int(max(2, round(float(bh) * float(self.cfg.debug_grid_player_ry_scale))))
            cv2.ellipse(m, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, thickness=-1)
        else:
            rx = int(max(1, self.cfg.debug_grid_player_fallback_rx))
            ry = int(max(1, self.cfg.debug_grid_player_fallback_ry))
            cv2.ellipse(m, (px, py), (rx, ry), 0.0, 0.0, 360.0, 255, thickness=-1)
        return m

    def _apply_player_donut_on_grid(
        self,
        occ: np.ndarray,
        player_center_roi: Optional[Tuple[int, int]],
        roi_shape: Tuple[int, int],
    ) -> np.ndarray:
        if (not bool(self.cfg.debug_grid_player_donut_enable)) or (player_center_roi is None):
            return occ
        if occ is None or occ.size == 0 or occ.ndim != 2:
            return occ

        h, w = roi_shape
        if h <= 1 or w <= 1:
            return occ

        px, py = map(float, player_center_roi)
        gh, gw = occ.shape[:2]
        cx = float(np.clip(px * (gw - 1) / float(max(1, w - 1)), 0.0, float(gw - 1)))
        cy = float(np.clip(py * (gh - 1) / float(max(1, h - 1)), 0.0, float(gh - 1)))

        yy, xx = np.indices((gh, gw), dtype=np.float32)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        rin = float(max(0.0, self.cfg.debug_grid_player_donut_r_inner))
        rout = float(max(rin + 1e-6, self.cfg.debug_grid_player_donut_r_outer))
        donut = (rr >= rin) & (rr <= rout)
        if not np.any(donut):
            return occ

        out = occ.astype(np.float32, copy=True)
        mode = str(self.cfg.debug_grid_player_donut_mode).strip().lower()
        if mode == "gaussian":
            sigma = float(max(1e-6, self.cfg.debug_grid_player_donut_sigma))
            strength = float(np.clip(self.cfg.debug_grid_player_donut_strength, 0.0, 1.0))
            # Attenuate strongest near inner boundary, weaker toward outer boundary.
            att = np.exp(-0.5 * ((rr - rin) / sigma) ** 2).astype(np.float32)
            factor = np.clip(1.0 - strength * att, 0.0, 1.0)
            out[donut] *= factor[donut]
        else:
            out[donut] = 0.0
        return out

    def _player_sprite_mask_from_hitbox(
        self,
        roi_shape: Tuple[int, int],
        player_center_roi: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        h, w = roi_shape
        m = np.zeros((h, w), np.uint8)
        if (not bool(self.cfg.player_sprite_reject_enable)) or (player_center_roi is None):
            return m

        px, py = map(int, player_center_roi)
        if not (0 <= px < w and 0 <= py < h):
            return m

        rx = int(max(1, self.cfg.player_sprite_rx))
        ry = int(max(1, self.cfg.player_sprite_ry))
        cy = int(py + int(self.cfg.player_sprite_center_y_offset))
        cv2.ellipse(m, (px, cy), (rx, ry), 0.0, 0.0, 360.0, 255, thickness=-1)

        top = int(max(0, self.cfg.player_sprite_extra_top))
        bot = int(max(0, self.cfg.player_sprite_extra_bottom))
        y1 = int(np.clip(cy - ry - top, 0, h - 1))
        y2 = int(np.clip(cy + ry + bot, y1 + 1, h))
        x1 = int(np.clip(px - rx, 0, w - 1))
        x2 = int(np.clip(px + rx + 1, x1 + 1, w))
        m[y1:y2, x1:x2] = 255
        return m

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
        if bool(self.cfg.player_sprite_reject_enable):
            spm = self._player_sprite_mask_from_hitbox((h, w), player_center_roi)
            if spm is not None and spm.size > 0:
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(spm))

        # 2) subtract item regions (template + color + ttl)
        item_boxes_det: List[BBox] = []
        if self.cfg.use_item_reject:
            item_boxes_det.extend(self._detect_item_boxes_template(roi_bgr))
            item_boxes_det.extend(self._detect_item_boxes_color(roi_bgr))
            item_boxes_det = _nms_boxes(item_boxes_det, max_iou=0.35, max_keep=int(self.cfg.item_max_boxes))
        item_boxes_alive = self._update_item_ttl(item_boxes_det)
        mask = self._suppress_boxes(mask, item_boxes_alive, expand_px=int(self.cfg.item_box_expand_px))

        # 3) contour candidates
        pts = self._extract_points(mask, player_center_roi=player_center_roi)

        # 4) track-based suppression (slow/static artifacts)
        pts, rej_track_slow, rej_track_up, rej_new_near = self._filter_by_tracks(
            pts,
            player_center_roi=player_center_roi,
        )
        held_pts = self._update_near_hold(pts, player_center_roi=player_center_roi)
        if held_pts:
            dd = float(max(1.0, self.cfg.hold_dedup_dist_px))
            dd2 = dd * dd
            merged = list(pts)
            for hx, hy in held_pts:
                dup = False
                for x, y in pts:
                    dx = float(hx) - float(x)
                    dy = float(hy) - float(y)
                    if dx * dx + dy * dy <= dd2:
                        dup = True
                        break
                if not dup:
                    merged.append((float(hx), float(hy)))
            pts = merged

        # 5) top-k nearest to player (fallback: first K)
        K = int(max(1, self.cfg.topk))
        if player_center_roi is not None and pts:
            px, py = map(float, player_center_roi)
            scored = []
            for (x, y) in pts:
                dx = x - px
                dy = y - py
                d2 = dx * dx + dy * dy
                scored.append((d2, (x, y)))
            scored.sort(key=lambda t: t[0])
            topk = [p for _, p in scored[:K]]
        else:
            topk = pts[:K]

        # Debug spatial grids (ID-free): occupancy + temporal delta
        g = int(max(2, self.cfg.debug_grid_size))
        if bool(self.cfg.debug_grid_use_final_points):
            occ_small = np.zeros((g, g), dtype=np.float32)
            r = int(max(0, self.cfg.debug_grid_point_radius_cells))
            for (x, y) in pts:
                gx = int(np.clip(round(float(x) * (g - 1) / max(1.0, float(w - 1))), 0, g - 1))
                gy = int(np.clip(round(float(y) * (g - 1) / max(1.0, float(h - 1))), 0, g - 1))
                x1 = max(0, gx - r)
                x2 = min(g, gx + r + 1)
                y1 = max(0, gy - r)
                y2 = min(g, gy + r + 1)
                occ_small[y1:y2, x1:x2] = 1.0
        else:
            # Legacy debug mode: raw mask coverage
            grid_mask = mask
            if mask is not None and mask.size > 0 and bool(self.cfg.debug_grid_suppress_player):
                pcore = self._player_core_mask_for_grid((h, w), player_center_roi, player_bbox_roi)
                if pcore is not None and pcore.size > 0:
                    grid_mask = cv2.bitwise_and(mask, cv2.bitwise_not(pcore))
            if grid_mask is not None and grid_mask.size > 0:
                occ_small = cv2.resize(grid_mask, (g, g), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            else:
                occ_small = np.zeros((g, g), dtype=np.float32)
        occ_small = self._apply_player_donut_on_grid(
            occ_small,
            player_center_roi=player_center_roi,
            roi_shape=(h, w),
        )
        occ = np.tanh(occ_small / float(max(1e-6, self.cfg.debug_grid_gain))).astype(np.float32)
        delta = np.clip(occ - self._prev_dbg_occ, -1.0, 1.0).astype(np.float32)
        self._prev_dbg_occ = occ.copy()

        self.last_mask_u8 = mask
        self.last_points_roi = pts
        self.last_points_topk_roi = topk
        self._dbg = {
            "n": int(len(pts)),
            "topk": int(len(topk)),
            "points": pts[: int(self.cfg.debug_max_draw)],
            "points_topk": topk,
            "hold_points": held_pts[: int(self.cfg.debug_max_draw)],
            "grid_occ": occ,
            "grid_delta": delta,
            "near_hold_n": int(len(self._near_hold)),
            "reject_track_total": int(rej_track_slow + rej_track_up + rej_new_near),
            "player_center_roi": player_center_roi,
            "K": K,
        }
        return topk

    def get_debug(self) -> Dict[str, Any]:
        return self._dbg or {}
