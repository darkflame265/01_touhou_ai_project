# env/reimu_tracker_cv.py
"""
CV 기반 레이무 트래커

원리:
- ROI 고정
- MOG2 -> threshold -> morph(open) -> erode -> dilate
- contour 후보:
  - area/size/aspect 필터
  - (큰 후보에 한해) distanceTransform peak count로 탄알 군집 제거
- LOCK 전:
  - 후보들을 track 유지(association + TTL)
  - 일정 시간 동안 크기 안정 + (최소 이동) 트랙만 LOCK 후보 선정
- LOCK 후:
  - CSRT tracker로만 추적
  - R 누르기 전까지 절대 unlock 하지 않음 (외부에서 reset() 호출)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import heapq

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class TrackerConfig:
    # ROI
    roi_left: int = 20
    roi_top: int = 210
    roi_right_margin: int = 210
    roi_bottom_margin: int = 10

    # MOG2
    mog2_history: int = 120
    mog2_var_threshold: int = 28
    mog2_detect_shadows: bool = False

    # Morphology
    open_k: int = 3
    open_iter: int = 1
    erode_iter: int = 1
    dilate_iter: int = 1

    # Candidate filters
    w_min: int = 18
    w_max: int = 70
    h_min: int = 28
    h_max: int = 90
    area_min: int = 400
    area_max: int = 6000

    aspect_min: float = 0.45
    aspect_max: float = 0.78

    # Peak check (탄알 군집 제거)
    peak_bin_ratio: float = 0.60
    peak_max_count: int = 1
    peak_min_area: int = 25

    # 작은 후보는 peak 검사 스킵 (비싼 distanceTransform 절감)
    peak_check_min_area: int = 900
    peak_check_min_wh: int = 26 * 38  # ~988

    # Association / lock
    assoc_dist: float = 60.0
    cand_ttl_sec: float = 0.25

    # 0.25 -> 0.18~0.20
    lock_hold_sec: float = 0.19

    size_stable_tol: float = 0.35

    # 락 후보는 "최소 이동" 필요
    lock_min_disp_px: float = 2.0

    lock_pad_px: int = 6

    # 후보 수 줄이기
    max_candidates_per_frame: int = 45

    # debug
    debug_max_candidates: int = 80


def _clamp_bbox(b: BBox, W: int, H: int) -> BBox:
    x, y, w, h = b
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return (x, y, w, h)


def _bbox_center(b: BBox) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + w * 0.5, y + h * 0.5)


def _roi_rect(W: int, H: int, left: int, top: int, right_margin: int, bottom_margin: int) -> Tuple[int, int, int, int]:
    x0 = int(left)
    y0 = int(top)
    x1 = int(W - right_margin)
    y1 = int(H - bottom_margin)

    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))
    return x0, y0, x1, y1


def _make_csrt_tracker():
    try:
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
    except Exception:
        pass
    try:
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
    except Exception:
        pass
    return None


def _tracker_init_ok(ret) -> bool:
    if ret is None:
        return True
    return bool(ret)


def _ensure_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        if f.size > 0 and float(np.nanmax(f)) <= 1.5:
            f = f * 255.0
        frame = np.clip(f, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        pass
    else:
        return frame

    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


def _peaks_count_from_mask(mask_u8: np.ndarray, peak_bin_ratio: float, peak_min_area: int) -> int:
    if mask_u8 is None or mask_u8.size == 0:
        return 0

    m = (mask_u8 > 0).astype(np.uint8)
    if int(m.sum()) < 50:
        return 0

    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    mx = float(dist.max())
    if mx <= 1e-6:
        return 0

    thr = mx * float(peak_bin_ratio)
    peak = (dist >= thr).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(peak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = 0
    for c in cnts:
        if cv2.contourArea(c) >= float(peak_min_area):
            keep += 1
            # peak_max_count=1이면 여기서 더 볼 필요 없음 (미세 최적화)
            if keep >= 2:
                break
    return int(keep)


@dataclass
class _Track:
    hist: Deque[Tuple[float, float, float, BBox]]  # (t, cx, cy, bbox_roi_xywh)
    last: float


class ReimuTrackerCV:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()

        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=int(self.cfg.mog2_history),
            varThreshold=float(self.cfg.mog2_var_threshold),
            detectShadows=bool(self.cfg.mog2_detect_shadows),
        )

        # morphology kernel 캐시 (프레임마다 만들지 않음)
        k = max(1, int(self.cfg.open_k))
        self._open_kernel = np.ones((k, k), np.uint8)

        self._csrt = None
        self.locked: bool = False
        self.lock_bbox: Optional[BBox] = None  # full-frame coords

        self._next_id: int = 1
        self._tracks: Dict[int, _Track] = {}

        self._dbg_roi_xyxy: Optional[Tuple[int, int, int, int]] = None
        self._dbg_candidates_full: List[BBox] = []
        self._dbg_lock_cand_full: Optional[BBox] = None
        self._dbg_fg_roi: Optional[np.ndarray] = None

    def reset(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=int(self.cfg.mog2_history),
            varThreshold=float(self.cfg.mog2_var_threshold),
            detectShadows=bool(self.cfg.mog2_detect_shadows),
        )
        self._next_id = 1
        self._tracks.clear()

        self._csrt = None
        self.locked = False
        self.lock_bbox = None

        self._dbg_roi_xyxy = None
        self._dbg_candidates_full = []
        self._dbg_lock_cand_full = None
        self._dbg_fg_roi = None

    def get_debug(self):
        return {
            "roi_xyxy": self._dbg_roi_xyxy,
            "candidates": list(self._dbg_candidates_full),
            "lock_cand": self._dbg_lock_cand_full,
            "locked_bbox": self.lock_bbox if self.locked else None,
            "fg_roi": self._dbg_fg_roi,
            "locked": bool(self.locked),
        }

    def step(self, frame_bgr: np.ndarray, now: Optional[float] = None) -> Tuple[Optional[BBox], float]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None, 0.0

        frame_bgr = _ensure_uint8_bgr(frame_bgr)
        if frame_bgr is None or frame_bgr.size == 0:
            return None, 0.0

        H, W = frame_bgr.shape[:2]
        if now is None:
            import time
            now = time.time()
        now = float(now)

        x0, y0, x1, y1 = _roi_rect(
            W, H,
            left=self.cfg.roi_left,
            top=self.cfg.roi_top,
            right_margin=self.cfg.roi_right_margin,
            bottom_margin=self.cfg.roi_bottom_margin,
        )
        self._dbg_roi_xyxy = (x0, y0, x1, y1)

        # LOCK: CSRT update only
        if self.locked and self.lock_bbox is not None:
            if self._csrt is not None:
                try:
                    ok, b = self._csrt.update(frame_bgr)
                except Exception:
                    ok, b = False, None
                if ok and b is not None:
                    self.lock_bbox = _clamp_bbox(tuple(map(int, b)), W, H)

            self._dbg_candidates_full = []
            self._dbg_lock_cand_full = None
            return self.lock_bbox, 1.0

        # UNLOCK: detect candidates
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            self._dbg_candidates_full = []
            self._dbg_lock_cand_full = None
            return None, 0.0

        cands_roi, fg = self._detect_candidates_roi(roi)
        self._dbg_fg_roi = fg

        # debug candidates (full coords)
        self._dbg_candidates_full = []
        for (bx, by, bw, bh) in cands_roi:
            self._dbg_candidates_full.append(_clamp_bbox((x0 + bx, y0 + by, bw, bh), W, H))
        if len(self._dbg_candidates_full) > int(self.cfg.debug_max_candidates):
            self._dbg_candidates_full = self._dbg_candidates_full[: int(self.cfg.debug_max_candidates)]

        self._update_tracks(cands_roi, now)
        lock_roi_bbox = self._pick_lock_candidate(now)

        if lock_roi_bbox is None:
            self._dbg_lock_cand_full = None
            return None, 0.0

        rx, ry, rw, rh = lock_roi_bbox
        cand_full = _clamp_bbox((x0 + rx, y0 + ry, rw, rh), W, H)

        pad = int(self.cfg.lock_pad_px)
        cand_full = _clamp_bbox(
            (cand_full[0] - pad, cand_full[1] - pad, cand_full[2] + pad * 2, cand_full[3] + pad * 2),
            W, H
        )

        self._dbg_lock_cand_full = cand_full

        trk = _make_csrt_tracker()
        if trk is None:
            return None, 0.0

        try:
            ret = trk.init(frame_bgr, tuple(map(float, cand_full)))
            ok_init = _tracker_init_ok(ret)
        except Exception:
            ok_init = False

        if not ok_init:
            # 실패 시 과감히 초기화(다음 기회)
            self._tracks.clear()
            self._next_id = 1
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=int(self.cfg.mog2_history),
                varThreshold=float(self.cfg.mog2_var_threshold),
                detectShadows=bool(self.cfg.mog2_detect_shadows),
            )
            return None, 0.0

        self._csrt = trk
        self.locked = True
        self.lock_bbox = cand_full

        self._dbg_candidates_full = []
        self._dbg_lock_cand_full = None
        return self.lock_bbox, 1.0

    def _detect_candidates_roi(self, roi_bgr: np.ndarray) -> Tuple[List[BBox], np.ndarray]:
        fg = self.bg.apply(roi_bgr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        fg = cv2.morphologyEx(
            fg, cv2.MORPH_OPEN, self._open_kernel, iterations=int(self.cfg.open_iter)
        )
        fg = cv2.erode(fg, None, iterations=int(self.cfg.erode_iter))
        fg = cv2.dilate(fg, None, iterations=int(self.cfg.dilate_iter))

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ✅ 최적화 1) 후보가 많으면 "전체정렬" 대신 상위 K개만 뽑기 (부분정렬)
        max_keep = int(self.cfg.max_candidates_per_frame)
        if len(cnts) > max_keep:
            cnts = heapq.nlargest(max_keep, cnts, key=cv2.contourArea)

        cands: List[BBox] = []
        for c in cnts:
            if len(cands) >= max_keep:
                break

            area = float(cv2.contourArea(c))
            if area < float(self.cfg.area_min) or area > float(self.cfg.area_max):
                continue

            x, y, w, h = cv2.boundingRect(c)
            if not (self.cfg.w_min <= w <= self.cfg.w_max and self.cfg.h_min <= h <= self.cfg.h_max):
                continue

            ar = (w / float(h)) if h > 0 else 999.0
            if ar < float(self.cfg.aspect_min) or ar > float(self.cfg.aspect_max):
                continue

            # 큰 후보만 peak 검사
            do_peak = (area >= float(self.cfg.peak_check_min_area)) and ((w * h) >= int(self.cfg.peak_check_min_wh))
            if do_peak:
                sub = fg[y:y + h, x:x + w]
                pk = _peaks_count_from_mask(
                    sub,
                    peak_bin_ratio=self.cfg.peak_bin_ratio,
                    peak_min_area=self.cfg.peak_min_area,
                )
                if pk > int(self.cfg.peak_max_count):
                    continue

            cands.append((int(x), int(y), int(w), int(h)))

        return cands, fg

    def _update_tracks(self, cands_roi: List[BBox], now: float):
        used = set()
        ttl = float(self.cfg.cand_ttl_sec)
        assoc = float(self.cfg.assoc_dist)
        assoc2 = assoc * assoc

        # ✅ 최적화 2) 후보 중심점 프레임당 1회 계산
        cand_centers: List[Tuple[float, float]] = []
        for b in cands_roi:
            cand_centers.append(_bbox_center(b))

        # update existing tracks
        for tid, tr in list(self._tracks.items()):
            if (now - tr.last) > ttl:
                del self._tracks[tid]
                continue

            lx, ly = float(tr.hist[-1][1]), float(tr.hist[-1][2])

            best_i = None
            best_d2 = 1e18
            for i, (cx, cy) in enumerate(cand_centers):
                if i in used:
                    continue
                dx = cx - lx
                dy = cy - ly
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i

            if best_i is not None and best_d2 <= assoc2:
                used.add(best_i)
                b = cands_roi[best_i]
                cx, cy = cand_centers[best_i]
                tr.hist.append((now, float(cx), float(cy), b))
                tr.last = now

        # new tracks
        for i, b in enumerate(cands_roi):
            if i in used:
                continue
            cx, cy = cand_centers[i]
            self._tracks[self._next_id] = _Track(
                hist=deque([(now, float(cx), float(cy), b)], maxlen=60),
                last=now,
            )
            self._next_id += 1

    def _pick_lock_candidate(self, now: float) -> Optional[BBox]:
        best: Optional[BBox] = None
        best_score = -1.0

        hold = float(self.cfg.lock_hold_sec)
        stable_tol = float(self.cfg.size_stable_tol)
        min_disp = float(self.cfg.lock_min_disp_px)

        for tr in self._tracks.values():
            # hold 내 포인트들만
            pts = [p for p in tr.hist if (now - p[0]) <= hold]
            n = len(pts)
            if n < 4:
                continue

            t0 = pts[0][0]
            t1 = pts[-1][0]
            if (t1 - t0) < hold * 0.8:
                continue

            # 최소 이동 조건
            dx = float(pts[-1][1] - pts[0][1])
            dy = float(pts[-1][2] - pts[0][2])
            if (dx * dx + dy * dy) < (min_disp * min_disp):
                continue

            # size stability (min/max만)
            w0 = float(pts[0][3][2])
            h0 = float(pts[0][3][3])
            w_min = 1e18
            w_max = -1e18
            h_min = 1e18
            h_max = -1e18
            for p in pts:
                w = float(p[3][2])
                h = float(p[3][3])
                if w < w_min:
                    w_min = w
                if w > w_max:
                    w_max = w
                if h < h_min:
                    h_min = h
                if h > h_max:
                    h_max = h

            wv = float((w_max - w_min) / max(1.0, w0))
            hv = float((h_max - h_min) / max(1.0, h0))
            if wv > stable_tol or hv > stable_tol:
                continue

            b_last = pts[-1][3]
            score = (t1 - t0) * 10.0 + (b_last[2] * b_last[3]) * 0.001
            if score > best_score:
                best_score = score
                best = b_last

        return best
