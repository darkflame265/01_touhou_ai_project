# env/reimu_tracker_cv.py
"""
CV 기반 레이무 트래커 (touhou_02 reimu_track_test.py 방식을 그대로 클래스화)

원리(=touhou_02 그대로):
- ROI: 고정 사각형(LEFT/TOP/RIGHT_MARGIN/BOTTOM_MARGIN)
- MOG2(roi_bgr에 apply) -> threshold(200) -> morph(open) -> erode -> dilate
- contour 후보들에 대해:
  - area/size/aspect(w/h) 필터
  - distanceTransform peak count로 탄알 군집 제거
- LOCK 전:
  - 후보들을 track으로 유지(association + TTL)
  - 일정 시간 동안 크기 안정적인 track만 LOCK 후보 선정
- LOCK 후:
  - CSRT tracker로만 추적
  - 추적 실패 시에만 detector+tracker reset 후 재탐색

출력:
- step(frame, now=None) -> (bbox_xywh or None, conf)
  - LOCK 성공/유지: (bbox, 1.0)
  - 그 외: (None, 0.0)

디버그:
- get_debug()로 ROI/candidates/lock_cand/locked_bbox 제공
- LOCK 상태에서는 touhou_02 느낌 그대로 candidates/lock_cand를 비움(그리지 않게)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


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
    # touhou_02와 동일한 의도: CSRT 사용 (OpenCV 버전차 대응)
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


def _peaks_count_from_mask(mask_u8: np.ndarray, peak_bin_ratio: float, peak_min_area: int) -> int:
    """
    touhou_02 peaks_count_from_mask와 동일:
    - m.sum() < 50 이면 0
    - distanceTransform(L2,5) -> thr=mx*ratio
    - thr 이상 peak 이진화 -> contour area >= peak_min_area count
    """
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
        a = cv2.contourArea(c)
        if a >= float(peak_min_area):
            keep += 1
    return int(keep)


@dataclass
class TrackerConfig:
    # ROI (touhou_02 동일)
    roi_left: int = 20
    roi_top: int = 210
    roi_right_margin: int = 210
    roi_bottom_margin: int = 20

    # MOG2 (touhou_02 동일)
    mog2_history: int = 120
    mog2_var_threshold: int = 28
    mog2_detect_shadows: bool = False

    # morphology (touhou_02 동일)
    open_k: int = 3
    open_iter: int = 1
    erode_iter: int = 1
    dilate_iter: int = 1

    # size/area (touhou_02 동일)
    w_min: int = 18
    w_max: int = 70
    h_min: int = 28
    h_max: int = 90
    area_min: int = 400
    area_max: int = 6000

    # aspect (touhou_02 동일)
    aspect_min: float = 0.45
    aspect_max: float = 0.78

    # peak 제거 (touhou_02 동일)
    peak_bin_ratio: float = 0.60
    peak_max_count: int = 1
    peak_min_area: int = 25

    # track 유지/확정 (touhou_02 동일)
    assoc_dist: float = 60.0
    cand_ttl_sec: float = 0.25
    lock_hold_sec: float = 0.25
    size_stable_tol: float = 0.35

    # LOCK 후보 bbox 패딩 (touhou_02 동일)
    lock_pad_px: int = 6

    # 디버그 후보 표시 제한(성능/가독성용, 원리에는 영향 없음)
    debug_max_candidates: int = 80


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

        # LOCK 후 CSRT
        self._csrt = None
        self.locked: bool = False
        self.lock_bbox: Optional[BBox] = None  # full-frame coords

        # LOCK 전 tracks
        self._next_id: int = 1
        self._tracks: Dict[int, _Track] = {}

        # debug snapshots
        self._dbg_roi_xyxy: Optional[Tuple[int, int, int, int]] = None
        self._dbg_candidates_full: List[BBox] = []
        self._dbg_lock_cand_full: Optional[BBox] = None
        self._dbg_fg_roi: Optional[np.ndarray] = None

    def reset(self):
        # touhou_02 detector.reset + tracker.reset 동일 효과
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

        H, W = frame_bgr.shape[:2]
        if now is None:
            import time
            now = time.time()
        now = float(now)

        # ROI
        x0, y0, x1, y1 = _roi_rect(
            W, H,
            left=self.cfg.roi_left,
            top=self.cfg.roi_top,
            right_margin=self.cfg.roi_right_margin,
            bottom_margin=self.cfg.roi_bottom_margin,
        )
        self._dbg_roi_xyxy = (x0, y0, x1, y1)

        # =========================================================
        # LOCK 상태: "절대로 unlock/re-detect 하지 않는다"
        # - update 실패해도 locked 유지
        # - 대신 동일 bbox로 CSRT 재초기화(re-init)만 시도
        # =========================================================
        if self.locked and (self._csrt is not None) and (self.lock_bbox is not None):
            ok, b = self._csrt.update(frame_bgr)
            if ok:
                self.lock_bbox = _clamp_bbox(tuple(map(int, b)), W, H)

                # LOCK이면 후보/주황 표시 안 함 (touhou_02 느낌)
                self._dbg_candidates_full = []
                self._dbg_lock_cand_full = None
                return self.lock_bbox, 1.0

            # ❗여기부터가 핵심 수정:
            # update가 실패해도 unlock 하지 말고, 현재 lock_bbox로 CSRT를 다시 init 해본다.
            # (재탐색/ detector reset 금지)
            trk = _make_csrt_tracker()
            if trk is not None:
                # 혹시 bbox가 화면 밖으로 삐져나가면 clamp
                bb = _clamp_bbox(self.lock_bbox, W, H)
                ok2 = trk.init(frame_bgr, tuple(map(float, bb)))
                if ok2:
                    self._csrt = trk
                    self.lock_bbox = bb

                    self._dbg_candidates_full = []
                    self._dbg_lock_cand_full = None
                    return self.lock_bbox, 1.0

            # re-init도 실패해도 "LOCK 유지"
            # -> 마지막 bbox를 계속 반환 (너 요구사항: 절대 풀리지 않음)
            self._dbg_candidates_full = []
            self._dbg_lock_cand_full = None
            return self.lock_bbox, 1.0

        # -------------------------
        # UNLOCK 상태: candidate 탐색 (기존 그대로)
        # -------------------------
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            self._dbg_candidates_full = []
            self._dbg_lock_cand_full = None
            return None, 0.0

        cands_roi, fg = self._detect_candidates_roi(roi)
        self._dbg_fg_roi = fg

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

        ok = trk.init(frame_bgr, tuple(map(float, cand_full)))
        if not ok:
            self._tracks.clear()
            self._next_id = 1
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=int(self.cfg.mog2_history),
                varThreshold=float(self.cfg.mog2_var_threshold),
                detectShadows=bool(self.cfg.mog2_detect_shadows),
            )
            self._dbg_lock_cand_full = None
            return None, 0.0

        self._csrt = trk
        self.locked = True
        self.lock_bbox = cand_full

        # LOCK 직후에도 후보/주황 표시 안 함
        self._dbg_candidates_full = []
        self._dbg_lock_cand_full = None

        return self.lock_bbox, 1.0

    # -------------------------
    # Internal: touhou_02 detector logic
    # -------------------------
    def _detect_candidates_roi(self, roi_bgr: np.ndarray) -> Tuple[List[BBox], np.ndarray]:
        # touhou_02: bg.apply(roi_bgr) (GRAY 변환 안 함)
        fg = self.bg.apply(roi_bgr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        k = int(self.cfg.open_k)
        k = max(1, k)
        kernel = np.ones((k, k), np.uint8)

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=int(self.cfg.open_iter))
        fg = cv2.erode(fg, None, iterations=int(self.cfg.erode_iter))
        fg = cv2.dilate(fg, None, iterations=int(self.cfg.dilate_iter))

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cands: List[BBox] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < float(self.cfg.area_min) or area > float(self.cfg.area_max):
                continue

            x, y, w, h = cv2.boundingRect(c)
            if not (self.cfg.w_min <= w <= self.cfg.w_max and self.cfg.h_min <= h <= self.cfg.h_max):
                continue

            ar = (w / float(h)) if h > 0 else 999.0
            if ar < float(self.cfg.aspect_min) or ar > float(self.cfg.aspect_max):
                continue

            sub = fg[y:y + h, x:x + w]
            pk = _peaks_count_from_mask(sub, peak_bin_ratio=self.cfg.peak_bin_ratio, peak_min_area=self.cfg.peak_min_area)
            if pk > int(self.cfg.peak_max_count):
                continue

            cands.append((int(x), int(y), int(w), int(h)))

        return cands, fg

    def _update_tracks(self, cands_roi: List[BBox], now: float):
        used = set()

        # prune + association (touhou_02 동일)
        for tid, tr in list(self._tracks.items()):
            if (now - tr.last) > float(self.cfg.cand_ttl_sec):
                del self._tracks[tid]
                continue

            lx, ly = float(tr.hist[-1][1]), float(tr.hist[-1][2])

            best_i = None
            best_d = 1e18
            for i, b in enumerate(cands_roi):
                if i in used:
                    continue
                cx, cy = _bbox_center(b)
                d = (cx - lx) ** 2 + (cy - ly) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i

            if best_i is not None and (best_d ** 0.5) <= float(self.cfg.assoc_dist):
                used.add(best_i)
                b = cands_roi[best_i]
                cx, cy = _bbox_center(b)
                tr.hist.append((now, float(cx), float(cy), b))
                tr.last = now

        # new tracks (touhou_02 동일)
        for i, b in enumerate(cands_roi):
            if i in used:
                continue
            cx, cy = _bbox_center(b)
            self._tracks[self._next_id] = _Track(
                hist=deque([(now, float(cx), float(cy), b)], maxlen=60),
                last=now,
            )
            self._next_id += 1

    def _pick_lock_candidate(self, now: float) -> Optional[BBox]:
        best = None
        best_score = -1.0

        # touhou_02 동일
        for tr in self._tracks.values():
            pts = [p for p in tr.hist if (now - p[0]) <= float(self.cfg.lock_hold_sec)]
            if len(pts) < 4:
                continue

            t0 = pts[0][0]
            t1 = pts[-1][0]
            if (t1 - t0) < float(self.cfg.lock_hold_sec) * 0.8:
                continue

            ws = np.array([p[3][2] for p in pts], dtype=np.float32)
            hs = np.array([p[3][3] for p in pts], dtype=np.float32)
            w0, h0 = float(ws[0]), float(hs[0])

            wv = float((ws.max() - ws.min()) / max(1.0, w0))
            hv = float((hs.max() - hs.min()) / max(1.0, h0))
            if wv > float(self.cfg.size_stable_tol) or hv > float(self.cfg.size_stable_tol):
                continue

            b_last = pts[-1][3]
            score = (t1 - t0) * 10.0 + (b_last[2] * b_last[3]) * 0.001
            if score > best_score:
                best_score = score
                best = b_last

        return best
