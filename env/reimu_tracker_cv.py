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
  - 일정 시간 동안 유지된 트랙만 LOCK 후보 선정
- LOCK 후:
  - CSRT tracker로만 추적
  - (추가) 위치가 1초간 거의 안 움직이면 재탐색(UNLOCK)
  - (추가) 재탐색 직후 일정 시간 ROI 확장(벽 근처 대응)

변경점(이 버전):
- LOCK 후보 선정에서 min_disp(최소 이동) 조건 제거 유지
- LOCK 후보 선정에서 size_stable_tol(크기 안정) 조건 제거 유지
- ✅ 배경(천천히/부드럽게/일정하게) 움직이는 트랙을 LOCK에서 감점/제외하는 규칙 추가
- ✅ 레이무가 주로 있는 "아래쪽" 위치 prior를 score에 추가(가벼운 편향)

추가 변경점(이번 요청):
- ✅ area 기준을 cv2.contourArea가 아니라 bbox 면적(w*h)로 통일
  -> 디버그 뷰의 area(=w*h)와 필터의 area가 같은 의미가 됨
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import heapq
import time
import math
from pathlib import Path

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

    # 재탐색 시 ROI 확장(벽 근처 대응)
    reacq_expand_sec: float = 2.0
    reacq_left: int = 0
    reacq_right_margin: int = 0
    # top/bottom은 기존 유지

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
    h_max: int = 115

    # ✅ area는 bbox 면적(w*h) 기준으로 통일
    area_min: int = 500
    area_max: int = 5000

    # aspect = w/h
    aspect_min: float = 0.40
    aspect_max: float = 0.78

    # Peak check (탄알 군집 제거)
    peak_bin_ratio: float = 0.60
    peak_max_count: int = 1
    peak_min_area: int = 25

    # 작은 후보는 peak 검사 스킵
    peak_check_min_area: int = 900      # ✅ bbox_area 기준으로 사용
    peak_check_min_wh: int = 26 * 38    # ~988

    # Association / lock
    assoc_dist: float = 82.0
    cand_ttl_sec: float = 0.30

    # hold 시간 내에 "지속적으로 관측된 트랙"만 락 후보
    lock_hold_sec: float = 0.19

    lock_pad_px: int = 12

    # 후보 수 줄이기
    max_candidates_per_frame: int = 45

    # debug
    debug_max_candidates: int = 80

    # LOCK 상태 이상 감지: 1초간 위치가 거의 안 변하면 재탐색
    lock_static_unlock_enable: bool = True
    lock_static_sec: float = 2.0
    lock_static_move_thr_px: float = 6.0
    lock_static_min_frames: int = 4

    # CSRT update 실패 시에도 재탐색 트리거
    unlock_on_csrt_fail: bool = False
    unlock_fail_consecutive: int = 6

    # Template gate: allow LOCK only when MOG2 candidate overlaps reimu template match.
    template_gate_enable: bool = False
    template_glob: str = "assets/reimu_*.png"
    template_match_thr: float = 0.66
    template_scales: Tuple[float, ...] = (0.90, 1.00, 1.10)
    template_max_boxes: int = 64
    template_match_iou_thr: float = 0.10

    # =========================
    # ✅ 배경 트랙 억제(LOCK 후보 선정용)
    # =========================
    lock_y_prior_weight: float = 0.35  # 0이면 사용 안함
    reject_smooth_bg: bool = True

    smooth_total_disp_px: float = 3.0
    smooth_max_step_px: float = 2.0
    smooth_dir_std_deg: float = 12.0

    # Action-motion consistency gate before first LOCK
    action_verify_enable: bool = True
    action_verify_window: int = 10
    action_verify_min_pairs: int = 2
    action_verify_motion_min_px: float = 0.3
    action_verify_cos_thr: float = 0.15
    action_verify_pass_ratio: float = 0.40
    action_verify_score_bonus: float = 0.50

    # Respawn/reacquire mode (temporary relaxed lock conditions)
    respawn_mode_sec: float = 1.0
    respawn_lock_hold_sec: float = 0.10
    respawn_min_track_points: int = 2
    respawn_disable_smooth_reject: bool = True
    respawn_y_prior_weight: float = 0.55


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


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = map(int, a)
    bx, by, bw, bh = map(int, b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    ua = float(max(1, aw * ah))
    ub = float(max(1, bw * bh))
    return inter / max(1e-6, (ua + ub - inter))


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
            if keep >= 2:
                break
    return int(keep)


def _dir_std_deg(vxs: List[float], vys: List[float]) -> float:
    if not vxs:
        return 999.0
    ang = [math.atan2(vy, vx) for vx, vy in zip(vxs, vys)]
    if len(ang) <= 1:
        return 999.0
    un = [ang[0]]
    for a in ang[1:]:
        prev = un[-1]
        da = a - prev
        while da > math.pi:
            a -= 2 * math.pi
            da = a - prev
        while da < -math.pi:
            a += 2 * math.pi
            da = a - prev
        un.append(a)
    m = sum(un) / len(un)
    var = sum((x - m) ** 2 for x in un) / max(1, len(un) - 1)
    return float(math.degrees(math.sqrt(var)))


def _track_motion_stats(pts: List[Tuple[float, float, float, BBox]]) -> Tuple[float, float, float]:
    if len(pts) < 2:
        return 0.0, 0.0, 999.0

    total = 0.0
    max_step = 0.0
    vxs: List[float] = []
    vys: List[float] = []

    for i in range(1, len(pts)):
        dx = float(pts[i][1] - pts[i - 1][1])
        dy = float(pts[i][2] - pts[i - 1][2])
        step = math.hypot(dx, dy)
        total += step
        if step > max_step:
            max_step = step
        if step > 1e-6:
            vxs.append(dx)
            vys.append(dy)

    dstd = _dir_std_deg(vxs, vys)
    return float(total), float(max_step), float(dstd)


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

        k = max(1, int(self.cfg.open_k))
        self._open_kernel = np.ones((k, k), np.uint8)

        self._csrt = None
        self.locked: bool = False
        self.lock_bbox: Optional[BBox] = None  # full-frame coords

        self._next_id: int = 1
        self._tracks: Dict[int, _Track] = {}

        self._lock_last_center: Optional[Tuple[float, float]] = None
        self._lock_static_since: Optional[float] = None
        self._lock_static_streak: int = 0
        self._lock_fail_streak: int = 0

        self._reacq_expand_until: float = 0.0
        self._respawn_mode_until: float = 0.0

        self._dbg_roi_xyxy: Optional[Tuple[int, int, int, int]] = None
        self._dbg_candidates_full: List[BBox] = []
        self._dbg_lock_cand_full: Optional[BBox] = None
        self._dbg_fg_roi: Optional[np.ndarray] = None
        self._dbg_action_verify_ratio: float = -1.0
        self._dbg_action_verify_pairs: int = 0

        self._recent_action_vecs: Deque[Optional[Tuple[float, float]]] = deque(
            maxlen=max(4, int(self.cfg.action_verify_window) * 4)
        )
        self._tmpl_gray: List[np.ndarray] = self._load_reimu_templates()
        self._dbg_template_candidates_full: List[BBox] = []

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

        self._lock_last_center = None
        self._lock_static_since = None
        self._lock_static_streak = 0
        self._lock_fail_streak = 0
        self._reacq_expand_until = 0.0
        self._respawn_mode_until = time.time() + float(self.cfg.respawn_mode_sec)

        self._dbg_roi_xyxy = None
        self._dbg_candidates_full = []
        self._dbg_template_candidates_full = []
        self._dbg_lock_cand_full = None
        self._dbg_fg_roi = None
        self._dbg_action_verify_ratio = -1.0
        self._dbg_action_verify_pairs = 0
        self._recent_action_vecs.clear()

    def get_debug(self):
        return {
            "roi_xyxy": self._dbg_roi_xyxy,
            "candidates": list(self._dbg_candidates_full),
            "template_candidates": list(self._dbg_template_candidates_full),
            "lock_cand": self._dbg_lock_cand_full,
            "locked_bbox": self.lock_bbox if self.locked else None,
            "fg_roi": self._dbg_fg_roi,
            "locked": bool(self.locked),
            "reacq_expand_until": float(self._reacq_expand_until),
            "respawn_mode_until": float(self._respawn_mode_until),
            "action_verify_ratio": float(self._dbg_action_verify_ratio),
            "action_verify_pairs": int(self._dbg_action_verify_pairs),
        }

    def _load_reimu_templates(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        try:
            for p in sorted(Path(".").glob(str(self.cfg.template_glob))):
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                if im.ndim == 3 and im.shape[2] == 4:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
                if im.ndim == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if im.dtype != np.uint8:
                    im = np.clip(im, 0, 255).astype(np.uint8)
                h, w = im.shape[:2]
                if h >= 6 and w >= 6:
                    out.append(im)
        except Exception:
            pass
        return out

    def _detect_template_candidates_roi(self, roi_bgr: np.ndarray) -> List[BBox]:
        if (not bool(self.cfg.template_gate_enable)) or (len(self._tmpl_gray) == 0):
            return []
        h, w = roi_bgr.shape[:2]
        if h < 6 or w < 6:
            return []
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        thr = float(self.cfg.template_match_thr)
        boxes: List[BBox] = []
        max_boxes = int(max(1, self.cfg.template_max_boxes))

        for t in self._tmpl_gray:
            th0, tw0 = t.shape[:2]
            for sc in self.cfg.template_scales:
                tw = int(max(6, round(tw0 * float(sc))))
                th = int(max(6, round(th0 * float(sc))))
                if tw >= w or th >= h:
                    continue
                tt = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
                res = cv2.matchTemplate(gray, tt, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= thr)
                for yy, xx in zip(ys.tolist(), xs.tolist()):
                    b = (int(xx), int(yy), int(tw), int(th))
                    keep = True
                    for ob in boxes:
                        if _bbox_iou(b, ob) >= 0.35:
                            keep = False
                            break
                    if keep:
                        boxes.append(b)
                        if len(boxes) >= max_boxes:
                            return boxes
        return boxes

    def _force_unlock_and_reacq(self, now: float):
        self._csrt = None
        self.locked = False
        self.lock_bbox = None

        self._tracks.clear()
        self._next_id = 1

        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=int(self.cfg.mog2_history),
            varThreshold=float(self.cfg.mog2_var_threshold),
            detectShadows=bool(self.cfg.mog2_detect_shadows),
        )

        self._lock_last_center = None
        self._lock_static_since = None
        self._lock_static_streak = 0
        self._lock_fail_streak = 0
        self._recent_action_vecs.clear()
        self._dbg_template_candidates_full = []

        self._reacq_expand_until = float(now) + float(self.cfg.reacq_expand_sec)
        self._respawn_mode_until = float(now) + float(self.cfg.respawn_mode_sec)

    def _push_expected_action(self, expected_move_vec: Optional[Tuple[float, float]]) -> None:
        if expected_move_vec is None:
            self._recent_action_vecs.append(None)
            return
        ax, ay = map(float, expected_move_vec)
        n = float(math.hypot(ax, ay))
        if n <= 1e-6:
            self._recent_action_vecs.append(None)
            return
        self._recent_action_vecs.append((ax / n, ay / n))

    def _action_motion_match_ratio(self, pts: List[Tuple[float, float, float, BBox]]) -> Tuple[float, int]:
        m = len(pts) - 1
        if m <= 0:
            return -1.0, 0
        acts = list(self._recent_action_vecs)
        if len(acts) < m:
            return -1.0, 0
        acts = acts[-m:]

        pass_n = 0
        used = 0
        cos_thr = float(self.cfg.action_verify_cos_thr)
        min_motion = float(max(0.01, self.cfg.action_verify_motion_min_px))

        for i in range(m):
            a = acts[i]
            if a is None:
                continue
            dx = float(pts[i + 1][1] - pts[i][1])
            dy = float(pts[i + 1][2] - pts[i][2])
            md = float(math.hypot(dx, dy))
            if md < min_motion:
                continue
            used += 1
            cosv = (dx * a[0] + dy * a[1]) / max(1e-6, md)
            if cosv >= cos_thr:
                pass_n += 1

        if used <= 0:
            return -1.0, 0
        return float(pass_n) / float(used), int(used)

    def step(
        self,
        frame_bgr: np.ndarray,
        now: Optional[float] = None,
        expected_move_vec: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[BBox], float]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None, 0.0

        frame_bgr = _ensure_uint8_bgr(frame_bgr)
        if frame_bgr is None or frame_bgr.size == 0:
            return None, 0.0

        self._push_expected_action(expected_move_vec)
        self._dbg_action_verify_ratio = -1.0
        self._dbg_action_verify_pairs = 0

        H, W = frame_bgr.shape[:2]
        if now is None:
            now = time.time()
        now = float(now)

        # =========================
        # LOCK
        # =========================
        if self.locked and self.lock_bbox is not None:
            ok = True
            b = None
            if self._csrt is not None:
                try:
                    ok, b = self._csrt.update(frame_bgr)
                except Exception:
                    ok, b = False, None

            if (not ok or b is None):
                self._lock_fail_streak += 1
                if bool(self.cfg.unlock_on_csrt_fail):
                    if self._lock_fail_streak >= int(max(1, self.cfg.unlock_fail_consecutive)):
                        self._force_unlock_and_reacq(now)
                else:
                    self._dbg_candidates_full = []
                    self._dbg_lock_cand_full = None
                    return self.lock_bbox, 1.0
            else:
                self._lock_fail_streak = 0
                self.lock_bbox = _clamp_bbox(tuple(map(int, b)), W, H)

                if bool(self.cfg.lock_static_unlock_enable):
                    cx, cy = _bbox_center(self.lock_bbox)
                    thr = float(self.cfg.lock_static_move_thr_px)
                    thr2 = thr * thr
                    if self._lock_last_center is None:
                        self._lock_last_center = (cx, cy)
                        self._lock_static_since = None
                        self._lock_static_streak = 0
                    else:
                        dx = float(cx - self._lock_last_center[0])
                        dy = float(cy - self._lock_last_center[1])
                        d2 = dx * dx + dy * dy

                        if d2 <= thr2:
                            self._lock_static_streak += 1
                            if self._lock_static_since is None:
                                self._lock_static_since = now
                            else:
                                if (
                                    (now - self._lock_static_since) >= float(self.cfg.lock_static_sec)
                                    and self._lock_static_streak >= int(max(1, self.cfg.lock_static_min_frames))
                                ):
                                    self._force_unlock_and_reacq(now)
                        else:
                            self._lock_last_center = (cx, cy)
                            self._lock_static_since = None
                            self._lock_static_streak = 0

            if self.locked and self.lock_bbox is not None:
                self._dbg_candidates_full = []
                self._dbg_template_candidates_full = []
                self._dbg_lock_cand_full = None
                self._dbg_fg_roi = None
                self._dbg_roi_xyxy = None
                return self.lock_bbox, 1.0

        # =========================
        # UNLOCK: detect candidates
        # =========================
        expand = (now < float(self._reacq_expand_until))
        if expand:
            x0, y0, x1, y1 = _roi_rect(
                W, H,
                left=int(self.cfg.reacq_left),
                top=int(self.cfg.roi_top),
                right_margin=int(self.cfg.reacq_right_margin),
                bottom_margin=int(self.cfg.roi_bottom_margin),
            )
        else:
            x0, y0, x1, y1 = _roi_rect(
                W, H,
                left=int(self.cfg.roi_left),
                top=int(self.cfg.roi_top),
                right_margin=int(self.cfg.roi_right_margin),
                bottom_margin=int(self.cfg.roi_bottom_margin),
            )

        self._dbg_roi_xyxy = (x0, y0, x1, y1)

        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            self._dbg_candidates_full = []
            self._dbg_lock_cand_full = None
            self._dbg_fg_roi = None
            return None, 0.0

        cands_roi, fg = self._detect_candidates_roi(roi)
        self._dbg_fg_roi = fg
        tmpl_roi = self._detect_template_candidates_roi(roi)

        self._dbg_candidates_full = []
        for (bx, by, bw, bh) in cands_roi:
            self._dbg_candidates_full.append(_clamp_bbox((x0 + bx, y0 + by, bw, bh), W, H))
        if len(self._dbg_candidates_full) > int(self.cfg.debug_max_candidates):
            self._dbg_candidates_full = self._dbg_candidates_full[: int(self.cfg.debug_max_candidates)]
        self._dbg_template_candidates_full = []
        for (bx, by, bw, bh) in tmpl_roi:
            self._dbg_template_candidates_full.append(_clamp_bbox((x0 + bx, y0 + by, bw, bh), W, H))
        if len(self._dbg_template_candidates_full) > int(self.cfg.debug_max_candidates):
            self._dbg_template_candidates_full = self._dbg_template_candidates_full[: int(self.cfg.debug_max_candidates)]

        self._update_tracks(cands_roi, now)
        lock_roi_bbox = self._pick_lock_candidate(now, roi_h=(y1 - y0), tmpl_roi=tmpl_roi)

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

        cx, cy = _bbox_center(self.lock_bbox)
        self._lock_last_center = (float(cx), float(cy))
        self._lock_static_since = None
        self._lock_static_streak = 0
        self._lock_fail_streak = 0

        self._reacq_expand_until = 0.0

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

        max_keep = int(self.cfg.max_candidates_per_frame)
        if len(cnts) > max_keep:
            cnts = heapq.nlargest(max_keep, cnts, key=cv2.contourArea)

        cands: List[BBox] = []
        for c in cnts:
            if len(cands) >= max_keep:
                break

            # ✅ area는 bbox 면적(w*h)로 통일하기 위해, 먼저 boundingRect를 구한다.
            x, y, w, h = cv2.boundingRect(c)

            if not (self.cfg.w_min <= w <= self.cfg.w_max and self.cfg.h_min <= h <= self.cfg.h_max):
                continue

            bbox_area = float(w * h)
            if bbox_area < float(self.cfg.area_min) or bbox_area > float(self.cfg.area_max):
                continue

            ar = (w / float(h)) if h > 0 else 999.0
            if ar < float(self.cfg.aspect_min) or ar > float(self.cfg.aspect_max):
                continue

            # 큰 후보만 peak 검사 (✅ bbox_area 기준)
            do_peak = (bbox_area >= float(self.cfg.peak_check_min_area)) and ((w * h) >= int(self.cfg.peak_check_min_wh))
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

        cand_centers: List[Tuple[float, float]] = []
        for b in cands_roi:
            cand_centers.append(_bbox_center(b))

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

        for i, b in enumerate(cands_roi):
            if i in used:
                continue
            cx, cy = cand_centers[i]
            self._tracks[self._next_id] = _Track(
                hist=deque([(now, float(cx), float(cy), b)], maxlen=60),
                last=now,
            )
            self._next_id += 1

    def _pick_lock_candidate(self, now: float, roi_h: int, tmpl_roi: Optional[List[BBox]] = None) -> Optional[BBox]:
        best: Optional[BBox] = None
        best_score = -1.0
        tmpl_roi = tmpl_roi or []
        use_gate = bool(self.cfg.template_gate_enable)
        iou_thr = float(max(0.0, self.cfg.template_match_iou_thr))

        in_respawn = bool(float(now) <= float(self._respawn_mode_until))
        hold = float(self.cfg.respawn_lock_hold_sec if in_respawn else self.cfg.lock_hold_sec)
        y_w = float(self.cfg.respawn_y_prior_weight if in_respawn else self.cfg.lock_y_prior_weight)
        use_smooth_reject = bool((not self.cfg.respawn_disable_smooth_reject) if in_respawn else self.cfg.reject_smooth_bg)
        min_pts = int(max(2, self.cfg.respawn_min_track_points if in_respawn else 3))

        for tr in self._tracks.values():
            pts = [p for p in tr.hist if (now - p[0]) <= hold]
            n = len(pts)
            if n < min_pts:
                continue

            t0 = pts[0][0]
            t1 = pts[-1][0]
            if (t1 - t0) < hold * 0.5:
                continue

            if use_smooth_reject:
                total_disp, max_step, dstd = _track_motion_stats(pts)
                if (
                    total_disp <= float(self.cfg.smooth_total_disp_px)
                    and max_step <= float(self.cfg.smooth_max_step_px)
                    and dstd <= float(self.cfg.smooth_dir_std_deg)
                ):
                    continue

            b_last = pts[-1][3]
            if use_gate:
                ok_match = False
                for tb in tmpl_roi:
                    if _bbox_iou(b_last, tb) >= iou_thr:
                        ok_match = True
                        break
                if not ok_match:
                    continue
            cy_last = float(pts[-1][2])

            score = (t1 - t0) * 10.0 + (b_last[2] * b_last[3]) * 0.001

            ratio = -1.0
            used = 0
            if bool(self.cfg.action_verify_enable):
                ratio, used = self._action_motion_match_ratio(pts)
                # Soft scoring only: do not reject lock candidates by action verify.
                if ratio >= 0.0 and used >= int(self.cfg.action_verify_min_pairs):
                    pass_thr = float(self.cfg.action_verify_pass_ratio)
                    score += float(self.cfg.action_verify_score_bonus) * float(ratio - pass_thr)

            if roi_h > 0 and y_w > 0.0:
                y_norm = max(0.0, min(1.0, cy_last / float(roi_h)))
                score += y_norm * y_w

            if score > best_score:
                best_score = score
                best = b_last
                if ratio >= 0.0:
                    self._dbg_action_verify_ratio = float(ratio)
                    self._dbg_action_verify_pairs = int(used)

        return best
