# env/ui_guard.py
import numpy as np
from env.ui_lives import count_lives_from_img

class UIGuard:
    """
    UI lives 안정화:
      - raw 측정값이 N프레임 연속 동일할 때만 확정값으로 채택
      - 급락(예: 3 -> 0 같은)은 1회 피격으로 보기 어렵기 때문에 무시/클램프
      - ui 패널이 없으면 None 반환(기존 유지)
    """
    def __init__(self, screen, state):
        self.screen = screen
        self.s = state

        # ---- lives stabilization ----
        self._raw_last = None          # 마지막 raw
        self._raw_same_cnt = 0         # raw가 연속으로 같은 횟수
        self._stable_lives = None      # 확정 lives

        # 튜닝 파라미터
        self._need_stable_frames = 3   # ★ 2~4 추천 (3이면 60fps 기준 50ms 정도)
        self._max_reasonable_drop = 1  # 한 번에 1 이상 감소면 의심 (동방 목숨은 보통 1씩)

    def ui_panel_present(self, img_bgr) -> bool:
        return bool(self.screen.ui_panel_present(img_bgr))

    def update_ui_absent(self, ui_ok: bool):
        if not ui_ok:
            self.s.ui_absent_count += 1
        else:
            self.s.ui_absent_count = 0

    def _stabilize_lives(self, raw: int) -> int | None:
        # 첫 측정은 바로 확정
        if self._stable_lives is None:
            self._stable_lives = int(raw)
            self._raw_last = int(raw)
            self._raw_same_cnt = 1
            return int(self._stable_lives)

        raw = int(raw)

        # raw 연속 카운트
        if self._raw_last is None or raw != int(self._raw_last):
            self._raw_last = raw
            self._raw_same_cnt = 1
        else:
            self._raw_same_cnt += 1

        # 아직 안정화 안 됐으면 기존 확정값 유지
        if self._raw_same_cnt < int(self._need_stable_frames):
            return int(self._stable_lives)

        # 여기부터는 "raw가 N프레임 연속 동일" = 후보 확정
        cand = raw
        prev = int(self._stable_lives)

        # 급락 방지 (플래시/이펙트에서 0으로 튀는 케이스 방어)
        if cand < prev - int(self._max_reasonable_drop):
            # 1) 완전히 무시(추천) -> prev 유지
            # return prev

            # 2) 클램프(대안) -> prev-1 로만 반영
            cand = prev - int(self._max_reasonable_drop)

        # 비정상 상승(갑자기 늘어남)도 대부분 오검출이라 막기
        if cand > prev + 1:
            cand = prev

        self._stable_lives = int(cand)
        return int(self._stable_lives)

    def ui_lives_safe(self, img_bgr, ui_panel_ok: bool):
        if not ui_panel_ok:
            return None

        try:
            v = count_lives_from_img(img_bgr, debug=False)
        except Exception:
            return None

        if v is None:
            return None
        if not isinstance(v, (int, np.integer)):
            return None

        v = int(v)
        if v < 0 or v > 12:
            return None

        # ✅ stabilize
        return self._stabilize_lives(v)

    def ui_lives_raw(self, img_bgr, ui_panel_ok: bool):
        """
        Raw lives count without stabilization.
        Used for immediate drop-trigger handling.
        """
        if not ui_panel_ok:
            return None
        try:
            v = count_lives_from_img(img_bgr, debug=False)
        except Exception:
            return None
        if v is None or (not isinstance(v, (int, np.integer))):
            return None
        v = int(v)
        if v < 0 or v > 12:
            return None
        return v
