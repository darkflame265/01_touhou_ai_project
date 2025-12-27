# env/screen_util/window.py
from __future__ import annotations
from typing import Tuple, Optional

import win32gui

Rect = Tuple[int, int, int, int]


def find_touhou_window():
    """
    '동방홍마향' 창 HWND 탐색.
    (네 기존 구현 그대로 두면 됨)
    """
    found = {"hwnd": None}

    def enum_handler(hwnd, _):
        if found["hwnd"] is not None:
            return
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "동방홍마향" in title:
                found["hwnd"] = hwnd

    win32gui.EnumWindows(enum_handler, None)
    return found["hwnd"]


def get_client_rect_screen(hwnd) -> Rect:
    """
    윈도우 '클라이언트 영역'을 스크린 좌표로 변환해 반환.
    실패 시 window rect로 fallback.
    """
    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
        (sx0, sy0) = win32gui.ClientToScreen(hwnd, (l, t))
        (sx1, sy1) = win32gui.ClientToScreen(hwnd, (r, b))
        if sx1 > sx0 and sy1 > sy0:
            return (int(sx0), int(sy0), int(sx1), int(sy1))
    except Exception:
        pass

    try:
        L, T, R, B = win32gui.GetWindowRect(hwnd)
        return (int(L), int(T), int(R), int(B))
    except Exception:
        return (0, 0, 640, 480)


class ClientRectCache:
    """
    N프레임마다 client rect를 갱신해주는 캐시.
    """
    def __init__(self, hwnd, refresh_every: int = 30):
        self.hwnd = hwnd
        self.refresh_every = max(1, int(refresh_every))
        self._i = 0
        self._rect: Rect = get_client_rect_screen(hwnd)

    def get(self) -> Rect:
        self._i += 1
        if (self._i % self.refresh_every) == 0:
            try:
                self._rect = get_client_rect_screen(self.hwnd)
            except Exception:
                pass
        return self._rect

    def force_refresh(self) -> Rect:
        self._rect = get_client_rect_screen(self.hwnd)
        return self._rect
