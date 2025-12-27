# env/screen_util/window.py
import win32gui


def find_touhou_window(title_keyword: str = "동방홍마향"):
    """
    '동방홍마향' 창 HWND 탐색.
    """
    found = {"hwnd": None}

    def enum_handler(hwnd, _):
        if found["hwnd"] is not None:
            return
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_keyword in title:
                found["hwnd"] = hwnd

    win32gui.EnumWindows(enum_handler, None)
    return found["hwnd"]
