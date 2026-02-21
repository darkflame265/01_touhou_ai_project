# ppo_runner/hotkeys_win.py
import ctypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
VK_ESCAPE = 0x1B
VK_P = 0x50

def esc_pressed() -> bool:
    esc = (user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0
    pkey = (user32.GetAsyncKeyState(VK_P) & 0x8000) != 0
    return bool(esc or pkey)
