# ppo_runner/hotkeys_win.py
import ctypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
VK_ESCAPE = 0x1B

def esc_pressed() -> bool:
    return (user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0
