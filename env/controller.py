# env/controller.py
import ctypes
from ctypes import wintypes
import time

user32 = ctypes.WinDLL("user32", use_last_error=True)

# SendInput signature
user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# ULONG_PTR 호환
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_uint32


# --- Windows INPUT full layout ---
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


def _send_input(inp: INPUT) -> bool:
    arr = (INPUT * 1)(inp)
    p = ctypes.byref(arr[0])
    sent = user32.SendInput(1, p, ctypes.sizeof(INPUT))
    return sent == 1


def _send_scancode(scan: int, is_down: bool, extended: bool = False) -> bool:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    if not is_down:
        flags |= KEYEVENTF_KEYUP

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(
        wVk=0,
        wScan=scan,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    return _send_input(inp)


# =========================
# Key mapping (Scan codes)
# =========================
# NOTE: 네 프로젝트에서 공격키는 z 고정, 폭탄키는 x 로 가정
SCAN = {
    "z": (0x2C, False),     # Z
    "x": (0x2D, False),     # X (bomb)
    "shift": (0x2A, False), # LSHIFT
    "left": (0x4B, True),
    "right": (0x4D, True),
    "up": (0x48, True),
    "down": (0x50, True),
}

ATTACK_KEY = "z"
BOMB_KEY = "x"

MOVE_KEYS = {
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "SLOW": "shift",
}

# ---- internal state ----
_HELD = set()
_ATTACK_HOLD = True

# ✅ 항상 SLOW(Shift) 유지 모드
_ALWAYS_SLOW = False


# =========================
# Public controls
# =========================
def set_attack_hold(enabled: bool):
    global _ATTACK_HOLD
    _ATTACK_HOLD = bool(enabled)
    if not _ATTACK_HOLD:
        _key_up(ATTACK_KEY)


def set_always_slow(enabled: bool):
    """
    enabled=True면 Shift를 항상 누른 상태로 유지.
    """
    global _ALWAYS_SLOW
    _ALWAYS_SLOW = bool(enabled)

    slow_key = MOVE_KEYS["SLOW"]
    if _ALWAYS_SLOW:
        _key_down(slow_key)
    else:
        _key_up(slow_key)


# =========================
# Low-level key ops
# =========================
def _key_down(key: str):
    if key in _HELD:
        return
    sc, ext = SCAN[key]
    _send_scancode(sc, is_down=True, extended=ext)
    _HELD.add(key)


def _key_up(key: str):
    sc, ext = SCAN[key]
    _send_scancode(sc, is_down=False, extended=ext)
    _HELD.discard(key)


def tap_key(key: str, hold_sec: float = 0.03):
    """
    '짧게 눌렀다 떼기' (폭탄처럼 순간 입력에 사용)
    """
    sc, ext = SCAN[key]
    _send_scancode(sc, is_down=True, extended=ext)
    time.sleep(max(0.0, float(hold_sec)))
    _send_scancode(sc, is_down=False, extended=ext)


def press_bomb(hold_sec: float = 0.03):
    tap_key(BOMB_KEY, hold_sec=hold_sec)


# =========================
# Main input API
# =========================
def press_keys(action_keys):
    """
    action_keys: ["LEFT"], ["SLOW","UP"], ["UP","RIGHT"] 등
    + ["BOMB"] 를 포함해도 됨(이 경우 x 를 '딸깍' 누름)
    """
    # 0) 폭탄은 '짧게 탭' (동시 입력 필요 없다고 했으니 여기서만 처리)
    if "BOMB" in action_keys:
        press_bomb()
        # bomb는 순간 입력이라 이후 로직에서 방향/슬로우에 영향 주지 않게 제거
        action_keys = [k for k in action_keys if k != "BOMB"]

    # 1) 공격키 유지
    if _ATTACK_HOLD:
        _key_down(ATTACK_KEY)
    else:
        _key_up(ATTACK_KEY)

    # 2) SLOW 처리
    slow_name = "SLOW"
    slow_key = MOVE_KEYS[slow_name]

    if _ALWAYS_SLOW:
        _key_down(slow_key)
    else:
        if slow_name in action_keys:
            _key_down(slow_key)
        else:
            _key_up(slow_key)

    # 3) 방향키 처리
    for name, key in MOVE_KEYS.items():
        if name == slow_name:
            continue
        if name in action_keys:
            _key_down(key)
        else:
            _key_up(key)


def release_all(force: bool = False):
    """
    모든 키 해제.

    force=True:
      - ALWAYS_SLOW 무시
      - Shift 포함 전부 해제
    """
    global _ALWAYS_SLOW

    if force:
        _ALWAYS_SLOW = False

    # 방향/슬로우 해제
    for _, key in MOVE_KEYS.items():
        _key_up(key)

    # 공격 해제
    _key_up(ATTACK_KEY)

    # 폭탄은 원래 홀드 안 하지만, 혹시 꼬였을까봐 한 번 Up 보장
    _key_up(BOMB_KEY)


# =========================
# ✅ 종료 안전장치 (핵심)
# =========================
def cleanup_inputs_on_exit():
    """
    🔥 학습 종료 시 반드시 호출해야 하는 함수 🔥

    - ALWAYS_SLOW 강제 해제
    - 모든 키 KeyUp
    - Shift 토글 탭(Down → Up)으로
      Windows 키 상태 꼬임을 물리적으로 복구
    """
    global _ALWAYS_SLOW
    _ALWAYS_SLOW = False

    # 1) 논리적 해제
    release_all(force=True)
    time.sleep(0.05)

    # 2) 물리적 안전장치: Shift 토글 탭
    sc, ext = SCAN["shift"]
    _send_scancode(sc, is_down=True, extended=ext)
    time.sleep(0.03)
    _send_scancode(sc, is_down=False, extended=ext)

    print("[CTRL] cleanup_inputs_on_exit: Shift safely released")
