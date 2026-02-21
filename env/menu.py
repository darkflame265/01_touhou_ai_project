# env/menu.py
import time
import os
import ctypes
from ctypes import wintypes

import win32gui
import cv2
import numpy as np

from env.screen import find_touhou_window
from ppo_runner.hotkeys import esc_pressed

# =========================
# WinAPI SendInput (FULL STRUCT)
# =========================
user32 = ctypes.WinDLL("user32", use_last_error=True)

user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_uint32


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


# =========================
# DirectInput scancodes
# =========================
SC_Z = 0x2C
SC_X = 0x2D
SC_UP = 0x48  # extended=True로 보내야 함


# =========================
# Templates (LOBBY existence only)
# =========================
TEMPLATES: dict[str, np.ndarray] = {}
TEMPLATES_HALF: dict[str, np.ndarray] = {}
_TEMPLATES_READY = False


def _load_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None or g.size == 0:
        raise RuntimeError(f"template load failed: {path}")
    return g


def _ensure_lobby_templates_loaded():
    """
    import 시점에 무조건 로드하지 않고, 최초 사용 시 1회 로드.
    (프로세스 시작 체감 지연도 조금 줄어듦)
    """
    global _TEMPLATES_READY
    if _TEMPLATES_READY:
        return

    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    practice = _load_gray(os.path.join(base, "lobby_practice.png"))
    quit_ = _load_gray(os.path.join(base, "lobby_quit.png"))

    TEMPLATES["practice"] = practice
    TEMPLATES["quit"] = quit_

    # half-scale 템플릿(매칭 속도용)
    TEMPLATES_HALF["practice"] = cv2.resize(practice, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    TEMPLATES_HALF["quit"] = cv2.resize(quit_, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    _TEMPLATES_READY = True
    print("[MENU] templates loaded:", list(TEMPLATES.keys()))


def _match_template(gray: np.ndarray, tmpl: np.ndarray) -> float:
    th, tw = tmpl.shape[:2]
    if gray.shape[0] < th or gray.shape[1] < tw:
        return 0.0
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def _abort_requested() -> bool:
    try:
        return bool(esc_pressed())
    except Exception:
        return False


# =========================
# Focus helper
# =========================
def focus_touhou_window(max_try: int = 5, sleep: float = 0.05) -> bool:
    hwnd = find_touhou_window()
    if not hwnd:
        print("[MENU][ERR] Touhou window not found")
        return False

    u32 = ctypes.WinDLL("user32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GetForegroundWindow = u32.GetForegroundWindow
    GetForegroundWindow.restype = wintypes.HWND

    GetWindowThreadProcessId = u32.GetWindowThreadProcessId
    GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    GetWindowThreadProcessId.restype = wintypes.DWORD

    AttachThreadInput = u32.AttachThreadInput
    AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
    AttachThreadInput.restype = wintypes.BOOL

    SetForegroundWindow = u32.SetForegroundWindow
    SetForegroundWindow.argtypes = (wintypes.HWND,)
    SetForegroundWindow.restype = wintypes.BOOL

    SetActiveWindow = u32.SetActiveWindow
    SetActiveWindow.argtypes = (wintypes.HWND,)
    SetActiveWindow.restype = wintypes.HWND

    SetFocus = u32.SetFocus
    SetFocus.argtypes = (wintypes.HWND,)
    SetFocus.restype = wintypes.HWND

    BringWindowToTop = u32.BringWindowToTop
    BringWindowToTop.argtypes = (wintypes.HWND,)
    BringWindowToTop.restype = wintypes.BOOL

    SetWindowPos = u32.SetWindowPos
    SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND,
        wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
        wintypes.UINT
    )
    SetWindowPos.restype = wintypes.BOOL

    GetCurrentThreadId = k32.GetCurrentThreadId
    GetCurrentThreadId.restype = wintypes.DWORD

    SW_RESTORE = 9
    HWND_TOPMOST = wintypes.HWND(-1)
    HWND_NOTOPMOST = wintypes.HWND(-2)
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040

    def _is_fg():
        return GetForegroundWindow() == hwnd

    for _ in range(max_try):
        if _abort_requested():
            return False
        try:
            win32gui.ShowWindow(hwnd, SW_RESTORE)
        except Exception:
            pass

        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        if _is_fg():
            return True

        try:
            SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        except Exception:
            pass

        try:
            BringWindowToTop(hwnd)
        except Exception:
            pass

        fg = GetForegroundWindow()
        fg_pid = wintypes.DWORD(0)
        target_pid = wintypes.DWORD(0)

        fg_tid = GetWindowThreadProcessId(fg, ctypes.byref(fg_pid)) if fg else 0
        target_tid = GetWindowThreadProcessId(hwnd, ctypes.byref(target_pid))
        cur_tid = GetCurrentThreadId()

        try:
            if fg_tid and fg_tid != cur_tid:
                AttachThreadInput(fg_tid, cur_tid, True)
            if target_tid and target_tid != cur_tid:
                AttachThreadInput(target_tid, cur_tid, True)

            SetActiveWindow(hwnd)
            SetFocus(hwnd)
            SetForegroundWindow(hwnd)

        finally:
            try:
                if fg_tid and fg_tid != cur_tid:
                    AttachThreadInput(fg_tid, cur_tid, False)
            except Exception:
                pass
            try:
                if target_tid and target_tid != cur_tid:
                    AttachThreadInput(target_tid, cur_tid, False)
            except Exception:
                pass

        if _is_fg():
            return True

        time.sleep(sleep)

    print("[MENU][ERR] focus failed: cannot bring Touhou window to foreground")
    return False


# =========================
# SendInput helpers
# =========================
def _send_input(inp: INPUT) -> bool:
    arr = (INPUT * 1)(inp)
    p = ctypes.byref(arr[0])
    sent = user32.SendInput(1, p, ctypes.sizeof(INPUT))
    return sent == 1


def tap_scancode(scan: int, extended=False, press=0.02, gap=0.03, label=""):
    flags_down = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    flags_up = flags_down | KEYEVENTF_KEYUP

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags_down, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(press)

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags_up, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(gap)


# =========================
# ROI helpers
# =========================
def _roi(img_bgr, x1r, y1r, x2r, y2r):
    h, w = img_bgr.shape[:2]
    x1 = int(np.clip(w * x1r, 0, w - 1))
    x2 = int(np.clip(w * x2r, 0, w))
    y1 = int(np.clip(h * y1r, 0, h - 1))
    y2 = int(np.clip(h * y2r, 0, h))
    if x2 <= x1 or y2 <= y1:
        return img_bgr[0:1, 0:1]
    return img_bgr[y1:y2, x1:x2]


def _menu_highlight_score(img_bgr_roi):
    if img_bgr_roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    g = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    white_ratio = float((g >= 210).mean())
    std = float(g.std())
    score = (mean * 0.55) + (white_ratio * 420.0) + (std * 1.1)
    return score, mean, white_ratio, std


# =========================
# Location detection (SCORE / LOBBY / OTHER only)
# =========================
def detect_location(screen, img=None, need_selected: bool = False):
    """
    ✅ 최적화 포인트
    - img를 넘기면 screen.capture()를 여기서 하지 않는다 (중복 캡처 방지)
    - 템플릿 매칭은 half-scale로 수행
    - need_selected=False면 selected 판정(하이라이트 스코어) 자체를 생략

    return dict:
      state: 'SCORE' | 'LOBBY' | 'OTHER'
      selected_name: 'PRACTICE' | 'QUIT' | None
      scores: debug dict
    """
    _ensure_lobby_templates_loaded()

    if img is None:
        img = screen.capture()

    # 1) SCORE는 확정 판정
    try:
        if screen.is_score_screen(img):
            return {"state": "SCORE", "selected_name": None, "scores": {}}
    except Exception:
        pass

    # 2) LOBBY는 "오른쪽 메뉴 템플릿 존재"로만 확정
    menu_roi = _roi(img, 0.55, 0.28, 0.98, 0.92)
    menu_gray = cv2.cvtColor(menu_roi, cv2.COLOR_BGR2GRAY)

    # half-scale로 줄여서 matchTemplate 비용 감소
    menu_gray_half = cv2.resize(menu_gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    practice_t = TEMPLATES_HALF.get("practice", None)
    quit_t = TEMPLATES_HALF.get("quit", None)

    practice_tm = _match_template(menu_gray_half, practice_t) if practice_t is not None else 0.0
    quit_tm = _match_template(menu_gray_half, quit_t) if quit_t is not None else 0.0

    menu_present = (max(practice_tm, quit_tm) >= 0.70)

    if not menu_present:
        return {
            "state": "OTHER",
            "selected_name": None,
            "scores": {"practice_tm": practice_tm, "quit_tm": quit_tm},
        }

    # selected가 필요 없으면 여기서 끝
    if not need_selected:
        return {
            "state": "LOBBY",
            "selected_name": None,
            "scores": {"practice_tm": practice_tm, "quit_tm": quit_tm},
        }

    # 3) LOBBY 내부에서만 selected 추정(verify용)
    practice_roi = _roi(img, 0.67, 0.40, 0.97, 0.58)
    quit_roi = _roi(img, 0.67, 0.76, 0.97, 0.92)

    pr_score, pr_mean, pr_wr, pr_std = _menu_highlight_score(practice_roi)
    qt_score, qt_mean, qt_wr, qt_std = _menu_highlight_score(quit_roi)

    selected = None
    margin = 12.0
    if pr_score >= qt_score + margin:
        selected = "PRACTICE"
    elif qt_score >= pr_score + margin:
        selected = "QUIT"

    return {
        "state": "LOBBY",
        "selected_name": selected,
        "scores": {
            "practice_tm": practice_tm,
            "quit_tm": quit_tm,
            "pr_score": pr_score,
            "qt_score": qt_score,
            "pr_mean": pr_mean,
            "qt_mean": qt_mean,
            "pr_white_ratio": pr_wr,
            "qt_white_ratio": qt_wr,
            "pr_std": pr_std,
            "qt_std": qt_std,
        },
    }


# =========================
# Core routines
# =========================
def enter_practice_from_cursor():
    """
    로비에서 Practice가 선택되어 있다는 가정 하에
    Z를 6번 눌러서 연습 모드 진입.
    """
    print("[MENU] FAST practice entry")
    if not focus_touhou_window():
        print("[MENU][ERR] focus failed -> skip key inputs")
        return
    time.sleep(0.05)

    t0 = time.time()
    for i in range(6):
        if _abort_requested():
            print("[MENU][STOP] abort requested during practice entry")
            return
        tap_scancode(SC_Z, label=f"Z{i+1}/6", press=0.02, gap=0.02)
        time.sleep(0.7)
    dt = time.time() - t0
    print(f"[MENU] Z x6 done in {dt:.3f}s")


def recover_from_score_to_lobby(screen, max_sec=3.0) -> bool:
    print("[MENU][RECOVER_SCORE] start")
    if not focus_touhou_window():
        print("[MENU][ERR] focus failed -> skip key inputs")
        return False
    time.sleep(0.05)

    t0 = time.time()
    z_sent = False

    while (time.time() - t0) < max_sec:
        if _abort_requested():
            print("[MENU][STOP] abort requested during score recovery")
            return False
        img = screen.capture()
        try:
            if not screen.is_score_screen(img):
                print("[MENU][RECOVER_SCORE] done")
                return True
        except Exception:
            pass

        # SCORE면 X 연타
        tap_scancode(SC_X, label="X(score)", press=0.02, gap=0.02)
        time.sleep(0.06)

        # ✅ 한 번만 Z (예: 0.6초 지나면 1회)
        if (not z_sent) and (time.time() - t0) >= 0.6:
            tap_scancode(SC_Z, label="Z(score_once)", press=0.02, gap=0.02)
            time.sleep(0.12)
            z_sent = True

    print("[MENU][RECOVER_SCORE] timeout")
    return False



def recover_to_lobby(
    screen,
    max_sec: float = 10.0,
    other_x_presses: int = 6,
    other_x_interval: float = 0.4,
) -> bool:
    """
    철학:
      - 우리가 확정하는 건 LOBBY / SCORE 뿐.
      - 나머지는 OTHER로 보고 'X 연타'로 무조건 로비로 보낸다.
      - SCORE면 recover_from_score_to_lobby()를 우선 사용.
      - OTHER면 X를 일정 횟수/간격으로 눌러 탈출.

    ✅ 최적화:
    - 루프당 capture 1회
    - detect_location(screen, img=, need_selected=False)로
      selected(하이라이트 스코어) 계산은 생략
    """
    print("[MENU][RECOVER_LOBBY] start")
    if not focus_touhou_window():
        print("[MENU][ERR] focus failed -> cannot send keys")
        return False

    t0 = time.time()
    cycles = 0

    while (time.time() - t0) < max_sec:
        if _abort_requested():
            print("[MENU][STOP] abort requested during lobby recovery")
            return False
        img = screen.capture()
        st = detect_location(screen, img=img, need_selected=False)
        state = st.get("state")

        if state == "LOBBY":
            print(f"[MENU][RECOVER_LOBBY] done -> LOBBY (cycles={cycles})")
            return True

        if state == "SCORE":
            recover_from_score_to_lobby(screen, max_sec=3.0)
            time.sleep(0.20)
            cycles += 1
            continue

        # OTHER: X burst
        for i in range(other_x_presses):
            if _abort_requested():
                print("[MENU][STOP] abort requested during OTHER->LOBBY inputs")
                return False
            tap_scancode(SC_X, label=f"X(back){cycles}-{i+1}", press=0.02, gap=0.02)
            time.sleep(other_x_interval)

            # 마지막에 가끔 1번만 Z
            # if i == other_x_presses - 1 and (cycles % 3) == 2:
            #     tap_scancode(SC_Z, label=f"Z(confirm){cycles}", press=0.02, gap=0.02)
            #     time.sleep(0.15)

        cycles += 1

    print(f"[MENU][RECOVER_LOBBY] timeout (cycles={cycles})")
    return False


def ensure_practice_cursor_from_lobby(screen, verify=True, max_try=3) -> bool:
    """
    로비에서 Practice 커서 정렬:
      - Quit로 기준점(X 1회)
      - UP 5회로 Practice로
      - verify면 detect_location().selected_name == PRACTICE 확인

    ✅ 최적화:
    - verify=True인 경우에만 selected 판정(need_selected=True)을 수행.
    - verify 전 상태 확인은 need_selected=False로 가볍게.
    """
    if not focus_touhou_window():
        print("[MENU][ALIGN] focus failed -> cannot send keys")
        return False

    for attempt in range(max_try):
        if _abort_requested():
            print("[MENU][STOP] abort requested during cursor align")
            return False
        img0 = screen.capture()
        st = detect_location(screen, img=img0, need_selected=False)
        if st.get("state") != "LOBBY":
            print(f"[MENU][ALIGN] not in LOBBY (state={st.get('state')})")
            return False

        tap_scancode(SC_X, label="X(to Quit baseline)", press=0.02, gap=0.02)
        time.sleep(0.25)

        for i in range(5):
            if _abort_requested():
                print("[MENU][STOP] abort requested during cursor align inputs")
                return False
            tap_scancode(SC_UP, extended=True, label=f"UP{i+1}/5", press=0.02, gap=0.02)
            time.sleep(0.12)

        time.sleep(0.30)

        if not verify:
            return True

        img1 = screen.capture()
        st2 = detect_location(screen, img=img1, need_selected=True)
        ok = (st2.get("state") == "LOBBY" and st2.get("selected_name") == "PRACTICE")
        print(f"[MENU][ALIGN] verify attempt {attempt+1}/{max_try} -> {st2.get('selected_name')} ok={ok}")
        if ok:
            return True

        time.sleep(0.25)

    print("[MENU][ALIGN] failed to align PRACTICE after retries")
    return False


def boot_into_practice(screen, max_sec_lobby: float = 10.0) -> bool:
    """
    에피소드 시작 전 공통 루틴(핵심):
      1) 어떤 상태든 recover_to_lobby()로 로비 확보
      2) 로비에서 practice 커서 정렬
      3) enter_practice_from_cursor()
    """
    print("[MENU][BOOT2] start -> make LOBBY then enter PRACTICE")
    if _abort_requested():
        print("[MENU][STOP] abort requested before boot")
        raise KeyboardInterrupt("ESC/P pressed in menu")

    ok = recover_to_lobby(screen, max_sec=max_sec_lobby)
    if not ok:
        if _abort_requested():
            raise KeyboardInterrupt("ESC/P pressed in menu")
        print("[MENU][BOOT2] failed: cannot recover to lobby")
        return False

    ok = ensure_practice_cursor_from_lobby(screen, verify=True, max_try=3)
    if not ok:
        print("[MENU][BOOT2] failed: cannot align practice cursor (continue anyway)")
        if _abort_requested():
            raise KeyboardInterrupt("ESC/P pressed in menu")

    enter_practice_from_cursor()
    if _abort_requested():
        raise KeyboardInterrupt("ESC/P pressed in menu")
    return True
