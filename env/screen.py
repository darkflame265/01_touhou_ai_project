import win32gui
import mss
import cv2
import numpy as np


def find_touhou_window():
    target_hwnd = None

    def enum_handler(hwnd, _):
        nonlocal target_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "동방홍마향" in title:
                target_hwnd = hwnd

    win32gui.EnumWindows(enum_handler, None)
    return target_hwnd

class Screen:
    def __init__(self):
        self.sct = mss.mss()
        self.hwnd = find_touhou_window()

        if not self.hwnd:
            raise Exception("동방홍마향 창을 찾을 수 없음")

        title = win32gui.GetWindowText(self.hwnd)
        rect = win32gui.GetWindowRect(self.hwnd)

        print("[DEBUG] 잡은 창 제목:", title)
        print("[DEBUG] 창 좌표 (left, top, right, bottom):", rect)

    def capture(self):
        # 🔹 최소화된 창이면 복원
        win32gui.ShowWindow(self.hwnd, 9)  # SW_RESTORE
        win32gui.SetForegroundWindow(self.hwnd)

        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)

        monitor = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top
        }

        img = np.array(self.sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        normalized = (resized / 255.0).astype(np.float32)
        return normalized


    def detect_death(self, img):
        """
        return: (hit, gameover)
        hit: 피격(1~2번째 죽음)
        gameover: 전체 화면 백색(3번째, 게임오버)
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1️⃣ 전체 화면 평균 밝기 (게임오버)
        full_brightness = gray.mean() / 255.0
        gameover = full_brightness > 0.80

        # 2️⃣ 플레이어 근처 ROI (피격)
        x1 = int(w * 0.35)
        x2 = int(w * 0.65)
        y1 = int(h * 0.60)
        y2 = int(h * 0.95)

        roi = gray[y1:y2, x1:x2]
        bright_ratio = (roi > 230).mean()
        hit = bright_ratio > 0.02

        # 🔹 이벤트 기반 디버그 출력
        if gameover:
            print("[DEBUG] GAMEOVER detected",
                  f"(full={full_brightness:.3f})")
            return True, True

        if hit:
            print("[DEBUG] HIT detected",
                  f"(roi={bright_ratio:.3f})")
            return True, False

        # 🔹 아무 이벤트 없을 때는 숫자만 (원하면 주석 처리 가능)
        print(f"[DEBUG] full={full_brightness:.3f}, roi={bright_ratio:.3f}")

        return False, False

    def get_playfield_gray(self, img_bgr):
        """
        오른쪽 UI 패널을 제외한 '플레이 필드' 영역만 회색조로 반환
        (비율은 네 창 기준으로 적당히 잡은 값. 필요하면 조정)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 오른쪽 UI 패널이 대략 30~35% 정도를 차지하니 잘라내기
        x2 = int(w * 0.70)
        play = gray[:, :x2]
        return play


    def playfield_motion_score(self, prev_play_gray, curr_play_gray):
        """
        프레임 간 변화량(움직임 정도)을 0~1 근사치로 반환
        """
        diff = cv2.absdiff(prev_play_gray, curr_play_gray)
        return float(diff.mean()) / 255.0
