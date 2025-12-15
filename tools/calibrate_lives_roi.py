import json
import os
import cv2

from env.screen import Screen

CONFIG_PATH = os.path.join("env", "ui_config.json")

def main():
    screen = Screen()
    img = screen.capture()

    print("잔기(목숨) UI 영역을 마우스로 드래그해서 선택하세요. Enter로 확정, Esc로 취소.")
    cv2.namedWindow("Select LIVES UI ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select LIVES UI ROI", img.shape[1], img.shape[0])

    r = cv2.selectROI("Select LIVES UI ROI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, r)
    if w == 0 or h == 0:
        print("선택이 취소되었습니다.")
        return

    cfg = {"lives_roi": {"x": x, "y": y, "w": w, "h": h}}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("저장 완료:", CONFIG_PATH)
    print(cfg)

if __name__ == "__main__":
    main()
