# env/ui_lives.py
import json
import os
import cv2
import numpy as np

CONFIG_PATH = os.path.join("env", "ui_config.json")


def load_lives_roi():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"{CONFIG_PATH} 가 없습니다. 먼저 python -m tools.calibrate_lives_roi 를 실행하세요."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["lives_roi"]


def count_lives_from_img(img_bgr, debug=False):
    roi_cfg = load_lives_roi()
    x, y, w, h = roi_cfg["x"], roi_cfg["y"], roi_cfg["w"], roi_cfg["h"]

    roi = img_bgr[y:y+h, x:x+w]

    # =========================
    # ✅ ROI 스샷: 딱 한 번만 저장
    # =========================
    # ROI 위치 어긋나면 python -m tools.calibrate_lives_roi 로 드래고하여 위치 재설정.
    if not hasattr(count_lives_from_img, "_roi_dumped"):
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite("debug/lives_roi_once.png", roi)
        print("[DEBUG] lives ROI snapshot saved: debug/lives_roi_once.png")
        count_lives_from_img._roi_dumped = True

    # HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 빨강 HSV 범위
    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 노이즈 정리
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 연결요소 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = max(10, int((w * h) * 0.01))
    count = 0
    areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            count += 1
            areas.append(int(area))

    if debug:
        print(f"[DEBUG] ROI=({w}x{h}) labels={num_labels-1} min_area={min_area} kept={count} areas={areas}")
        cv2.imshow("LIVES ROI (BGR)", roi)
        cv2.imshow("LIVES MASK (RED)", mask)
        cv2.waitKey(1)

    return count

