import time
import cv2
from env.screen import Screen
from env.ui_lives import count_lives_from_img

screen = Screen()

time.sleep(2)
print("3초 후부터 잔기 측정 시작. 게임 화면에서 일부러 맞아보면 숫자가 변하는지 확인!")

time.sleep(3)

while True:
    img = screen.capture()
    lives = count_lives_from_img(img, debug=True)
    print("lives:", lives)
    time.sleep(0.2)

    # ESC 누르면 종료(윈도우 떠 있을 때만)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
