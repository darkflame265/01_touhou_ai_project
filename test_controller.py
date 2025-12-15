import time
from env.controller import press_keys, release_all
from env.actions import ACTIONS

print("3초 후 시작합니다. 게임 창을 클릭하세요.")
time.sleep(3)

for action in ACTIONS:
    print("action:", action)
    press_keys(action)
    time.sleep(1)

release_all()
print("테스트 종료")
