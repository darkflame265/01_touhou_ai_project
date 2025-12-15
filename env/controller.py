# env/controller.py

import pydirectinput

# 공격 키 (항상 누르고 있을 키)
ATTACK_KEY = "z"

# 이동 키 목록
MOVE_KEYS = {
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
}

def press_keys(action_keys):
    """
    action_keys: ["LEFT", "UP"] 같은 리스트
    """

    # 공격 키는 항상 누른 상태 유지
    pydirectinput.keyDown(ATTACK_KEY)

    # 이동 키 처리
    for name, key in MOVE_KEYS.items():
        if name in action_keys:
            pydirectinput.keyDown(key)
        else:
            pydirectinput.keyUp(key)


def release_all():
    # 이동 키 전부 떼기
    for key in MOVE_KEYS.values():
        pydirectinput.keyUp(key)

    # 공격 키도 떼기 (프로그램 종료 시)
    pydirectinput.keyUp(ATTACK_KEY)
