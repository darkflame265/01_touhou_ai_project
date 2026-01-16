# env/actions.py
from enum import Enum


class Action(Enum):
    """
    ✅ 상시 SLOW 전제:
    - Shift는 set_always_slow(True) 같은 방식으로 환경에서 항상 눌려있다고 가정
    - 여기 액션은 '방향(8방향) + 폭탄(딸깍) + 정지(Stop)' 담당
      -> 총 10개 액션
    """

    # 8방향 이동(느림 이동은 환경에서 항상 유지)
    SLOW_LEFT = ["LEFT"]
    SLOW_RIGHT = ["RIGHT"]
    SLOW_UP = ["UP"]
    SLOW_DOWN = ["DOWN"]

    SLOW_UP_LEFT = ["UP", "LEFT"]
    SLOW_UP_RIGHT = ["UP", "RIGHT"]
    SLOW_DOWN_LEFT = ["DOWN", "LEFT"]
    SLOW_DOWN_RIGHT = ["DOWN", "RIGHT"]

    # 폭탄: 이동과 결합하지 않는 "단독 탭" 액션
    BOMB = ["BOMB"]

    # 정지: 이동키 입력 없음 (Shift는 환경에서 상시 유지 가정)
    SLOW_STOP = []


ACTIONS = list(Action)
