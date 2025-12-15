# env/actions.py
from enum import Enum

class Action(Enum):
    NONE = []

    LEFT = ["LEFT"]
    RIGHT = ["RIGHT"]
    UP = ["UP"]
    DOWN = ["DOWN"]

    SLOW_LEFT = ["LEFT"]
    SLOW_RIGHT = ["RIGHT"]
    SLOW_UP = ["UP"]
    SLOW_DOWN = ["DOWN"]


ACTIONS = list(Action)
