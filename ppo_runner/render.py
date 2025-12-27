# ppo_runner/render.py
import cv2

def apply_no_render(env):
    """
    env 내부 debug 플래그들을 최대한 끈다.
    (인게임 렉 줄이기 목적)
    """
    # obs_builder 쪽
    try:
        if hasattr(env, "obs") and hasattr(env.obs, "show_reimu_debug"):
            env.obs.show_reimu_debug = False
    except Exception:
        pass

    try:
        if hasattr(env, "obs") and hasattr(env.obs, "show_obs_debug"):
            env.obs.show_obs_debug = False
    except Exception:
        pass

    # env.debug 같은 별도 디버그 객체가 있을 수도 있어 방어적으로 off
    dbg = getattr(env, "debug", None)
    if dbg is not None:
        for name, val in (
            ("show_tracker_debug", False),
            ("show_roi_window", False),
            ("show_full_window", False),
            ("show_mask_window", False),
        ):
            try:
                if hasattr(dbg, name):
                    setattr(dbg, name, val)
            except Exception:
                pass


def pump_cv_events_once():
    """
    OpenCV 창 띄웠을 때 이벤트 펌프만 (키 처리 X)
    """
    try:
        cv2.waitKey(1)
    except Exception:
        pass
