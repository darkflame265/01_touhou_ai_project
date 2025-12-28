# env/game_env_util/obs_pack.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObsPackConfig:
    """Obs/frame_stack packing 관련 설정 (필요하면 확장)"""
    # 현재는 비워둬도 됨
    pass


class ObsPacker:
    """
    역할:
      - obs/state를 CHW로 정규화
      - frame_stack을 채널 concat으로 pack
      - ep_total_reward 누적
      - frame_stack 초기화(filling)
    """

    def __init__(self, state, cfg: Optional[ObsPackConfig] = None):
        self.s = state
        self.cfg = cfg or ObsPackConfig()

    # -------------------------
    # shape utils
    # -------------------------
    def as_chw(self, obs: np.ndarray) -> np.ndarray:
        if obs is None:
            return None
        obs = np.asarray(obs)
        if obs.ndim == 2:
            return obs[None, :, :]
        if obs.ndim == 3:
            return obs
        raise ValueError(f"Unexpected obs shape: {obs.shape}")

    def pack_frames_concat(self) -> np.ndarray:
        if len(self.s.frame_stack) == 0:
            return self.as_chw(self.s.prev_state)
        frames = [self.as_chw(x) for x in list(self.s.frame_stack)]
        return np.concatenate(frames, axis=0)

    # -------------------------
    # stack ops
    # -------------------------
    def reset_stack_fill(self, init_state_chw: np.ndarray):
        """frame_stack을 init_state로 frame_stack_size만큼 채움"""
        self.s.prev_state = self.as_chw(init_state_chw)
        self.s.frame_stack.clear()
        for _ in range(int(self.s.frame_stack_size)):
            self.s.frame_stack.append(self.s.prev_state)

    def push_prev_state(self, state_chw: np.ndarray):
        """prev_state 갱신 + frame_stack append"""
        self.s.prev_state = self.as_chw(state_chw)
        self.s.frame_stack.append(self.s.prev_state)

    # -------------------------
    # episode reward accumulate
    # -------------------------
    def ep_add(self, x: float):
        try:
            self.s.ep_total_reward += float(x)
        except Exception:
            pass
