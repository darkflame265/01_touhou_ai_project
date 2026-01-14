# env/game_env_util/obs_pack.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObsPackConfig:
    """
    Obs/frame_stack packing 관련 설정

    - drop_dup_frames:
        FrameSkipper가 is_dup=True로 준 프레임을 frame_stack에 넣지 않음.
        (기본 True 권장: 속도/미래 힌트가 dup로 망가지는 걸 줄임)
    - copy_on_push:
        deque에 push할 때 항상 copy()해서 참조 공유로 인한 스택 오염 방지
    - pad_to_stack:
        stack이 부족하면 마지막 프레임으로 패딩해서 항상 고정 shape 반환
    """
    drop_dup_frames: bool = True
    copy_on_push: bool = True
    pad_to_stack: bool = True


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

    def _maybe_copy(self, x: np.ndarray) -> np.ndarray:
        if x is None:
            return None
        return x.copy() if self.cfg.copy_on_push else x

    # -------------------------
    # packing
    # -------------------------
    def pack_frames_concat(self) -> np.ndarray:
        """
        frame_stack (deque)에 쌓인 프레임들을 axis=0으로 concat.
        항상 (C * frame_stack_size, H, W) 형태를 반환하도록 패딩 가능.
        """
        fs = int(getattr(self.s, "frame_stack_size", 1))
        fs = max(1, fs)

        # stack이 비어 있으면 prev_state로
        if len(self.s.frame_stack) == 0:
            base = self.as_chw(self.s.prev_state)
            if base is None:
                raise ValueError("prev_state is None and frame_stack is empty.")
            # pad
            if self.cfg.pad_to_stack and fs > 1:
                frames = [base] * fs
                return np.concatenate(frames, axis=0)
            return base

        frames = [self.as_chw(x) for x in list(self.s.frame_stack)]
        last = frames[-1]

        # 부족하면 마지막 프레임으로 패딩
        if self.cfg.pad_to_stack and len(frames) < fs:
            frames.extend([last] * (fs - len(frames)))

        # 초과하면 최신 fs개만
        if len(frames) > fs:
            frames = frames[-fs:]

        return np.concatenate(frames, axis=0)

    # -------------------------
    # stack ops
    # -------------------------
    def reset_stack_fill(self, init_state_chw: np.ndarray):
        """frame_stack을 init_state로 frame_stack_size만큼 채움"""
        init = self.as_chw(init_state_chw)
        init = self._maybe_copy(init)

        self.s.prev_state = init
        self.s.frame_stack.clear()

        fs = int(getattr(self.s, "frame_stack_size", 1))
        fs = max(1, fs)
        for _ in range(fs):
            # 각 원소가 같은 참조를 공유하지 않도록 copy
            self.s.frame_stack.append(self._maybe_copy(init))

    def push_prev_state(self, state_chw: np.ndarray, is_dup: bool = False):
        """
        prev_state 갱신 + frame_stack append

        - is_dup=True 이고 cfg.drop_dup_frames=True 이면 push를 스킵.
          (속도 힌트용 프레임스택은 dup가 들어가면 학습에 악영향이 큼)
        """
        if bool(is_dup) and bool(self.cfg.drop_dup_frames):
            # prev_state는 유지(혹은 갱신) 선택지가 있는데,
            # 여기서는 "스택과 일관"을 위해 prev_state도 갱신하지 않음.
            return

        x = self.as_chw(state_chw)
        x = self._maybe_copy(x)
        self.s.prev_state = x
        self.s.frame_stack.append(x)

    # -------------------------
    # episode reward accumulate
    # -------------------------
    def ep_add(self, x: float):
        try:
            self.s.ep_total_reward += float(x)
        except Exception:
            pass
