# ppo_runner/runner.py
from __future__ import annotations

import os
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch

from env.game_env import GameEnv
from env.controller import release_all, set_attack_hold, cleanup_inputs_on_exit
from env.menu import boot_into_practice
from env.actions import ACTIONS

from agents.dqn_agent import DQNAgent

from sim.sim_env import SimEnv, SimConfig

from .hotkeys import esc_pressed
from .render import apply_no_render, pump_cv_events_once
from .stats_log import (
    ensure_stats_header,
    append_run_header,
    maybe_update_records,
    update_stats_in_file,
    stats_one_line,
)


# ----------------------------
# Utilities
# ----------------------------
def safe_release_inputs():
    """공격홀드/키 stuck 방지"""
    try:
        set_attack_hold(False)
    except Exception:
        pass
    try:
        release_all()
    except Exception:
        pass


def _safe_save_checkpoint(agent, ckpt_path: str) -> bool:
    try:
        ret = agent.save(ckpt_path)
        return bool(ret) if isinstance(ret, bool) else True
    except Exception as e:
        print(f"[WARN] checkpoint save failed (ignored): {e}")
        return False


def _try_clear_agent_rollout(agent):
    for name in ("clear", "reset_buffer", "reset_storage", "clear_buffer", "clear_rollout"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


def _find_bomb_index() -> int | None:
    for i, a in enumerate(ACTIONS):
        if getattr(a, "name", "") == "BOMB":
            return i
    return None


def _find_stop_index() -> int | None:
    for i, a in enumerate(ACTIONS):
        if getattr(a, "name", "") == "SLOW_STOP":
            return i
    return None


def _find_stop_name() -> str | None:
    return "SLOW_STOP" if _find_stop_index() is not None else None


def _build_action_mask_from_img(env: GameEnv, img: np.ndarray | None) -> np.ndarray:
    mask = np.ones((len(ACTIONS),), dtype=np.bool_)

    try:
        if img is not None and hasattr(env, "masker") and env.masker is not None:
            m2 = env.masker.get_action_mask(img)
            if m2 is not None and len(m2) == len(mask):
                mask &= m2.astype(np.bool_, copy=False)
    except Exception:
        pass

    bidx = _find_bomb_index()
    if bidx is not None:
        mask[int(bidx)] = False

    if not bool(mask.any()):
        mask[:] = True
        if bidx is not None:
            mask[int(bidx)] = False

    return mask


def _build_action_mask_for_sim() -> np.ndarray:
    mask = np.zeros((len(ACTIONS),), dtype=np.bool_)

    n = min(8, len(ACTIONS))
    mask[:n] = True

    sidx = _find_stop_index()
    if sidx is not None:
        mask[int(sidx)] = True

    bidx = _find_bomb_index()
    if bidx is not None:
        mask[int(bidx)] = False

    if not bool(mask.any()):
        mask[:n] = True
        if sidx is not None:
            mask[int(sidx)] = True
        if bidx is not None:
            mask[int(bidx)] = False

    return mask


def _clamp_action_idx(i: int) -> int:
    try:
        x = int(i)
    except Exception:
        return 0
    if x < 0:
        return 0
    if x >= len(ACTIONS):
        return len(ACTIONS) - 1
    return x


def _agent_set_eval_mode(agent) -> None:
    for attr in ("actor_critic", "net", "model", "policy", "policy_net", "q", "q_net"):
        m = getattr(agent, attr, None)
        if m is not None and hasattr(m, "eval"):
            try:
                m.eval()
            except Exception:
                pass


# ----------------------------
# Sim frame stacker (runner-side)
# ----------------------------
class _SimFrameStacker:
    def __init__(self, obs_channels: int = 4, stack: int = 4):
        self.c = int(obs_channels)
        self.k = int(stack)
        self.buf: deque[np.ndarray] = deque(maxlen=self.k)

    def reset(self, first_obs: np.ndarray) -> np.ndarray:
        self.buf.clear()
        o = np.asarray(first_obs, dtype=np.float32)
        for _ in range(self.k):
            self.buf.append(o)
        return self._pack()

    def step(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32)
        self.buf.append(o)
        return self._pack()

    def _pack(self) -> np.ndarray:
        frames = list(self.buf)
        stacked = np.concatenate(frames, axis=0)  # (16,H,W)
        return stacked.astype(np.float32, copy=False)


# ----------------------------
# Load agent/env
# ----------------------------
def _load_agent_and_env(ckpt_path: str, no_render: bool, use_sim: bool, eval_mode: bool):
    # TF32 (CUDA일 때만)
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass

    if use_sim:
        cfg = SimConfig(seed=0)

        # ✅ SIM 학습에서는 렌더가 CPU를 크게 잡아먹어서 기본 OFF
        # 사용자가 --no-render를 줬으면 당연히 OFF, 안 줘도 OFF가 기본값.
        cfg.render = False

        env = SimEnv(cfg)
        obs_channels = 4
        stack_size = 4
    else:
        env = GameEnv(screen_mode="low")
        if no_render:
            apply_no_render(env)
        obs_channels = int(getattr(env.obs, "obs_channels", 1))
        stack_size = int(getattr(env.s, "frame_stack_size", 4))

    input_channels = obs_channels * stack_size

    # ✅ Replay: SIM에서도 20k로 키움 (uint8 저장 기준)
    replay_size = 20_000

    # ✅ 추천 튜닝: batch↑, train 빈도↑, eps decay↑
    agent = DQNAgent(
        input_channels=input_channels,
        num_actions=len(ACTIONS),
        obs_channels_per_frame=obs_channels,

        lr=2e-4,
        gamma=0.99,
        batch_size=256,            # 64 -> 128
        grad_accum_steps=4,   # ✅ 추가


        replay_size=replay_size,   # 20k
        learning_starts=4000,      # 2000 -> 4000 (초기 분포 조금 더 모으기)

        train_every_steps=16,       # 16 -> 8 (학습 더 자주)
        target_update_every_steps=2000,
        double_dqn=True,

        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=300_000,   # 200k -> 300k (커리큘럼 후반 적응 여지)
    )

    if os.path.exists(ckpt_path):
        agent.load(ckpt_path, load_optimizer=False)
        print(f"[DQN] checkpoint loaded: {ckpt_path}")
    else:
        print("[DQN] no checkpoint found, training from scratch")

    if eval_mode:
        agent.eps_start = 0.0
        agent.eps_end = 0.0

    print(f"[DQN] input_channels={input_channels} (obs_channels={obs_channels} * stack={stack_size})")
    print(f"[DQN] replay_size={replay_size}")
    try:
        print(f"[HW] Agent device={agent.device} | AMP={getattr(agent, 'use_amp', False)}")
    except Exception:
        pass

    return env, agent


# ----------------------------
# Main loop
# ----------------------------
def run(
    episodes: int = 1,
    no_render: bool = False,
    eval_mode: bool = False,
    ckpt_path: str = "checkpoints/dqn_v1.pth",
    use_sim: bool = False,
    save_every_episodes: int = 1,
    stats_every_episodes: int = 1,
    print_survival_every: int = 10,
    survival_window: int = 50,
):
    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    log_path = os.path.join(os.path.dirname(ckpt_path) if ckpt_dir else ".", f"{pth_name}_episode_log.txt")

    stats = ensure_stats_header(log_path)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_run_header(log_path, run_ts, int(episodes), bool(eval_mode), stats)
    print(stats_one_line(stats))

    env, agent = _load_agent_and_env(ckpt_path, no_render=no_render, use_sim=use_sim, eval_mode=eval_mode)

    if eval_mode:
        _agent_set_eval_mode(agent)

    sim_stacker = _SimFrameStacker(obs_channels=4, stack=4) if use_sim else None
    sim_action_mask = _build_action_mask_for_sim() if use_sim else None

    # ✅ GPU 점유율 올리기: update 트리거 1번에 여러 번 학습
    # - SIM: CPU가 env에 묶이기 쉬워서, 학습을 덩어리로 돌려 GPU를 더 활용
    updates_per_trigger = 1

    best_wall = 0.0
    best_game = 0.0
    best_reward = -1e9
    surv_hist = deque(maxlen=int(max(1, survival_window)))

    print("\n[INFO] ESC 중단: Windows 전역 감지(GetAsyncKeyState)")
    time.sleep(0.05)

    stop_requested = False
    stop_name = _find_stop_name()
    do_io = (not eval_mode)

    try:
        for ep in range(1, int(episodes) + 1):
            if esc_pressed():
                print("[STOP] ESC pressed before episode start -> stopping.")
                stop_requested = True
                break

            print(f"\n========== EPISODE {ep}/{episodes} ==========")

            if not use_sim:
                safe_release_inputs()
                time.sleep(0.05)

                print("[MENU] [practice 준비/진입 중.]")
                ok = boot_into_practice(env.screen, max_sec_lobby=12.0)
                if not ok:
                    print("[EP_PREP][WARN] boot_into_practice failed (continue)")
                print("[MENU] [practice 준비/진입 완료]")

                safe_release_inputs()
                time.sleep(0.05)

            raw = env.reset()
            state = sim_stacker.reset(raw) if use_sim else raw

            if (not no_render) and (not use_sim):
                pump_cv_events_once()

            ep_t0 = time.time()
            done = False
            total_reward = 0.0
            steps = 0
            slow_count = 0
            stop_count = 0
            aborted = False
            local_updates = 0

            while not done:
                if esc_pressed():
                    stop_requested = True
                    aborted = True
                    print("[STOP] ESC pressed -> aborting NOW (NO SAVE/NO UPDATE for this episode).")
                    if not use_sim:
                        safe_release_inputs()
                    done = True
                    break

                # -------- current-state mask (for action selection)
                if use_sim:
                    action_mask = sim_action_mask
                else:
                    img_for_mask = getattr(env.s, "last_action_mask_img", None)
                    action_mask = _build_action_mask_from_img(env, img_for_mask)

                action_idx, log_prob, value = agent.select_action(state, action_mask=action_mask)
                action_idx = _clamp_action_idx(action_idx)

                # -------- step
                if use_sim:
                    raw_next, reward, done, _info = env.step(action_idx)
                    next_state = sim_stacker.step(raw_next)
                    exec_idx = int(action_idx)
                else:
                    next_state, reward, done = env.step(action_idx)
                    exec_idx = getattr(env.s, "exec_action_idx", action_idx)

                exec_name = ACTIONS[int(exec_idx)].name

                if exec_name.startswith("SLOW"):
                    slow_count += 1
                if stop_name and exec_name == stop_name:
                    stop_count += 1

                # -------- next-state mask (for masked DQN target)
                if use_sim:
                    next_action_mask = sim_action_mask
                else:
                    img_for_next_mask = getattr(env.s, "last_action_mask_img", None)
                    next_action_mask = _build_action_mask_from_img(env, img_for_next_mask)

                if do_io:
                    agent.store(
                        state,
                        int(action_idx),
                        reward,
                        done,
                        log_prob,
                        value,
                        action_mask=action_mask,
                        next_state=next_state,
                        next_action_mask=next_action_mask,
                    )

                state = next_state
                total_reward += float(reward)
                steps += 1

                # ✅ 학습 덩어리로 실행
                if do_io and agent.should_update():
                    for _ in range(int(max(1, updates_per_trigger))):
                        out = agent.update(last_state=state, last_done=False)
                        if out is None:
                            break
                        local_updates += 1

                if (not no_render) and (not use_sim):
                    pump_cv_events_once()

            wall_sec = time.time() - ep_t0
            game_sec = float(steps) / 60.0
            surv_hist.append(float(wall_sec))

            best_game = max(best_game, float(game_sec))
            best_wall = max(best_wall, float(wall_sec))
            best_reward = max(best_reward, float(total_reward))

            if use_sim:
                sps = steps / max(wall_sec, 1e-9)
                speed_x = sps / 60.0
                print(
                    f"[SIM] wall={wall_sec:.3f}s game={game_sec:.3f}s "
                    f"steps={steps} SPS={sps:.0f} (~{speed_x:.1f}x vs 60fps)"
                )

            if (ep % int(max(1, print_survival_every))) == 0 or ep == 1:
                avg_surv = float(np.mean(surv_hist)) if len(surv_hist) else 0.0
                print(
                    f"[SURV] best_wall={best_wall:.2f}s best_game={best_game:.2f}s | "
                    f"avg_wall(last {len(surv_hist)})={avg_surv:.2f}s"
                )

            slow_ratio = slow_count / max(1, steps)
            stop_ratio = (stop_count / max(1, steps)) if stop_name else 0.0

            note_parts = []
            if eval_mode:
                note_parts.append("EVAL_BEST")
            if aborted:
                note_parts.append("ABORTED")
            note = ",".join(note_parts)

            print(
                f"[DQN] episode end | steps={steps} total_reward={total_reward:.1f} "
                f"wall_sec={wall_sec:.2f} game_sec={game_sec:.2f} slow_ratio={slow_ratio:.3f} "
                f"stop_ratio={stop_ratio:.3f} updates(ep)={local_updates} {note}"
            )

            if not do_io:
                if stop_requested or aborted:
                    break
                continue

            ep_tag = f"({ep}/{episodes})"
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{ep_tag}\t{total_reward:.6f}\t{wall_sec:.3f}\t{game_sec:.3f}\t{note}\n")
            except Exception as e:
                print(f"[WARN] episode log append failed (ignored): {e}")

            if aborted:
                _try_clear_agent_rollout(agent)
                print("[STOP] Episode aborted -> stopping.")
                break

            if (ep % int(max(1, stats_every_episodes))) == 0 or ep == int(episodes):
                maybe_update_records(stats, total_reward, wall_sec, run_ts, ep_tag)
                update_stats_in_file(log_path, stats)
            print(stats_one_line(stats))

            if (ep % int(max(1, save_every_episodes))) == 0 or ep == int(episodes):
                ok_save = _safe_save_checkpoint(agent, ckpt_path)
                if ok_save:
                    print("[DQN] checkpoint saved")
                else:
                    print("[WARN] checkpoint save failed -> continue training without stopping")

            if stop_requested:
                print("[STOP] Training stopped by ESC.")
                break

            print(f"[DBG] global_step={agent.global_step} replay={len(agent.replay)} updates_per_trigger={updates_per_trigger}")

            if (not use_sim) and (ep < int(episodes)):
                time.sleep(0.02)

    finally:
        cleanup_inputs_on_exit()
        if not no_render:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if eval_mode:
        print(
            f"\n[EVAL_SUMMARY] best_game_sec={best_game:.2f}s | "
            f"best_wall_sec={best_wall:.2f}s | best_reward={best_reward:.3f}"
        )

    print("\n[DQN] Finished.")
