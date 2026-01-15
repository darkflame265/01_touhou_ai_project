# ppo_runner/runner.py
from __future__ import annotations

import os
import time
from datetime import datetime
from collections import Counter, deque

import cv2
import numpy as np
import platform
import torch
import subprocess

from env.game_env import GameEnv
from env.controller import release_all, set_attack_hold, cleanup_inputs_on_exit
from env.menu import boot_into_practice
from env.actions import ACTIONS
from agents.ppo_agent import PPOAgent

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


def _torch_mem_str(device: str = "cuda") -> str:
    if (not torch.cuda.is_available()) or (device != "cuda"):
        return "CUDA: N/A"
    try:
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        max_reserv = torch.cuda.max_memory_reserved() / (1024**2)
        return (
            f"VRAM alloc={alloc:.0f}MiB reserv={reserv:.0f}MiB "
            f"(max alloc={max_alloc:.0f}MiB max reserv={max_reserv:.0f}MiB)"
        )
    except Exception as e:
        return f"VRAM: ERROR({e})"


def _nvidia_smi_query() -> str | None:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if not out:
            return None
        lines = [x.strip() for x in out.splitlines() if x.strip()]
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) >= 5:
            util_gpu, util_mem, mem_used, mem_total, temp = parts[:5]
            return f"nvidia-smi util={util_gpu}% memUtil={util_mem}% mem={mem_used}/{mem_total}MiB temp={temp}C"
        return f"nvidia-smi raw={lines[0]}"
    except Exception:
        return None


def _print_runtime_gpu_stats(agent, prefix: str = "[GPU]"):
    dev = getattr(agent, "device", "cpu")
    if isinstance(dev, str) and dev.startswith("cuda") and torch.cuda.is_available():
        print(f"{prefix} {_torch_mem_str('cuda')}")
    else:
        print(f"{prefix} CUDA: OFF (agent.device={dev})")

    smi = _nvidia_smi_query()
    if smi is not None:
        print(f"{prefix} {smi}")
    else:
        print(f"{prefix} nvidia-smi: N/A (not found or failed)")


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


def _build_action_mask_from_img(env: GameEnv, img: np.ndarray | None) -> np.ndarray:
    """
    GameEnv용 마스크:
    - runner는 캡처하지 않고 env가 저장한 last_action_mask_img 사용
    """
    mask = np.ones((len(ACTIONS),), dtype=np.bool_)

    try:
        if img is not None and hasattr(env, "masker") and env.masker is not None:
            m2 = env.masker.get_action_mask(img)
            if m2 is not None and len(m2) == len(mask):
                mask &= m2.astype(np.bool_, copy=False)
    except Exception:
        pass

    # BOMB는 무조건 금지
    bidx = _find_bomb_index()
    if bidx is not None:
        mask[int(bidx)] = False

    # 전부 False면 복구
    if not bool(mask.any()):
        mask[:] = True
        if bidx is not None:
            mask[int(bidx)] = False

    return mask


def _build_action_mask_for_sim() -> np.ndarray:
    """
    SimEnv용 마스크:
    - "8방향 이동만 허용"
    - ACTIONS가 더 많아도 첫 8개만 True, 나머지 False
    """
    mask = np.zeros((len(ACTIONS),), dtype=np.bool_)
    n = min(8, len(ACTIONS))
    mask[:n] = True

    bidx = _find_bomb_index()
    if bidx is not None:
        mask[int(bidx)] = False

    if not bool(mask.any()):
        mask[:n] = True
        if bidx is not None:
            mask[int(bidx)] = False
    return mask


# ----------------------------
# Sim frame stacker (runner-side)
# ----------------------------
class _SimFrameStacker:
    """
    SimEnv raw obs: (4,128,128)
    PPO input: (16,128,128) = 4ch * stack4
    """
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
def _load_agent_and_env(ckpt_path: str, no_render: bool, use_sim: bool):
    if use_sim:
        # ✅ SimConfig 기본 render=True는 유지
        cfg = SimConfig(seed=0)

        # ✅ CLI가 최종 결정: --no-render면 cfg.render만 꺼준다
        if no_render:
            cfg.render = False

        env = SimEnv(cfg)

        # sim은 GameEnv처럼 env.obs/env.s가 없으니 여기서 고정
        obs_channels = 4
        stack_size = 4

    else:
        env = GameEnv(screen_mode="low")
        if no_render:
            apply_no_render(env)

        obs_channels = int(getattr(env.obs, "obs_channels", 1))
        stack_size = int(getattr(env.s, "frame_stack_size", 4))

    # ✅ input_channels 계산은 공통
    input_channels = obs_channels * stack_size

    agent = PPOAgent(
        input_channels=input_channels,
        num_actions=len(ACTIONS),
        obs_channels_per_frame=obs_channels,
    )

    if os.path.exists(ckpt_path):
        agent.load(ckpt_path, load_optimizer=False)
        print(f"[PPO] checkpoint loaded: {ckpt_path}")
    else:
        print("[PPO] no checkpoint found, training from scratch")

    # 하이퍼파라미터 override (네 기존 유지)
    agent.ent_coef = 0.04
    agent.ent_min = 0.005
    agent.ent_decay = 0.9995
    agent.ent_warmup_updates = 30
    agent.clip_eps = 0.15
    agent.rollout_steps = 128
    agent.update_epochs = 5

    print(f"[PPO] input_channels={input_channels} (obs_channels={obs_channels} * stack={stack_size})")
    print(
        "[PPO][OVERRIDE] "
        f"ent_coef={agent.ent_coef:.3f}, ent_min={agent.ent_min:.3f}, "
        f"clip_eps={agent.clip_eps:.2f}, rollout_steps={agent.rollout_steps}, update_epochs={agent.update_epochs}"
    )
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
    ckpt_path: str = "checkpoints/lunatic_v1_ch4.pth",
    use_sim: bool = False,

    # monitoring
    monitor_gpu: bool = False,
    monitor_every_steps: int = 400,
    monitor_every_sec: float = 0.0,

    # ✅ episode-boundary I/O throttles
    save_every_episodes: int = 50,
    stats_every_episodes: int = 20,

    # ✅ smooth update knobs (step-based)
    update_every_steps: int = 8,          # 업데이트 트리거 주기(너무 작으면 잦게, 너무 크면 뭉침)
    update_max_per_trigger: int = 1,      # 한 번 트리거에 몇 번 update()까지 허용 (끊김 방지용)
    update_time_budget_ms: float = 6.0,   # 추가 안전장치: 트리거 1회당 시간 예산(ms)

    # survival prints
    print_survival_every: int = 10,
    survival_window: int = 50,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    log_path = os.path.join(os.path.dirname(ckpt_path), f"{pth_name}_episode_log.txt")

    stats = ensure_stats_header(log_path)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_run_header(log_path, run_ts, int(episodes), bool(eval_mode), stats)
    print(stats_one_line(stats))

    env, agent = _load_agent_and_env(ckpt_path, no_render=no_render, use_sim=use_sim)

    # sim stacker + sim action mask
    sim_stacker = _SimFrameStacker(obs_channels=4, stack=4) if use_sim else None
    sim_action_mask = _build_action_mask_for_sim() if use_sim else None

    # survival tracking
    best_surv = 0.0
    surv_hist = deque(maxlen=int(max(1, survival_window)))

    if monitor_gpu:
        if torch.cuda.is_available() and str(getattr(agent, "device", "")).startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        _print_runtime_gpu_stats(agent, prefix="[GPU][INIT]")

    print("\n[INFO] ESC 중단: Windows 전역 감지(GetAsyncKeyState)")
    time.sleep(0.05)

    stop_requested = False

    try:
        for ep in range(1, int(episodes) + 1):
            if esc_pressed():
                print("[STOP] ESC pressed before episode start -> stopping.")
                stop_requested = True
                break

            print(f"\n========== EPISODE {ep}/{episodes} ==========")

            # GameEnv only: 메뉴/입력 처리
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

            # reset
            raw = env.reset()
            if use_sim:
                state = sim_stacker.reset(raw)
            else:
                state = raw

            if (not no_render) and (not use_sim):
                pump_cv_events_once()

            ep_t0 = time.time()
            done = False
            total_reward = 0.0
            steps = 0
            slow_count = 0
            action_counter = Counter()
            aborted = False

            last_mon_t = time.time()
            last_mon_step = 0

            # smooth-update counters
            local_updates = 0  # per-episode updates

            while not done:
                if esc_pressed():
                    stop_requested = True
                    aborted = True
                    print("[STOP] ESC pressed -> aborting NOW (NO SAVE/NO UPDATE for this episode).")
                    if not use_sim:
                        safe_release_inputs()
                    done = True
                    break

                # action mask
                if use_sim:
                    action_mask = sim_action_mask
                else:
                    img_for_mask = getattr(env.s, "last_action_mask_img", None)
                    action_mask = _build_action_mask_from_img(env, img_for_mask)

                action_idx, log_prob, value = agent.select_action(state, action_mask=action_mask)

                if use_sim:
                    raw_next, reward, done, _info = env.step(action_idx)
                    next_state = sim_stacker.step(raw_next)
                    exec_idx = int(action_idx) % 8
                    exec_name = ACTIONS[int(exec_idx)].name if int(exec_idx) < len(ACTIONS) else f"SIM_{exec_idx}"
                else:
                    next_state, reward, done = env.step(action_idx)
                    exec_idx = getattr(env.s, "exec_action_idx", action_idx)
                    exec_name = ACTIONS[int(exec_idx)].name

                action_counter[exec_name] += 1
                if exec_name.startswith("SLOW"):
                    slow_count += 1

                # store transition
                if not eval_mode:
                    agent.store(state, int(action_idx), reward, done, log_prob, value, action_mask=action_mask)

                state = next_state
                total_reward += float(reward)
                steps += 1

                # -----------------------
                # ✅ Smooth PPO update (step-based)
                # -----------------------
                if not eval_mode:
                    # 트리거 간격
                    if int(update_every_steps) > 0 and (steps % int(update_every_steps)) == 0:
                        # 시간 예산 내에서, 최대 N번 update
                        t_budget_end = time.perf_counter() + (float(update_time_budget_ms) / 1000.0)
                        k = 0
                        while agent.should_update():
                            agent.update(last_state=state, last_done=False)
                            local_updates += 1
                            k += 1
                            if k >= int(max(1, update_max_per_trigger)):
                                break
                            if time.perf_counter() >= t_budget_end:
                                break

                # GPU monitor
                if monitor_gpu:
                    do_print = False
                    if monitor_every_steps and monitor_every_steps > 0:
                        if (steps - last_mon_step) >= int(monitor_every_steps):
                            do_print = True
                            last_mon_step = steps
                    if monitor_every_sec and monitor_every_sec > 0:
                        now_t = time.time()
                        if (now_t - last_mon_t) >= float(monitor_every_sec):
                            do_print = True
                            last_mon_t = now_t
                    if do_print:
                        _print_runtime_gpu_stats(agent, prefix=f"[GPU][ep{ep} step{steps}]")

                # GameEnv only: cv pump
                if (not no_render) and (not use_sim):
                    pump_cv_events_once()

            survival_sec = time.time() - ep_t0
            best_surv = max(best_surv, float(survival_sec))
            surv_hist.append(float(survival_sec))

            # sim speed info
            if use_sim:
                sps = steps / max(survival_sec, 1e-9)
                print(f"[SIM] wall={survival_sec:.3f}s steps={steps} SPS={sps:.0f} (~{sps/60.0:.1f}x vs 60fps)")

            if (ep % int(max(1, print_survival_every))) == 0 or ep == 1:
                avg_surv = float(np.mean(surv_hist)) if len(surv_hist) else 0.0
                print(f"[SURV] best={best_surv:.2f}s | avg(last {len(surv_hist)})={avg_surv:.2f}s")

            slow_ratio = slow_count / max(1, steps)
            top_actions = action_counter.most_common(5)
            top_actions_str = ";".join(f"{k}:{v}" for k, v in top_actions)

            note_parts = []
            if eval_mode:
                note_parts.append("EVAL")
            if aborted:
                note_parts.append("ABORTED")
            note = ",".join(note_parts)

            print(
                f"[PPO] episode end | steps={steps} total_reward={total_reward:.1f} "
                f"survival_sec={survival_sec:.2f} slow_ratio={slow_ratio:.3f} "
                f"updates(ep)={local_updates} top_actions={top_actions_str} {note}"
            )

            # per-episode lightweight append
            ep_tag = f"({ep}/{episodes})"
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{ep_tag}\t{total_reward:.6f}\t{survival_sec:.3f}\t{note}\n")
            except Exception as e:
                print(f"[WARN] episode log append failed (ignored): {e}")

            if aborted:
                if not eval_mode:
                    _try_clear_agent_rollout(agent)
                print("[STOP] Episode aborted -> stopping.")
                break

            # -----------------------
            # ✅ No end-of-episode update
            # -----------------------
            # (끊김 줄이기 목적) 필요하면 여기서 0~1회만 돌리는 정도로만 추가 가능.

            if not eval_mode:
                # stats update (throttled)
                do_stats = (int(stats_every_episodes) > 0) and ((ep % int(stats_every_episodes)) == 0 or ep == int(episodes))
                if do_stats:
                    maybe_update_records(stats, total_reward, survival_sec, run_ts, ep_tag)
                    update_stats_in_file(log_path, stats)
                print(stats_one_line(stats))

                # checkpoint save (throttled)
                do_save = (int(save_every_episodes) > 0) and ((ep % int(save_every_episodes)) == 0 or ep == int(episodes))
                if do_save:
                    ok_save = _safe_save_checkpoint(agent, ckpt_path)
                    if ok_save:
                        print("[PPO] checkpoint saved")
                    else:
                        print("[WARN] checkpoint save failed -> continue training without stopping")
                else:
                    print("[PPO] checkpoint skipped")
            else:
                print(stats_one_line(stats))

            if stop_requested:
                print("[STOP] Training stopped by ESC.")
                break

            # ✅ episode boundary sleep 최소화
            if (not use_sim) and (ep < int(episodes)):
                time.sleep(0.02)

    finally:
        cleanup_inputs_on_exit()
        if not no_render:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n[PPO] Finished.")
