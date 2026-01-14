# ppo_runner/runner.py
from __future__ import annotations

import os
import time
from datetime import datetime
from collections import Counter

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

from .hotkeys import esc_pressed
from .render import apply_no_render, pump_cv_events_once
from .stats_log import (
    ensure_stats_header,
    append_run_header,
    maybe_update_records,
    update_stats_in_file,
    stats_one_line,
)


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

def _print_hw_info():
    # OS / CPU
    try:
        print(f"[HW] OS={platform.platform()}")
        print(f"[HW] CPU={platform.processor() or 'UNKNOWN'} | cores(logical)={os.cpu_count()}")
    except Exception as e:
        print(f"[HW][WARN] CPU info failed: {e}")

    # PyTorch / CUDA
    try:
        print(f"[HW] torch={torch.__version__}")
        print(f"[HW] torch.cuda.is_available()={torch.cuda.is_available()}")
        print(f"[HW] torch.version.cuda={torch.version.cuda}")
        print(f"[HW] cuDNN={torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"[HW][WARN] torch info failed: {e}")

    # GPU
    if torch.cuda.is_available():
        try:
            n = torch.cuda.device_count()
            print(f"[HW] CUDA device_count={n}")
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                print(
                    f"[HW] GPU[{i}] {name} | cc={cap[0]}.{cap[1]} | VRAM={total_gb:.2f} GB"
                )
        except Exception as e:
            print(f"[HW][WARN] GPU info failed: {e}")

def _torch_mem_str(device: str = "cuda") -> str:
    if (not torch.cuda.is_available()) or (device != "cuda"):
        return "CUDA: N/A"
    try:
        # bytes -> MiB
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
    """
    returns one-line summary or None if nvidia-smi not available.
    Query: util.gpu, util.mem, mem.used, mem.total, temperature.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if not out:
            return None

        # If multiple GPUs, take GPU0 line (or join them)
        lines = [x.strip() for x in out.splitlines() if x.strip()]
        # Format per line: "18, 12, 1234, 8192, 55"
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) >= 5:
            util_gpu = parts[0]
            util_mem = parts[1]
            mem_used = parts[2]
            mem_total = parts[3]
            temp = parts[4]
            return f"nvidia-smi util={util_gpu}% memUtil={util_mem}% mem={mem_used}/{mem_total}MiB temp={temp}C"
        return f"nvidia-smi raw={lines[0]}"
    except Exception:
        return None


def _print_runtime_gpu_stats(agent, prefix: str = "[GPU]"):
    # 1) torch VRAM
    dev = getattr(agent, "device", "cpu")
    if isinstance(dev, str) and dev.startswith("cuda") and torch.cuda.is_available():
        print(f"{prefix} {_torch_mem_str('cuda')}")
    else:
        print(f"{prefix} CUDA: OFF (agent.device={dev})")

    # 2) nvidia-smi util
    smi = _nvidia_smi_query()
    if smi is not None:
        print(f"{prefix} {smi}")
    else:
        print(f"{prefix} nvidia-smi: N/A (not found or failed)")


def _try_clear_agent_rollout(agent):
    for name in ("clear", "reset_buffer", "reset_storage", "clear_buffer", "clear_rollout"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


def _safe_save_checkpoint(agent, ckpt_path: str) -> bool:
    try:
        ret = agent.save(ckpt_path)
        return bool(ret) if isinstance(ret, bool) else True
    except Exception as e:
        print(f"[WARN] checkpoint save failed (ignored): {e}")
        return False


def _load_agent_and_env(ckpt_path: str, no_render: bool):
    env = GameEnv(screen_mode="low")
    if no_render:
        apply_no_render(env)

    obs_channels = int(getattr(env.obs, "obs_channels", 1))
    stack_size = int(getattr(env.s, "frame_stack_size", 4))
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
    # ✅ 에이전트 디바이스(실제 학습 device)도 출력
    try:
        print(f"[HW] Agent device={agent.device} | AMP={getattr(agent, 'use_amp', False)}")
    except Exception:
        pass

    return env, agent


def _find_bomb_index() -> int | None:
    for i, a in enumerate(ACTIONS):
        if getattr(a, "name", "") == "BOMB":
            return i
    return None


def _build_action_mask(env: GameEnv) -> np.ndarray:
    """
    1) env.masker가 있으면 화면 캡처 1회로 mask 계산
    2) 안전장치: BOMB는 항상 금지(정책이 아예 뽑지 못하도록)
    """
    mask = np.ones((len(ACTIONS),), dtype=np.bool_)

    # (A) 가능하면 env.masker 기반 마스크를 AND로 적용
    try:
        img = env.screen.capture()
        if hasattr(env, "masker") and env.masker is not None:
            m2 = env.masker.get_action_mask(img)
            if m2 is not None and len(m2) == len(mask):
                mask &= m2.astype(np.bool_, copy=False)
    except Exception:
        pass

    # (B) 최종적으로 BOMB는 무조건 금지
    bidx = _find_bomb_index()
    if bidx is not None:
        mask[int(bidx)] = False

    # (C) 전부 False면 위험하니(샘플링 불가) 전부 True로 복구 후 BOMB만 금지
    if not bool(mask.any()):
        mask[:] = True
        if bidx is not None:
            mask[int(bidx)] = False

    return mask


def run(
    episodes: int = 1,
    no_render: bool = False,
    eval_mode: bool = False,
    ckpt_path: str = "checkpoints/lunatic_v1_ch4.pth",
    # ✅ 모니터링 옵션
    monitor_gpu: bool = False,
    monitor_every_steps: int = 200,
    monitor_every_sec: float = 0.0,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    log_path = os.path.join(os.path.dirname(ckpt_path), f"{pth_name}_episode_log.txt")

    # stats 파일 헤더/블록 보장 + RUN 헤더 append
    stats = ensure_stats_header(log_path)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_run_header(log_path, run_ts, int(episodes), bool(eval_mode), stats)
    print(stats_one_line(stats))

    env, agent = _load_agent_and_env(ckpt_path, no_render=no_render)

    if monitor_gpu:
        # torch max 메모리 peak 측정용 reset (원하면)
        if torch.cuda.is_available() and getattr(agent, "device", "").startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        _print_runtime_gpu_stats(agent, prefix="[GPU][INIT]")

    print("\n[INFO] ESC 중단: Windows 전역 감지(GetAsyncKeyState)")
    time.sleep(0.2)

    stop_requested = False

    try:
        for ep in range(1, int(episodes) + 1):
            if esc_pressed():
                print("[STOP] ESC pressed before episode start -> stopping.")
                stop_requested = True
                break

            print(f"\n========== EPISODE {ep}/{episodes} ==========")

            # ✅ 메뉴 들어가기 전에 무조건 입력/공격홀드 해제
            safe_release_inputs()
            time.sleep(0.05)

            # 에피소드 시작 전(로비/스코어 등)에서만 메뉴 제어
            print("[MENU] [practice 준비/진입 중.]")
            ok = boot_into_practice(env.screen, max_sec_lobby=12.0)
            if not ok:
                print("[EP_PREP][WARN] boot_into_practice failed (continue)")
            print("[MENU] [practice 준비/진입 완료]")

            # 메뉴 끝난 직후에도 한 번 더(잔류 방지)
            safe_release_inputs()
            time.sleep(0.05)

            state = env.reset()

            if not no_render:
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


            # ✅ 인게임 루프: update 금지(렉 방지)
            while not done:
                if esc_pressed():
                    stop_requested = True
                    aborted = True
                    print("[STOP] ESC pressed -> aborting NOW (release inputs, NO SAVE/NO UPDATE for this episode).")
                    safe_release_inputs()
                    done = True
                    break

                # ✅ action mask 생성 + PPO 샘플링에 반영
                action_mask = _build_action_mask(env)
                action_idx, log_prob, value = agent.select_action(state, action_mask=action_mask)

                next_state, reward, done = env.step(action_idx)

                # ✅ 실제 실행된 액션 기준으로 통계/저장/slow 카운트
                exec_idx = getattr(env.s, "exec_action_idx", action_idx)
                exec_name = ACTIONS[int(exec_idx)].name
                action_counter[exec_name] += 1
                if exec_name.startswith("SLOW"):
                    slow_count += 1

                if not eval_mode:
                    agent.store(state, int(action_idx), reward, done, log_prob, value, action_mask=action_mask)

                state = next_state
                total_reward += float(reward)
                steps += 1
                # ---- optional GPU monitor ----
                if monitor_gpu:
                    do_print = False

                    # step-based
                    if monitor_every_steps and monitor_every_steps > 0:
                        if (steps - last_mon_step) >= int(monitor_every_steps):
                            do_print = True
                            last_mon_step = steps

                    # time-based
                    if monitor_every_sec and monitor_every_sec > 0:
                        now_t = time.time()
                        if (now_t - last_mon_t) >= float(monitor_every_sec):
                            do_print = True
                            last_mon_t = now_t

                    if do_print:
                        _print_runtime_gpu_stats(agent, prefix=f"[GPU][ep{ep} step{steps}]")


                if not no_render:
                    pump_cv_events_once()

            # 에피소드 종료 후(로비/스코어) 무거운 작업
            survival_sec = time.time() - ep_t0
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
                f"top_actions={top_actions_str} {note}"
            )

            # episode log append (원본 포맷 유지)
            ep_tag = f"({ep}/{episodes})"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{ep_tag}\t{total_reward:.6f}\t{survival_sec:.3f}\t{note}\n")

            if aborted:
                if not eval_mode:
                    _try_clear_agent_rollout(agent)
                print("[STOP] Episode aborted -> stopping.")
                break

            if not eval_mode:
                # ✅ update는 여기서만
                updates = 0
                while agent.should_update():
                    agent.update(last_state=state, last_done=True)
                    updates += 1
                if updates:
                    print(f"[PPO] updates_after_episode={updates}")

                # stats 갱신/기록
                maybe_update_records(stats, total_reward, survival_sec, run_ts, ep_tag)
                update_stats_in_file(log_path, stats)
                print(stats_one_line(stats))

                # ✅ 에피소드마다 저장
                ok_save = _safe_save_checkpoint(agent, ckpt_path)
                if ok_save:
                    print("[PPO] checkpoint saved")
                else:
                    print("[WARN] checkpoint save failed -> continue training without stopping")
            else:
                print(stats_one_line(stats))

            if stop_requested:
                print("[STOP] Training stopped by ESC.")
                break

            if ep < int(episodes):
                time.sleep(0.2)

    finally:
        cleanup_inputs_on_exit()
        if not no_render:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n[PPO] Finished.")
