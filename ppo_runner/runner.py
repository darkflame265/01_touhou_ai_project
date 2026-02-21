from __future__ import annotations

import os
import time
from datetime import datetime
from collections import Counter, deque

import cv2
import numpy as np
import torch
import subprocess

from env.game_env import GameEnv
from env.controller import release_all, set_attack_hold, cleanup_inputs_on_exit
from env.menu import boot_into_practice
from env.actions import ACTIONS
from agents.ppo_agent import PPOAgent
from agents.mlp_ppo_agent import MLPPPOAgent

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

MLP_TOPK = 32


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


def _find_stop_index() -> int | None:
    for i, a in enumerate(ACTIONS):
        if getattr(a, "name", "") == "SLOW_STOP":
            return i
    return None


def _find_stop_name() -> str | None:
    return "SLOW_STOP" if _find_stop_index() is not None else None


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
    ✅ SimEnv용 마스크 (이름 기반):
    - 8방향 이동 + SLOW_STOP 허용
    - BOMB는 항상 금지
    """
    allow = {
        "SLOW_LEFT", "SLOW_RIGHT", "SLOW_UP", "SLOW_DOWN",
        "SLOW_UP_LEFT", "SLOW_UP_RIGHT", "SLOW_DOWN_LEFT", "SLOW_DOWN_RIGHT",
        "SLOW_STOP",
    }
    mask = np.zeros((len(ACTIONS),), dtype=np.bool_)

    for i, a in enumerate(ACTIONS):
        name = str(getattr(a, "name", "")).upper()
        if name in allow:
            mask[i] = True

    bidx = _find_bomb_index()
    if bidx is not None:
        mask[int(bidx)] = False

    # 전부 False면(이름 mismatch) fallback: 전체 True 후 bomb만 False
    if not bool(mask.any()):
        mask[:] = True
        if bidx is not None:
            mask[int(bidx)] = False

    return mask


def _build_mlp_vector_from_obs(obs, k: int = MLP_TOPK) -> np.ndarray:
    spatial = getattr(obs, "last_bullet_spatial_vec", None)
    if spatial is not None:
        arr = np.asarray(spatial, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return arr

    px, py = getattr(obs, "last_xy_norm", (0.5, 0.78))
    conf = float(getattr(obs, "last_conf", 0.0))

    dxdy = getattr(obs, "last_bullets_dxdy", None)
    vdxdy = getattr(obs, "last_bullets_vdxdy", None)

    if dxdy is None or vdxdy is None:
        bullets = getattr(obs, "last_bullets_xy_norm", []) or []
        n = int(len(bullets))
        n_norm = float(np.clip(n / float(max(1, k)), 0.0, 1.0))
        feats = [float(px), float(py), float(conf), float(n_norm)]
        for i in range(int(k)):
            if i < n:
                bx, by = bullets[i]
                dx = float(bx) - float(px)
                dy = float(by) - float(py)
                feats.extend([dx, dy, 0.0, 0.0])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
        return np.asarray(feats, dtype=np.float32)

    dxdy_len = int(len(dxdy))
    vdxdy_len = int(len(vdxdy))

    n = 0
    for (dx, dy) in dxdy[:k]:
        if abs(float(dx)) > 1e-9 or abs(float(dy)) > 1e-9:
            n += 1
    n_norm = float(np.clip(n / float(max(1, k)), 0.0, 1.0))

    feats = [float(px), float(py), float(conf), float(n_norm)]
    for i in range(int(k)):
        if i < dxdy_len:
            dx, dy = dxdy[i]
        else:
            dx, dy = 0.0, 0.0

        if i < vdxdy_len:
            vdx, vdy = vdxdy[i]
        else:
            vdx, vdy = 0.0, 0.0

        feats.extend([float(dx), float(dy), float(vdx), float(vdy)])

    return np.asarray(feats, dtype=np.float32)


def _build_mlp_state_from_env(env: GameEnv, k: int = MLP_TOPK) -> np.ndarray:
    return _build_mlp_vector_from_obs(env.obs, k=int(k))


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
    for attr in ("actor_critic", "net", "model", "policy", "policy_net"):
        m = getattr(agent, attr, None)
        if m is not None and hasattr(m, "eval"):
            try:
                m.eval()
            except Exception:
                pass


def _select_action_best_effort(agent, state, action_mask: np.ndarray | None):
    try:
        return agent.select_action(state, action_mask=action_mask, deterministic=True)
    except TypeError:
        pass
    except Exception:
        pass

    for name in ("select_action_deterministic", "act_deterministic", "select_action_eval"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                out = fn(state, action_mask=action_mask)
                if isinstance(out, (int, np.integer)):
                    return int(out), 0.0, 0.0
                if isinstance(out, (tuple, list)) and len(out) >= 1:
                    a = out[0]
                    lp = out[1] if len(out) > 1 else 0.0
                    v = out[2] if len(out) > 2 else 0.0
                    return int(a), lp, v
            except Exception:
                pass

    try:
        return agent.select_action(state, action_mask=action_mask)
    except Exception:
        return 0, 0.0, 0.0


# ----------------------------
# PPO debug metrics (best-effort)
# ----------------------------
def _fmt_num(x, nd: int = 4) -> str:
    try:
        if x is None:
            return "NA"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        v = float(x)
        if not np.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"
    except Exception:
        return "NA"


def _get_first_attr(obj, names: list[str]):
    for n in names:
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
    return None


def _ppo_debug_line(agent) -> str:
    ent = _get_first_attr(agent, ["last_entropy", "entropy", "ent", "ent_mean"])
    kl = _get_first_attr(agent, ["last_kl", "approx_kl", "kl", "kl_mean"])
    clipfrac = _get_first_attr(agent, ["last_clipfrac", "clipfrac", "clip_frac", "clip_fraction"])

    lr = None
    opt = _get_first_attr(agent, ["optimizer", "opt"])
    if opt is not None:
        try:
            lr = opt.param_groups[0].get("lr", None)
        except Exception:
            lr = None
    if lr is None:
        lr = _get_first_attr(agent, ["lr", "learning_rate", "last_lr"])

    return f"[PPO] ent={_fmt_num(ent, 4)} kl={_fmt_num(kl, 5)} clipfrac={_fmt_num(clipfrac, 4)} lr={_fmt_num(lr, 6)}"


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
def _load_agent_and_env(ckpt_path: str, no_render: bool, use_sim: bool, use_mlp_agent: bool = False):
    if use_sim and use_mlp_agent:
        raise ValueError("--mlp-agent is currently supported only for real game env (not --sim).")

    if use_sim:
        cfg = SimConfig(seed=0)
        if no_render:
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

    if use_mlp_agent:
        sample_vec = _build_mlp_state_from_env(env, k=MLP_TOPK)
        input_dim = int(np.asarray(sample_vec, dtype=np.float32).reshape(-1).shape[0])
        agent = MLPPPOAgent(
            input_dim=int(input_dim),
            num_actions=len(ACTIONS),
            lr=1e-4,
            rollout_steps=256,
            update_epochs=3,
            mini_batch_size=128,
        )
    else:
        agent = PPOAgent(
            input_channels=input_channels,
            num_actions=len(ACTIONS),
            obs_channels_per_frame=obs_channels,
        )

    if os.path.exists(ckpt_path):
        try:
            agent.load(ckpt_path, load_optimizer=False)
            print(f"[PPO] checkpoint loaded: {ckpt_path}")
        except RuntimeError as e:
            if use_mlp_agent:
                print(f"[PPO-MLP][WARN] checkpoint shape mismatch -> start from scratch: {e}")
            else:
                raise
    else:
        print("[PPO] no checkpoint found, training from scratch")

    # overrides (기존 유지)
    agent.ent_coef = 0.02
    agent.ent_min = 0.005
    agent.ent_decay = 0.9995
    agent.ent_warmup_updates = 30
    agent.clip_eps = 0.12
    agent.rollout_steps = 256
    agent.update_epochs = 3

    if use_mlp_agent:
        print(f"[PPO-MLP] input_dim={input_dim} (spatial-grid vector)")
    else:
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
    use_mlp_agent: bool = False,

    # monitoring
    monitor_gpu: bool = False,
    monitor_every_steps: int = 400,
    monitor_every_sec: float = 0.0,

    # training I/O (eval에서는 무시됨)
    save_every_episodes: int = 1,
    stats_every_episodes: int = 1,

    # smooth update knobs (step-based)
    update_every_steps: int = 32,
    update_max_per_trigger: int = 1,

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

    env, agent = _load_agent_and_env(
        ckpt_path,
        no_render=no_render,
        use_sim=use_sim,
        use_mlp_agent=use_mlp_agent,
    )

    if eval_mode:
        _agent_set_eval_mode(agent)

    sim_stacker = _SimFrameStacker(obs_channels=4, stack=4) if use_sim else None
    sim_action_mask = _build_action_mask_for_sim() if use_sim else None

    # best tracking
    best_wall = 0.0
    best_game = 0.0
    best_reward = -1e9
    best_steps = 0

    # 기존 wall 기반(유지: SURV 출력용)
    surv_hist = deque(maxlen=int(max(1, survival_window)))

    # 최근 20개 평균(게임시간) 표시용
    sim_surv_hist20 = deque(maxlen=20)

    # ✅ 최근 20개 "클리어(success)" 개수 표시용
    sim_clear_hist20 = deque(maxlen=20)

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
            if use_sim:
                state = sim_stacker.reset(raw)
            elif use_mlp_agent:
                state = _build_mlp_state_from_env(env, k=MLP_TOPK)
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

                # action mask
                if use_sim:
                    action_mask = sim_action_mask
                else:
                    img_for_mask = getattr(env.s, "last_action_mask_img", None)
                    action_mask = _build_action_mask_from_img(env, img_for_mask)

                # select action
                if eval_mode:
                    action_idx, log_prob, value = _select_action_best_effort(agent, state, action_mask)
                else:
                    action_idx, log_prob, value = agent.select_action(state, action_mask=action_mask)

                action_idx = _clamp_action_idx(action_idx)

                if use_sim:
                    raw_next, reward, done, _info = env.step(action_idx)
                    next_state = sim_stacker.step(raw_next)

                    exec_idx = int(action_idx)
                    exec_name = str(getattr(ACTIONS[int(exec_idx)], "name", ""))

                    # ✅ (선택) 여기서도 쓰고 싶으면 가능하지만,
                    # 실제 "최근20 clear" 집계는 에피소드 끝에서 기록하는 게 안전함.
                    # (에피소드 단위로 1개 bool만 넣고 싶으니까)
                else:
                    _next_obs, reward, done = env.step(action_idx)
                    if use_mlp_agent:
                        next_state = _build_mlp_state_from_env(env, k=MLP_TOPK)
                    else:
                        next_state = _next_obs
                    exec_idx = getattr(env.s, "exec_action_idx", action_idx)
                    exec_name = str(getattr(ACTIONS[int(exec_idx)], "name", ""))

                action_counter[exec_name] += 1
                if exec_name.startswith("SLOW"):
                    slow_count += 1

                # store transition (train only)
                if do_io:
                    agent.store(state, int(action_idx), reward, done, log_prob, value, action_mask=action_mask)

                state = next_state
                total_reward += float(reward)
                steps += 1

                # -------------------------------------------------
                # ✅ PPO update: "시간 예산" 제거 -> step/rollout 기준만
                # -------------------------------------------------
                if do_io:
                    if int(update_every_steps) > 0 and (steps % int(update_every_steps)) == 0:
                        k = 0
                        while agent.should_update():
                            agent.update(last_state=state, last_done=False)
                            local_updates += 1
                            k += 1
                            if k >= int(max(1, update_max_per_trigger)):
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

                if (not no_render) and (not use_sim):
                    pump_cv_events_once()

            wall_sec = time.time() - ep_t0
            game_sec = float(steps) / 60.0

            # 기존 SURV용(벽시계) 유지
            surv_hist.append(float(wall_sec))

            # 최근 20개 평균(게임시간) 표시 + ✅ 최근20 clear 집계
            if use_sim:
                sim_surv_hist20.append(float(game_sec))
                avg20 = float(np.mean(sim_surv_hist20)) if len(sim_surv_hist20) else 0.0

                # ✅ 에피소드 단위 success 기록
                # SimEnv.info["success"]는 time_limit 도달 여부(=완주)로 넣어둔 상태
                try:
                    ep_success = bool(_info.get("success", False))  # _info는 마지막 step의 info
                except Exception:
                    ep_success = False
                sim_clear_hist20.append(bool(ep_success))

                clear_n = int(np.sum(np.asarray(sim_clear_hist20, dtype=np.int32)))
                clear_d = int(len(sim_clear_hist20))
            else:
                avg20 = 0.0
                clear_n = 0
                clear_d = 0

            # best tracking
            if float(game_sec) > float(best_game):
                best_game = float(game_sec)
            if float(wall_sec) > float(best_wall):
                best_wall = float(wall_sec)
            if float(total_reward) > float(best_reward):
                best_reward = float(total_reward)
            if int(steps) > int(best_steps):
                best_steps = int(steps)

            # sim prints
            if use_sim:
                sps = steps / max(wall_sec, 1e-9)
                speed_x = sps / 60.0
                cur_diff = float(getattr(getattr(env, "cfg", None), "difficulty", -1.0))

                # ✅ avg20 옆에 clear=8/20 출력
                print(
                    f"[SIM] game={game_sec:.3f}s steps={steps}  "
                    f"avg20={avg20:.3f}s clear={clear_n}/{max(1, clear_d)} "
                    f"SPS={sps:.0f} (~{speed_x:.1f}x vs 60fps)"
                )
                print(f"[BST] game={best_game:.3f}s steps={best_steps} diff={cur_diff:.1f}")
                print(_ppo_debug_line(agent))

            if (ep % int(max(1, print_survival_every))) == 0 or ep == 1:
                avg_surv = float(np.mean(surv_hist)) if len(surv_hist) else 0.0
                print(f"[SURV] best_wall={best_wall:.2f}s best_game={best_game:.2f}s | avg_wall(last {len(surv_hist)})={avg_surv:.2f}s")

            slow_ratio = slow_count / max(1, steps)
            stop_count = int(action_counter.get(stop_name, 0)) if stop_name else 0
            stop_ratio = (stop_count / max(1, steps)) if stop_name else 0.0

            note_parts = []
            if eval_mode:
                note_parts.append("EVAL_BEST")
            if aborted:
                note_parts.append("ABORTED")
            note = ",".join(note_parts)

            print(
                f"[PPO] episode end | steps={steps} total_reward={total_reward:.1f} "
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

            maybe_update_records(stats, total_reward, wall_sec, run_ts, ep_tag)
            update_stats_in_file(log_path, stats)
            print(stats_one_line(stats))

            ok_save = _safe_save_checkpoint(agent, ckpt_path)
            if ok_save:
                print("[PPO] checkpoint saved")
            else:
                print("[WARN] checkpoint save failed -> continue training without stopping")

            if stop_requested:
                print("[STOP] Training stopped by ESC.")
                break

            if (not use_sim) and (ep < int(episodes)):
                time.sleep(0.02)

    except KeyboardInterrupt:
        stop_requested = True
        print("[STOP] ESC/P pressed during menu prep -> stopping.")
    finally:
        cleanup_inputs_on_exit()
        if not no_render:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if eval_mode:
        print(f"\n[EVAL_SUMMARY] best_game_sec={best_game:.2f}s | best_wall_sec={best_wall:.2f}s | best_reward={best_reward:.3f}")

    print("\n[PPO] Finished.")
