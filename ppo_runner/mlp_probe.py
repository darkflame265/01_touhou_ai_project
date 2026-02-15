# ppo_runner/mlp_probe.py
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Tuple, Dict, List, Callable

import numpy as np

from env.game_env import GameEnv
from env.menu import boot_into_practice
from env.controller import release_all, set_attack_hold
from ppo_runner.hotkeys import esc_pressed


def _safe_release_inputs() -> None:
    try:
        set_attack_hold(False)
    except Exception:
        pass
    try:
        release_all()
    except Exception:
        pass


def _build_random_action_sampler(seed: int = 0) -> tuple[Callable[[], int], List[str]]:
    """
    ACTIONS 이름을 보고:
      - STOP/IDLE/NOOP 계열은 낮은 확률
      - ATTACK/SHOT/FIRE 계열도 낮은 확률
    """
    from env.actions import ACTIONS

    names = [str(a).upper() for a in ACTIONS]
    n = len(names)

    w = np.ones(n, dtype=np.float32)

    # STOP/IDLE/NOOP 줄이기
    for i, nm in enumerate(names):
        if ("STOP" in nm) or ("IDLE" in nm) or ("NOOP" in nm):
            w[i] *= 0.10

    # 발사/공격 줄이기
    for i, nm in enumerate(names):
        if ("ATTACK" in nm) or ("SHOT" in nm) or ("FIRE" in nm):
            w[i] *= 0.25

    if float(w.sum()) <= 0:
        w[:] = 1.0
    w /= w.sum()

    rng = np.random.default_rng(seed)

    def sample() -> int:
        return int(rng.choice(np.arange(n), p=w))

    return sample, names


def _vectorize_from_obs(env: GameEnv) -> tuple[np.ndarray, dict]:
    obs = env.obs

    # 1) player
    x_n, y_n = getattr(obs, "last_xy_norm", (0.5, 0.78))
    conf = float(getattr(obs, "last_conf", 0.0))

    # 2) bullet candidate mask (있으면 fill ratio)
    bcm = getattr(obs, "bullet_candidate_mask", None)
    if bcm is not None:
        bcm_arr = np.asarray(bcm)
        fill = float(np.mean(bcm_arr > 0.5)) if bcm_arr.size > 0 else 0.0
    else:
        fill = 0.0

    # 3) risk heatmap (있으면 통계 + COM)
    rh = getattr(obs, "risk_heatmap", None)
    if rh is not None:
        r = np.asarray(rh, dtype=np.float32)
        if r.size > 0:
            r = np.clip(r, 0.0, 1.0)
            r_mean = float(r.mean())
            r_p90 = float(np.percentile(r, 90.0))
            r_max = float(r.max())

            s = float(r.sum())
            if s > 1e-6:
                h, w = r.shape[:2]
                ys = np.arange(h, dtype=np.float32)
                xs = np.arange(w, dtype=np.float32)
                yy, xx = np.meshgrid(ys, xs, indexing="ij")
                cx = float((r * xx).sum() / s) / max(1.0, (w - 1))
                cy = float((r * yy).sum() / s) / max(1.0, (h - 1))
            else:
                cx, cy = 0.5, 0.5
        else:
            r_mean, r_p90, r_max, cx, cy = 0.0, 0.0, 0.0, 0.5, 0.5
    else:
        r_mean, r_p90, r_max, cx, cy = 0.0, 0.0, 0.0, 0.5, 0.5

    vec = np.array([x_n, y_n, conf, fill, r_mean, r_p90, r_max, cx, cy], dtype=np.float32)

    dbg = {
        "x_n": float(x_n),
        "y_n": float(y_n),
        "conf": float(conf),
        "fill": float(fill),
        "risk_mean": float(r_mean),
        "risk_p90": float(r_p90),
        "risk_max": float(r_max),
        "risk_com_x": float(cx),
        "risk_com_y": float(cy),
    }
    return vec, dbg


def _step_env(env: GameEnv, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    env.step 반환 형태(3/4/5 튜플)를 모두 지원.
    return: (obs_like, reward, done, info)
    """
    ret = env.step(action)

    if isinstance(ret, tuple) and len(ret) == 3:
        obs, reward, done = ret
        return obs, float(reward), bool(done), {}

    if isinstance(ret, tuple) and len(ret) == 4:
        obs, reward, done, info = ret
        return obs, float(reward), bool(done), (info or {})

    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        done = bool(terminated) or bool(truncated)
        return obs, float(reward), done, (info or {})

    raise RuntimeError(f"Unexpected env.step return: {ret!r}")

def _vectorize_from_obs(env: GameEnv, k: int = 16) -> tuple[np.ndarray, dict]:
    obs = env.obs

    px, py = getattr(obs, "last_xy_norm", (0.5, 0.78))
    conf = float(getattr(obs, "last_conf", 0.0))

    bullets = getattr(obs, "last_bullets_xy_norm", []) or []
    n = int(len(bullets))
    n_norm = float(np.clip(n / float(max(1, k)), 0.0, 1.0))

    feats: List[float] = [float(px), float(py), float(conf), float(n_norm)]

    # top-k 상대좌표
    for i in range(k):
        if i < n:
            bx, by = bullets[i]
            dx = float(bx) - float(px)
            dy = float(by) - float(py)
            feats.extend([dx, dy])
        else:
            feats.extend([0.0, 0.0])

    vec = np.asarray(feats, dtype=np.float32)

    dbg = {
        "px": float(px),
        "py": float(py),
        "conf": float(conf),
        "n_bullets": int(n),
        "k": int(k),
        "first_bullet": bullets[0] if n > 0 else None,
        "vec_dim": int(vec.shape[0]),
    }
    return vec, dbg



def run_mlp_probe(episodes: int = 1, no_render: bool = False) -> None:
    """
    --mlp 모드:
      - practice 진입 자동화는 기존 루틴 그대로 사용
      - 학습/agent 없음
      - 인게임 프레임에서 벡터만 추출해서 저장

    저장:
      - runs/mlp_vectors_<timestamp>.npz
    """
    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("runs", f"mlp_vectors_{ts}.npz")

    env = GameEnv(screen_mode="low")
    if no_render:
        try:
            from ppo_runner.render import apply_no_render
            apply_no_render(env)
        except Exception:
            pass

    sample_action, action_names = _build_random_action_sampler(seed=0)
    print(f"[MLP] action_space={len(action_names)}")

    all_vecs: List[np.ndarray] = []
    all_meta: List[List[float]] = []

    try:
        for ep in range(1, int(episodes) + 1):
            if esc_pressed():
                print("[MLP] ESC -> stop")
                break

            print(f"\n[MLP] EP {ep}/{episodes} boot_into_practice...")
            _safe_release_inputs()
            time.sleep(0.05)

            ok = boot_into_practice(env.screen, max_sec_lobby=12.0)
            if not ok:
                print("[MLP][WARN] boot_into_practice failed (continue anyway)")

            _safe_release_inputs()
            time.sleep(0.05)

            try:
                env.reset()
            except TypeError:
                env.reset()

            t0 = time.time()
            frames = 0
            cur_action = sample_action()  # ✅ 초기값

            while True:
                if esc_pressed():
                    print("[MLP] ESC -> stop")
                    raise KeyboardInterrupt

                # ✅ N프레임마다 새 액션(덜덜거림 완화)
                if frames % 6 == 0:
                    cur_action = sample_action()

                _, reward, done, info = _step_env(env, cur_action)

                vec, dbg = _vectorize_from_obs(env)

                all_vecs.append(vec)
                all_meta.append([time.time(), float(ep)])

                frames += 1
                if frames % 60 == 0:
                    aname = action_names[cur_action] if 0 <= cur_action < len(action_names) else str(cur_action)
                    print(
                        f"[MLP] t={time.time()-t0:5.1f}s "
                        f"action={cur_action}:{aname} "
                        f"done={done} r={reward:.3f} "
                        f"vec={vec.tolist()} dbg={dbg}"
                    )

                if done:
                    try:
                        env.reset()
                    except TypeError:
                        env.reset()

                if (time.time() - t0) >= 10.0:
                    print(f"[MLP] collected frames={frames}")
                    break

                time.sleep(0.0)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        _safe_release_inputs()

    if len(all_vecs) == 0:
        print("[MLP] no vectors collected")
        return

    arr = np.stack(all_vecs, axis=0)  # (T, D)
    meta = np.asarray(all_meta, dtype=np.float64)  # (T,2): [unix_time, episode]

    np.savez_compressed(out_path, vec=arr, meta=meta)
    print(f"[MLP] saved: {out_path}  shape={arr.shape}")
