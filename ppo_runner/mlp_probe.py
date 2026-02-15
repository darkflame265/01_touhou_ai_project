# ppo_runner/mlp_probe.py
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Tuple, Dict, List

import numpy as np

from env.game_env import GameEnv
from env.menu import boot_into_practice
from env.controller import release_all, set_attack_hold
from ppo_runner.hotkeys import esc_pressed

MLP_TOPK = 32


def _safe_release_inputs() -> None:
    try:
        set_attack_hold(False)
    except Exception:
        pass
    try:
        release_all()
    except Exception:
        pass


def _build_random_action_sampler(seed: int = 0):
    from env.actions import ACTIONS

    names = [str(a).upper() for a in ACTIONS]
    n = len(names)

    w = np.ones(n, dtype=np.float32)

    for i, nm in enumerate(names):
        if "STOP" in nm or "IDLE" in nm or "NOOP" in nm:
            w[i] *= 0.10
        if "ATTACK" in nm or "SHOT" in nm or "FIRE" in nm:
            w[i] *= 0.25

    if float(w.sum()) <= 0:
        w[:] = 1.0
    w /= w.sum()

    rng = np.random.default_rng(seed)

    def sample() -> int:
        return int(rng.choice(np.arange(n), p=w))

    return sample, names


def _step_env(env: GameEnv, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
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
        return obs, float(reward), bool(done), (info or {})

    raise RuntimeError(f"Unexpected env.step return: {ret!r}")


def _vectorize_topk_with_vel(env: GameEnv, K: int = MLP_TOPK) -> tuple[np.ndarray, dict]:
    """
    vec = [px, py, conf, n_norm,  (dx,dy,vdx,vdy)*K]
    dxdy/vdxdy는 obs_builder가 slot-based로 이미 K padding해서 제공.
    """
    obs = env.obs

    px, py = getattr(obs, "last_xy_norm", (0.5, 0.78))
    conf = float(getattr(obs, "last_conf", 0.0))

    dxdy = getattr(obs, "last_bullets_dxdy", None)
    vdxdy = getattr(obs, "last_bullets_vdxdy", None)

    if dxdy is None or vdxdy is None:
        # obs_builder 패치가 안 들어간 경우를 대비한 fallback
        bullets = getattr(obs, "last_bullets_xy_norm", []) or []
        n = int(len(bullets))
        n_norm = float(np.clip(n / float(max(1, K)), 0.0, 1.0))
        feats: List[float] = [float(px), float(py), float(conf), float(n_norm)]
        for i in range(int(K)):
            if i < n:
                bx, by = bullets[i]
                dx = float(bx) - float(px)
                dy = float(by) - float(py)
                feats.extend([dx, dy, 0.0, 0.0])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
        vec = np.asarray(feats, dtype=np.float32)
        return vec, {"vec_dim": int(vec.shape[0]), "fallback": True}

    # dxdy/vdxdy는 K 길이로 padding되어 있다고 가정(ObsBuilder에서 그렇게 만듦)
    dxdy_len = int(len(dxdy))
    vdxdy_len = int(len(vdxdy))

    n = 0
    for (dx, dy) in dxdy[:K]:
        if abs(float(dx)) > 1e-9 or abs(float(dy)) > 1e-9:
            n += 1
    n_norm = float(np.clip(n / float(max(1, K)), 0.0, 1.0))

    feats: List[float] = [float(px), float(py), float(conf), float(n_norm)]

    for i in range(int(K)):
        if i < dxdy_len:
            dx, dy = dxdy[i]
        else:
            dx, dy = 0.0, 0.0

        if i < vdxdy_len:
            vdx, vdy = vdxdy[i]
        else:
            vdx, vdy = 0.0, 0.0

        feats.extend([float(dx), float(dy), float(vdx), float(vdy)])

    vec = np.asarray(feats, dtype=np.float32)
    dbg = {
        "px": float(px),
        "py": float(py),
        "conf": float(conf),
        "n_bullets": int(n),
        "k": int(K),
        "vec_dim": int(vec.shape[0]),
        "example_dxdy": dxdy[0] if len(dxdy) > 0 else None,
        "example_vdxdy": vdxdy[0] if len(vdxdy) > 0 else None,
    }
    return vec, dbg


def run_mlp_probe(
    episodes: int = 1,
    no_render: bool = False,
    max_seconds: float = 60.0,
) -> None:
    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("runs", f"mlp_vectors_{ts}.npz")

    env = GameEnv(screen_mode="low")

    # MLP probe can pass through brief UI-hide frames (death flash/respawn)
    # without being treated as an immediate ABORT.
    try:
        env.s.ui_absent_needed = max(int(getattr(env.s, "ui_absent_needed", 2)), 12)
    except Exception:
        pass

    if no_render:
        try:
            from ppo_runner.render import apply_no_render
            apply_no_render(env)
        except Exception:
            pass

    sample_action, action_names = _build_random_action_sampler(seed=0)
    print(f"[MLP] action_space={len(action_names)}")

    K = MLP_TOPK
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
            cur_action = 0

            while True:
                if esc_pressed():
                    print("[MLP] ESC -> stop")
                    raise KeyboardInterrupt

                if frames % 6 == 0:
                    cur_action = sample_action()

                _, reward, done, _info = _step_env(env, cur_action)

                vec, dbg = _vectorize_topk_with_vel(env, K=K)
                all_vecs.append(vec)
                all_meta.append([time.time(), float(ep)])

                frames += 1

                if frames % 60 == 0:
                    a_name = action_names[cur_action] if 0 <= cur_action < len(action_names) else str(cur_action)
                    print(
                        f"[MLP] t={time.time()-t0:5.1f}s "
                        f"action={cur_action}:{a_name} done={done} r={reward:.3f} "
                        f"dbg={dbg}"
                    )

                if done:
                    end_reason = str(getattr(env.s, "episode_end_reason", ""))
                    end_pen = float(getattr(env.s, "episode_end_pen", 0.0))
                    print(f"[MLP] done=True reason={end_reason or 'N/A'} pen={end_pen:.3f}")
                    if end_reason == "DEATH:AFTER_UI_ZERO":
                        print("[MLP] episode finished by UI lives/gameover condition")
                        print(f"[MLP] collected frames={frames}")
                        break
                    try:
                        env.reset()
                    except TypeError:
                        env.reset()

                if (time.time() - t0) >= float(max_seconds):
                    print(f"[MLP] max_seconds reached: {float(max_seconds):.1f}s")
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

    arr = np.stack(all_vecs, axis=0)
    meta = np.asarray(all_meta, dtype=np.float64)

    np.savez_compressed(out_path, vec=arr, meta=meta)
    print(f"[MLP] saved: {out_path}  shape={arr.shape}")
