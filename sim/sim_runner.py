# sim/sim_runner.py
from __future__ import annotations

import time
import numpy as np

from sim.sim_env import SimEnv, SimConfig


def main():
    # 원하는대로 여기서 수동 조절
    cfg = SimConfig(seed=0, render=True)
    env = SimEnv(cfg)

    obs = env.reset()

    # random policy demo
    t0 = time.time()
    steps = 0
    ep_r = 0.0

    # ---- periodic stats ----
    ep = 0
    print_every = 10
    surv_hist = []
    rew_hist = []
    best_surv = -1e9
    best_rew = -1e9

    try:
        import cv2  # noqa
        use_vis = False  # sim_env 자체 render가 있으니 기본 False
    except Exception:
        use_vis = False

    while True:
        a = int(np.random.randint(0, 8))
        obs, r, done, info = env.step(a)
        ep_r += float(r)
        steps += 1

        if use_vis:
            import cv2
            ch0 = (obs[0] * 255).astype(np.uint8)
            ch2 = (obs[2] * 255).astype(np.uint8)
            ch3 = (obs[3] * 255).astype(np.uint8)
            viz = np.concatenate([ch0, ch2, ch3], axis=1)
            cv2.imshow("sim (ch0|ch2|ch3)", viz)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        if done:
            ep += 1
            wall = time.time() - t0
            fps = steps / max(wall, 1e-9)

            surv_hist.append(float(wall))
            rew_hist.append(float(ep_r))
            best_surv = max(best_surv, float(wall))
            best_rew = max(best_rew, float(ep_r))

            print(f"[EP {ep}] steps={steps} ep_r={ep_r:.3f} wall={wall:.3f}s fps={fps:.1f} info={info}")

            if (ep % print_every) == 0:
                k = min(print_every, len(surv_hist))
                avg_surv = sum(surv_hist[-k:]) / max(1, k)
                avg_rew = sum(rew_hist[-k:]) / max(1, k)
                print(
                    f"[ROLLING {k}] avg_survival={avg_surv:.3f}s avg_reward={avg_rew:.3f} "
                    f"| best_survival={best_surv:.3f}s best_reward={best_rew:.3f}"
                )

            obs = env.reset()
            steps = 0
            ep_r = 0.0
            t0 = time.time()


if __name__ == "__main__":
    main()
