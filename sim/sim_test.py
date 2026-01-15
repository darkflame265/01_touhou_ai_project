# sim/sim_test.py
from __future__ import annotations

import numpy as np

from sim.sim_env import SimEnv, SimConfig


def test_basic_contract():
    env = SimEnv(SimConfig(seed=123, obs_out_size=128, world_size=256))
    obs = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 128, 128)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0

    total = 0.0
    done_seen = False
    for _ in range(2000):
        a = int(np.random.randint(0, 8))
        obs, r, done, info = env.step(a)
        total += r
        assert obs.shape == (4, 128, 128)
        assert obs.dtype == np.float32
        assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
        if done:
            done_seen = True
            break

    # With default spawn prob, likely to end by hit within 2000 steps
    assert done_seen, "Expected at least one episode termination (hit or max_steps) within 2000 steps."
    print("PASS: basic contract, total reward:", total)


if __name__ == "__main__":
    test_basic_contract()
