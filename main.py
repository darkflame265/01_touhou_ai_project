# main.py
import os
import time
import sys
import argparse

from env.game_env import GameEnv
from env.actions import ACTIONS
from agents.dqn_agent import DQNAgent

CKPT_PATH = "checkpoints/dqn_practice_v2_pool.pth"
PRINT_EVERY_STEPS = 10
SAVE_ON_END = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true", help="evaluation mode (no learning, epsilon=0)")
    return p.parse_args()


def main():
    args = parse_args()
    is_eval = args.eval

    env = GameEnv(screen_mode="high")
    state = env.reset()

    agent = DQNAgent(
        input_channels=4,
        num_actions=len(ACTIONS),
    )

    # 체크포인트 로드(있으면)
    if os.path.exists(CKPT_PATH):
        agent.load(CKPT_PATH, load_optimizer=not is_eval)
        print("[DEBUG] checkpoint loaded:", CKPT_PATH)

    time.sleep(2)

    # 평가 모드면 탐험 끔
    if is_eval:
        agent.epsilon = 0.02
        print("[DEBUG] === EVAL MODE (epsilon=0, no learning) ===")
    else:
        print("[DEBUG] === TRAIN MODE (learning on) ===")

    total_reward = 0.0
    step_count = 0
    last_loss = None

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        if not is_eval:
            agent.memory.push(state, action, reward, next_state, float(done))
            last_loss = agent.learn()

        state = next_state
        total_reward += reward
        step_count += 1

        if step_count % PRINT_EVERY_STEPS == 0:
            loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
            print(
                f"step={step_count} reward={reward:.1f} total={total_reward:.1f} "
                f"eps={agent.epsilon:.3f} loss={loss_str}"
            )

        if done:
            print(f"[DEBUG] episode done. total_reward={total_reward:.1f} steps={step_count}")
            break

    # TRAIN에서만 저장
    if (not is_eval) and SAVE_ON_END:
        agent.save(CKPT_PATH)
        print("[DEBUG] checkpoint saved:", CKPT_PATH)

    print("[DEBUG] exiting.")
    sys.exit(0)


if __name__ == "__main__":
    main()
