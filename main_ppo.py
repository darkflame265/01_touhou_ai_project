import time
import sys

from env.game_env import GameEnv
from env.actions import ACTIONS
from agents.ppo_agent import PPOAgent


def main():
    env = GameEnv()
    state = env.reset()

    agent = PPOAgent(
        input_channels=4,
        num_actions=len(ACTIONS),
    )

    time.sleep(3)

    total_reward = 0.0
    step = 0

    while True:
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.store(state, action, reward, done, log_prob, value)

        state = next_state
        total_reward += reward
        step += 1

        if done:
            print(f"[PPO] episode done | steps={step} total_reward={total_reward:.1f}")
            agent.finish_episode()
            break

    print("[PPO] training finished")
    sys.exit(0)


if __name__ == "__main__":
    main()
