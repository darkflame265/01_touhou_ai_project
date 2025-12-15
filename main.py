import os
import time

from env.game_env import GameEnv
from env.actions import ACTIONS
from agents.dqn_agent import DQNAgent

CKPT_PATH = "checkpoints/dqn_practice.pth"
PRINT_EVERY_STEPS = 50
SAVE_EVERY_STEPS = 2000

env = GameEnv()
state = env.reset()

agent = DQNAgent(
    input_channels=1,
    num_actions=len(ACTIONS),
)

# ✅ 이어서 학습(체크포인트 있으면 로드)
if os.path.exists(CKPT_PATH):
    agent.load(CKPT_PATH, load_optimizer=True)
    print("[DEBUG] checkpoint loaded:", CKPT_PATH)

time.sleep(3)

total_reward = 0.0
step_count = 0

while True:
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)

    # ✅ done은 float(0/1)로 저장하면 학습이 깔끔함
    agent.memory.push(state, action, reward, next_state, float(done))

    loss = agent.learn()

    state = next_state
    total_reward += reward
    step_count += 1

    # ✅ 로그 출력
    if step_count % PRINT_EVERY_STEPS == 0:
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        print(
            f"step={step_count} "
            f"reward={reward:.1f} total={total_reward:.1f} "
            f"eps={agent.epsilon:.3f} loss={loss_str}"
        )

    # ✅ 체크포인트 저장
    if step_count % SAVE_EVERY_STEPS == 0:
        agent.save(CKPT_PATH)
        print("[DEBUG] checkpoint saved:", CKPT_PATH)

    # ✅ 에피소드 종료 처리
    if done:
        print(f"[DEBUG] episode done. total_reward={total_reward:.1f} steps={step_count}")
        agent.save(CKPT_PATH)
        print("[DEBUG] checkpoint saved:", CKPT_PATH)

        # 다음 에피소드로 리셋
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        time.sleep(1)
