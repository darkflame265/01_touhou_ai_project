# main_ppo.py
import argparse

from ppo_runner import run
from ppo_runner.mlp_probe import run_mlp_probe  # ✅ 추가


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--mlp-agent", action="store_true", help="use MLP PPO agent with vector observations")

    # ✅ 추가: MLP 준비 모드 (학습 안 함)
    p.add_argument("--mlp", action="store_true", help="boot practice and extract vector features only (no training)")
    p.add_argument("--mlp-max-seconds", type=float, default=60.0, help="safety cap for MLP probe duration per episode")

    return p.parse_args()


def main():
    args = parse_args()

    # ✅ MLP 모드면 학습/agent 없이 “부팅 + 벡터 추출”만
    if args.mlp:
        run_mlp_probe(
            episodes=int(args.episodes),
            no_render=bool(args.no_render),
            max_seconds=float(args.mlp_max_seconds),
        )
        return

    # 기존 PPO 루트는 그대로 유지
    if args.mlp_agent:
        ckpt_path = "checkpoints/lunatic_mlp_v1.pth"
    else:
        ckpt_path = "checkpoints/sim_v1.pth" if args.sim else "checkpoints/lunatic_v1_ch4.pth"

    run(
        episodes=int(args.episodes),
        no_render=bool(args.no_render),
        eval_mode=bool(args.eval),
        ckpt_path=ckpt_path,
        use_sim=bool(args.sim),
        use_mlp_agent=bool(args.mlp_agent),
    )


if __name__ == "__main__":
    main()
