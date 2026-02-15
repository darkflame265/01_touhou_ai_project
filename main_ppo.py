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

    # ✅ 추가: MLP 준비 모드 (학습 안 함)
    p.add_argument("--mlp", action="store_true", help="boot practice and extract vector features only (no training)")

    return p.parse_args()


def main():
    args = parse_args()

    # ✅ MLP 모드면 학습/agent 없이 “부팅 + 벡터 추출”만
    if args.mlp:
        run_mlp_probe(
            episodes=int(args.episodes),
            no_render=bool(args.no_render),
        )
        return

    # 기존 PPO 루트는 그대로 유지
    run(
        episodes=int(args.episodes),
        no_render=bool(args.no_render),
        eval_mode=bool(args.eval),
        ckpt_path="checkpoints/sim_v1.pth" if args.sim else "checkpoints/lunatic_v1_ch4.pth",
        use_sim=bool(args.sim),
    )


if __name__ == "__main__":
    main()
