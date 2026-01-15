# main_ppo.py
import argparse
from ppo_runner import run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--sim", action="store_true")  # ✅ 추가
    return p.parse_args()


def main():
    args = parse_args()
    run(
        episodes=int(args.episodes),
        no_render=bool(args.no_render),
        eval_mode=bool(args.eval),
        ckpt_path = "checkpoints/sim_v1.pth" if args.sim else "checkpoints/lunatic_v1_ch4.pth",
        use_sim=bool(args.sim),
    )


if __name__ == "__main__":
    main()
