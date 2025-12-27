# ppo_runner/cli.py
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-render", action="store_true", help="disable debug/obs windows")
    p.add_argument("--eval", action="store_true", help="evaluation mode (no training, no checkpoint, no STATS update)")
    return p.parse_args()
