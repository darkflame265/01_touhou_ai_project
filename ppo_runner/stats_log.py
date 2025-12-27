# ppo_runner/stats_log.py
import os
import re
import tempfile
from datetime import datetime

STATS_BEGIN = "# === PPO_STATS_BEGIN ==="
STATS_END = "# === PPO_STATS_END ==="

def default_stats():
    return {
        "total_completed": 0,
        "best_reward": None,
        "best_reward_ts": "",
        "best_reward_ep": "",
        "best_reward_run": "",
        "best_survival": None,
        "best_survival_ts": "",
        "best_survival_ep": "",
        "best_survival_run": "",
    }

def _parse_float_or_none(s: str):
    try:
        return float(s)
    except Exception:
        return None

def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _atomic_write(path: str, text: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="tmp_stats_", suffix=".txt", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _extract_stats_block(text: str):
    if (STATS_BEGIN not in text) or (STATS_END not in text):
        return None, text

    pattern = re.compile(re.escape(STATS_BEGIN) + r"(.*?)" + re.escape(STATS_END), re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None, text

    block = m.group(1)
    stats = default_stats()

    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ("=" not in line):
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k == "total_completed":
            try:
                stats[k] = int(v)
            except Exception:
                pass
        elif k in ("best_reward", "best_survival"):
            stats[k] = _parse_float_or_none(v)
        else:
            if k in stats:
                stats[k] = v

    return stats, text

def _format_stats_block(stats: dict) -> str:
    br = "" if stats["best_reward"] is None else f"{stats['best_reward']:.6f}"
    bs = "" if stats["best_survival"] is None else f"{stats['best_survival']:.3f}"

    lines = [
        STATS_BEGIN,
        f"total_completed={int(stats['total_completed'])}",
        f"best_reward={br}",
        f"best_reward_ts={stats['best_reward_ts']}",
        f"best_reward_run={stats['best_reward_run']}",
        f"best_reward_ep={stats['best_reward_ep']}",
        f"best_survival={bs}",
        f"best_survival_ts={stats['best_survival_ts']}",
        f"best_survival_run={stats['best_survival_run']}",
        f"best_survival_ep={stats['best_survival_ep']}",
        STATS_END,
        "",
    ]
    return "\n".join(lines)

def ensure_stats_header(log_path: str) -> dict:
    text = _read_text(log_path)
    stats, _ = _extract_stats_block(text)
    if stats is not None:
        return stats

    stats = default_stats()
    new_text = _format_stats_block(stats) + text
    _atomic_write(log_path, new_text)
    return stats

def update_stats_in_file(log_path: str, stats: dict):
    text = _read_text(log_path)
    cur_stats, _ = _extract_stats_block(text)

    new_block = _format_stats_block(stats)
    if cur_stats is None:
        new_text = new_block + text
    else:
        pattern = re.compile(re.escape(STATS_BEGIN) + r".*?" + re.escape(STATS_END) + r"\n?", re.DOTALL)
        new_text = pattern.sub(new_block.strip() + "\n", text, count=1)

    _atomic_write(log_path, new_text)

def maybe_update_records(stats: dict, reward: float, survival_sec: float, run_ts: str, ep_tag: str):
    stats["total_completed"] = int(stats.get("total_completed", 0)) + 1

    br = stats.get("best_reward", None)
    if (br is None) or (float(reward) > float(br)):
        stats["best_reward"] = float(reward)
        stats["best_reward_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats["best_reward_run"] = run_ts
        stats["best_reward_ep"] = ep_tag

    bs = stats.get("best_survival", None)
    if (bs is None) or (float(survival_sec) > float(bs)):
        stats["best_survival"] = float(survival_sec)
        stats["best_survival_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats["best_survival_run"] = run_ts
        stats["best_survival_ep"] = ep_tag

def stats_one_line(stats: dict) -> str:
    br = stats["best_reward"]
    bs = stats["best_survival"]
    br_s = "N/A" if br is None else f"{br:.3f}"
    bs_s = "N/A" if bs is None else f"{bs:.2f}s"
    return f"[STATS] total_completed={stats['total_completed']} | best_reward={br_s} | best_survival={bs_s}"

def append_run_header(log_path: str, run_ts: str, episodes: int, is_eval: bool, stats: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n========\n")
        f.write(f"[RUN] {run_ts}  episodes={episodes} eval={str(bool(is_eval))}\n")
        f.write(stats_one_line(stats) + "\n")
        f.write("idx\treward\tsurvival_sec\tnote\n")
