# ppo_runner/train_loop.py
import os
import time
from datetime import datetime
from collections import Counter

from env.game_env import GameEnv
from env.controller import release_all, set_attack_hold, cleanup_inputs_on_exit
from env.menu import detect_location, boot_into_practice
from env.actions import ACTIONS
from agents.ppo_agent import PPOAgent

from ppo_runner.cli import parse_args
from ppo_runner.hotkeys_win import esc_pressed
from ppo_runner.render import apply_no_render, pump_cv_events_once
from ppo_runner.stats_log import (
    ensure_stats_header,
    append_run_header,
    maybe_update_records,
    update_stats_in_file,
    stats_one_line,
)

def safe_release_inputs():
    try:
        set_attack_hold(False)
    except Exception:
        pass
    try:
        release_all()
    except Exception:
        pass

def try_clear_agent_rollout(agent):
    for name in ("clear", "reset_buffer", "reset_storage", "clear_buffer", "clear_rollout"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn()
                print(f"[PPO] agent.{name}() called (abort cleanup)")
            except Exception:
                pass
            break

def safe_save_checkpoint(agent, ckpt_path: str) -> bool:
    try:
        ret = agent.save(ckpt_path)
        return bool(ret) if isinstance(ret, bool) else True
    except Exception as e:
        print(f"[WARN] checkpoint save failed (ignored): {e}")
        return False

def run_main():
    args = parse_args()
    is_eval = bool(args.eval)

    CKPT_PATH = "checkpoints/lunatic_v1_ch4.pth"
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]
    log_path = os.path.join(os.path.dirname(CKPT_PATH), f"{pth_name}_episode_log.txt")

    stats = ensure_stats_header(log_path)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_run_header(log_path, run_ts, int(args.episodes), is_eval, stats)
    print(stats_one_line(stats))

    env = GameEnv(screen_mode="low")
    if args.no_render:
        apply_no_render(env)

    # 부팅 시 1회만(인게임 중에는 절대 안 돌림)
    print("\n[BOOT] 현재 화면 위치 감지 중.")
    st = detect_location(env.screen)
    print(f"[BOOT] state={st.get('state')} selected={st.get('selected_name')}")
    print("[BOOT] practice 진입 준비.")

    obs_channels = int(getattr(env.obs, "obs_channels", 1))
    stack_size = int(getattr(env.s, "frame_stack_size", 4))
    input_channels = obs_channels * stack_size
    print(f"[PPO] input_channels={input_channels} (obs_channels={obs_channels} * stack={stack_size})")

    agent = PPOAgent(
        input_channels=input_channels,
        num_actions=len(ACTIONS),
        obs_channels_per_frame=obs_channels,
    )

    if os.path.exists(CKPT_PATH):
        agent.load(CKPT_PATH, load_optimizer=False)
        print(f"[PPO] checkpoint loaded: {CKPT_PATH}")
    else:
        print("[PPO] no checkpoint found, training from scratch" if not is_eval
              else "[PPO][EVAL] no checkpoint found (evaluating random policy)")

    # 하이퍼파라미터 override(네 기존 유지)
    agent.ent_coef = 0.04
    agent.ent_min = 0.005
    agent.ent_decay = 0.9995
    agent.ent_warmup_updates = 30
    agent.clip_eps = 0.15
    agent.rollout_steps = 128
    agent.update_epochs = 5

    print("\n[INFO] ESC 중단: Windows 전역 감지(GetAsyncKeyState)")
    print(" - 게임 창이 포커스여도 ESC를 잡고 즉시 종료합니다.\n")
    time.sleep(0.5)

    stop_requested = False

    try:
        for ep in range(1, int(args.episodes) + 1):
            if esc_pressed():
                stop_requested = True
                print("[STOP] ESC pressed before episode start -> stopping.")
                break

            print(f"\n========== EPISODE {ep}/{args.episodes} ==========")

            # 에피소드 시작 전(로비/스코어)에서만 메뉴 제어
            print("[MENU] [practice 준비/진입 중.]")
            ok = boot_into_practice(env.screen, max_sec_lobby=12.0)
            if not ok:
                print("[EP_PREP][WARN] boot_into_practice failed (will continue and let env/reset try)")
            print("[MENU] [practice 준비/진입 완료]")

            safe_release_inputs()
            state = env.reset()

            if not args.no_render:
                pump_cv_events_once()

            ep_t0 = time.time()
            done = False
            total_reward = 0.0
            steps = 0
            slow_count = 0
            action_counter = Counter()
            aborted = False

            # ✅ 인게임 루프: update/save/무거운 작업 금지
            while not done:
                if esc_pressed():
                    stop_requested = True
                    aborted = True
                    print("[STOP] ESC pressed -> aborting NOW (release inputs, NO SAVE/NO UPDATE for this episode).")
                    safe_release_inputs()
                    done = True
                    break

                action_idx, log_prob, value = agent.select_action(state)

                action_name = ACTIONS[action_idx].name
                action_counter[action_name] += 1
                if action_name.startswith("SLOW"):
                    slow_count += 1

                next_state, reward, done = env.step(action_idx)

                if not is_eval:
                    exec_idx = getattr(env.s, "exec_action_idx", action_idx)
                    agent.store(state, exec_idx, reward, done, log_prob, value)

                state = next_state
                total_reward += float(reward)
                steps += 1

                if not args.no_render:
                    pump_cv_events_once()

            # 에피소드 종료 후(로비/스코어로 나간 뒤) 무거운 작업 수행
            survival_sec = time.time() - ep_t0
            slow_ratio = slow_count / max(1, steps)
            top_actions_str = ";".join(f"{k}:{v}" for k, v in action_counter.most_common(5))
            note_parts = []
            if is_eval:
                note_parts.append("EVAL")
            if aborted:
                note_parts.append("ABORTED")
            note = ",".join(note_parts)

            print(
                f"[PPO] episode end | steps={steps} total_reward={total_reward:.1f} "
                f"survival_sec={survival_sec:.2f} slow_ratio={slow_ratio:.3f} "
                f"top_actions={top_actions_str} {note}"
            )

            ep_tag = f"({ep}/{args.episodes})"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{ep_tag}\t{total_reward:.6f}\t{survival_sec:.3f}\t{note}\n")

            if aborted:
                if not is_eval:
                    try_clear_agent_rollout(agent)
                print("[STOP] Episode aborted -> stopping.")
                break

            if not is_eval:
                # ✅ update는 여기서만(인게임 중엔 절대 X)
                updates = 0
                while agent.should_update():
                    agent.update(last_state=state, last_done=True)
                    updates += 1
                if updates > 0:
                    print(f"[PPO] updates_after_episode={updates}")

                maybe_update_records(stats, total_reward, survival_sec, run_ts, ep_tag)
                update_stats_in_file(log_path, stats)
                print(stats_one_line(stats))

                # ✅ 에피소드마다 저장(요구사항 유지)
                if safe_save_checkpoint(agent, CKPT_PATH):
                    print("[PPO] checkpoint saved")
                else:
                    print("[WARN] checkpoint save failed -> continue training without stopping")
            else:
                print(stats_one_line(stats))

            if stop_requested:
                print("[STOP] Training stopped by ESC. Exiting.")
                break

            if ep < args.episodes:
                time.sleep(0.2)

    finally:
        cleanup_inputs_on_exit()
        if not args.no_render:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n[PPO] Finished.")
