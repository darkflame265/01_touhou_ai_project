# env/game_env.py

import time
import numpy as np
from env.screen import Screen
from env.controller import press_keys, release_all
from env.actions import ACTIONS
from env.ui_lives import count_lives_from_img

class GameEnv:
    def __init__(self):
        self.screen = Screen()
        self.prev_state = None
        self.done = False

        self.lives = 3
        self.last_death_time = 0.0
        self.death_cooldown = 1.0  # 1초 쿨다운

        self.prev_lives = None
        self.last_hit_time = 0.0
        self.hit_cooldown = 1.0  # 중복 hit 방지(초)

        self.prev_play_gray = None
        self.no_motion_count = 0

        self.prev_play_mean = None
        self.prev_play_std = None
        self.scene_change_count = 0


    def reset(self):
        release_all()
        time.sleep(0.5)

        self.lives = 3
        self.done = False
        self.last_hit_time = 0.0

        img = self.screen.capture()
        state = self.screen.preprocess(img)
        self.prev_state = state

        # UI 잔기(예비 잔기) 저장
        self.prev_ui_lives = count_lives_from_img(img)

        img = self.screen.capture()
        play = self.screen.get_playfield_gray(img)
        self.prev_play_gray = play
        self.no_motion_count = 0

        self.prev_play_mean = float(play.mean())
        self.prev_play_std = float(play.std())
        self.scene_change_count = 0



        return np.zeros_like(state)


    def step(self, action_idx):
        action = ACTIONS[action_idx]
        press_keys(action)
        time.sleep(0.03)

        img = self.screen.capture()
        state = self.screen.preprocess(img)
        diff_state = np.abs(state - self.prev_state)

        reward = 0.1
        done = False

        now = time.time()
        ui_now = count_lives_from_img(img)

        if (now - self.last_hit_time) > self.hit_cooldown:
            if ui_now < self.prev_ui_lives:
                self.lives -= 1
                reward = -10
                self.last_hit_time = now
                print(f"[DEBUG] HIT! internal lives={self.lives} (ui {self.prev_ui_lives}->{ui_now})")

        self.prev_ui_lives = ui_now


        if self.lives <= 0:
            reward = -100
            done = True
            print("[DEBUG] GAME OVER! internal lives=0")

            release_all()

        # --- (1) 플레이필드 변화량 기반: 멈춤(컨티뉴 화면) 감지 ---
        # --- playfield 추출
        curr_play = self.screen.get_playfield_gray(img)

        # --- (A) freeze 감지: 스토리 Continue 화면에 강함
        motion = self.screen.playfield_motion_score(self.prev_play_gray, curr_play)
        self.prev_play_gray = curr_play

        if motion < 0.004:          # practice는 보통 이 조건을 잘 안 만족함
            self.no_motion_count += 1
        else:
            self.no_motion_count = 0

        # --- (B) scene change 감지: practice 로비 이동에 강함
        mean_now = float(curr_play.mean())
        std_now = float(curr_play.std())

        d_mean = abs(mean_now - self.prev_play_mean)
        d_std = abs(std_now - self.prev_play_std)

        # 장면이 확 바뀌면 mean/std가 크게 튄다
        if d_mean > 6.0 or d_std > 8.0:   # 값은 스케일(0~255) 기준이라 이런 숫자 대가 잘 맞음
            self.scene_change_count += 1
        else:
            self.scene_change_count = 0

        self.prev_play_mean = mean_now
        self.prev_play_std = std_now

        # --- 디버그(원하면)
        # print(f"[DEBUG] motion={motion:.4f}, d_mean={d_mean:.2f}, d_std={d_std:.2f}, freeze={self.no_motion_count}, scene={self.scene_change_count}")

        # --- 에피소드 종료 조건 (둘 중 하나면 done)
        # 1) 멈춤이 1초 가까이 지속되면(continue 화면)
        if self.no_motion_count >= 30:  # step sleep이 0.03이면 약 0.9초
            print("[DEBUG] EPISODE END: frozen/continue detected")
            reward = -100
            done = True
            release_all()

        # 2) 장면 급변이 연속으로 몇 프레임 뜨면(로비 이동)
        if self.scene_change_count >= 3:
            print("[DEBUG] EPISODE END: lobby/scene change detected")
            reward = -100
            done = True
            release_all()



        self.prev_state = state
        return diff_state, reward, done

    def _calc_reward(self, state):
        brightness = state.mean()
        reward = 0.1  # 살아있으면 기본 보상

        #print(f"[DEBUG] brightness: {brightness:.4f}")

        # 사망 감지
        if brightness > 0.80:
            reward = -100
            self.done = True
            release_all()
            print("[DEBUG] 죽음 감지됨")

        return reward