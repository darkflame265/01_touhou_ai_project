# 프로젝트 개요: 동방홍마향 AI (01_touhou_ai)

PyTorch + PPO 강화학습으로 동방홍마향(東方紅魔郷)을 자동 플레이하는 AI.
**목표**: 루나틱 난이도 클리어

---

## 실행 방법

```bash
# 1. 가상환경 활성화
venv\Scripts\activate

# 2. 동방홍마향 실행 (창모드 필수) -> Practice 모드 -> 캐릭/무기 선택 -> 게임화면 진입

# 3. 학습 실행
python main_ppo.py --episodes 1              # CNN 기반 PPO
python main_ppo.py --mlp --episodes 1        # MLP 벡터 추출만 (학습 없음)
python main_ppo.py --mlp-agent --episodes 1  # MLP 기반 PPO 학습
python main_ppo.py --eval --episodes 1       # 평가 모드 (학습 없음)
python main_ppo.py --sim --episodes 1        # 시뮬레이터 환경 사용
```

---

## 프로젝트 구조

```
main_ppo.py          # 진입점 (argparse -> ppo_runner 또는 mlp_probe 호출)
env/
  game_env.py        # GameEnv: 핵심 RL 환경 (reset/step)
  screen.py          # 화면 캡처, UI 감지, 사망 감지
  controller.py      # 키보드 입력 제어 (pynput 등)
  obs_builder.py     # ObsBuilder: 4채널 관찰 생성 + 트래커 통합
  reimu_tracker_cv.py       # 플레이어(레이무) 위치 추적 (OpenCV)
  bullet_tracker_cv.py      # 탄환 위치 추적 (OpenCV)
  actions.py         # 행동 공간 정의
  menu.py            # 게임 메뉴 자동화 (boot_into_practice)
  episode_guard.py   # 에피소드 종료 조건 관리
  ui_guard.py        # UI 목숨 수 감지
  ui_lives.py        # 목숨 UI 파싱
  game_env_util/     # 환경 내부 유틸
    reward_engine.py       # 보상 계산
    action_masking.py      # 경계 밖 행동 마스킹
    action_executor.py     # 행동 실행
    frame_skipper.py       # 중복 프레임 스킵
    obs_pack.py            # 프레임 스택 패킹
agents/
  ppo_agent.py       # CNN 기반 PPO Agent
  mlp_ppo_agent.py   # MLP 기반 PPO Agent (현재 주력)
  dqn_agent.py       # DQN Agent (미사용)
  replay_buffer.py   # 리플레이 버퍼
models/
  shared/            # 공유 모델 (ActorCriticMLP 등)
  low/               # CNN 기반 저해상도 모델
ppo_runner/
  runner.py          # 학습 루프 메인
  train_loop.py      # 학습 루프 세부
  mlp_probe.py       # MLP 벡터 추출 전용 모드
  hotkeys.py         # ESC/P 핫키
  render.py          # 디버그 렌더링
  stats_log.py       # 통계 로깅
sim/
  sim_env.py         # 시뮬레이터 환경 (실제 게임 없이 테스트)
vision/              # 데이터 수집/라벨링/모델 학습 도구
checkpoints/         # 저장된 체크포인트 (.pth)
runs/                # 에피소드 벡터 데이터 (.npz)
```

---

## 핵심 설계

### 관찰 공간 (ObsBuilder)
- **4채널** float32 이미지 (128x128)
  - ch0: 현재 그레이스케일 + 플레이어 마커
  - ch1: 이전 프레임 그레이스케일
  - ch2: 프레임 차이 (absdiff)
  - ch3: 플레이어 위치 가우시안 힌트맵
- **Frame stack = 4** (4스텝 히스토리 concat)
- 플레이필드 크롭: UI 패널 제거 후 게임 영역만 사용

### 보상 함수 (RewardEngine)
```
alive_reward   = +0.03  (매 스텝 생존 보상)
hit_pen        = -1.5   (피격)
death_pen      = -1.5   (사망)
abort_pen      = -1.5   (UI 이탈 등 비정상 종료)
risk_penalty           (탄환 밀집도 기반 위험 패널티)
diff_risk_penalty      (위험 증가량 패널티)
position_penalties     (하단/우측 구석 페널티)
```

### 학습 알고리즘 (PPO)
- Clip eps: 0.10
- GAE lambda: 0.95
- Rollout steps: 256
- Mini batch: 128
- Entropy decay: 0.9990 (탐험 점진적 감소)
- Value clip: 0.2

### 행동 마스킹
- 화면 경계 근처 이동 행동 비활성화 (margin_px=90)
- 폭탄 비활성화 (disable_bomb=True)

---

## 현재 상태 (2026-03)

- CNN 기반 40시간+ 학습해도 회피 안 됨 -> MLP 방식으로 전환 중
- `--mlp` 모드로 벡터 데이터 수집 중 (`runs/*.npz`)
- MLP Agent (`mlp_ppo_agent.py`) + `ActorCriticMLP` 모델 구조 완성
- 체크포인트: `checkpoints/lunatic_mlp_v1.pth` (MLP), `checkpoints/lunatic_v1_ch4.pth` (CNN)

---

## 주의사항

- 실제 게임 창이 포커스된 상태에서 실행해야 함
- 게임은 반드시 창모드로 실행
- 기본 목숨 수: 3
- CUDA 병렬 학습 불가 (실게임 기반이므로 순차 실행만 가능)
- ESC 또는 P 키로 학습 중단 가능
