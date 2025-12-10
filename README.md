# TITAN v2.0 - 강화학습 기반 게임 AI 시스템

## 📋 개요

TITAN은 AION2 게임을 위한 강화학습 기반 AI 시스템입니다. v2.0에서는 핵심 알고리즘과 학습 안정성이 크게 개선되었습니다.

## 🚀 v2.0 주요 개선사항

### 1. 상태 변화 기반 보상 시스템 (Critical Fix)

**문제점:** 기존 시스템은 버튼을 누르는 것만으로 보상을 받았습니다.
```python
# 기존 (v1.0) - 잘못된 보상
if action == ATK_WEAK:
    reward += 0.5  # 실제 킬 여부와 무관
```

**해결책:** 액션 전/후 상태를 비교하여 실제 결과에 따라 보상합니다.
```python
# 개선 (v2.0) - 상태 변화 기반 보상
if monsters_before > 0 and monsters_after < monsters_before:
    reward += 5.0 * (monsters_before - monsters_after)  # 실제 킬에만 보상
```

### 2. 프레임 변화 감지 (Static Frame Detection)

정지 이미지에서 학습하는 것을 방지합니다.

```python
class FrameChangeDetector:
    def is_static(self, frame) -> bool:
        # 다운샘플링으로 빠른 비교
        small = frame[::10, ::10, :]
        diff = np.abs(small - prev_frame).mean() / 255.0
        return diff < 0.02  # 2% 미만 변화 = 정지
```

### 3. Double DQN

Q값 과대추정 문제를 해결합니다.

```python
# 기존 DQN
next_q = target_net(next_state)
target = reward + gamma * next_q.max()

# Double DQN (v2.0)
best_action = policy_net(next_state).argmax()  # 정책망으로 액션 선택
target = reward + gamma * target_net(next_state)[best_action]  # 타겟망으로 평가
```

### 4. 실제 상태 벡터

더미값 대신 실제 게임 상태를 반영합니다.

```python
# 기존 (v1.0) - 더미값
status = [1.0, 1.0, 0.0, 0.0, 1.0]  # 항상 고정

# 개선 (v2.0) - 실제 상태
status = [
    monster_count / 10,    # 몬스터 비율
    loot_count / 5,        # 루트 비율
    has_target,            # 타겟 유무
    target_distance,       # 타겟 거리
    time_factor            # 시간 컨텍스트
]
```

### 5. Gradient Clipping

학습 안정성을 위한 그래디언트 클리핑입니다.

```python
self.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
```

### 6. 에피소드 관리

연속적인 학습 대신 에피소드 단위로 관리합니다.

```python
def _check_episode_end(self):
    # 1. 최대 길이 도달
    if episode_steps >= 10000: return True
    # 2. 장기 정체
    if no_change_count >= 100: return True
    # 3. 필드 클리어
    if monsters_before > 0 and monsters_after == 0: return True
```

### 7. 메트릭 추적

학습 과정을 상세히 모니터링합니다.

```python
class MetricsTracker:
    - 이동 평균 보상 (100, 1000 스텝)
    - 액션 분포
    - 킬/루트 통계
    - 보상 이유 분포
```

## 📁 프로젝트 구조

```
titan_improved/
├── config/
│   ├── __init__.py
│   ├── constants.py      # 전역 상수 (매직넘버 제거)
│   └── settings.py       # 시스템 설정
├── core/
│   ├── __init__.py
│   ├── brain.py          # 신경망 (Double DQN 지원)
│   ├── vision.py         # YOLO + HSV 비전
│   ├── memory.py         # 경험 리플레이 버퍼
│   ├── driver.py         # 입력 제어
│   ├── streamer.py       # 실시간 스트리밍
│   ├── window_manager.py # 윈도우 관리
│   ├── frame_detector.py # [NEW] 프레임 변화 감지
│   ├── reward.py         # [NEW] 상태 변화 기반 보상
│   └── metrics.py        # [NEW] 메트릭 추적
├── tools/
│   ├── __init__.py
│   ├── snapshot.py       # 스냅샷 로거
│   └── self_trainer.py   # YOLO 자가 학습
├── utils/
│   ├── __init__.py
│   └── logger.py         # 로깅 시스템
├── main.py               # 메인 실행 파일 (v2.0)
└── README.md
```

## ⚙️ 설정

### 환경변수

```bash
# 디바이스
export TITAN_DEVICE=cuda

# 학습 파라미터
export TITAN_BATCH_SIZE=512
export TITAN_LR=0.0003
export TITAN_MEMORY_CAPACITY=100000

# 기능 토글
export TITAN_USE_TRT=true      # TensorRT 사용
export TITAN_DOUBLE_DQN=true   # Double DQN 사용
export TITAN_GRAD_CLIP=10.0    # Gradient Clipping
```

### 보상 설정 (constants.py)

```python
class RewardConfig:
    # 상태 변화 기반 (큰 보상)
    KILL_REWARD = 5.0        # 몬스터 킬
    LOOT_REWARD = 3.0        # 루팅 성공
    MONSTER_FOUND = 0.5      # 몬스터 발견
    
    # 액션 시도 (작은 보상)
    ATTACK_ATTEMPT = 0.02    # 공격 시도
    SEARCH_ATTEMPT = 0.05    # 탐색 시도
    
    # 페널티
    STATIC_FRAME = -0.05     # 정지 화면
    WRONG_ACTION = -0.10     # 잘못된 액션
    STAGNATION = -0.15       # 장기 정체
```

## 🎮 사용법

### 설치

```bash
pip install -r requirements.txt
```

### 실행 방법

#### 1. 런처 사용 (권장)
```bash
# AI + GUI 동시 실행
python launcher.py

# AI만 실행
python launcher.py --ai

# GUI만 실행 (별도 창에서)
python launcher.py --gui
```

#### 2. 개별 실행
```bash
# 터미널 1: AI 실행
python main.py

# 터미널 2: GUI 대시보드
python gui.py
```

### 조작

**키보드 단축키:**
| 키 | 기능 | 설명 |
|----|------|------|
| `Delete` | 일시정지/재개 | AI 동작 토글 |
| `F5` | 시작 | GUI에서 AI 시작 |
| `Escape` | 정지 | GUI에서 AI 정지 |
| `Ctrl+C` | 종료 | 체크포인트 자동 저장 |

**GUI 버튼:**
- `▶ Start`: AI 시작 및 스트림 연결
- `⏸ Pause`: 일시정지/재개 토글
- `⏹ Stop`: AI 완전 정지

**GUI-AI 통신 (IPC):**
- GUI와 AI는 공유 메모리(`TitanIPC`)를 통해 명령을 주고받습니다.
- GUI에서 AI를 완전 제어할 수 있습니다.
- AI 상태(스텝, 보상, 킬, 루트, 엡실론)가 200ms마다 GUI에 동기화됩니다.

### 로그 해석

```
# 기존 로그 (v1.0)
Step=0 Reward=0.49 Action=ATK_WEAK Data=1/50

# 개선된 로그 (v2.0)
Step=0 R=+0.02(atk) Act=ATK_WEAK M:1→1 ε=0.95 Data=1/50
       ↑            ↑          ↑    ↑
       보상(이유)   액션       몬스터 변화  탐험율
       
# 실제 킬 발생 시
Step=5 R=+5.00(KILL+1) Act=ATK_STR M:2→1 ε=0.90 Data=2/50

# 정지 화면 감지
Step=10 R=-0.05(STATIC) Act=SEARCH M:0→0 ε=0.85 Data=2/50
```

## 📊 보상 시스템 비교

| 상황 | v1.0 보상 | v2.0 보상 |
|------|-----------|-----------|
| 정지 이미지 + 공격 | +0.49 | **-0.05** |
| 몬스터 있음 + 공격 (미스) | +0.49 | +0.02 |
| 몬스터 킬 성공 | +0.49 | **+5.00** |
| 몬스터 발견 | - | **+0.50** |
| 몬스터 없음 + 공격 | +0.49 | **-0.10** |
| 루팅 성공 | +2.00 | **+3.00** |

## 🔧 개발자 참고

### 새 보상 이유 추가

```python
# core/reward.py
class RewardConfig:
    NEW_REWARD = 1.0  # 새 보상 값

def calculate_reward(...):
    if new_condition:
        reward += RewardConfig.NEW_REWARD
        reasons.append("NEW")
```

### 메트릭 확장

```python
# core/metrics.py
class MetricsTracker:
    def log_custom_metric(self, name, value):
        self.custom_metrics[name].append(value)
```

## 📝 버전 히스토리

### v2.0.0 (Current)
- 상태 변화 기반 보상 시스템
- 프레임 변화 감지
- Double DQN
- 실제 상태 벡터
- Gradient Clipping
- 에피소드 관리
- 메트릭 추적

### v1.0.0
- 초기 버전
- 기본 DQN + LSTM
- HSV + YOLO 비전

## ⚠️ 주의사항

1. **정지 이미지 테스트**: v2.0은 정지 이미지에서 학습하지 않습니다.
2. **체크포인트 호환성**: v1.0 체크포인트는 v2.0과 호환되지 않을 수 있습니다.
3. **GPU 메모리**: Double DQN은 추가 메모리가 필요합니다.

---

## 🖥️ GUI 대시보드

PyQt5 기반의 실시간 모니터링 대시보드를 제공합니다.

### 기능

- **실시간 게임 화면** - 공유 메모리 기반 저지연 스트리밍
- **보상 그래프** - 실시간 보상 추이 및 이동 평균
- **손실 그래프** - 학습 손실 모니터링
- **액션 분포** - 각 액션의 선택 비율
- **통계 카드** - 스텝, 보상, 킬, 루트 요약
- **제어 패널** - 시작/정지/일시정지 버튼
- **시스템 로그** - 실시간 로그 표시

### 스크린샷 설명

```
┌─────────────────────────────────────────────────────────┐
│  🎮 TITAN v2.0                        ● Connected       │
├─────────────────────────┬───────────────────────────────┤
│                         │  [Steps]  [Reward]  [Kills]   │
│    실시간 게임 화면      │     124     +45.2      3      │
│                         ├───────────────────────────────┤
│                         │  📈 Rewards | 📉 Loss | 🎯 Act │
│                         │  ┌─────────────────────────┐  │
│                         │  │     보상 그래프         │  │
│                         │  └─────────────────────────┘  │
├─────────────────────────┼───────────────────────────────┤
│  System Log             │  Control Panel                │
│  [12:34:56] Kill! +5.0  │  [▶ Start] [⏸ Pause] [⏹ Stop]│
│  [12:34:57] Loot +3.0   │  Epsilon: 0.850              │
└─────────────────────────┴───────────────────────────────┘
```

### 다크 테마

GUI는 눈이 편안한 Catppuccin Mocha 다크 테마를 사용합니다.

## 📄 라이센스

Private - Anthropic Internal Use Only
