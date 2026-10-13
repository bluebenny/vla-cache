# PLAN001 요약: Thor에서 B0 / V' LIBERO-Spatial (10 task × 5 에피소드)

## 결론
- 두 모드 모두 50 에피소드를 예외 없이 마쳤고, CSV 검증(6.1)은 전부 통과했다.
- 성공률: B0 38/50(76%), V' 41/50(82%).
- **V'는 B0보다 빠르지 않았다.** 재사용 스텝의 `t_total_ms`는 V' 487.3 ms, B0 471.2 ms(중앙값)로 V'/B0 = 1.034(평균 기준 1.019)다.
  - 재사용 prefill은 99.3 ms로 V'의 비재사용 prefill(105.7 ms)보다 약 6 ms 짧다.
  - 그러나 V'의 모듈 내 attention 통계 계산(`attn_info`, 재사용 스텝 평균 11.1 ms)과 select(평균 2.8 ms)가 더해져 B0 prefill(85.8 ms)보다 길다.
  - decode(스텝당 약 365 ms, 전체의 약 75%)는 두 모드가 같다.
- 성공률 차이는 에피소드 50개 기준이라 모드 간 우열로 해석하기에는 표본이 작다.

## 환경
- 장비: Jetson Thor, `# R39 (release), REVISION: 2.1`, `NV Power Mode: 120W`(변경하지 않음)
- 컨테이너 `plan001`: `nvcr.io/nvidia/pytorch:26.05-py3`, venv(`--system-site-packages`)
- 저장소: 브랜치 `exp/plan001-thor`, 커밋 `83d760b`(기반 `17cace9`)
- 체크포인트: `openvla/openvla-7b-finetuned-libero-spatial`(가중치는 기존 사본 복사, 나머지는 HF에서 새로 받음)
- 패키지
```
Python 3.12.3
accelerate                       1.15.0
huggingface_hub                  0.36.2
keras                            3.15.1
mujoco                           3.3.0
numpy                            2.1.0
robosuite                        1.4.1
tensorflow                       2.21.0
timm                             0.9.10
tokenizers                       0.21.1
torch                            2.12.0a0+5aff3928d8.nv26.5.50603568
torchvision                      0.27.0a0+6214cb20.nv26.5.50603568
transformers                     4.47.0
transformers @ git+https://github.com/siyuhsu/transformers.git@2302fce58afa3a4f8461625b1394f9e9c8a7f1ea
```
- 계획서와 다른 점
  - numpy는 `numpy<2`가 아니라 컨테이너의 2.1.0을 유지했다(NV torch 빌드 기준). 그래서 TensorFlow는 2.15가 아니라 2.21.0이다.
  - mujoco 3.3.0, bddl 1.0.1로 고정했다(이전 Thor 실행에서 쓰던 버전).
  - LIBERO 사본의 `torch.load(init_states_path)`에 `weights_only=False`를 넣었다(torch 2.6 이상 기본값 때문).
  - LIBERO는 editable 설치로 import되지 않아 `PYTHONPATH`로 잡았다.
  - TensorFlow 전처리는 공개 코드 그대로 썼다. `tf_free.py` 대체는 쓰지 않았다.
- 실행: 2026-10-13 03:26:51 시작, B0 04:26:58 종료, V' 05:28:31 종료. seed 7, 초기 상태 0~4.

## 검증과 통계
아래는 `PLAN001-report.py`의 출력이다(`python3 PLAN001-report.py .`).

### B0: validation
- rows 6640, episodes 50, successes 38/50 (eval log: 38), `Caught exception` 0
- ordering `t_vision <= t_generate <= t_total`: 0 rows violate, max excess 0.000 ms
- problems: none
- per-task successes (task 0..9, of 5): [4, 3, 4, 5, 2, 5, 5, 5, 3, 2]
- mean episode length: 132.8 steps

### Vp: validation
- rows 6677, episodes 50, successes 41/50 (eval log: 41), `Caught exception` 0
- ordering `t_vision <= t_generate <= t_total`: 0 rows violate, max excess 0.000 ms
- problems: none
- per-task successes (task 0..9, of 5): [3, 4, 4, 4, 4, 4, 5, 5, 3, 5]
- mean episode length: 133.5 steps

### Time statistics (ms; mean / median / p90)

**all steps**

| mode | n | t_total_ms | t_generate_ms | t_vision_ms | t_prune_ms |
|---|---|---|---|---|---|
| B0 | 6640 | 478.5 / 471.2 / 473.5 | 476.6 / 469.4 / 471.7 | 15.6 / 15.6 / 15.7 | 0.0 / 0.0 / 0.0 |
| Vp | 6677 | 487.4 / 487.4 / 492.6 | 482.9 / 483.0 / 488.1 | 15.5 / 15.5 / 15.6 | 2.8 / 2.7 / 3.1 |

**step == 0**

| mode | n | t_total_ms | t_generate_ms | t_vision_ms | t_prune_ms |
|---|---|---|---|---|---|
| B0 | 50 | 495.3 / 479.6 / 484.5 | 487.8 / 472.9 / 476.3 | 21.0 / 17.1 / 18.6 | 0.0 / 0.0 / 0.0 |
| Vp | 50 | 505.9 / 499.8 / 502.3 | 497.9 / 491.9 / 494.5 | 21.4 / 18.0 / 18.5 | 0.0 / 0.0 / 0.0 |

**step > 0**

| mode | n | t_total_ms | t_generate_ms | t_vision_ms | t_prune_ms |
|---|---|---|---|---|---|
| B0 | 6590 | 478.3 / 471.2 / 473.5 | 476.6 / 469.4 / 471.7 | 15.5 / 15.5 / 15.7 | 0.0 / 0.0 / 0.0 |
| Vp | 6627 | 487.2 / 487.3 / 492.5 | 482.8 / 483.0 / 488.1 | 15.4 / 15.5 / 15.6 | 2.8 / 2.7 / 3.1 |

### JSONL stage detail (ms; mean / median / p90)

| mode | subset | n | prefill_ms | decode_ms | decode_wall_ms | attn_info_ms | select_static_wall_ms | select_task_wall_ms | select_schedule_wall_ms | n_reusable |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | no reuse | 6640 | 85.8 / 85.6 / 86.8 | 372.1 / 364.9 / 366.4 | 150.6 / 147.7 / 148.9 | - | - | - | - | 0.0 / 0.0 / 0.0 |
| Vp | reuse | 6627 | 99.3 / 98.7 / 103.8 | 364.9 / 365.0 / 366.1 | 148.6 / 148.5 / 149.5 | 11.1 / 10.0 / 14.9 | 1.0 / 0.9 / 1.3 | 1.1 / 1.0 / 1.1 | 0.3 / 0.3 / 0.4 | 83.6 / 85.0 / 91.0 |
| Vp | no reuse | 50 | 105.7 / 105.4 / 106.8 | 366.3 / 365.0 / 366.2 | 150.3 / 148.3 / 148.9 | 16.8 / 19.0 / 19.5 | - | - | - | 0.0 / 0.0 / 0.0 |

### V' / B0 ratio of t_total_ms

- V' reuse steps: n=6627, mean 487.2, median 487.3
- B0 steps with step > 0: n=6590, mean 478.3, median 471.2
- ratio V'/B0: mean 1.0185, median 1.0342

## 해석할 때 주의할 점
- REPORT001 6절: V'는 재사용 prefill에서 pruning 전인 0·1층에도 명시 causal mask를 쓴다. 그래서 B0→V'의 재사용 스텝 `prefill_ms`에는 VLA-Cache 본래 효과 외에 32층 중 2개 층의 커널 차이가 섞여 있다.
- `decode_ms`(CUDA event 합)가 `decode_wall_ms`보다 크다. 두 값의 차이 원인은 확인하지 않았다. 모드 간 비교에는 영향이 없다(두 모드에서 같은 양상).
- `t_total_ms`, `t_prune_ms`는 wall, `t_vision_ms`, `t_generate_ms`는 CUDA event 기준이다.
- 스모크(task 0 한 에피소드)에서는 스텝당 약 610~625 ms였고 본 실험은 약 470~490 ms였다. 차이 원인은 확인하지 않았다.

## 산출물 (`src/openvla/` 기준, 모두 git 추적 대상 아님)
- `experiments/logs/PLAN001-steps-B0.csv`, `PLAN001-steps-Vp.csv`: 스텝별 CSV
- `experiments/logs/EVAL-libero_spatial-openvla-{B0,Vp}-*--PLAN001-timing.jsonl`: 원본 타이밍
- `experiments/logs/EVAL-...--PLAN001.txt`, `PLAN001-{B0,Vp}.out`, `PLAN001-all.out`: 실행 로그
- `experiments/logs/PLAN001-report.py`, `PLAN001-stats.md`, `PLAN001-env.txt`
- `experiments/logs/SMOKE-*`, `EVAL-...--smoke*`: 스모크 런
- `rollouts/2026_10_13/`: mp4 102개(스모크 2개 포함)
