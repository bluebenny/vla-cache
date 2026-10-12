# PLAN-VLA-cache-modified-001: Thor에서 B0 / V' LIBERO-Spatial 실험

이 문서는 **Jetson Thor에서 실행되는 Claude Code**에게 주는 작업 지시서다. 위에서부터 순서대로 수행한다.
막히면 임의로 우회하지 말고 "중단 조건"에 따라 사용자에게 보고한다.

## 0. 목표와 범위
- **실험**: LIBERO-Spatial 10개 task × task당 에피소드 5개(초기 상태 0~4) = 50 에피소드를 수행한다. 이것을 **B0**와 **V'(`Vp`)** 각각에서 반복한다.
- **산출물**
  - 스텝(행동 1개)마다 1줄인 CSV. 모드당 1개, 컬럼은 2.3절과 같다.
  - 원본 타이밍 JSONL, 실행 로그
  - 결과 요약(6절)
- **하지 않는 것**
  - B0/V'의 추론 로직은 바꾸지 않는다. 이 계획에서 바꾸는 코드는 CSV 기록뿐이다.
  - B1/V 실행, 전원 모드 변경, git push는 사용자 지시가 있을 때만 한다.

## 1. 먼저 읽을 것 (clone 후)
1. `CLAUDE.md`(저장소 루트): 모드 정의, 포크 transformers, 불변 규칙, 타이밍 필드
2. `REPORT001.md`: V/V'/B0의 차이. 특히 3절과 6절(B0 대비 V'의 차이, 재사용 prefill의 명시 mask)
3. `src/openvla/README_VLA_Cache.md`: 실행 방법

## 2. 측정 지표 정의 (이 실험용 매핑)
원래 지표 목록은 다른 실험(C0/CP/CS/CPS, skip 스텝)용이다. 이 실험에는 아래처럼 대응시킨다. **컬럼 이름과 순서는 원래 목록 그대로 둔다.**

### 2.1 컬럼별 값
| # | 컬럼 | 이 실험에서의 값 | 출처 (`get_vla_action`의 `StageTimer` 행) |
|---|---|---|---|
| 1 | `cond` | `B0` 또는 `Vp` | `cfg.vla_mode` |
| 2 | `task_id` | 0~9 | eval 루프 `task_id` |
| 3 | `episode` | 0~4(초기 상태 번호) | eval 루프 `episode_idx` |
| 4 | `step` | 대기 10스텝을 뺀 0부터의 스텝 번호 = `t - num_steps_wait` | eval 루프 |
| 5 | `step_type` | 항상 `infer`(B0와 V'는 건너뛰기가 없음) | 상수 |
| 6 | `t_total_ms` | 행동 1개를 내는 전체 시간에서 이미지 전처리를 뺀 값 = `step_total_wall_ms − center_crop_wall_ms − preprocess_wall_ms` | 아래 2.2 참고 |
| 7 | `t_vision_ms` | vision backbone + projector 시간(이 실험에는 비전 프루닝 없음) | `vision_ms`(CUDA event) |
| 8 | `t_prune_ms` | **V'**: VLA-Cache 토큰 선택(`select`) 시간. **B0**: 0 | `select_wall_ms`(V'), 0(B0) |
| 9 | `t_generate_ms` | `vla.generate` 전체(비전 + LLM prefill + decode). V'의 모듈 내 attention 통계 계산(`attn_info`)도 여기에 포함 | `generate_ms`(CUDA event, 3.1에서 추가) |
| 10 | `t_fit_ms` | 0(skip 없음) | 상수 |
| 11 | `n_vis_tokens` | 빈 값(이 실험에서는 기록하지 않음) | - |
| 12 | `z_trans` | 빈 값(이 실험에서는 기록하지 않음) | - |
| 13 | `success` | 에피소드 마지막 줄에만 1/0, 나머지 줄은 빈 값 | eval 루프 `done` |
| 14 | `episode_len` | 에피소드 마지막 줄에만, 그 에피소드의 기록된 스텝 수 | 버퍼 길이 |

### 2.2 정의 메모
- **전처리 제외**
  - `get_vla_action` 밖의 `get_libero_image`(LIBERO 리사이즈)는 원래 측정 구간에 들어가지 않는다.
  - 안에서는 `center_crop`(TF crop/resize)과 `preprocess`(processor: 이미지 변환 + 프롬프트 토크나이즈 + H2D)를 뺀다.
  - 토크나이즈는 processor 호출 안에서 분리할 수 없어 함께 빠진다(1 ms 미만으로 추정).
- **CUDA event와 wall 구분**
  - `t_total_ms`, `t_prune_ms`는 CPU 작업이 섞여 있으므로 wall(perf_counter)을 쓴다.
  - `t_vision_ms`, `t_generate_ms`는 CUDA event를 쓴다.
  - 원본 JSONL에는 모든 구간의 event 값과 wall 값이 함께 있으므로 나중에 다시 계산할 수 있다.
- **V'의 `t_prune_ms` 범위**
  - 포함: `find_static_patches` + task-relevant 선택(오버레이 그리기 포함) + 층별 비율(schedule)
  - 각 에피소드 첫 스텝은 재사용이 없어 거의 0이다.
  - V'의 attention 통계 계산(`attn_info`)은 prefill 안에서 일어나므로 `t_generate_ms`에 들어간다(별도 값은 JSONL의 `attn_info_ms`).
- **예외로 끝난 에피소드**: `run_libero_eval.py`는 예외를 잡고 그 에피소드를 끝낸다. 예외가 난 스텝은 기록하지 않고, 그때까지의 마지막 줄에 `success=0`, `episode_len`을 쓴다. 로그의 `Caught exception` 수를 6절에 보고한다.

### 2.3 CSV 헤더 (정확히 이 순서)
```
cond,task_id,episode,step,step_type,t_total_ms,t_vision_ms,t_prune_ms,t_generate_ms,t_fit_ms,n_vis_tokens,z_trans,success,episode_len
```
- 시간은 소수 3자리로 쓴다.
- 파일
  - `src/openvla/experiments/logs/PLAN001-steps-B0.csv`
  - `src/openvla/experiments/logs/PLAN001-steps-Vp.csv`

## 3. 코드 변경 (CSV 기록만 추가)
브랜치: clone한 브랜치(`claude/affectionate-pasteur-6h7oj2`, 이미 merge되었으면 `main`)에서 로컬 브랜치 `exp/plan001-thor`를 만든다.
CLAUDE.md의 불변 규칙을 지킨다. B0/B1/V/V'의 추론 경로는 그대로 둔다.

### 3.1 `src/openvla/experiments/robot/vla_timing.py`
- `attach_stage_timer`에서 `vla.generate`를 감싸 `"generate"` 구간을 잰다.
  - `predict_action`이 `self.generate(...)`를 부르므로 인스턴스 속성으로 덮어쓰면 된다.
  ```python
  generate = vla.generate
  def timed_generate(*args, **kwargs):
      with timer.stage("generate"):
          return generate(*args, **kwargs)
  vla.generate = timed_generate
  ```
- `StageTimer.end_step`이 만든 행을 `self.last_row`에 저장한다. `__init__`에서 `self.last_row = None`으로 초기화한다.
- 순수 함수 `step_metrics_row(timer_row, cond, task_id, episode, step) -> dict`를 추가한다. 2.1의 1~12번 컬럼을 채우고 13, 14번은 빈 값으로 둔다.
  - 없는 키는 0으로 처리한다(예: B0의 `select_wall_ms`, 첫 스텝의 `center_crop` 등).

### 3.2 `src/openvla/experiments/robot/libero/run_libero_eval.py`
- `GenerateConfig`에 `step_csv_path: Optional[str] = None`을 추가한다.
  - 설정되면 timing을 강제로 켠다(`cfg.timing = True`).
- 에피소드마다 행 버퍼를 쓴다.
  - 각 정책 스텝에서 `get_action` 호출 직전에 `stage_timer.last_row = None`으로 둔다.
  - 호출이 정상 반환되면 `step_metrics_row(stage_timer.last_row, vla_mode, task_id, episode_idx, t - cfg.num_steps_wait)`를 버퍼에 넣는다.
  - 에피소드가 끝나면(while 루프 종료 후) 버퍼가 비어 있지 않을 때 마지막 행에 `success=int(done)`, `episode_len=len(buffer)`를 넣는다. 그다음 CSV에 append한다(파일이 없으면 헤더부터 쓴다). 쓴 뒤 flush한다.
  - 버퍼가 비어 있으면(첫 스텝에서 예외) 로그에 경고만 남긴다.
- 기존 로그, 동영상 저장, 타이밍 JSONL 동작은 그대로 둔다.

### 3.3 테스트
- `src/openvla/tests/test_modes_e2e.py::test_stage_timing_rows`에 다음 확인을 추가한다: `generate_ms` 키가 있고, `step_metrics_row`의 결과가 헤더와 같은 키 순서이고, B0에서는 `t_prune_ms == 0`이다.
- `step_metrics_row` 단위 테스트를 1개 추가한다(입력 행 → 기대 dict).
- Thor에서는 테스트를 **CPU로 강제**해서 돌린다. tiny 모델이 CPU에 있는데 `openvla_utils.DEVICE`가 cuda가 되면 불일치가 나기 때문이다.
  ```bash
  cd src/openvla && CUDA_VISIBLE_DEVICES="" python -m pytest tests -q
  ```
  기존 32개를 포함해 모두 통과해야 한다.
- 로컬 커밋까지만 하고 push는 하지 않는다.

## 4. 환경 구성 (Thor)
### 4.1 원칙
- **CUDA torch를 절대 덮어쓰지 않는다.** `pyproject.toml`이 `torch==2.2.0`, `tensorflow==2.15.0` 등을 고정하고 있어서 `pip install -e .`(의존성 포함)을 하면 망가진다. torch를 의존성으로 가진 패키지는 모두 `--no-deps`로 설치한다.
- 설치 단계마다 아래로 확인한다. `True`와 bf16 matmul이 나와야 한다.
  ```bash
  python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available());x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16);print((x@x).float().abs().mean())"
  ```
- 기록할 것(6절 보고용)
  - `cat /etc/nv_tegra_release`, `nvpmodel -q`, `python --version`
  - torch, torchvision, transformers(포크 커밋), timm, tokenizers, tensorflow, numpy, mujoco, robosuite 버전

### 4.2 순서
1. **기존 환경 확인**: Thor에서 CUDA가 되는 torch 환경이 이미 있는지 확인하고 사용자에게 알린다(예: JetPack 제공, 사용자 venv).
   - 있으면 그 Python 버전에 맞춰 진행한다.
   - 없으면 Python 3.10 venv(`uv venv -p 3.10`)를 만들고, Thor(SBSA, CUDA 13)를 지원하는 aarch64 CUDA torch/torchvision을 설치한다(예: PyTorch cu130 aarch64 인덱스).
   - 어느 쪽도 안 되면 중단하고 보고한다.
2. **clone**
   ```bash
   git clone https://github.com/bluebenny/vla-cache.git && cd vla-cache
   git checkout claude/affectionate-pasteur-6h7oj2   # PR #1; merge되었으면 main
   git log --oneline -1                               # 커밋 해시를 기록
   ```
3. **포크 transformers와 모델 의존성** (`--no-deps`)
   ```bash
   pip install --no-deps "transformers @ git+https://github.com/siyuhsu/transformers.git@2302fce58afa3a4f8461625b1394f9e9c8a7f1ea"
   pip install --no-deps timm==0.9.10 accelerate
   pip install "tokenizers==0.21.1" "huggingface_hub<1" safetensors regex requests tqdm pyyaml packaging filelock psutil sentencepiece "numpy<2"
   ```
   - `modeling_prismatic.py`는 timm이 0.9.10/11/12/16이 아니면 예외를 낸다.
   - `low_cpu_mem_usage=True`에는 accelerate가 필요하다.
4. **TF 계열** (eval 경로가 import함)
   - import 체인: `prismatic/__init__` → `prismatic.vla.datasets.rlds` → `dlimp`, `tensorflow_datasets`. `libero_utils`와 `process_image`는 `tensorflow`를 쓴다.
   - `tensorflow`(CPU면 충분하다. Python 3.10이면 2.15.x)를 설치한다.
   - 다음은 `--no-deps`로 설치한다: `tensorflow_datasets==4.9.3`, `tensorflow_graphics==2021.12.3`, `git+https://github.com/moojink/dlimp_openvla`, `draccus==0.8.0`
   - 그 밖에 `wandb`(run_libero_eval이 import함), `opencv-python-headless`, `scikit-image==0.25.0`, `imageio[ffmpeg]`, `jsonlines`, `einops`, `rich`
   - TF를 설치할 수 없으면 **중단하고 보고한다**. TF 리사이즈와 crop을 다른 라이브러리로 바꾸면 전처리 수치가 공개 코드와 달라진다.
5. **openvla 패키지**: `cd src/openvla && pip install --no-deps -e .`
   - `experiments`가 namespace 패키지로 잡히게 하기 위해서다. 안 되면 `export PYTHONPATH=$PWD`.
6. **LIBERO**
   ```bash
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
   pip install --no-deps -e ~/LIBERO        # LIBERO requirements는 transformers==4.21 등을 고정하므로 쓰지 않는다
   pip install robosuite==1.4.1 bddl easydict cloudpickle gym mujoco numba scipy termcolor  # torch가 바뀌지 않았는지 확인
   ```
   - LIBERO는 처음 import할 때 경로 설정을 **대화형으로 묻는다**. 미리 응답한다: `printf 'N\n' | python -c "import libero.libero"`. 또는 `~/.libero/config.yaml`을 만든다.
   - 헤드리스 렌더링: `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`
7. **import 스모크 테스트**: `cd src/openvla && python -c "import experiments.robot.libero.run_libero_eval"`
   - 빠진 모듈은 하나씩 `--no-deps`로 설치하고, 설치할 때마다 torch를 확인한다.
8. **체크포인트**: `python vla_cache_scripts/download_model_local.py --model_id openvla/openvla-7b-finetuned-libero-spatial`
   - `checkpoints/` 아래에 약 15 GB가 받아진다. 반드시 `src/openvla`에서 실행한다. `get_vla`가 `./prismatic/`의 `modeling_prismatic.py`를 체크포인트 폴더로 동기화하기 때문이다.

## 5. 실행
### 5.1 스모크 런 (모드별 1 task × 1 에피소드)
CSV와 타이밍이 기록되는지 확인하는 용도라 짧게 돌린다. `num_trials_per_task=1`로 실행하고, task 0에서 끝나면 중단해도 된다.
```bash
cd src/openvla
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial --num_trials_per_task 1 \
  --vla_mode Vp --timing True --step_csv_path experiments/logs/SMOKE-steps-Vp.csv --run_id_note smoke
```
확인할 것
- 로드 로그
  - `[*] VLA mode: Vp | LLM attn_implementation: sdpa (VPrimeLlamaSdpaAttention)`
  - B0이면 `sdpa (LlamaSdpaAttention)`
- CSV 헤더가 2.3과 같은지, `step`이 0부터 연속인지, 마지막 줄에만 `success`/`episode_len`이 있는지
- V'의 2번째 스텝부터 `t_prune_ms > 0`인지, JSONL에 `reuse: true`가 있는지
- `Caught exception`이 없는지
- 스모크 CSV와 JSONL은 본 실험 결과와 섞이지 않게 둔다(파일명 `SMOKE-`).

### 5.2 본 실험 (모드별 10 task × 5 에피소드, 프로세스를 분리해 순서대로)
- 오래 걸리므로 `nohup`으로 백그라운드에서 돌리고 로그를 주기적으로 확인한다.
- B0가 끝난 뒤 V'를 돌린다. 같은 시드(기본 7)와 같은 초기 상태를 쓴다.
```bash
cd src/openvla
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
nohup bash -c '
for MODE in B0 Vp; do
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial --num_trials_per_task 5 \
    --vla_mode $MODE --timing True \
    --step_csv_path experiments/logs/PLAN001-steps-$MODE.csv \
    --run_id_note PLAN001 > experiments/logs/PLAN001-$MODE.out 2>&1
done' > experiments/logs/PLAN001-all.out 2>&1 &
```
- 진행 확인: `tail -n 5 experiments/logs/PLAN001-*.out`, `wc -l experiments/logs/PLAN001-steps-*.csv`
- 실행 중에는 다른 GPU 작업을 하지 않는다. 전원 모드(`nvpmodel`, `jetson_clocks`)는 바꾸지 않고, 시작 전 상태를 기록만 한다(바꾸려면 사용자 승인이 필요하다).

## 6. 검증과 보고
### 6.1 CSV 검증 (모드별)
- 에피소드 50개: `(task_id, episode)` 조합이 0~9 × 0~4로 모두 있어야 한다.
- 각 에피소드에서 `step`이 0..`episode_len`−1로 연속이고, 마지막 줄에만 `success`, `episode_len`이 있어야 한다.
- 값 검사
  - `step_type`은 모두 `infer`이고 `t_fit_ms`는 모두 0이다.
  - B0의 `t_prune_ms`는 모두 0이다.
  - 모든 줄에서 `t_vision_ms ≤ t_generate_ms ≤ t_total_ms`이다(오차 수 ms 이내).
- `success` 합계가 eval 로그의 `# successes`와 일치해야 한다.

### 6.2 요약 (`experiments/logs/PLAN001-summary.md`로 저장하고 사용자에게 보고)
- 환경: 4.1에서 기록한 항목과 저장소 커밋 해시
- 모드별 성공률(전체, task별)과 `Caught exception` 수
- 모드별 시간 통계(`infer` 줄; 평균, 중앙값, p90)
  - 대상: `t_total_ms`, `t_generate_ms`, `t_vision_ms`, `t_prune_ms`
  - 전체 통계에 더해, 에피소드 첫 스텝(`step==0`, V'에서 재사용 없음)과 나머지 스텝을 따로 계산한다.
- V'만: JSONL 기준 `attn_info_ms`, `prefill_ms`(재사용/비재사용), `select_static/task/schedule_wall_ms` 평균
- B0 대비 V'의 `t_total_ms` 비율(재사용 스텝 기준). REPORT001 6절의 주의점(재사용 prefill의 0·1층 커널 차이)을 함께 적는다.

### 6.3 보존
- 산출물 위치: `experiments/logs/`의 CSV, JSONL, `.txt`, `.out`, summary, 그리고 `rollouts/`의 mp4
- 위치를 보고한다. git에 올릴지, push할지는 사용자에게 묻는다.

## 7. 중단 조건 (사용자에게 보고하고 지시를 기다린다)
- CUDA torch가 동작하지 않거나, 설치 과정에서 torch가 바뀌었을 때
- TF, LIBERO, MuJoCo EGL 렌더링을 설치하거나 실행할 수 없을 때(대체 구현으로 우회하지 않는다)
- 스모크 런에서 로드 로그나 CSV가 5.1의 기대와 다를 때
- 본 실험 중 `Caught exception`이 반복될 때(예: V'에서 재사용할 패치가 0개일 때의 TypeError. 공개 코드와 같은 알려진 동작이지만 빈도를 보고한다)
- 테스트(3.3)가 실패할 때
