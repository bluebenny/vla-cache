# CLAUDE.md

이 저장소는 VLA-Cache(OpenVLA, OpenVLA-OFT)의 공개 코드다. 현재 작업은 `src/openvla`(OpenVLA)에
B0와 V' 실행 모드를 추가하고, B1/V와 같은 조건에서 비교할 수 있게 하는 것이다. `src/openvla-oft`는 건드리지 않는다.

## 실행 모드

| 모드 | 정의 | LLM attention | generate `output_attentions` |
|---|---|---|---|
| **B0** | VLA-Cache 끔, attention 출력 끔 | SDPA (`attn_implementation="sdpa"` 명시) | False |
| **B1** | 공개 baseline (`use_vla_cache=False`) | SDPA 클래스지만 eager로 폴백 | True |
| **V** | 공개 VLA-Cache 그대로 (`use_vla_cache=True`) | SDPA 클래스지만 eager로 폴백 | True |
| **V' (`Vp`)** | B0와 같은 SDPA 경로에서 V와 같은 재사용 결정 | SDPA (명시) + 모듈 내 통계 | False |

- 목표: B1→V와 B0→V'가 **같은 조건 변화(VLA-Cache만 켜짐)**가 되게 한다.
- 선택 플래그: `--vla_mode {B0,B1,V,Vp}`. 지정하지 않으면 기존대로 `--use_vla_cache True|False`가 V/B1을 정한다.
- B1/V의 기존 동작(로딩 인자, generate 인자, 선택 로직, 출력)은 **바꾸지 않는다**.

## V' 요구사항 (원문 요약)
1. 선택 로직은 공개 코드와 동일하게 유지한다.
   - `find_static_patches`: 호출부 `top_k=130`, 기준 0.996
   - `task_relevant_selection`: 15번 층, text 위치 257~291 × vision 위치 1~256, head 평균, top-120
   - `get_layer_mask_schedule`: 전 층 entropy, min-max 정규화, `growth_factor=0.55`
2. generate의 `output_attentions`를 끈다(prefill, decode 모두).
   - 필요한 값(15번 층 text→vision 블록, 층별 entropy 스칼라)은 prefill 중 **attention 모듈 안에서** 계산한다.
   - 전체 행렬은 반환하지 않는다.
3. 재사용 적용(`reusable_patches`, `proportion_attn_var`)과 KV cache crop은 V와 동일하게 SDPA 경로에서 동작해야 한다.
4. select 구현(오버레이 그리기 포함)은 V와 동일하게 둔다. 최적화는 **별도 옵션으로만** 넣는다.
5. 플래그로 B0/B1/V/V'를 선택한다. 기존 B1/V 동작은 바꾸지 않는다.
6. 구간별 시간(비전, 프리필, 디코드, select, attention 정보 계산)을 CUDA event로 기록하는 옵션을 둔다.
7. 고정 입력 동일성 검증 스크립트를 둔다. 같은 프레임 쌍과 직전 attention에서 V와 V'의 재사용 인덱스와 층별 비율을 비교한다.

## 수정된 LLaMA 코드는 이 저장소에 없다
`src/openvla/pyproject.toml`이 포크를 고정한다.
- 포크: `transformers @ git+https://github.com/siyuhsu/transformers.git@vla-cache-openvla`
- 검증한 커밋: `2302fce58afa3a4f8461625b1394f9e9c8a7f1ea`(4.47.0 트리)
- `modeling_llama.py`는 4.40 스타일 코드다.

포크에서 VLA-Cache와 관련된 곳:
- `models/llama/modeling_llama.py`
  - `LlamaModel.__init__`: `pruning_loc = [2, 6, 9, 11]`
  - `LlamaModel.forward`
    - `config.reusable_patches`, `config.proportion_attn_var`를 읽는다.
    - pruning 층에서 hidden/position/`cache_position`을 부분집합으로 줄이고 `causal_mask[..., mask, :]`로 행을 자른다.
    - 재사용 prefill에서는 `past_seen_tokens=0`으로 강제한다.
    - `all_self_attns += (cache_position,)`를 **항상** 붙인다. V의 `attentions[-1]`이 이 값이다.
    - 모든 LM forward 시작에서 `torch.cuda.synchronize()`를 호출하고 prefill latency를 print한다. 모든 모드에 동일하게 적용된다. CPU에서는 stub이 필요하다.
- `cache_utils.py` `DynamicCache`
  - `cache_position` 속성이 있다. `modeling_prismatic.prepare_inputs_for_generation`이 사용한다.
  - `update()`는 부분 `cache_position`이 들어오면 `index_copy_`로 그 위치만 덮어쓴다. 재사용 위치에는 이전 프레임 K/V가 남는다.
  - `crop()`
  - `LlamaModel.forward`는 전달받은 cache를 `DynamicCache.from_legacy_cache`로 **새 객체로 다시 감싼다**.
    - 텐서는 공유하므로 `index_copy_`는 원래 cache의 텐서도 덮어쓴다.
    - 갱신된 cache는 `outputs.past_key_values`다. `generate`와 `predict_action`이 이것을 이어 쓴다.
    - 테스트에서 cache를 재사용하려면 반환된 cache를 `deepcopy`해서 넘긴다.

### 재사용 로직과 attention 구현의 관계 (중요)
- 재사용 메커니즘 자체는 `LlamaModel.forward`와 `DynamicCache`에 있다. 어텐션 클래스(eager, SDPA)는 수정되지 않았다.
- 결정 입력(전체 attention 확률)은 `output_attentions=True`일 때만 나온다. 이때 `LlamaSdpaAttention`이 eager로 폴백한다.
- 기본 attention 구현은 sdpa로 해석된다. `PrismaticPreTrainedModel._supports_sdpa`가 property여서 클래스 수준에서 truthy로 평가되기 때문이다.
  - 그래서 B1/V도 SDPA 클래스이고 eager로 폴백한다(소스 추론). 로드할 때 `get_vla`가 실제 값을 출력한다.
- **SDPA 함정**: SDPA + `output_attentions=False`이면 재사용 prefill에서 `_update_causal_mask`가 `None`을 반환한다.
  - `_ignore_causal_mask_sdpa` 때문이다: attention_mask가 전부 1이고, q_len과 kv_len이 같다.
  - 그러면 행 자르기가 건너뛰어지고 SDPA가 `is_causal=True`로 동작한다. q_len < kv_len이면 PyTorch는 **top-left 정렬**을 쓴다.
  - 결과적으로 pruning 후 남은 query의 causal mask가 틀린다. 에러는 나지 않는다.
  - V'는 재사용 prefill에서 원본 `_update_causal_mask`를 `output_attentions=True`로 불러 mask를 실체화한다. 이 mask는 V의 eager mask와 비트 단위로 같다.
- 공개 코드 엣지 케이스: 남는 패치가 0개이면 `reusable_patches=None`인데 비율은 설정된다. 그러면 `len(None)`에서 TypeError가 나고, eval 루프가 에피소드를 끝낸다. V'도 결정 동일성을 위해 똑같이 둔다.

## 불변 규칙
- **B1/V 경로를 바꾸지 않는다.** 로딩 인자, generate 인자, V 선택 블록이 대상이다. 공유 헬퍼로 쪼갤 때는 원본과의 비트 동일성 테스트가 있어야 한다(`tests/test_vla_cache_utils_refactor.py`).
- V' 통계는 eager와 **같은 연산 순서와 dtype**을 따른다: `matmul/sqrt(d) + mask`, `softmax(fp32).to(bf16)`. 그다음 V와 같은 헬퍼(`layer_attention_entropy`, `text_vision_relation`)를 쓴다.
- `VPrimeLlamaSdpaAttention.forward`는 포크의 `LlamaSdpaAttention.forward` 복사본이다.
  - 포크를 올리면 다시 맞춰야 한다. `install_vprime`이 소스 해시를 비교해 다르면 경고한다.
- 15번 층의 `cache_position`이 V의 `attentions[-1]`과 같으려면 `max(pruning_loc) < 15`여야 한다. `install_vprime`이 이를 확인한다.
- select/통계 계산의 최적화는 기본값을 바꾸지 말고 별도 플래그로만 넣는다.
- `modeling_prismatic.py`는 `trust_remote_code`로 체크포인트 폴더의 사본이 로드된다(`check_model_logic_mismatch`가 동기화).
  - 이 파일에 새 로컬 모듈 import를 넣지 않는다. V' 코드는 `experiments/robot/`에 두고 로드된 인스턴스에 설치한다.

## 파일 맵 (`src/openvla`)
- `experiments/robot/openvla_utils.py`
  - `get_vla`: 모드별 attention 구현과 V' 설치
  - `get_vla_action`: 모드 분기, V/V' 선택, 타이밍
- `experiments/robot/vla_cache_utils.py`: 공개 선택 로직과 공유 헬퍼
- `experiments/robot/vla_cache_sdpa.py`: 모드 상수, `resolve_vla_mode`, `AttnStats`/`AttnStatsCollector`, `VPrimeLlamaSdpaAttention`, `install_vprime`
- `experiments/robot/vla_timing.py`: `StageTimer`(CUDA event, CPU에서는 perf_counter), `attach_stage_timer`(hook 설치)
- `experiments/robot/tiny_openvla.py`: CPU 테스트와 검증 스크립트용 작은 무작위 OpenVLA, 가짜 processor, 경량 import stub
- `experiments/robot/libero/run_libero_eval.py`: `--vla_mode`, `--timing`, `--timing_log_path`
- `prismatic/extern/hf/modeling_prismatic.py`: `predict_action`이 `attentions`가 None(B0/V')일 때도 동작
- `vla_cache_scripts/verify_vprime_equivalence.py`: 고정 입력 동일성 검증(검사 A/B/C)
- `vla_cache_scripts/run_openvla_{b0,vprime}.sh`: 실행 예시
- `tests/`: CPU 단위 테스트

## 타이밍 로그 (`--timing True`)
스텝마다 JSONL 한 줄을 쓴다. `<name>_ms`는 CUDA event 기준(CPU에서는 wall), `<name>_wall_ms`는 perf_counter 기준이다.
- `vision`: vision backbone + projector
- `prefill`: LM forward, q_len>1
- `decode`: LM forward, q_len==1의 합. `decode_n`은 스텝 수
- `select`: 하위 구간 `select_static`, `select_task`, `select_schedule`. CPU 위주이므로 wall도 같이 본다.
- `attn_info`: V' 모듈 내 통계 계산의 층별 합. **prefill 안에 중첩된다.**
- `preprocess`, `predict_total`, `step_total`과 메타(`mode`, `reuse`, `n_reusable`, `task_id`, `episode`, `t`)

V에서는 attention 정보 계산이 eager attention과 select(`task_relevant_selection`, `get_layer_mask_schedule`) 안에 섞여 있어 따로 떼어낼 수 없다.

## 환경 제약과 테스트
- 이 VM에는 GPU도 Jetson Thor도 없다. 실제 측정(Thor)은 사용자가 한다. 여기서는 CPU에서 작은 무작위 LLaMA로 단위 테스트만 한다.
- CPU 테스트 환경: Python 3.10 venv. `download.pytorch.org`가 차단되어 PyPI의 torch 2.2.0을 쓴다.
  ```bash
  uv venv -p 3.10 <venv> && uv pip install -p <venv>/bin/python -r src/openvla/tests/requirements-cpu.txt
  cd src/openvla && <venv>/bin/python -m pytest tests -q
  <venv>/bin/python vla_cache_scripts/verify_vprime_equivalence.py --tiny_random
  ```
- 검증 상태: 이 VM에서 torch 2.14.1(CPU 실행)과 포크 transformers(`2302fce`)로 `pytest tests` 32개가 통과했다. `--tiny_random` 검증도 A/B PASS다.
  핀 버전(torch 2.2.0)은 PyPI CUDA 휠의 다운로드가 느려 별도로 확인한다.
- 테스트는 `prismatic` 패키지 초기화(TF, draccus 등 무거운 import)와 tensorflow, CPU에서의 `torch.cuda.Event/synchronize`를 stub으로 우회한다(`experiments/robot/tiny_openvla.py:install_light_import_stubs`).
- 브랜치 작업 후에는 커밋하고 push한다. PR은 요청이 있을 때만 만든다.
