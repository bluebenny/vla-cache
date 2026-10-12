# REPORT001: V와 V', B0와 V'의 차이

- 기준 코드: 브랜치 `claude/affectionate-pasteur-6h7oj2`, 커밋 `8a20091` (bluebenny/vla-cache#1)
- 기준 포크: `siyuhsu/transformers@2302fce`(4.47.0 트리). 아래에서 `[fork] 파일:줄`로 표기한다.
- 이 문서는 코드 사실과 CPU tiny 모델 확인 결과만 담는다. Thor 성능 수치는 측정 전이다.
- 경로는 저장소 루트 기준이다. `ou` = `src/openvla/experiments/robot/openvla_utils.py`, `sdpa` = `src/openvla/experiments/robot/vla_cache_sdpa.py`, `vcu` = `src/openvla/experiments/robot/vla_cache_utils.py`.

## 1. 요약

| 항목 | B0 | B1 | V | V' (`Vp`) |
|---|---|---|---|---|
| 로딩 `attn_implementation` | `"sdpa"` 명시 | 미지정(sdpa로 해석, 추론) | 미지정(sdpa로 해석, 추론) | `"sdpa"` 명시 |
| `install_vprime` | 없음 | 없음 | 없음 | 있음 |
| generate `output_attentions` | False | True | True | False |
| 실제 attention 계산 | SDPA | eager 수식(폴백) | eager 수식(폴백) | SDPA |
| prefill causal mask | None(`is_causal`) | 항상 실체화 | 항상 실체화 | 재사용 없음: None / 재사용: 실체화(강제) |
| decode causal mask | None | 실체화(1×kv) | 실체화(1×kv) | None |
| KV cache | 매 스텝 새로 만듦 | 매 스텝 새로 만듦 | 직전 cache 재사용 | 직전 cache 재사용 |
| select 단계 | 없음 | 없음 | 있음(전체 행렬 축약 포함) | 있음(축약된 값만 사용) |
| 재사용 결정 입력 | - | - | 직전 prefill의 전체 attention | 직전 prefill 중 모듈 안에서 계산한 통계 |
| `last_caches` | cache, `attentions=None` | cache, attentions | cache, attentions | cache, `attentions=None`, `attn_stats` |

핵심 결론:
1. **V와 V'의 재사용 알고리즘은 같다.** 선택 코드, 재사용 적용, KV crop이 같은 코드다. 차이는 attention 계산 방식(eager↔SDPA)과 통계를 어디서 언제 계산하느냐뿐이다.
   - 같은 Q/K가 들어가면 V'의 통계와 결정은 V와 비트 단위로 같다(검사 A/B).
   - 다만 실주행에서는 SDPA와 eager의 수치 차이 때문에 결정이 조금 달라질 수 있다. tiny 모델에서 실제로 재현했다(2.4절).
2. **B0와 V'는 에피소드 첫 스텝의 출력이 같다.** 두 번째 스텝부터는 cache 재사용, select, pruning, 명시 mask, 통계 계산이 V'에만 있다.
3. **B1→V와 B0→V'는 "VLA-Cache만 켜짐"이라는 점은 같지만, 비용이 생기는 방식이 다르다(4절).**
   - B0→V'에는 통계용 QKᵀ·softmax 계산(`attn_info`)이 새로 생긴다.
   - 재사용 prefill에서 SDPA 커널이 `is_causal` 경로에서 명시 mask 경로로 바뀐다.

## 2. V vs V'

### 2.1 같은 것
- **선택 코드**: 두 모드 모두 다음을 같은 인자로 부른다(`ou:412-433`(V'), `ou:435-455`(V)).
  - `find_static_patches(image, prev_image, top_k=130)`(기준 0.996)
  - task-relevant 선택(top-120, 오버레이 그리기 포함)
  - schedule(`growth_factor=0.55`)
  - V는 `task_relevant_selection`/`get_layer_mask_schedule`을 부르고, 이 둘은 V'가 부르는 `task_relevant_selection_from_scores`/`layer_mask_schedule_from_entropies`를 내부에서 호출한다(`vcu:179-198`, `vcu:18-31`). 리팩터 결과는 공개 원본 사본과 비트 동일하다(`src/openvla/tests/test_vla_cache_utils_refactor.py`).
- **재사용 적용**: 포크의 `LlamaModel.forward`를 그대로 쓴다.
  - pruning 층은 `[2, 6, 9, 11]`이다(`[fork] modeling_llama.py:935`). 이 층들에서 hidden, `position_ids`, `cache_position`을 줄이고 mask 행을 자른다(`[fork] modeling_llama.py:1022-1047`).
  - 재사용 prefill에서는 `past_seen_tokens=0`이다(`[fork] modeling_llama.py:994`).
- **KV cache**
  - 부분 위치는 `DynamicCache.update`의 `index_copy_`로 덮어쓴다(`[fork] cache_utils.py:465-466`).
  - 생성이 끝나면 `predict_action`이 프롬프트 길이로 crop한다(`src/openvla/prismatic/extern/hf/modeling_prismatic.py:520-527`).
  - forward가 cache를 `from_legacy_cache`로 다시 감싸는 동작(`[fork] modeling_llama.py:990`)도 같다.
- **재사용 prefill의 causal mask 값**: V'는 원본 `_update_causal_mask`를 `output_attentions=True`로 불러 만든다(`sdpa:244-252`). 그래서 V의 mask와 값이 같다.
- **포크의 계측 코드**: 모든 LM forward의 `torch.cuda.synchronize()`(`[fork] modeling_llama.py:1017`)와 prefill latency print(`:1101`)가 양쪽에 똑같이 있다.
- **엣지 케이스**: 남는 패치가 0개이면 `reusable_patches=None`인 채로 비율만 설정되어 TypeError가 난다. 양쪽 동일하다(`ou:429-431`, `ou:451-453`).

### 2.2 다른 것 (실행 순서대로)

| 단계 | V | V' | 근거 |
|---|---|---|---|
| 로딩 | `from_pretrained`에 attention 인자 없음 | `attn_implementation="sdpa"`로 로드한 뒤 `install_vprime`(층마다 `VPrimeLlamaSdpaAttention`으로 클래스 교체, `_update_causal_mask` 래퍼, collector 부착) | `ou:269-290`, `sdpa:259-295` |
| 선택에 쓰는 입력 | `last_caches["attentions"]`(전체 행렬 튜플) | `last_caches["attn_stats"]`(`AttnStats`) | `ou:398-399` |
| select: text→vision 점수 | `token_attention_merge`: 15층 `[1,H,q,N]`을 fp32로 변환하고 head 평균한 뒤 text 행 평균, `.cpu()` | 이미 계산된 `[256]` fp32 벡터를 그대로 사용. `.cpu()`는 `get_top_attention_patches`에서 일어남 | `vcu:115-138`, `vcu:144` |
| select: schedule | 32층 각각 `[q,N]` head 평균, 정규화, entropy를 계산한 뒤 schedule | 32개 스칼라로 바로 schedule | `vcu:18-64` |
| generate 인자 | `output_attentions=True` | `output_attentions=False` | `ou:485-486` |
| prefill attention | `LlamaSdpaAttention`이 경고를 남기고 `LlamaAttention.forward`(eager)로 폴백: `matmul/√d + mask`, `softmax(fp32).to(bf16)`, `@V` | `torch.nn.functional.scaled_dot_product_attention` | `[fork] modeling_llama.py:624`, `:371-378`, `sdpa:227-234` |
| prefill mask | 재사용 여부와 무관하게 항상 실체화 | 재사용 없음: None, `is_causal=True`. 재사용: 래퍼가 실체화 | `[fork] modeling_llama.py:1142`, `sdpa:249-251` |
| prefill 중 추가 계산 | 없음(attention 확률은 eager 계산의 부산물) | 층마다 `eager_attention_weights`(QKᵀ, mask, fp32 softmax, bf16 변환), entropy 축약, 15층 text→vision 축약 = `attn_info` | `sdpa:97-103`, `sdpa:123-133`, `sdpa:206` |
| decode 6스텝 | eager 수식, 1×kv mask 실체화 | SDPA, mask None | 같음 |
| 반환 | `results.attentions`: prefill 1개와 decode 6개 스텝, 각 32층 + `cache_position` | `results.attentions = None` | `modeling_prismatic.py:521` |
| 다음 스텝까지 보관 | `attentions[0]`: 32×`[1,32,q_l,N]` bf16 + `cache_position` | bf16 스칼라 32개 + fp32 `[256]` | `modeling_prismatic.py:527`, `ou:490` |
| 15층 text 행 위치 출처 | 최종 `cache_position`(`attentions[-1]`) | 15층의 `cache_position` | 같음: `max(pruning_loc)=11 < 15`, `install_vprime`이 검사 |
| print | `>> VLA-Cache inference mode` | `>> VLA-Cache inference mode (V': SDPA)` | `ou:414`, `ou:436` |
| 타이밍 로그 | `attn_info` 없음. `select_task`/`select_schedule`에 전체 행렬 축약이 포함됨 | `attn_info` 있음(prefill에 중첩). select 하위 구간은 가벼움 | `src/openvla/experiments/robot/vla_timing.py` |

### 2.3 크기 (OpenVLA-7B 기준 계산식)
- `N = 256 + L`. L은 BOS와 29871을 포함한 프롬프트 토큰 수다. 예: L=36이면 N=292다.
- **V가 보관하는 `attentions[0]`**: 32층 × 32 head × q × N × 2바이트.
  - pruning이 없으면 32×32×292×292×2 ≈ **166.5 MiB**다. 재사용 스텝에서는 2층부터 q가 줄어든다.
  - decode 스텝의 attention은 6 × 32 × 32 × 1 × (N+k) × 2 ≈ 3.6 MB이고, `results`와 함께 해제된다.
- **V'가 보관하는 통계**: 32 × 2바이트 + 256 × 4바이트 = **1,088바이트**.
- V'의 층별 통계 계산은 `[1,32,q,N]` fp32 softmax 임시 텐서를 만든다. 이것은 V의 eager attention이 만드는 임시 텐서와 같은 크기이고, 층마다 해제된다.
- tiny 모델(4 head)로 확인한 형상: V `attentions[0]`는 33개(32층 + `cache_position` int64 `[292]`)다. 층 텐서는 `[1,4,292,292]` bf16이고, 재사용 스텝 2층부터는 `[1,4,236,292]`다. V' `attn_stats`는 bf16 스칼라 32개와 fp32 `[256]`이다.

### 2.4 수치 차이와 그 영향 (tiny 모델, CPU, torch 2.2.0, 확인 완료)
같은 프레임 쌍(합성), 같은 모델로 V, V'(SDPA), V'(`eager_debug`)를 비교했다.

| 프롬프트 | 재사용 패치 수 V / V' | Jaccard | 2층 비율 V / V' | 6, 9, 11층 비율 | entropy 최대 차이 | 15층 점수 최대 차이 | 액션 |
|---|---|---|---|---|---|---|---|
| `"x"` | 68 / 67 | 0.985 | 0.832 / 0.918 | 같음 | 0.03125 | 1.7e-4 | 같음 |
| `"pick up the black bowl ..."` | 69 / 69 | 1.0 | 0.918 / 0.918 | 같음 | 0 | 1.6e-4 | 같음 |

- `"x"`의 경우 직전 prefill에서 SDPA와 eager의 미세한 수치 차이가 생겼다.
  - 한 층의 bf16 entropy가 한 단계(0.03125) 달라졌다.
  - min-max 정규화가 이 차이를 키워 2층 비율이 0.832에서 0.918로 바뀌었다.
  - 그 결과 2층에서 제거된 토큰이 56개 대 61개가 되었고, 층별 query 수는 236 대 231이 되었다.
  - 즉 **bf16 entropy 양자화와 min-max 정규화 때문에 결정이 작은 수치 차이에 민감하다.**
- 두 프롬프트 모두 `eager_debug`(V'의 통계 경로는 그대로 두고 attention 출력만 eager로 계산)에서는 결정과 액션이 V와 정확히 같았다.
  - 따라서 이 차이는 구현 차이가 아니라 attention 커널의 수치 차이에서 온다.
- tiny 모델과 CPU의 수치는 GPU bf16 커널을 대표하지 않는다. 실제 크기는 Thor에서 검증 스크립트의 검사 C로 확인해야 한다.

## 3. B0 vs V'

### 3.1 같은 것
- `attn_implementation="sdpa"` 명시 로딩(`ou:270`), `output_attentions=False`(`ou:486`)
- decode 커널: SDPA, mask None, `q_len=1`
- 포크 forward의 synchronize/print, `predict_action`의 crop 호출(B0의 crop된 cache는 다음 스텝에서 버려짐)
- **에피소드 첫 스텝**: V'는 cache와 직전 통계가 없어 재사용하지 않는다. 층마다 통계만 추가로 계산하고, attention 출력 경로(mask None, `is_causal`)는 B0와 같다.
  - tiny 모델에서 B0와 V'의 첫 스텝 prefill logits와 액션이 **비트 단위로 같음**을 확인했다.
  - collector를 끈 상태에서 복사한 forward가 원본과 같다는 것은 `test_collector_off_keeps_original_sdpa_outputs`가 확인한다.

### 3.2 다른 것

| 항목 | B0 | V' | 근거 |
|---|---|---|---|
| 모델 패치 | 원본 `LlamaSdpaAttention` | `VPrimeLlamaSdpaAttention`(collector 확인 분기 추가), LM forward마다 `_update_causal_mask` 래퍼 Python 호출 1회 | `sdpa:144-241`, `sdpa:293` |
| KV cache | 매 스텝 `prompt_cache=None`으로 두고 새 `DynamicCache()` 생성. prefill이 전 토큰을 전 층에서 계산 | 2번째 스텝부터 직전 cache(crop됨)를 재사용. 2층 이후 재사용 위치의 K/V는 이전 프레임 값(누적 가능) | `ou:457-463` |
| select 단계 | 없음 | `find_static_patches`(CPU numpy 코사인 유사도), top-k와 PIL 오버레이, schedule(`.tolist()` 동기화 포함), mask 인덱스 `torch.tensor(..., device)` H2D | `ou:412-433` |
| prefill 토큰 수 | 모든 층에서 N | 0, 1층은 N. 2층부터 `N - (선택된 재사용 패치 수)`. 층별 비율에 따라 2, 6, 9, 11층에서 더 줄 수 있음 | `[fork] modeling_llama.py:1022-1047` |
| prefill mask와 SDPA 경로 | 모든 스텝, 모든 층에서 mask None, `is_causal=True` | 재사용 스텝에서는 **0층부터 31층까지 모두** 실체화된 mask `[1,1,q_l,N]`를 사용. 첫 스텝은 B0와 같음 | 아래 확인 결과 |
| prefill 추가 계산 | 없음 | 매 prefill(첫 스텝 포함), 층마다 통계 계산(`attn_info`) | `sdpa:206` |
| decode | 새 cache의 KV를 사용 | 재사용 위치에 이전 프레임 K/V가 섞인 cache를 사용. 커널은 같지만 입력이 달라 액션이 달라질 수 있음(VLA-Cache의 근사) | |
| `last_caches` | `{past_key_values, attentions: None}` | 위에 더해 `attn_stats` | `ou:488-490` |
| 예외 가능성 | 없음 | 남는 패치가 0개일 때 TypeError(공개 V와 동일한 동작) | 2.1절 |
| print | `>> VLA-Cache disabled` | `>> VLA-Cache inference mode (V': SDPA)` | `ou:458`, `ou:414` |
| 타이밍 로그 | select*, attn_info 없음, `reuse=False`, `n_reusable=0` | select*(2번째 스텝부터 하위 구간), attn_info, `reuse`, `n_reusable` | `ou:494-499` |

**SDPA에 전달되는 mask 확인 결과** (tiny 모델; 층 0/1/2/6/9/11/15/31을 기록; 값은 `(mask 형상, query 수)`):
- B0 1·2번째 스텝: 모든 층이 `(None, 292)`
- V' 1번째 스텝: 모든 층이 `(None, 292)`로 B0와 같다.
- V' 2번째 스텝(재사용)
  - 0, 1층: `([1,1,292,292], 292)`
  - 2층부터: `([1,1,231,292], 231)`
- 참고로 V 1번째 스텝은 모든 층이 `([1,1,292,292], 292)`이다. 2번째 스텝의 2층부터는 `([1,1,236,292], 236)`이다.

CUDA SDPA는 임의 mask가 있으면 flash 백엔드를 쓸 수 없다. 그래서 V'의 재사용 prefill은 B0와 다른 백엔드(mem-efficient 또는 math)로 갈 가능성이 높다. 실제로 어떤 백엔드가 선택되는지는 Thor에서 미확인이다.

## 4. 비교 설계 관점: B1→V와 B0→V'의 비대칭
두 쌍 모두 "VLA-Cache 켬"이 유일한 설정 변화다. 다만 켰을 때 추가되거나 바뀌는 연산의 종류가 다르다.

| 변화 요소 | B1→V | B0→V' |
|---|---|---|
| attention 커널 | 양쪽 모두 eager. 재사용 시 행 수만 줄어듦 | 재사용 prefill에서 SDPA `is_causal`이 SDPA 명시 mask로 바뀜(백엔드 변경 가능) |
| 결정 통계의 원천 | eager attention 확률(어차피 계산되는 부산물) | 통계 전용 QKᵀ와 softmax를 **새로** 계산(`attn_info`) |
| 통계 축약 위치 | 다음 스텝 select(`select_task`, `select_schedule`) | 현재 prefill 안(`attn_info`) |
| attention 반환과 보관 비용 | B1, V 모두 부담(양쪽 동일) | B0, V' 모두 없음 |
| cache 재사용, select, pruning | 추가됨 | 추가됨 |

측정할 때 분리해 볼 필드(`--timing True`):
- `prefill_ms`: `reuse` 값으로 재사용 스텝과 비재사용(에피소드 첫) 스텝을 나눈다. V'의 비재사용 스텝과 B0의 차이는 순수한 `attn_info` 비용이다.
- `attn_info_ms`: V'의 통계 계산 비용이다. `prefill_ms` 안에 포함되어 있다.
- `select_ms`, `select_static/task/schedule_wall_ms`: CPU 위주 구간이다. V와 V'는 `select_task`와 `select_schedule`의 내용이 다르다(2.2절).
- `decode_ms`, `decode_n`: V(eager)와 V'(SDPA)의 decode 커널 차이를 반영한다.

## 5. 검증 근거와 미확인 사항
- **CPU 단위 테스트** 32개가 통과했다(torch 2.2.0, 2.14.1, 포크 `2302fce`). 주요 테스트:
  - `test_attn_stats.py`: 재사용 유무 각각에서 V' 모듈 내 통계, logits, KV가 eager V와 비트 단위로 같다(bf16/fp32).
  - `test_sdpa_reuse_mask.py`: mask를 강제하면 V'와 V가 수치 오차 안에서 일치한다. 강제하지 않으면 포크의 SDPA 경로가 크게 틀린다.
  - `test_modes_e2e.py`: B0가 동작하고, B1의 attention 출력이 유지되고, V'(`eager_debug`)가 3스텝 동안 V를 재현하고, timing 로그가 기록된다. `install_vprime`이 V 경로를 바꾸지 않는다.
- **검증 스크립트**(`--tiny_random`): 검사 A/B가 PASS다. 검사 C(SDPA)에서는 기본 프롬프트로 결정과 액션이 V와 같았다. 2.4절처럼 프롬프트에 따라 결정이 조금 달라질 수 있다.
- **이 문서를 위한 추가 확인**(scratchpad 스크립트, 저장소 변경 없음): 3.1절의 첫 스텝 동일성, 3.2절의 mask 형상, 2.3절의 형상, 2.4절의 수치 차이.
  - tiny 모델에서는 B0와 V'의 2번째 스텝 액션도 같았다. 그러나 무작위 소형 모델에서 나온 결과이므로 일반화할 수 없다.
- **미확인(Thor에서 확인 필요)**
  1. B1/V의 실제 attention 구현: 로드 로그의 `[*] VLA mode ... attn_implementation` 줄로 확인한다. sdpa로 해석된다는 것은 소스 추론이다.
  2. 각 경로에서 선택되는 SDPA 백엔드(B0와 V' 재사용 prefill)
  3. 실제 체크포인트와 실제 프레임에서의 검사 C 수치(재사용 인덱스 Jaccard, 비율 차이, 액션 일치)
  4. 단계별 시간: `attn_info_ms`가 prefill에서 차지하는 비율, V의 select 축약 비용과 V'의 `attn_info` 비교
