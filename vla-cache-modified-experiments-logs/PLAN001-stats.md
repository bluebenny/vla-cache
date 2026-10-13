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
