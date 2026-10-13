"""PLAN001: validates the per-step CSVs (6.1) and prints the summary tables (6.2) as markdown."""
import csv
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict

LOGS = sys.argv[1]
HEADER = "cond,task_id,episode,step,step_type,t_total_ms,t_vision_ms,t_prune_ms,t_generate_ms,t_fit_ms,n_vis_tokens,z_trans,success,episode_len".split(",")
TIMES = ["t_total_ms", "t_generate_ms", "t_vision_ms", "t_prune_ms"]


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]


def stats(xs):
    return f"{st.mean(xs):.1f} / {st.median(xs):.1f} / {p90(xs):.1f}" if xs else "-"


def load(mode):
    with open(f"{LOGS}/PLAN001-steps-{mode}.csv") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == HEADER, reader.fieldnames
        return list(reader)


def validate(mode, rows):
    problems = []
    episodes = defaultdict(list)
    for row in rows:
        episodes[(int(row["task_id"]), int(row["episode"]))].append(row)
    expected = {(t, e) for t in range(10) for e in range(5)}
    if set(episodes) != expected:
        problems.append(f"missing episodes: {sorted(expected - set(episodes))}")
    order_viol = 0
    for key, ep in episodes.items():
        if [int(r["step"]) for r in ep] != list(range(len(ep))):
            problems.append(f"{key}: steps not contiguous")
        if any(r["success"] != "" or r["episode_len"] != "" for r in ep[:-1]):
            problems.append(f"{key}: success/episode_len before the last row")
        if ep[-1]["success"] not in ("0", "1") or ep[-1]["episode_len"] != str(len(ep)):
            problems.append(f"{key}: bad last row {ep[-1]['success']!r} {ep[-1]['episode_len']!r}")
    max_slack = 0.0
    for r in rows:
        if r["cond"] != mode or r["step_type"] != "infer" or float(r["t_fit_ms"]) != 0:
            problems.append("bad cond/step_type/t_fit")
            break
        if mode == "B0" and float(r["t_prune_ms"]) != 0:
            problems.append("B0 t_prune_ms != 0")
            break
        v, g, t = float(r["t_vision_ms"]), float(r["t_generate_ms"]), float(r["t_total_ms"])
        slack = max(v - g, g - t)
        max_slack = max(max_slack, slack)
        order_viol += slack > 0
    successes = sum(int(ep[-1]["success"]) for ep in episodes.values() if ep[-1]["success"] in ("0", "1"))
    return episodes, successes, problems, order_viol, max_slack


def eval_log(mode):
    path = sorted(glob.glob(f"{LOGS}/EVAL-libero_spatial-openvla-{mode}-*--PLAN001.txt"))[-1]
    text = open(path).read()
    succ = re.findall(r"# successes: (\d+)", text)
    return path, int(succ[-1]) if succ else None, text.count("Caught exception"), re.findall(r"Caught exception: (.*)", text)


def jsonl(mode):
    path = sorted(glob.glob(f"{LOGS}/EVAL-libero_spatial-openvla-{mode}-*--PLAN001-timing.jsonl"))[-1]
    return [json.loads(line) for line in open(path)]


out = []
data = {}
for mode in ("B0", "Vp"):
    rows = load(mode)
    episodes, successes, problems, order_viol, max_slack = validate(mode, rows)
    path, log_succ, n_exc, exc = eval_log(mode)
    if log_succ != successes:
        problems.append(f"success mismatch: csv {successes} vs log {log_succ}")
    data[mode] = (rows, episodes, successes)
    out.append(f"### {mode}: validation")
    out.append(f"- rows {len(rows)}, episodes {len(episodes)}, successes {successes}/50 (eval log: {log_succ}), `Caught exception` {n_exc}")
    out.append(f"- ordering `t_vision <= t_generate <= t_total`: {order_viol} rows violate, max excess {max_slack:.3f} ms")
    out.append(f"- problems: {problems or 'none'}")
    if exc:
        out.append(f"- exceptions: {sorted(set(exc))}")
    per_task = [sum(int(episodes[(t, e)][-1]["success"] or 0) for e in range(5) if (t, e) in episodes) for t in range(10)]
    out.append(f"- per-task successes (task 0..9, of 5): {per_task}")
    out.append(f"- mean episode length: {len(rows) / max(len(episodes), 1):.1f} steps")
    out.append("")

out.append("### Time statistics (ms; mean / median / p90)")
for label, pick in (("all steps", lambda r: True), ("step == 0", lambda r: r["step"] == "0"), ("step > 0", lambda r: r["step"] != "0")):
    out.append(f"\n**{label}**\n")
    out.append("| mode | n | " + " | ".join(TIMES) + " |")
    out.append("|---|---|" + "---|" * len(TIMES))
    for mode in ("B0", "Vp"):
        sel = [r for r in data[mode][0] if pick(r)]
        out.append(f"| {mode} | {len(sel)} | " + " | ".join(stats([float(r[c]) for r in sel]) for c in TIMES) + " |")

out.append("\n### JSONL stage detail (ms; mean / median / p90)\n")
keys = ["prefill_ms", "decode_ms", "decode_wall_ms", "attn_info_ms", "select_static_wall_ms", "select_task_wall_ms", "select_schedule_wall_ms", "n_reusable"]
out.append("| mode | subset | n | " + " | ".join(keys) + " |")
out.append("|---|---|---|" + "---|" * len(keys))
reuse_rows = {}
for mode in ("B0", "Vp"):
    rows = jsonl(mode)
    for label, sel in (("reuse", [r for r in rows if r["reuse"]]), ("no reuse", [r for r in rows if not r["reuse"]])):
        if not sel:
            continue
        out.append(f"| {mode} | {label} | {len(sel)} | " + " | ".join(stats([float(r[k]) for r in sel if k in r]) for k in keys) + " |")
    reuse_rows[mode] = rows

# B0 vs V' on reuse steps: V' rows with reuse=True (CSV row i <-> JSONL row i within a run), B0 rows with step > 0
vp_rows, vp_json = data["Vp"][0], reuse_rows["Vp"]
out.append("\n### V' / B0 ratio of t_total_ms\n")
if len(vp_rows) == len(vp_json):
    vp_reuse = [float(r["t_total_ms"]) for r, j in zip(vp_rows, vp_json) if j["reuse"]]
else:
    out.append(f"- note: CSV rows {len(vp_rows)} != JSONL rows {len(vp_json)}; using step > 0 instead of the reuse flag")
    vp_reuse = [float(r["t_total_ms"]) for r in vp_rows if r["step"] != "0"]
b0_rest = [float(r["t_total_ms"]) for r in data["B0"][0] if r["step"] != "0"]
out.append(f"- V' reuse steps: n={len(vp_reuse)}, mean {st.mean(vp_reuse):.1f}, median {st.median(vp_reuse):.1f}")
out.append(f"- B0 steps with step > 0: n={len(b0_rest)}, mean {st.mean(b0_rest):.1f}, median {st.median(b0_rest):.1f}")
out.append(f"- ratio V'/B0: mean {st.mean(vp_reuse) / st.mean(b0_rest):.4f}, median {st.median(vp_reuse) / st.median(b0_rest):.4f}")
print("\n".join(out))
