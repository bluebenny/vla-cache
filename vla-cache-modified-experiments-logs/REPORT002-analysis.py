"""
REPORT002-analysis.py

Reproduces every number in REPORT002-PLAN001-results.md from the PLAN001 artifacts in this folder.
Usage (numpy only):  python REPORT002-analysis.py [this folder]
Sections: [1] CSV/JSONL consistency + success, [2] step-time decomposition, [3] warm-up phase and
comparison without it.
"""
import os
import sys

import csv, glob, json, math
import numpy as np

HERE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def _section_plan001_analysis(D):
    def load_csv(m): return list(csv.DictReader(open(f"{D}/PLAN001-steps-{m}.csv")))
    def load_jsonl(m): return [json.loads(l) for l in open(glob.glob(f"{D}/EVAL-*-{m}-*--PLAN001-timing.jsonl")[0])]
    R = {}
    for m in ["B0", "Vp"]:
        rows, js = load_csv(m), load_jsonl(m)
        assert len(rows) == len(js), (m, len(rows), len(js))
        # 1) CSV <-> JSONL consistency (definitions in PLAN-001 2.1)
        bad = 0
        for r, j in zip(rows, js):
            assert (int(r["task_id"]), int(r["episode"]), int(r["step"])) == (j["task_id"], j["episode"], j["t"] - 10)
            t_total = j["step_total_wall_ms"] - j.get("center_crop_wall_ms", 0) - j.get("preprocess_wall_ms", 0)
            checks = [(float(r["t_total_ms"]), t_total), (float(r["t_vision_ms"]), j["vision_ms"]),
                      (float(r["t_generate_ms"]), j["generate_ms"]),
                      (float(r["t_prune_ms"]), j.get("select_wall_ms", 0) if m == "Vp" else 0)]
            bad += any(abs(a - b) > 1e-3 + 1e-6 * abs(b) for a, b in checks)
        R[m] = (rows, js)
        print(f"[{m}] rows={len(rows)} csv/jsonl definition mismatches={bad}")
    
    def stats(x):
        x = np.asarray(x, float); return f"{x.mean():7.1f} / {np.median(x):7.1f} / {np.percentile(x,90):7.1f}  (n={len(x)})"
    
    # 2) success, paired by (task, episode)
    succ = {}
    for m, (rows, js) in R.items():
        succ[m] = {(int(r["task_id"]), int(r["episode"])): int(r["success"]) for r in rows if r["success"] != ""}
    keys = sorted(succ["B0"])
    b = sum(succ["B0"][k] and not succ["Vp"][k] for k in keys); c = sum(succ["Vp"][k] and not succ["B0"][k] for k in keys)
    both = sum(succ["B0"][k] and succ["Vp"][k] for k in keys); none_ = sum((not succ["B0"][k]) and (not succ["Vp"][k]) for k in keys)
    def binom_two_sided(b, c):
        n = b + c; k = min(b, c)
        return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    def wilson(s, n, z=1.96):
        p = s / n; d = 1 + z*z/n; c0 = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
        return c0 - h, c0 + h
    for m in R:
        s = sum(succ[m].values()); lo, hi = wilson(s, 50); print(f"[{m}] success {s}/50 = {s/50:.0%}  Wilson95 [{lo:.2f}, {hi:.2f}]")
    print(f"paired: both={both} none={none_} B0only={b} Vponly={c}  McNemar exact p={binom_two_sided(b, c):.3f}")
    
    # 3) per-step timing decomposition (exclude step 0 = no reuse; exclude first episode of run = warm-up)
    def comp(j, k): return j.get(k, 0.0)
    def steps(m, reuse=None, skip_first_episode=True):
        out = []
        for j in R[m][1]:
            if skip_first_episode and j["task_id"] == 0 and j["episode"] == 0: continue
            if reuse is not None and j["reuse"] != reuse: continue
            if reuse is None and j["t"] == 10: continue
            out.append(j)
        return out
    print("\n== components, steps > 0, first episode of each run excluded (mean / median / p90 ms) ==")
    for m, sub in [("B0", steps("B0")), ("Vp", steps("Vp", reuse=True))]:
        print(f"-- {m} --")
        t_total = [j["step_total_wall_ms"] - comp(j,"center_crop_wall_ms") - comp(j,"preprocess_wall_ms") for j in sub]
        print(" t_total         ", stats(t_total))
        print(" generate (event)", stats([j["generate_ms"] for j in sub]))
        print(" vision          ", stats([j["vision_ms"] for j in sub]))
        print(" prefill         ", stats([j["prefill_ms"] for j in sub]))
        print(" prefill-attn_info", stats([j["prefill_ms"] - comp(j,"attn_info_ms") for j in sub]))
        print(" attn_info       ", stats([comp(j,"attn_info_ms") for j in sub]))
        print(" decode (6 tok)  ", stats([j["decode_ms"] for j in sub]))
        print(" gen overhead    ", stats([j["generate_ms"] - j["vision_ms"] - j["prefill_ms"] - j["decode_ms"] for j in sub]))
        print(" select (wall)   ", stats([comp(j,"select_wall_ms") for j in sub]))
        print(" outside generate", stats([j["step_total_wall_ms"] - comp(j,"center_crop_wall_ms") - comp(j,"preprocess_wall_ms") - j["generate_ms"] - comp(j,"select_wall_ms") for j in sub]))
    vp_noreuse = [j for j in R["Vp"][1] if not j["reuse"] and not (j["task_id"] == 0 and j["episode"] == 0)]
    print("-- Vp no-reuse (episode first steps, excl. run's first) --")
    print(" prefill         ", stats([j["prefill_ms"] for j in vp_noreuse]))
    print(" prefill-attn_info", stats([j["prefill_ms"] - j["attn_info_ms"] for j in vp_noreuse]))
    print(" attn_info       ", stats([j["attn_info_ms"] for j in vp_noreuse]))
    b0_first = [j for j in R["B0"][1] if j["t"] == 10 and not (j["task_id"] == 0 and j["episode"] == 0)]
    print("-- B0 episode first steps --  prefill", stats([j["prefill_ms"] for j in b0_first]))
    
    # 4) does reuse shrink prefill? regress (prefill - attn_info) on n_reusable for Vp reuse steps
    sub = steps("Vp", reuse=True)
    x = np.array([j["n_reusable"] for j in sub], float); y = np.array([j["prefill_ms"] - j["attn_info_ms"] for j in sub])
    A = np.vstack([x, np.ones_like(x)]).T; slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"\n(prefill - attn_info) vs n_reusable: slope={slope*1000:.1f} us/token, intercept={icpt:.1f} ms, corr={np.corrcoef(x,y)[0,1]:.2f}; n_reusable range {x.min():.0f}-{x.max():.0f}")
    xa = np.array([j["n_reusable"] for j in sub], float); ya = np.array([j["attn_info_ms"] for j in sub])
    print(f"attn_info vs n_reusable: corr={np.corrcoef(xa,ya)[0,1]:.2f}")
    
    # 5) drift / hardware-state control: vision and decode by run quarter
    print("\n== drift check (median by run quarter): vision_ms | decode_ms | prefill_ms ==")
    for m in R:
        js = R[m][1]; q = np.array_split(np.arange(len(js)), 4)
        print(m, [f"{np.median([js[i]['vision_ms'] for i in idx]):.1f} | {np.median([js[i]['decode_ms'] for i in idx]):.1f} | {np.median([js[i]['prefill_ms'] for i in idx]):.1f}" for idx in q])
    
    # 6) warm-up: first episode of each run
    for m in R:
        e0 = [float(r["t_total_ms"]) for r in R[m][0] if r["task_id"] == "0" and r["episode"] == "0"]
        print(f"[{m}] first episode t_total: first 3 steps {[round(v) for v in e0[:3]]}, median {np.median(e0):.1f}")
    
    # 7) episode-level: per-step mean (paired) and episode time
    ep = {m: {} for m in R}
    for m in R:
        for r in R[m][0]:
            k = (int(r["task_id"]), int(r["episode"])); ep[m].setdefault(k, []).append(float(r["t_total_ms"]))
    d = np.array([np.mean(ep["Vp"][k]) - np.mean(ep["B0"][k]) for k in keys])
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f"\nper-episode mean t_total, paired Vp - B0: {d.mean():.1f} ms (95% CI {d.mean()-1.96*se:.1f} .. {d.mean()+1.96*se:.1f}); Vp slower in {int((d>0).sum())}/50 episodes")
    for m in R:
        L = np.array([len(ep[m][k]) for k in keys]); T = np.array([sum(ep[m][k]) / 1000 for k in keys])
        Ls = np.array([len(ep[m][k]) for k in keys if succ[m][k]])
        print(f"[{m}] episode len mean {L.mean():.1f} (success-only {Ls.mean():.1f}); model time per episode mean {T.mean():.1f} s, total {T.sum()/60:.1f} min")


def _section_plan001_extra(D):
    rows = {m: list(csv.DictReader(open(f"{D}/PLAN001-steps-{m}.csv"))) for m in ["B0", "Vp"]}
    js = {m: [json.loads(l) for l in open(glob.glob(f"{D}/EVAL-*-{m}-*--PLAN001-timing.jsonl")[0])] for m in ["B0", "Vp"]}
    # warm-up span: rolling median of t_total over the run's first 400 steps
    for m in rows:
        t = np.array([float(r["t_total_ms"]) for r in rows[m]])
        slow = np.where(t > 520)[0]
        print(f"[{m}] steps with t_total>520ms: {len(slow)}; first 400 steps by 50: {[round(np.median(t[i:i+50])) for i in range(0, 400, 50)]}")
        eps = [(r["task_id"], r["episode"]) for r in rows[m]]
        print(f"     slow steps' episodes: {sorted(set(eps[i] for i in slow))[:10]}")
    # B0 decode tail
    for m in js:
        d = np.array([j["decode_ms"] for j in js[m]])
        big = np.where(d > 380)[0]
        print(f"[{m}] decode>380ms: {len(big)} steps; their (task,episode,t) first: {[(js[m][i]['task_id'], js[m][i]['episode'], js[m][i]['t']) for i in big[:6]]}")
    # paired per-episode mean t_total excluding the run's first episode (B0 warm-up)
    ep = {m: {} for m in rows}
    for m in rows:
        for r in rows[m]:
            ep[m].setdefault((int(r["task_id"]), int(r["episode"])), []).append(r)
    keys = [k for k in sorted(ep["B0"]) if k != (0, 0)]
    d = np.array([np.mean([float(r["t_total_ms"]) for r in ep["Vp"][k]]) - np.mean([float(r["t_total_ms"]) for r in ep["B0"][k]]) for k in keys])
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f"paired per-episode mean t_total Vp-B0 (excl. episode (0,0)): {d.mean():.1f} ms, 95% CI {d.mean()-1.96*se:.1f}..{d.mean()+1.96*se:.1f}, Vp slower in {(d>0).sum()}/{len(d)}")
    # episode lengths where both succeeded
    both = [k for k in sorted(ep["B0"]) if ep["B0"][k][-1]["success"] == "1" and ep["Vp"][k][-1]["success"] == "1"]
    lb = np.array([len(ep["B0"][k]) for k in both]); lv = np.array([len(ep["Vp"][k]) for k in both])
    dd = lv - lb; se = dd.std(ddof=1)/math.sqrt(len(dd))
    print(f"both-success episodes n={len(both)}: len B0 {lb.mean():.1f}, Vp {lv.mean():.1f}, paired diff {dd.mean():.1f} (95% CI {dd.mean()-1.96*se:.1f}..{dd.mean()+1.96*se:.1f})")
    # per-task successes
    for m in rows:
        print(m, [sum(int(ep[m][(t, e)][-1]["success"]) for e in range(5)) for t in range(10)])
    # smoke speed
    for m in ["B0", "Vp"]:
        s = [float(r["t_total_ms"]) for r in csv.DictReader(open(f"{D}/SMOKE-steps-{m}.csv"))]
        print(f"smoke {m}: n={len(s)} median {np.median(s):.1f}")


def _section_plan001_extra2(D):
    js = {m: [json.loads(l) for l in open(glob.glob(f"{D}/EVAL-*-{m}-*--PLAN001-timing.jsonl")[0])] for m in ["B0", "Vp"]}
    SLOW = {(0, 0), (0, 1), (0, 2)}
    for m in js:
        slow = [j for j in js[m] if (j["task_id"], j["episode"]) in SLOW and j["t"] > 10]
        rest = [j for j in js[m] if (j["task_id"], j["episode"]) not in SLOW and j["t"] > 10 and (m == "B0" or j["reuse"])]
        f = lambda xs, k: np.median([x[k] for x in xs])
        print(f"[{m}] task0 ep0-2 (n={len(slow)}): decode {f(slow,'decode_ms'):.1f}, prefill {f(slow,'prefill_ms'):.1f}, vision {f(slow,'vision_ms'):.1f} | rest (n={len(rest)}): decode {f(rest,'decode_ms'):.1f}, prefill {f(rest,'prefill_ms'):.1f}, vision {f(rest,'vision_ms'):.1f}")
    def tot(j): return j["step_total_wall_ms"] - j.get("center_crop_wall_ms", 0) - j.get("preprocess_wall_ms", 0)
    print("== excluding task 0 episodes 0-2 in BOTH modes, steps > 0 (V': reuse steps) ==")
    res = {}
    for m in js:
        sub = [j for j in js[m] if (j["task_id"], j["episode"]) not in SLOW and j["t"] > 10]
        x = np.array([tot(j) for j in sub]); res[m] = x
        print(f"[{m}] t_total mean {x.mean():.1f} median {np.median(x):.1f} p90 {np.percentile(x,90):.1f} n={len(x)}; decode mean {np.mean([j['decode_ms'] for j in sub]):.1f}")
    print(f"V'/B0: mean {res['Vp'].mean()/res['B0'].mean():.4f}, median {np.median(res['Vp'])/np.median(res['B0']):.4f}, diff mean {res['Vp'].mean()-res['B0'].mean():.1f} ms")
    # Budget: B0 time share
    b = [j for j in js["B0"] if (j["task_id"], j["episode"]) not in SLOW and j["t"] > 10]
    tt = np.mean([tot(j) for j in b])
    for k in ["vision_ms", "prefill_ms", "decode_ms"]:
        print(f"B0 share {k}: {np.mean([j[k] for j in b])/tt:.1%}")
    print(f"B0 decode per token: {np.mean([j['decode_ms'] for j in b])/6:.1f} ms")


def _section_paired_without_warmup(D):
    ep = {}
    for m in ["B0", "Vp"]:
        d = {}
        for r in csv.DictReader(open(f"{D}/PLAN001-steps-{m}.csv")):
            if r["step"] == "0":
                continue
            d.setdefault((int(r["task_id"]), int(r["episode"])), []).append(float(r["t_total_ms"]))
        ep[m] = d
    keys = [k for k in sorted(ep["B0"]) if k not in {(0, 0), (0, 1), (0, 2)}]
    d = np.array([np.mean(ep["Vp"][k]) - np.mean(ep["B0"][k]) for k in keys])
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f"paired per-episode (steps>0, excl task0 ep0-2): {d.mean():.1f} ms, "
          f"95% CI {d.mean()-1.96*se:.1f}..{d.mean()+1.96*se:.1f}, Vp slower in {(d>0).sum()}/{len(d)}")


def _section_components_without_warmup(D):
    """Mean ms per step, steps > 0 (V': reuse steps), task 0 episodes 0-2 excluded in both modes."""
    slow = {(0, 0), (0, 1), (0, 2)}
    comp = {}
    for m in ["B0", "Vp"]:
        js = [json.loads(l) for l in open(glob.glob(f"{D}/EVAL-*-{m}-*--PLAN001-timing.jsonl")[0])]
        sub = [j for j in js if (j["task_id"], j["episode"]) not in slow and j["t"] > 10]
        g = lambda j, k: j.get(k, 0.0)
        cols = {
            "t_total": lambda j: j["step_total_wall_ms"] - g(j, "center_crop_wall_ms") - g(j, "preprocess_wall_ms"),
            "generate": lambda j: j["generate_ms"],
            "vision": lambda j: j["vision_ms"],
            "prefill": lambda j: j["prefill_ms"],
            "prefill_minus_attn_info": lambda j: j["prefill_ms"] - g(j, "attn_info_ms"),
            "attn_info": lambda j: g(j, "attn_info_ms"),
            "decode": lambda j: j["decode_ms"],
            "generate_overhead": lambda j: j["generate_ms"] - j["vision_ms"] - j["prefill_ms"] - j["decode_ms"],
            "select_wall": lambda j: g(j, "select_wall_ms"),
            "rest_outside_generate": lambda j: j["step_total_wall_ms"] - g(j, "center_crop_wall_ms")
            - g(j, "preprocess_wall_ms") - j["generate_ms"] - g(j, "select_wall_ms"),
            "n_reusable": lambda j: j["n_reusable"],
        }
        comp[m] = {k: float(np.mean([f(j) for j in sub])) for k, f in cols.items()}
        comp[m]["n"] = len(sub)
    print("component                 B0       Vp     Vp-B0")
    for k in comp["B0"]:
        print(f"{k:24s} {comp['B0'][k]:8.1f} {comp['Vp'][k]:8.1f} {comp['Vp'][k]-comp['B0'][k]:8.1f}")

if __name__ == '__main__':
    _section_plan001_analysis(HERE)
    _section_plan001_extra(HERE)
    _section_plan001_extra2(HERE)
    _section_paired_without_warmup(HERE)
    _section_components_without_warmup(HERE)
