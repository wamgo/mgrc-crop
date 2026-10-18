"""Pilot: does a source-proposed threshold path, verified on the target with labels of both granularities, beat
MGRC as calibrated now?

Methods on the same target draws (two levels, high-probability version, delta = 0.1):
  mgrc        planned MGRC (source planner chooses rule, beta, gamma; thresholds from target labels)
  path_fine   source path, verified with crop codes only (the existing two-threshold baseline)
  path_mixed  source path, verified with both label kinds: along the path (t_p(lambda), t_f(lambda)) the group
              term is tested on all group-labelled parcels at gamma*alpha and the code term on the code-labelled
              parcels at (1-gamma)*alpha, each at delta/2; lambda is the last index at which both tests reject.
              Same budget split beta and gamma as MGRC's planned design.
  margin25    source thresholds at alpha / 2.5 (the margin that was safe in hindsight on every transfer)

The path is fixed from source data before target labels are drawn, so fixed-sequence testing along it is valid.
Usage: MGRC_TAXON=pheno python pilot_path.py
"""
import concurrent.futures, sys
import numpy as np, pandas as pd
import experiments as E
import run_v2 as R

ALPHAS, BUDGETS, RHO, REPS = [0.02, 0.05], [25, 50, 100, 200], 10, 30


def path_mixed(F, C, path, alpha, gamma):
    """Fixed-sequence test along the source path, from the strict end (+inf, last row) towards the lenient end
    (row 0). A path point is accepted only if both terms are certified there; the scan stops at the first point that
    fails, and the last accepted point is returned. Each point is tested once, so the family-wise error of the whole
    sequence is at most the sum of the two per-term levels delta/2 + delta/2."""
    ac, af = gamma * alpha, (1 - gamma) * alpha
    k = len(path) - 1                                   # +inf end: always valid (answers nothing)
    for i in range(len(path) - 2, -1, -1):
        tp, tf = path[i]
        tf = max(tf, tp) if af > 0 else np.inf           # no risk budget for the code term: never answer a code
        kc = int(np.sum((C.sg >= tp) & C.ep))
        kf = int(np.sum((F.sf >= tf) & F.ef))
        okc = ac > 0 and E.crit(kc, C.n, 0.5) <= ac
        okf = af <= 0 or E.crit(kf, F.n, 0.5) <= af
        if not (okc and okf):
            break
        k = i
    tp, tf = path[k]
    return tp, (max(tf, tp) if af > 0 else np.inf)


def run_pair(tag, model, target, seed):
    E.MODE['mode'], E.MODE['delta'] = 'pac', 0.1
    src, tgt, _ = E.load_pair(tag, model, target)
    path = E.path_from_source(src, levels=np.linspace(.002, .6, 300))
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in ALPHAS:
        r_src = E.source_split(src, alpha)
        plans = {B: E.plan(src, alpha, B, RHO, E.BETAS, ['info'], seed=seed)['info'] for B in BUDGETS}
        for rep in range(REPS):
            perm = rng.permutation(tgt.n)
            T, pool = tgt.sub(perm[:15000]), perm[15000:]
            tp, tf = E.cal_levelwise(src, r_src * alpha / 2.5, (1 - r_src) * alpha / 2.5)
            base = {'tag': tag, 'model': model, 'target': target, 'alpha': alpha, 'rep': rep}
            for B in BUDGETS:
                d = plans[B]
                p2 = rng.permutation(pool)
                nf = max(int(round(d[1] * B)), 1)
                nc = int(round((B - nf) * RHO))
                F, C = tgt.sub(p2[:nf]), tgt.sub(p2[:nf + nc])
                gamma = d[2] if d[2] is not None else 0.9
                res = {'mgrc': E.apply_design(d, F, C, alpha, T),
                       'path_fine': T.levelwise(*E.cal_path(tgt.sub(p2[:B]), path, alpha)),
                       'path_mixed': T.levelwise(*path_mixed(F, C, path, alpha, gamma)),
                       'margin25': T.levelwise(tp, tf)}
                for m, (risk, fine, par, cov) in res.items():
                    rows.append({**base, 'B': B, 'method': m, 'risk': risk, 'fine': fine, 'parent': par, 'hcov': cov,
                                 'rule': d[0]})
    return rows


if __name__ == '__main__':
    P = R.pairs('breizh', 'year') + R.pairs('breizh', 'main')[::4]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as ex:
        futs = [ex.submit(run_pair, *p, 7000 + i) for i, p in enumerate(P)]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            rows += f.result()
            print('DONE', i + 1, len(P), flush=True)
    out = E.ROOT / 'results' / 'exp' / 'v2' / 'pilot_path.parquet'
    pd.DataFrame(rows).to_parquet(out)
    print('saved', out.name, len(rows))
