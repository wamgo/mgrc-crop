"""Three-level mixed-granularity risk control on EuroCropsML (LV <-> EE, 2021).

Levels: L1 = cereals (HCAT 330101) / other temporary crops (rest of 3301 except 330109) / grassland, green
fodder and permanent crops (330109, 3302, 3303). This follows the WorldCereal definition of temporary crops
(FAO: less than one-year cycle; perennial crops and temporary pastures excluded), so that the product can
supply L1 labels (exp_worldcereal.py). L2 = HCAT 6-digit groups (13); L3 = 24 fine classes. Prices in
fine-label units: fine 1, L2 1/rho2, L1 1/rho1.
A design (b3, q, gamma) spends b3*B on fine labels and splits the rest between L2 (share q) and L1 labels;
gamma splits alpha over the three error terms. Label sets are nested: fine-labelled parcels also count as
L2- and L1-labelled, L2-labelled parcels as L1-labelled. Designs are chosen on source labels only; the
three-level planner also considers the two-level designs (dropping L1) and the fine-label-only rules
(flat selective, HSC climbing), so it can decide not to buy a label type.

Usage: python exp_multilevel.py main <pac|crc> [workers]
"""
from pathlib import Path
import itertools, json, sys, time
from pilot_common import setup
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, pandas as pd
import hcrc_multi as M

TAX = json.loads((ROOT / 'data' / 'taxonomy_eurocrops.json').read_text(encoding='utf8'))
FINE = TAX['fine']
L1_NAMES = ['cereals', 'other_temporary', 'grass_fodder_permanent']
A1 = np.array([0 if c.startswith('330101') else 2 if c.startswith(('330109', '3302', '3303')) else 1 for c in FINE])
A2 = np.array(TAX['parent_of_fine'])
H3 = M.Hierarchy([A1, A2, np.arange(len(FINE))])
H2 = M.Hierarchy([A2, np.arange(len(FINE))])
PROBS = ROOT / 'results' / 'probs'
OUT = ROOT / 'results' / 'exp' / 'multilevel'
OUT.mkdir(parents=True, exist_ok=True)

STEP = 0.1
G3 = [g for g in itertools.product(*[np.round(np.arange(0, 1 + 1e-9, STEP), 2)] * 3)
      if abs(sum(g) - 1) < 1e-9 and g[0] > 0]
G2 = [(round(1 - a, 2), a) for a in np.round(np.arange(0, 1 + 1e-9, STEP), 2) if a < 1]
B3S, QS = [0.0, 0.1, 0.25, 0.5, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]


def designs3(fine_only=False):
    out = []
    for b3 in ([1.0] if fine_only else B3S):
        for q in ([1.0] if b3 == 1.0 else QS):
            for g in G3:
                if b3 == 0 and g[2] > 0:
                    continue
                out.append((b3, q, g))
    return out


def designs2():
    return [(b, None, g) for b in B3S for g in G2 if not (b == 0 and g[1] > 0)]


def sizes(d, B, rho2, rho1, L):
    b3, q, _ = d
    nf = int(round(b3 * B))
    rem = B - nf
    if L == 2:
        return [nf + int(round(rem * rho2)), nf]
    n2 = nf + int(round(q * rem * rho2))
    return [n2 + int(round((1 - q) * rem * rho1)), n2, nf]


class Cal:
    """Threshold cache for one ordered label pool: thresholds depend only on (level, n_l, alpha_l)."""

    def __init__(self, S, alpha, delta, mode):
        self.S, self.alpha, self.delta, self.mode, self.c = S, alpha, delta, mode, {}

    def thresholds(self, ns, gamma):
        m = max(sum(g > 0 for g in gamma), 1)
        t = []
        for l, (n, g) in enumerate(zip(ns, gamma)):
            k = (l, n, g, m)
            if k not in self.c:
                self.c[k] = M.thr(self.S.s[l, :n], self.S.e[l, :n], g * self.alpha, self.delta / m, self.mode)
            t.append(self.c[k])
        return np.maximum.accumulate(np.array(t, float))


FINE_ONLY = [('flat', None), ('hsc', None)]


def execute(L, d, B, rho2, rho1, cal, E, alpha, delta, mode):
    """Calibrate design (L, d) on the ordered pools in cal and evaluate on E (both dicts keyed by L = 2, 3).
    L = 'flat' or 'hsc' are the fine-label-only rules, calibrated on the first B parcels."""
    if L in ('flat', 'hsc'):
        F = cal[3].S.sub(np.arange(B))
        if L == 'flat':
            return M.evaluate_flat(E[3], M.calibrate_flat(F, alpha, delta, mode))
        return E[3].evaluate_climb(M.calibrate_climb(F, alpha, delta, mode))
    return E[L].evaluate(cal[L].thresholds(sizes(d, B, rho2, rho1, L), d[2]))


def plan(srcs, cands, alpha, delta, mode, B, rho2, rho1, sims=10, seed=0):
    """cands: list of (L, design); srcs: {L: source Scores}. Same simulated label draws for every candidate."""
    rng = np.random.default_rng(seed)
    n = srcs[3].n
    score = np.zeros(len(cands))
    for _ in range(sims):
        perm = rng.permutation(n)
        E = {L: S.sub(perm[-5000:]) for L, S in srcs.items()}
        cal = {L: Cal(S.sub(perm[:-5000]), alpha, delta, mode) for L, S in srcs.items()}
        for i, (L, d) in enumerate(cands):
            score[i] += execute(L, d, B, rho2, rho1, cal, E, alpha, delta, mode)['hcov']
    return cands[int(np.argmax(score))]


def source_rule(src, alpha, delta, mode, seed=0):
    """gamma for the source-calibrated rule: best info coverage on one half of the source, calibrated on the other."""
    perm = np.random.default_rng(seed).permutation(src.n)
    A, Bh = src.sub(perm[:src.n // 2]), src.sub(perm[src.n // 2:])
    cal = Cal(A, alpha, delta, mode)
    g = max(G3, key=lambda g: Bh.evaluate(cal.thresholds([A.n] * 3, g))['hcov'])
    return Cal(src, alpha, delta, mode).thresholds([src.n] * 3, g)


def load(tag, model, target):
    d = np.load(PROBS / f'{tag}_{model}.npz')
    return (d['source_hold__p'], d['source_hold__y']), (d[f'{target}__p'], d[f'{target}__y'])


def run_pair(tag, model, target, budgets, prices, alpha=0.1, delta=0.1, mode='pac', reps=50,
             test_n=15000, seed=0):
    (ps, ys), (pt, yt) = load(tag, model, target)
    src3, src2 = M.Scores(H3, ps, ys), M.Scores(H2, ps, ys)
    tgt3, tgt2 = M.Scores(H3, pt, yt), M.Scores(H2, pt, yt)
    t_src = source_rule(src3, alpha, delta, mode, seed)
    srcs = {3: src3, 2: src2}
    c3, c2, cf = [(3, d) for d in designs3()], [(2, d) for d in designs2()], [(3, d) for d in designs3(True)]
    plans = {}
    for B, (r2, r1) in itertools.product(budgets, prices):
        args = (alpha, delta, mode, B, r2, r1)
        plans[B, r2, r1] = {'mgrc3': plan(srcs, c3 + c2 + FINE_ONLY, *args, seed=seed),
                            'mgrc2': plan(srcs, c2 + FINE_ONLY, *args, seed=seed),
                            'td3_fine': plan(srcs, cf, *args, seed=seed)}
    rng = np.random.default_rng(seed + 1)
    rows = []
    for rep in range(reps):
        perm = rng.permutation(tgt3.n)
        test, pool = perm[:test_n], perm[test_n:]
        T3, T2 = tgt3.sub(test), tgt2.sub(test)
        P3, P2 = tgt3.sub(pool), tgt2.sub(pool)
        base = {'tag': tag, 'model': model, 'target': target, 'mode': mode, 'alpha': alpha, 'rep': rep}

        def rec(method, res, **kw):
            sh = np.r_[np.zeros(3 - len(res['shares'])), res['shares']]
            rows.append({**base, **kw, 'method': method, 'risk': res['risk'], 'hcov': res['hcov'],
                         'l1': sh[0], 'l2': sh[-2], 'fine': sh[-1], 'root': 1 - sh.sum()})

        for (B, r2, r1), pl in plans.items():
            kw = {'B': B, 'rho2': r2, 'rho1': r1}
            if (B, r2, r1) == next(iter(plans)):
                rec('source3', T3.evaluate(t_src), B=np.nan, rho2=np.nan, rho1=np.nan)
            F = P3.sub(np.arange(B))
            rec('flat', M.evaluate_flat(T3, M.calibrate_flat(F, alpha, delta, mode)), **kw)
            rec('hsc3', T3.evaluate_climb(M.calibrate_climb(F, alpha, delta, mode)), **kw)
            cal = {3: Cal(P3, alpha, delta, mode), 2: Cal(P2, alpha, delta, mode)}
            E = {3: T3, 2: T2}
            for name, (L, d) in pl.items():
                res = execute(L, d, B, r2, r1, cal, E, alpha, delta, mode)
                if d is None:
                    rec(name, res, **kw, levels=str(L))
                else:
                    rec(name, res, **kw, levels=str(L), b3=d[0], q=d[1] if d[1] is not None else np.nan,
                        gamma=','.join(map(str, d[2])), n_labels=','.join(map(str, sizes(d, B, r2, r1, L))))
    return rows


SPEC = {'main': ([10, 25, 50, 100, 200], [(2, 20), (5, 20), (10, 40)])}


def pairs():
    doms = json.loads((ROOT / 'data' / 'eurocrops_domains.json').read_text(encoding='utf8'))
    out = []
    for s in ['LV', 'EE']:
        o = 'EE' if s == 'LV' else 'LV'
        out += [(f'EC2021_{s}', m, f'2021_EC_{t}') for t in sorted(doms) if t.startswith(o) and len(t) > 2
                for m in ['mlp', 'lgbm', 'tempcnn', 'ltae']]
    return out


if __name__ == '__main__':
    import concurrent.futures
    exp, mode = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    budgets, prices = SPEC[exp]
    out = OUT / f'{exp}_{mode}.parquet'
    P = pairs()
    t0, rows = time.time(), []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_pair, *p, budgets, prices, mode=mode, seed=30000 + i) for i, p in enumerate(P)]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            rows += f.result()
            print('DONE', exp, mode, i + 1, len(P), round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_parquet(out)
    print('SAVED', out.name, len(rows), flush=True)
