"""Three-level MGRC with free but noisy level-1 labels taken from ESA WorldCereal 2021.

Every target pool parcel gets a WorldCereal level-1 label z at no cost (wc_sample.py). The budget B buys
fine and L2 labels as in exp_multilevel.py; those n2 truth-labelled parcels are kept out of the noisy L1
calibration set, which is the rest of the pool. Since ell_1 <= ell_1^z + 1[g1 = z != y1], the true L1 term
is at most the noisy one plus the shared-error rate eta_s = P(g1 = z != y1) <= eta = P(z != y1). Variants:

  wc_naive   noisy L1 labels, no correction (no guarantee);
  wc_eta     L1 level gamma1*alpha - UCB(eta) from the truth-labelled parcels;
  wc_shared  L1 level gamma1*alpha - UCB(eta_s) from the truth-labelled parcels;
  free_clean oracle reference: error-free L1 labels for the whole pool at no cost;
  mgrc2      two-level MGRC with bought labels only (reference without WorldCereal).

With a correction the confidence delta is split equally over the bound and the active levels. Each variant is
planned on source parcels with the source WorldCereal labels; the candidate list includes the two-level designs.

Usage: python exp_worldcereal.py pac [workers]   (Proposition 3 is a high-probability statement)
"""
from pathlib import Path
import itertools, json, sys, time
import numpy as np, pandas as pd
import hcrc_multi as M
import exp_multilevel as X

WCD = X.ROOT / 'data' / 'worldcereal'
OUT = X.OUT
VARIANTS = ['wc_naive', 'wc_eta', 'wc_shared', 'free_clean']
BS = [0.0, 0.1, 0.25, 0.5, 1.0]


class Pool:
    """Ordered label pool: the first n2 parcels get bought labels, the rest only z (or, for free_clean, y1)."""

    def __init__(self, S3, S2, z, alpha, delta, mode):
        self.S3, self.S2, self.alpha, self.delta, self.mode = S3, S2, alpha, delta, mode
        y1 = X.A1[S3.y]
        self.noisy_err = S3.node[0] != z
        self.disagree = z != y1
        self.shared = (S3.node[0] == z) & (z != y1)
        self.cal2 = X.Cal(S2, alpha, delta, mode)
        self.c = {}

    def thresholds3(self, variant, n2, nf, gamma):
        L1_alpha = gamma[0] * self.alpha
        corr = variant in ('wc_eta', 'wc_shared')
        m = sum(g > 0 for g in gamma) + corr
        dl = self.delta / m
        if corr:
            bad = (self.disagree if variant == 'wc_eta' else self.shared)[:n2]
            eta_bar = float(M.ucb(bad.sum(), n2, dl)) if n2 > 0 else 1.0
            L1_alpha -= eta_bar
        k = (variant, n2, gamma[0], m)
        if k not in self.c:
            S = self.S3
            if variant == 'free_clean':
                s1, e1 = S.s[0], S.e[0]
            else:
                s1, e1 = S.s[0, n2:], self.noisy_err[n2:]
            self.c[k] = M.thr(s1, e1, L1_alpha, dl, self.mode) if L1_alpha > 0 else np.inf
        t1 = self.c[k]
        t2 = self._bought(1, n2, gamma[1], dl)
        t3 = self._bought(2, nf, gamma[2], dl)
        return np.maximum.accumulate(np.array([t1, t2, t3], float))

    def _bought(self, l, n, g, dl):
        k = ('b', l, n, g, dl)
        if k not in self.c:
            self.c[k] = M.thr(self.S3.s[l, :n], self.S3.e[l, :n], g * self.alpha, dl, self.mode)
        return self.c[k]

    def thresholds2(self, n2, nf, gamma):
        return self.cal2.thresholds([n2, nf], gamma)


def cands(variant):
    c2 = [(2, d) for d in X.designs2()] + X.FINE_ONLY
    if variant == 'mgrc2':  # bought labels only, no WorldCereal
        return c2
    return [(3, (b, None, g)) for b in BS for g in X.G3 if not (b == 0 and g[2] > 0)] + c2


def run_design(pool, variant, L, d, B, rho2, T3, T2):
    if L in ('flat', 'hsc'):
        F = pool.S3.sub(np.arange(B))
        a, dl, md = pool.alpha, pool.delta, pool.mode
        if L == 'flat':
            return M.evaluate_flat(T3, M.calibrate_flat(F, a, dl, md)), (B, B)
        return T3.evaluate_climb(M.calibrate_climb(F, a, dl, md)), (B, B)
    nf = int(round(d[0] * B))
    n2 = nf + int(round((B - nf) * rho2))
    if L == 2:
        return T2.evaluate(pool.thresholds2(n2, nf, d[2])), (n2, nf)
    return T3.evaluate(pool.thresholds3(variant, n2, nf, d[2])), (n2, nf)


def plan(src3, src2, zs, variant, alpha, delta, mode, B, rho2, sims=10, seed=0):
    rng = np.random.default_rng(seed)
    C = cands(variant)
    score = np.zeros(len(C))
    for _ in range(sims):
        perm = rng.permutation(src3.n)
        e, p = perm[-5000:], perm[:-5000]
        pool = Pool(src3.sub(p), src2.sub(p), zs[p], alpha, delta, mode)
        E3, E2 = src3.sub(e), src2.sub(e)
        for i, (L, d) in enumerate(C):
            score[i] += run_design(pool, variant, L, d, B, rho2, E3, E2)[0]['hcov']
    return C[int(np.argmax(score))]


def run_pair(tag, model, target, budgets, rho2s, alpha=0.1, delta=0.1, mode='pac', reps=50, test_n=15000, seed=0):
    (ps, ys), (pt, yt) = X.load(tag, model, target)
    w = np.load(WCD / f'wc_{tag}.npz')
    d = np.load(X.PROBS / f'{tag}_{model}.npz')
    for dom in ['source_hold', target]:
        assert np.array_equal(w[f'{dom}__ids'], d[f'{dom}__ids'])
    zs, zt = w['source_hold__z'], w[f'{target}__z']
    src3, src2 = M.Scores(X.H3, ps, ys), M.Scores(X.H2, ps, ys)
    tgt3, tgt2 = M.Scores(X.H3, pt, yt), M.Scores(X.H2, pt, yt)
    y1 = X.A1[yt]
    stats = {'eta': float(np.mean(zt != y1)), 'eta_shared': float(np.mean((tgt3.node[0] == zt) & (zt != y1))),
             'z_missing': float(np.mean(zt < 0)), 'acc_l1_model': float(np.mean(tgt3.node[0] == y1))}
    plans = {}
    for B, r2 in itertools.product(budgets, rho2s):
        a = (alpha, delta, mode, B, r2)
        plans[B, r2] = {v: plan(src3, src2, zs, v, *a, seed=seed) for v in VARIANTS + ['mgrc2']}
    rng = np.random.default_rng(seed + 1)
    rows = []
    for rep in range(reps):
        perm = rng.permutation(tgt3.n)
        test, p = perm[:test_n], perm[test_n:]
        T3, T2 = tgt3.sub(test), tgt2.sub(test)
        pool = Pool(tgt3.sub(p), tgt2.sub(p), zt[p], alpha, delta, mode)
        for (B, r2), pl in plans.items():
            for v, (L, dd) in pl.items():
                res, (n2, nf) = run_design(pool, v, L, dd, B, r2, T3, T2)
                sh = np.r_[np.zeros(3 - len(res['shares'])), res['shares']]
                rows.append({'tag': tag, 'model': model, 'target': target, 'mode': mode, 'alpha': alpha, 'rep': rep,
                             'B': B, 'rho2': r2, 'method': v, 'levels': str(L), 'b3': dd[0] if dd else np.nan,
                             'gamma': ','.join(map(str, dd[2])) if dd else '',
                             'n2': n2, 'nf': nf, 'risk': res['risk'], 'hcov': res['hcov'], 'l1': sh[0], 'l2': sh[1],
                             'fine': sh[2], 'root': 1 - sh.sum(), **stats})
    return rows


if __name__ == '__main__':
    import concurrent.futures
    mode = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = OUT / f'worldcereal_{mode}.parquet'
    P = X.pairs()
    t0, rows = time.time(), []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_pair, *p, [10, 25, 50, 100, 200], [1, 2, 10], mode=mode, seed=40000 + i)
                for i, p in enumerate(P)]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            rows += f.result()
            print('DONE', mode, i + 1, len(P), round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_parquet(out)
    print('SAVED', out.name, len(rows), flush=True)
