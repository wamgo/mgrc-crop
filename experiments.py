"""Target-domain calibration experiments for hierarchical selective crop classification.

Every rule is reduced to threshold tests on per-parcel scores, so each configuration can be
re-drawn many times. Labelling cost: one fine (field-survey) label costs 1, one parent-only
(photo-interpretation) label costs 1/rho. Mixed-granularity calibration spends a budget B on
n_f fine labels and n_c=(B-n_f)*rho coarse labels; parcels with fine labels also join the coarse set.
"""
from pathlib import Path
import sys, json, itertools, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_common import setup, save_json
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, pandas as pd
import hcrc

TAX = json.loads((ROOT / 'data/taxonomy.json').read_text(encoding='utf8'))
PARENT = np.array(TAX['parent_of_fine'])
GRID = hcrc.GRID
_M = np.bincount(PARENT)
COV_PARENT = 1 - np.log(_M) / np.log(len(PARENT))  # HSC-style information coverage of a parent answer
PROBS = ROOT / 'results' / 'probs'
OUT = ROOT / 'results' / 'exp'
OUT.mkdir(parents=True, exist_ok=True)


class Scores:
    """Pre-computed quantities for every answer rule."""

    def __init__(self, p, y):
        p = p.astype(np.float64)
        self.n = len(y)
        self.y = y
        g, sg, f, sf = hcrc.top_down(p, PARENT)
        self.sg, self.sf = sg, sf
        self.ep = g != PARENT[y]
        self.ef = (g == PARENT[y]) & (f != y)
        order = np.sort(p, 1)
        f0 = p.argmax(1)
        self.p1, self.p2 = order[:, -1], order[:, -2]
        self.u = np.where(PARENT[None, :] != PARENT[f0][:, None], p, 0).max(1)
        pp = hcrc.parent_probs(p, PARENT)
        self.sg0 = pp[np.arange(self.n), PARENT[f0]]
        self.e_leaf = f0 != y
        self.e_par0 = PARENT[f0] != PARENT[y]
        self.true_p = p[np.arange(self.n), y]
        self.ef0 = (~self.e_par0) & self.e_leaf  # leaf wrong although its parent is right
        self.cov_td = COV_PARENT[g]
        self.cov_bu = COV_PARENT[PARENT[f0]]

    def sub(self, idx):
        s = object.__new__(Scores)
        for k, v in self.__dict__.items():
            s.__dict__[k] = v[idx] if isinstance(v, np.ndarray) else v
        s.n = len(idx)
        return s

    # ---- answers on this score set: returns (risk, fine_frac, parent_frac, root_frac)
    def levelwise(self, tp, tf):
        tf = max(tf, tp)
        fine = self.sf >= tf
        par = (~fine) & (self.sg >= tp)
        err = (fine & (self.ep | self.ef)) | (par & self.ep)
        return err.mean(), fine.mean(), par.mean(), (fine + par * self.cov_td).mean()

    def climb(self, th):
        fine = self.p1 >= th
        par = (~fine) & (self.sg0 >= th)
        err = (fine & self.e_leaf) | (par & self.e_par0)
        return err.mean(), fine.mean(), par.mean(), (fine + par * self.cov_bu).mean()

    def climb2(self, tp, tf):
        """Bottom-up rule with separate thresholds; tf >= tp makes the error split exactly by level."""
        tf = max(tf, tp)
        fine = self.p1 >= tf
        par = (~fine) & (self.sg0 >= tp)
        err = (fine & self.e_leaf) | (par & self.e_par0)
        return err.mean(), fine.mean(), par.mean(), (fine + par * self.cov_bu).mean()

    def flat(self, t):
        fine = self.p1 >= t
        return (fine & self.e_leaf).mean(), fine.mean(), 0.0, fine.mean()

    def cplca(self, q):
        c = 1 - q
        fine = (self.p1 >= c) & (self.p2 < c)
        par = (self.p2 >= c) & (self.u < c)
        err = (fine & self.e_leaf) | (par & self.e_par0)
        return err.mean(), fine.mean(), par.mean(), (fine + par * self.cov_bu).mean()


# ---------------------------------------------------------------- threshold selection
from scipy.stats import beta as _beta
MODE = {'mode': 'crc', 'delta': 0.1}


def crit(k, n, delta_share=1.0):
    if MODE['mode'] == 'crc':
        return (k + 1.0) / (n + 1.0)
    d = MODE['delta'] * delta_share
    k = np.asarray(k, float)
    return np.where(k >= n, 1.0, _beta.ppf(1 - d, k + 1, np.maximum(n - k, 1e-9)))


def counts_on_grid(score, err):
    order = np.argsort(score)
    s, e = score[order], err[order].astype(float)
    tail = np.r_[np.cumsum(e[::-1])[::-1], 0.0]
    return tail[np.searchsorted(s, GRID, side='left')]


def pick(counts, n, alpha, delta_share=1.0, values=GRID):
    if n == 0:
        return values[-1]
    ok = crit(counts, n, delta_share) <= alpha
    return values[np.argmax(ok)] if ok.any() else values[-1]


def thr(score, err, alpha, delta_share=1.0):
    return pick(counts_on_grid(score, err), len(score), alpha, delta_share)


# ---------------------------------------------------------------- calibrators on Scores
def cal_levelwise(s, ap, af, coarse=None):
    """coarse: Scores restricted to parent-labelled parcels (only .sg/.ep used)."""
    c = coarse if coarse is not None else s
    tp = thr(c.sg, c.ep, ap, 0.5)
    tf = thr(s.sf, s.ef, af, 0.5)
    return tp, max(tf, tp)


def cal_climb2(s, ap, af, coarse=None):
    c = coarse if coarse is not None else s
    tp = thr(c.sg0, c.e_par0, ap, 0.5)
    tf = thr(s.p1, s.ef0, af, 0.5)
    return tp, max(tf, tp)


def cal_by_grid(s, rule, alpha):
    errs = np.round(np.array([getattr(s, rule)(t)[0] for t in GRID]) * s.n)
    return pick(errs, s.n, alpha)


def cal_climb(s, alpha):
    # risk is non-increasing in theta for climbing (more abstention), use exact grid scan
    return cal_by_grid(s, 'climb', alpha)


def cal_flat(s, alpha):
    return thr(s.p1, s.e_leaf, alpha)


def cal_cp(s, alpha):
    sc = np.sort(1 - s.true_p)
    k = int(np.ceil((s.n + 1) * (1 - alpha)))
    return sc[k - 1] if k <= s.n else np.inf


def source_split(src, alpha, ratios=np.linspace(.05, .95, 19)):
    """Choose alpha_parent/alpha on held-out SOURCE labels to maximise fine answers (empirical risk)."""
    best = None
    for r in ratios:
        tp = hcrc.crc_threshold(src.sg, src.ep, r * alpha)
        tf = max(hcrc.crc_threshold(src.sf, src.ef, (1 - r) * alpha), tp)
        _, fine, par, cov = src.levelwise(tp, tf)
        key = (cov, fine)
        if best is None or key > best[0]:
            best = (key, r)
    return best[1]


def path_from_source(src, levels=np.linspace(.004, .6, 150)):
    pts = []
    for a in levels:
        r = source_split(src, a)
        tp = hcrc.crc_threshold(src.sg, src.ep, r * a)
        tf = max(hcrc.crc_threshold(src.sf, src.ef, (1 - r) * a), tp)
        pts.append((tp, tf))
    path = np.maximum.accumulate(np.array(pts)[::-1], axis=0)
    return np.vstack([path, [np.inf, np.inf]])


def cal_path(s, path, alpha):
    errs = np.round(np.array([s.levelwise(tp, tf)[0] for tp, tf in path]) * s.n)
    k = int(pick(errs, s.n, alpha, values=np.arange(len(path))))
    return path[k]




# ---------------------------------------------------------------- source-only design planner
def utility(res, w):
    risk, fine, par, cov = res
    return cov if w == 'info' else fine + w * par


def apply_design(design, F, C, alpha, rule_eval):
    """design = (rule, beta, ratio); F fine-labelled, C parent-labelled (superset of F)."""
    rule, b, r = design
    if rule == 'hsc':
        return rule_eval.climb(cal_climb(F, alpha))
    if rule == 'flat':
        return rule_eval.flat(cal_flat(F, alpha))
    if rule == 'td':
        tp, tf = cal_levelwise(F, r * alpha, (1 - r) * alpha, coarse=C)
        return rule_eval.levelwise(tp, tf)
    tp, tf = cal_climb2(F, r * alpha, (1 - r) * alpha, coarse=C)
    return rule_eval.climb2(tp, tf)


def candidate_designs(betas, ratios=np.linspace(.1, .9, 9)):
    out = [('hsc', 1.0, None), ('flat', 1.0, None)]
    for rule in ['td', 'cl']:
        for b in betas:
            for r in ratios:
                out.append((rule, b, float(r)))
    return out


def plan(src, alpha, B, rho, betas, ws, sims=10, seed=0):
    """Pick the design maximising simulated utility on SOURCE held-out labels only (no target labels)."""
    rng = np.random.default_rng(seed)
    designs = candidate_designs(betas)
    score = {w: np.zeros(len(designs)) for w in ws}
    for _ in range(sims):
        perm = rng.permutation(src.n)
        E = src.sub(perm[-5000:])
        for i, d in enumerate(designs):
            nf = max(int(round(d[1] * B)), 1)
            nc = int(round((B - nf) * rho))
            F, C = src.sub(perm[:nf]), src.sub(perm[:nf + nc])
            res = apply_design(d, F, C, alpha, E)
            for w in ws:
                score[w][i] += utility(res, w)
    return {w: designs[int(np.argmax(score[w]))] for w in ws}


# ---------------------------------------------------------------- experiment driver
def load_pair(tag, model, target):
    d = np.load(PROBS / f'{tag}_{model}.npz')
    src = Scores(d['source_hold__p'], d['source_hold__y'])
    tgt = Scores(d[f'{target}__p'], d[f'{target}__y'])
    return src, tgt, d[f'{target}__xy']


WS = ['info', 0.3, 0.5, 0.7, 0.9]


def evaluate_draw(rng, src, tgt, T, pool_idx, B, rho, alpha, cfg):
    rows = []

    def rec(method, res, extra=None):
        risk, fine, par, cov = res
        rows.append({'method': method, 'risk': risk, 'fine': fine, 'parent': par, 'root': 1 - fine - par,
                     'hcov': cov, 'u50': fine + .5 * par, **(extra or {})})

    tp, tf = cal_levelwise(src, cfg['r_src'] * alpha, (1 - cfg['r_src']) * alpha)
    rec('source_levelwise', T.levelwise(tp, tf))
    perm = rng.permutation(pool_idx)
    cal = tgt.sub(perm[:int(B)])
    rec('flat_selective', T.flat(cal_flat(cal, alpha)))
    if MODE['mode'] == 'crc':
        rec('cp_lca', T.cplca(cal_cp(cal, alpha)))
    rec('hsc_climb', T.climb(cal_climb(cal, alpha)))
    tp, tf = cal_path(cal, cfg['path'], alpha)
    rec('levelwise_path_fine', T.levelwise(tp, tf))
    for w, d in cfg['plan'].items():
        nf = max(int(round(d[1] * B)), 1)
        nc = int(round((B - nf) * rho))
        F, C = tgt.sub(perm[:nf]), tgt.sub(perm[:nf + nc])  # same first labels; parent labels extend them
        tag = 'info' if w == 'info' else f'w{w}'
        rec(f'planned_{tag}', apply_design(d, F, C, alpha, T),
            {'rule': d[0], 'beta': d[1], 'ratio': d[2] if d[2] is not None else np.nan, 'n_fine': nf, 'n_coarse': nf + nc})
    for rule in ['td', 'cl']:  # fixed reference designs: beta=0.25, ratio=0.5
        nf = max(int(round(.25 * B)), 1)
        nc = int(round((B - nf) * rho))
        rec(f'mixed_{rule}_b0.25', apply_design((rule, .25, .5), tgt.sub(perm[:nf]), tgt.sub(perm[:nf + nc]), alpha, T))
    return rows


BETAS = [0.1, 0.25, 0.5, 0.75, 1.0]


def run_pair(tag, model, target, budgets, rhos, alphas, reps, design='random', mode='crc', delta=0.1,
             seed=0, test_n=20000):
    MODE['mode'], MODE['delta'] = mode, delta
    src, tgt, xy = load_pair(tag, model, target)
    rng = np.random.default_rng(seed)
    path = path_from_source(src)
    out = []
    for alpha in alphas:
        r_src = source_split(src, alpha)
        plans = {(B, rho): plan(src, alpha, B, rho, BETAS, WS, seed=seed) for B in budgets for rho in rhos}
        for rep in range(reps):
            if design == 'random':
                perm = rng.permutation(tgt.n)
                test_idx, pool_idx = perm[:test_n], perm[test_n:]
            else:  # 10 km blocks; half of the blocks form the calibration pool
                blk = (np.floor(xy[:, 0] / 1e4) * 1000 + np.floor(xy[:, 1] / 1e4)).astype(int)
                ub = np.unique(blk)
                inpool = np.isin(blk, rng.choice(ub, len(ub) // 2, replace=False))
                pool_idx = np.flatnonzero(inpool)
                rest = np.flatnonzero(~inpool)
                test_idx = rng.choice(rest, min(test_n, len(rest)), replace=False)
            T = tgt.sub(test_idx)
            for (B, rho), pl in plans.items():
                cfg = {'r_src': r_src, 'path': path, 'plan': pl}
                for r in evaluate_draw(rng, src, tgt, T, pool_idx, B, rho, alpha, cfg):
                    out.append({'tag': tag, 'model': model, 'target': target, 'design': design, 'mode': mode,
                                'alpha': alpha, 'B': B, 'rho': rho, 'rep': rep, **r})
    return out


if __name__ == '__main__':
    import concurrent.futures
    exp = sys.argv[1] if len(sys.argv) > 1 else 'main'
    risk_mode = sys.argv[2] if len(sys.argv) > 2 else 'crc'
    regions = ['frh01', 'frh02', 'frh03', 'frh04']
    models = ['mlp', 'lgbm', 'tempcnn']
    pairs = [(f'L2A2017_{s}', m, f'2017_L2A_{t}') for s, t, m in itertools.product(regions, regions, models) if s != t]
    if exp == 'year':
        pairs = [(f'L1C2017_{s}', m, t) for s in ['frh01', 'frh04'] for m in models
                 for t in ['2018_L1C_frh01', '2018_L1C_frh04', '2017_L1C_' + ('frh04' if s == 'frh01' else 'frh01')]]
    spec = {'main': ([25, 50, 100, 200, 400], [10], [0.1], 50, 'random'),
            'year': ([25, 50, 100, 200, 400], [10], [0.1], 50, 'random'),
            'alpha': ([100], [10], [0.05, 0.15, 0.2], 50, 'random'),
            'rho': ([50, 100], [1, 2, 5, 20], [0.1], 50, 'random'),
            'spatial': ([50, 100, 400], [10], [0.1], 50, 'spatial')}[exp]
    t0 = time.time()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_pair, *p, *spec, mode=risk_mode, seed=i) for i, p in enumerate(pairs)]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            rows += f.result()
            print('DONE', i + 1, len(pairs), round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_parquet(OUT / f'{exp}_{risk_mode}.parquet')
    print('SAVED', exp, risk_mode, len(rows))
