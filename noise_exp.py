"""Parent-label noise experiment: naive MGRC vs noise-corrected MGRC.

Photo-interpreted parent labels z may differ from the true group pi(y). For the coarse term,
1[non-root, g_hat != pi(y)] <= 1[non-root, g_hat != z] + 1[z != pi(y)], so the true coarse risk
is at most the noisy coarse risk plus the interpretation error rate eta. Calibrating the coarse
term at level gamma*alpha - eta_bar, with eta_bar >= eta a bound known before calibration (e.g.
from an independent audit with its failure probability accounted for). The noisy-label theorem
requires identically distributed interpreted coarse labels; this script pools clean and noisy
labels and therefore evaluates the correction empirically. Fine-labelled
parcels carry field-checked parent labels (noise-free). This file implements:

  * three noise models applied only to the extra parent-only parcels: symmetric, confusion-matrix,
    adversarial (interpreter copies the model's own parent prediction when it is wrong);
  * naive calibration (uses noisy labels, no correction) and corrected calibration (level shifted
    by eta_bar);
  * a source-side planner that simulates the noise and the correction;
  * the standard target-side evaluation protocol.
"""
from pathlib import Path
import sys, itertools, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_common import setup
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, pandas as pd
import experiments as E
import hcrc

G = int(E.PARENT.max() + 1)
OUT = ROOT / 'results' / 'exp'


def parent_pred(p):
    """Top-down parent prediction g_hat = argmax_g P_g."""
    return hcrc.parent_probs(p, E.PARENT).argmax(1)


def source_confusion(src_y, src_g):
    conf = np.zeros((G, G), float)
    np.add.at(conf, (E.PARENT[src_y], src_g), 1)
    row = conf.sum(1, keepdims=True)
    return conf / np.maximum(row, 1)


def corrupt(rng, ytrue, ghat, eta, kind, conf):
    """Return a possibly noisy parent label z for every parcel."""
    n = len(ytrue)
    z = ytrue.copy()
    if n == 0:
        return z
    if kind == 'adversarial':
        wrong = np.flatnonzero(ghat != ytrue)
        if len(wrong) == 0:
            return z
        prob = min(1.0, eta * n / len(wrong))
        flip = wrong[rng.random(len(wrong)) < prob]
        z[flip] = ghat[flip]
        return z
    mask = rng.random(n) < eta
    idx = np.flatnonzero(mask)
    for i in idx:
        if kind == 'symmetric':
            choices = [k for k in range(G) if k != ytrue[i]]
            z[i] = rng.choice(choices)
        else:  # confusion
            row = conf[ytrue[i]].copy()
            row[ytrue[i]] = 0.0
            total = row.sum()
            z[i] = rng.choice(G, p=row / total) if total > 0 else rng.choice([k for k in range(G) if k != ytrue[i]])
    return z


def calibrate_with_noise(fine_sg, fine_ep, coarse_sg, coarse_ep_noisy, fine_sf, fine_ef, alpha, r, eta_bar):
    """Two-threshold calibration under coarse-label noise. Shift the coarse level by eta_bar."""
    ac = r * alpha - eta_bar
    af = (1 - r) * alpha
    sg = np.r_[fine_sg, coarse_sg]
    ep = np.r_[fine_ep, coarse_ep_noisy]
    tp = E.thr(sg, ep, ac, 0.5) if ac > 0 else E.GRID[-1]
    tf = E.thr(fine_sf, fine_ef, af, 0.5)
    return tp, max(tf, tp)


class Env:
    """Everything needed to apply a design on a Scores subset with coarse-label noise."""

    def __init__(self, scores, parent_pred_full):
        self.s = scores
        self.gtd = parent_pred_full

    def apply(self, rng, perm, nf, nc, alpha, r, eta, kind, conf, eta_bar):
        fine = self.s.sub(perm[:nf])
        coarse_idx = perm[nf:nf + nc]
        g_coarse = self.gtd[coarse_idx]
        y_coarse = self.s.y[coarse_idx]
        z = corrupt(rng, E.PARENT[y_coarse], g_coarse, eta, kind, conf)
        coarse_ep = g_coarse != z
        tp, tf = calibrate_with_noise(fine.sg, fine.ep, self.s.sg[coarse_idx], coarse_ep,
                                      fine.sf, fine.ef, alpha, r, eta_bar)
        return tp, tf


def plan_noisy(rng, src_env, alpha, B, rho, eta, kind, conf, eta_bar, ws, sims=8):
    designs = [('hsc', 1.0, None)] + [('td', b, float(r)) for b in [0.1, 0.25, 0.5, 0.75, 1.0]
                                       for r in np.linspace(.1, .9, 9)]
    score = {w: np.zeros(len(designs)) for w in ws}
    src = src_env.s
    for _ in range(sims):
        perm = rng.permutation(src.n)
        Ev = src.sub(perm[-5000:])
        for i, d in enumerate(designs):
            if d[0] == 'hsc':
                nf = int(B)
                res = Ev.climb(E.cal_climb(src.sub(perm[:nf]), alpha))
            else:
                nf = max(int(round(d[1] * B)), 1)
                nc = int(round((B - nf) * rho))
                tp, tf = src_env.apply(rng, perm, nf, nc, alpha, d[2], eta, kind, conf, eta_bar)
                res = Ev.levelwise(tp, tf)
            for w in ws:
                score[w][i] += E.utility(res, w)
    return {w: designs[int(np.argmax(score[w]))] for w in ws}


def run(tag, model, target, etas, kinds, reps=30, B=100, alpha=0.1, rho=10, seed=0, test_n=20000):
    E.MODE['mode'], E.MODE['delta'] = 'pac', 0.1
    src, tgt, _ = E.load_pair(tag, model, target)
    probs = np.load(E.PROBS / f'{tag}_{model}.npz')
    src_g = parent_pred(probs['source_hold__p'].astype(np.float64))
    tgt_g = parent_pred(probs[f'{target}__p'].astype(np.float64))
    conf = source_confusion(src.y, src_g)
    src_env = Env(src, src_g)
    tgt_env = Env(tgt, tgt_g)
    rng = np.random.default_rng(seed)
    ws = ['info', 0.9]
    rows = []
    for kind, eta in itertools.product(kinds, etas):
        plans_naive = plan_noisy(rng, src_env, alpha, B, rho, eta, kind, conf, 0.0, ws)
        plans_corr = plan_noisy(rng, src_env, alpha, B, rho, eta, kind, conf, eta, ws)
        for rep in range(reps):
            perm = rng.permutation(tgt.n)
            T = tgt.sub(perm[:test_n])
            pool = perm[test_n:]
            cal = rng.permutation(pool)
            base = {'tag': tag, 'model': model, 'target': target, 'kind': kind, 'eta': eta, 'B': B, 'rep': rep}
            res = T.climb(E.cal_climb(tgt.sub(cal[:B]), alpha))
            rows.append({**base, 'method': 'hsc_climb', 'risk': res[0], 'fine': res[1], 'parent': res[2],
                         'root': 1 - res[1] - res[2], 'hcov': res[3], 'u0.9': res[1] + .9 * res[2],
                         'rule': 'hsc', 'beta': 1.0, 'ratio': np.nan})
            for corr, plan in [(False, plans_naive), (True, plans_corr)]:
                for w, dsg in plan.items():
                    rule, b, r = dsg
                    if rule == 'hsc':
                        res = T.climb(E.cal_climb(tgt.sub(cal[:B]), alpha))
                    else:
                        nf = max(int(round(b * B)), 1)
                        nc = int(round((B - nf) * rho))
                        tp, tf = tgt_env.apply(rng, cal, nf, nc, alpha, r, eta, kind, conf, eta if corr else 0.0)
                        res = T.levelwise(tp, tf)
                    tag_name = 'info' if w == 'info' else f'w{w}'
                    rows.append({**base, 'method': f"mgrc_{'corrected' if corr else 'naive'}_{tag_name}",
                                 'risk': res[0], 'fine': res[1], 'parent': res[2], 'root': 1 - res[1] - res[2],
                                 'hcov': res[3], 'u0.9': res[1] + .9 * res[2], 'rule': rule, 'beta': b,
                                 'ratio': r if r is not None else np.nan})
    return rows


if __name__ == '__main__':
    import concurrent.futures
    regions = ['frh01', 'frh02', 'frh03', 'frh04']
    models = ['mlp', 'lgbm', 'tempcnn']
    pairs = [(f'L2A2017_{s}', m, f'2017_L2A_{t}') for s, t, m in itertools.product(regions, regions, models) if s != t]
    etas = [0.02, 0.05, 0.1, 0.2]
    kinds = ['symmetric', 'confusion', 'adversarial']
    t0 = time.time()
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run, *p, etas, kinds, reps=30, seed=200 + i) for i, p in enumerate(pairs)]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            all_rows += f.result()
            print('DONE', i + 1, len(pairs), round(time.time() - t0), flush=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT / 'noise_pac.parquet')
    print('SAVED noise', len(df))
