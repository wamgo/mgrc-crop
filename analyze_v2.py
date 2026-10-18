"""Summaries of the version-2 experiments (results/exp/v2) for the JSTARS manuscript.

Unit of replication: a transfer (source domain -> target domain) with one classifier. Repetitions within a setting
share the classifier and the target domain, so per-run frequencies are summarised per setting first.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
V2 = ROOT / 'results' / 'exp' / 'v2'
KEYS = ['tag', 'model', 'target', 'alpha', 'B', 'rho']
BASE = ['flat_selective', 'hsc_climb', 'levelwise_path_fine']


def load(exp, mode, dataset):
    d = pd.read_parquet(V2 / f'{exp}_{mode}__{dataset}.parquet')
    d['over'] = (d.risk > d.alpha).astype(float)
    d['answered'] = d.fine + d.parent
    d['sel_err'] = np.where(d.answered > 0, d.risk / d.answered.where(d.answered > 0, 1), np.nan)
    return d


def per_setting(d):
    cols = ['risk', 'over', 'fine', 'parent', 'hcov', 'answered', 'sel_err']
    return d.groupby(KEYS + ['method'])[cols].mean().reset_index()


def gains(s, method='planned_info', metric='hcov', base='best'):
    piv = s.pivot_table(index=KEYS, columns='method', values=metric)
    ref = piv[BASE].max(1) if base == 'best' else piv[base]
    g = (piv[method] - ref).rename('gain').reset_index()
    out = g.groupby(['alpha', 'B', 'rho']).gain.agg(
        mean='mean', wins=lambda x: int((x > 1e-9).sum()), ties=lambda x: int((x.abs() <= 1e-9).sum()),
        losses=lambda x: int((x < -1e-9).sum()), n='count').reset_index()
    return out


def boot_ci(g, by=('alpha', 'B', 'rho'), unit='target', reps=2000, seed=0):
    """Cluster bootstrap over target domains of the mean paired gain."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, x in g.groupby(list(by)):
        units = x[unit].unique()
        m = x.groupby(unit).gain.mean().reindex(units).values
        bs = rng.choice(m, (reps, len(m))).mean(1)
        rows.append(dict(zip(by, k), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 220, 'display.max_rows', 300)
    exp, mode, ds = sys.argv[1:4]
    d = load(exp, mode, ds)
    s = per_setting(d)
    print('settings', s[['tag', 'model', 'target']].drop_duplicates().shape[0], 'models', sorted(d.model.unique()))
    show = ['source_levelwise', 'source_margin', 'flat_selective', 'hsc_climb', 'levelwise_path_fine', 'coarse_only',
            'planned_info', 'planned_w0.9']
    print(s[s.method.isin(show)].groupby(['alpha', 'B', 'method'])[['risk', 'over', 'fine', 'parent', 'hcov', 'sel_err']]
          .mean().round(3).to_string())
    g = gains(s)
    piv = s.pivot_table(index=KEYS, columns='method', values='hcov')
    gg = (piv['planned_info'] - piv[BASE].max(1)).rename('gain').reset_index()
    print(g.merge(boot_ci(gg), on=['alpha', 'B', 'rho']).round(3).to_string())
