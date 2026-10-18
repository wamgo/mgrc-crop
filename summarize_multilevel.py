"""Summaries of the three-level and WorldCereal experiments (results/exp/multilevel/*.parquet)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

D = Path(__file__).resolve().parent / 'results' / 'exp' / 'multilevel'
KEY = ['tag', 'model', 'target', 'B']


def per_setting(df, prices):
    return df.groupby(KEY + prices + ['method'])[['risk', 'hcov', 'l1', 'l2', 'fine', 'root', 'over']].mean().reset_index()


def table(df, prices):
    s = per_setting(df, prices)
    out = s.groupby(['B'] + prices + ['method'])[['risk', 'over', 'hcov', 'l1', 'l2', 'fine']].mean().round(3)
    return out


def gain(df, prices, a, b):
    s = per_setting(df, prices)
    piv = s.pivot_table(index=KEY + prices, columns='method', values='hcov')
    ref = piv[b].max(1) if isinstance(b, list) else piv[b]
    d = (piv[a] - ref).rename('g').reset_index()
    rows = []
    for k, x in d.groupby(['B'] + prices):
        w, l = int((x.g > 1e-9).sum()), int((x.g < -1e-9).sum())
        rows.append({**dict(zip(['B'] + prices, k if isinstance(k, tuple) else (k,))), 'mean': round(x.g.mean(), 4),
                     'wins': w, 'ties': len(x) - w - l, 'losses': l, 'p': binomtest(w, w + l).pvalue if w + l else 1.0})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 200, 'display.max_rows', 500)
    for f in sorted(D.glob('*.parquet')):
        df = pd.read_parquet(f)
        df['over'] = (df.risk > df.alpha).astype(float)
        prices = ['rho2', 'rho1'] if 'rho1' in df else ['rho2']
        print('=' * 20, f.name, len(df))
        src = df[df.method == 'source3'] if 'source3' in set(df.method) else None
        if src is not None:
            print('source3 violation', round(src.over.mean(), 3), 'risk', round(src.risk.mean(), 3))
            df = df[df.method != 'source3']
        print(table(df, prices))
        if f.name.startswith('main'):
            print(gain(df, prices, 'mgrc3', 'mgrc2'))
            print(gain(df, prices, 'mgrc3', ['flat', 'hsc3', 'td3_fine']))
            print(gain(df, prices, 'mgrc2', ['flat', 'hsc3', 'td3_fine']))
        else:
            for v in ['wc_naive', 'wc_eta', 'wc_shared', 'free_clean']:
                print(v, 'vs mgrc2')
                print(gain(df, prices, v, 'mgrc2'))
            print(df.groupby(['tag', 'target'])[['eta', 'eta_shared', 'z_missing', 'acc_l1_model']].mean().round(4))
