"""How large a safety margin would source calibration need, and what coverage would it leave?

For every transfer and error level, thresholds are fitted on the held-out source pool at alpha / m (m = margin
factor), with the same alpha split as the source-calibrated baseline, and applied to all parcels of the target
domain. No target label is used. Output: realised risk / alpha and information-weighted coverage per (pair, alpha, m).

Usage: MGRC_TAXON=pheno [MGRC_DATASET=eurocrops] python margin_hindsight.py <breizh|eurocrops>
"""
import sys
import numpy as np, pandas as pd
import experiments as E
import run_v2 as R

ALPHAS = [0.02, 0.05, 0.1]
MARGINS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]


def one(tag, model, target):
    src, tgt, _ = E.load_pair(tag, model, target)
    rows = []
    for a in ALPHAS:
        r = E.source_split(src, a)
        for m in MARGINS:
            tp, tf = E.cal_levelwise(src, r * a / m, (1 - r) * a / m)
            risk, fine, par, cov = tgt.levelwise(tp, tf)
            rows.append({'tag': tag, 'model': model, 'target': target, 'alpha': a, 'm': m,
                         'risk': risk, 'ratio': risk / a, 'hcov': cov, 'fine': fine, 'parent': par})
    return rows


if __name__ == '__main__':
    ds = sys.argv[1]
    E.MODE['mode'], E.MODE['delta'] = 'pac', 0.1
    exps = ['main', 'year'] if ds == 'breizh' else ['main']
    rows = []
    for exp in exps:
        for p in R.pairs(ds, exp):
            for row in one(*p):
                row['exp'] = exp
                rows.append(row)
    out = E.ROOT / 'results' / 'exp' / 'v2' / f'margin_hindsight__{ds}.parquet'
    pd.DataFrame(rows).to_parquet(out)
    print('saved', out.name, len(rows))
