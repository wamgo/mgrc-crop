"""Supplementary S2: is the binomial Clopper-Pearson certificate still valid when calibration parcels are drawn
without replacement from a finite target domain?

For a domain of N parcels with K = floor(R N) + 1 errors (true rate just above alpha, the worst case), a sample of
n parcels without replacement has k ~ Hypergeometric(N, K, n). A certificate is wrongly granted when
UCB_delta(k, n) <= alpha although K / N > alpha. We compute that probability exactly and compare it with delta.
"""
import itertools, json
import numpy as np
from scipy.stats import beta, hypergeom
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ucb(k, n, d):
    k = np.asarray(k, float)
    return np.where(k >= n, 1.0, beta.ppf(1 - d, k + 1, np.maximum(n - k, 1e-12)))


def false_certificate(N, n, alpha, d):
    """Max over true error counts K with K/N > alpha of P(UCB(k, n) <= alpha), k ~ Hypergeom(N, K, n)."""
    ks = np.arange(n + 1)
    ok = ucb(ks, n, d) <= alpha                       # sample counts that grant a certificate
    worst = 0.0
    K0 = int(np.floor(alpha * N)) + 1                   # smallest error count above alpha
    # P(certificate) decreases in K, so the worst case above alpha is the smallest K above alpha
    for K in (K0, K0 + 1):
        worst = max(worst, float(hypergeom.pmf(ks, N, K, n)[ok].sum()))
    return worst


if __name__ == '__main__':
    rows = []
    for alpha, d in [(0.02, 0.05), (0.02, 0.1), (0.05, 0.05), (0.1, 0.05), (0.1, 0.1), (0.018, 0.05), (0.09, 0.05)]:
        for N in [2000, 10000, 40000]:
            for n in [22, 50, 114, 165, 200, 400, 1000, 2000]:
                if n >= N:
                    continue
                fc = false_certificate(N, n, alpha, d)
                rows.append({'alpha': alpha, 'delta': d, 'N': N, 'n': n, 'false_certificate': fc, 'ok': fc <= d})
    worst = max(r['false_certificate'] / r['delta'] for r in rows)
    print('configurations', len(rows), 'all <= delta:', all(r['ok'] for r in rows), 'max ratio to delta', round(worst, 4))
    for r in rows:
        if r['false_certificate'] / r['delta'] > 0.9:
            print(r)
    (ROOT / 'results' / 'finite_population_check.json').write_text(json.dumps(rows, indent=1), encoding='utf8')
