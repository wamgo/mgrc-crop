"""All numbers, tables and figures of the JSTARS manuscript (version 2 experiments, phenology groups).

Writes paper_jstars/tables/*.tex, paper_jstars/figures/*.pdf|png and results/exp/v2/paper_numbers.json.
Usage: MGRC_TAXON=pheno python paper_v2_numbers.py
"""
import json
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_v2 as A

ROOT = Path(__file__).resolve().parent
V2 = ROOT / 'results' / 'exp' / 'v2'
P = ROOT / 'paper_jstars'
FIG, TAB = P / 'figures', P / 'tables'
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'font.family': 'DejaVu Sans', 'savefig.bbox': 'tight'})
NUM = {}


def group_of(exp, ds, target):
    if ds == 'eurocrops':
        return 'EuroCropsML, cross-country'
    if exp == 'year':
        return 'BreizhCrops, cross-year' if target.startswith('2018') else 'BreizhCrops, same-year (L1C)'
    return 'BreizhCrops, cross-region'


GROUPS = ['BreizhCrops, cross-region', 'BreizhCrops, cross-year', 'BreizhCrops, same-year (L1C)',
          'EuroCropsML, cross-country']


def load_main():
    out = []
    for exp, ds in [('main', 'breizh'), ('year', 'breizh'), ('main', 'eurocrops')]:
        s = A.per_setting(A.load(exp, 'pac', ds))
        s['grp'] = [group_of(exp, ds, t) for t in s.target]
        out.append(s)
    return pd.concat(out, ignore_index=True)


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', dpi=600)
    fig.savefig(FIG / f'{name}.png', dpi=200)
    plt.close(fig)


def f3(x):
    return f'{x:.3f}'


# ------------------------------------------------------------------ main results
S = load_main()
NUM['settings'] = int(S[['tag', 'model', 'target']].drop_duplicates().shape[0])
src = S[S.method == 'source_levelwise'].copy()
src['ratio'] = src.risk / src.alpha
NUM['source'] = {g: {'viol': round(float(x.over.mean()), 3), 'ratio_mean': round(float(x.ratio.mean()), 2),
                     'ratio_max': round(float(x.ratio.max()), 2)} for g, x in src.groupby('grp')}
NUM['source_all'] = {'viol': round(float(src.over.mean()), 3), 'ratio_max': round(float(src.ratio.max()), 2)}

# Table: source calibration by transfer group
rows = []
for g in GROUPS:
    x = src[src.grp == g]
    rows.append(f'{g} & {x[["tag", "model", "target"]].drop_duplicates().shape[0]} & {x.over.mean():.2f} & '
                f'{x.ratio.mean():.2f} & {x.ratio.max():.2f} \\\\')
(TAB / 'tab_source.tex').write_text('\n'.join(rows) + '\n', encoding='utf8')

# gains of MGRC over best code-only method, with cluster bootstrap CI over target domains
piv = S.pivot_table(index=A.KEYS + ['grp'], columns='method', values='hcov')
gain = (piv['planned_info'] - piv[A.BASE].max(1)).rename('gain').reset_index()
ci = A.boot_ci(gain, by=('grp', 'alpha', 'B'))
gs = gain.groupby(['grp', 'alpha', 'B']).gain.agg(mean='mean', wins=lambda x: int((x > 1e-9).sum()),
                                                 losses=lambda x: int((x < -1e-9).sum()), n='count').reset_index()
gs = gs.merge(ci, on=['grp', 'alpha', 'B'])
gs.to_csv(V2 / 'gain_table.csv', index=False)
rows = []
for g in GROUPS:
    first = True
    for a in [0.02, 0.05, 0.1]:
        x = gs[(gs.grp == g) & (gs.alpha == a)].set_index('B')
        cells = []
        for B in [25, 50, 100, 200, 400]:
            r = x.loc[B]
            cells.append(f'{r["mean"]:+.2f}' + (f' ({int(r.wins)}/{int(r.n)})' if r.wins else ''))
        lab = g.split(', ')[1] if first else ''
        rows.append(f'{lab} & {a:g} & ' + ' & '.join(cells) + r' \\')
        first = False
    rows.append(r'\addlinespace')
(TAB / 'tab_gain.tex').write_text('\n'.join(rows[:-1]) + '\n', encoding='utf8')
NUM['gain_strict'] = {f'{g}|{a}|{B}': [round(float(r['mean']), 3), int(r.wins), int(r.n), round(float(r.lo), 3),
                                        round(float(r.hi), 3)]
                      for (g, a, B), r in gs.set_index(['grp', 'alpha', 'B']).iterrows() if a <= 0.05}

# mixing versus best single granularity (coarse-only included)
single = (piv['planned_info'] - piv[A.BASE + ['coarse_only']].max(1)).rename('g').reset_index()
NUM['vs_best_single'] = {'max': round(float(single.g.max()), 3), 'mean': round(float(single.g.mean()), 4),
                         'share_better': round(float((single.g > 1e-3).mean()), 3),
                         'share_worse': round(float((single.g < -1e-3).mean()), 3)}

# realised error of MGRC: aggregate violation per alpha, worst setting, error among labelled parcels
mg = S[S.method == 'planned_info']
NUM['mgrc_viol'] = {str(a): round(float(x.over.mean()), 3) for a, x in mg.groupby('alpha')}
NUM['mgrc_viol_max_setting'] = round(float(mg.over.max()), 3)
NUM['mgrc_ratio_max'] = round(float((mg.risk / mg.alpha).max()), 3)
NUM['sel_err_002'] = {str(B): round(float(x.sel_err.mean()), 4) for B, x in mg[mg.alpha == 0.02].groupby('B')}
NUM['answered_002'] = {str(B): round(float(x.answered.mean()), 3) for B, x in mg[mg.alpha == 0.02].groupby('B')}
codeonly = S[S.method.isin(A.BASE)].groupby(['alpha', 'B']).hcov.max()
NUM['codeonly_zero'] = {f'{a}|{B}': round(float(v), 3) for (a, B), v in codeonly.items() if a <= 0.05}

# planner choices
d_all = []
for exp, ds in [('main', 'breizh'), ('year', 'breizh'), ('main', 'eurocrops')]:
    d = A.load(exp, 'pac', ds)
    d_all.append(d[d.method == 'planned_info'][['alpha', 'B', 'rule', 'beta', 'ratio']])
pc = pd.concat(d_all)
NUM['planner'] = {f'{a}|{B}': {'hsc_or_flat': round(float(x.rule.isin(['hsc', 'flat']).mean()), 2),
                              'beta_mean': round(float(x[~x.rule.isin(['hsc', 'flat'])].beta.mean()), 2)
                              if (~x.rule.isin(['hsc', 'flat'])).any() else None}
                  for (a, B), x in pc.groupby(['alpha', 'B'])}

# ------------------------------------------------------------------ figure: coverage and violation vs budget
METH = [('source_levelwise', 'Source-calibrated', 'k', '--'), ('hsc_climb', 'HSC (codes)', 'C1', '-'),
        ('levelwise_path_fine', 'Two-threshold (codes)', 'C2', '-'), ('flat_selective', 'Flat (codes)', 'C0', '-'),
        ('coarse_only', 'Groups only', 'C7', ':'), ('planned_info', 'MGRC', 'C3', '-')]
fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.7), sharex=True, gridspec_kw={'hspace': 0.12, 'wspace': 0.25})
for j, a in enumerate([0.02, 0.05, 0.1]):
    x = S[S.alpha == a]
    for m, lab, col, ls in METH:
        y = x[x.method == m].groupby('B')
        axs[0, j].plot(y.hcov.mean().index, y.hcov.mean().values, marker='o', ms=2.5, color=col, ls=ls,
                       lw=1.6 if m == 'planned_info' else 1, label=lab)
        axs[1, j].plot(y.over.mean().index, y.over.mean().values, marker='o', ms=2.5, color=col, ls=ls,
                       lw=1.6 if m == 'planned_info' else 1)
    axs[0, j].set_title(rf'Error level $\alpha={a:g}$')
    axs[1, j].axhline(0.1, color='gray', ls=':', lw=1)
    axs[1, j].set_ylim(0, 1)
    axs[1, j].set_xscale('log')
    axs[1, j].set_xticks([25, 50, 100, 200, 400], ['25', '50', '100', '200', '400'])
    axs[1, j].set_xlabel('Budget $B$ (crop-code units)')
axs[0, 0].set_ylabel('Information-weighted\ncoverage')
axs[1, 0].set_ylabel('Share of runs with\nerror above $\\alpha$')
axs[0, 0].legend(loc='upper center', bbox_to_anchor=(1.7, 1.38), ncol=6, frameon=False)
save(fig, 'fig_budget')

# ------------------------------------------------------------------ safety margin chosen in hindsight
h = pd.concat([pd.read_parquet(V2 / f'margin_hindsight__{d}.parquet').assign(ds=d) for d in ['breizh', 'eurocrops']])
h['grp'] = [group_of(e, d, t) for e, d, t in zip(h.exp, h.ds, h.target)]
mx = h.groupby(['alpha', 'm']).ratio.max().unstack()
vi = h.groupby(['alpha', 'm']).ratio.apply(lambda r: (r > 1).mean()).unstack()
safe = {a: min(m for m in mx.columns if mx.loc[a, m] <= 1) for a in mx.index}
NUM['margin_safe'] = {str(a): float(m) for a, m in safe.items()}
NUM['margin_max_ratio'] = {f'{a}|{m}': round(float(mx.loc[a, m]), 2) for a in mx.index for m in mx.columns}
NUM['margin_viol'] = {f'{a}|{m}': round(float(vi.loc[a, m]), 3) for a in vi.index for m in vi.columns}
rows = []
for a in [0.02, 0.05, 0.1]:
    rows.append(f'{a:g} & ' + ' & '.join(f'{mx.loc[a, m]:.2f}' for m in [1.0, 1.5, 2.0, 2.5, 3.0]) + ' & '
                + ' & '.join(f'{vi.loc[a, m]:.2f}' for m in [1.0, 1.5, 2.0]) + r' \\')
(TAB / 'tab_margin.tex').write_text('\n'.join(rows) + '\n', encoding='utf8')
rows = []
for g in GROUPS:
    first = True
    for a in [0.02, 0.05, 0.1]:
        mh = h[(h.alpha == a) & (h.m == safe[a]) & (h.grp == g)].hcov.mean()
        x = S[(S.alpha == a) & (S.grp == g) & (S.method == 'planned_info')].groupby('B').hcov.mean()
        lab = g.split(', ')[1] if first else ''
        rows.append(f'{lab} & {a:g} & {safe[a]:g} & {mh:.3f} & ' + ' & '.join(
            (r'\textbf{' + f'{x[B]:.3f}' + '}' if x[B] > mh + 1e-9 else f'{x[B]:.3f}') for B in [25, 50, 100, 200, 400])
            + r' \\')
        first = False
    rows.append(r'\addlinespace')
(TAB / 'tab_margin_cov.tex').write_text('\n'.join(rows[:-1]) + '\n', encoding='utf8')

# ------------------------------------------------------------------ price ratio
rr = []
for ds in ['breizh', 'eurocrops']:
    s = A.per_setting(A.load('rho', 'pac', ds))
    s['ds'] = ds
    rr.append(s)
R = pd.concat(rr)
pr = R.pivot_table(index=A.KEYS + ['ds'], columns='method', values='hcov')
g = (pr['planned_info'] - pr[A.BASE].max(1)).rename('g').reset_index()
t = g[g.alpha == 0.02].groupby(['rho', 'B']).g.mean().unstack()
NUM['rho_002'] = {f'{int(r)}|{int(B)}': round(float(t.loc[r, B]), 3) for r in t.index for B in t.columns}
t10 = g.assign(rho=g.rho.astype(int))
rows = []
for rho in [1, 2, 5, 10, 20]:
    if rho == 10:
        x = gain[gain.alpha == 0.02].groupby('B').gain.mean()
        rows.append(f'{rho} & ' + ' & '.join(f'{x[B]:+.2f}' for B in [25, 50, 100, 200]) + r' \\')
    else:
        rows.append(f'{rho} & ' + ' & '.join(f'{t.loc[rho, B]:+.2f}' for B in [25, 50, 100, 200]) + r' \\')
(TAB / 'tab_rho.tex').write_text('\n'.join(rows) + '\n', encoding='utf8')

# ------------------------------------------------------------------ spatial, crc
for exp, mode, key in [('spatial', 'pac', 'spatial'), ('main', 'crc', 'crc')]:
    out = {}
    for ds in ['breizh', 'eurocrops']:
        s = A.per_setting(A.load(exp, mode, ds))
        gg = A.gains(s)
        for _, r in gg[gg.alpha <= 0.05].iterrows():
            out[f'{ds}|{r.alpha}|{int(r.B)}'] = [round(float(r['mean']), 3), int(r.wins), int(r.n)]
        mgx = s[s.method == 'planned_info']
        out[f'{ds}|viol_mean'] = round(float(mgx.over.mean()), 3)
        out[f'{ds}|ratio_mean_by_alpha'] = {str(a): round(float((x.risk / x.alpha).mean()), 3)
                                            for a, x in mgx.groupby('alpha')}
    NUM[key] = out

# ------------------------------------------------------------------ noise at alpha = 0.02, B = 100
rows, nz = [], {}
for ds, lab in [('breizh', 'BreizhCrops'), ('eurocrops', 'EuroCropsML')]:
    d = pd.read_parquet(V2 / f'noise_pac__{ds}.parquet')
    d = d[d.method.isin(['hsc_climb', 'mgrc_naive_info', 'mgrc_corrected_info'])].copy()
    d['over'] = (d.risk > d.alpha).astype(float)
    s = d.groupby(['tag', 'model', 'target', 'kind', 'eta', 'method'])[['over', 'hcov']].mean().reset_index()
    t = s.groupby(['kind', 'eta', 'method'])[['over', 'hcov']].mean().unstack('method')
    first = True
    for kind in ['symmetric', 'confusion', 'adversarial']:
        for eta in [0.005, 0.01, 0.02]:
            r = t.loc[(kind, eta)]
            nz[f'{ds}|{kind}|{eta}'] = {m: [round(float(r[('over', m)]), 3), round(float(r[('hcov', m)]), 3)]
                                        for m in ['mgrc_naive_info', 'mgrc_corrected_info']}
            rows.append(f'{lab if first else ""} & {kind if eta == 0.005 else ""} & {eta:g} & '
                        f'{r[("over", "mgrc_naive_info")]:.2f} & {r[("hcov", "mgrc_naive_info")]:.2f} & '
                        f'{r[("over", "mgrc_corrected_info")]:.2f} & {r[("hcov", "mgrc_corrected_info")]:.2f} \\\\')
            first = False
    rows.append(r'\midrule')
(TAB / 'tab_noise.tex').write_text('\n'.join(rows[:-1]) + '\n', encoding='utf8')
NUM['noise'] = nz

# ------------------------------------------------------------------ source-path pilot
pp = pd.read_parquet(V2 / 'pilot_path.parquet')
pp['over'] = (pp.risk > pp.alpha).astype(float)
ps = pp.groupby(['tag', 'model', 'target', 'alpha', 'B', 'method'])[['hcov', 'over']].mean().reset_index()
NUM['pilot_path'] = {f'{a}|{B}|{m}': round(float(x.hcov.mean()), 3)
                     for (a, B, m), x in ps.groupby(['alpha', 'B', 'method'])}

# ------------------------------------------------------------------ accuracy table (phenology groups)
acc = pd.read_csv(ROOT / 'results' / 'tables' / 'accuracy_all_pheno.csv')
acc['product'] = np.where(acc.source.str.startswith('L1C'), 'L1C', np.where(acc.dataset == 'eurocrops', 'EC', 'L2A'))
ga = acc.groupby(['product', 'model', 'kind'])[['fine_acc', 'group_acc']].mean()
cols = [('L2A', 'source'), ('L2A', 'cross-region'), ('L1C', 'source'), ('L1C', 'cross-year'),
        ('L1C', 'same-year region'), ('EC', 'source'), ('EC', 'cross-country')]
rows = []
for m, lab in [('mlp', 'MLP'), ('lgbm', 'LightGBM'), ('tempcnn', 'TempCNN'), ('ltae', 'L-TAE')]:
    rows.append(f'{lab} & ' + ' & '.join(f"{ga.loc[(p, m, k), 'fine_acc']:.2f}/{ga.loc[(p, m, k), 'group_acc']:.2f}"
                                         for p, k in cols) + r' \\')
(TAB / 'tab_accuracy.tex').write_text('\n'.join(rows) + '\n', encoding='utf8')

# ------------------------------------------------------------------ released-map figure
if '--skip-map' not in sys.argv:
    me = np.load(V2 / 'map_example.npz')
    ua = pd.read_csv(V2 / 'map_example_ua.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={'wspace': 0.05})
    xy = me['xy'] / 1e3
    cols4 = ['#4c72b0', '#dd8452', '#55a868', '#c4c4a0']
    names = ['winter crops', 'maize', 'spring/summer crops', 'grassland, fallow, other']
    for ax, lvl, title in [(axs[0], me['lvl_code'], r'(a) Crop codes only ($\alpha=0.02$, $B=100$)'),
                           (axs[1], me['lvl'], r'(b) MGRC ($\alpha=0.02$, $B=100$)')]:
        none = lvl == 0
        ax.scatter(xy[none, 0], xy[none, 1], s=0.15, color='0.85', rasterized=True, label='no label')
        if (lvl == 1).any():
            for gi in [3, 2, 0, 1]:
                m = (lvl == 1) & (me['node'] == gi)
                ax.scatter(xy[m, 0], xy[m, 1], s=0.15, color=cols4[gi], rasterized=True, label=names[gi])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    axs[1].legend(markerscale=12, loc='lower left', bbox_to_anchor=(1.0, 0.0), frameon=False)
    save(fig, 'fig_map')
    NUM['map_example'] = {'answered': round(float((me['lvl'] > 0).mean()), 3),
                          'answered_codes_only': round(float((me['lvl_code'] > 0).mean()), 3),
                          'ua': ua.round(4).to_dict('records')}
else:
    NUM['map_example'] = json.loads((ROOT / 'reference/paper_numbers.json').read_text(encoding='utf8'))['map_example']
    print('Map statistics copied from the archived reference; map coordinates are not redistributed.')

(V2 / 'paper_numbers.json').write_text(json.dumps(NUM, indent=1), encoding='utf8')
print(json.dumps({k: NUM[k] for k in ['settings', 'source_all', 'source', 'vs_best_single', 'mgrc_viol',
                                      'mgrc_viol_max_setting', 'mgrc_ratio_max', 'sel_err_002', 'answered_002',
                                      'margin_safe', 'spatial', 'crc', 'map_example']}, indent=1))
