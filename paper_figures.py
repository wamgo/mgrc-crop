"""Publication figures and LaTeX tables for the MGRC paper."""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_common import setup, save_json
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PFIG = ROOT / 'paper' / 'figures'
PFIG.mkdir(parents=True, exist_ok=True)
PTAB = ROOT / 'paper' / 'tables'
PTAB.mkdir(parents=True, exist_ok=True)
EXP = ROOT / 'results' / 'exp'
TAX = json.loads((ROOT / 'data/taxonomy.json').read_text(encoding='utf8'))
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'font.family': 'DejaVu Sans', 'savefig.bbox': 'tight'})
REG = {'frh01': "Côtes-d'Armor (FRH01)", 'frh02': 'Finistère (FRH02)', 'frh03': 'Ille-et-Vilaine (FRH03)',
       'frh04': 'Morbihan (FRH04)'}
KEYS = ['tag', 'model', 'target', 'alpha', 'B', 'rho']
COL = {'flat_selective': ('Flat selective', 'C0', '-'), 'hsc_climb': ('HSC climbing', 'C1', '-'),
       'levelwise_path_fine': ('Two-threshold, fine labels', 'C2', '-'),
       'source_levelwise': ('Source-calibrated', 'k', '--'), 'planned': ('MGRC (ours)', 'C3', '-')}


def study_area():
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, r in enumerate(['frh01', 'frh02', 'frh03', 'frh04']):
        xy = np.load(ROOT / 'data' / f'feat_2017_L2A_{r}.npz')['xy']
        sel = np.random.default_rng(0).choice(len(xy), 6000, replace=False)
        ax.scatter(xy[sel, 0] / 1e3, xy[sel, 1] / 1e3, s=.3, color=f'C{i}', label=REG[r], rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (km, Lambert-93)')
    ax.set_ylabel('Northing (km, Lambert-93)')
    ax.legend(markerscale=12, loc='lower left', frameon=False)
    fig.savefig(PFIG / 'fig_study_area.pdf', dpi=300)
    fig.savefig(PFIG / 'fig_study_area.png', dpi=200)
    plt.close(fig)


def load(name):
    df = pd.read_parquet(EXP / f'{name}.parquet')
    df['u0.5'] = df.u50
    for w in [0.3, 0.7, 0.9]:
        df[f'u{w}'] = df.fine + w * df.parent
    df['over'] = (df.risk > df.alpha).astype(float)
    return df


def per_setting(df):
    return df.groupby(KEYS + ['method'])[['risk', 'over', 'fine', 'parent', 'root', 'hcov', 'u0.3', 'u0.5', 'u0.7', 'u0.9']].mean().reset_index()


def planned_name(w):
    return 'planned_info' if w == 'info' else f'planned_w{w}'


UCOL = {'info': 'hcov', 0.3: 'u0.3', 0.5: 'u0.5', 0.7: 'u0.7', 0.9: 'u0.9'}
ULAB = {'info': 'information-weighted coverage', 0.3: r'$U_{0.3}$', 0.5: r'$U_{0.5}$', 0.7: r'$U_{0.7}$', 0.9: r'$U_{0.9}$'}


def fig_budget(s, name, title):
    utils = ['info', 0.9, 0.5]
    fig, axs = plt.subplots(1, 4, figsize=(8.6, 2.5), gridspec_kw={'wspace': 0.42})
    for ax, w in zip(axs, utils):
        c = UCOL[w]
        for m in ['flat_selective', 'hsc_climb', 'levelwise_path_fine', 'planned', 'source_levelwise']:
            mm = planned_name(w) if m == 'planned' else m
            x = s[s.method == mm].groupby('B')[c].mean()
            lab, col, ls = COL[m]
            ax.plot(x.index, x.values, marker='o', ms=2.5, color=col, ls=ls, lw=1.8 if m == 'planned' else 1, label=lab)
        ax.set_xscale('log')
        ax.set_xticks([25, 50, 100, 200, 400], ['25', '50', '100', '200', '400'])
        ax.set_xlabel('Budget $B$')
        ax.set_title(ULAB[w])
    axs[0].set_ylabel('Mean utility (target test)')
    ax = axs[3]
    for m in ['source_levelwise', 'hsc_climb', 'planned']:
        mm = 'planned_info' if m == 'planned' else m
        x = s[s.method == mm].groupby('B').over.mean()
        lab, col, ls = COL[m]
        ax.plot(x.index, x.values, marker='o', ms=2.5, color=col, ls=ls, label=lab)
    ax.axhline(0.1, color='gray', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xticks([25, 50, 100, 200, 400], ['25', '50', '100', '200', '400'])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Budget $B$')
    ax.set_title(r'$\Pr(R>\alpha)$ over runs')
    axs[0].legend(loc='upper left', bbox_to_anchor=(0, -0.3), ncol=5, frameon=False)
    fig.savefig(PFIG / f'fig_{name}_budget.pdf')
    fig.savefig(PFIG / f'fig_{name}_budget.png', dpi=200)
    plt.close(fig)


def gains(s, w, base='hsc_climb'):
    c = UCOL[w]
    piv = s.pivot_table(index=KEYS, columns='method', values=c)
    ref = piv[['flat_selective', 'hsc_climb', 'levelwise_path_fine']].max(1) if base == 'best' else piv[base]
    d = (piv[planned_name(w)] - ref).rename('gain').reset_index()
    out = []
    for (B, rho, alpha), x in d.groupby(['B', 'rho', 'alpha']):
        wins, losses = int((x.gain > 1e-9).sum()), int((x.gain < -1e-9).sum())
        p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
        out.append({'utility': str(w), 'B': B, 'rho': rho, 'alpha': alpha, 'mean': x.gain.mean(), 'sd': x.gain.std(),
                    'wins': wins, 'losses': losses, 'ties': len(x) - wins - losses, 'p_sign': p})
    return pd.DataFrame(out)


def fig_rho(s):
    ws = ['info', 0.9, 0.7, 0.5]
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.3), sharey=True)
    for ax, B in zip(axs, [50, 100]):
        for i, w in enumerate(ws):
            g = gains(s[s.B == B], w)
            ax.errorbar(g.rho, g['mean'], yerr=g['sd'], marker='o', ms=3, capsize=2, label=ULAB[w], color=f'C{i + 3}')
        ax.axhline(0, color='gray', lw=.8)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20], ['1', '2', '5', '10', '20'])
        ax.set_xlabel(r'Price ratio $\rho$ (parent labels per fine label)')
        ax.set_title(f'$B={B}$')
    axs[0].set_ylabel('Utility gain over HSC (mean $\\pm$ sd)')
    axs[1].legend(frameon=False)
    fig.savefig(PFIG / 'fig_rho.pdf')
    fig.savefig(PFIG / 'fig_rho.png', dpi=200)
    plt.close(fig)


def fig_composition(s, name):
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    xs, labels, centers = [], [], []
    i = 0
    for B in [25, 50, 100, 400]:
        start = i
        for m, lab in [('hsc_climb', 'HSC'), ('planned_w0.9', 'MGRC\n$w$=0.9'), ('planned_info', 'MGRC\ninfo')]:
            x = s[(s.method == m) & (s.B == B)][['fine', 'parent', 'root']].mean()
            ax.bar(i, x.fine, color='C2', width=.8)
            ax.bar(i, x.parent, bottom=x.fine, color='C1', width=.8)
            ax.bar(i, x.root, bottom=x.fine + x.parent, color='lightgray', width=.8)
            xs.append(i)
            labels.append(lab)
            i += 1
        centers.append(((start + i - 1) / 2, B))
        i += .7
    ax.set_xticks(xs, labels, fontsize=6)
    for c, B in centers:
        ax.text(c, 1.03, f'$B={B}$', ha='center', fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Share of target parcels')
    ax.legend(['fine answer', 'parent answer', 'root (no answer)'], ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.25), frameon=False)
    fig.savefig(PFIG / f'fig_{name}_composition.pdf')
    fig.savefig(PFIG / f'fig_{name}_composition.png', dpi=200)
    plt.close(fig)


def fig_shift(s):
    """Realised error of source-calibrated rule vs target-calibrated rules, per transfer (main, B=100)."""
    x = s[(s.B == 100)]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    data = [x[x.method == m].risk.values for m in ['source_levelwise', 'hsc_climb', 'levelwise_path_fine', 'planned_info']]
    ax.boxplot(data, widths=.5)
    ax.set_xticks([1, 2, 3, 4], ['Source-\ncalibrated', 'HSC', 'Two-thr.\nfine', 'MGRC\ninfo'])
    ax.axhline(.1, color='C3', ls=':', lw=1)
    ax.set_ylabel('Realised hierarchical error')
    fig.savefig(PFIG / 'fig_shift.pdf')
    fig.savefig(PFIG / 'fig_shift.png', dpi=200)
    plt.close(fig)


def tex_main_table(s, name, budgets=(25, 50, 100, 400)):
    rows = []
    meths = [('source_levelwise', 'Source-calibrated'), ('flat_selective', 'Flat selective'), ('hsc_climb', 'HSC climbing'),
             ('levelwise_path_fine', 'Two-threshold (fine)'), ('planned_w0.5', r'MGRC, $w=0.5$'),
             ('planned_w0.9', r'MGRC, $w=0.9$'), ('planned_info', 'MGRC, info')]
    for B in budgets:
        for m, lab in meths:
            x = s[(s.method == m) & (s.B == B)]
            rows.append(f'{B if m == "source_levelwise" else ""} & {lab} & {x.risk.mean():.3f} & {x.over.mean():.3f} & '
                        f'{x.fine.mean():.3f} & {x.parent.mean():.3f} & {x["u0.5"].mean():.3f} & {x["u0.9"].mean():.3f} & {x.hcov.mean():.3f} \\\\')
        rows.append(r'\midrule')
    body = '\n'.join(rows[:-1])
    (PTAB / f'tab_{name}.tex').write_text(body + '\n', encoding='utf8')


def tex_gain_table(s, name, base='hsc_climb'):
    lines = []
    for w in ['info', 0.9, 0.7, 0.5, 0.3]:
        g = gains(s, w, base)
        cells = []
        for _, r in g.sort_values('B').iterrows():
            star = '$^{*}$' if r.p_sign < 0.05 and r['mean'] > 0 else ''
            cells.append(f"{r['mean']:+.3f}{star} ({int(r.wins)}/{int(r.ties)}/{int(r.losses)})")
        lines.append(f'{ULAB[w] if w != "info" else "Info. coverage"} & ' + ' & '.join(cells) + r' \\')
    suffix = '' if base == 'hsc_climb' else '_best'
    (PTAB / f'tab_{name}_gain{suffix}.tex').write_text('\n'.join(lines) + '\n', encoding='utf8')


if __name__ == '__main__':
    if '--skip-study-area' not in sys.argv:
        study_area()
    report = {}
    for name in ['main_pac', 'year_pac', 'spatial_pac', 'rho_pac', 'alpha_pac', 'main_crc', 'year_crc']:
        f = EXP / f'{name}.parquet'
        if not f.exists():
            print('missing', name)
            continue
        s = per_setting(load(name))
        if name in ('main_pac', 'year_pac', 'main_crc', 'year_crc', 'spatial_pac'):
            fig_budget(s, name, {'main_pac': '12 cross-region transfers x 3 models, high-probability version',
                                 'year_pac': 'L1C: cross-year transfers and same-year controls',
                                 'spatial_pac': 'Spatially blocked calibration', 'main_crc': 'Expectation version',
                                 'year_crc': 'L1C transfers, expectation version'}[name])
            tex_main_table(s, name, budgets=sorted(s.B.unique()))
            tex_gain_table(s, name)
            tex_gain_table(s, name, 'best')
            fig_composition(s, name) if name == 'main_pac' else None
        if name == 'main_pac':
            fig_shift(s)
        if name == 'rho_pac':
            fig_rho(s)
        allg = pd.concat([gains(s, w).assign(base='hsc') for w in UCOL] + [gains(s, w, 'best').assign(base='best') for w in UCOL])
        allg.to_csv(PTAB / f'gains_{name}.csv', index=False)
        viol = s.groupby(['method', 'B', 'rho', 'alpha']).agg(viol=('over', 'mean'), viol_max=('over', 'max'),
                                                            risk=('risk', 'mean')).reset_index()
        viol.to_csv(PTAB / f'violation_{name}.csv', index=False)
        report[name] = {'gains': allg.round(4).to_dict('records'), 'violation': viol.round(4).to_dict('records')}
        print('done', name)
    save_json(ROOT / 'results' / 'paper_numbers.json', report)
