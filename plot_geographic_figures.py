"""Regenerate the two geographic display figures; upstream parcel data are required.

Only plotting resolution changes from the original manuscript scripts. This script does
not distribute coordinates and does not change scientific results.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTF = ROOT / 'paper_jstars' / 'figures'
V2 = ROOT / 'results' / 'exp' / 'v2'
PAGEW = 7.16

def save(fig, name):
    fig.savefig(OUTF / f'{name}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def study_area():
    fig, axs = plt.subplots(1, 2, figsize=(PAGEW, 2.7), gridspec_kw={'width_ratios': [1.25, 1]})
    rng = np.random.default_rng(0)
    names = {'frh01': "FRH01 Côtes-d'Armor", 'frh02': 'FRH02 Finistère', 'frh03': 'FRH03 Ille-et-Vilaine',
             'frh04': 'FRH04 Morbihan'}
    ax = axs[0]
    for i, r in enumerate(names):
        xy = np.load(ROOT / 'data' / f'feat_2017_L2A_{r}.npz')['xy']
        s = rng.choice(len(xy), 5000, replace=False)
        ax.scatter(xy[s, 0] / 1e3, xy[s, 1] / 1e3, s=.2, color=f'C{i}', label=names[r], rasterized=True)
    ax.set_xlabel('Easting (km, Lambert-93)')
    ax.set_ylabel('Northing (km)')
    ax.set_title('(a) BreizhCrops, Brittany')
    ax.legend(markerscale=10, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2)
    ax.set_aspect('equal')
    ax = axs[1]
    doms = json.loads((ROOT / 'data' / 'eurocrops_domains.json').read_text(encoding='utf8'))
    regs = sorted(d for d in doms if len(d) > 2)
    for i, r in enumerate(regs):
        xy = np.load(ROOT / 'data' / f'feat_2021_EC_{r}.npz')['xy']
        s = rng.choice(len(xy), min(4000, len(xy)), replace=False)
        ax.scatter(xy[s, 0] / 1e3, xy[s, 1] / 1e3, s=.2, color=plt.cm.tab10(i), label=r, rasterized=True)
    ax.set_xlabel('Easting (km, UTM 35N)')
    ax.set_title('(b) EuroCropsML target regions')
    ax.legend(markerscale=10, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=4)
    ax.set_aspect('equal')
    save(fig, 'fig_study_area')


def map_figure():
    NUM = {}
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path, default=OUTF)
    ap.add_argument('--only', choices=['study-area','map','all'], default='all')
    args = ap.parse_args()
    ROOT = args.data_root.resolve()
    OUTF = args.output.resolve()
    V2 = ROOT / 'results' / 'exp' / 'v2'
    OUTF.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'legend.fontsize':6,
                         'xtick.labelsize':6,'ytick.labelsize':6,'font.family':'DejaVu Sans'})
    if args.only in ['study-area','all']: study_area()
    if args.only in ['map','all']: map_figure()
    print('Geographic PDFs rendered at 600 dpi:', OUTF)
