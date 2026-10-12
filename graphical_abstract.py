"""Graphical abstract for the MGRC paper (Elsevier: 531 x 1328 px min, readable at 5 x 13 cm)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_common import setup
ROOT = setup('direction_2_crop_hierarchy')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import paper_figures as P

s = P.per_setting(P.load('main_pac'))
fig = plt.figure(figsize=(13 / 2.54 * 2, 5 / 2.54 * 2))  # 26 x 10 cm, keeps 13:5 ratio
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 26)
ax.set_ylim(0, 10)
ax.axis('off')


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15', fc=fc, ec='0.3', lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9.5, wrap=True)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=14, lw=1.3, color='0.25'))


box(0.4, 6.2, 5.2, 2.6, 'Source region\ncrop classifier\n(23 crop codes)', '#e8eef7')
box(0.4, 1.2, 5.2, 3.6, 'Target region / year\n\nmany cheap group labels\n(cereal, maize, grassland)\n+ few crop-code labels', '#fdf0e2')
arrow(5.8, 7.5, 7.4, 6.3)
arrow(5.8, 3.0, 7.4, 4.3)
box(7.6, 2.3, 7.0, 5.4, 'Exact split of the\nhierarchical error\n\n' + r'$\ell=\ell_c(t_p)+\ell_f(t_f)$' +
    '\n\n$t_p$: group labels\n$t_f$: code labels\n\nexpectation or high-probability\nbound, robust to known\ngroup-label error', '#eaf5ea')
arrow(14.8, 5.0, 16.0, 5.0)

ins = fig.add_axes([0.70, 0.17, 0.285, 0.72])
for m, lab, c, lw in [('hsc_climb', 'Code labels only (HSC)', 'C1', 1.4), ('planned_info', 'Mixed labels (MGRC)', 'C3', 2.4)]:
    x = s[s.method == m].groupby('B').hcov.mean()
    ins.plot(x.index, x.values, 'o-', color=c, lw=lw, ms=4, label=lab)
ins.set_xscale('log')
ins.set_xticks([25, 50, 100, 200, 400], ['25', '50', '100', '200', '400'])
ins.set_xlabel('Labelling budget (code-label units)', fontsize=8.5)
ins.set_ylabel('Map informativeness', fontsize=8.5)
ins.set_title('Nominal error level ($\\alpha$=0.1), 36 cross-region settings', fontsize=8.5)
ins.tick_params(labelsize=8)
ins.legend(fontsize=8, frameon=False, loc='lower right')
out = ROOT / 'paper' / 'graphical_abstract'
fig.savefig(str(out) + '.png', dpi=300)
fig.savefig(str(out) + '.pdf')
print('saved', out)
