"""Supplementary Table S3: register groups versus phenology groups (cross-region BreizhCrops, cross-country EuroCropsML)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
t = pd.read_csv(ROOT / 'results' / 'exp' / 'v2_register' / 'register_vs_pheno.csv')
lines = []
for ds, lab in [('breizh', 'BreizhCrops (12 RPG groups)'), ('eurocrops', 'EuroCropsML (13 HCAT groups)')]:
    first = True
    for a in [0.02, 0.1]:
        x = t[(t.ds == ds) & (t.alpha == a)].set_index('B')
        reg = ' & '.join(f'{x.loc[B, "reg_gain"]:+.2f}' for B in [25, 50, 100, 200, 400])
        phe = ' & '.join(f'{x.loc[B, "pheno_gain"]:+.2f}' for B in [25, 50, 100, 200, 400])
        lines.append(f'{lab if first else ""} & {a:g} & register & {reg} \\\\')
        lines.append(f' & & phenology & {phe} \\\\')
        first = False
    lines.append(r'\addlinespace')
(ROOT / 'paper_jstars' / 'tables' / 'supp_register.tex').write_text('\n'.join(lines[:-1]) + '\n', encoding='utf8')
print('ok')
