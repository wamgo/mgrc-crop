"""Supplementary tables for the JSTARS version: phenology taxonomies with parcel counts, and the source-path pilot."""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
TAB = ROOT / 'paper_jstars' / 'tables'
TAB.mkdir(parents=True, exist_ok=True)
BZ_NAMES = {'BTH': 'Winter soft wheat', 'MIE': 'Silage maize', 'MIS': 'Grain maize', 'ORH': 'Winter barley',
            'ORP': 'Spring barley', 'TTH': 'Winter triticale', 'AVH': 'Winter oats', 'MCR': 'Mixed cereals',
            'SRS': 'Buckwheat', 'CZH': 'Winter rapeseed', 'J6S': 'Fallow, 6 years or more',
            'J5M': 'Fallow, 5 years or less', 'FET': 'Fescue', 'BVF': 'Fodder beet', 'PPH': 'Permanent grassland',
            'PRL': 'Long-rotation grassland', 'PTR': 'Temporary grassland', 'RGA': 'Italian ryegrass',
            'PTC': 'Potatoes', 'FLA': 'Other vegetables and flowers', 'SNE': 'Temporarily non-cultivated area',
            'BTA': 'Buffer strip', 'VRG': 'Orchards'}


def tex(s):
    return s.replace('_', ' ').replace('&', r'\&')


def counts(files):
    ys = []
    for f in files:
        d = np.load(f)
        ys += [d[k] for k in d.files if k.endswith('__y')]
    return np.bincount(np.concatenate(ys))


def table(tax_file, prob_files, names, out):
    t = json.loads((ROOT / 'data' / tax_file).read_text(encoding='utf8'))
    P = np.array(t['parent_of_fine'])
    key = 'eurocrops' if 'eurocrops' in tax_file else 'breizh'
    archive = ROOT / 'data' / 'supplementary_class_counts.json'
    c = np.array(json.loads(archive.read_text(encoding='utf8'))[key]) if archive.exists() else counts(prob_files)
    lines = []
    for g, gname in enumerate(t['parents']):
        first = True
        for k in np.flatnonzero(P == g):
            code = t['fine'][k]
            lines.append(f"{tex(gname) if first else ''} & {code} & {tex(names(k, code))} & {int(c[k]):,}".replace(',', '{,}')
                         + r' \\')
            first = False
        lines.append(r'\addlinespace')
    (TAB / out).write_text('\n'.join(lines[:-1]) + '\n', encoding='utf8')


if __name__ == '__main__':
    pr = ROOT / 'results' / 'probs'
    table('taxonomy_pheno.json', [pr / 'L2A2017_frh01_ltae.npz'], lambda k, code: BZ_NAMES[code], 'supp_tax_bz.tex')
    ec = json.loads((ROOT / 'data' / 'taxonomy_eurocrops_pheno.json').read_text(encoding='utf8'))
    table('taxonomy_eurocrops_pheno.json', [pr / 'EC2021_LV_ltae.npz', pr / 'EC2021_EE_ltae.npz'],
          lambda k, code: ec['fine_names'][k], 'supp_tax_ec.tex')
    # source-path pilot
    d = pd.read_parquet(ROOT / 'results' / 'exp' / 'v2' / 'pilot_path.parquet')
    s = d.groupby(['tag', 'model', 'target', 'alpha', 'B', 'method']).hcov.mean().reset_index()
    t = s.groupby(['alpha', 'B', 'method']).hcov.mean().unstack('method')
    lines = []
    for (a, B), r in t.iterrows():
        lines.append(f'{a:g} & {B} & {r.margin25:.3f} & {r.path_fine:.3f} & {r.path_mixed:.3f} & {r.mgrc:.3f} \\\\')
    (TAB / 'supp_path.tex').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('ok')
