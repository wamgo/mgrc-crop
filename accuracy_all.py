"""Accuracy and calibration of every source model, per dataset, with the right taxonomy for each.

Writes results/tables/accuracy_all_pheno.csv (one row per model file and domain) and prints the summary used in
the paper: fine and group accuracy and ECE on the held-out source pool and on the transfer targets.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
TAX = {'breizh': json.loads((ROOT / 'data/taxonomy_pheno.json').read_text(encoding='utf8')),
       'eurocrops': json.loads((ROOT / 'data/taxonomy_eurocrops_pheno.json').read_text(encoding='utf8'))}


def ece(p, y, bins=15):
    conf = p.max(1)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    ok = p.argmax(1) == y
    return sum(abs(ok[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def kind(tag, dom):
    if dom == 'source_hold':
        return 'source'
    if tag.startswith('EC'):
        return 'cross-country'
    if tag.startswith('L1C'):
        return 'cross-year' if dom.startswith('2018') else 'same-year region'
    return 'cross-region'


rows = []
for f in sorted((ROOT / 'results/probs').glob('*.npz')):
    tag, model = f.stem.rsplit('_', 1)
    ds = 'eurocrops' if tag.startswith('EC') else 'breizh'
    parent = np.array(TAX[ds]['parent_of_fine'])
    d = np.load(f)
    for k in d.files:
        if not k.endswith('__p'):
            continue
        dom = k[:-3]
        p, y = d[k].astype(np.float64), d[dom + '__y']
        pp = np.stack([p[:, parent == g].sum(1) for g in range(parent.max() + 1)], 1)
        rows.append({'dataset': ds, 'source': tag, 'model': model, 'domain': dom, 'kind': kind(tag, dom), 'n': len(y),
                     'fine_acc': (p.argmax(1) == y).mean(), 'group_acc': (pp.argmax(1) == parent[y]).mean(),
                     'ece': ece(p, y)})
acc = pd.DataFrame(rows)
(ROOT / 'results' / 'tables').mkdir(parents=True, exist_ok=True)
acc.to_csv(ROOT / 'results' / 'tables' / 'accuracy_all_pheno.csv', index=False)
acc['product'] = np.where(acc.source.str.startswith('L1C'), 'L1C', np.where(acc.dataset == 'eurocrops', 'EC', 'L2A'))
pd.set_option('display.width', 200)
print(acc.groupby(['product', 'model', 'kind'])[['fine_acc', 'group_acc', 'ece']].mean().round(3).unstack('kind').to_string())
