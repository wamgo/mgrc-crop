"""Descriptive tables for the paper: data per region/year, model accuracy in and out of domain."""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_common import setup, save_json
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, pandas as pd

TAX = json.loads((ROOT / 'data/taxonomy.json').read_text(encoding='utf8'))
PARENT = np.array(TAX['parent_of_fine'])
TAB = ROOT / 'results' / 'tables'
TAB.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted((ROOT / 'data').glob('feat_*.npz')):
    d = np.load(f)
    _, year, level, region = f.stem.split('_')
    c = np.bincount(d['y'], minlength=len(PARENT))
    rows.append({'year': year, 'level': level, 'region': region, 'parcels': len(d['y']),
                 **{TAX['fine'][k]: int(c[k]) for k in range(len(c))}})
data = pd.DataFrame(rows)
data.to_csv(TAB / 'data_counts.csv', index=False)
print(data[['year', 'level', 'region', 'parcels']].to_string())

tax_rows = [{'code': TAX['fine'][k], 'parent': TAX['parents'][PARENT[k]],
             'rpg_group': TAX['rpg_group_codes'][PARENT[k]]} for k in range(len(PARENT))]
pd.DataFrame(tax_rows).to_csv(TAB / 'taxonomy.csv', index=False)

acc = []
for f in sorted((ROOT / 'results/probs').glob('*.npz')):
    tag, model = f.stem.rsplit('_', 1)
    d = np.load(f)
    for k in d.files:
        if not k.endswith('__p'):
            continue
        dom = k[:-3]
        p = d[k].astype(np.float32)
        y = d[dom + '__y']
        pp = np.stack([p[:, PARENT == g].sum(1) for g in range(PARENT.max() + 1)], 1)
        conf = p.max(1)
        bins = np.minimum((conf * 15).astype(int), 14)
        ece = sum(abs((p.argmax(1) == y)[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                  for b in range(15) if (bins == b).any())
        acc.append({'source': tag, 'model': model, 'domain': dom, 'n': len(y),
                    'fine_acc': (p.argmax(1) == y).mean(), 'parent_acc': (pp.argmax(1) == PARENT[y]).mean(),
                    'ece_fine': ece})
acc = pd.DataFrame(acc)
acc.to_csv(TAB / 'model_accuracy.csv', index=False)
acc['in_domain'] = acc.domain == 'source_hold'
print(acc.groupby(['source', 'model', 'in_domain'])[['fine_acc', 'parent_acc', 'ece_fine']].mean().round(3).to_string())
save_json(ROOT / 'results' / 'describe.json', {'data': data.to_dict('records'), 'accuracy': acc.round(4).to_dict('records')})
