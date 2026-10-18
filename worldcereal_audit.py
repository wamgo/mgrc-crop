"""Aggregate the WorldCereal product audit in supplementary S6.

Requires locally generated WorldCereal/probability arrays, not redistributed
here. Writes counts/rates only; no parcel IDs, labels or coordinates.
"""
from pathlib import Path
import argparse, json
import numpy as np

def audit(root):
    tax=json.loads((root/'data/taxonomy_eurocrops.json').read_text(encoding='utf8'))
    a=np.array([0 if c.startswith('330101') else 2 if c.startswith(('330109','3302','3303')) else 1 for c in tax['fine']])
    n=bad=nodata=0
    class_n=np.zeros(3,dtype=int);class_bad=np.zeros(3,dtype=int)
    shared={m:0 for m in ['lgbm','ltae']};regions=[]
    for tag in ['EC2021_LV','EC2021_EE']:
        with np.load(root/'data/worldcereal'/f'wc_{tag}.npz',allow_pickle=False) as w:
            for key in w.files:
                if not key.endswith('__z') or key.startswith('source_hold'):continue
                domain=key[:-3];z=w[key];wid=w[domain+'__ids']
                labels=None
                for model in shared:
                    with np.load(root/'results/probs'/f'{tag}_{model}.npz',allow_pickle=False) as d:
                        assert np.array_equal(wid,d[domain+'__ids']), 'Parcel order mismatch'
                        y=a[d[domain+'__y']]
                        if labels is not None:assert np.array_equal(labels,y)
                        labels=y
                        prob=np.asarray(d[domain+'__p'],dtype=np.float64)
                        g=np.stack([prob[:,a==i].sum(1) for i in range(3)],axis=1).argmax(1)
                        shared[model]+=int(((g==z)&(z!=y)).sum())
                e=z!=labels;n+=len(z);bad+=int(e.sum());nodata+=int((z<0).sum())
                class_n+=np.bincount(labels,minlength=3);class_bad+=np.bincount(labels[e],minlength=3)
                regions.append({'domain':domain,'n':len(z),'disagreement':float(e.mean())})
    return {'target_parcels':n,'disagreements':bad,'disagreement_rate':bad/n,'no_data_count':nodata,
            'regions':regions,'classes':[{'name':name,'n':int(class_n[i]),'disagreements':int(class_bad[i]),'rate':float(class_bad[i]/class_n[i])} for i,name in enumerate(['cereals','other temporary crops','grassland, fodder and permanent crops'])],
            'shared_error_rate':{m:v/n for m,v in shared.items()}}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path)
    args=ap.parse_args();report=audit(args.data_root.resolve());text=json.dumps(report,indent=2)+'\n';print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf8')
