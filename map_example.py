"""Released-map example: one transfer, one calibration draw, every target parcel labelled as the released map would.

Outputs results/exp/v2/map_example.npz (parcel xy, answer level, answered node, true labels) and
results/exp/v2/map_example_ua.csv (per released class: parcels, realised user's accuracy on the whole target domain).
Usage: MGRC_TAXON=pheno python map_example.py
"""
import json
import numpy as np, pandas as pd
import experiments as E
import hcrc

TAG, MODEL, TARGET = 'L2A2017_frh01', 'ltae', '2017_L2A_frh02'
ALPHA, B, RHO, SEED = 0.02, 100, 10, 12345


def answers(p, design, tp, tf, th=None):
    """Per-parcel answer: level (0 none, 1 group, 2 code) and node index (group or code)."""
    g, sg, f, sf = hcrc.top_down(p, E.PARENT)
    f0 = p.argmax(1)
    pp = hcrc.parent_probs(p, E.PARENT)
    sg0 = pp[np.arange(len(p)), E.PARENT[f0]]
    rule = design[0]
    if rule == 'td':
        code = sf >= max(tf, tp)
        grp = ~code & (sg >= tp)
        return np.where(code, 2, np.where(grp, 1, 0)), np.where(code, f, g)
    if rule == 'hsc':
        code = p.max(1) >= th
        grp = ~code & (sg0 >= th)
    else:  # 'cl'
        code = p.max(1) >= max(tf, tp)
        grp = ~code & (sg0 >= tp)
    return np.where(code, 2, np.where(grp, 1, 0)), np.where(code, f0, E.PARENT[f0])


if __name__ == '__main__':
    E.MODE['mode'], E.MODE['delta'] = 'pac', 0.1
    d = np.load(E.PROBS / f'{TAG}_{MODEL}.npz')
    src, tgt, xy = E.load_pair(TAG, MODEL, TARGET)
    pt, yt = d[f'{TARGET}__p'].astype(np.float64), d[f'{TARGET}__y']
    design = E.plan(src, ALPHA, B, RHO, E.BETAS, ['info'], seed=SEED)['info']
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(tgt.n)
    nf = max(int(round(design[1] * B)), 1)
    nc = int(round((B - nf) * RHO))
    F, C = tgt.sub(perm[:nf]), tgt.sub(perm[:nf + nc])
    tp = tf = th = None
    if design[0] == 'hsc':
        th = E.cal_climb(F, ALPHA)
    elif design[0] == 'td':
        tp, tf = E.cal_levelwise(F, design[2] * ALPHA, (1 - design[2]) * ALPHA, coarse=C)
    else:
        tp, tf = E.cal_climb2(F, design[2] * ALPHA, (1 - design[2]) * ALPHA, coarse=C)
    lvl, node = answers(pt, design, tp, tf, th)
    # code-only comparison: HSC climbing on B crop codes
    th_code = E.cal_climb(tgt.sub(perm[:B]), ALPHA)
    lvl_code, node_code = answers(pt, ('hsc',), None, None, th_code)
    correct = np.where(lvl == 2, node == yt, np.where(lvl == 1, node == E.PARENT[yt], True))
    print('design', design, 'n codes', nf, 'n group labels', nf + nc, 'thresholds', tp, tf, th)
    print('MGRC  : code %.3f group %.3f none %.3f  realised error %.4f' % ((lvl == 2).mean(), (lvl == 1).mean(),
                                                                           (lvl == 0).mean(), (~correct).mean()))
    print('codes : answered %.3f' % (lvl_code > 0).mean())
    names = E.TAX['parents']
    rows = []
    for g in range(len(names)):
        m = (lvl == 1) & (node == g)
        if m.any():
            rows.append({'released class': names[g], 'level': 'group', 'parcels': int(m.sum()),
                         'share': m.mean(), 'user accuracy': float((E.PARENT[yt[m]] == g).mean())})
    fine_names = E.TAX.get('fine_names', E.TAX['fine'])
    for k in np.unique(node[lvl == 2]):
        m = (lvl == 2) & (node == k)
        rows.append({'released class': fine_names[k], 'level': 'code', 'parcels': int(m.sum()), 'share': m.mean(),
                     'user accuracy': float((yt[m] == k).mean())})
    ua = pd.DataFrame(rows)
    ua.to_csv(E.ROOT / 'results' / 'exp' / 'v2' / 'map_example_ua.csv', index=False)
    print(ua.round(4).to_string(index=False))
    np.savez_compressed(E.ROOT / 'results' / 'exp' / 'v2' / 'map_example.npz', xy=xy, lvl=lvl, node=node, y=yt,
                        lvl_code=lvl_code, design=np.array([str(x) for x in design]))
