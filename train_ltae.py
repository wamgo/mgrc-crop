"""Lightweight Temporal Attention Encoder (L-TAE, Sainte Fare Garnot & Landrieu, 2020) on parcel sequences.

Runs on the CUDA interpreter with numpy and torch only. Same source split as train_models.py
(40k train, 5k dev for early stopping and temperature scaling, rest = held-out source pool) and the
same output format (results/probs/<tag>_ltae.npz), so experiments.py can use it like the other models.

Usage: python train_ltae.py region|year|eurocrops [--epochs N]
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
D = ROOT / 'data'
OUT = ROOT / 'results' / 'probs'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261011
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LTAE(nn.Module):
    def __init__(self, c_in, n_classes, d_model=128, n_head=16, d_k=8, mlp=128, dropout=0.2, T=1000):
        super().__init__()
        self.n_head, self.d_k, self.d_model = n_head, d_k, d_model
        self.inconv = nn.Sequential(nn.Linear(c_in, d_model), nn.LayerNorm(d_model))
        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.Q, std=math.sqrt(2.0 / d_k))
        self.fc_k = nn.Linear(d_model, n_head * d_k)
        self.T = T
        self.attn_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp), nn.BatchNorm1d(mlp), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(mlp, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, n_classes))

    def pos_enc(self, doy):
        d = self.d_model // self.n_head
        i = torch.arange(d, device=doy.device)
        denom = torch.pow(self.T, 2 * (i // 2) / d)
        ang = doy.float().unsqueeze(-1) / denom
        pe = torch.where(i % 2 == 0, torch.sin(ang), torch.cos(ang))
        return pe.repeat(1, 1, self.n_head)  # (B, T, d_model)

    def forward(self, x, doy, mask):
        B, T, _ = x.shape
        e = self.inconv(x) + self.pos_enc(doy)
        k = self.fc_k(e).view(B, T, self.n_head, self.d_k)
        scores = torch.einsum('bthd,hd->bht', k, self.Q) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        a = self.attn_drop(torch.softmax(scores, dim=-1))
        v = e.view(B, T, self.n_head, self.d_model // self.n_head)
        out = torch.einsum('bht,bthd->bhd', a, v).reshape(B, self.d_model)
        return self.head(self.mlp(out))


def load(name):
    s = np.load(D / f'seq_{name}.npz')
    f = np.load(D / f'feat_{name}.npz')
    assert np.array_equal(s['ids'], f['ids']), name
    return {'x': s['x'], 'doy': s['doy'], 'mask': s['mask'], 'y': f['y'], 'xy': f['xy'], 'ids': f['ids']}


def batches(d, idx, bs, norm, train=False, rng=None):
    for b in range(0, len(idx), bs):
        j = idx[b:b + bs]
        x = (d['x'][j].astype(np.float32) - norm[0]) / norm[1]
        m = d['mask'][j].copy()
        if train:  # randomly drop 20% of the valid observations, keep at least one
            drop = rng.random(m.shape) < 0.2
            keep = m & ~drop
            keep[keep.sum(1) == 0] = m[keep.sum(1) == 0]
            m = keep
        x[~m] = 0
        yield (torch.from_numpy(x).to(DEV), torch.from_numpy(d['doy'][j].astype(np.int64)).to(DEV),
               torch.from_numpy(m).to(DEV), torch.from_numpy(d['y'][j]).to(DEV))


def logits(model, d, idx, norm):
    model.eval()
    out = []
    with torch.no_grad():
        for x, doy, m, _ in batches(d, idx, 2048, norm):
            out.append(model(x, doy, m).float().cpu())
    return torch.cat(out).numpy()


def fit_temperature(z, y):
    zt, yt = torch.from_numpy(z), torch.from_numpy(y)
    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(zt / logt.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logt.detach().exp())


def softmax(z, T):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def run(src_name, others, tag, n_classes, epochs=60, tr_n=40000, dev_n=5000):
    out = OUT / f'{tag}_ltae.npz'
    if out.exists():
        print('exists', out.name)
        return
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    S = load(src_name)
    tr, dev, hold = np.arange(tr_n), np.arange(tr_n, tr_n + dev_n), np.arange(tr_n + dev_n, len(S['y']))
    xm = S['x'][tr][S['mask'][tr]].astype(np.float32)
    norm = (xm.mean(0), xm.std(0) + 1e-6)
    model = LTAE(S['x'].shape[2], n_classes).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * math.ceil(tr_n / 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    best, state, hist, t0 = math.inf, None, [], time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        for x, doy, m, y in batches(S, perm, 512, norm, True, rng):
            opt.zero_grad()
            F.cross_entropy(model(x, doy, m), y).backward()
            opt.step()
            sched.step()
        zd = logits(model, S, dev, norm)
        vl = float(F.cross_entropy(torch.from_numpy(zd), torch.from_numpy(S['y'][dev])))
        hist.append(vl)
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    T = fit_temperature(logits(model, S, dev, norm), S['y'][dev])
    res, meta = {}, {'ltae:T': T, 'ltae:history': hist, 'ltae:seconds': time.time() - t0}
    domains = {'source_hold': (S, hold)}
    for name in others:
        d = load(name)
        domains[name] = (d, np.arange(len(d['y'])))
    for dn, (d, idx) in domains.items():
        p = softmax(logits(model, d, idx, norm), T)
        res[f'{dn}__p'] = p.astype(np.float16)
        res[f'{dn}__y'] = d['y'][idx]
        res[f'{dn}__xy'] = d['xy'][idx]
        res[f'{dn}__ids'] = d['ids'][idx]
        meta[f'ltae:{dn}:acc'] = float((p.argmax(1) == d['y'][idx]).mean())
    np.savez_compressed(out, **res)
    torch.save(state, OUT / f'{tag}_ltae.pt')
    mf = OUT / f'{tag}_meta.json'
    old = json.loads(mf.read_text(encoding='utf8')) if mf.exists() else {}
    old.update(meta)
    mf.write_text(json.dumps(old, indent=2), encoding='utf8')
    print('TRAINED', tag, json.dumps({k: round(v, 4) for k, v in meta.items() if k.endswith('acc')}),
          'T', round(T, 3), 'sec', round(meta['ltae:seconds']), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'region'
    epochs = int(sys.argv[sys.argv.index('--epochs') + 1]) if '--epochs' in sys.argv else 60
    regions = ['frh01', 'frh02', 'frh03', 'frh04']
    K = len(json.loads((D / 'taxonomy.json').read_text(encoding='utf8'))['fine'])
    if mode == 'eurocrops':
        K = len(json.loads((D / 'taxonomy_eurocrops.json').read_text(encoding='utf8'))['fine'])
        doms = json.loads((D / 'eurocrops_domains.json').read_text(encoding='utf8'))
        for s in ['LV', 'EE']:
            o = 'EE' if s == 'LV' else 'LV'
            run(f'2021_EC_{s}', [f'2021_EC_{r}' for r in sorted(doms) if r.startswith(o) and len(r) > 2],
                f'EC2021_{s}', K, epochs)
    if mode == 'region':
        for s in regions:
            run(f'2017_L2A_{s}', [f'2017_L2A_{t}' for t in regions if t != s], f'L2A2017_{s}', K, epochs)
    elif mode == 'year':
        for s in ['frh01', 'frh04']:
            o = 'frh04' if s == 'frh01' else 'frh01'
            run(f'2017_L1C_{s}', ['2018_L1C_frh01', '2018_L1C_frh04', f'2017_L1C_{o}'], f'L1C2017_{s}', K, epochs)
