"""Train MLP / LightGBM / temporal CNN on each source region; store probabilities for every region.

Source split (hash order, fixed): 40k train, 5k dev (early stopping + temperature), 15k held-out source.
Temperature scaling is fitted on source dev only. No target labels are used here.
"""
from pathlib import Path
import sys, json, math, time
from pilot_common import setup, save_json
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, torch, lightgbm as lgb
from sklearn.preprocessing import StandardScaler

SEED = 20261011
REGIONS = ['frh01', 'frh02', 'frh03', 'frh04']
OUT = ROOT / 'results' / 'probs'
OUT.mkdir(parents=True, exist_ok=True)
K = 23  # overwritten per dataset in run()


def load(year, level, region):
    return dict(np.load(ROOT / 'data' / f'feat_{year}_{level}_{region}.npz'))


class TempCNN(torch.nn.Module):
    def __init__(self, c=10, k=K, h=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(c, h, 3, padding=1), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(.2),
            torch.nn.Conv1d(h, h, 3, padding=1), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(.2),
            torch.nn.Conv1d(h, h, 3, padding=1), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(.2),
            torch.nn.Flatten(), torch.nn.Linear(h * 12, 128), torch.nn.ReLU(), torch.nn.Dropout(.3), torch.nn.Linear(128, k))

    def forward(self, x):
        return self.net(x)


def mlp():
    return torch.nn.Sequential(torch.nn.Linear(140, 256), torch.nn.ReLU(), torch.nn.Dropout(.2),
                               torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(128, K))


def fit_torch(model, xtr, ytr, xdev, ydev, epochs=40, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xtr, ytr = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xdev_t, ydev_t = torch.from_numpy(xdev), torch.from_numpy(ydev)
    best, state, hist = math.inf, None, []
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(xtr)).split(256):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.cross_entropy(model(xdev_t), ydev_t).item()
        hist.append(vl)
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()

    def logits(x):
        with torch.no_grad():
            return torch.cat([model(b) for b in torch.from_numpy(x).split(2048)]).numpy()
    return logits, hist


def fit_temperature(logit_dev, ydev):
    lt, yt = torch.from_numpy(logit_dev.astype(np.float32)), torch.from_numpy(ydev)
    logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lt / logt.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logt.exp())


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def run(year, level, src, others, tag):
    global K
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(6)
    S = load(year, level, src)
    K = 24 if level == 'EC' else 23
    tr, dev, hold = np.arange(40000), np.arange(40000, 45000), np.arange(45000, len(S['y']))
    scaler = StandardScaler().fit(S['X'][tr])
    cmean, cstd = S['cube'][tr].mean((0, 1)), S['cube'][tr].std((0, 1)) + 1e-6
    feats = lambda d: scaler.transform(d['X']).astype(np.float32)
    cube = lambda d: ((d['cube'] - cmean) / cstd).transpose(0, 2, 1).astype(np.float32).copy()
    domains = {'source_hold': {k: v[hold] for k, v in S.items()}}
    for (y2, l2, r2) in others:
        domains[f'{y2}_{l2}_{r2}'] = load(y2, l2, r2)
    meta_path = OUT / f'{tag}_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf8')) if meta_path.exists() else {}
    trained = False
    for name in ['mlp', 'lgbm', 'tempcnn']:
        out = OUT / f'{tag}_{name}.npz'
        if out.exists():
            continue
        t0 = time.time()
        if name == 'mlp':
            f, hist = fit_torch(mlp(), feats(S)[tr], S['y'][tr], feats(S)[dev], S['y'][dev])
            predict = lambda d: f(feats(d))
        elif name == 'tempcnn':
            f, hist = fit_torch(TempCNN(k=K), cube(S)[tr], S['y'][tr], cube(S)[dev], S['y'][dev], epochs=30)
            predict = lambda d: f(cube(d))
        else:
            clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=.05, num_leaves=63, subsample=.8, subsample_freq=1,
                                     colsample_bytree=.6, min_child_samples=20, n_jobs=6, random_state=SEED, verbose=-1)
            clf.fit(S['X'][tr], S['y'][tr], eval_set=[(S['X'][dev], S['y'][dev])],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            hist = [int(clf.best_iteration_)]
            predict = lambda d: clf.predict(d['X'], raw_score=True)
        T = fit_temperature(predict({k: v[dev] for k, v in S.items()}), S['y'][dev])
        res = {}
        for dn, d in domains.items():
            p = softmax(predict(d), T)
            res[f'{dn}__p'] = p.astype(np.float16)
            res[f'{dn}__y'] = d['y']
            res[f'{dn}__xy'] = d['xy']
            res[f'{dn}__ids'] = d['ids']
            meta[f'{name}:{dn}:acc'] = float((p.argmax(1) == d['y']).mean())
        np.savez_compressed(out, **res)
        meta[f'{name}:T'] = T
        meta[f'{name}:seconds'] = time.time() - t0
        meta[f'{name}:history'] = hist
        trained = True
        print('TRAINED', tag, name, json.dumps({k: v for k, v in meta.items() if k.startswith(name) and 'acc' in k}), flush=True)
    if trained:
        save_json(meta_path, meta)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'region'
    if mode == 'region':
        for src in REGIONS:
            run(2017, 'L2A', src, [(2017, 'L2A', r) for r in REGIONS if r != src], f'L2A2017_{src}')
    elif mode == 'eurocrops':  # cross-country: LV <-> EE, targets are NUTS-3 regions of the other country
        doms = json.loads((ROOT / 'data' / 'eurocrops_domains.json').read_text(encoding='utf8'))
        for src in ['LV', 'EE']:
            other = 'EE' if src == 'LV' else 'LV'
            run(2021, 'EC', src, [(2021, 'EC', r) for r in sorted(doms) if r.startswith(other) and len(r) > 2],
                f'EC2021_{src}')
    else:  # cross-year on L1C: train 2017, test 2018 same and other region
        for src in ['frh01', 'frh04']:
            run(2017, 'L1C', src, [(2018, 'L1C', 'frh01'), (2018, 'L1C', 'frh04'),
                                   (2017, 'L1C', 'frh04' if src == 'frh01' else 'frh01')], f'L1C2017_{src}')
