"""Sample ESA WorldCereal 2021 (AEZ 22190) at EuroCropsML parcel centres and map to the level-1 classes.

z = 2 (grassland, green fodder, permanent crops) where temporary crops = 0; z = 0 (cereals) where temporary
crops = 100 and any of winter cereals, spring cereals or maize = 100; z = 1 (other temporary crops) otherwise;
z = -1 where any product has no data. Run with python -I (the GeoTIFFs are downloaded data); rasterio is
loaded from deps/.

Output: data/worldcereal/wc_<tag>.npz with, per domain key of results/probs/<tag>_mlp.npz, the arrays
<domain>__z, <domain>__raw (n x 4 product values) and <domain>__ids.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
import numpy as np
import rasterio
from rasterio.warp import transform

WC = HERE / 'data' / 'worldcereal'
PRODUCTS = ['tc-annual_temporarycrops', 'tc-wintercereals_wintercereals', 'tc-springcereals_springcereals',
            'tc-maize-main_maize']


def tif(prod):
    f = sorted(WC.glob(f'22190_{prod}_*_classification.tif'))
    assert len(f) == 1, (prod, f)
    return f[0]


def sample(path, xy):
    """Pixel values at UTM 35N points; every touched tile is decoded once."""
    with rasterio.open(path) as r:
        out = np.full(len(xy), 255, np.int32)
        if len(xy):
            xs, ys = transform('EPSG:32635', r.crs, xy[:, 0].tolist(), xy[:, 1].tolist())
            rows, cols = rasterio.transform.rowcol(r.transform, xs, ys)
            rows, cols = np.asarray(rows), np.asarray(cols)
            inside = np.flatnonzero((rows >= 0) & (rows < r.height) & (cols >= 0) & (cols < r.width))
            bh, bw = r.block_shapes[0]
            tile = (rows[inside] // bh) * (r.width // bw + 1) + cols[inside] // bw
            for t in np.unique(tile):
                sel = inside[tile == t]
                r0, c0 = rows[sel[0]] // bh * bh, cols[sel[0]] // bw * bw
                win = rasterio.windows.Window(c0, r0, min(bw, r.width - c0), min(bh, r.height - r0))
                a = r.read(1, window=win)
                out[sel] = a[rows[sel] - r0, cols[sel] - c0]
        return out, r.crs.to_string(), r.nodata


def to_z(raw):
    tc, wcer, scer, mz = raw.T
    valid = np.all(np.isin(raw[:, :1], [0, 100]), 1)
    cereal = (wcer == 100) | (scer == 100) | (mz == 100)
    z = np.where(tc == 100, np.where(cereal, 0, 1), 2)
    return np.where(valid, z, -1)


if __name__ == '__main__':
    import time
    doms, xys = [], []
    files = {tag: np.load(HERE / 'results' / 'probs' / f'{tag}_mlp.npz') for tag in ['EC2021_LV', 'EC2021_EE']}
    for tag, d in files.items():
        for k in d.files:
            if k.endswith('__xy'):
                doms.append((tag, k[:-4], len(d[k])))
                xys.append(d[k])
    xy = np.concatenate(xys)
    raw = []
    for p in PRODUCTS:
        t0 = time.time()
        v, crs, nd = sample(tif(p), xy)
        raw.append(v)
        print(p, crs, 'nodata', nd, 'values', dict(zip(*map(np.ndarray.tolist, np.unique(v, return_counts=True)))),
              round(time.time() - t0), 's', flush=True)
    raw = np.stack(raw, 1)
    i = 0
    out = {tag: {} for tag in files}
    for tag, dom, n in doms:
        r = raw[i:i + n]
        i += n
        out[tag][f'{dom}__raw'], out[tag][f'{dom}__z'], out[tag][f'{dom}__ids'] = r, to_z(r), files[tag][f'{dom}__ids']
        print(tag, dom, n, 'z counts', dict(zip(*map(np.ndarray.tolist, np.unique(to_z(r), return_counts=True)))))
    for tag in files:
        np.savez_compressed(WC / f'wc_{tag}.npz', **out[tag])
