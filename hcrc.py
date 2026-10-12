"""Hierarchical risk control for two-level (fine -> parent) classification.

All rules map fine-class probabilities to an answer at depth 2 (fine class), 1 (parent) or 0 (root/abstain).
Hierarchical error = answer is neither the true fine class nor one of its ancestors (root is never wrong).
Calibration follows conformal risk control (Angelopoulos et al., ICLR 2024): for a loss that is
non-increasing in a threshold t and bounded by 1, pick the smallest t with (n*R_hat(t)+1)/(n+1) <= alpha.
"""
import numpy as np

GRID = np.r_[np.linspace(0.0, 1.0, 2001), np.inf]


def parent_probs(p, parent):
    g = parent.max() + 1
    return np.stack([p[:, parent == j].sum(1) for j in range(g)], 1)


def top_down(p, parent):
    """Parent = argmax parent mass; fine = argmax fine prob inside that parent."""
    pp = parent_probs(p, parent)
    g = pp.argmax(1)
    masked = np.where(parent[None, :] == g[:, None], p, -1.0)
    f = masked.argmax(1)
    return g, pp[np.arange(len(p)), g], f, p[np.arange(len(p)), f]


def crc_threshold(score, err, alpha, grid=GRID):
    """Smallest grid t with (sum(err * [score>=t]) + 1)/(n+1) <= alpha; inf if none (always abstain).

    err is the 0/1 loss incurred if the sample is answered; loss(t) = err * [score >= t] is non-increasing in t.
    """
    n = len(score)
    if n == 0:
        return np.inf
    order = np.argsort(score)
    s, e = score[order], err[order].astype(float)
    tail = np.r_[np.cumsum(e[::-1])[::-1], 0.0]  # tail[i] = sum e[j] for j >= i
    idx = np.searchsorted(s, grid, side='left')
    risk_sum = tail[idx]
    ok = (risk_sum + 1.0) / (n + 1.0) <= alpha
    return grid[np.argmax(ok)] if ok.any() else np.inf


# ---------------------------------------------------------------- answer rules
def answer_levelwise(p, parent, t_parent, t_fine):
    g, sg, f, sf = top_down(p, parent)
    t_fine = max(t_fine, t_parent)  # nesting keeps the loss decomposition exact
    depth = np.where(sf >= t_fine, 2, np.where(sg >= t_parent, 1, 0))
    return depth, g, f


def answer_climb(p, parent, theta):
    """HSC-style climbing: start at the argmax leaf, climb until node confidence >= theta."""
    f = p.argmax(1)
    g = parent[f]
    pp = parent_probs(p, parent)
    sf = p[np.arange(len(p)), f]
    sg = pp[np.arange(len(p)), g]
    depth = np.where(sf >= theta, 2, np.where(sg >= theta, 1, 0))
    return depth, g, f


def answer_flat(p, parent, t):
    f = p.argmax(1)
    depth = np.where(p.max(1) >= t, 2, 0)
    return depth, parent[f], f


def answer_cp_lca(p, parent, q):
    """Split-conformal set {k: 1-p_k <= q}; answer its least common ancestor (empty set -> root)."""
    sets = (1.0 - p) <= q
    size = sets.sum(1)
    groups = np.stack([(sets & (parent[None, :] == j)).any(1) for j in range(parent.max() + 1)], 1)
    ngroups = groups.sum(1)
    depth = np.where(size == 1, 2, np.where((size > 1) & (ngroups == 1), 1, 0))
    f = np.where(size >= 1, np.argmax(sets, 1), p.argmax(1))
    g = np.where(ngroups >= 1, np.argmax(groups, 1), parent[p.argmax(1)])
    return depth, g, f


def hier_error(depth, g, f, y, parent):
    return ((depth == 2) & (f != y)) | ((depth == 1) & (g != parent[y]))


# ---------------------------------------------------------------- calibrators
def calibrate_levelwise(p, y, parent, alpha_parent, alpha_fine):
    """Separate CRC per level. E[loss] <= alpha_parent + alpha_fine under exchangeability."""
    g, sg, f, sf = top_down(p, parent)
    e_parent = g != parent[y]
    e_fine = (g == parent[y]) & (f != y)
    tp = crc_threshold(sg, e_parent, alpha_parent)
    tf = crc_threshold(sf, e_fine, alpha_fine)
    return {'t_parent': tp, 't_fine': max(tf, tp)}


def calibrate_levelwise_coarse(p, y_parent_coarse, p_fine, y_fine, parent, alpha_parent, alpha_fine):
    """Mixed granularity: parent term from (possibly many) parent-only labels, fine term from fine labels."""
    g, sg, _, _ = top_down(p, parent)
    tp = crc_threshold(sg, g != y_parent_coarse, alpha_parent)
    g2, _, f2, sf2 = top_down(p_fine, parent)
    tf = crc_threshold(sf2, (g2 == parent[y_fine]) & (f2 != y_fine), alpha_fine)
    return {'t_parent': tp, 't_fine': max(tf, tp)}


def path_from_source(p, y, parent, levels=np.linspace(0.005, 0.5, 100), split_grid=np.linspace(0.05, 0.95, 19)):
    """Learn a monotone 1-D path lambda -> (t_parent, t_fine) on source data.

    For each source risk level a, choose the alpha split maximising fine answers (ties: non-root answers),
    using empirical thresholds (no finite-sample slack). Thresholds are then made non-decreasing along the path.
    """
    g, sg, f, sf = top_down(p, parent)
    ep = g != parent[y]
    ef = (g == parent[y]) & (f != y)
    n = len(y)

    def emp(score, err, a):
        order = np.argsort(score)
        s, e = score[order], err[order]
        tail = np.r_[np.cumsum(e[::-1])[::-1], 0] / n
        idx = np.searchsorted(s, GRID, side='left')
        ok = tail[idx] <= a
        return GRID[np.argmax(ok)]

    path = []
    for a in levels:
        best = None
        for r in split_grid:
            tp = emp(sg, ep, r * a)
            tf = max(emp(sf, ef, (1 - r) * a), tp)
            fine = np.mean(sf >= tf)
            nonroot = np.mean(sg >= tp)
            key = (fine, nonroot)
            if best is None or key > best[0]:
                best = (key, tp, tf)
        path.append(best[1:])
    path = np.array(path, float)[::-1]  # increasing strictness along lambda
    path = np.maximum.accumulate(path, axis=0)
    return np.vstack([path, [np.inf, np.inf]])


def calibrate_path(p, y, parent, path, alpha):
    """Single CRC along a source-learned monotone path; loss is non-increasing in the path index."""
    errs = []
    for tp, tf in path:
        d, g, f = answer_levelwise(p, parent, tp, tf)
        errs.append(hier_error(d, g, f, y, parent).sum())
    errs = np.array(errs, float)
    n = len(y)
    ok = (errs + 1) / (n + 1) <= alpha
    k = int(np.argmax(ok)) if ok.any() else len(path) - 1
    return {'t_parent': path[k, 0], 't_fine': path[k, 1]}


def calibrate_climb(p, y, parent, alpha):
    errs = []
    for th in GRID:
        d, g, f = answer_climb(p, parent, th)
        errs.append(hier_error(d, g, f, y, parent).sum())
    errs = np.array(errs, float)
    ok = (errs + 1) / (len(y) + 1) <= alpha
    return {'theta': GRID[np.argmax(ok)] if ok.any() else np.inf}


def calibrate_flat(p, y, parent, alpha):
    f = p.argmax(1)
    return {'t': crc_threshold(p.max(1), f != y, alpha)}


def calibrate_cp(p, y, alpha):
    s = np.sort(1.0 - p[np.arange(len(y)), y])
    k = int(np.ceil((len(y) + 1) * (1 - alpha)))
    return {'q': s[k - 1] if k <= len(y) else np.inf}


def summarize(depth, g, f, y, parent):
    err = hier_error(depth, g, f, y, parent)
    return {'risk': err.mean(), 'fine': np.mean(depth == 2), 'parent': np.mean(depth == 1),
            'root': np.mean(depth == 0),
            'fine_acc': np.mean(f[depth == 2] == y[depth == 2]) if (depth == 2).any() else np.nan}
