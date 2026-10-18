"""Multi-level mixed-granularity risk control.

Hierarchy: levels 1..L, level L = fine classes. anc[l][k] is the level-(l+1) ancestor of fine class k
(0-based lists, coarse to fine; anc[L-1] is the identity). Top-down rule with t_1 <= ... <= t_L:
    g_1 = argmax_v P^1_v,  g_l = argmax over children of g_{l-1} of P^l_v,  s_l = P^l_{g_l};
    answer the deepest level l with s_l >= t_l, or the root if there is none.
s_l is non-increasing along the path and t_l non-decreasing, so passing level l implies passing every
coarser level and the answer depth is d = max{l : s_l >= t_l}.

Error split: an answer at depth d is wrong iff the path leaves the true path at some level <= d. With
e_l = 1[g_1..g_{l-1} correct, g_l wrong] (first deviation at level l),
    loss = sum_l 1[s_l >= t_l] e_l,
the terms are disjoint, term l depends on t_l only, and e_l needs only the level-l label.
"""
from functools import lru_cache
import numpy as np
from scipy.stats import beta as _beta

GRID = np.r_[np.linspace(0.0, 1.0, 2001), np.inf]


class Hierarchy:
    def __init__(self, anc):
        self.anc = [np.asarray(a) for a in anc]
        self.L = len(self.anc)
        self.K = len(self.anc[-1])
        assert np.array_equal(self.anc[-1], np.arange(self.K))
        for l in range(1, self.L):
            for v in np.unique(self.anc[l]):
                assert len(np.unique(self.anc[l - 1][self.anc[l] == v])) == 1, 'levels must nest'
        self.n_nodes = [int(a.max()) + 1 for a in self.anc]
        # parent[l][v]: level-l parent of level-(l+1) node v
        self.parent = [None] + [np.array([self.anc[l - 1][np.flatnonzero(self.anc[l] == v)[0]]
                                          for v in range(self.n_nodes[l])]) for l in range(1, self.L)]
        leaves = [np.bincount(a, minlength=n) for a, n in zip(self.anc, self.n_nodes)]
        self.info = [1 - np.log(c) / np.log(self.K) for c in leaves]

    def mass(self, p, l):
        return np.stack([p[:, self.anc[l] == v].sum(1) for v in range(self.n_nodes[l])], 1)


def _depth(passed, L):
    return np.where(passed.any(0), L - np.argmax(passed[::-1], 0), 0)


class Scores:
    def __init__(self, H, p, y):
        p = np.asarray(p, np.float64)
        n = len(y)
        self.H, self.n, self.y = H, n, np.asarray(y)
        self.s = np.zeros((H.L, n))
        self.e = np.zeros((H.L, n), bool)
        self.node = np.zeros((H.L, n), int)
        ok = np.ones(n, bool)
        prev = None
        for l in range(H.L):
            P = H.mass(p, l)
            if prev is not None:
                P = np.where(H.parent[l][None, :] == prev[:, None], P, -1.0)
            g = P.argmax(1)
            self.node[l], self.s[l] = g, P[np.arange(n), g]
            right = g == H.anc[l][self.y]
            self.e[l] = ok & ~right
            ok = ok & right
            prev = g
        leaf = p.argmax(1)  # bottom-up climbing (HSC): ancestors of the argmax leaf
        self.cl_node = np.stack([H.anc[l][leaf] for l in range(H.L)])
        self.cl_s = np.stack([H.mass(p, l)[np.arange(n), self.cl_node[l]] for l in range(H.L)])
        self.cl_wrong = np.stack([self.cl_node[l] != H.anc[l][self.y] for l in range(H.L)])

    def sub(self, idx):
        o = object.__new__(Scores)
        o.H, o.n, o.y = self.H, len(idx), self.y[idx]
        for k in ('s', 'e', 'node', 'cl_node', 'cl_s', 'cl_wrong'):
            setattr(o, k, getattr(self, k)[:, idx])
        return o

    def _summary(self, d, wrong, node):
        H = self.H
        shares = np.array([np.mean(d == l + 1) for l in range(H.L)])
        info = np.zeros(self.n)
        for l in range(H.L):
            info += (d == l + 1) * H.info[l][node[l]]
        return {'risk': float(wrong.mean()), 'shares': shares, 'hcov': float(info.mean())}

    def evaluate(self, t):
        t = np.maximum.accumulate(np.asarray(t, float))
        d = _depth(self.s >= t[:, None], self.H.L)
        wrong = np.zeros(self.n, bool)
        for l in range(self.H.L):
            wrong |= (d >= l + 1) & self.e[l]
        return self._summary(d, wrong, self.node)

    def evaluate_climb(self, theta):
        d = _depth(self.cl_s >= theta, self.H.L)
        wrong = np.zeros(self.n, bool)
        for l in range(self.H.L):
            wrong |= (d == l + 1) & self.cl_wrong[l]
        return self._summary(d, wrong, self.cl_node)


# ---------------------------------------------------------------- calibration
def ucb(k, n, delta):
    k = np.asarray(k, float)
    return np.where(k >= n, 1.0, _beta.ppf(1 - delta, k + 1, np.maximum(n - k, 1e-9)))


@lru_cache(maxsize=4096)
def _ucb_table(n, delta):
    return ucb(np.arange(n + 1), n, delta)


def crit(k, n, alpha, delta, mode):
    if mode == 'crc':
        return (np.asarray(k, float) + 1) / (n + 1) <= alpha
    return _ucb_table(int(n), float(delta))[np.asarray(k, int)] <= alpha


def counts(score, err, grid=GRID):
    o = np.argsort(score)
    s, e = score[o], err[o].astype(float)
    tail = np.r_[np.cumsum(e[::-1])[::-1], 0.0]
    return tail[np.searchsorted(s, grid, side='left')]


def thr(score, err, alpha, delta, mode='pac'):
    n = len(score)
    if n == 0 or alpha <= 0:
        return np.inf
    ok = crit(counts(score, err), n, alpha, delta, mode)
    return GRID[np.argmax(ok)] if ok.any() else np.inf


def calibrate_levels(sets, alphas, delta, mode='pac', errs=None):
    """sets[l]: Scores of the parcels whose level-(l+1) label is known (finer-labelled parcels included).
    errs[l] optionally replaces sets[l].e[l] (e.g. first-deviation indicators computed from noisy labels).
    Term l is calibrated at level alphas[l] and confidence delta/m, m = number of levels with alphas[l] > 0
    (a level with alpha 0 gets t = inf, so its term is zero and spends no confidence). Returns nested thresholds."""
    L = len(sets)
    m = max(sum(a > 0 for a in alphas), 1)
    t = []
    for l in range(L):
        e = sets[l].e[l] if errs is None or errs[l] is None else errs[l]
        t.append(thr(sets[l].s[l], e, alphas[l], delta / m, mode))
    return np.maximum.accumulate(np.array(t, float))


def calibrate_climb(S, alpha, delta, mode='pac'):
    err = np.zeros(len(GRID))
    for i, th in enumerate(GRID):
        d = _depth(S.cl_s >= th, S.H.L)
        w = np.zeros(S.n, bool)
        for l in range(S.H.L):
            w |= (d == l + 1) & S.cl_wrong[l]
        err[i] = w.sum()
    ok = crit(err, S.n, alpha, delta, mode)
    return GRID[np.argmax(ok)] if ok.any() else np.inf


def calibrate_flat(S, alpha, delta, mode='pac'):
    L = S.H.L
    return thr(S.cl_s[L - 1], S.cl_wrong[L - 1], alpha, delta, mode)


def evaluate_flat(S, t):
    L = S.H.L
    ans = S.cl_s[L - 1] >= t
    d = np.where(ans, L, 0)
    return S._summary(d, ans & S.cl_wrong[L - 1], S.cl_node)


def min_sample(alpha, delta, mode='pac'):
    n = 1
    while not crit(0, n, alpha, delta, mode):
        n += 1
    return n
