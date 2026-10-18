import unittest
import numpy as np
import hcrc
import hcrc_multi as M

# three levels: 3 roots -> 6 groups -> 12 leaves
G2 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
G1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
H3 = M.Hierarchy([G1, G2, np.arange(12)])


def synth(n, rng, H=H3):
    K = H.K
    y = rng.integers(0, K, n)
    z = rng.normal(0, 1, (n, K))
    z[np.arange(n), y] += 2.0
    for a, b in ((H.anc[0], 0.6), (H.anc[1], 0.9)):
        z += b * (a[None, :] == a[y][:, None])
    p = np.exp(z)
    return p / p.sum(1, keepdims=True), y


def brute_loss(H, p, y, t):
    """Direct definition: answer deepest passed node on the top-down path; wrong if not an ancestor."""
    t = np.maximum.accumulate(np.asarray(t, float))
    out = np.zeros(len(y), bool)
    for i in range(len(y)):
        prev, ans = None, None
        for l in range(H.L):
            P = H.mass(p[i:i + 1], l)[0]
            if prev is not None:
                P = np.where(H.parent[l] == prev, P, -1)
            g = int(P.argmax())
            if P[g] >= t[l]:
                ans = (l, g)
            prev = g
        if ans is not None:
            out[i] = ans[1] != H.anc[ans[0]][y[i]]
    return out


class T(unittest.TestCase):
    def test_split_matches_definition(self):
        rng = np.random.default_rng(0)
        p, y = synth(1500, rng)
        S = M.Scores(H3, p, y)
        for t in [(0.2, 0.4, 0.6), (0.5, 0.5, 0.5), (0.1, 0.7, 0.3), (0.3, 0.3, 0.9)]:
            tt = np.maximum.accumulate(np.array(t))
            split = np.zeros(len(y), bool)
            for l in range(3):
                term = (S.s[l] >= tt[l]) & S.e[l]
                self.assertFalse((split & term).any())  # disjoint
                split |= term
            self.assertTrue(np.array_equal(split, brute_loss(H3, p, y, t)))
            self.assertAlmostEqual(S.evaluate(t)['risk'], split.mean())

    def test_two_level_reduces_to_hcrc(self):
        PARENT = np.array([0, 0, 1, 0, 1, 2, 2, 3, 3])
        H2 = M.Hierarchy([PARENT, np.arange(9)])
        rng = np.random.default_rng(1)
        y = rng.integers(0, 9, 3000)
        z = rng.normal(0, 1, (3000, 9)); z[np.arange(3000), y] += 2
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        S = M.Scores(H2, p, y)
        for tp, tf in [(0.3, 0.6), (0.5, 0.5), (0.4, 0.9)]:
            d, g, f = hcrc.answer_levelwise(p, PARENT, tp, tf)
            self.assertAlmostEqual(S.evaluate([tp, tf])['risk'], hcrc.hier_error(d, g, f, y, PARENT).mean())

    def test_mixed_calibration_validity(self):
        rng = np.random.default_rng(2)
        pt, yt = synth(30000, rng)
        T_ = M.Scores(H3, pt, yt)
        viol = []
        for r in range(300):
            p, y = synth(1500, rng)
            S = M.Scores(H3, p, y)
            # 40 fine, 300 level-2, 1500 level-1 labels (nested sets)
            sets = [S.sub(np.arange(1500)), S.sub(np.arange(300)), S.sub(np.arange(40))]
            t = M.calibrate_levels(sets, [0.05, 0.03, 0.02], 0.1)
            viol.append(T_.evaluate(t)['risk'] > 0.1)
        self.assertLessEqual(np.mean(viol), 0.1 + 0.035)

    def test_climb_validity_and_min_sample(self):
        rng = np.random.default_rng(3)
        pt, yt = synth(30000, rng)
        T_ = M.Scores(H3, pt, yt)
        viol = []
        for r in range(200):
            p, y = synth(200, rng)
            th = M.calibrate_climb(M.Scores(H3, p, y), 0.1, 0.1)
            viol.append(T_.evaluate_climb(th)['risk'] > 0.1)
        self.assertLessEqual(np.mean(viol), 0.1 + 0.04)
        self.assertEqual(M.min_sample(0.1, 0.1), 22)
        self.assertEqual(M.min_sample(0.09, 0.05), 32)
        self.assertEqual(M.min_sample(0.1, 0.1, 'crc'), 9)


if __name__ == '__main__':
    unittest.main()
