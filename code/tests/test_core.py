"""Unit tests for core routines (run: python3 -m pytest -q tests  from code/)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import scitovski_eps, b_aki, mc_maxc, project
from clustering import prune, run, grid_labels, grid_ari
from decluster import nn_zbz, gk74

def _seq(rng):
    # background Poisson events + one M6 mainshock with 150 close aftershocks
    nb = 300; tb = np.sort(rng.uniform(0, 10, nb)); xb = rng.uniform(0, 300, (nb, 2)); mb = 3 + rng.exponential(0.43, nb)
    ta = 5 + np.sort(rng.exponential(0.02, 150)); xa = rng.normal(150, 3, (150, 2)); ma = 3 + rng.exponential(0.43, 150)
    t = np.r_[tb, 5.0, ta]; x = np.vstack([xb, [[150, 150]], xa]); m = np.r_[mb, 6.0, ma]; o = np.argsort(t)
    lab = np.r_[np.zeros(nb), 0, np.ones(150)][o]; return t[o], x[o], np.minimum(m[o], 5.9 + (np.r_[mb, 6.0, ma][o] == 6.0) * 0.1), lab

def test_declustering_flags_aftershocks():
    rng = np.random.default_rng(5); t, x, m, lab = _seq(rng)
    z = nn_zbz(t, x, m); g = gk74(t * 365.25, x, m)
    for bg in (z['background'], g):
        assert (~bg[lab == 1]).mean() > 0.7 and bg[lab == 0].mean() > 0.7

def test_scitovski_eps_two_scale():
    rng = np.random.default_rng(0)
    dense = rng.normal(0.5, 0.01, (400, 2)); sparse = rng.uniform(0, 1, (100, 2))
    e = scitovski_eps(np.vstack([dense, sparse]), 5)
    assert e['r1'] < e['eps'] < e['r2'] and e['n1'] > e['n2']

def test_b_aki_recovers_b1():
    rng = np.random.default_rng(1)
    m = np.round(3.0 - 0.05 + rng.exponential(1 / np.log(10), 20000), 1)
    b, sb, n = b_aki(m, 3.0)
    assert abs(b - 1.0) < 0.05

def test_mc_maxc():
    rng = np.random.default_rng(2); m = np.round(2.95 + rng.exponential(0.43, 5000), 1)
    assert 2.9 <= mc_maxc(m) <= 3.3

def test_prune_relabels():
    lab = np.array([0] * 60 + [1] * 10 + [2] * 55 + [-1] * 5)
    p = prune(lab, 50); assert set(np.unique(p)) == {-1, 0, 1} and (p == -1).sum() == 15

def test_dbscan_hdbscan_find_blobs():
    rng = np.random.default_rng(3)
    XY = np.vstack([rng.normal((0, 0), 3, (200, 2)), rng.normal((100, 0), 3, (200, 2)), rng.uniform(-50, 150, (50, 2))])
    for m in ('DBKM', 'HDB'):
        lab = run(m, XY); assert lab.max() + 1 == 2

def test_grid_ari_identity():
    rng = np.random.default_rng(4); XY = rng.uniform(0, 200, (500, 2)); lab = (XY[:, 0] > 100).astype(int)
    g = grid_labels(XY, lab); a, n = grid_ari(g, g); assert a == 1.0

def test_projection_km():
    P = project(np.array([28.0, 29.0]), np.array([38.5, 38.5]))
    assert abs(np.hypot(*(P[1] - P[0])) - 87.1) < 1.0
