"""Tests for Rough-DBSCAN and the MBS completeness estimator (run from code/: python -m pytest -q tests)."""
import sys, os, ast, numpy as np
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, D)
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from rough_dbscan import rough_dbscan, leaders
from common import b_aki

def _blobs(rng):
    c = np.array([[0, 0], [10, 0], [0, 10]])
    return np.vstack([rng.normal(ci, 0.6, (150, 2)) for ci in c] + [rng.uniform(-5, 15, (60, 2))])

def test_leaders_cover_all_points_within_tau():
    rng = np.random.default_rng(0); X = _blobs(rng); L, cnt, lab = leaders(X, 0.5)
    assert cnt.sum() == len(X) and (np.hypot(*(X - L[lab]).T) <= 0.5 + 1e-12).all()

def test_rough_dbscan_tau0_equals_exact():
    rng = np.random.default_rng(1); X = _blobs(rng)
    ex = DBSCAN(eps=1.0, min_samples=5).fit_predict(X); rl, nl = rough_dbscan(X, 1.0, 5, tau=0.0)
    assert nl == len(X) and adjusted_rand_score(ex, rl) == 1.0

def test_rough_dbscan_small_tau_close_to_exact():
    rng = np.random.default_rng(2); X = _blobs(rng)
    ex = DBSCAN(eps=1.0, min_samples=5).fit_predict(X); rl, nl = rough_dbscan(X, 1.0, 5, tau=0.25)
    assert nl < len(X) and adjusted_rand_score(ex, rl) > 0.9

def _load_mbs():
    src = open(os.path.join(D, '10_catalogue.py')).read(); tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mbs'][0]
    ns = {'np': np, 'b_aki': b_aki}; exec(compile(ast.Module([fn], []), 'mbs', 'exec'), ns); return ns['mbs']

def test_mbs_recovers_completeness():
    rng = np.random.default_rng(3); mbs = _load_mbs()
    m = 1.5 + rng.exponential(1 / np.log(10), 60000)                       # GR with b = 1 above 1.5
    keep = rng.uniform(size=m.size) < np.clip((m - 1.5) / 1.0, 0, 1)       # linear detection ramp 1.5-2.5
    mc = mbs(np.round(m[keep], 1))
    assert 2.2 <= mc <= 2.9
