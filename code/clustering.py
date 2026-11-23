"""Clustering methods and comparison metrics."""
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score
from common import scitovski_eps
REGN = dict(lon0=26.0, lon1=31.0, lat0=36.0, lat1=41.0)
MINPTS = 5          # 4 neighbours + the point itself (convention that reproduced Scitovski's eps)
MIN_ZONE = 50       # a priori minimum zone size applied identically to every method

def prune(lab, min_zone=MIN_ZONE):
    lab = lab.copy(); ids, c = np.unique(lab[lab >= 0], return_counts=True)
    for i, n in zip(ids, c):
        if n < min_zone: lab[lab == i] = -1
    u = {v: k for k, v in enumerate(np.unique(lab[lab >= 0]))}
    return np.array([u.get(v, -1) for v in lab])

def run(method, XY, lonlat=None, eps_scale=1.0, minpts=MINPTS, mcs=MIN_ZONE, xi=0.05, eps=None, return_eps=False):
    if method == 'SCIT':
        B = np.c_[(lonlat[:, 0] - REGN['lon0']) / (REGN['lon1'] - REGN['lon0']), (lonlat[:, 1] - REGN['lat0']) / (REGN['lat1'] - REGN['lat0'])]
        e = eps if eps is not None else scitovski_eps(B, minpts)['eps']
        lab = DBSCAN(eps=e * eps_scale, min_samples=minpts).fit_predict(B)
    elif method == 'DBKM':
        e = eps if eps is not None else scitovski_eps(XY, minpts)['eps']
        lab = DBSCAN(eps=e * eps_scale, min_samples=minpts).fit_predict(XY)
    elif method == 'HDB':
        e = None; lab = HDBSCAN(min_cluster_size=mcs, min_samples=minpts, copy=True).fit_predict(XY)
    elif method == 'OPT':
        e = None; lab = OPTICS(min_samples=minpts, xi=xi, min_cluster_size=mcs).fit_predict(XY)
    lab = prune(lab, mcs)
    return (lab, e) if return_eps else lab

def ari(a, b): return adjusted_rand_score(a, b)

def jaccard_stability(ref, rep, idx):
    """Hennig (2007): for each reference zone, max Jaccard with any replicate zone on shared points idx."""
    r = ref[idx]; out = {}
    for z in np.unique(ref[ref >= 0]):
        A = set(np.where(r == z)[0]); best = 0.0
        for w in np.unique(rep[rep >= 0]):
            Bs = set(np.where(rep == w)[0]); j = len(A & Bs) / max(1, len(A | Bs)); best = max(best, j)
        out[z] = best
    return out

def grid_labels(XY, lab, cell=10.0):
    gx = np.floor(XY[:, 0] / cell).astype(int); gy = np.floor(XY[:, 1] / cell).astype(int); res = {}
    for a, b, l in zip(gx, gy, lab): res.setdefault((a, b), []).append(l)
    return {k: max(set(v), key=v.count) for k, v in res.items()}

def grid_ari(g1, g2):
    ks = sorted(set(g1) & set(g2))
    if len(ks) < 10: return np.nan, len(ks)
    return adjusted_rand_score([g1[k] for k in ks], [g2[k] for k in ks]), len(ks)
