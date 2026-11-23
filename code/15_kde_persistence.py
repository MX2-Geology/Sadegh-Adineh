"""wanat3 Analysis 6: persistence without re-clustering. For each raw zone (default configuration of each method), ratio of the
P_bg-weighted Gaussian kernel density (sigma 10 km, 2 km raster) averaged inside the zone (convex hull + 5 km) to that averaged over
the rest of the study region; 95 % CI from B = 200 event-bootstrap (multinomial) replicates. Computed for ETAS hom30 and ker30 weights,
and for unit weights (raw density) for comparison. Ratio ~1: background density inside the zone is not elevated."""
import numpy as np, pandas as pd, json
from scipy.ndimage import gaussian_filter
from common import *
from wa3 import *
A = load_A(); XY = A[['x', 'y']].values; rng = np.random.default_rng(SEED)
tgt = A[[c for c in A.columns if c.startswith('P_bg_hom30')]].notna().values.ravel()   # target-period events only
X0, X1 = XY[:, 0].min() - 30, XY[:, 0].max() + 30; Y0, Y1 = XY[:, 1].min() - 30, XY[:, 1].max() + 30; h = 2.0
ex = np.arange(X0, X1 + h, h); ey = np.arange(Y0, Y1 + h, h); cx = (ex[:-1] + ex[1:]) / 2; cy = (ey[:-1] + ey[1:]) / 2
GX, GY = np.meshgrid(cx, cy, indexing='ij'); G = np.c_[GX.ravel(), GY.ravel()]
reg = inside(REGPOLY, G)
def dens(w): H, _, _ = np.histogram2d(XY[tgt, 0], XY[tgt, 1], [ex, ey], weights=w); return gaussian_filter(H, 10.0 / h).ravel()
rows = []; B = 200
masks = {}
for m in ['SCIT', 'DBKM', 'HDB', 'OPT']:
    lab = A[f'z_{m}'].values
    for z in np.unique(lab[lab >= 0]): masks[(m, z)] = inside(hull(XY[lab == z]), G) & reg
W = {'ETAS_hom30': A.P_bg_hom30.values[tgt], 'ETAS_ker30': A.P_bg_ker30.values[tgt] if 'P_bg_ker30' in A else None, 'raw': np.ones(tgt.sum())}
W = {k: np.nan_to_num(v, nan=0.0) for k, v in W.items() if v is not None}
boots = {k: [] for k in W}
for k, w in W.items():
    D0 = dens(w); n = len(w)
    for b in range(B):
        c = rng.multinomial(n, np.full(n, 1 / n)); boots[k].append(dens(w * c))
    for (m, z), mk in masks.items():
        out = reg & ~mk; r0 = D0[mk].mean() / D0[out].mean()
        rb = np.array([Db[mk].mean() / Db[out].mean() for Db in boots[k]])
        rows.append(dict(method=m, zone=int(z), weights=k, ratio=round(r0, 2), ci_lo=round(np.quantile(rb, 0.025), 2), ci_hi=round(np.quantile(rb, 0.975), 2),
                         area_km2=int(mk.sum() * h * h)))
R = pd.DataFrame(rows); R.to_csv(f'{ROOT}/out/tables/T_kde_persistence.csv', index=False)
P = R.pivot_table(index=['method', 'zone'], columns='weights', values=['ratio', 'ci_lo', 'ci_hi']); print(P.round(2).to_string())
s = R[R.weights == 'ETAS_hom30']; print('zones with ETAS-weighted ratio CI_lo>1:', (s.ci_lo > 1).sum(), '/', len(s))
