"""Declustering: Zaliapin & Ben-Zion (2013) nearest-neighbour; Gardner & Knopoff (1974) windows."""
import numpy as np
from sklearn.mixture import GaussianMixture

def nn_zbz(t_yr, xy, m, b=1.0, df=1.6, seed=0):
    n = len(t_yr); eta = np.full(n, np.inf); par = np.full(n, -1); T = np.full(n, np.nan); R = np.full(n, np.nan)
    for j in range(1, n):
        dt = t_yr[j] - t_yr[:j]; ok = dt > 0
        if not ok.any(): continue
        r = np.hypot(*(xy[:j] - xy[j]).T); r = np.maximum(r, 0.1)  # 100 m floor for co-located epicentres
        tt = dt * 10 ** (-0.5 * b * m[:j]); rr = r ** df * 10 ** (-0.5 * b * m[:j])
        e = np.where(ok, dt * r ** df * 10 ** (-b * m[:j]), np.inf); i = np.argmin(e)
        eta[j] = e[i]; par[j] = i; T[j] = tt[i]; R[j] = rr[i]
    le = np.log10(eta[np.isfinite(eta)])
    gm = GaussianMixture(2, random_state=seed).fit(le[:, None])
    grid = np.linspace(le.min(), le.max(), 4000)[:, None]; p = gm.predict_proba(grid)
    lo = np.argmin(gm.means_.ravel())
    thr = float(grid[np.argmin(np.abs(p[:, lo] - 0.5)), 0])
    clustered = np.log10(eta) < thr
    return dict(eta=eta, parent=par, logT=np.log10(T), logR=np.log10(R), thr=thr, background=~clustered)

def gk74(t_day, xy, m):
    n = len(m); bg = np.ones(n, bool); order = np.argsort(-m)
    dwin = 10 ** (0.1238 * m + 0.983)
    twin = np.where(m >= 6.5, 10 ** (0.032 * m + 2.7389), 10 ** (0.5409 * m - 0.547))
    for i in order:
        if not bg[i]: continue
        dt = t_day - t_day[i]; r = np.hypot(*(xy - xy[i]).T)
        sel = (np.abs(dt) <= twin[i]) & (r <= dwin[i]) & (m <= m[i]); sel[i] = False
        # foreshocks: same window backwards (symmetric, standard 'double-sided' variant)
        bg[sel] = False
    return bg
