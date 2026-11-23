"""AUDIT synthetic ETAS benchmark (false-positive rate and power of the constructive support rule).
Simulates ETAS catalogues with the fitted hom30 parameters (Mc 3.0, b from beta), homogeneous background or homogeneous
background plus N_SRC stationary disc sources (radius R_SRC km, background-rate enhancement E). For each synthetic
catalogue: ETAS refit (same settings as 13_etas.py hom30), K=20 stochastic realisations, P_bg-weighted DBSCAN (eps 10/15/20 km)
and the DBSCAN realisation ensemble, bootstrap stability (B_BOOT replicates, 90% + 5 km jitter), and the period-B test against a
second, independent synthetic catalogue of 16 yr declustered with ZBZ (density-matched MinPts as in 17_constructive.py).
Also: KDE ratio of raw DBKM zones (P_bg weighted, as 15_kde_persistence.py, point estimate) -> selection-bias null.
Usage: python bench.py SCEN E REP   (SCEN in {H0, SRC}); writes bench/out/<SCEN>_E<E>_r<REP>.json"""
import sys, os, json, time, logging, datetime as dt
import numpy as np, pandas as pd
CODE = os.environ.get('WA_CODE', os.path.abspath('../../code')); sys.path.insert(0, CODE)
from common import project, scitovski_eps
from clustering import prune
from decluster import nn_zbz
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path
from shapely.geometry import MultiPoint
from etas.inversion import ETASParameterCalculation, expected_aftershocks, parameter_dict2array, branching_ratio
import types; _m = types.ModuleType("seismostats"); _m.ForecastCatalog = object; sys.modules.setdefault("seismostats", _m)  # stub: only used by unused forecast helpers
from etas.simulation import generate_aftershocks
from etas.mc_b_est import simulate_magnitudes
logging.basicConfig(level=logging.ERROR)
for n in ('etas', 'etas.inversion', 'etas.simulation'): logging.getLogger(n).setLevel(logging.ERROR)
SCEN, E, REP = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
B_BOOT = int(os.environ.get('B_BOOT', 50)); N_SRC = 3; R_SRC = 15.0; M_MAX = 7.5
OUT = os.environ.get('WA_BENCH_OUT', 'out'); os.makedirs(OUT, exist_ok=True)
seed = 1000 * (0 if SCEN == 'H0' else int(E)) + REP + 7; np.random.seed(seed); rng = np.random.default_rng(seed)
T = json.load(open(os.path.join(CODE, '../out/tables/T_etas_hom30.json')))
par = {k: (np.nan if v is None else v) for k, v in T['params'].items()}; par['log10_iota'] = None
beta = T['beta']; MC = 3.0; DM = 0.1
LON0, LON1, LAT0, LAT1 = 26.0, 31.0, 36.0, 41.0
mu_per_km2_day = 10 ** par['log10_mu']
REG = MultiPoint(project([26, 31, 31, 26], [36, 36, 41, 41])).convex_hull; REGP = Path(np.asarray(REG.exterior.coords))
from pyproj import Proj
PJ = Proj(proj='aeqd', lat_0=38.5, lon_0=28.0, datum='WGS84', units='km')
# source centres (projected km), fixed per replicate, >= 60 km inside the region, >= 120 km apart
src = []
if SCEN == 'SRC':
    while len(src) < N_SRC:
        p = rng.uniform([-200, -250], [200, 250])
        if REG.buffer(-60).contains(MultiPoint([p]).geoms[0]) and all(np.hypot(*(p - q)) > 120 for q in src): src.append(p)
src = np.array(src)
def lonlat(P): lo, la = PJ(P[:, 0], P[:, 1], inverse=True); return np.asarray(lo), np.asarray(la)
def background(t0, t1):
    days = (t1 - t0).days
    area = REG.area; nb = rng.poisson(mu_per_km2_day * area * days)
    P = []
    while len(P) < nb:
        q = rng.uniform([-260, -300], [260, 300], (nb * 2, 2)); q = q[REGP.contains_points(q)]; P.extend(q.tolist())
    P = np.array(P[:nb]); lab = np.full(len(P), -1)
    for i, c in enumerate(src):   # extra (E-1) x regional rate inside each disc
        ne = rng.poisson((E - 1) * mu_per_km2_day * np.pi * R_SRC ** 2 * days)
        r = R_SRC * np.sqrt(rng.random(ne)); th = rng.uniform(0, 2 * np.pi, ne)
        P = np.vstack([P, c + np.c_[r * np.cos(th), r * np.sin(th)]]); lab = np.r_[lab, np.full(ne, i)]
    lo, la = lonlat(P)
    cat = pd.DataFrame(dict(latitude=la, longitude=lo, time=[t0 + dt.timedelta(days=float(d)) for d in rng.uniform(0, days, len(P))],
                            magnitude=simulate_magnitudes(len(P), beta=beta, mc=MC - DM / 2, m_max=M_MAX + DM / 2), parent=0, generation=0, is_background=True))
    cat['src'] = lab
    cat = cat.sort_values('time').reset_index(drop=True); cat.index += 1; cat['gen_0_parent'] = cat.index
    th = parameter_dict2array(par)[2:]
    cat['expected_n_aftershocks'] = expected_aftershocks(cat['magnitude'], params=[th, MC - DM / 2], no_start=True, no_end=True)
    cat['n_aftershocks'] = np.random.poisson(cat['expected_n_aftershocks'])
    return cat
def simulate(t0, t1):
    cat = background(t0, t1); g = 0; L = (t1 - t0).days
    while True:
        s = cat.query('generation == @g and n_aftershocks > 0').copy()
        if len(s) == 0: break
        a = generate_aftershocks(s, g, par, beta, MC, delta_m=DM, m_max=M_MAX, timewindow_end=t1, timewindow_length=L)
        a.index += cat.index.max() + 1; a['src'] = -1; cat = pd.concat([cat, a], sort=True); g += 1
        if len(cat) > 60000: break
    cat = cat[(cat.longitude.between(LON0, LON1)) & (cat.latitude.between(LAT0, LAT1))].copy()
    cat['magnitude'] = np.round(cat.magnitude / DM) * DM
    return cat.sort_values('time').reset_index(drop=True)
t0 = time.time()
A = simulate(dt.datetime(2016, 1, 1), dt.datetime(2026, 10, 14))
Bc = simulate(dt.datetime(2000, 1, 1), dt.datetime(2016, 1, 1))
res = dict(t_sim=round(time.time()-t0), scen=SCEN, E=E, rep=REP, seed=seed, n_A=len(A), n_B=len(Bc), src=src.tolist(), Mmax_A=float(A.magnitude.max()))
# ---- ETAS refit on synthetic A (as 13_etas.py hom30)
cat = pd.DataFrame({'time': pd.to_datetime(A.time), 'latitude': A.latitude, 'longitude': A.longitude, 'magnitude': A.magnitude}); cat.index = pd.Index(np.arange(len(A)), name='id')
meta = dict(name='bench', catalog=cat, auxiliary_start='2016-01-01', timewindow_start='2016-07-01', timewindow_end='2026-10-14', mc=MC, m_ref=MC, delta_m=DM,
            coppersmith_multiplier=100, shape_coords=[[36, 26], [36, 31], [41, 31], [41, 26]], earth_radius=6.3781e3, bw_sq=100.0, free_background=False,
            free_productivity=False, n_iterations=None)
try:
    calc = ETASParameterCalculation(meta); calc.prepare(); th = calc.invert()
    pb = calc.target_events['P_background'].reindex(cat.index).values; res['exp_bg_fit'] = float(np.nansum(pb))
    thd = dict(th); res['theta_fit'] = {k: (None if v is None else float(v)) for k, v in thd.items()}; res['beta_fit'] = float(calc.beta)
    res['alpha_eff_fit'] = float(thd['a'] - thd['rho'] * thd['gamma'])
    try: res['br_fit'] = float(branching_ratio(parameter_dict2array(thd), calc.beta))
    except AssertionError: res['br_fit'] = None; res['br_note'] = 'alpha_eff >= beta: branching ratio undefined for unlimited magnitudes'
except Exception as e:
    res['error'] = str(e)[:200]; json.dump(res, open(f'{OUT}/{SCEN}_E{E:g}_r{REP}.json', 'w')); sys.exit()
res['t_fit'] = round(time.time()-t0)
tg = ~np.isnan(pb); Pbg = np.nan_to_num(pb, nan=0.0)
res['true_bg_target'] = int((A.is_background.values & tg).sum())
XY = project(A.longitude.values, A.latitude.values); n = len(A)
Rk = (rng.random((20, n)) < np.nan_to_num(pb, nan=1.0)) & tg
XYB = project(Bc.longitude.values, Bc.latitude.values)
tB = (pd.to_datetime(Bc.time) - pd.to_datetime(Bc.time).iloc[0]).dt.total_seconds().values / 3.15576e7
bgB = nn_zbz(tB, XYB, Bc.magnitude.values, seed=0)['background']; XB0 = XYB[bgB]
CELL = 10.0
def cells(P): return [tuple(c) for c in np.floor(P / CELL).astype(int)]
def best_j(S, others): return max([len(S & T) / len(S | T) for T in others] + [0.0])
def cl(P, mcs=25):
    if len(P) < mcs: return np.full(len(P), -1)
    e = scitovski_eps(P, 5)['eps']; return prune(DBSCAN(eps=e, min_samples=5).fit_predict(P), mcs)
def ensemble(R, P, Pb):
    K = R.shape[0]; cid = cells(P); U = sorted(set(cid)); ix = {c: i for i, c in enumerate(U)}; ci = np.array([ix[c] for c in cid])
    obs = np.zeros((K, len(U)), bool); lab_c = np.full((K, len(U)), -1)
    for k in range(K):
        s = np.where(R[k])[0]; lb = cl(P[s]); obs[k, ci[s]] = True
        for z in np.unique(lb[lb >= 0]):
            cc = np.unique(ci[s[lb == z]]); lab_c[k, cc] = z
    nobs = obs.sum(0); Pcell = np.where(nobs > 0, (lab_c >= 0).sum(0) / np.maximum(nobs, 1), 0)
    cand = np.where((Pcell >= 0.5) & (nobs >= K / 4))[0]
    if len(cand) == 0: return {}
    L = lab_c[:, cand]; both = (L[:, :, None] >= 0) & (L[:, None, :] >= 0)
    co = ((L[:, :, None] == L[:, None, :]) & both).sum(0) / np.maximum(both.sum(0), 1)
    cc = np.array([U[i] for i in cand]); nb = (np.abs(cc[:, None, :] - cc[None, :, :]).max(2) <= 1)
    ncomp, comp = connected_components(csr_matrix((co >= 0.5) & nb), directed=False); zc = {}; zid = 0
    for c in range(ncomp):
        mem = cand[comp == c]; ev = np.isin(ci, mem)
        if np.nansum(Pb[ev]) >= 25:
            for m in mem: zc[U[m]] = zid
            zid += 1
    return zc
def cellsets(zc):
    o = {}
    for c, z in zc.items(): o.setdefault(z, set()).add(c)
    return o
REP_ = [np.sort(rng.choice(n, int(0.9 * n), replace=False)) for _ in range(B_BOOT)]; REP_ = [(i, rng.normal(0, 5.0, (len(i), 2))) for i in REP_]
zones = []
def match(ev):
    if len(src) == 0: return -1
    c = XY[ev].mean(0); d = np.hypot(*(src - c).T); return int(np.argmin(d)) if d.min() <= R_SRC + 10 else -1
# (a) ensemble DBKM
zc = ensemble(Rk, XY, Pbg); sets = cellsets(zc); zev = np.array([zc.get(c, -1) for c in cells(XY)])
jbe = {z: [] for z in sets}
for idx, jit in REP_:
    P2 = XY[idx] + jit; z2c = ensemble(Rk[:, idx], P2, Pbg[idx]); z2 = np.array([z2c.get(c, -1) for c in cells(P2)]); zr = zev[idx]
    for z in sets:
        a = set(np.where(zr == z)[0]); jbe[z].append(max([len(a & set(np.where(z2 == w)[0])) / len(a | set(np.where(z2 == w)[0])) for w in np.unique(z2[z2 >= 0])] + [0.0]))
Bsets = [set(cells(XB0[l == w])) for l in [cl(XB0)] for w in np.unique(l[l >= 0])]
for z, S in sets.items():
    ev = zev == z; zones.append(dict(variant='ens-DBKM', zone=int(z), n=int(ev.sum()), exp_bg=float(Pbg[ev].sum()), J_boot=float(np.mean(jbe[z])),
                                      J_B=best_j(S, Bsets), src=match(ev)))
res['t_ens'] = round(time.time()-t0)
# (b) weighted DBSCAN
for eps in (10.0, 15.0, 20.0):
    lab = np.full(n, -1); lab[tg] = prune(DBSCAN(eps=eps, min_samples=5).fit_predict(XY[tg], sample_weight=Pbg[tg])); Z0 = np.unique(lab[lab >= 0])
    jb = {z: [] for z in Z0}
    for idx, jit in REP_:
        t2 = tg[idx]; l2 = np.full(len(idx), -1); l2[t2] = prune(DBSCAN(eps=eps, min_samples=5).fit_predict((XY[idx] + jit)[t2], sample_weight=Pbg[idx][t2])); r = lab[idx]
        for z in Z0:
            a = set(np.where(r == z)[0]); jb[z].append(max([len(a & set(np.where(l2 == w)[0])) / len(a | set(np.where(l2 == w)[0])) for w in np.unique(l2[l2 >= 0])] + [0.0]))
    msB = int(round(5 * len(XB0) / Pbg[tg].sum())); lB = prune(DBSCAN(eps=eps, min_samples=msB).fit_predict(XB0), 25)
    SB = [set(cells(XB0[lB == w])) for w in np.unique(lB[lB >= 0])]
    for z in Z0:
        ev = lab == z; zones.append(dict(variant=f'wDB{eps:g}', zone=int(z), n=int(ev.sum()), exp_bg=float(Pbg[ev].sum()), J_boot=float(np.mean(jb[z])),
                                          J_B=best_j(set(cells(XY[ev])), SB), src=match(ev), msB=msB))
    # oracle: true background labels as weights (upper bound on power)
    w = (A.is_background.values & tg).astype(float); lo_ = np.full(n, -1); lo_[tg] = prune(DBSCAN(eps=eps, min_samples=5).fit_predict(XY[tg], sample_weight=w[tg]))
    for z in np.unique(lo_[lo_ >= 0]):
        ev = lo_ == z; zones.append(dict(variant=f'oracle-wDB{eps:g}', zone=int(z), n=int(ev.sum()), exp_bg=float(w[ev].sum()), J_boot=np.nan, J_B=np.nan, src=match(ev)))
for zz in zones: zz['supported'] = bool(zz['J_boot'] >= 0.75 and zz['J_B'] >= 0.5) if not np.isnan(zz['J_boot']) else None
res['t_wdb'] = round(time.time()-t0)
res['zones'] = zones
# ---- KDE ratio of raw DBKM zones (selection-bias null under homogeneous truth)
lab = prune(DBSCAN(eps=scitovski_eps(XY, 5)['eps'], min_samples=5).fit_predict(XY))
h = 2.0; ex = np.arange(XY[:, 0].min() - 30, XY[:, 0].max() + 32, h); ey = np.arange(XY[:, 1].min() - 30, XY[:, 1].max() + 32, h)
cx = (ex[:-1] + ex[1:]) / 2; cy = (ey[:-1] + ey[1:]) / 2; GX, GY = np.meshgrid(cx, cy, indexing='ij'); G = np.c_[GX.ravel(), GY.ravel()]; reg = REGP.contains_points(G)
H, _, _ = np.histogram2d(XY[tg, 0], XY[tg, 1], [ex, ey], weights=Pbg[tg]); D0 = gaussian_filter(H, 10.0 / h).ravel()
kr = []
for z in np.unique(lab[lab >= 0]):
    hp = MultiPoint(XY[lab == z]).convex_hull.buffer(5.0); mk = Path(np.asarray(hp.exterior.coords)).contains_points(G) & reg
    kr.append(float(D0[mk].mean() / D0[reg & ~mk].mean()))
res['kde_ratios_raw_DBKM'] = kr; res['n_raw_DBKM_zones'] = len(kr); res['runtime_s'] = round(time.time() - t0)
with open(f'{OUT}/{SCEN}_E{E:g}_r{REP}.json', 'w') as f: json.dump(res, f, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'zones'}, default=float))
