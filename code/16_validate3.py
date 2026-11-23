"""wanat3 validation of raw zones (default configuration of each method) on catalogue A (AFAD):
 zone triggered fractions (ETAS hom30/ker30, ZBZ, GK74), ETAS re-clustering persistence (K=20), Hennig Jaccard (B=200, from 14),
 KDE ratio (from 15), stable/persistent labels by the pre-set rule (J >= 0.75 and persistence >= 0.5); fault tests for raw-zone events
 AND for ETAS background events; GNSS strain (Mann-Whitney only; the Kruskal-Wallis test of wanat2 is deleted); GCMT T-axes;
 axis-vs-fault; 3-D check; short-term aftershock incompleteness (STAI) audit; b-values (Supplement)."""
import numpy as np, pandas as pd, json
from shapely.geometry import Point, MultiPoint
from scipy.stats import mannwhitneyu, binomtest
from statsmodels.stats.multitest import multipletests
from common import *
from clustering import run, ari, jaccard_stability
from wa3 import *
rng = np.random.default_rng(SEED)
A = load_A(); XY = A[['x', 'y']].values; LL = A[['lon', 'lat']].values; A['time'] = pd.to_datetime(A.time, utc=True, format='ISO8601')
A['dfault'] = fault_dist(XY)
C = pd.read_csv(f'{ROOT}/out/tables/T_fair_stability_configs.csv'); ZJ = pd.read_csv(f'{ROOT}/out/tables/S_fair_stability_zone_jaccard.csv')
K = pd.read_csv(f'{ROOT}/out/tables/T_kde_persistence.csv')
M = ['SCIT', 'DBKM', 'HDB', 'OPT']; DEF = {r.method: int(r.cfg) for r in C[C['default']].itertuples()}
BEST = C[C['default']].set_index('method').ARI_all_med.idxmax()          # same a-priori rule as wanat2 (default configs)
BEST_CLU = C[C['default']].set_index('method').ARI_clu_med.idxmax()
out = dict(best_method_ARI_all=BEST, best_method_ARI_clustered=BEST_CLU)
RE = {t: real(t) for t in ('hom30', 'ker30') if f'P_bg_{t}' in A}
# ---------------- zone table for all methods
rows = []
for m in M:
    lab = A[f'z_{m}'].values
    pers = {}
    for t, R in RE.items():
        vals = {z: [] for z in np.unique(lab[lab >= 0])}
        for k in range(R.shape[0]):
            s = np.where(R[k])[0]; lb = run(m, XY[s], LL[s]); full = np.full(len(A), -1); full[s] = lb
            j = jaccard_stability(np.where(R[k], lab, -1), full, np.arange(len(A)))
            for z in vals: vals[z].append(j.get(z, 0.0))
        pers[t] = {z: float(np.mean(v)) for z, v in vals.items()}
    for z in np.unique(lab[lab >= 0]):
        E = A[lab == z]; zj = ZJ[(ZJ.cfg == DEF[m]) & (ZJ.zone == z)].iloc[0]
        kd = K[(K.method == m) & (K.zone == z) & (K.weights == 'ETAS_hom30')].iloc[0]
        big = E.loc[E.M.idxmax()]
        r = dict(method=m, zone=int(z), n=len(E), lon=round(E.lon.mean(), 2), lat=round(E.lat.mean(), 2), Mmax=float(E.M.max()),
                 Mmax_date=str(big.time.date()), J_mean=round(zj.J_mean, 3), J_p10=round(zj.J_p10, 3),
                 Ptr_ETAS_hom=round(float(1 - np.nanmean(E.P_bg_hom30)), 3),
                 Ptr_ETAS_ker=round(float(1 - np.nanmean(E.P_bg_ker30)), 3) if 'P_bg_ker30' in A else np.nan,
                 Ptr_ZBZ=round(float(1 - E.bg_zbz.mean()), 3), Ptr_GK=round(float(1 - E.bg_gk.mean()), 3),
                 persist_hom=round(pers['hom30'][z], 3), persist_ker=round(pers['ker30'][z], 3) if 'ker30' in pers else np.nan,
                 kde_ratio=kd.ratio, kde_lo=kd.ci_lo, kde_hi=kd.ci_hi, frac_within5km_fault=round(float(np.mean(E.dfault <= 5)), 3))
        r['stable'] = r['J_mean'] >= 0.75; r['persistent'] = r['persist_hom'] >= 0.5; r['stable_and_persistent'] = r['stable'] and r['persistent']
        rows.append(r)
Z = pd.DataFrame(rows); Z.to_csv(f'{ROOT}/out/tables/T_zones_all_methods.csv', index=False)
out['zone_counts'] = {m: dict(n=int((Z.method == m).sum()), stable=int(Z[Z.method == m].stable.sum()), persistent=int(Z[Z.method == m].persistent.sum()),
                              stable_and_persistent=int(Z[Z.method == m].stable_and_persistent.sum())) for m in M}
# ---------------- fault test for raw zones (rotation/translation null, 200 draws)
def null_f5(lab, nd=200, R=50.0):
    vals = []; zs = [z for z in np.unique(lab) if z >= 0]
    for _ in range(nd):
        P = []
        for z in zs:
            Q = XY[lab == z]; c = Q.mean(0)
            for _try in range(50):
                th = rng.uniform(0, 2 * np.pi); rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                Qn = (Q - c) @ rot.T + c + rng.uniform(-R, R, 2)
                if inside(REGPOLY, Qn[::10]).mean() >= 0.8: break
            P.append(Qn)
        vals.append(np.mean(fault_dist(np.vstack(P)) <= 5.0))
    return np.array(vals)
ft = []
for m in M:
    lab = A[f'z_{m}'].values; f5 = np.mean(A.dfault[lab >= 0] <= 5); nv = null_f5(lab)
    ft.append(dict(test='raw zones', method=m, f5=round(f5, 3), f5_noise=round(float(np.mean(A.dfault[lab < 0] <= 5)), 3), null_mean=round(nv.mean(), 3),
                   null_p95=round(np.quantile(nv, 0.95), 3), p=round((1 + (nv >= f5).sum()) / (1 + len(nv)), 4)))
# ---------------- fault test for ETAS background events: per realisation, fraction within 5 km vs null of the same number of points
# drawn uniformly from the 'sampled area' (union of 25 km discs around all catalogue-A epicentres, inside the region)
ex = np.arange(XY[:, 0].min(), XY[:, 0].max(), 2.0); ey = np.arange(XY[:, 1].min(), XY[:, 1].max(), 2.0)
GX, GY = np.meshgrid(ex, ey); G = np.c_[GX.ravel(), GY.ravel()]
from sklearn.neighbors import KDTree
near = KDTree(XY).query(G, k=1)[0].ravel() <= 25; G = G[near & inside(REGPOLY, G)]; dG = fault_dist(G)
for t, R in RE.items():
    f5r = np.array([np.mean(A.dfault.values[R[k]] <= 5) for k in range(R.shape[0])]); nb = int(R.sum(1).mean())
    nul = np.array([np.mean(dG[rng.choice(len(dG), nb)] <= 5) for _ in range(1000)])
    trig = ~np.isnan(A[f'P_bg_{t}'].values); w = A[f'P_bg_{t}'].values[trig]; dd = A.dfault.values[trig] <= 5
    ft.append(dict(test=f'ETAS background ({t})', method='-', f5=round(float(np.median(f5r)), 3), f5_noise=round(float(np.sum((1 - w) * dd) / np.sum(1 - w)), 3),
                   null_mean=round(nul.mean(), 3), null_p95=round(np.quantile(nul, 0.95), 3),
                   p=round((1 + (nul >= np.median(f5r)).sum()) / (1 + len(nul)), 4), f5_realisation_range=[round(f5r.min(), 3), round(f5r.max(), 3)], n_bg_mean=nb))
f5all = np.mean(A.dfault <= 5); nul = np.array([np.mean(dG[rng.choice(len(dG), len(A))] <= 5) for _ in range(1000)])
ft.append(dict(test='all catalogue-A events', method='-', f5=round(f5all, 3), null_mean=round(nul.mean(), 3), null_p95=round(np.quantile(nul, 0.95), 3),
               p=round((1 + (nul >= f5all).sum()) / (1 + len(nul)), 4)))
ft = pd.DataFrame(ft); ft.to_csv(f'{ROOT}/out/tables/T_fault_test.csv', index=False)
out['note_fault_test'] = "f5_noise column: noise events for raw zones; P_triggered-weighted fraction for ETAS rows"
# ---------------- BEST-method domains: GNSS, GCMT, geometry, depth, b (Supplement)
lab = A[f'z_{BEST}'].values; L, _ = fault_lines()
grid = pd.read_csv(f'{ROOT}/out/data/gnss_strain_grid.csv'); GP = project(grid.lon, grid.lat)
gc = pd.read_csv(f'{ROOT}/out/data/gcmt_box.csv'); GC = project(gc.lon, gc.lat)
def axial_mean(az, w=None):
    a = np.radians(np.asarray(az) * 2); w = np.ones(len(a)) if w is None else w
    return (np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 2) % 180
def axdiff(a, b): d = np.abs((np.asarray(a) - np.asarray(b)) % 180); return np.minimum(d, 180 - d)
D = []; tgt_years = (A.time.max() - pd.Timestamp('2016-07-01', tz='UTC')).days / 365.25
for z in np.unique(lab[lab >= 0]):
    E = A[lab == z]; Q = E[['x', 'y']].values; hb = MultiPoint(Q).convex_hull.buffer(10)
    w, v = np.linalg.eigh(np.cov(Q.T)); az = np.degrees(np.arctan2(v[0, 1], v[1, 1])) % 180; elong = np.sqrt(w[1] / max(w[0], 1e-9))
    segs = []
    for Lq in L:
        if Lq.intersects(hb):
            I = Lq.intersection(hb)
            for pp in (I.geoms if hasattr(I, 'geoms') else [I]):
                c = np.array(pp.coords)
                for s0, s1 in zip(c[:-1], c[1:]): d = s1 - s0; segs.append((np.degrees(np.arctan2(d[0], d[1])) % 180, np.hypot(*d)))
    fs = axial_mean([s[0] for s in segs], np.array([s[1] for s in segs])) if segs else np.nan
    gz = grid[inside(hb, GP)]; gcz = gc[inside(hb, GC)]; nm = gcz[gcz['style'].isin(['normal', 'oblique'])]
    gaz = [grid.ext_az.values[np.argmin(np.hypot(*(GP - p).T))] for p in project(nm.lon, nm.lat)] if len(nm) else []
    dz = E.loc[~E.depth_fixed.astype(bool), 'depth']
    m_ = np.round(E.M.values, 1); mc = max(3.0, mc_maxc(m_)); b, sb, nb = b_aki(m_, mc)
    bb = [b_aki(rng.choice(m_[m_ >= mc - 0.05], nb), mc)[0] for _ in range(500)] if nb >= 30 else [np.nan]
    dm = np.diff(m_); dpos = dm[dm >= 0.2 - 1e-9]; bpos = np.log10(np.e) / (dpos.mean() - 0.15) if len(dpos) > 20 else np.nan
    ti = np.sort(E.time.values.astype('int64') / 8.64e13); it = np.diff(ti); it = it[it > 0]
    D.append(dict(zone=int(z), n=len(E), area_km2=round(MultiPoint(Q).convex_hull.area), zone_axis_az=round(az), elongation=round(elong, 2),
                  fault_strike=round(fs) if segs else np.nan, axis_vs_fault=round(float(axdiff(az, fs))) if segs else np.nan,
                  n_gnss_nodes=len(gz), I2=round(gz.I2.mean()) if len(gz) else np.nan, gnss_ext_az=round(axial_mean(gz.ext_az)) if len(gz) else np.nan,
                  n_gcmt=len(gcz), n_gcmt_normal=int((gcz['style'] == 'normal').sum()), T_az=round(axial_mean(nm.T_az)) if len(nm) else np.nan,
                  T_vs_gnss=round(float(np.median(axdiff(nm.T_az.values, gaz)))) if len(nm) else np.nan,
                  n_depth_free=len(dz), D50=round(dz.median(), 1) if len(dz) > 10 else np.nan, D90=round(dz.quantile(0.9), 1) if len(dz) > 10 else np.nan,
                  Mc=mc, n_above_Mc=nb, b=round(b, 2), b_lo=round(np.nanpercentile(bb, 2.5), 2) if nb >= 30 else np.nan,
                  b_hi=round(np.nanpercentile(bb, 97.5), 2) if nb >= 30 else np.nan, b_positive=round(bpos, 2) if np.isfinite(bpos) else np.nan,
                  CV=round(it.std() / it.mean(), 2) if len(it) > 5 else np.nan,
                  bg_rate_per_1e4km2_yr=round(np.nansum(E.P_bg_hom30) / tgt_years / max(MultiPoint(Q).convex_hull.area, 1) * 1e4, 2)))
D = pd.DataFrame(D).merge(Z[Z.method == BEST][['zone', 'stable', 'persistent']], on='zone'); D.to_csv(f'{ROOT}/out/tables/T_domains.csv', index=False)
I2ev = grid.I2.values[[np.argmin(np.hypot(*(GP - p).T)) for p in XY]]
out['I2_zone_vs_noise_MWU_p'] = float(mannwhitneyu(I2ev[lab >= 0], I2ev[lab < 0], alternative='greater').pvalue)
out['I2_median_zone_noise'] = [float(np.median(I2ev[lab >= 0])), float(np.median(I2ev[lab < 0]))]
gR = grid[grid.lon.between(26, 31) & grid.lat.between(36, 41)].I2.values; pv = []
for k in range(RE['hom30'].shape[0]): pv.append(mannwhitneyu(I2ev[RE['hom30'][k]], gR, alternative='greater').pvalue)
out['I2_background_vs_grid'] = dict(p_median=float(np.median(pv)), p_max=float(np.max(pv)), median_I2_grid=float(np.median(gR)),
                                    median_I2_bg=float(np.median([np.median(I2ev[RE['hom30'][k]]) for k in range(RE['hom30'].shape[0])])))
sub = D.dropna(subset=['fault_strike']); sub = sub[sub.elongation >= 1.5]
out['axis_vs_fault_elongated'] = dict(n=int(len(sub)), median_deg=float(sub.axis_vs_fault.median()) if len(sub) else None,
                                      p_binomial_lt45=float(binomtest(int((sub.axis_vs_fault < 45).sum()), len(sub), 0.5, alternative='greater').pvalue) if len(sub) else None)
tg = D.dropna(subset=['T_vs_gnss']); out['T_vs_gnss'] = dict(n=int(len(tg)), median=float(tg.T_vs_gnss.median()) if len(tg) else None,
                                                             n_le20=int((tg.T_vs_gnss <= 20).sum()), range=[float(tg.T_vs_gnss.min()), float(tg.T_vs_gnss.max())] if len(tg) else None)
# Utsu pairwise b tests (Supplement), among BEST zones with >= 50 events above Mc
pairs = []; Rr = D[D.n_above_Mc >= 50]
for i, a in Rr.iterrows():
    for j, c in Rr.iterrows():
        if a.zone >= c.zone: continue
        n1, n2, b1, b2 = a.n_above_Mc, c.n_above_Mc, a.b, c.b; N = n1 + n2
        dA = -2 * N * np.log(N) + 2 * n1 * np.log(n1 + n2 * b1 / b2) + 2 * n2 * np.log(n1 * b2 / b1 + n2) - 2
        pairs.append(dict(z1=a.zone, z2=c.zone, b1=b1, b2=b2, dAIC=round(dA, 2), p_utsu=min(1.0, float(np.exp(-dA / 2 - 2)))))
pairs = pd.DataFrame(pairs)
if len(pairs): pairs['p_BH'] = multipletests(pairs.p_utsu, method='fdr_bh')[1]; pairs['sig'] = pairs.p_BH < 0.05
pairs.to_csv(f'{ROOT}/out/tables/S_b_pairwise.csv', index=False)
out['b_pairs'] = dict(n=int(len(pairs)), sig=int(pairs.sig.sum()) if len(pairs) else 0, b_range=[float(D.b.min()), float(D.b.max())])
# 3-D check (HDBSCAN on free-depth events <= 40 km)
s = (~A.depth_fixed.astype(bool)) & (A.depth <= 40); l3 = run('HDB', np.c_[XY[s], A.depth.values[s]]); l2 = run('HDB', XY[s])
out['3D'] = dict(n=int(s.sum()), zones_3D=int(l3.max() + 1), zones_2D=int(l2.max() + 1), ARI=round(ari(l3, l2), 3))
# STAI audit: Helmstetter et al. (2006) Mc(t) = M - 4.5 - 0.75 log10(t[days]); window until Mc(t) <= Mc=3.0; plus empirical first-day Mc
MS = A[A.M >= 5.5].sort_values('time'); st = []
for _, e in MS.iterrows():
    tstar = 10 ** ((e.M - 4.5 - 3.0) / 0.75)   # days
    dt = (A.time - e.time).dt.total_seconds() / 86400; r = np.hypot(*(XY - [e.x, e.y]).T)
    w1 = (dt > 0) & (dt <= 1) & (r <= 50); w2 = (dt > 1) & (dt <= 30) & (r <= 50)
    st.append(dict(date=str(e.time.date()), M=e.M, lon=e.lon, lat=e.lat, tstar_hours=round(tstar * 24, 2), n_in_tstar=int(((dt > 0) & (dt <= tstar) & (r <= 50)).sum()),
                   n_day1=int(w1.sum()), Mc_day1_maxc02=mc_maxc(A.M[w1]) if w1.sum() >= 20 else np.nan, n_day2_30=int(w2.sum()),
                   Mc_day2_30_maxc02=mc_maxc(A.M[w2]) if w2.sum() >= 20 else np.nan,
                   meanPbg_day1=round(float(np.nanmean(A.P_bg_hom30[w1])), 3) if w1.sum() else np.nan))
st = pd.DataFrame(st); st.to_csv(f'{ROOT}/out/tables/S_stai_audit.csv', index=False)
afad_all = pd.read_csv(f'{ROOT}/out/data/afad_all_region.csv', usecols=['time', 'lat', 'lon', 'M']); afad_all['time'] = pd.to_datetime(afad_all.time, utc=True, format='ISO8601')
pa = project(afad_all.lon, afad_all.lat); dd = []
for _, e in MS.iterrows():
    dt = (afad_all.time - e.time).dt.total_seconds() / 86400; r = np.hypot(*(pa - [e.x, e.y]).T)
    for lo, hi, nm_ in [(0, 0.25, '0-6h'), (0.25, 1, '6-24h'), (1, 30, '1-30d'), (30, 365, '30-365d')]:
        w = (dt > lo) & (dt <= hi) & (r <= 50)
        if w.sum() >= 30: dd.append(dict(date=str(e.time.date()), M=e.M, window=nm_, n_allM=int(w.sum()), Mc_maxc02=mc_maxc(afad_all.M[w])))
dd = pd.DataFrame(dd); dd.to_csv(f'{ROOT}/out/tables/S_stai_mc_windows.csv', index=False)
out['STAI'] = dict(n_mainshocks_M5_5=int(len(MS)), events_in_helmstetter_windows=int(st.n_in_tstar.sum()),
                   max_Mc_0_6h=float(dd[dd.window == '0-6h'].Mc_maxc02.max()) if len(dd) else None,
                   median_Mc_1_30d=float(dd[dd.window == '1-30d'].Mc_maxc02.median()) if len(dd) else None)
json.dump(out, open(f'{ROOT}/out/tables/T_validation.json', 'w'), indent=1, default=float)
print(Z.to_string()); print(ft.to_string()); print(D.to_string()); print(st.to_string()); print(dd.to_string()); print(json.dumps(out, indent=1, default=float))
