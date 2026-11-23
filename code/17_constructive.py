"""wanat3 Analysis 1: constructive zoning from the declustered catalogue.
 (a) P_bg-weighted DBSCAN: sample_weight = ETAS P_bg, weighted MinPts = 5 (i.e. >= 5 expected background events within eps),
     eps in {Scitovski rule on catalogue A, 10, 15, 20 km}; zones < 50 events pruned (as everywhere).
 (b) Realisation ensemble: HDBSCAN (min_cluster_size 25, min_samples 5) and DBSCAN (Scitovski eps recomputed per realisation, MinPts 5),
     zones < 25 events pruned, on each of the K = 20 ETAS stochastic-declustering realisations; per 10-km cell, zone-membership
     probability P_cell = fraction of realisations (in which the cell holds background events) where it is inside a zone;
     consensus zones = connected components of cells with P_cell >= 0.5 linked by co-assignment >= 0.5, kept if >= 25 expected
     background events (sum P_bg).
 Pre-set diagnostic chain and support rule (fixed before running): bootstrap stability (B = 200, 90 % subsample + 5 km jitter;
 cell-set Jaccard, stable if mean >= 0.75); independent-period agreement with the same method on catalogue B (ISC 2000-2015, ZBZ
 background; persistent if best cell-set Jaccard >= 0.5); fault proximity (fraction within 5 km vs rotation/translation null);
 orientation (axis vs fault strike); geodetic consistency (GNSS I2 inside vs outside, Mann-Whitney). A zone is 'supported' only if
 it is stable AND persistent."""
import numpy as np, pandas as pd, json, sys, time
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.stats import mannwhitneyu
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from common import *
from clustering import prune
from wa3 import *
from shapely.geometry import MultiPoint
rng = np.random.default_rng(SEED); t0 = time.time()
BN = int(sys.argv[1]) if len(sys.argv) > 1 else 200
A = load_A(); XY = A[['x', 'y']].values; n = len(A)
Bc = pd.read_csv(f'{ROOT}/out/data/cat_B_ISC_2000_2015.csv'); XYB = Bc[['x', 'y']].values
CELL = 10.0
def cells(P): return [tuple(c) for c in np.floor(P / CELL).astype(int)]
def cl(method, P, mcs=25):
    if len(P) < mcs: return np.full(len(P), -1)
    if method == 'HDB': return prune(HDBSCAN(min_cluster_size=mcs, min_samples=5, copy=True).fit_predict(P), mcs)
    e = scitovski_eps(P, 5)['eps']; return prune(DBSCAN(eps=e, min_samples=5).fit_predict(P), mcs)
def ensemble(R, P, Pbg, method):
    """R: K x n boolean background masks; returns dict(cellzone, Pcell, member)"""
    K = R.shape[0]; cid = cells(P); U = sorted(set(cid)); ix = {c: i for i, c in enumerate(U)}; ci = np.array([ix[c] for c in cid])
    obs = np.zeros((K, len(U)), bool); lab_c = np.full((K, len(U)), -1); member = np.zeros(n if len(P) == n else len(P)); bgc = np.zeros(len(P))
    for k in range(K):
        s = np.where(R[k])[0]; lb = cl(method, P[s]); obs[k, ci[s]] = True; bgc[s] += 1; member[s[lb >= 0]] += 1
        for z in np.unique(lb[lb >= 0]):
            cc, cnt = np.unique(ci[s[lb == z]], return_counts=True); lab_c[k, cc] = z   # a cell takes the zone of its clustered bg events
    nobs = obs.sum(0); Pcell = np.where(nobs > 0, (lab_c >= 0).sum(0) / np.maximum(nobs, 1), 0)
    cand = np.where((Pcell >= 0.5) & (nobs >= K / 4))[0]
    if len(cand) == 0: return dict(U=U, ci=ci, Pcell=Pcell, zone_of_cell={}, member=member / np.maximum(bgc, 1), nzones=0)
    L = lab_c[:, cand]; both = (L[:, :, None] >= 0) & (L[:, None, :] >= 0)
    co = ((L[:, :, None] == L[:, None, :]) & both).sum(0) / np.maximum(both.sum(0), 1)
    adj = (co >= 0.5)
    # only link spatially adjacent cells (8-neighbourhood) so components are contiguous
    cc = np.array([U[i] for i in cand]); nb = (np.abs(cc[:, None, :] - cc[None, :, :]).max(2) <= 1)
    ncomp, comp = connected_components(csr_matrix(adj & nb), directed=False)
    zone_of_cell = {}; zid = 0
    for c in range(ncomp):
        mem = cand[comp == c]; ev = np.isin(ci, mem)
        if np.nansum(Pbg[ev]) >= 25:
            for m in mem: zone_of_cell[U[m]] = zid
            zid += 1
    return dict(U=U, ci=ci, Pcell=Pcell, zone_of_cell=zone_of_cell, member=member / np.maximum(bgc, 1), nzones=zid)
def cellsets(zc):
    out = {}
    for c, z in zc.items(): out.setdefault(z, set()).add(c)
    return out
def best_j(S, others):
    return max([len(S & T) / len(S | T) for T in others] + [0.0])
def single_partition_cells(P, method, mcs=25):
    lb = cl(method, P, mcs); cid = cells(P); out = {}
    for z in np.unique(lb[lb >= 0]): out[z] = set(c for c, l in zip(cid, lb) if l == z)
    return out
Bsets = {m: list(single_partition_cells(XYB[Bc.bg_zbz.values], m).values()) for m in ('HDB', 'DBKM')}
BsetsGK = {m: list(single_partition_cells(XYB[Bc.bg_gk.values], m).values()) for m in ('HDB', 'DBKM')}
A_zbz = {m: single_partition_cells(XY[A.bg_zbz.values], m) for m in ('HDB', 'DBKM')}
dfa = fault_dist(XY)
grid = pd.read_csv(f'{ROOT}/out/data/gnss_strain_grid.csv'); GP = project(grid.lon, grid.lat)
I2ev = grid.I2.values[[np.argmin(np.hypot(*(GP - p).T)) for p in XY]]
def null_f5_sets(groups, nd=200, Rr=50.0):
    vals = []
    for _ in range(nd):
        P = []
        for Q in groups:
            c = Q.mean(0)
            for _t in range(50):
                th = rng.uniform(0, 2 * np.pi); rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                Qn = (Q - c) @ rot.T + c + rng.uniform(-Rr, Rr, 2)
                if inside(REGPOLY, Qn).mean() >= 0.8: break
            P.append(Qn)
        vals.append(np.mean(fault_dist(np.vstack(P)) <= 5))
    return np.array(vals)
REP = [(np.sort(rng.choice(n, int(0.9 * n), replace=False)), None) for _ in range(BN)]
REP = [(idx, rng.normal(0, 5.0, (len(idx), 2))) for idx, _ in REP]
summary = []; zrows = []; maps = {}; MSB = {}
for tag in ('hom30', 'ker30'):
    if f'P_bg_{tag}' not in A: continue
    R = real(tag); Pbg = np.nan_to_num(A[f'P_bg_{tag}'].values, nan=0.0)
    # ---------- (b) ensemble
    for method in ('HDB', 'DBKM'):
        ens = ensemble(R, XY, Pbg, method); sets = cellsets(ens['zone_of_cell']); maps[(tag, method)] = ens
        cid = cells(XY); zev = np.array([ens['zone_of_cell'].get(c, -1) for c in cid])
        # bootstrap
        def _one(idx, jit):
            P2 = XY[idx] + jit; e2 = ensemble(R[:, idx], P2, Pbg[idx], method); s2 = list(cellsets(e2['zone_of_cell']).values())
            z2 = np.array([e2['zone_of_cell'].get(c, -1) for c in cells(P2)]); zr = zev[idx]; o_c, o_e = {}, {}
            for z, S in sets.items():
                o_c[z] = best_j(S, s2); a = set(np.where(zr == z)[0])
                o_e[z] = max([len(a & set(np.where(z2 == w)[0])) / len(a | set(np.where(z2 == w)[0])) for w in np.unique(z2[z2 >= 0])] + [0.0])
            return o_c, o_e
        res = Parallel(n_jobs=6)(delayed(_one)(idx, jit) for idx, jit in REP)
        jb = {z: [r[0][z] for r in res] for z in sets}; jbe = {z: [r[1][z] for r in res] for z in sets}
        groups = [XY[zev == z] for z in sets]
        f5 = np.mean(dfa[zev >= 0] <= 5) if (zev >= 0).any() else np.nan
        nv = null_f5_sets(groups) if len(groups) else np.array([np.nan])
        for z, S in sets.items():
            ev = zev == z; Q = XY[ev]
            zrows.append(dict(variant=f'ensemble-{method}', etas=tag, zone=int(z), n_cells=len(S), n_events=int(ev.sum()), exp_bg=round(float(Pbg[ev].sum()), 1),
                              lon=round(A.lon[ev].mean(), 2), lat=round(A.lat[ev].mean(), 2), mean_Pcell=round(float(np.mean([ens['Pcell'][ens['U'].index(c)] for c in S])), 2),
                              J_boot=round(float(np.mean(jbe[z])), 3), J_boot_cells=round(float(np.mean(jb[z])), 3), J_period_B_ZBZ=round(best_j(S, Bsets[method]), 3), J_period_B_GK=round(best_j(S, BsetsGK[method]), 3),
                              J_A_ZBZ=round(best_j(S, list(A_zbz[method].values())), 3), frac_within5km=round(float(np.mean(dfa[ev] <= 5)), 3)))
        summary.append(dict(variant=f'ensemble-{method}', etas=tag, n_zones=len(sets), events_in_zones=int((zev >= 0).sum()),
                            f5_zones=round(f5, 3), f5_null_mean=round(float(np.nanmean(nv)), 3), p_fault=round((1 + np.sum(nv >= f5)) / (1 + len(nv)), 4) if len(groups) else np.nan,
                            I2_MWU_p=float(mannwhitneyu(I2ev[zev >= 0], I2ev[zev < 0], alternative='greater').pvalue) if (zev >= 0).sum() > 5 else np.nan, I2_med_in=round(float(np.median(I2ev[zev >= 0])), 1), I2_med_out=round(float(np.median(I2ev[zev < 0])), 1),
                            cells_Pcell_ge_0_5=int((ens['Pcell'] >= 0.5).sum()), cells_observed=int((ens['Pcell'] > 0).sum() + (ens['Pcell'] == 0).sum())))
    # ---------- (a) P_bg-weighted DBSCAN
    tg = ~np.isnan(A[f'P_bg_{tag}'].values)
    for eps in [scitovski_eps(XY, 5)['eps'], 10.0, 15.0, 20.0]:
        lab = np.full(n, -1); lab[tg] = prune(DBSCAN(eps=eps, min_samples=5).fit_predict(XY[tg], sample_weight=Pbg[tg]))
        Z0 = np.unique(lab[lab >= 0])
        def _w(idx, jit):
            t2 = tg[idx]; l2 = np.full(len(idx), -1)
            l2[t2] = prune(DBSCAN(eps=eps, min_samples=5).fit_predict((XY[idx] + jit)[t2], sample_weight=Pbg[idx][t2]))
            r = lab[idx]; o = {}
            for z in Z0:
                Aset = set(np.where(r == z)[0]); o[z] = max([len(Aset & set(np.where(l2 == w)[0])) / len(Aset | set(np.where(l2 == w)[0])) for w in np.unique(l2[l2 >= 0])] + [0.0])
            return o
        res = Parallel(n_jobs=6)(delayed(_w)(idx, jit) for idx, jit in REP)
        jb = {z: [r[z] for r in res] for z in Z0}
        # period B: same eps and MinPts on the ZBZ-background subset of catalogue B (unit weights); zones >= 25 events
        # density-matched: MinPts scaled by the ratio of background counts (B ZBZ background vs expected ETAS background in A)
        XB0 = XYB[Bc.bg_zbz.values]; msB = int(round(5 * len(XB0) / Pbg[tg].sum())); MSB[(tag, round(eps, 1))] = msB
        lB = prune(DBSCAN(eps=eps, min_samples=msB).fit_predict(XB0), 25)
        SB = [set(cells(XB0[lB == w])) for w in np.unique(lB[lB >= 0])]
        groups = [XY[lab == z] for z in jb]; f5 = np.mean(dfa[lab >= 0] <= 5) if (lab >= 0).any() else np.nan
        nv = null_f5_sets(groups) if len(groups) else np.array([np.nan])
        for z in jb:
            ev = lab == z; S = set(cells(XY[ev]))
            zrows.append(dict(variant=f'weighted-DBSCAN eps={eps:.1f}', etas=tag, zone=int(z), n_cells=len(S), n_events=int(ev.sum()), exp_bg=round(float(Pbg[ev].sum()), 1),
                              lon=round(A.lon[ev].mean(), 2), lat=round(A.lat[ev].mean(), 2), mean_Pcell=np.nan, J_boot=round(float(np.mean(jb[z])), 3),
                              J_period_B_ZBZ=round(best_j(S, SB), 3), J_period_B_GK=np.nan, J_A_ZBZ=np.nan, frac_within5km=round(float(np.mean(dfa[ev] <= 5)), 3), J_boot_cells=np.nan))
        summary.append(dict(variant=f'weighted-DBSCAN eps={eps:.1f}', etas=tag, n_zones=len(jb), events_in_zones=int((lab >= 0).sum()), f5_zones=round(f5, 3) if len(jb) else np.nan,
                            f5_null_mean=round(float(np.nanmean(nv)), 3), p_fault=round((1 + np.sum(nv >= f5)) / (1 + len(nv)), 4) if len(groups) else np.nan,
                            I2_MWU_p=float(mannwhitneyu(I2ev[lab >= 0], I2ev[lab < 0], alternative='greater').pvalue) if (lab >= 0).sum() > 5 else np.nan, I2_med_in=round(float(np.median(I2ev[lab >= 0])), 1) if (lab >= 0).any() else np.nan, I2_med_out=round(float(np.median(I2ev[lab < 0])), 1)))
        if tag == 'hom30' and abs(eps - 15.0) < 1e-6: A['zc_wdb15'] = lab
Zr = pd.DataFrame(zrows)
if len(Zr):
    Zr['bg_ge25'] = Zr.exp_bg >= 25; Zr['stable'] = Zr.J_boot >= 0.75; Zr['persistent'] = Zr.J_period_B_ZBZ >= 0.5; Zr['supported'] = Zr.stable & Zr.persistent
Zr.to_csv(f'{ROOT}/out/tables/T_constructive_zones.csv', index=False)
S = pd.DataFrame(summary)
if len(Zr):
    g = Zr.groupby(['variant', 'etas']); S = S.merge(g.agg(n_stable=('stable', 'sum'), n_persistent=('persistent', 'sum'), n_supported=('supported', 'sum'), n_bg_ge25=('bg_ge25', 'sum'),
                                                        Jboot_med=('J_boot', 'median'), Jper_med=('J_period_B_ZBZ', 'median')).reset_index(), on=['variant', 'etas'], how='left')
S.to_csv(f'{ROOT}/out/tables/T_constructive_summary.csv', index=False)
# ---------- quantification: required local enhancement of background density
area = REGPOLY.area; T = (pd.to_datetime(A.time.max(), utc=True) - pd.Timestamp('2016-07-01', tz='UTC')).days / 365.25
lam = np.nansum(A.P_bg_hom30) / area          # background events per km^2 over the target period
e_rule = scitovski_eps(XY, 5)['eps']
q = dict(T_years=round(T, 2), region_area_km2=round(area), expected_bg=round(float(np.nansum(A.P_bg_hom30)), 1),
         bg_density_per_1e4km2_per_yr=round(lam / T * 1e4, 2), eps_rule_km=round(e_rule, 2))
for eps in (e_rule, 10.0, 15.0, 20.0):
    mu = lam * np.pi * eps ** 2; q[f'expected_bg_within_{eps:.1f}km'] = round(mu, 3); q[f'enhancement_needed_MinPts5_{eps:.1f}km'] = round(5 / mu, 1)
kd = pd.read_csv(f'{ROOT}/out/tables/T_kde_persistence.csv'); kh = kd[kd.weights == 'ETAS_hom30']
q['kde_ratio_raw_zones_max'] = float(kh.ratio.max()); q['kde_ratio_raw_zones_median'] = float(kh.ratio.median())
for nm, r in (('median', kh.ratio.median()), ('max', kh.ratio.max())):
    for eps in (e_rule, 10.0):
        q[f'years_to_MinPts5_at_{nm}_KDE_ratio_{eps:.1f}km'] = round(T * 5 / (r * lam * np.pi * eps ** 2), 1)
# number of 'core' background points under the fitted homogeneous rate: Poisson P(N>=5) at mean mu*ratio
from scipy.stats import poisson
q['P_core_rule_eps_at_median_ratio'] = round(float(poisson.sf(4, kh.ratio.median() * lam * np.pi * e_rule ** 2)), 4)
json.dump(dict(minpts_B_density_matched={f'{k[0]}_{k[1]}': v for k, v in MSB.items()}, quantification=q, runtime_s=round(time.time() - t0), B=BN), open(f'{ROOT}/out/tables/T_constructive_quant.json', 'w'), indent=1, default=float)
# membership map export (primary)
for (tag, method), ens in maps.items():
    pd.DataFrame([dict(cx=c[0], cy=c[1], Pcell=p, zone=ens['zone_of_cell'].get(c, -1)) for c, p in zip(ens['U'], ens['Pcell'])]).to_csv(
        f'{ROOT}/out/data/membership_{tag}_{method}.csv', index=False)
A.to_csv(f'{ROOT}/out/data/cat_A_labels_constructive.csv', index=False)
print(S.to_string()); print(Zr.to_string()); print(json.dumps(q, indent=1)); print('runtime', time.time() - t0)
