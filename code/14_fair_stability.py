"""wanat3 RQ1 fair stability comparison on catalogue A (AFAD, Mc 3.0). Predefined grid (fixed before running):
 SCIT/DBKM: eps = Scitovski rule x {0.75, 1, 1.25}, MinPts {5, 9};  HDB: min_cluster_size {25, 50, 100} x min_samples {5, 10, 20};
 OPT: xi {0.02, 0.05, 0.10} x min_samples {5, 10, 20} (min_cluster_size 50). All partitions pruned identically (zones < 50 events -> noise).
 B = 200 bootstrap replicates (90 % subsample without replacement + 5 km Gaussian epicentral jitter), identical replicates for all configs.
 Metrics per replicate: ARI_all (noise = its own label -1, i.e. noise-noise pairs count as agreement), ARI_clustered (events clustered in
 both reference and replicate), noise agreement, per-zone Hennig (2007) Jaccard."""
import numpy as np, pandas as pd, json, time, sys
from joblib import Parallel, delayed
from common import *
from clustering import run, ari, jaccard_stability
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from clustering import prune, REGN
BN = int(sys.argv[1]) if len(sys.argv) > 1 else 200
A = pd.read_csv(f'{ROOT}/out/data/cat_A_AFAD_2016_2026.csv'); XY = A[['x', 'y']].values; LL = A[['lon', 'lat']].values
def unit(LL): return np.c_[(LL[:, 0] - REGN['lon0']) / 5.0, (LL[:, 1] - REGN['lat0']) / 5.0]
CFG = []
for m in ('SCIT', 'DBKM'):
    for f in (0.75, 1.0, 1.25):
        for k in (5, 9): CFG.append(dict(method=m, eps_scale=f, minpts=k, default=(f == 1.0 and k == 5)))
for c in (25, 50, 100):
    for k in (5, 10, 20): CFG.append(dict(method='HDB', mcs=c, minpts=k, default=(c == 50 and k == 5)))
for x in (0.02, 0.05, 0.10):
    for k in (5, 10, 20): CFG.append(dict(method='OPT', xi=x, minpts=k, default=(x == 0.05 and k == 5)))
def fit(cfg, XY_, LL_, eps_fixed=None):
    m = cfg['method']; k = cfg['minpts']
    if m == 'SCIT': return prune(DBSCAN(eps=eps_fixed * cfg['eps_scale'], min_samples=k).fit_predict(unit(LL_)))
    if m == 'DBKM': return prune(DBSCAN(eps=eps_fixed * cfg['eps_scale'], min_samples=k).fit_predict(XY_))
    if m == 'HDB': return prune(HDBSCAN(min_cluster_size=cfg['mcs'], min_samples=k, copy=True).fit_predict(XY_))
    if m == 'OPT': return prune(OPTICS(min_samples=k, xi=cfg['xi'], min_cluster_size=50).fit_predict(XY_))
# eps is fixed from the reference catalogue (rule applied once, as in Scitovski 2018) and kept for replicates
EPS = {'SCIT': {k: scitovski_eps(unit(LL), k)['eps'] for k in (5, 9)}, 'DBKM': {k: scitovski_eps(XY, k)['eps'] for k in (5, 9)}}
rng = np.random.default_rng(SEED); REPS = []
for b in range(BN):
    idx = np.sort(rng.choice(len(A), int(0.9 * len(A)), replace=False)); jit = rng.normal(0, 5.0, (len(idx), 2)); REPS.append((idx, jit))
def one(ci):
    cfg = CFG[ci]; e = EPS.get(cfg['method'], {}).get(cfg['minpts'])
    ref = fit(cfg, XY, LL, e); out = []; jz = []
    for b, (idx, jit) in enumerate(REPS):
        XYb = XY[idx] + jit; kmdeg = np.c_[1 / (111.32 * np.cos(np.radians(LL[idx, 1]))), np.full(len(idx), 1 / 111.32)]
        LLb = LL[idx] + jit * kmdeg; lb = fit(cfg, XYb, LLb, e); r = ref[idx]
        both = (r >= 0) & (lb >= 0)
        out.append(dict(cfg=ci, rep=b, ARI_all=ari(r, lb), ARI_clustered=ari(r[both], lb[both]) if both.sum() > 10 else np.nan,
                        frac_both_clustered=both.mean(), noise_agree=np.mean((r < 0) == (lb < 0)), n_zones_rep=int(lb.max() + 1)))
        jz.append(jaccard_stability(ref, lb, idx))
    zones = []
    for z in np.unique(ref[ref >= 0]):
        v = np.array([j.get(z, 0.0) for j in jz])
        zones.append(dict(cfg=ci, zone=int(z), n=int((ref == z).sum()), J_mean=v.mean(), J_median=np.median(v), J_p10=np.quantile(v, 0.1),
                          lon=LL[ref == z, 0].mean(), lat=LL[ref == z, 1].mean()))
    return out, zones, ref
t0 = time.time()
res = Parallel(n_jobs=6)(delayed(one)(i) for i in range(len(CFG)))
R = pd.DataFrame([r for o, z, _ in res for r in o]); Z = pd.DataFrame([r for o, z, _ in res for r in z])
C = pd.DataFrame(CFG); C['cfg'] = range(len(CFG))
C['eps_used'] = [EPS[c['method']][c['minpts']] * c['eps_scale'] if c['method'] in EPS else np.nan for c in CFG]
C['n_zones'] = [int(r.max() + 1) for _, _, r in res]; C['noise_frac'] = [float((r < 0).mean()) for _, _, r in res]
g = R.groupby('cfg'); C['ARI_all_med'] = g.ARI_all.median().values; C['ARI_all_p05'] = g.ARI_all.quantile(0.05).values
C['ARI_clu_med'] = g.ARI_clustered.median().values; C['ARI_clu_p05'] = g.ARI_clustered.quantile(0.05).values
C['noise_agree_med'] = g.noise_agree.median().values
zg = Z.groupby('cfg'); C['zones_J>=0.75'] = zg.apply(lambda s: int((s.J_mean >= 0.75).sum())).values
C['frac_events_in_J>=0.75_zones'] = zg.apply(lambda s: s[s.J_mean >= 0.75].n.sum() / len(A)).values
C['J_zone_median'] = zg.J_mean.median().values
C.to_csv(f'{ROOT}/out/tables/T_fair_stability_configs.csv', index=False); R.to_csv(f'{ROOT}/out/tables/S_fair_stability_replicates.csv', index=False)
Z.to_csv(f'{ROOT}/out/tables/S_fair_stability_zone_jaccard.csv', index=False)
S = []
for m, s in C.groupby('method', sort=False):
    d = s[s.default].iloc[0]
    S.append(dict(method=m, n_configs=len(s), default_ARI_all=d.ARI_all_med, default_ARI_clu=d.ARI_clu_med, default_zones=f"{d['zones_J>=0.75']}/{d.n_zones}",
                  best_ARI_all=s.ARI_all_med.max(), median_ARI_all=s.ARI_all_med.median(), best_ARI_clu=s.ARI_clu_med.max(), median_ARI_clu=s.ARI_clu_med.median(),
                  best_cfg_ARI_clu=int(s.loc[s.ARI_clu_med.idxmax(), 'cfg']), best_frac_events_stable=s['frac_events_in_J>=0.75_zones'].max(),
                  median_frac_events_stable=s['frac_events_in_J>=0.75_zones'].median()))
S = pd.DataFrame(S).round(3); S.to_csv(f'{ROOT}/out/tables/T_fair_stability_summary.csv', index=False)
# default reference labels for later scripts
for (o, z, ref), c in zip(res, CFG):
    if c['default']: A[f"z_{c['method']}"] = ref
A.to_csv(f'{ROOT}/out/data/cat_A_labels.csv', index=False)
json.dump(dict(B=BN, seed=SEED, eps=EPS, runtime_s=round(time.time() - t0)), open(f'{ROOT}/out/tables/T_fair_stability_meta.json', 'w'), indent=1)
print(C.round(3).to_string()); print(S.to_string()); print('runtime', time.time() - t0)
