"""Scitovski (2018) Croatia reproduction with exact DBSCAN (wanat2) and with leaders-based Rough-DBSCAN (wanat3)."""
import numpy as np, pandas as pd, json
from sklearn.cluster import DBSCAN
from common import scitovski_eps, ROOT, SEED
from rough_dbscan import rough_dbscan
d = pd.read_csv(f'{ROOT}/data/raw/usgs_croatia_1950_2017-05-25.csv'); d = d[d.type == 'earthquake'].sort_values('time').reset_index(drop=True)
L = dict(l0=12.5, l1=21, p0=41.5, p1=47.5)
B = np.c_[(d.longitude - L['l0']) / (L['l1'] - L['l0']), (d.latitude - L['p0']) / (L['p1'] - L['p0'])]
paper = dict(n_events=5324, eps=0.016, n_clusters=15, n_noise=1006, sizes=[1081, 1081, 999, 541, 156, 87, 80, 69, 57, 41, 40, 31, 24, 19, 12])
rng = np.random.default_rng(SEED); rows = []; S12 = []
def summ(lab):
    s = sorted(np.bincount(lab[lab >= 0]).tolist(), reverse=True) if (lab >= 0).any() else []
    S12.append((sum(x >= 12 for x in s), int((lab < 0).sum()) + sum(x for x in s if x < 12)))
    return len(s), int((lab < 0).sum()), s[:15]
for conv, k in [('self_included', 4), ('self_excluded', 5)]:
    e = scitovski_eps(B, minpts=k)['eps']
    for eps_name, eps in [('scitovski_rule', e), ('paper_0.016', 0.016)]:
        nc, nn, s = summ(DBSCAN(eps=eps, min_samples=k).fit_predict(B))
        rows.append(dict(algorithm='exact DBSCAN', convention=conv, eps=round(eps, 5), tau=None, order='-', n_leaders=None, n_clusters=nc, n_noise=nn, sizes=s))
        for tf in (0.25, 0.5):
            for oname in ('time', 'random'):
                order = None if oname == 'time' else rng.permutation(len(B))
                lab, nl = rough_dbscan(B, eps, k, tf * eps, order)
                nc, nn, s = summ(lab)
                rows.append(dict(algorithm='Rough-DBSCAN', convention=conv, eps=round(eps, 5), tau=f'{tf}*eps', order=oname, n_leaders=nl, n_clusters=nc, n_noise=nn, sizes=s))
R = pd.DataFrame(rows); R['n_clusters_ge12'] = [a for a, b in S12]; R['n_noise_if_lt12_noise'] = [b for a, b in S12]; R['abs_dclusters'] = (R.n_clusters - 15).abs(); R['abs_dnoise'] = (R.n_noise - 1006).abs()
R.to_csv(f'{ROOT}/out/tables/T_croatia_rough.csv', index=False)
best = R.sort_values(['abs_dclusters', 'abs_dnoise']).iloc[0].to_dict()
json.dump(dict(n_events=len(d), paper=paper, closest_run=best), open(f'{ROOT}/out/tables/T_croatia_rough.json', 'w'), indent=1, default=str)
print(R.drop(columns=['sizes']).to_string()); print(best)
