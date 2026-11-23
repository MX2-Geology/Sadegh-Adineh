"""Detectability of stationary sources (Section 3.4), with the same inputs and arithmetic as the quantification block of
17_constructive.py. scikit-learn counts the point itself in min_samples, so with MinPts = 5 a core point needs 4 further
events within eps (the convention of the clustering, 'sklearn_4_further'). The stricter count of 5 further events is the
one reported by 17_constructive.py ('strict_5_further'). Output: out/tables/T_detectability.json"""
import json, numpy as np, pandas as pd
from scipy.stats import poisson
from common import *
from wa3 import *
A = load_A(); XY = A[['x', 'y']].values
area = REGPOLY.area; T = (pd.to_datetime(A.time.max(), utc=True) - pd.Timestamp('2016-07-01', tz='UTC')).days / 365.25
lam = np.nansum(A.P_bg_hom30) / area; e_rule = scitovski_eps(XY, 5)['eps']
kd = pd.read_csv(f'{ROOT}/out/tables/T_kde_persistence.csv'); kh = kd[kd.weights == 'ETAS_hom30']; med, mx = kh.ratio.median(), kh.ratio.max()
out = dict(T_years=round(T, 2), eps_rule_km=round(e_rule, 2), kde_median=round(float(med), 2), kde_max=round(float(mx), 2), radii={})
for eps in (e_rule, 10.0, 15.0, 20.0):
    mu = lam * np.pi * eps ** 2; d = dict(expected_bg_within=round(mu, 3))
    for k, lab in [(4, 'sklearn_4_further'), (5, 'strict_5_further')]:
        d[lab] = dict(enhancement_needed=round(k / mu, 2), years_at_median=round(T * k / (med * mu), 1), years_at_max=round(T * k / (mx * mu), 1))
    out['radii'][f'{eps:.1f}km'] = d
mu = lam * np.pi * e_rule ** 2 * med
out['P_core_rule_eps_median'] = dict(sklearn_4_further=round(float(poisson.sf(3, mu)), 5), strict_5_further=round(float(poisson.sf(4, mu)), 5))
json.dump(out, open(f'{ROOT}/out/tables/T_detectability.json', 'w'), indent=1); print(json.dumps(out, indent=1))
