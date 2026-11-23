"""wanat3 Stage C2: analysis catalogues at the a-priori Mc (T_Mc_choice.json): A = AFAD 2016-01-01..2026-10-14 (AFAD preferred
magnitude, ML/Mw; justified in T_catalogue_audit.json), B = ISC 2000-2015 (independent period/agency). ZBZ and GK74 declustering."""
import numpy as np, pandas as pd, json
from common import *
from decluster import nn_zbz, gk74
MC = json.load(open(f'{ROOT}/out/tables/T_Mc_choice.json'))['Mc_analysis']
A = pd.read_csv(f'{ROOT}/out/data/afad_all_region.csv', low_memory=False); B = pd.read_csv(f'{ROOT}/out/data/isc_B_all_region.csv', low_memory=False)
out = {}
for k, d in [('A_AFAD_2016_2026', A), ('B_ISC_2000_2015', B)]:
    d = d.copy(); d['time'] = pd.to_datetime(d.time, utc=True, format='ISO8601'); d['M'] = np.round(d.M.astype(float), 1)
    d = d[d.M >= MC - 1e-6].sort_values('time').reset_index(drop=True)
    xy = project(d.lon, d.lat); d['x'], d['y'] = xy[:, 0], xy[:, 1]
    t_yr = (d.time - d.time.iloc[0]).dt.total_seconds().values / 3.15576e7
    z = nn_zbz(t_yr, xy, d.M.values, seed=SEED)
    d['log_eta'] = np.log10(z['eta']); d['logT'] = z['logT']; d['logR'] = z['logR']; d['bg_zbz'] = z['background']; d['zbz_parent'] = z['parent']
    d['bg_gk'] = gk74(t_yr * 365.25, xy, d.M.values)
    vc = d.depth.round(1).value_counts(normalize=True)
    fixed = [v for v in vc.index[:5] if vc[v] > 0.03 and v in (0.0, 5.0, 7.0, 10.0, 33.0)]
    d['depth_fixed'] = d.depth.round(1).isin(fixed)
    d.to_csv(f'{ROOT}/out/data/cat_{k}.csv', index=False)
    out[k] = dict(n=len(d), t0=str(d.time.min()), t1=str(d.time.max()), Mmax=float(d.M.max()), Mc=MC, zbz_thr=round(z['thr'], 3),
                  frac_bg_zbz=round(float(d.bg_zbz.mean()), 3), frac_bg_gk=round(float(d.bg_gk.mean()), 3),
                  magtypes=d.magType.astype(str).str.upper().value_counts(normalize=True).round(3).to_dict(),
                  fixed_depth_values=fixed, frac_fixed_depth=round(float(d.depth_fixed.mean()), 3), mc_maxc_of_subset=mc_maxc(d.M),
                  top_depths=vc.head(4).round(3).to_dict())
json.dump(out, open(f'{ROOT}/out/tables/T_catalogues.json', 'w'), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
