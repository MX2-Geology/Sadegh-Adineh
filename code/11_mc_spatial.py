"""Spatial Mc (MAXC+0.2 per 0.5 deg cell, >=100 events M>=1) for AFAD period A; chooses analysis Mc a priori as
max(3.0, 95th percentile of cell Mc, 95th percentile of moving-window MBS), rounded up to 0.1."""
import numpy as np, pandas as pd, json
from common import *
A = pd.read_csv(f'{ROOT}/out/data/afad_all_region.csv')
A['cx'] = np.floor(A.lon / 0.5) * 0.5; A['cy'] = np.floor(A.lat / 0.5) * 0.5
rows = [dict(lon=cx, lat=cy, n=len(g), Mc=mc_maxc(g.M)) for (cx, cy), g in A[A.M >= 1].groupby(['cx', 'cy']) if len(g) >= 100]
S = pd.DataFrame(rows); S.to_csv(f'{ROOT}/out/tables/S_Mc_spatial_AFAD.csv', index=False)
J = json.load(open(f'{ROOT}/out/tables/T_catalogue_audit.json'))
p95 = float(S.Mc.quantile(0.95)); mbs95 = J['Mc_AFAD']['moving_window_MBS_p95']
MC = float(np.ceil(max(3.0, p95, mbs95) * 10 - 1e-9) / 10)
out = dict(n_cells=len(S), cell_Mc_median=float(S.Mc.median()), cell_Mc_p95=p95, cell_Mc_max=float(S.Mc.max()), moving_MBS_p95=mbs95, Mc_analysis=MC,
           n_events_ge_Mc=int((A.M >= MC - 1e-6).sum()), n_ge_3_0=int((A.M >= 3.0 - 1e-6).sum()), n_ge_3_2=int((A.M >= 3.2 - 1e-6).sum()))
json.dump(out, open(f'{ROOT}/out/tables/T_Mc_choice.json', 'w'), indent=1); print(S.sort_values('Mc').tail(8).to_string()); print(out)
