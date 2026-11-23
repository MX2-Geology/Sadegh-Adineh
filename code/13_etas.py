"""wanat3 ETAS (Mizrahi et al. 2021 `etas` package, commit 097f08b) on catalogue A (AFAD). Usage:
   python 13_etas.py TAG MC FREE_BG   e.g. 'hom30 3.0 0' (spatially homogeneous mu) or 'ker30 3.0 1' (kernel-smoothed mu,
   Gaussian sigma = 10 km, i.e. bw_sq = 100 km^2, chosen a priori). Auxiliary 2016-01-01..2016-07-01, target 2016-07-01..2026-10-14.
   Branching ratio from etas.inversion.branching_ratio (unbounded magnitudes). K=20 Zhuang et al. (2002) stochastic realisations."""
import numpy as np, pandas as pd, json, logging, sys, time
from common import ROOT, SEED
from etas.inversion import ETASParameterCalculation, branching_ratio, parameter_dict2array
logging.basicConfig(level=logging.WARNING)
TAG, MC, FB = sys.argv[1], float(sys.argv[2]), bool(int(sys.argv[3]))
np.random.seed(SEED); t0 = time.time()
d = pd.read_csv(f'{ROOT}/out/data/cat_A_AFAD_2016_2026.csv'); d['time'] = pd.to_datetime(d.time, utc=True, format='ISO8601')
cat = pd.DataFrame({'time': d.time.dt.tz_localize(None), 'latitude': d.lat, 'longitude': d.lon, 'magnitude': np.round(d.M, 1)})
cat.index = pd.Index(np.arange(len(d)), name='id')
meta = dict(name=f'WA_ETAS_{TAG}', catalog=cat, auxiliary_start='2016-01-01', timewindow_start='2016-07-01',
            timewindow_end='2026-10-14', mc=MC, m_ref=MC, delta_m=0.1, coppersmith_multiplier=100,
            shape_coords=[[36, 26], [36, 31], [41, 31], [41, 26]], earth_radius=6.3781e3, bw_sq=100.0,
            free_background=FB, free_productivity=False, n_iterations=None)
calc = ETASParameterCalculation(meta); calc.prepare(); theta = calc.invert()
te = calc.target_events; pb = te['P_background'].reindex(cat.index)
K = 20; rng = np.random.default_rng(SEED)
real = (rng.random((K, len(d))) < np.nan_to_num(pb.values, nan=1.0)) & pb.notna().values
np.save(f'{ROOT}/out/data/etas_{TAG}_realisations.npy', real)
pd.DataFrame({'id': d.id, 'P_bg': pb.values}).to_csv(f'{ROOT}/out/data/etas_{TAG}_Pbg.csv', index=False)
par = {k: (None if v is None else float(v)) for k, v in dict(theta).items()}
try: br = float(branching_ratio(parameter_dict2array(dict(theta)), calc.beta))
except Exception as e: br = f'error: {str(e)[:100]}'
out = dict(tag=TAG, mc=MC, free_background=FB, bw_sq_km2=100.0 if FB else None, params=par, beta=float(calc.beta), b=float(calc.beta / np.log(10)),
           branching_ratio=br, one_minus_mean_Pbg=float(1 - np.nanmean(pb)), n_target=int(pb.notna().sum()), mean_P_bg=float(np.nanmean(pb)),
           expected_bg=float(np.nansum(pb)), K=K, seed=SEED, runtime_s=round(time.time() - t0))
json.dump(out, open(f'{ROOT}/out/tables/T_etas_{TAG}.json', 'w'), indent=1); print(json.dumps(out, indent=1))
