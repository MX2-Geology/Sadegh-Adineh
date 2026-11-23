"""Seed and completeness sensitivity of the constructive zones, the Denizli box and the depth statistics (Sections 2.1, 3.4, 4).
Inputs: out/tables (package seed, Mc 3.0), audit/runs/seed1-3 (17_constructive.py with SEED = 1, 2, 3),
audit/runs/Mc25 and audit/runs/Mc35 (12_prepare, 13_etas hom30 and 17_constructive at Mc 2.5 and 3.5; see audit/README.md),
out/data/cat_A_labels_constructive.csv. Output: out/tables/T_sensitivity.json and T_sensitivity_seed.csv"""
import json, pandas as pd
from common import ROOT
A_ = f'{ROOT}/audit/runs'
def zones(path, run):
    z = pd.read_csv(path); z['run'] = run; return z
Z = pd.concat([zones(f'{ROOT}/out/tables/T_constructive_zones.csv', 'seed20261028')] +
              [zones(f'{A_}/seed{s}/tables/T_constructive_zones_UNROUNDED.csv', f'seed{s}') for s in (1, 2, 3)])
den = Z[Z.lon.between(29.2, 29.4) & Z.lat.between(37.85, 37.95)]
ens = den[den.variant.str.startswith('ensemble')]; w15 = den[den.variant == 'weighted-DBSCAN eps=15.0']
sim = Z[(Z.variant == 'weighted-DBSCAN eps=15.0') & Z.lon.between(28.0, 28.15) & Z.lat.between(39.1, 39.2)]
sea = Z[(Z.variant == 'weighted-DBSCAN eps=20.0') & Z.lon.between(27.6, 27.8) & Z.lat.between(36.6, 36.75)]
out = dict(seeds=['20261028', '1', '2', '3'],
    denizli_ensemble=dict(n_combinations=int(len(ens)), n_supported=int(ens.supported.sum()), J_boot_min=round(ens.J_boot.min(), 3), J_boot_max=round(ens.J_boot.max(), 3)),
    denizli_wdb15=dict(n=int(len(w15)), n_supported=int(w15.supported.sum()), J_boot_min=round(w15.J_boot.min(), 3), J_boot_max=round(w15.J_boot.max(), 3), J_B=sorted(set(w15.J_period_B_ZBZ.round(3)))),
    simav_wdb15_ker30=dict(n=int(len(sim)), n_supported=int(sim.supported.sum()), J_boot_min=round(sim.J_boot.min(), 3), J_boot_max=round(sim.J_boot.max(), 3), J_B=sorted(set(sim.J_period_B_ZBZ.round(3))), n_events=sorted(set(sim.n_events))),
    se_aegean_wdb20=dict(n=int(len(sea)), n_supported=int(sea.supported.sum()), J_boot_min=round(sea.J_boot.min(), 3), J_boot_max=round(sea.J_boot.max(), 3), J_B=sorted(set(sea.J_period_B_ZBZ.round(3))), n_events=sorted(set(sea.n_events)), n_cells=sorted(set(sea.n_cells))))
pd.concat([ens, w15, sim, sea])[['run', 'variant', 'etas', 'lon', 'lat', 'n_events', 'exp_bg', 'J_boot', 'J_period_B_ZBZ', 'supported']].round(3).to_csv(f'{ROOT}/out/tables/T_sensitivity_seed.csv', index=False)
for m in ('Mc25', 'Mc35'):
    c = json.load(open(f'{A_}/{m}/tables/T_catalogues.json')); e = json.load(open(f'{A_}/{m}/tables/T_etas_hom30.json'))
    q = json.load(open(f'{A_}/{m}/tables/T_constructive_quant.json'))['quantification']; z = pd.read_csv(f'{A_}/{m}/tables/T_constructive_zones.csv')
    s = z[z.supported]
    out[m] = dict(n_A=c['A_AFAD_2016_2026']['n'], n_B=c['B_ISC_2000_2015']['n'], b=round(e['b'], 2), branching_ratio=round(e['branching_ratio'], 3), expected_bg=round(e['expected_bg']),
                  eps_rule_km=q['eps_rule_km'], bg_density=q['bg_density_per_1e4km2_per_yr'],
                  supported=s[['variant', 'etas', 'lon', 'lat', 'n_events', 'exp_bg', 'J_boot', 'J_period_B_ZBZ']].to_dict('records'))
A = pd.read_csv(f'{ROOT}/out/data/cat_A_labels_constructive.csv'); A['t'] = pd.to_datetime(A.time, utc=True, format='mixed')
b = A[A.lon.between(28.9, 29.9) & A.lat.between(37.65, 38.15) & (A.M >= 3.0)]; yr = b.t.dt.year
pb = b.groupby(yr).P_bg_hom30.sum()
out['denizli_box'] = dict(box='28.9-29.9E, 37.65-38.15N', n=int(len(b)), n_2019=int((yr == 2019).sum()), n_2026=int((yr == 2026).sum()),
                          sum_Pbg_hom30=round(float(b.P_bg_hom30.sum()), 1), sum_Pbg_hom30_2019_2026=round(float(pb.get(2019, 0) + pb.get(2026, 0)), 1),
                          share_2019_2026=round(float((pb.get(2019, 0) + pb.get(2026, 0)) / b.P_bg_hom30.sum()), 3))
d = A.depth.astype(float)
out['depth'] = dict(n=int(len(A)), frac_exact_7=round(float((d == 7.0).mean()), 3), frac_round01_7=round(float((d.round(1) == 7.0).mean()), 3), frac_gt40=round(float((d > 40).mean()), 3))
json.dump(out, open(f'{ROOT}/out/tables/T_sensitivity.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
