import json, glob, numpy as np, pandas as pd, sys
R = [json.load(open(f)) for f in sorted(glob.glob('out/*.json'))]
runs = []; Z = []
for r in R:
    runs.append(dict(scen=r['scen'], E=r['E'], rep=r['rep'], n_A=r.get('n_A'), n_B=r.get('n_B'), Mmax_A=r.get('Mmax_A'), br_fit=r.get('br_fit'),
                     alpha_eff=r.get('alpha_eff_fit'), beta=r.get('beta_fit'), exp_bg_fit=r.get('exp_bg_fit'), true_bg=r.get('true_bg_target'),
                     error=r.get('error'), n_raw=r.get('n_raw_DBKM_zones'), kde_med=np.median(r['kde_ratios_raw_DBKM']) if r.get('kde_ratios_raw_DBKM') else np.nan,
                     kde_max=np.max(r['kde_ratios_raw_DBKM']) if r.get('kde_ratios_raw_DBKM') else np.nan, runtime=r.get('runtime_s')))
    for z in r.get('zones', []): Z.append(dict(scen=r['scen'], E=r['E'], rep=r['rep'], **z))
runs = pd.DataFrame(runs); Z = pd.DataFrame(Z)
ok = runs[runs.error.isna()].copy()
ok['fit_ok'] = ok.br_fit.astype(float) > 0.1  # degenerate EM solution: branching ratio ~0, all events background
out = dict(n_files=len(runs), n_ok=int(len(ok)), n_err=int(runs.error.notna().sum()), errors=runs[runs.error.notna()][['scen','E','rep','error']].to_dict('records'),
           n_degenerate=int((~ok.fit_ok).sum()), degenerate=ok[~ok.fit_ok][['scen','E','rep','br_fit','alpha_eff','exp_bg_fit','true_bg']].to_dict('records'))
V = ['ens-DBKM', 'wDB10', 'wDB15', 'wDB20']
tab = []
for subset in ('all', 'fit_ok'):
  for (scen, E), g0 in ok.groupby(['scen', 'E']):
    g = g0 if subset == 'all' else g0[g0.fit_ok]
    if not len(g): continue
    zz = Z[(Z.scen == scen) & (Z.E == E) & Z.rep.isin(g.rep)]
    row = dict(subset=subset, scen=scen, E=E, n_cat=len(g), n_degenerate=int((~g0.fit_ok).sum()), n_A_med=float(g.n_A.median()), n_A_min=int(g.n_A.min()), n_A_max=int(g.n_A.max()),
               br_med=float(g.br_fit.median()), br_min=float(g.br_fit.min()), br_max=float(g.br_fit.max()),
               bg_fit_over_true_med=float((g.exp_bg_fit / g.true_bg).median()),
               kde_med_med=float(g.kde_med.median()), kde_med_q05=float(g.kde_med.quantile(0.05)), kde_med_q95=float(g.kde_med.quantile(0.95)),
               kde_max_med=float(g.kde_max.median()), kde_max_q95=float(g.kde_max.quantile(0.95)), kde_max_max=float(g.kde_max.max()),
               frac_kde_med_ge_1_9=float((g.kde_med >= 1.9).mean()), frac_kde_max_ge_3_94=float((g.kde_max >= 3.94).mean()))
    for v in V + ['oracle-wDB10', 'oracle-wDB15', 'oracle-wDB20']:
        zv = zz[zz.variant == v]
        if not v.startswith('oracle'):
            sup = zv[zv.supported == True]
            row[f'{v}_cats_with_any_supported'] = int(sup.rep.nunique()); row[f'{v}_supported_total'] = int(len(sup))
            row[f'{v}_supported_false'] = int((sup.src < 0).sum()) if len(sup) else 0
            if scen == 'SRC': row[f'{v}_src_zone'] = int(zv[zv.src >= 0].groupby('rep').src.nunique().sum())
            if scen == 'SRC': row[f'{v}_src_hits'] = int(sup[sup.src >= 0].groupby('rep').src.nunique().sum()); row[f'{v}_src_total'] = 3 * len(g)
            row[f'{v}_zones_total'] = int(len(zv))
        else:
            if scen == 'SRC': row[f'{v}_src_found'] = int(zv[zv.src >= 0].groupby('rep').src.nunique().sum()); row[f'{v}_src_total'] = 3 * len(g)
            row[f'{v}_zones_total'] = int(len(zv))
    # any-variant (family-wise) false positive
    fz = zz[(zz.variant.isin(V)) & (zz.supported == True) & (zz.src < 0)]
    row['any_variant_cats_with_false_supported'] = int(fz.rep.nunique())
    tab.append(row)
out['by_scenario'] = tab
json.dump(out, open('bench_summary.json', 'w'), indent=1, default=float)
runs.to_csv('bench_runs.csv', index=False); Z.to_csv('bench_zones.csv', index=False)
T = pd.DataFrame(tab)
print(out['n_files'], out['n_ok'], out['n_err']); print(T.T.to_string())
