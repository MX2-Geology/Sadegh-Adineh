"""Numbers-consistency check: every key number quoted in doc/manuscript.md (and highlights/supplement) is recomputed from
out/tables and must appear verbatim in the text. Exit code 1 on any failure. Log: logs/30_check_numbers.log"""
import json, re, sys, pandas as pd, numpy as np
from common import ROOT
T = f'{ROOT}/out/tables'; J = lambda f: json.load(open(f'{T}/{f}'))
ms = open(f'{ROOT}/doc/manuscript.md').read(); hl = open(f'{ROOT}/doc/highlights.md').read(); sp = open(f'{ROOT}/doc/supplement.md').read()
cat = J('T_catalogues.json'); mc = J('T_Mc_choice.json'); au = J('T_catalogue_audit.json'); e0 = J('T_etas_hom30.json'); ek = J('T_etas_ker30.json'); e2 = J('T_etas_hom32.json')
va = J('T_validation.json'); cq = J('T_constructive_quant.json')['quantification']; cr = J('T_croatia_rough.json'); fm = J('T_fair_stability_meta.json')
FS = pd.read_csv(f'{T}/T_fair_stability_summary.csv').set_index('method'); CR = pd.read_csv(f'{T}/T_croatia_rough.csv'); FT = pd.read_csv(f'{T}/T_fault_test.csv')
Z = pd.read_csv(f'{T}/T_zones_all_methods.csv'); K = pd.read_csv(f'{T}/T_kde_persistence.csv'); CZ = pd.read_csv(f'{T}/T_constructive_zones.csv'); CS = pd.read_csv(f'{T}/T_constructive_summary.csv')
f = lambda x, d=3: f'{x:.{d}f}'; c = lambda x: f'{int(x):,}'
zd = Z[Z.method == 'DBKM']; st = zd[zd.stable]; kh = K[K.weights == 'ETAS_hom30']
ex = CR[CR.algorithm.str.startswith('exact')]; ro = CR[~CR.algorithm.str.startswith('exact')]
den = CZ[(CZ.lon.between(29.2, 29.35)) & (CZ.lat.between(37.85, 37.95)) & CZ.variant.str.startswith('ensemble')]
checks = [
 ('n catalogue A', c(cat['A_AFAD_2016_2026']['n'])), ('n catalogue B', c(cat['B_ISC_2000_2015']['n'])), ('Mc', f(mc['Mc_analysis'], 1)),
 ('AFAD merged rows', c(au['afad_rows_merged_in_region'])), ('Mw xmatch n', str(au['crossmatch_vs_independent_Mw']['Mw']['n'])), ('ML xmatch n', c(au['crossmatch_vs_independent_Mw']['ML']['n'])),
 ('KK2016 resid', f"−{abs(au['crossmatch_vs_independent_Mw']['ML']['KK2016_resid_median']):.2f}"), ('MBS p95', f"{au['Mc_AFAD']['moving_window_MBS_p95']:.2f}"),
 ('fixed depth %', f"{round(100*cat['A_AFAD_2016_2026']['frac_fixed_depth'])}%"), ('MD % B', f"MD {round(100*cat['B_ISC_2000_2015']['magtypes']['MD'])}%"),
 ('B MBS all', f"{au['Mc_ISC_B']['all_MBS']:.1f}"), ('STAI n', str(va['STAI']['n_mainshocks_M5_5'])), ('STAI Mc 0-6h', f"{va['STAI']['max_Mc_0_6h']:.1f}"), ('STAI Mc 1-30d', f"{va['STAI']['median_Mc_1_30d']:.1f}"),
 ('croatia n', c(cr['n_events'])), ('croatia exact range', f"{ex.n_clusters.min()}–{ex.n_clusters.max()}"), ('croatia rough range', f"{ro.n_clusters.min()}–{ro.n_clusters.max()}"),
 ('croatia >=12 range', f"{CR.n_clusters_ge12.min()}–{CR.n_clusters_ge12.max()}"), ('croatia noise lt12', f"{c(CR.n_noise_if_lt12_noise.min())}–{c(CR.n_noise_if_lt12_noise.max())}"),
 ('closest noise', str(cr['closest_run']['n_noise'])), ('eps SCIT', f"{fm['eps']['SCIT']['5']:.5f}"), ('eps km', f"{fm['eps']['DBKM']['5']:.2f} km"),
 ('branching hom30', f(e0['branching_ratio'])), ('branching ker30', f(ek['branching_ratio'])), ('branching hom32', f(e2['branching_ratio'])), ('mean Pbg', f(e0['mean_P_bg'])),
 ('exp bg', c(round(e0['expected_bg']))), ('exp bg ker', c(round(ek['expected_bg']))), ('exp bg hom32', str(round(e2['expected_bg']))), ('n target', c(e0['n_target'])), ('n target 32', c(e2['n_target'])),
 ('b hom30', f"b = {e0['b']:.2f}"), ('ZBZ A', f(cat['A_AFAD_2016_2026']['frac_bg_zbz'])), ('GK A', f(cat['A_AFAD_2016_2026']['frac_bg_gk'])),
 ('trig ETAS', f"{st.Ptr_ETAS_hom.min():.2f} to {st.Ptr_ETAS_hom.max():.2f}"), ('trig ZBZ', f"{st.Ptr_ZBZ.min():.2f} to {st.Ptr_ZBZ.max():.2f}"), ('trig GK', f"{st.Ptr_GK.min():.2f} to {st.Ptr_GK.max():.2f}"),
 ('abstract trig ETAS', f"{st.Ptr_ETAS_hom.min():.2f}–{st.Ptr_ETAS_hom.max():.2f} (ETAS)"),
 ('DBKM max persist', f"{zd.persist_hom.max():.2f}"), ('KDE median', f"{kh.ratio.median():.2f}"), ('KDE ker median', f"{K[K.weights=='ETAS_ker30'].ratio.median():.2f}"),
 ('KDE lo>1', f"{(kh.ci_lo > 1).sum()} of {len(kh)}"), ('KDE raw', f"{K[K.weights=='raw'].ratio.median():.2f}"),
 ('bg density', f"{cq['bg_density_per_1e4km2_per_yr']:.2f}"), ('exp within eps', f"{cq['expected_bg_within_6.2km']:.2f}"), ('enh 6.2', f"{cq['enhancement_needed_MinPts5_6.2km']}×"),
 ('enh 10', f"{cq['enhancement_needed_MinPts5_10.0km']}×"), ('years med 6.2', f"{cq['years_to_MinPts5_at_median_KDE_ratio_6.2km']} and {cq['years_to_MinPts5_at_median_KDE_ratio_10.0km']} yr"), ('years med 10', f"{cq['years_to_MinPts5_at_median_KDE_ratio_10.0km']} yr"),
 ('years max', f"{cq['years_to_MinPts5_at_max_KDE_ratio_6.2km']} and {cq['years_to_MinPts5_at_max_KDE_ratio_10.0km']} yr"), ('area', c(cq['region_area_km2'])), ('T', f"{cq['T_years']}"),
 ('fault hom30', f"0.343 against a null of 0.288"), ('fault all', f"{FT[FT.test=='all catalogue-A events'].f5.iloc[0]:.3f}"), ('axis', f"median misfit {va['axis_vs_fault_elongated']['median_deg']:.0f}°"),
 ('axis p', f"p = {va['axis_vs_fault_elongated']['p_binomial_lt45']:.2f}"), ('3D ARI', f"{va['3D']['ARI']:.3f}"), ('3D n', c(va['3D']['n'])), ('I2 med', f"{va['I2_median_zone_noise'][0]:.1f} and {va['I2_median_zone_noise'][1]:.1f}"),
 ('T-gnss', f"median of {va['T_vs_gnss']['median']}°"), ('ens I2', f"{CS[CS.variant=='ensemble-DBKM'].I2_med_in.iloc[0]:.1f}"),
 ('Denizli DBKM hom30', f"{den[(den.variant=='ensemble-DBKM')&(den.etas=='hom30')].J_boot.iloc[0]:.3f}"), ('Denizli HDB hom30', f"{den[(den.variant=='ensemble-HDB')&(den.etas=='hom30')].J_boot.iloc[0]:.3f}"),
]
for m in FS.index:
    r = FS.loc[m]; checks += [(f'{m} default', f"| {m} | {int(r.n_configs)} | {r.default_ARI_all:.3f} | {r.default_ARI_clu:.3f} | {r.default_zones} | {r.best_ARI_all:.3f} / {r.median_ARI_all:.3f} | {r.best_ARI_clu:.3f} / {r.median_ARI_clu:.3f} | {r.best_frac_events_stable:.3f} / {r.median_frac_events_stable:.3f} |")]
for _, r in zd.iterrows():
    checks.append((f'T3 zone {r.zone}', f"| {int(r.zone)} | {int(r.n)} | {r.Mmax_date} M{r.Mmax:.1f} | {r.J_mean:.3f} | {'yes' if r.stable else 'no'} | {r.Ptr_ETAS_hom:.2f}/{r.Ptr_ZBZ:.2f}/{r.Ptr_GK:.2f} | {r.kde_ratio:.2f} ({r.kde_lo:.2f}–{r.kde_hi:.2f}) | {r.frac_within5km_fault:.2f} |"))
for _, r in FT[FT.test == 'raw zones'].iterrows(): checks.append((f'fault {r.method}', f"| {r.method} zones | {r.f5:.3f} | {r.null_mean:.3f} ({r.null_p95:.3f}) |"))
# detectability under both MinPts counts (18_detectability.py), seed and Mc sensitivity (19_sensitivity.py)
import os
DT = J('T_detectability.json'); RD = DT['radii']; S4 = {k: RD[k]['sklearn_4_further'] for k in RD}; S5 = {k: RD[k]['strict_5_further'] for k in RD}
checks += [('enh 4 further', f"{S4['6.2km']['enhancement_needed']:.1f}× at the rule radius, {S4['10.0km']['enhancement_needed']:.1f}× at 10 km, {S4['15.0km']['enhancement_needed']:.1f}× at 15 km and {S4['20.0km']['enhancement_needed']:.2f}× at 20 km"),
 ('years 4 median', f"{S4['6.2km']['years_at_median']} yr at 6.22 km and {S4['10.0km']['years_at_median']} yr at 10 km"), ('years 4 max', f"{S4['6.2km']['years_at_max']} and {S4['10.0km']['years_at_max']} yr"),
 ('P core', f"{DT['P_core_rule_eps_median']['sklearn_4_further']:.3f}"), ('kde max', f"({DT['kde_max']:.2f})"),
 ('enh 5 further', f"{S5['6.2km']['enhancement_needed']:.1f}×, {S5['10.0km']['enhancement_needed']:.1f}×, {S5['15.0km']['enhancement_needed']:.1f}× and {S5['20.0km']['enhancement_needed']:.2f}×"),
 ('disc years', f"in about {S4['10.0km']['years_at_median']:.0f} or {S4['15.0km']['years_at_median']:.0f} years, and within the rule radius in about {S4['6.2km']['years_at_median']:.0f} years")]
SE = J('T_sensitivity.json'); de, dw, si, ae, m25, m35, bx = (SE[k] for k in ('denizli_ensemble', 'denizli_wdb15', 'simav_wdb15_ker30', 'se_aegean_wdb20', 'Mc25', 'Mc35', 'denizli_box'))
w10 = [z for z in m25['supported'] if z['variant'] == 'weighted-DBSCAN eps=10.0']; dn25 = [z for z in w10 if z['lon'] > 29][0]; sg25 = [z for z in w10 if z['lat'] > 39][0]
en25 = [z for z in m25['supported'] if z['variant'] == 'ensemble-DBKM'][0]; ag25 = [z for z in m25['supported'] if z['variant'] == 'weighted-DBSCAN eps=20.0'][0]; z35 = m35['supported'][0]
checks += [('seed ens', f"J_boot {de['J_boot_min']:.3f}–{de['J_boot_max']:.3f} over the {de['n_combinations']} seed and variant combinations and was supported in {de['n_supported']} of them"),
 ('seed wdb15 Denizli', f"(J_boot {dw['J_boot_min']:.3f}–{dw['J_boot_max']:.3f}; period J {min(dw['J_B']):.3f}–{max(dw['J_B']):.3f})"),
 ('seed Simav', f"(ker30; {si['J_boot_min']:.3f}–{si['J_boot_max']:.3f}; {si['J_B'][0]:.3f})"), ('seed SE Aegean', f"({ae['J_boot_min']:.3f}–{ae['J_boot_max']:.3f}; {min(ae['J_B']):.3f}–{max(ae['J_B']):.3f})"),
 ('Mc35', f"At Mc 3.5 ({m35['n_A']:,} events; rule ε {m35['eps_rule_km']:.2f} km), the only supported zone was a 20-km weighted-DBSCAN zone of {z35['n_events']} events near {z35['lon']:.2f}°E, {z35['lat']:.2f}°N"),
 ('Mc25', f"At Mc 2.5 ({m25['n_A']:,} events; rule ε {m25['eps_rule_km']:.1f} km), Denizli was supported by the DBKM ensemble (J_boot {en25['J_boot']:.3f}, period J {en25['J_period_B_ZBZ']:.3f}) and by 10-km weighted DBSCAN ({dn25['J_boot']:.3f}, {dn25['J_period_B_ZBZ']:.3f})"),
 ('Mc25 others', f"a {sg25['n_events']:,}-event zone near {sg25['lon']:.2f}°E, {sg25['lat']:.2f}°N and a 20-km aggregate of {ag25['n_events']:,} events"),
 ('Denizli box', f"{bx['n_2019'] + bx['n_2026']} of the {bx['n']} events occurred in 2019 and 2026, but these two years hold only {bx['sum_Pbg_hom30_2019_2026']} of the {bx['sum_Pbg_hom30']} expected background events ({bx['share_2019_2026']*100:.0f}%)"),
 ('depth', f"{round(100*SE['depth']['frac_round01_7'])}% of these events have a depth of 7.0 km to 0.1 km precision ({round(100*SE['depth']['frac_exact_7'])}% exactly 7.0 km)"),
 ('conc seed', f"supported in {de['n_supported']} of {de['n_combinations']} seed and variant combinations")]
# synthetic benchmark (audit/bench; optional, checked only if its summary exists)
BNF = f'{ROOT}/audit/bench/bench_numbers.json'
if os.path.exists(BNF):
    BN = json.load(open(BNF)); H0 = BN['H0']; SC = [BN[k] for k in ('SRC, E = 5', 'SRC, E = 10', 'SRC, E = 20')]; ALL = [H0] + SC
    best = lambda r: max(r[f'hits_{v}'] for v in ('ens-DBKM', 'wDB10', 'wDB15', 'wDB20'))
    checks += [('bench n', f"We processed {sum(r['n'] for r in ALL)} synthetic catalogues: {H0['n']} under H0 and {sum(r['n'] for r in SC)} with sources"),
     ('bench deg', f"In {sum(r['deg'] for r in ALL)} of them, the ETAS refit was degenerate"), ('bench err', f"and in {sum(r['err'] for r in ALL)} the refit failed"),
     ('bench H0 false', f"No converged H0 catalogue produced a supported zone ({H0['false_cats_converged']} of {H0['ok']}), but {H0['false_cats_degenerate']} of the {H0['deg']} degenerate fits did"),
     ('bench KDE', f"median KDE ratio of raw DBKM zones is {H0['kde_med_ok'][0]:.2f}–{H0['kde_med_ok'][1]:.2f} and the maximum {H0['kde_max_ok'][0]:.2f}–{H0['kde_max_ok'][1]:.2f}"),
     ('bench nA', f"The H0 catalogues hold {H0['nA_med']:,} events (median; range {H0['nA_min']:,}–{H0['nA_max']:,})")]
    for r, e in zip(SC, (5, 10, 20)):
        h = best(r); real = f"none of {r['src_total_ok']} sources for any real variant" if h == 0 else f"{h} of {r['src_total_ok']} sources for"
        checks.append((f'bench power E{e}', f"at E = {e}, {real}")); checks.append((f'bench oracle E{e}', f"against {r['hits_oracle-wDB10']} of {r['src_total_all']} for the oracle at 10 km"))
        checks.append((f'bench abstract E{e}', f"{h} of {r['src_total_ok']} at {e} times"))
    for r, e in zip(SC, (5, 10, 20)): checks.append((f'bench sup E{e}', f"{r['n']} ({r['ok']} / {r['deg']} / {r['err']})"))
    checks.append(('bench sup H0', f"{H0['n']} ({H0['ok']} / {H0['deg']} / {H0['err']})"))
hl_checks = [('HL trig', f"{round(100*st.Ptr_ETAS_hom.min())}–{round(100*st.Ptr_ETAS_hom.max())}%"), ('HL bg', f"{cq['expected_bg_within_6.2km']:.1f} background events per decade")]
bad = 0; out = []
for k, v in checks:
    ok = v in ms; bad += not ok; out.append(f"{'OK  ' if ok else 'FAIL'} {k}: '{v}'")
for k, v in hl_checks:
    ok = v in hl; bad += not ok; out.append(f"{'OK  ' if ok else 'FAIL'} {k}: '{v}' (highlights)")
for l in hl.strip().splitlines():
    if l.startswith('-'): L = len(l[1:].strip()); ok = L <= 85; bad += not ok; out.append(f"{'OK  ' if ok else 'FAIL'} highlight length {L} <= 85")
# old wanat2 values must not leak into the new manuscript
for old in ['6,439', '12,626', '0.989', '5,913', '6.99', 'p = 0.006', 'Kruskal–Wallis p']:
    ok = old not in ms; bad += not ok; out.append(f"{'OK  ' if ok else 'FAIL'} stale value absent: '{old}'")
out.append(f'{len(out) - bad} passed, {bad} failed'); s = '\n'.join(out); print(s); open(f'{ROOT}/logs/30_check_numbers.log', 'w').write(s + '\n'); sys.exit(1 if bad else 0)
