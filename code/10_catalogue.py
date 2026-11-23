"""wanat3 Stage C1: AFAD 2016-2026 catalogue (period A) + ISC 2000-2015 (period B); magnitude audit; AFAD-vs-reference
Mw cross-match; Kadirioglu & Kartal (2016) ML->Mw check; Mc by MAXC+0.2 and MBS (Woessner & Wiemer 2005), incl. time variation."""
import numpy as np, pandas as pd, json, sys
from common import *
REG = dict(lon0=26.0, lon1=31.0, lat0=36.0, lat1=41.0)
def inreg(d): return d[d.lon.between(REG['lon0'], REG['lon1']) & d.lat.between(REG['lat0'], REG['lat1'])]
# ---------- AFAD (merge M>=2.5 bbox file with the low-magnitude file; dedupe on eventID)
a1 = pd.read_csv(f'{ROOT}/data/w_anatolia/earthquakes/afad_bbox_2016_2026_M2.5.csv')
a2 = pd.read_csv(f'{ROOT}/data/w_anatolia_extra/cat/afad_M1_lowmag.csv'); a2.columns = [c[0].lower() + c[1:] for c in a2.columns]
a2 = a2.rename(columns={'eventID': 'eventID'})
for a in (a1, a2):
    a.columns = [c if c != 'EventID' else 'eventID' for c in a.columns]
a1['src'] = 'afad_M2.5'; a2['src'] = 'afad_lowmag'
a1['time'] = pd.to_datetime(a1.date, format='mixed', utc=True); a2['time'] = pd.to_datetime(a2.date, format='mixed', utc=True)
A = pd.concat([a1, a2]).drop_duplicates('eventID').rename(columns={'latitude': 'lat', 'longitude': 'lon', 'magnitude': 'M', 'type': 'magType'})
A['magType'] = A.magType.str.upper().replace({'MW': 'MW'}); A['magType'] = A.magType.replace({'MW': 'Mw'}).str.replace('ML', 'ML')
A = inreg(A).sort_values('time').reset_index(drop=True)
A['id'] = A.eventID
info = {'afad_rows_M2.5_file_in_region': int(len(inreg(a1.rename(columns={'latitude': 'lat', 'longitude': 'lon'})))),
        'afad_rows_merged_in_region': int(len(A)), 'afad_t0': str(A.time.min()), 'afad_t1': str(A.time.max())}
A['M'] = A.M.astype(float)
# magnitude-type table (S)
bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 9]
mt = pd.crosstab(pd.cut(A.M, bins, right=False), A.magType); mt.to_csv(f'{ROOT}/out/tables/S_afad_magtype_by_bin.csv')
yr = pd.crosstab(A.time.dt.year, A.magType); yr.to_csv(f'{ROOT}/out/tables/S_afad_magtype_by_year.csv')
# ---------- cross-match AFAD vs independent moment magnitudes in ISC bulletin (MagAuthor != AFAD)
I = pd.read_csv(f'{ROOT}/data/w_anatolia/earthquakes/isc_fdsn_bbox_2016_2026_M2.5.csv')
I['time'] = pd.to_datetime(I.Time, format='mixed', utc=True)
I = I[I.MagType.str.upper().str.startswith('MW') & (I.MagAuthor != 'AFAD')].reset_index(drop=True)
tA = A.time.values.astype('int64') / 1e9; tI = I.time.values.astype('int64') / 1e9
xyA = project(A.lon, A.lat); xyI = project(I.Longitude, I.Latitude)
o = np.argsort(tA); tAs = tA[o]; rows = []
for i in range(len(I)):
    lo, hi = np.searchsorted(tAs, [tI[i] - 10, tI[i] + 10]); cand = o[lo:hi]
    if len(cand) == 0: continue
    r = np.hypot(*(xyA[cand] - xyI[i]).T); ok = r <= 30
    if not ok.any(): continue
    j = cand[ok][np.argmin(np.abs(tA[cand[ok]] - tI[i]))]
    rows.append(dict(afad_id=A.id[j], afad_type=A.magType[j], afad_M=A.M[j], ref_Mw=I.Magnitude[i], ref_author=I.MagAuthor[i]))
X = pd.DataFrame(rows); X.to_csv(f'{ROOT}/out/tables/S_afad_vs_ref_Mw_matches.csv', index=False)
KK = lambda ml: 0.8095 * ml + 1.3003     # Kadirioglu & Kartal (2016) eq. 3c, OLS, valid 3.3<=ML<=6.6
cm = {}
for t in ['ML', 'Mw']:
    s = X[X.afad_type == t]
    if len(s) < 5: continue
    res = s.ref_Mw - s.afad_M; ent = dict(n=int(len(s)), median_ref_minus_afad=round(float(res.median()), 3), mean=round(float(res.mean()), 3),
                                        sd=round(float(res.std()), 3), afad_M_range=[float(s.afad_M.min()), float(s.afad_M.max())])
    if t == 'ML':
        s2 = s[s.afad_M >= 3.3]; rk = s2.ref_Mw - KK(s2.afad_M)
        ent['KK2016_resid_median'] = round(float(rk.median()), 3); ent['KK2016_resid_sd'] = round(float(rk.std()), 3); ent['n_ML>=3.3'] = int(len(s2))
        ent['identity_resid_median_ML>=3.3'] = round(float((s2.ref_Mw - s2.afad_M).median()), 3)
        p = np.polyfit(s.afad_M, s.ref_Mw, 1); ent['OLS_refMw_on_afadML'] = [round(float(p[0]), 3), round(float(p[1]), 3)]
    cm[t] = ent
info['crossmatch_vs_independent_Mw'] = cm
info['ref_authors'] = X.ref_author.value_counts().to_dict() if len(X) else {}
# ---------- Mc
def mbs(m, dm=0.1, win=0.5, nmin=50):
    """Woessner & Wiemer (2005) MBS: smallest Mc for which |b_ave(Mc..Mc+win) - b(Mc)| <= dbShiBolt(Mc)."""
    m = np.round(np.asarray(m) / dm) * dm; grid = np.round(np.arange(np.min(m), np.max(m) - win, dm), 2); bs = {}
    for mc in grid:
        mm = m[m >= mc - 1e-9]
        if len(mm) < nmin: break
        b = np.log10(np.e) / (mm.mean() - (mc - dm / 2)); sb = 2.3 * b ** 2 * np.sqrt(((mm - mm.mean()) ** 2).sum() / (len(mm) * (len(mm) - 1)))
        bs[mc] = (b, sb)
    ks = sorted(bs)
    for mc in ks:
        nxt = [bs[k][0] for k in ks if mc - 1e-9 <= k <= mc + win + 1e-9]
        if len([k for k in ks if k > mc + win - 1e-9]) == 0: break
        if abs(np.mean(nxt) - bs[mc][0]) <= bs[mc][1]: return float(mc)
    return np.nan
mc = {'all_maxc+0.2': mc_maxc(A.M), 'all_MBS': mbs(A.M)}
for t in ['ML', 'Mw']: mc[f'{t}_only_maxc+0.2'] = mc_maxc(A.M[A.magType == t])
ty = []
for y, g in A.groupby(A.time.dt.year):
    ty.append(dict(year=int(y), n=len(g), Mc_maxc02=mc_maxc(g.M), Mc_MBS=mbs(g.M), frac_ML=round(float((g.magType == 'ML').mean()), 3)))
ty = pd.DataFrame(ty); ty.to_csv(f'{ROOT}/out/tables/S_Mc_by_year_AFAD.csv', index=False)
# moving windows of 1000 events (step 500) on M>=2.0 events
g2 = A[A.M >= 2.0].reset_index(drop=True); mw = []
for s in range(0, len(g2) - 1000, 500):
    w = g2.iloc[s:s + 1000]; mw.append(dict(t_mid=str(w.time.iloc[500]), Mc_maxc02=mc_maxc(w.M), Mc_MBS=mbs(w.M)))
mw = pd.DataFrame(mw); mw.to_csv(f'{ROOT}/out/tables/S_Mc_moving_AFAD.csv', index=False)
mc['moving_window_maxc02_p95'] = float(mw.Mc_maxc02.quantile(0.95)); mc['moving_window_MBS_p95'] = float(mw.Mc_MBS.quantile(0.95))
mc['moving_window_maxc02_max'] = float(mw.Mc_maxc02.max()); mc['moving_window_MBS_max'] = float(np.nanmax(mw.Mc_MBS))
mc['annual_maxc02_max'] = float(ty.Mc_maxc02.max()); mc['annual_MBS_max'] = float(np.nanmax(ty.Mc_MBS))
info['Mc_AFAD'] = mc
# ---------- ISC period B 2000-2015
isc = inreg(load_isc()); B = isc[(isc.time >= pd.Timestamp('2000-01-01', tz='UTC')) & (isc.time < pd.Timestamp('2016-01-01', tz='UTC'))].copy()
B['magType'] = B.magType.str.upper()
mcB = {'all_maxc+0.2': mc_maxc(B.M), 'all_MBS': mbs(B.M)}
tyB = []
for y, g in B.groupby(B.time.dt.year):
    tyB.append(dict(year=int(y), n=len(g), Mc_maxc02=mc_maxc(g.M), Mc_MBS=mbs(g.M), top_type=g.magType.value_counts().index[0],
                    frac_top=round(float(g.magType.value_counts(normalize=True).iloc[0]), 3)))
tyB = pd.DataFrame(tyB); tyB.to_csv(f'{ROOT}/out/tables/S_Mc_by_year_ISC_B.csv', index=False)
mcB['annual_maxc02_max'] = float(tyB.Mc_maxc02.max()); mcB['annual_MBS_max'] = float(np.nanmax(tyB.Mc_MBS))
info['Mc_ISC_B'] = mcB; info['ISC_B_magtypes'] = B.magType.value_counts(normalize=True).round(3).head(5).to_dict(); info['ISC_B_n_all'] = int(len(B))
A.to_csv(f'{ROOT}/out/data/afad_all_region.csv', index=False); B.to_csv(f'{ROOT}/out/data/isc_B_all_region.csv', index=False)
json.dump(info, open(f'{ROOT}/out/tables/T_catalogue_audit.json', 'w'), indent=1, default=str)
print(mt); print(yr); print(ty.to_string()); print(tyB.to_string()); print(json.dumps(info, indent=1, default=str))
