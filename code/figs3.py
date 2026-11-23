"""wanat3 Figs 2-7 and Supplement figures from saved outputs (no re-analysis except re-running the Croatia DBSCAN/Rough-DBSCAN
configurations listed in T_croatia_rough.csv, deterministic). 400 dpi JPG, lat/long ticks, scale bar, north arrow on maps."""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.colors as mcol
from matplotlib.patches import Polygon
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from figbase import *
from common import ROOT, scitovski_eps, PROJ
from rough_dbscan import rough_dbscan
_t = plt.get_cmap('tab20'); COL = [_t(i) for i in range(20) if i not in (14, 15)]
T = f'{ROOT}/out/tables'; D = f'{ROOT}/out/data'
A = pd.read_csv(f'{D}/cat_A_labels_constructive.csv')
F = json.load(open(f'{ROOT}/data/raw/gem_gaf.geojson'))
def faults(ax, c='#b2182b', lw=0.5):
    for f in F['features']:
        gm = f['geometry']; parts = gm['coordinates'] if gm['type'] == 'MultiLineString' else [gm['coordinates']]
        for p in parts: p = np.array(p); ax.plot(p[:, 0], p[:, 1], color=c, lw=lw, transform=PC, zorder=3)
def pts(ax, lon, lat, lab, s=0.8):
    n = lab < 0; ax.scatter(lon[n], lat[n], s=0.3, c='lightgrey', transform=PC, zorder=2)
    ax.scatter(lon[~n], lat[~n], s=s, c=[COL[i % len(COL)] for i in lab[~n]], transform=PC, zorder=4)
# ---------- Fig 2: Scitovski / Croatia
d = pd.read_csv(f'{ROOT}/data/raw/usgs_croatia_1950_2017-05-25.csv'); d = d[d.type == 'earthquake']
B = np.c_[(d.longitude - 12.5) / 8.5, (d.latitude - 41.5) / 6.0]
CR = pd.read_csv(f'{T}/T_croatia_rough.csv')
fig = plt.figure(figsize=(7.2, 6.8))
l1 = DBSCAN(eps=0.016, min_samples=5).fit_predict(B)
l2, nl = rough_dbscan(B, 0.016, 5, 0.25 * 0.016, None)
for i, (lab, t) in enumerate([(l1, 'exact DBSCAN'), (l2, f'Rough-DBSCAN ($\\tau$=$\\varepsilon$/4, {nl} leaders)')]):
    ax = fig.add_subplot(2, 2, i + 1, projection=PC); basemap(ax, [12.5, 21, 41.5, 47.5], labels=(i == 0, True), sb='ll')
    pts(ax, d.longitude.values, d.latitude.values, lab, 0.6)
    ax.set_title(f'({"ab"[i]}) Croatia, {t}\n$\\varepsilon$=0.016: {lab.max()+1} clusters, {(lab<0).sum()} noise (paper 15, 1006)', fontsize=7)
ax = fig.add_subplot(223); lab_ = [f"{'E' if a.startswith('exact') else 'R'} {e:.4f}\n{t if isinstance(t, str) else ''} {o if o != '-' else ''}" for a, e, t, o in zip(CR.algorithm, CR.eps, CR.tau, CR.order)]
x = np.arange(len(CR)); ax.bar(x - 0.2, CR.n_clusters, 0.4, label='clusters (all)', color='grey'); ax.bar(x + 0.2, CR.n_clusters_ge12, 0.4, label='clusters $\\geq$12 events', color='k')
ax.axhline(15, color='r', ls='--', lw=0.8, label='paper: 15'); ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=5)
ax.set_xlabel('run (rows of Table S5)'); ax.set_ylabel('number of clusters'); ax.legend(fontsize=6); ax.set_title('(c) all 20 Croatia runs vs published partition', fontsize=7)
ax = fig.add_subplot(224); XY = A[['x', 'y']].values; e = scitovski_eps(XY, 5)
r = np.sort(NearestNeighbors(n_neighbors=5).fit(XY).kneighbors(XY)[0][:, -1]); ax.plot(r, 'k', lw=0.8)
for v, s in [(e['r1'], 'r$_1$'), (e['r2'], 'r$_2$'), (e['eps'], '$\\varepsilon$')]: ax.axhline(v, ls='--', lw=0.7); ax.text(len(r) * 0.02, v * 1.12, s)
ax.set_yscale('log'); ax.set_xlabel('event rank'); ax.set_ylabel('4th-neighbour distance (km)'); ax.set_title(f'(d) W Anatolia cat. A: $\\varepsilon$ = {e["eps"]:.2f} km', fontsize=7)
fig.tight_layout(); save(fig, 'Fig02_scitovski_reproduction')
# ---------- Fig 3: four methods (default configs)
C = pd.read_csv(f'{T}/T_fair_stability_configs.csv'); Cd = C[C['default']].set_index('method')
names = dict(SCIT='DBSCAN, Scitovski $\\varepsilon$ (unit square)', DBKM='DBSCAN, Scitovski $\\varepsilon$ (km)', HDB='HDBSCAN (mcs 50, ms 5)', OPT='OPTICS ($\\xi$=0.05, ms 5)')
fig = plt.figure(figsize=(7.2, 7.4))
for i, m in enumerate(['SCIT', 'DBKM', 'HDB', 'OPT']):
    ax = fig.add_subplot(2, 2, i + 1, projection=PC); basemap(ax, labels=(i % 2 == 0, i >= 2)); faults(ax, '#555555', 0.4)
    pts(ax, A.lon.values, A.lat.values, A[f'z_{m}'].values)
    ax.set_title(f'({"abcd"[i]}) {names[m]}\n{int(Cd.loc[m, "n_zones"])} zones, noise {Cd.loc[m, "noise_frac"]:.2f}', fontsize=7)
fig.tight_layout(); save(fig, 'Fig03_method_zones')
# ---------- Fig 4: fair stability
ZJ = pd.read_csv(f'{T}/S_fair_stability_zone_jaccard.csv')
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
mk = dict(SCIT='o', DBKM='s', HDB='^', OPT='D')
for j, m in enumerate(mk):
    s = C[C.method == m]; ax[0].scatter(s.ARI_all_med, s.ARI_clu_med, marker=mk[m], color=f'C{j}', s=18, alpha=0.6, label=m)
    sd = s[s['default']]; ax[0].scatter(sd.ARI_all_med, sd.ARI_clu_med, marker=mk[m], facecolor='none', edgecolor='k', s=60, lw=1.2)
ax[0].plot([0, 1], [0, 1], 'k:', lw=0.6); ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel('median ARI, all events (noise = one label)'); ax[0].set_ylabel('median ARI, events clustered in both'); ax[0].legend(fontsize=6, loc='lower right')
ax[0].set_title('(a) 30 configurations, B = 200 (circled: defaults)', fontsize=7)
data = [ZJ[ZJ.cfg == int(Cd.loc[m, 'cfg'])].J_mean.values for m in mk]
ax[1].boxplot(data, tick_labels=list(mk), showfliers=False); 
for j, v in enumerate(data): ax[1].scatter(np.full(len(v), j + 1) + np.random.default_rng(1).uniform(-0.12, 0.12, len(v)), v, s=8, color=f'C{j}', zorder=3)
ax[1].axhline(0.75, color='r', ls='--', lw=0.8); ax[1].set_ylabel('per-zone mean Jaccard (Hennig 2007)'); ax[1].set_title('(b) per-zone Jaccard, default configurations', fontsize=7)
fig.tight_layout(); save(fig, 'Fig04_fair_stability')
# ---------- Fig 5: triggering and persistence
Z = pd.read_csv(f'{T}/T_zones_all_methods.csv')
fig = plt.figure(figsize=(7.2, 6.6))
a1 = fig.add_subplot(221, projection=PC); basemap(a1); faults(a1, '#555555', 0.4)
s = A.P_bg_hom30.notna(); sc = a1.scatter(A.lon[s], A.lat[s], s=1, c=1 - A.P_bg_hom30[s], cmap='plasma', vmin=0, vmax=1, transform=PC, zorder=4)
fig.colorbar(sc, ax=a1, shrink=0.7, label='ETAS P(triggered)'); a1.set_title('(a) ETAS (hom30) triggering probability', fontsize=7)
a2 = fig.add_subplot(222); zd = Z[Z.method == 'DBKM'].sort_values('zone'); x = np.arange(len(zd))
for k, (c, lab_) in enumerate([('Ptr_ETAS_hom', 'ETAS hom30'), ('Ptr_ETAS_ker', 'ETAS ker30'), ('Ptr_ZBZ', 'ZBZ'), ('Ptr_GK', 'GK74')]):
    a2.bar(x + (k - 1.5) * 0.2, zd[c], 0.2, label=lab_)
a2.set_xticks(x); a2.set_xticklabels(zd.zone); a2.set_ylim(0, 1.25); a2.set_xlabel('DBKM zone'); a2.set_ylabel('triggered fraction'); a2.legend(fontsize=5.5, ncol=4, loc='upper center')
a2.set_title('(b) zone triggered fractions by declustering method', fontsize=7)
a3 = fig.add_subplot(212)
for j, m in enumerate(mk):
    s = Z[Z.method == m]; a3.errorbar(s.persist_hom, s.kde_ratio, yerr=[s.kde_ratio - s.kde_lo, s.kde_hi - s.kde_ratio], fmt=mk[m], ms=4, color=f'C{j}', alpha=0.7, lw=0.6, label=m)
a3.axhline(1, color='k', lw=0.6); a3.axvline(0.5, color='r', ls='--', lw=0.8); a3.set_xlabel('ETAS re-clustering persistence (mean max Jaccard, K = 20)')
a3.set_ylabel('background KDE ratio inside/outside (95% CI)'); a3.legend(fontsize=6); a3.set_title('(c) two persistence measures for all 75 raw zones', fontsize=7)
fig.tight_layout(); save(fig, 'Fig05_triggering_persistence')
# ---------- Fig 6: validation map with leader lines (numbers only; values in Table 3)
grid = pd.read_csv(f'{D}/gnss_strain_grid.csv'); gc = pd.read_csv(f'{D}/gcmt_box.csv')
fig = plt.figure(figsize=(7.2, 6.6)); ax = fig.add_subplot(111, projection=PC); basemap(ax)
gg = grid.pivot(index='lat', columns='lon', values='I2')
cf_ = ax.pcolormesh(gg.columns, gg.index, gg.values, cmap='YlOrBr', vmin=0, vmax=150, alpha=0.6, transform=PC, zorder=1, shading='nearest')
fig.colorbar(cf_, ax=ax, shrink=0.6, pad=0.02, label='GNSS I$_2$ (nanostrain/yr)'); faults(ax, '#b2182b', 0.5)
for _, r in grid.iloc[::3].iterrows():
    a = np.radians(r.ext_az); L = 0.08; ax.plot([r.lon - L * np.sin(a), r.lon + L * np.sin(a)], [r.lat - L * np.cos(a), r.lat + L * np.cos(a)], color='k', lw=0.5, transform=PC, zorder=2)
nm = gc[gc['style'] == 'normal']
for _, r in nm.iterrows():
    a = np.radians(r.T_az); L = 0.12; ax.plot([r.lon - L * np.sin(a), r.lon + L * np.sin(a)], [r.lat - L * np.cos(a), r.lat + L * np.cos(a)], color='tab:blue', lw=1.2, transform=PC, zorder=6)
lab = A.z_DBKM.values; cen = np.array([28.6, 38.4])
for _, z in zd.iterrows():
    E = A[lab == z.zone]; ax.scatter(E.lon, E.lat, s=0.8, color=COL[int(z.zone) % len(COL)], transform=PC, zorder=4)
    p = np.array([z.lon, z.lat]); v = p - cen; v = v / (np.linalg.norm(v) + 1e-9); q = p + 0.45 * v
    q = np.clip(q, [26.1, 36.1], [30.9, 40.9])
    ax.annotate(f"{int(z.zone)}{'*' if z.stable else ''}", xy=p, xytext=q, fontsize=7, ha='center', va='center', xycoords=PC._as_mpl_transform(ax), textcoords=PC._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle='-', lw=0.6, color='k'), bbox=dict(fc='white', ec='k', lw=0.4, pad=0.2), zorder=9)
ax.set_title('DBKM zones (number; * bootstrap-stable, J$\\geq$0.75); GNSS I$_2$ and extension axes (black);\nGCMT normal-fault T-axes (blue); GEM GAF faults (red). Zone values in Table 3.', fontsize=7)
save(fig, 'Fig06_validation_map')
# ---------- Fig 7: constructive zoning
CZ = pd.read_csv(f'{T}/T_constructive_zones.csv'); MB = pd.read_csv(f'{D}/membership_hom30_DBKM.csv')
def cellpoly(cx, cy, h=10.0):
    xs = np.array([cx, cx + 1, cx + 1, cx]) * h; ys = np.array([cy, cy, cy + 1, cy + 1]) * h; lo, la = PROJ(xs, ys, inverse=True); return np.c_[lo, la]
fig = plt.figure(figsize=(7.2, 7.0))
ax = fig.add_subplot(221, projection=PC); basemap(ax, labels=(True, False)); faults(ax, '#555555', 0.3)
cmap = plt.get_cmap('viridis'); 
for r in MB[MB.Pcell > 0].itertuples():
    ax.add_patch(Polygon(cellpoly(r.cx, r.cy), closed=True, fc=cmap(r.Pcell), ec='none', alpha=0.85, transform=PC, zorder=2))
for r in MB[MB.zone >= 0].itertuples():
    ax.add_patch(Polygon(cellpoly(r.cx, r.cy), closed=True, fc='none', ec='r', lw=0.5, transform=PC, zorder=5))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcol.Normalize(0, 1)); fig.colorbar(sm, ax=ax, shrink=0.7, label='cell membership probability')
ax.set_title('(a) ensemble DBSCAN over 20 ETAS realisations:\nP$_{cell}$ and consensus zones (red)', fontsize=7)
ax = fig.add_subplot(222, projection=PC); basemap(ax, labels=(False, False)); faults(ax, '#555555', 0.3)
pts(ax, A.lon.values, A.lat.values, A.zc_wdb15.values); ax.set_title('(b) P$_{bg}$-weighted DBSCAN, $\\varepsilon$ = 15 km\n(weighted MinPts 5)', fontsize=7)
Bc = pd.read_csv(f'{D}/cat_B_ISC_2000_2015.csv'); b0 = Bc[Bc.bg_zbz]; XB = b0[['x', 'y']].values; eB = scitovski_eps(XB, 5)['eps']
from clustering import prune
lB = prune(DBSCAN(eps=eB, min_samples=5).fit_predict(XB), 25)
ax = fig.add_subplot(223, projection=PC); basemap(ax, labels=(True, True)); faults(ax, '#555555', 0.3)
pts(ax, b0.lon.values, b0.lat.values, lB, 0.5)
for r in MB[MB.zone >= 0].itertuples():
    ax.add_patch(Polygon(cellpoly(r.cx, r.cy), closed=True, fc='none', ec='r', lw=0.4, transform=PC, zorder=5))
ax.set_title(f'(c) independent period B (ISC 2000–2015, ZBZ background):\nDBSCAN $\\varepsilon$ = {eB:.1f} km; red = panel (a) zones', fontsize=7)
ax = fig.add_subplot(224)
for j, (v, mk_) in enumerate([('ensemble-DBKM', 'o'), ('ensemble-HDB', '^'), ('weighted-DBSCAN eps=15.0', 's'), ('weighted-DBSCAN eps=20.0', 'D'), ('weighted-DBSCAN eps=10.0', 'v')]):
    s = CZ[(CZ.variant == v) & (CZ.etas == 'hom30')]; ax.scatter(s.J_boot, s.J_period_B_ZBZ, marker=mk_, s=10 + s.exp_bg / 4, color=f'C{j}', alpha=0.7, label=v.replace('weighted-DBSCAN ', 'wDBSCAN '))
ax.axvline(0.75, color='r', ls='--', lw=0.8); ax.axhline(0.5, color='r', ls='--', lw=0.8); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.text(0.77, 0.95, 'supported', fontsize=6, color='r'); ax.set_xlabel('bootstrap Jaccard (B = 200)'); ax.set_ylabel('period-B cell Jaccard')
ax.legend(fontsize=5, loc='upper left'); ax.set_title('(d) pre-set support rule (hom30; size $\\propto$ expected background)', fontsize=7)
fig.tight_layout(); save(fig, 'Fig07_constructive_zoning')
# ---------- Supplement: Mc in time, b/depth
mv = pd.read_csv(f'{T}/S_Mc_moving_AFAD.csv'); yr = pd.read_csv(f'{T}/S_Mc_by_year_AFAD.csv'); yb = pd.read_csv(f'{T}/S_Mc_by_year_ISC_B.csv')
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2)); t = pd.to_datetime(mv.t_mid, utc=True, format='ISO8601')
ax[0].plot(t, mv.Mc_maxc02, 'k.-', ms=3, lw=0.6, label='MAXC+0.2'); ax[0].plot(t, mv.Mc_MBS, 'r.-', ms=3, lw=0.6, label='MBS (Woessner & Wiemer 2005)')
ax[0].axhline(3.0, color='b', ls='--', lw=0.8, label='analysis Mc 3.0'); ax[0].set_ylabel('Mc'); ax[0].legend(fontsize=6); ax[0].set_title('(a) AFAD, moving windows of 1000 events', fontsize=7)
ax[1].plot(yr.year, yr.Mc_maxc02, 'ko-', ms=3, label='AFAD MAXC+0.2'); ax[1].plot(yr.year, yr.Mc_MBS, 'ro-', ms=3, label='AFAD MBS')
ax[1].plot(yb.year, yb.Mc_maxc02, 'ks--', ms=3, label='ISC B MAXC+0.2'); ax[1].plot(yb.year, yb.Mc_MBS, 'rs--', ms=3, label='ISC B MBS')
ax[1].axhline(3.0, color='b', ls='--', lw=0.8); ax[1].legend(fontsize=5.5); ax[1].set_title('(b) annual Mc', fontsize=7); ax[1].set_xlabel('year')
fig.tight_layout(); save(fig, 'FigS1_Mc_time')
Dm = pd.read_csv(f'{T}/T_domains.csv').sort_values('zone'); fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3)); x = np.arange(len(Dm)); c = ['k' if s else 'grey' for s in Dm.stable]
ax[0].errorbar(x, Dm.b, yerr=[Dm.b - Dm.b_lo, Dm.b_hi - Dm.b], fmt='none', ecolor='grey', lw=0.8); ax[0].scatter(x, Dm.b, c=c, s=14, label='b (Aki–Utsu, 95% CI)')
ax[0].scatter(x, Dm.b_positive, marker='x', c='tab:red', s=14, label='b-positive'); ax[0].set_xticks(x); ax[0].set_xticklabels(Dm.zone); ax[0].legend(fontsize=6); ax[0].set_xlabel('DBKM zone'); ax[0].set_ylabel('b')
ax[1].vlines(x, Dm.D50 - 0, Dm.D90, color=c, lw=2); ax[1].scatter(x, Dm.D50, c='w', edgecolor='k', s=14, zorder=3); ax[1].invert_yaxis(); ax[1].set_xticks(x); ax[1].set_xticklabels(Dm.zone)
ax[1].set_ylabel('depth (km): median to D90, free-depth events'); ax[1].set_xlabel('DBKM zone')
fig.tight_layout(); save(fig, 'FigS2_b_depth')
print('figures done')
