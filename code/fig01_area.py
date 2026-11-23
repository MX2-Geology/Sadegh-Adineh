"""Fig. 1 (final): study area - catalogue A (AFAD 2016-2026, M>=3.0), GEM GAF faults, NGL MIDAS GNSS velocities relative
to stable Eurasia (IGS14 velocities rotated with the ITRF2014 PMM Eurasia pole, Altamimi et al., 2017), GCMT faulting
styles, location inset. Full legend outside the map frame. Depth colour scale 0-40 km; deeper events are clipped to the
top colour (share printed in the caption). Data and plotted values are unchanged from v3; only layout/legend/colours."""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figbase import *
from common import ROOT
A = pd.read_csv(f'{ROOT}/out/data/cat_A_AFAD_2016_2026.csv')
g = pd.read_csv(f'{ROOT}/out/data/gnss_midas_eurasia.csv'); g = g[g.lon.between(26, 31) & g.lat.between(36, 41)]
c = pd.read_csv(f'{ROOT}/out/data/gcmt_box.csv'); cc = c[(c.lon.between(26, 31)) & (c.lat.between(36, 41))]
F = json.load(open(f'{ROOT}/data/raw/gem_gaf.geojson'))
FAULT = '#b2182b'
fig = plt.figure(figsize=(7.2, 5.6))
ax = fig.add_axes([0.065, 0.075, 0.60, 0.89], projection=PC); basemap(ax)
for f in F['features']:
    gm = f['geometry']; parts = gm['coordinates'] if gm['type'] == 'MultiLineString' else [gm['coordinates']]
    for p in parts:
        p = np.array(p); ax.plot(p[:, 0], p[:, 1], color=FAULT, lw=0.7, transform=PC, zorder=3)
DMAX = 40; clip = float((A.depth > DMAX).mean())
msize = lambda M: 1.5 * 2.2 ** (np.asarray(M) - 3)
sc = ax.scatter(A.lon, A.lat, s=msize(A.M), c=A.depth.clip(upper=DMAX), cmap='viridis_r', vmin=0, vmax=DMAX, lw=0, alpha=0.7, transform=PC, zorder=2)
q = ax.quiver(g.lon.values, g.lat.values, g.ve_eu.values, g.vn_eu.values, scale=250, width=0.0035, headwidth=3.5, color='k', transform=PC, zorder=5)
# GCMT faulting styles: Okabe-Ito colours (colour-blind safe)
col = {'normal': '#0072B2', 'strike-slip': '#E69F00', 'oblique': '#FFFFFF', 'thrust': '#CC79A7'}
for s, k in col.items():
    cs = cc[cc['style'] == s]; ax.scatter(cs.lon, cs.lat, marker='*', s=28, c=k, edgecolor='k', lw=0.35, transform=PC, zorder=6)
LAB = [('Gediz G.', 28.30, 38.43), ('B. Menderes G.', 28.05, 37.76), ('K. Menderes G.', 27.97, 38.20), ('Simav G.', 28.97, 39.09),
       ('Gulf of Gökova', 27.85, 36.97), ('Gulf of Edremit', 26.75, 39.52), ('Bakırçay G.', 27.25, 39.10)]
for n, x, y in LAB:
    ax.plot(x, y, 'k+', ms=4, mew=0.9, transform=PC, zorder=7)
    ax.text(x + 0.04, y + 0.03, n, fontsize=6, style='italic', transform=PC, zorder=7, clip_on=True,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.6))
# ---- right-hand column: inset, colour bar, legend (nothing drawn over the data)
X0, W = 0.715, 0.27
ia = fig.add_axes([X0 + 0.02, 0.775, W - 0.04, 0.19], projection=PC); ia.set_extent([19, 46, 33, 44], PC)
ia.add_feature(LAND); ia.add_feature(COAST); ia.set_facecolor('#dfeaf2')
ia.plot([26, 31, 31, 26, 26], [36, 36, 41, 41, 36], color='k', lw=1.0, transform=PC)
ia.text(33, 39, 'Anatolia', fontsize=5, transform=PC); ia.text(20.0, 34.0, 'Mediterranean Sea', fontsize=4.5, transform=PC)
ia.set_title('Location (box = map)', fontsize=6, pad=2)
ca = fig.add_axes([X0 + 0.01, 0.705, W - 0.02, 0.022])
cb = fig.colorbar(sc, cax=ca, orientation='horizontal', extend='max'); cb.ax.tick_params(labelsize=6)
cb.set_label(f'Epicentre depth (km); > {DMAX} km shown in darkest colour', fontsize=6, labelpad=2)
la = fig.add_axes([X0, 0.075, W, 0.56]); la.axis('off')
def leg(handles, title, y):
    L = la.legend(handles=handles, title=title, loc='upper left', bbox_to_anchor=(0.0, y), fontsize=6, frameon=False,
                  handlelength=1.6, borderaxespad=0, labelspacing=0.4, alignment='left', title_fontproperties={'size': 6.5, 'weight': 'bold'})
    la.add_artist(L); return L
leg([Line2D([], [], ls='none', marker='o', mfc='#5b8c5a', mec='none', alpha=0.8, ms=np.sqrt(msize(m)), label=f'M {m}') for m in (3, 4, 5, 6)],
    f'AFAD epicentres, M ≥ 3.0, 2016–2026\n(size = magnitude, colour = depth)', 1.02)
leg([Line2D([], [], ls='none', marker='*', mfc=k, mec='k', mew=0.4, ms=7, label=f'{s} ({(cc["style"] == s).sum()})') for s, k in col.items()],
    'GCMT focal mechanisms, 1979–2024\n(faulting style, number of events)', 0.76)
leg([Line2D([], [], color=FAULT, lw=1.0, label='Active fault trace\n(GEM GAF; Styron and Pagani, 2020)'),
     Line2D([], [], ls='none', marker='+', color='k', ms=6, mew=0.9, label='Graben / gulf (label point)'),
     Patch(fc='#f2efe6', ec='#555555', lw=0.4, label='Land'), Patch(fc='#dfeaf2', ec='#555555', lw=0.4, label='Sea')],
    'Other symbols', 0.50)
# GNSS key: arrow drawn at the map's scale, outside the map frame
la.text(0.0, 0.17, 'GNSS velocity relative to\nstable Eurasia (mm/yr)', fontsize=6.5, fontweight='bold', transform=la.transAxes, va='bottom')
la.text(0.0, 0.09, 'NGL MIDAS; Eurasia-fixed frame\n(ITRF2014 PMM, Altamimi et al., 2017)', fontsize=5.5, transform=la.transAxes, va='bottom')
qk = ax.quiverkey(q, X0 + 0.04, 0.10, 20, '20 mm/yr', labelpos='E', fontproperties={'size': 6}, coordinates='figure')
save(fig, 'Fig01_study_area')
print('n', len(A), 'clipped frac', round(clip, 4), 'gcmt', len(cc), dict(cc['style'].value_counts()), 'gnss', len(g), 'gcmt dates', cc.date.min(), cc.date.max())
