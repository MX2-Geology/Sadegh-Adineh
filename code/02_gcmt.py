"""Parse GCMT NDK files (1976-2024) inside the study box."""
import glob, numpy as np, pandas as pd
from common import ROOT, BOX
rows = []
for f in sorted(glob.glob(f'{ROOT}/data/raw/gcmt/*.ndk')):
    L = open(f).read().splitlines()
    for i in range(0, len(L) - 4, 5):
        l1, l3, l4, l5 = L[i], L[i + 2], L[i + 3], L[i + 4]
        c = l3.split(); lat, lon, dep = float(c[3]), float(c[5]), float(c[7])
        if not (BOX['lon0'] <= lon <= BOX['lon1'] and BOX['lat0'] <= lat <= BOX['lat1']): continue
        ex = int(l4.split()[0]); t = l5.split()
        m0 = float(t[10]) * 10 ** ex
        rows.append(dict(date=l1[5:15], lat=lat, lon=lon, depth=dep, Mw=round(2 / 3 * np.log10(m0) - 10.7, 2),
                         T_pl=int(t[2]), T_az=int(t[3]), P_pl=int(t[8]), P_az=int(t[9]), strike=int(t[11]), dip=int(t[12]), rake=int(t[13])))
g = pd.DataFrame(rows)
def style(r):
    if r.P_pl >= 52 and r.T_pl <= 35: return 'normal'
    if r.T_pl >= 52 and r.P_pl <= 35: return 'thrust'
    if r.P_pl <= 40 and r.T_pl <= 20 or r.T_pl <= 40 and r.P_pl <= 20: return 'strike-slip'
    return 'oblique'
g['style'] = g.apply(style, axis=1)   # Zoback (1992)-type plunge criteria, simplified
g.to_csv(f'{ROOT}/out/data/gcmt_box.csv', index=False)
print(len(g), g.date.min(), g.date.max()); print(g['style'].value_counts())
