"""GNSS velocities (NGL MIDAS, IGS14) -> Eurasia-fixed velocities and gridded strain-rate field."""
import numpy as np, pandas as pd
from common import ROOT, project
c = pd.read_csv(f'{ROOT}/data/raw/ngl_midas.txt', sep=r'\s+', header=None)
c = c.rename(columns={0: 'sta', 2: 't0', 3: 't1', 4: 'dur', 8: 've', 9: 'vn', 10: 'vu', 11: 'se', 12: 'sn', 24: 'lat', 25: 'lon'})
c['lon'] = ((c.lon + 180) % 360) - 180
r = c[(c.lon.between(24, 32)) & (c.lat.between(35, 42))].copy()
n_all = len(r)
r = r[(r.dur >= 3.0) & (r.se < 0.001) & (r.sn < 0.001) & (r.t1 >= 2015)].copy()
# Eurasia-fixed: ITRF2014 plate motion model, Eurasia pole (Altamimi et al., 2017): wx,wy,wz = -0.085,-0.531,0.770 mas/yr
mas = np.pi / 180 / 3600e3; w = np.array([-0.085, -0.531, 0.770]) * mas; R = 6371e3
la, lo = np.radians(r.lat.values), np.radians(r.lon.values)
X = R * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
V = np.cross(w, X)
eE = np.c_[-np.sin(lo), np.cos(lo), 0 * lo]; eN = np.c_[-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)]
r['ve_eu'] = (r.ve - (V * eE).sum(1)) * 1e3; r['vn_eu'] = (r.vn - (V * eN).sum(1)) * 1e3
r['se'] *= 1e3; r['sn'] *= 1e3
r = r[np.hypot(r.ve_eu, r.vn_eu) < 60]
r[['sta', 'lon', 'lat', 't0', 't1', 'dur', 've_eu', 'vn_eu', 'se', 'sn']].to_csv(f'{ROOT}/out/data/gnss_midas_eurasia.csv', index=False)
# gridded strain rate: Gaussian-weighted least squares of v = v0 + L x (Shen-type, fixed scale D)
P = project(r.lon, r.lat); Ve = r.ve_eu.values; Vn = r.vn_eu.values
gl, gt = np.meshgrid(np.arange(25.0, 31.01, 0.2), np.arange(36.0, 41.01, 0.2)); G = project(gl.ravel(), gt.ravel())
D = 75.0; rows = []
for (gx, gy), lo_, la_ in zip(G, gl.ravel(), gt.ravel()):
    dx = P[:, 0] - gx; dy = P[:, 1] - gy; dist = np.hypot(dx, dy)
    wgt = np.exp(-(dist / D) ** 2); m = dist < 3 * D
    if m.sum() < 6 or (dist < D).sum() < 3: continue
    A = np.zeros((2 * m.sum(), 6)); n = m.sum()
    A[:n, 0] = 1; A[:n, 2] = dx[m]; A[:n, 3] = dy[m]
    A[n:, 1] = 1; A[n:, 4] = dx[m]; A[n:, 5] = dy[m]
    sw = np.sqrt(np.r_[wgt[m] / r.se.values[m] ** 2, wgt[m] / r.sn.values[m] ** 2])
    sol, *_ = np.linalg.lstsq(A * sw[:, None], np.r_[Ve[m], Vn[m]] * sw, rcond=None)
    Lm = np.array([[sol[2], sol[3]], [sol[4], sol[5]]]) * 1e3  # mm/yr/km -> nanostrain/yr (1e-9/yr)
    E = 0.5 * (Lm + Lm.T); ev, evec = np.linalg.eigh(E)
    e1 = ev[1]; v1 = evec[:, 1]                      # most extensional
    az = np.degrees(np.arctan2(v1[0], v1[1])) % 180  # azimuth from north
    rows.append(dict(lon=lo_, lat=la_, n_sta=int((dist < D).sum()), e1=e1, e2=ev[0], dil=ev.sum(),
                     I2=np.sqrt(E[0, 0] ** 2 + E[1, 1] ** 2 + 2 * E[0, 1] ** 2), ext_az=az, maxshear=(ev[1] - ev[0]) / 2))
g = pd.DataFrame(rows); g.to_csv(f'{ROOT}/out/data/gnss_strain_grid.csv', index=False)
print('stations in 24-32E/35-42N:', n_all, 'used:', len(r), 'grid nodes:', len(g))
print(g[['I2', 'dil', 'ext_az']].describe().round(1))
