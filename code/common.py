"""Shared utilities for the Western Anatolia seismogenic-zoning pipeline."""
import numpy as np, pandas as pd, glob, os
from pyproj import Proj
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX = dict(lon0=25.0, lon1=31.0, lat0=36.0, lat1=41.0)
PROJ = Proj(proj='aeqd', lat_0=38.5, lon_0=28.0, datum='WGS84', units='km')
SEED = 20261028

def project(lon, lat):
    x, y = PROJ(np.asarray(lon), np.asarray(lat)); return np.c_[x, y]

def load_usgs():
    fs = sorted(glob.glob(f'{ROOT}/data/raw/usgs/usgs_????-??-??.csv'))
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates('id')
    d = d[d.type == 'earthquake'].copy()
    d['time'] = pd.to_datetime(d.time, utc=True, format='ISO8601')
    d = d.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'mag': 'M'})
    return d.sort_values('time').reset_index(drop=True)

def load_isc():
    fs = sorted(glob.glob(f'{ROOT}/data/raw/isc/isc_*.txt'))
    rows = []
    for f in fs:
        try:
            import io
            L = open(f).read().splitlines()
            L = [l for l in L if l and (not l.startswith('#') or l.startswith('#EventID'))]
            if len(L) < 2: continue
            t = pd.read_csv(io.StringIO('\n'.join(L)), sep='|')
        except Exception:
            continue
        if len(t): rows.append(t)
    d = pd.concat(rows)
    d.columns = [c.strip('#').strip() for c in d.columns]
    d = d.rename(columns={'EventID': 'id', 'Time': 'time', 'Latitude': 'lat', 'Longitude': 'lon', 'Depth/km': 'depth',
                          'Author': 'loc_author', 'MagType': 'magType', 'Magnitude': 'M', 'MagAuthor': 'mag_author',
                          'EventType': 'etype'})
    d = d.dropna(subset=['time', 'lat', 'lon', 'M']).drop_duplicates('id')
    d['time'] = pd.to_datetime(d.time, utc=True, format='ISO8601')
    d = d[d.etype.fillna('earthquake').str.contains('earthquake')]
    return d.sort_values('time').reset_index(drop=True)

def mc_maxc(m, dm=0.1, corr=0.2):
    m = np.round(np.asarray(m) / dm) * dm
    v, c = np.unique(np.round(m, 2), return_counts=True)
    return float(v[np.argmax(c)] + corr)

def b_aki(m, mc, dm=0.1):
    m = np.asarray(m); m = m[m >= mc - dm / 2 - 1e-9]
    if len(m) < 2: return np.nan, np.nan, len(m)
    b = np.log10(np.e) / (m.mean() - (mc - dm / 2))
    return b, b / np.sqrt(len(m)), len(m)   # Aki (1965) standard error

def scitovski_eps(B, minpts=4, frac=0.95):
    """Scitovski (2018) eps: radius of circle holding MinPts points (self included), 1-D LS 2-partition
    (globally optimal via exhaustive split on sorted radii), eps = right edge of smallest interval centred
    on the small-radius centroid r1 containing `frac` of that cluster."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=minpts).fit(B)
    rho = nn.kneighbors(B)[0][:, -1]
    s = np.sort(rho); n = len(s); cs = np.cumsum(s); cs2 = np.cumsum(s ** 2)
    k = np.arange(1, n)                       # split after k elements
    sse1 = cs2[k - 1] - cs[k - 1] ** 2 / k
    sse2 = (cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (n - k)
    kk = k[np.argmin(sse1 + sse2)]
    R1, R2 = s[:kk], s[kk:]
    r1, r2 = R1.mean(), R2.mean()
    delta = np.quantile(np.abs(R1 - r1), frac)
    return dict(eps=r1 + delta, r1=r1, r2=r2, n1=len(R1), n2=len(R2), rho=rho, lo=r1 - delta)
