"""wanat3 shared helpers: loading catalogue A with labels and ETAS background probabilities; zone geometry; fault distances."""
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import LineString, MultiPoint, Point
from shapely import STRtree
from matplotlib.path import Path
from common import ROOT, project, SEED
PRIMARY = 'hom30'      # a priori primary ETAS run (spatially homogeneous mu, Mc 3.0); 'ker30' = kernel-smoothed sensitivity run
def load_A():
    A = pd.read_csv(f'{ROOT}/out/data/cat_A_labels.csv')
    for tag in ('hom30', 'ker30', 'hom32'):
        try:
            p = pd.read_csv(f'{ROOT}/out/data/etas_{tag}_Pbg.csv'); A[f'P_bg_{tag}'] = p.P_bg.values
        except FileNotFoundError: pass
    return A
def real(tag): return np.load(f'{ROOT}/out/data/etas_{tag}_realisations.npy')
_g = None
def fault_lines():
    global _g
    if _g is None:
        g = gpd.read_file(f'{ROOT}/data/raw/gem_gaf.geojson').cx[25.5:31.5, 35.5:41.5]; L = []
        for geom in g.geometry:
            for part in (geom.geoms if geom.geom_type.startswith('Multi') else [geom]):
                c = np.array(part.coords); L.append(LineString(project(c[:, 0], c[:, 1])))
        _g = (L, STRtree(L))
    return _g
def fault_dist(P):
    L, T = fault_lines(); idx = T.nearest([Point(p) for p in P]); return np.array([L[i].distance(Point(p)) for i, p in zip(idx, P)])
REGPOLY = MultiPoint(project([26, 31, 31, 26], [36, 36, 41, 41])).convex_hull
def hull(Q, buf=5.0): return MultiPoint(Q).convex_hull.buffer(buf)
def inside(poly, P):
    if poly.geom_type == 'MultiPolygon': return np.any([inside(p, P) for p in poly.geoms], 0)
    return Path(np.asarray(poly.exterior.coords)).contains_points(P)
