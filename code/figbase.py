"""Map helpers: PlateCarree map with lat/long ticks, scale bar and north arrow; 400 dpi JPG output."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, cartopy.crs as ccrs, cartopy.feature as cf
from common import ROOT
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9})
PC = ccrs.PlateCarree()
EXT = [26, 31, 36, 41]
LAND = cf.NaturalEarthFeature('physical', 'land', '10m', facecolor='#f2efe6', edgecolor='none')
COAST = cf.NaturalEarthFeature('physical', 'coastline', '10m', facecolor='none', edgecolor='#555555', linewidth=0.4)
def basemap(ax, ext=EXT, labels=(True, True), scale_km=100, sb='lr'):
    ax.set_extent(ext, PC); ax.add_feature(LAND, zorder=0); ax.add_feature(COAST, zorder=1)
    ax.set_facecolor('#dfeaf2')
    gl = ax.gridlines(PC, draw_labels=True, linewidth=0.3, color='grey', alpha=0.5, linestyle=':')
    gl.top_labels = gl.right_labels = False; gl.left_labels = labels[0]; gl.bottom_labels = labels[1]
    gl.xlabel_style = gl.ylabel_style = {'size': 7}
    # scale bar (lower left), length computed at bar latitude
    lat = ext[2] + 0.05 * (ext[3] - ext[2]); lon0 = ext[0] + 0.05 * (ext[1] - ext[0]) if sb == 'll' else ext[1] - 0.06 * (ext[1] - ext[0]) - scale_km / (111.32 * np.cos(np.radians(lat)))
    dlon = scale_km / (111.32 * np.cos(np.radians(lat)))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((lon0 - 0.03 * (ext[1] - ext[0]), lat - 0.03 * (ext[3] - ext[2])), dlon + 0.06 * (ext[1] - ext[0]), 0.1 * (ext[3] - ext[2]), fc='white', ec='none', alpha=0.8, transform=PC, zorder=9))
    ax.plot([lon0, lon0 + dlon], [lat, lat], color='k', lw=2, transform=PC, zorder=10, solid_capstyle='butt')
    ax.text(lon0 + dlon / 2, lat + 0.04 * (ext[3] - ext[2]), f'{scale_km} km', ha='center', fontsize=7, transform=PC, zorder=10)
    # north arrow (upper right)
    x = ext[1] - 0.07 * (ext[1] - ext[0]); y = ext[3] - 0.17 * (ext[3] - ext[2])
    ax.annotate('N', xy=(x, y + 0.11 * (ext[3] - ext[2])), xytext=(x, y), ha='center', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2), xycoords=PC._as_mpl_transform(ax), textcoords=PC._as_mpl_transform(ax), zorder=10)
    return ax
def save(fig, name):
    fig.savefig(f'{ROOT}/out/figs/{name}.jpg', dpi=400, bbox_inches='tight', pil_kwargs={'quality': 92}); plt.close(fig)
