import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from shapely.geometry import box
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & prep ────────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson"
gdf = gpd.read_file(url)

california  = gdf[gdf['STATE'] == '06'].copy().to_crs(epsg=3310)
fresno      = california[california['NAME'] == 'Fresno']
kern        = california[california['NAME'] == 'Kern']
highlighted = california[california['NAME'].isin(['Kern', 'Fresno'])]
others      = california[~california['NAME'].isin(['Kern', 'Fresno'])]

# US states for inset
us_states   = gdf.dissolve(by='STATE').to_crs(epsg=4326)
ca_state    = gdf[gdf['STATE'] == '06'].dissolve().to_crs(epsg=4326)

# ── 2. Academic palette ───────────────────────────────────────────────────────
BG          = 'white'
OTHER_FILL  = '#d9d9d9'      # light neutral grey
OTHER_EDGE  = '#ffffff'      # white borders for clean separation
CA_EDGE     = '#555555'      # state outline
FRESNO_FILL = '#f4a261'      # muted amber (colorblind-considerate)
KERN_FILL   = '#e76f51'      # muted terracotta
HI_EDGE     = '#333333'
TEXT        = '#1a1a1a'
SUBTEXT     = '#555555'
INSET_CA    = '#f4a261'      # CA highlighted in inset

# ── 3. Main figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 9), facecolor=BG)   # 7in wide = common journal column

# --- Main map axis ---
ax = fig.add_axes([0.08, 0.10, 0.82, 0.72])
ax.set_facecolor('#f7f7f7')    # very subtle off-white map background
for sp in ax.spines.values():
    sp.set_linewidth(0.8)
    sp.set_color('#aaaaaa')

# Draw counties
others.plot(ax=ax, color=OTHER_FILL, edgecolor=OTHER_EDGE, linewidth=0.4, zorder=2)
fresno.plot(ax=ax, color=FRESNO_FILL, edgecolor=HI_EDGE,   linewidth=0.8, zorder=4)
kern.plot(  ax=ax, color=KERN_FILL,   edgecolor=HI_EDGE,   linewidth=0.8, zorder=4)

# State outline
california.dissolve().plot(ax=ax, color='none', edgecolor='#333333', linewidth=1.2, zorder=5)

ax.set_xticks([])
ax.set_yticks([])

# ── 4. Labels ─────────────────────────────────────────────────────────────────
label_cfg = [
    (fresno, 'Fresno\nCounty',  (0, 0)),
    (kern,   'Kern\nCounty',    (0, 0)),
]
for county, label, offset in label_cfg:
    cx = county.geometry.centroid.x.values[0] + offset[0]
    cy = county.geometry.centroid.y.values[0] + offset[1]
    ax.text(cx, cy, label,
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            fontfamily='sans-serif', color='#1a1a1a',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
            zorder=10)

# ── 5. Scale bar ──────────────────────────────────────────────────────────────
# 100 km scale bar in EPSG:3310 (meters)
bounds = california.total_bounds
sb_x0 = bounds[0] + (bounds[2] - bounds[0]) * 0.05
sb_y0 = bounds[1] + (bounds[3] - bounds[1]) * 0.04
bar_len = 100_000   # 100 km in meters

ax.plot([sb_x0, sb_x0 + bar_len], [sb_y0, sb_y0],
        color='#333333', linewidth=2.5, solid_capstyle='butt', zorder=8)
# tick ends
for x in [sb_x0, sb_x0 + bar_len]:
    ax.plot([x, x], [sb_y0 - 6000, sb_y0 + 6000], color='#333333', linewidth=1.5, zorder=8)

ax.text(sb_x0 + bar_len / 2, sb_y0 - 18000, '100 km',
        ha='center', va='top', fontsize=7, fontfamily='sans-serif', color='#333333', zorder=8)
ax.text(sb_x0, sb_y0 - 18000, '0',
        ha='center', va='top', fontsize=7, fontfamily='sans-serif', color='#333333', zorder=8)


## ── 7. Inset map (CA in USA context) ─────────────────────────────────────────
#ax_inset = fig.add_axes([0.68, 0.68, 0.24, 0.16])   # top-right corner
#ax_inset.set_facecolor('#eaf1fb')   # light blue = ocean
#for sp in ax_inset.spines.values():
    #sp.set_linewidth(0.5)
    #sp.set_color('#aaaaaa')

## Clip to contiguous US extent
#us_clip = us_states.cx[-130:-60, 22:52]
#us_clip.plot(ax=ax_inset, color='#cccccc', edgecolor='white', linewidth=0.3)
#ca_state.plot(ax=ax_inset, color='#e76f51', edgecolor='#333333', linewidth=0.5)

#ax_inset.set_xlim(-130, -60)
#ax_inset.set_ylim(22, 52)
#ax_inset.set_xticks([])
#ax_inset.set_yticks([])
#ax_inset.set_title('Location', fontsize=6, fontfamily='sans-serif',
                   #color='#555555', pad=2)

# ── 8. Legend ─────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=FRESNO_FILL, edgecolor=HI_EDGE, linewidth=0.8,
                   label='Fresno County'),
    mpatches.Patch(facecolor=KERN_FILL,   edgecolor=HI_EDGE, linewidth=0.8,
                   label='Kern County'),
    mpatches.Patch(facecolor=OTHER_FILL,  edgecolor='#aaaaaa', linewidth=0.4,
                   label='Other counties'),
]
legend = ax.legend(handles=legend_handles, loc='lower right',
                   fontsize=7.5, frameon=True, framealpha=0.95,
                   edgecolor='#aaaaaa', fancybox=False,
                   title='Counties under study', title_fontsize=8,
                   borderpad=0.7, labelspacing=0.45)
legend.get_frame().set_linewidth(0.6)

## thin top rule under title
#fig.add_artist(mpatches.FancyArrowPatch(
    #(0.08, 0.898), (0.92, 0.898), transform=fig.transFigure,
    #arrowstyle='-', color='#cccccc', linewidth=0.8, zorder=20))

# ── 10. Save ──────────────────────────────────────────────────────────────────
#print("Saved: valley_fever_map_academic.png")
plt.show()