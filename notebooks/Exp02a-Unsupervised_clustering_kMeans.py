#%%
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from sklearn.cluster import KMeans

DAS_APL_NAS_SERVER_PATH = "Z:/DAS-APL"
sys.path.append(DAS_APL_NAS_SERVER_PATH)

import das4whales as dw
from multispectralRepresentation import spectral_decomposition, normalize_band, crop

#%%
# =============================================================================
# CONFIGURATION
# =============================================================================

FILEPATH = 'Z:/DAS-APL/data/OOI_RCA2021/Optasence/NorthCable/North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'

# Frequency bands [Hz]
B1 = [40, 60]   # Blue
B2 = [30, 40]   # Green
B3 = [16, 28]   # Red

# Crop window
T_MIN, T_MAX = 0, 60
D_MIN, D_MAX = 20e3, 60e3

PERC           = 90
CHANNEL_STRIDE = 3

# KMeans
N_CLUSTERS  = 4
KMEANS_SEED = 42

# --- Cluster names (one per cluster, ordered by ascending Red intensity) ---
CLUSTER_NAMES = [
    'D',
    'C',
    'B',
    'A',
]

# --- One colour per cluster ---
CLUSTER_COLORS = ["#ffffff", "#1e1e1e", "#0717f7", "#EE0808"]

# Font sizes
LABEL_SIZE  = 18
TICK_SIZE   = 16
TITLE_SIZE  = 17
LEGEND_SIZE = 15
CBAR_SIZE   = 15

# Output paths (set to None to skip)
OUT_PDF = "Z:/DAS-APL/exp/multispectral_representation/Exp01-X3_kmeans.pdf"
OUT_PNG = "Z:/DAS-APL/exp/multispectral_representation/Exp01-X3_kmeans.png"

#%%
# =============================================================================
# HELPERS
# =============================================================================

def file_title(filepath):
    name  = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{6}Z)', name)
    if match:
        dt = datetime.strptime(match.group(1), '%Y-%m-%dT%H%M%SZ')
        return dt.strftime('%Y-%m-%d  %H:%M:%S UTC')
    return name

#%%
# =============================================================================
# 1. LOAD DATA
# =============================================================================

metadata = dw.data_handle.get_acquisition_parameters(FILEPATH, interrogator='optasense')
fs = metadata['fs']
dx = metadata['dx']
nx = metadata['nx']

selected_channels_m = [0, nx * dx, dx * CHANNEL_STRIDE]
selected_channels   = [int(x // dx) for x in selected_channels_m]

print(f"Loading 1 of every {CHANNEL_STRIDE} channels → {len(range(*selected_channels))} channels")
tr, time, dist, _ = dw.data_handle.load_das_data(FILEPATH, selected_channels, metadata)
print(f"Loaded data shape: {tr.shape}")

#%%
# =============================================================================
# 2. SPECTRAL DECOMPOSITION + CROP
# =============================================================================

tr_b01, tr_b02, tr_b03 = spectral_decomposition(tr, fs, [B1, B2, B3])

tr_b01_c, dist_c, time_c = crop(tr_b01, dist, time, D_MIN, D_MAX, T_MIN, T_MAX)
tr_b02_c, _,      _      = crop(tr_b02, dist, time, D_MIN, D_MAX, T_MIN, T_MAX)
tr_b03_c, _,      _      = crop(tr_b03, dist, time, D_MIN, D_MAX, T_MIN, T_MAX)

r = normalize_band(tr_b03_c, perc=PERC)
g = normalize_band(tr_b02_c, perc=PERC)
b = normalize_band(tr_b01_c, perc=PERC)

rgb = np.stack([r, g, b], axis=-1)
print(f"RGB image shape: {rgb.shape}")

#%%
# =============================================================================
# 3. K-MEANS SEGMENTATION
# =============================================================================

n_dist, n_time, _ = rgb.shape
X = rgb.reshape(-1, 3)

print(f"Fitting KMeans with {N_CLUSTERS} clusters on {X.shape[0]} pixels …")
km     = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_SEED, n_init='auto')
labels = km.fit_predict(X)

# Sort clusters by ascending Red-channel centroid value
center_red = km.cluster_centers_[:, 0]
order  = np.argsort(center_red)
remap  = np.empty_like(order)
remap[order] = np.arange(N_CLUSTERS)
labels = remap[labels]

mask         = labels.reshape(n_dist, n_time)
pixel_counts = np.bincount(labels, minlength=N_CLUSTERS)

#%%
# =============================================================================
# 4. PLOT
# =============================================================================

cmap_seg = mcolors.ListedColormap(CLUSTER_COLORS[:N_CLUSTERS])
extent   = [time_c[0], time_c[-1], dist_c[0] / 1e3, dist_c[-1] / 1e3]

# Layout: [RGB image | segmentation mask | barplot]
# Use gridspec for tight control over spacing
fig = plt.figure(figsize=(18, 7))
gs  = fig.add_gridspec(
    1, 3,
    width_ratios=[1, 1, 0.35],   # barplot narrower
    wspace=0.08,                  # tighter gap between panels
)

ax_rgb = fig.add_subplot(gs[0])
ax_seg = fig.add_subplot(gs[1], sharey=ax_rgb)
ax_bar = fig.add_subplot(gs[2])

# ── LEFT: RGB multispectral ──────────────────────────────────────────────────
ax_rgb.imshow(rgb, aspect='auto', origin='lower', extent=extent)
ax_rgb.set_title(f'{file_title(FILEPATH)}', fontsize=TITLE_SIZE, pad=8)
ax_rgb.set_xlabel('Time [s]', fontsize=LABEL_SIZE)
ax_rgb.set_ylabel('Distance [km]', fontsize=LABEL_SIZE)
ax_rgb.tick_params(axis='both', labelsize=TICK_SIZE)

rgb_legend = [
    Patch(color='red',   label=f'Red:   {B3[0]}–{B3[1]} Hz'),
    Patch(color='green', label=f'Green: {B2[0]}–{B2[1]} Hz'),
    Patch(color='blue',  label=f'Blue:  {B1[0]}–{B1[1]} Hz'),
]
ax_rgb.legend(handles=rgb_legend, loc='lower right', framealpha=0.7, fontsize=LEGEND_SIZE)

# ── MIDDLE: segmentation mask ────────────────────────────────────────────────
ax_seg.imshow(mask, aspect='auto', origin='lower', extent=extent,
              cmap=cmap_seg, vmin=-0.5, vmax=N_CLUSTERS - 0.5,
              interpolation='nearest')
# No title on the mask panel
ax_seg.set_xlabel('Time [s]', fontsize=LABEL_SIZE)
ax_seg.tick_params(axis='both', labelsize=TICK_SIZE)
plt.setp(ax_seg.get_yticklabels(), visible=False)   # shared y → hide duplicate labels

# Discrete colorbar acting as a stepped colour scale
norm_seg = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, N_CLUSTERS), ncolors=N_CLUSTERS)
sm       = plt.cm.ScalarMappable(cmap=cmap_seg, norm=norm_seg)
sm.set_array([])

divider = make_axes_locatable(ax_seg)
cax     = divider.append_axes("right", size="5%", pad=0.06)
cbar    = fig.colorbar(sm, cax=cax, ticks=[])
#cbar.ax.set_yticklabels(CLUSTER_NAMES[:N_CLUSTERS], fontsize=CBAR_SIZE)
# Draw lines between colour steps
cbar.ax.hlines(np.arange(0.5, N_CLUSTERS - 0.5), 0, 1,
               colors='white', linewidths=1.5, transform=cbar.ax.transData)

# ── RIGHT: barplot pixel counts ───────────────────────────────────────────────
bar_heights = pixel_counts / pixel_counts.sum() * 100   # percentage

# Horizontal bars, bottom cluster at bottom (matches image orientation)
y_pos = np.arange(N_CLUSTERS)
bars  = ax_bar.barh(
    y_pos, bar_heights,
    color=CLUSTER_COLORS[:N_CLUSTERS],
    edgecolor='black', linewidth=0.8,
    height=0.6
)

# Add percentage labels inside / outside bars
for bar, pct in zip(bars, bar_heights):
    x_label = bar.get_width() + 0.5
    ax_bar.text(x_label, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', ha='left',
                fontsize=LEGEND_SIZE - 1, color='black')

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(CLUSTER_NAMES[:N_CLUSTERS], fontsize=LEGEND_SIZE)
ax_bar.set_xlabel('Pixels [%]', fontsize=LABEL_SIZE)
ax_bar.tick_params(axis='x', labelsize=TICK_SIZE)
ax_bar.set_xlim(0, bar_heights.max() * 1.25)
ax_bar.spines[['top', 'right']].set_visible(False)

# Tight layout respecting gridspec wspace
fig.tight_layout(pad=1.0)

#%%
# =============================================================================
# 5. SAVE
# =============================================================================

if OUT_PDF:
    fig.savefig(OUT_PDF, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved PDF: {OUT_PDF}")

if OUT_PNG:
    fig.savefig(OUT_PNG, format="png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved PNG: {OUT_PNG}")

plt.show()
# %%