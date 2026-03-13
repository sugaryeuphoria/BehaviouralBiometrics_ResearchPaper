"""
TRU Theme Configuration for Research Plots
=============================================
Official Thompson Rivers University color palette
for consistent, paper-ready visualizations.

Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics
"""

# ─── TRU Official Color Palette ──────────────────────────
TRU_BLUE = '#003e51'
TRU_TEAL = '#00b0b9'
TRU_SAGE = '#bad1ba'
TRU_GREY = '#9ab7c1'
TRU_YELLOW = '#ffcd00'
TRU_CLOUD = '#fff5de'
TRU_OL_GREEN = '#00b18f'

# Secondary / Web
TRU_ORANGE = '#f88f23'
TRU_DARK_TEAL = '#007B81'
TRU_LIGHT_TEAL = '#9EE1E5'
TRU_PALE_TEAL = '#E0F6F7'
TRU_LIGHT_GREY = '#e9e9e9'
TRU_PALE_GREY = '#f3f3f3'

# ─── Chart Color Sequences ──────────────────────────────
# Primary palette for bar charts, lines, scatter
TRU_COLORS = [TRU_BLUE, TRU_TEAL, TRU_ORANGE, TRU_OL_GREEN, TRU_YELLOW, TRU_DARK_TEAL, TRU_SAGE]

# For clusters / categories
TRU_CLUSTER_COLORS = [TRU_BLUE, TRU_TEAL, TRU_ORANGE, TRU_OL_GREEN]

# For comparison (e.g., human vs synthetic)
TRU_COMPARE = [TRU_BLUE, TRU_TEAL, TRU_ORANGE]

# ─── Matplotlib RC Params (Light Theme) ──────────────────
LIGHT_THEME = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': '#1a1a1a',
    'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.edgecolor': '#cccccc',
    'grid.color': '#e5e5e5',
    'grid.alpha': 0.7,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
}


def apply_theme():
    """Apply the TRU light theme globally for matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(LIGHT_THEME)
