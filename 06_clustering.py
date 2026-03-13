"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 6: Participant Clustering & Typing Pattern Analysis
===========================================================
- Cluster participants by typing behavior
- PCA/t-SNE visualization
- Identify typing archetypes
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import json

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.size': 11,
})

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/texts', exist_ok=True)

log_entries = []
def log(msg):
    log_entries.append(msg)
    print(msg)

def save_log():
    with open('decision_log.txt', 'a') as f:
        f.write('\n' + '='*80 + '\n')
        f.write('PHASE 6: PARTICIPANT CLUSTERING\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')

# ─── LOAD DATA ───────────────────────────────────────────
log('[LOAD] Loading participant features...')
pf = pd.read_csv('outputs/texts/participant_features.csv')
log(f'[LOAD] Shape: {pf.shape}')

# ─── PREPARE FEATURES FOR CLUSTERING ─────────────────────
feature_cols = ['hold_mean', 'hold_std', 'hold_cv', 'dd_mean', 'dd_std', 'dd_cv',
                'dd_skew', 'dd_kurtosis', 'ud_mean', 'ud_std', 'overlap_ratio', 'dd_iqr']

X = pf[feature_cols].values
X = np.nan_to_num(X, nan=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── K-MEANS CLUSTERING ──────────────────────────────────
log('\n[CLUSTER] Running K-Means clustering...')

# Elbow method
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    log(f'  K={k}: inertia={km.inertia_:.1f}')

# Choose K=4 based on elbow
best_k = 4
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
pf['cluster'] = km_final.fit_predict(X_scaled)
log(f'\n[CLUSTER] Final K={best_k}')
log(f'  Cluster sizes: {pf["cluster"].value_counts().sort_index().to_dict()}')

# ─── PCA ──────────────────────────────────────────────────
log('\n[PCA] Running PCA...')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
log(f'  Explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}')
log(f'  Total: {sum(pca.explained_variance_ratio_):.3f}')

# ─── t-SNE ───────────────────────────────────────────────
log('\n[TSNE] Running t-SNE...')
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(X_scaled)

# ─── PLOT 13: Clustering Visualization ───────────────────
log('\n[PLOT 13] Creating clustering visualization...')

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Elbow plot
ax = axes[0, 0]
ax.plot(list(K_range), inertias, 'o-', color='#58a6ff', linewidth=2, markersize=8)
ax.axvline(best_k, color='#f778ba', linestyle='--', linewidth=2, label=f'Selected K={best_k}')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
ax.legend()

# PCA scatter
ax = axes[0, 1]
cluster_colors = ['#58a6ff', '#7ee787', '#f778ba', '#ffa657', '#d2a8ff']
for c in range(best_k):
    mask = pf['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=cluster_colors[c], s=80, 
               alpha=0.7, label=f'Cluster {c}', edgecolors='white', linewidths=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
ax.set_title('PCA: Typing Pattern Clusters', fontsize=13, fontweight='bold')
ax.legend()

# t-SNE scatter
ax = axes[1, 0]
for c in range(best_k):
    mask = pf['cluster'] == c
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=cluster_colors[c], s=80, 
               alpha=0.7, label=f'Cluster {c}', edgecolors='white', linewidths=0.5)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE: Typing Pattern Clusters', fontsize=13, fontweight='bold')
ax.legend()

# Cluster profiles (radar-like comparison)
ax = axes[1, 1]
cluster_profiles = pf.groupby('cluster')[['hold_mean', 'dd_mean', 'dd_std', 
                                            'overlap_ratio', 'wpm_estimate']].mean()
x_pos = np.arange(len(cluster_profiles.columns))
width = 0.2

for c in range(best_k):
    vals = cluster_profiles.loc[c].values
    # Normalize for comparison
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    ax.bar(x_pos + c * width, vals_norm, width, color=cluster_colors[c], 
           alpha=0.8, label=f'Cluster {c}')

ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels(['Hold\nTime', 'DD\nFlight', 'DD\nStd', 'Overlap\nRatio', 'WPM'], fontsize=9)
ax.set_ylabel('Normalized Value')
ax.set_title('Cluster Profiles Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/plots/13_clustering_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 13] Saved: outputs/plots/13_clustering_analysis.png')

# ─── CLUSTER ANALYSIS ────────────────────────────────────
log('\n[ANALYSIS] Typing Archetypes:')
archetype_names = {}
for c in range(best_k):
    profile = pf[pf['cluster'] == c]
    wpm = profile['wpm_estimate'].median()
    overlap = profile['overlap_ratio'].median()
    cv = profile['dd_cv'].median()
    
    if wpm > 80 and overlap > 0.15:
        name = 'Speed Typist (fast, rollover)'
    elif wpm > 60 and cv < 1.0:
        name = 'Steady Typist (moderate, consistent)'
    elif wpm < 50:
        name = 'Careful Typist (slow, deliberate)'
    else:
        name = 'Variable Typist (moderate, varied rhythm)'
    
    archetype_names[c] = name
    log(f'  Cluster {c} ({name}):')
    log(f'    Participants: {len(profile)}')
    log(f'    Median WPM: {wpm:.0f}')
    log(f'    Overlap ratio: {overlap:.3f}')
    log(f'    DD CV: {cv:.3f}')
    log(f'    Hold time: {profile["hold_mean"].median()*1000:.1f}ms')

# Save cluster assignments
pf.to_csv('outputs/texts/participant_clusters.csv', index=False)

# Save analysis
cluster_summary = {
    'n_clusters': best_k,
    'archetypes': archetype_names,
    'pca_variance_explained': [float(v) for v in pca.explained_variance_ratio_],
    'cluster_sizes': pf['cluster'].value_counts().sort_index().to_dict(),
}
with open('outputs/texts/phase6_cluster_summary.json', 'w') as f:
    json.dump(cluster_summary, f, indent=2, default=str)

log('\n[DONE] Phase 6 complete. Typing archetypes identified.')
save_log()
print('\n[LOG] Updated decision_log.txt')
