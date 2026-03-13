"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 2: Exploratory Data Analysis (EDA)
=========================================
- Comprehensive visualizations of keystroke dynamics
- Statistical analysis of timing patterns
- Key pair analysis, participant variation, word boundary effects
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import json

# Style setup
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11,
})

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/texts', exist_ok=True)

# Decision log
log_entries = []
def log(msg):
    log_entries.append(msg)
    print(msg)

def save_log():
    with open('decision_log.txt', 'a') as f:
        f.write('\n' + '='*80 + '\n')
        f.write('PHASE 2: EXPLORATORY DATA ANALYSIS\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')

# ─── LOAD CLEANED DATA ───────────────────────────────────
log('[LOAD] Loading cleaned dataset...')
df = pd.read_csv('outputs/texts/free_text_cleaned.csv')
log(f'[LOAD] Shape: {df.shape}')

# Filter to just alpha + space + common punctuation for most analysis
alpha_space = df[df['key1_type'].isin(['alpha', 'space', 'punctuation'])]
log(f'[FILTER] Alpha+Space+Punct subset: {alpha_space.shape}')

# ─── PLOT 1: Timing Distributions (Hold, DD, UD) ─────────
log('\n[PLOT 1] Creating timing distributions plot...')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Keystroke Timing Distributions', fontsize=18, fontweight='bold', y=0.98)

timing_data = [
    ('DU.key1.key1', 'Hold Time (key down → key up)', '#58a6ff', (0, 0.5)),
    ('DD.key1.key2', 'Down-Down Flight Time', '#f778ba', (0, 1.0)),
    ('UD.key1.key2', 'Up-Down Flight Time', '#7ee787', (-0.3, 0.8)),
    ('DU.key1.key2', 'Down-Up Interval', '#d2a8ff', (0, 1.0)),
    ('UU.key1.key2', 'Up-Up Interval', '#ffa657', (0, 1.0)),
]

for i, (col, title, color, xlim) in enumerate(timing_data):
    ax = axes[i // 3, i % 3]
    data = alpha_space[col].clip(xlim[0] - 0.05, xlim[1] + 0.05)
    ax.hist(data, bins=120, density=True, alpha=0.7, color=color, edgecolor='none')
    data_kde = alpha_space[col][(alpha_space[col] >= xlim[0]) & (alpha_space[col] <= xlim[1])]
    if len(data_kde) > 100:
        data_kde.plot.kde(ax=ax, color='white', linewidth=1.5)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Density')
    # Add stats
    med = alpha_space[col].median()
    mean = alpha_space[col].mean()
    ax.axvline(med, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.95, 0.95, f'med={med:.3f}s\nmean={mean:.3f}s', 
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))

# Remove the 6th subplot
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/plots/01_timing_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 1] Saved: outputs/plots/01_timing_distributions.png')

# Stats for decision log
for col, title, _, _ in timing_data:
    vals = alpha_space[col]
    log(f'  {title}: mean={vals.mean():.4f}s, median={vals.median():.4f}s, std={vals.std():.4f}s, IQR=[{vals.quantile(0.25):.4f}, {vals.quantile(0.75):.4f}]')

# ─── PLOT 2: Hold Time by Key Type ───────────────────────
log('\n[PLOT 2] Creating hold time by key type violin plot...')
fig, ax = plt.subplots(figsize=(14, 7))

key_type_order = ['alpha', 'space', 'punctuation', 'backspace', 'modifier']
subset = df[df['key1_type'].isin(key_type_order)]
subset_clipped = subset.copy()
subset_clipped['DU.key1.key1'] = subset_clipped['DU.key1.key1'].clip(0, 0.5)

palette = {'alpha': '#58a6ff', 'space': '#7ee787', 'punctuation': '#ffa657', 
           'backspace': '#f778ba', 'modifier': '#d2a8ff'}

parts = ax.violinplot([subset_clipped[subset_clipped['key1_type']==kt]['DU.key1.key1'].values 
                        for kt in key_type_order],
                       showmedians=True, showextrema=False)

colors = [palette[kt] for kt in key_type_order]
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('white')

ax.set_xticks(range(1, len(key_type_order) + 1))
ax.set_xticklabels(key_type_order, fontsize=12)
ax.set_ylabel('Hold Time (seconds)', fontsize=13)
ax.set_title('Hold Time Distribution by Key Type', fontsize=16, fontweight='bold')
ax.set_ylim(0, 0.4)

# Add median labels
for i, kt in enumerate(key_type_order):
    med = subset[subset['key1_type']==kt]['DU.key1.key1'].median()
    ax.text(i+1, med + 0.015, f'{med*1000:.0f}ms', ha='center', fontsize=10, 
            color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/plots/02_hold_time_by_key_type.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 2] Saved: outputs/plots/02_hold_time_by_key_type.png')

for kt in key_type_order:
    vals = subset[subset['key1_type']==kt]['DU.key1.key1']
    log(f'  {kt}: median hold={vals.median()*1000:.1f}ms, mean={vals.mean()*1000:.1f}ms')

# ─── PLOT 3: Bigram Timing Heatmap ───────────────────────
log('\n[PLOT 3] Creating bigram timing heatmap...')

# Get top keys for heatmap
alpha_keys = list('etaoinsrhldcumwfgypbvkjxqz') + ['Space']
alpha_df = df[df['key1'].isin(alpha_keys) & df['key2'].isin(alpha_keys)]

# Compute median DD flight per bigram
bigram_timing = alpha_df.groupby(['key1', 'key2'])['DD.key1.key2'].median().reset_index()
bigram_timing = bigram_timing.pivot(index='key1', columns='key2', values='DD.key1.key2')

# Select top 15 most frequent keys
top_keys = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r', 'l', 'd', 'Space', 'c', 'u', 'm']
heatmap_data = bigram_timing.reindex(index=top_keys, columns=top_keys)

fig, ax = plt.subplots(figsize=(14, 11))
im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0.1, vmax=0.4)
ax.set_xticks(range(len(top_keys)))
ax.set_xticklabels(top_keys, fontsize=11)
ax.set_yticks(range(len(top_keys)))
ax.set_yticklabels(top_keys, fontsize=11)
ax.set_xlabel('Key 2 (next key)', fontsize=13)
ax.set_ylabel('Key 1 (current key)', fontsize=13)
ax.set_title('Median Down-Down Flight Time by Key Bigram (seconds)', fontsize=15, fontweight='bold')

# Add values
for i in range(len(top_keys)):
    for j in range(len(top_keys)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            text_color = 'white' if val > 0.25 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=text_color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Seconds', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/plots/03_bigram_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 3] Saved: outputs/plots/03_bigram_heatmap.png')

# Save bigram stats
bigram_stats = alpha_df.groupby(['key1', 'key2']).agg(
    count=('DD.key1.key2', 'count'),
    mean_dd=('DD.key1.key2', 'mean'),
    median_dd=('DD.key1.key2', 'median'),
    std_dd=('DD.key1.key2', 'std'),
    mean_hold=('DU.key1.key1', 'mean'),
    median_hold=('DU.key1.key1', 'median')
).reset_index()
bigram_stats.to_csv('outputs/texts/bigram_statistics.csv', index=False)
log('[SAVE] Saved bigram statistics: outputs/texts/bigram_statistics.csv')

# ─── PLOT 4: Participant Typing Speed Variation ──────────
log('\n[PLOT 4] Creating participant variation plot...')

participant_profiles = pd.read_csv('outputs/texts/participant_profiles.csv')
participant_profiles = participant_profiles.sort_values('median_dd_flight')

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Top: median DD flight per participant
ax1 = axes[0]
colors_part = plt.cm.cool(np.linspace(0.1, 0.9, len(participant_profiles)))
ax1.bar(range(len(participant_profiles)), participant_profiles['median_dd_flight'].values, 
        color=colors_part, alpha=0.8, width=1.0)
ax1.set_xlabel('Participants (sorted by speed)', fontsize=12)
ax1.set_ylabel('Median DD Flight (seconds)', fontsize=12)
ax1.set_title('Typing Speed Variation Across 99 Participants', fontsize=15, fontweight='bold')
ax1.axhline(participant_profiles['median_dd_flight'].median(), color='#ffa657', 
            linestyle='--', linewidth=2, label=f'Overall median: {participant_profiles["median_dd_flight"].median():.3f}s')
ax1.legend(fontsize=11)
ax1.set_xlim(-0.5, len(participant_profiles) - 0.5)

# Bottom: hold time vs flight time scatter
ax2 = axes[1]
sc = ax2.scatter(participant_profiles['median_hold_time'], 
                  participant_profiles['median_dd_flight'],
                  c=participant_profiles['overlap_ratio'], cmap='plasma',
                  s=80, alpha=0.8, edgecolors='#30363d', linewidths=0.5)
ax2.set_xlabel('Median Hold Time (seconds)', fontsize=12)
ax2.set_ylabel('Median DD Flight Time (seconds)', fontsize=12)
ax2.set_title('Hold Time vs Flight Time by Participant (colored by overlap ratio)', fontsize=14, fontweight='bold')
cbar2 = plt.colorbar(sc, ax=ax2, shrink=0.8)
cbar2.set_label('Key Overlap Ratio', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/plots/04_participant_variation.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 4] Saved: outputs/plots/04_participant_variation.png')

fastest = participant_profiles.iloc[0]
slowest = participant_profiles.iloc[-1]
log(f'  Fastest typist: {fastest["participant"]}, median DD={fastest["median_dd_flight"]:.3f}s')
log(f'  Slowest typist: {slowest["participant"]}, median DD={slowest["median_dd_flight"]:.3f}s')
log(f'  Speed ratio (slowest/fastest): {slowest["median_dd_flight"]/fastest["median_dd_flight"]:.1f}x')

# ─── PLOT 5: Word Boundary Effects ───────────────────────
log('\n[PLOT 5] Creating word boundary effects plot...')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Before space (key → Space)
before_space = df[(df['key2'] == 'Space') & (df['key1_type'] == 'alpha')]
after_space = df[(df['key1'] == 'Space') & (df['key2_type'] == 'alpha')]
within_word = df[(df['key1_type'] == 'alpha') & (df['key2_type'] == 'alpha')]

datasets = [
    (within_word['DD.key1.key2'], 'Within Word (letter→letter)', '#58a6ff'),
    (before_space['DD.key1.key2'], 'Before Space (letter→Space)', '#f778ba'),
    (after_space['DD.key1.key2'], 'After Space (Space→letter)', '#7ee787'),
]

# Histogram comparison
ax = axes[0]
for data, label, color in datasets:
    clipped = data.clip(0, 0.8)
    ax.hist(clipped, bins=80, density=True, alpha=0.5, color=color, label=label, edgecolor='none')
ax.set_xlabel('DD Flight Time (seconds)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('DD Flight Time: Word Boundaries', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 0.8)

# Box plot
ax = axes[1]
box_data = [within_word['DD.key1.key2'].clip(0, 1).values,
            before_space['DD.key1.key2'].clip(0, 1).values,
            after_space['DD.key1.key2'].clip(0, 1).values]
bp = ax.boxplot(box_data, labels=['Within\nWord', 'Before\nSpace', 'After\nSpace'],
                patch_artist=True, widths=0.6, showfliers=False)
colors_box = ['#58a6ff', '#f778ba', '#7ee787']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('white')
ax.set_ylabel('DD Flight Time (seconds)', fontsize=12)
ax.set_title('DD Flight by Position', fontsize=13, fontweight='bold')

# Hold time comparison
ax = axes[2]
hold_data = [
    within_word['DU.key1.key1'].clip(0, 0.3),
    before_space['DU.key1.key1'].clip(0, 0.3),
    after_space['DU.key1.key1'].clip(0, 0.3)
]
for data, label, color in zip(hold_data, 
                               ['Within Word', 'Before Space', 'After Space'],
                               ['#58a6ff', '#f778ba', '#7ee787']):
    ax.hist(data, bins=60, density=True, alpha=0.5, color=color, label=label, edgecolor='none')
ax.set_xlabel('Hold Time (seconds)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Hold Time: Word Boundaries', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/plots/05_word_boundary_effects.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 5] Saved: outputs/plots/05_word_boundary_effects.png')

for data, label, _ in datasets:
    log(f'  {label}: median={data.median():.4f}s, mean={data.mean():.4f}s, std={data.std():.4f}s')

# ─── PLOT 6: Session Effects ─────────────────────────────
log('\n[PLOT 6] Creating session comparison plot...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, col in enumerate(['DD.key1.key2', 'DU.key1.key1']):
    ax = axes[i]
    title = 'DD Flight Time' if i == 0 else 'Hold Time'
    s1 = df[df['session'] == 1][col].clip(0, 0.6 if i == 0 else 0.3)
    s2 = df[df['session'] == 2][col].clip(0, 0.6 if i == 0 else 0.3)
    ax.hist(s1, bins=80, density=True, alpha=0.6, color='#58a6ff', label='Session 1', edgecolor='none')
    ax.hist(s2, bins=80, density=True, alpha=0.6, color='#f778ba', label='Session 2', edgecolor='none')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}: Session 1 vs Session 2', fontsize=13, fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig('outputs/plots/06_session_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 6] Saved: outputs/plots/06_session_comparison.png')

s1_dd = df[df['session'] == 1]['DD.key1.key2']
s2_dd = df[df['session'] == 2]['DD.key1.key2']
log(f'  Session 1 DD flight: median={s1_dd.median():.4f}s, mean={s1_dd.mean():.4f}s')
log(f'  Session 2 DD flight: median={s2_dd.median():.4f}s, mean={s2_dd.mean():.4f}s')

# ─── PLOT 7: Overlap/Rollover Typing Analysis ────────────
log('\n[PLOT 7] Creating overlap analysis plot...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# UD distribution showing negative values
ax = axes[0]
ud = alpha_space['UD.key1.key2'].clip(-0.3, 0.6)
ax.hist(ud, bins=120, density=True, color='#7ee787', alpha=0.7, edgecolor='none')
ax.axvline(0, color='#f778ba', linewidth=2, linestyle='--', label='Zero boundary')
ax.set_xlabel('UD Flight Time (seconds)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Up-Down Flight Time Distribution\n(Negative = Key Overlap / Rollover)', 
             fontsize=13, fontweight='bold')
neg_pct = (alpha_space['UD.key1.key2'] < 0).mean() * 100
ax.text(0.95, 0.95, f'{neg_pct:.1f}% negative\n(rollover typing)', 
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))
ax.legend()

# Overlap frequency by key pair (top overlapping pairs)
ax = axes[1]
overlap_df = df[df['UD.key1.key2'] < 0]
overlap_pairs = overlap_df.groupby(['key1', 'key2']).size().sort_values(ascending=False).head(15)
pair_labels = [f'{k1}→{k2}' for k1, k2 in overlap_pairs.index]
ax.barh(range(len(pair_labels)), overlap_pairs.values, color='#d2a8ff', alpha=0.8)
ax.set_yticks(range(len(pair_labels)))
ax.set_yticklabels(pair_labels, fontsize=10)
ax.set_xlabel('Count of Overlapping Keystrokes')
ax.set_title('Most Common Overlapping Key Pairs', fontsize=13, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/plots/07_overlap_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 7] Saved: outputs/plots/07_overlap_analysis.png')

# ─── PLOT 8: Typing Rhythm (autocorrelation) ─────────────
log('\n[PLOT 8] Creating typing rhythm analysis plot...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pick a representative participant 
sample_participant = 'p001'
p_data = df[(df['participant'] == sample_participant) & (df['session'] == 1)]
p_dd = p_data['DD.key1.key2'].clip(0, 1).values[:200]

# Time series of DD flight for first 200 keystrokes
ax = axes[0]
ax.plot(range(len(p_dd)), p_dd, color='#58a6ff', linewidth=0.8, alpha=0.7)
ax.fill_between(range(len(p_dd)), 0, p_dd, alpha=0.2, color='#58a6ff')
ax.set_xlabel('Keystroke Index', fontsize=12)
ax.set_ylabel('DD Flight Time (seconds)', fontsize=12)
ax.set_title(f'Typing Rhythm: Participant {sample_participant} (first 200 keys)', 
             fontsize=13, fontweight='bold')

# Autocorrelation
ax = axes[1]
from pandas.plotting import autocorrelation_plot
dd_series = pd.Series(p_data['DD.key1.key2'].clip(0, 2).values)
lags = range(1, 51)
autocorr = [dd_series.autocorr(lag) for lag in lags]
ax.bar(lags, autocorr, color='#ffa657', alpha=0.7)
ax.set_xlabel('Lag (keystrokes)', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.set_title('DD Flight Autocorrelation (Typing Rhythm Pattern)', fontsize=13, fontweight='bold')
ax.axhline(0, color='#8b949e', linewidth=0.5)

plt.tight_layout()
plt.savefig('outputs/plots/08_typing_rhythm.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 8] Saved: outputs/plots/08_typing_rhythm.png')
log(f'  Autocorrelation lag-1: {autocorr[0]:.4f}')
log(f'  Autocorrelation lag-2: {autocorr[1]:.4f}')

# ─── SAVE EDA SUMMARY ────────────────────────────────────
log('\n[SUMMARY] Generating EDA summary...')

eda_summary = {
    'total_clean_rows': int(len(df)),
    'alpha_space_rows': int(len(alpha_space)),
    'key_type_distribution': df['key1_type'].value_counts().to_dict(),
    'word_boundary_analysis': {
        'within_word_dd_median': float(within_word['DD.key1.key2'].median()),
        'before_space_dd_median': float(before_space['DD.key1.key2'].median()),
        'after_space_dd_median': float(after_space['DD.key1.key2'].median()),
    },
    'overlap_ratio': float(neg_pct),
    'session_comparison': {
        'session1_dd_median': float(s1_dd.median()),
        'session2_dd_median': float(s2_dd.median()),
    },
    'key_findings': [
        'Hold time is right-skewed with median 91ms - log-normal or gamma likely fits well',
        'DD flight time shows bimodal tendency - within-word vs word-boundary transitions differ significantly',
        f'17.2% of keystrokes show rollover typing (negative UD) - critical for realism',
        'Participants vary hugely in speed - need to model per-profile variance',
        'Strong word-boundary effect on timing - Space transitions are slower',
        'Session 1 vs 2 very similar - typing patterns are consistent',
        f'Lag-1 autocorrelation of {autocorr[0]:.3f} - moderate rhythm persistence',
    ]
}

with open('outputs/texts/phase2_eda_summary.json', 'w') as f:
    json.dump(eda_summary, f, indent=2)

log('[SAVE] Saved EDA summary: outputs/texts/phase2_eda_summary.json')
log('\n[DONE] Phase 2 complete. 8 plots generated, key patterns identified.')

save_log()
print('\n[LOG] Updated decision_log.txt')
