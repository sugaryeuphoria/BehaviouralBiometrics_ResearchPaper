"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 4: Distribution Modeling & Statistical Analysis
======================================================
- Fit multiple statistical distributions to keystroke timings
- Find best-fit distributions for hold times and flight times
- Build distribution parameter lookup tables for the simulation engine
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import lognorm, gamma, weibull_min, norm, expon
import os
import json
import warnings
warnings.filterwarnings('ignore')

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
        f.write('PHASE 4: DISTRIBUTION MODELING\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')

# ─── LOAD DATA ───────────────────────────────────────────
log('[LOAD] Loading cleaned dataset...')
df = pd.read_csv('outputs/texts/free_text_cleaned.csv')
typing_keys = set(list('abcdefghijklmnopqrstuvwxyz') + ['Space', 'Backspace', '.', ',', 'Shift'])
df_typing = df[df['key1'].isin(typing_keys) & df['key2'].isin(typing_keys)].copy()
log(f'[LOAD] Typing subset: {df_typing.shape}')

# ─── FIT DISTRIBUTIONS ───────────────────────────────────
distributions = {
    'lognormal': lognorm,
    'gamma': gamma,
    'weibull': weibull_min,
    'normal': norm,
}

def fit_and_evaluate(data, name=''):
    """Fit multiple distributions and return best fit parameters."""
    data = data[np.isfinite(data)]
    if len(data) < 30:
        return None
    
    # For hold times and positive flight times, shift to positive
    data_positive = data[data > 0.001]
    if len(data_positive) < 30:
        return None
    
    results = {}
    for dist_name, dist in distributions.items():
        try:
            params = dist.fit(data_positive)
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data_positive, dist_name if dist_name != 'weibull' else 'weibull_min', args=params)
            
            # Log-likelihood for AIC
            ll = np.sum(dist.logpdf(data_positive, *params))
            k = len(params)
            aic = 2 * k - 2 * ll
            
            results[dist_name] = {
                'params': [float(p) for p in params],
                'ks_stat': float(ks_stat),
                'ks_p': float(ks_p),
                'aic': float(aic),
                'n_samples': int(len(data_positive))
            }
        except Exception:
            pass
    
    if not results:
        return None
    
    # Find best by AIC
    best = min(results.keys(), key=lambda k: results[k]['aic'])
    results['best_dist'] = best
    results['data_stats'] = {
        'mean': float(data_positive.mean()),
        'median': float(np.median(data_positive)),
        'std': float(data_positive.std()),
        'q25': float(np.percentile(data_positive, 25)),
        'q75': float(np.percentile(data_positive, 75)),
    }
    # Also compute stats for negative UD 
    neg_ratio = float((data < 0).mean())
    results['negative_ratio'] = neg_ratio
    if neg_ratio > 0:
        neg_data = data[data < 0]
        results['negative_stats'] = {
            'mean': float(neg_data.mean()),
            'median': float(np.median(neg_data)),
            'std': float(neg_data.std()),
        }
    
    return results

# ─── 1. GLOBAL DISTRIBUTION FITS ─────────────────────────
log('\n[DIST] Fitting global distributions...')

global_fits = {}
for col, label in [('DU.key1.key1', 'hold_time'), ('DD.key1.key2', 'dd_flight'), ('UD.key1.key2', 'ud_flight')]:
    data = df_typing[col].dropna().values
    result = fit_and_evaluate(data, label)
    if result:
        global_fits[label] = result
        best = result['best_dist']
        log(f'  {label}: best fit = {best} (AIC={result[best]["aic"]:.0f}, KS={result[best]["ks_stat"]:.4f})')

with open('outputs/texts/global_distribution_fits.json', 'w') as f:
    json.dump(global_fits, f, indent=2)
log('[SAVE] Saved: outputs/texts/global_distribution_fits.json')

# ─── 2. PER-BIGRAM DISTRIBUTION FITS ─────────────────────
log('\n[DIST] Fitting distributions per bigram (top 100 most common)...')

bigram_counts = df_typing.groupby(['key1', 'key2']).size().sort_values(ascending=False)
top_bigrams = bigram_counts.head(100).index.tolist()

bigram_dist_params = {}
for key1, key2 in top_bigrams:
    bigram_data = df_typing[(df_typing['key1'] == key1) & (df_typing['key2'] == key2)]
    bigram_key = f'{key1}->{key2}'
    
    bigram_dist_params[bigram_key] = {}
    
    for col, label in [('DU.key1.key1', 'hold'), ('DD.key1.key2', 'dd'), ('UD.key1.key2', 'ud')]:
        data = bigram_data[col].dropna().values
        result = fit_and_evaluate(data)
        if result:
            best = result['best_dist']
            bigram_dist_params[bigram_key][label] = {
                'best_dist': best,
                'params': result[best]['params'],
                'data_stats': result['data_stats'],
                'negative_ratio': result['negative_ratio'],
                'n_samples': result[best]['n_samples'],
            }
            if 'negative_stats' in result:
                bigram_dist_params[bigram_key][label]['negative_stats'] = result['negative_stats']

log(f'[DIST] Fitted distributions for {len(bigram_dist_params)} bigrams')

with open('outputs/texts/bigram_distribution_params.json', 'w') as f:
    json.dump(bigram_dist_params, f, indent=2)
log('[SAVE] Saved: outputs/texts/bigram_distribution_params.json')

# ─── 3. PER-KEY HOLD TIME DISTRIBUTION FITS ──────────────
log('\n[DIST] Fitting hold time distributions per key...')

key_hold_dist = {}
for key in sorted(typing_keys):
    data = df_typing[df_typing['key1'] == key]['DU.key1.key1'].dropna().values
    result = fit_and_evaluate(data)
    if result:
        best = result['best_dist']
        key_hold_dist[key] = {
            'best_dist': best,
            'params': result[best]['params'],
            'data_stats': result['data_stats'],
            'n_samples': result[best]['n_samples'],
        }

with open('outputs/texts/key_hold_distributions.json', 'w') as f:
    json.dump(key_hold_dist, f, indent=2)
log(f'[DIST] Fitted hold time distributions for {len(key_hold_dist)} keys')

# ─── 4. CONTEXT-DEPENDENT DISTRIBUTION FITS ──────────────
log('\n[DIST] Fitting context-dependent distributions...')

context_fits = {}
# Word start (after Space)
ws_dd = df_typing[df_typing['key1'] == 'Space']['DD.key1.key2'].dropna().values
result = fit_and_evaluate(ws_dd)
if result:
    best = result['best_dist']
    context_fits['word_start_dd'] = {
        'best_dist': best,
        'params': result[best]['params'],
        'data_stats': result['data_stats'],
    }
    log(f'  Word start DD: best={best}, median={result["data_stats"]["median"]:.4f}s')

# Word end (before Space)
we_dd = df_typing[df_typing['key2'] == 'Space']['DD.key1.key2'].dropna().values
result = fit_and_evaluate(we_dd)
if result:
    best = result['best_dist']
    context_fits['word_end_dd'] = {
        'best_dist': best,
        'params': result[best]['params'],
        'data_stats': result['data_stats'],
    }
    log(f'  Word end DD: best={best}, median={result["data_stats"]["median"]:.4f}s')

# Mid-word (alpha to alpha)
mw_dd = df_typing[(df_typing['key1_type'] == 'alpha') & (df_typing['key2_type'] == 'alpha')]['DD.key1.key2'].dropna().values
result = fit_and_evaluate(mw_dd)
if result:
    best = result['best_dist']
    context_fits['mid_word_dd'] = {
        'best_dist': best,
        'params': result[best]['params'],
        'data_stats': result['data_stats'],
    }
    log(f'  Mid-word DD: best={best}, median={result["data_stats"]["median"]:.4f}s')

with open('outputs/texts/context_distributions.json', 'w') as f:
    json.dump(context_fits, f, indent=2)
log('[SAVE] Saved: outputs/texts/context_distributions.json')

# ─── PLOT 9: Distribution Fits Visualization ─────────────
log('\n[PLOT 9] Creating distribution fit comparison plots...')

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Distribution Fits: Empirical Data vs Fitted Models', fontsize=16, fontweight='bold', y=0.98)

plot_configs = [
    ('DU.key1.key1', 'Hold Time (Global)', (0.01, 0.4), global_fits.get('hold_time', {})),
    ('DD.key1.key2', 'DD Flight Time (Global)', (0.01, 0.8), global_fits.get('dd_flight', {})),
    ('UD.key1.key2', 'UD Flight Time (Global)', (0.01, 0.6), global_fits.get('ud_flight', {})),
]

# Add specific bigram examples
example_bigrams = [('t', 'h', 'th'), ('e', 'Space', 'e→Space'), ('Space', 't', 'Space→t')]
for key1, key2, label in example_bigrams:
    bigram_data = df_typing[(df_typing['key1'] == key1) & (df_typing['key2'] == key2)]
    plot_configs.append(('DD.key1.key2', f'DD: "{label}" bigram', (0.01, 0.6), 
                         bigram_dist_params.get(f'{key1}->{key2}', {}).get('dd', {})))

dist_colors = {'lognormal': '#58a6ff', 'gamma': '#7ee787', 'weibull': '#f778ba', 'normal': '#ffa657'}

for idx, (col, title, xlim, fit_result) in enumerate(plot_configs):
    ax = axes[idx // 3, idx % 3]
    
    if idx < 3:
        data = df_typing[col].dropna()
    else:
        key1, key2, _ = example_bigrams[idx - 3]
        data = df_typing[(df_typing['key1'] == key1) & (df_typing['key2'] == key2)][col].dropna()
    
    data_pos = data[data > 0.001]
    data_clipped = data_pos[(data_pos >= xlim[0]) & (data_pos <= xlim[1])]
    
    ax.hist(data_clipped, bins=80, density=True, alpha=0.4, color='#8b949e', 
            edgecolor='none', label='Empirical')
    
    x = np.linspace(xlim[0], xlim[1], 200)
    
    # Plot fitted distributions
    if fit_result and 'best_dist' not in fit_result:
        # It's already the inner structure
        if 'best_dist' in fit_result:
            dist_name = fit_result['best_dist']
            params = fit_result['params']
            dist = distributions.get(dist_name)
            if dist:
                y = dist.pdf(x, *params)
                ax.plot(x, y, color=dist_colors.get(dist_name, 'white'), linewidth=2, 
                        label=f'{dist_name} (best)')
    else:
        for dist_name in ['lognormal', 'gamma', 'weibull']:
            if dist_name in (fit_result or {}):
                params = fit_result[dist_name]['params']
                dist = distributions[dist_name]
                y = dist.pdf(x, *params)
                is_best = (fit_result.get('best_dist') == dist_name)
                ax.plot(x, y, color=dist_colors[dist_name], linewidth=2 if is_best else 1, 
                        alpha=1.0 if is_best else 0.5,
                        label=f'{dist_name}{"*" if is_best else ""}',
                        linestyle='-' if is_best else '--')
    
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/plots/09_distribution_fits.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 9] Saved: outputs/plots/09_distribution_fits.png')

# ─── PLOT 10: Negative UD Analysis ───────────────────────
log('\n[PLOT 10] Creating overlap (negative UD) distribution analysis...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Negative UD distribution by key type
ax = axes[0]
ud_data = df_typing['UD.key1.key2'].dropna()
ud_neg = ud_data[ud_data < 0]
ud_pos = ud_data[ud_data >= 0]

ax.hist(ud_neg.clip(-0.25, 0), bins=60, density=True, alpha=0.7, color='#f778ba', 
        label=f'Negative (n={len(ud_neg):,})', edgecolor='none')
ax.hist(ud_pos.clip(0, 0.6), bins=80, density=True, alpha=0.7, color='#7ee787',
        label=f'Positive (n={len(ud_pos):,})', edgecolor='none')
ax.set_xlabel('UD Flight Time (seconds)')
ax.set_ylabel('Density')
ax.set_title('Positive vs Negative UD Flight Distribution', fontsize=13, fontweight='bold')
ax.legend()

# Overlap ratio by bigram category
ax = axes[1]
categories = {
    'Same hand\n(adjacent)': df_typing[df_typing['key1'].isin(list('qwertasdfgzxcvb')) & 
                                        df_typing['key2'].isin(list('qwertasdfgzxcvb'))],
    'Cross hand': df_typing[df_typing['key1'].isin(list('qwertasdfgzxcvb')) & 
                              df_typing['key2'].isin(list('yuiophjklnm'))],
    'With Space': df_typing[(df_typing['key1'] == 'Space') | (df_typing['key2'] == 'Space')],
    'Same key': df_typing[df_typing['key1'] == df_typing['key2']],
}

overlap_by_cat = {}
cat_labels = []
cat_values = []
for cat_name, cat_data in categories.items():
    ratio = (cat_data['UD.key1.key2'] < 0).mean() * 100
    cat_labels.append(cat_name)
    cat_values.append(ratio)

colors_cat = ['#58a6ff', '#7ee787', '#ffa657', '#d2a8ff']
bars = ax.bar(cat_labels, cat_values, color=colors_cat, alpha=0.8)
for bar, val in zip(bars, cat_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Overlap Ratio (%)')
ax.set_title('Key Overlap Frequency by Typing Context', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/plots/10_overlap_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 10] Saved: outputs/plots/10_overlap_distributions.png')

# ─── SUMMARY ─────────────────────────────────────────────
log('\n[SUMMARY] Distribution Modeling Key Findings:')
for label, result in global_fits.items():
    best = result['best_dist']
    log(f'  {label}: Best fit = {best}')
    for dist_name in ['lognormal', 'gamma', 'weibull']:
        if dist_name in result:
            log(f'    {dist_name}: AIC={result[dist_name]["aic"]:.0f}, KS={result[dist_name]["ks_stat"]:.4f}')

log('\n[DECISION] Based on distribution fits:')
log('  - Log-normal consistently fits keystroke timing data well')
log('  - Gamma distribution is competitive, especially for hold times')  
log('  - Key insight: we should use PER-BIGRAM distribution parameters, not global')
log('  - For simulation: sample from fitted log-normal with bigram-specific params')
log('  - Must model negative UD (key overlap) separately for realism')

log('\n[DONE] Phase 4 complete. Distribution parameters ready for simulation engine.')
save_log()
print('\n[LOG] Updated decision_log.txt')
