"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 8: Evaluation — Testing Simulation Against ML Models
============================================================
- Generate simulated keystroke sequences
- Extract window features from simulated data
- Pass through trained classifiers
- Compare detection rates for naive vs our simulation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json
import sys
import os

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

# Import simulation engine
sys.path.insert(0, '.')
from importlib.machinery import SourceFileLoader
sim_module = SourceFileLoader("sim", "07_simulation_engine.py").load_module()
HumanKeystrokeSimulator = sim_module.HumanKeystrokeSimulator

log_entries = []
def log(msg):
    log_entries.append(msg)
    print(msg)

def save_log():
    with open('decision_log.txt', 'a') as f:
        f.write('\n' + '='*80 + '\n')
        f.write('PHASE 8: EVALUATION\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')

# ─── LOAD MODELS ─────────────────────────────────────────
log('[LOAD] Loading trained models...')
rf_model = joblib.load('outputs/models/random_forest_model.joblib')
gb_model = joblib.load('outputs/models/gradient_boosting_model.joblib')
ada_model = joblib.load('outputs/models/adaboost_model.joblib')
scaler = joblib.load('outputs/models/scaler.joblib')

with open('outputs/models/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

models = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'AdaBoost': ada_model,
}

log(f'[LOAD] Loaded 3 models and scaler. Features: {len(feature_cols)}')

# ─── GENERATE TEST TEXTS ─────────────────────────────────
log('\n[GEN] Generating test texts...')

test_texts = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
    "In the beginning, there was nothing. Then, there was everything. The universe expanded rapidly in all directions.",
    "The weather today is beautiful. Clear skies and warm temperatures make it perfect for a walk outside.",
    "Programming requires patience and attention to detail. Every semicolon and bracket matters in writing code.",
    "Research in natural language processing has advanced significantly with the development of large language models.",
    "Education is the most powerful weapon which you can use to change the world. Knowledge opens doors.",
    "The integration of technology in everyday life has transformed how we communicate, work, and learn.",
    "Data science combines statistics, programming, and domain knowledge to extract insights from large datasets.",
    "The human brain processes information through complex networks of neurons that fire in specific patterns.",
    "Climate change represents one of the most significant challenges facing humanity in the twenty first century.",
    "Software engineering principles help teams build reliable and maintainable systems at scale.",
    "The art of writing well involves choosing the right words and arranging them in a clear and compelling way.",
    "Cybersecurity is becoming increasingly important as more of our personal and professional lives move online.",
    "Music has the unique ability to evoke emotions and bring people together across cultures and generations.",
]

log(f'[GEN] Created {len(test_texts)} test texts')

# ─── EXTRACT FEATURES FROM SIMULATED DATA ────────────────
def extract_window_features(keystrokes, window_size=20):
    """Extract the same window features used for training."""
    windows = []
    n_windows = len(keystrokes) // window_size
    
    for w in range(max(1, n_windows)):
        start = w * window_size
        end = min(start + window_size, len(keystrokes))
        window = keystrokes[start:end]
        
        if len(window) < 5:
            continue
        
        dd_times = [k['dd_time'] for k in window[1:]]
        hold_times = [k['hold_time'] for k in window]
        
        if len(dd_times) == 0:
            continue
        
        # Compute UD approximation (dd - hold of previous key)
        ud_times = []
        for j in range(1, len(window)):
            ud = window[j]['dd_time'] - window[j-1]['hold_time']
            ud_times.append(ud)
        
        # Compute UU approximation
        uu_times = []
        for j in range(1, len(window)):
            uu = window[j]['dd_time'] + window[j]['hold_time'] - window[j-1]['hold_time']
            uu_times.append(uu)
        
        dd_arr = np.array(dd_times)
        hold_arr = np.array(hold_times)
        ud_arr = np.array(ud_times) if ud_times else np.array([0])
        uu_arr = np.array(uu_times) if uu_times else np.array([0])
        
        feat = {
            'dd_mean': float(dd_arr.mean()),
            'dd_std': float(dd_arr.std()) if len(dd_arr) > 1 else 0.0,
            'dd_median': float(np.median(dd_arr)),
            'dd_iqr': float(np.percentile(dd_arr, 75) - np.percentile(dd_arr, 25)) if len(dd_arr) >= 4 else 0.0,
            'dd_min': float(dd_arr.min()),
            'dd_max': float(dd_arr.max()),
            'dd_range': float(dd_arr.max() - dd_arr.min()),
            'dd_skew': float(pd.Series(dd_arr).skew()) if len(dd_arr) > 2 else 0.0,
            'hold_mean': float(hold_arr.mean()),
            'hold_std': float(hold_arr.std()) if len(hold_arr) > 1 else 0.0,
            'hold_median': float(np.median(hold_arr)),
            'ud_mean': float(ud_arr.mean()),
            'ud_std': float(ud_arr.std()) if len(ud_arr) > 1 else 0.0,
            'ud_neg_ratio': float((ud_arr < 0).mean()),
            'uu_mean': float(uu_arr.mean()),
            'uu_std': float(uu_arr.std()) if len(uu_arr) > 1 else 0.0,
            'hold_to_dd_ratio': float(hold_arr.mean() / max(dd_arr.mean(), 0.001)),
            'cv_dd': float(dd_arr.std() / max(dd_arr.mean(), 0.001)) if len(dd_arr) > 1 else 0.0,
            'cv_hold': float(hold_arr.std() / max(hold_arr.mean(), 0.001)) if len(hold_arr) > 1 else 0.0,
        }
        windows.append(feat)
    
    return windows

# ─── GENERATE SIMULATED DATA ─────────────────────────────
log('\n[SIM] Generating simulated keystroke data...')

sim_windows_all = []
for speed in ['slow', 'medium', 'fast']:
    sim = HumanKeystrokeSimulator(speed_profile=speed)
    for text in test_texts:
        keystrokes = sim.simulate(text)
        windows = extract_window_features(keystrokes)
        sim_windows_all.extend(windows)

log(f'[SIM] Generated {len(sim_windows_all)} simulated windows')

# Also generate naive synthetic for comparison
naive_windows = []
for _ in range(len(sim_windows_all)):
    feat = {}
    for col in feature_cols:
        if 'mean' in col or 'median' in col:
            feat[col] = np.random.uniform(0.1, 0.3)
        elif 'std' in col or 'iqr' in col:
            feat[col] = 0.01
        elif 'skew' in col:
            feat[col] = np.random.normal(0, 0.1)
        elif 'neg_ratio' in col:
            feat[col] = 0.0
        elif 'ratio' in col or 'cv' in col:
            feat[col] = np.random.uniform(0.2, 0.6)
        elif 'min' in col:
            feat[col] = np.random.uniform(0.08, 0.15)
        elif 'max' in col:
            feat[col] = np.random.uniform(0.2, 0.4)
        elif 'range' in col:
            feat[col] = np.random.uniform(0.05, 0.15)
        else:
            feat[col] = np.random.uniform(0.1, 0.3)
    naive_windows.append(feat)

log(f'[SIM] Generated {len(naive_windows)} naive synthetic windows for comparison')

# ─── EVALUATE WITH MODELS ────────────────────────────────
log('\n[EVAL] Evaluating simulated data against trained models...')

sim_df = pd.DataFrame(sim_windows_all)
naive_df = pd.DataFrame(naive_windows)

# Ensure column order matches
for col in feature_cols:
    if col not in sim_df.columns:
        sim_df[col] = 0.0
    if col not in naive_df.columns:
        naive_df[col] = 0.0

X_sim = sim_df[feature_cols].values
X_naive = naive_df[feature_cols].values

# Handle NaN/inf
X_sim = np.nan_to_num(X_sim, nan=0.0, posinf=10.0, neginf=-10.0)
X_naive = np.nan_to_num(X_naive, nan=0.0, posinf=10.0, neginf=-10.0)

X_sim_scaled = scaler.transform(X_sim)
X_naive_scaled = scaler.transform(X_naive)

eval_results = {}

for name, model in models.items():
    # Predict: 1 = human, 0 = synthetic
    sim_preds = model.predict(X_sim_scaled)
    sim_probs = model.predict_proba(X_sim_scaled)[:, 1]
    naive_preds = model.predict(X_naive_scaled)
    naive_probs = model.predict_proba(X_naive_scaled)[:, 1]
    
    # Our simulation: what % classified as human?
    sim_human_rate = sim_preds.mean()
    sim_avg_prob = sim_probs.mean()
    
    # Naive: what % classified as human?
    naive_human_rate = naive_preds.mean()
    naive_avg_prob = naive_probs.mean()
    
    eval_results[name] = {
        'simulation': {
            'classified_as_human': float(sim_human_rate),
            'avg_human_probability': float(sim_avg_prob),
        },
        'naive_synthetic': {
            'classified_as_human': float(naive_human_rate),
            'avg_human_probability': float(naive_avg_prob),
        }
    }
    
    log(f'\n  {name}:')
    log(f'    Our simulation → {sim_human_rate*100:.1f}% classified as human (avg prob: {sim_avg_prob:.3f})')
    log(f'    Naive synthetic → {naive_human_rate*100:.1f}% classified as human (avg prob: {naive_avg_prob:.3f})')

# Save results
with open('outputs/texts/phase8_evaluation_results.json', 'w') as f:
    json.dump(eval_results, f, indent=2)
log('\n[SAVE] Saved: outputs/texts/phase8_evaluation_results.json')

# ─── PLOT 14: Evaluation Results ─────────────────────────
log('\n[PLOT 14] Creating evaluation comparison plots...')

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Bar chart: Human classification rate
ax = axes[0]
model_names = list(eval_results.keys())
sim_rates = [eval_results[m]['simulation']['classified_as_human'] * 100 for m in model_names]
naive_rates = [eval_results[m]['naive_synthetic']['classified_as_human'] * 100 for m in model_names]

x = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x - width/2, sim_rates, width, label='Our Simulation', color='#7ee787', alpha=0.8)
bars2 = ax.bar(x + width/2, naive_rates, width, label='Naive Synthetic', color='#f778ba', alpha=0.8)

# Add goal line
ax.axhline(50, color='#ffa657', linestyle='--', linewidth=2, alpha=0.7, label='Goal: 50% (indistinguishable)')

for bar, val in zip(bars1, sim_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
            ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, naive_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
            ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=10)
ax.set_ylabel('% Classified as Human', fontsize=12)
ax.set_title('Classification as Human: Simulation vs Naive', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 110)

# Human probability distribution
ax = axes[1]
for name, model in models.items():
    probs = model.predict_proba(X_sim_scaled)[:, 1]
    color = {'Random Forest': '#58a6ff', 'Gradient Boosting': '#7ee787', 'AdaBoost': '#f778ba'}[name]
    ax.hist(probs, bins=30, density=True, alpha=0.5, color=color, label=name, edgecolor='none')

ax.axvline(0.5, color='#ffa657', linestyle='--', linewidth=2, label='Decision boundary')
ax.set_xlabel('Human Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Simulated Data: Human Probability Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# Feature comparison: simulated vs real
ax = axes[2]
human_df = pd.read_csv('outputs/texts/window_features.csv')
compare_feats = ['dd_mean', 'dd_std', 'hold_mean', 'hold_std', 'ud_neg_ratio']

real_vals = [human_df[f].median() for f in compare_feats]
sim_vals = [sim_df[f].median() for f in compare_feats]

x = np.arange(len(compare_feats))
width = 0.35
ax.bar(x - width/2, real_vals, width, label='Real Human', color='#58a6ff', alpha=0.8)
ax.bar(x + width/2, sim_vals, width, label='Our Simulation', color='#7ee787', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(['DD\nMean', 'DD\nStd', 'Hold\nMean', 'Hold\nStd', 'UD Neg\nRatio'], fontsize=10)
ax.set_ylabel('Median Value')
ax.set_title('Feature Comparison: Real vs Simulated', fontsize=13, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/plots/14_evaluation_results.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 14] Saved: outputs/plots/14_evaluation_results.png')

# ─── PLOT 15: Timing Distribution Comparison ─────────────
log('\n[PLOT 15] Creating timing distribution comparison...')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Simulate a longer text for better comparison
long_text = " ".join(test_texts)
sim = HumanKeystrokeSimulator(speed_profile='medium')
ks = sim.simulate(long_text)
sim_dd = [k['dd_time'] for k in ks[1:]]
sim_hold = [k['hold_time'] for k in ks]

# Real human data
real_df = pd.read_csv('outputs/texts/free_text_cleaned.csv')
typing_keys = set(list('abcdefghijklmnopqrstuvwxyz') + ['Space', '.', ','])
real_typing = real_df[real_df['key1'].isin(typing_keys) & real_df['key2'].isin(typing_keys)]

# DD comparison
ax = axes[0]
ax.hist(real_typing['DD.key1.key2'].clip(0, 0.8), bins=80, density=True, 
        alpha=0.5, color='#58a6ff', label='Real Human', edgecolor='none')
ax.hist(np.clip(sim_dd, 0, 0.8), bins=40, density=True, 
        alpha=0.5, color='#7ee787', label='Simulated', edgecolor='none')
ax.set_xlabel('DD Flight Time (seconds)')
ax.set_ylabel('Density')
ax.set_title('DD Flight Time: Real vs Simulated', fontsize=13, fontweight='bold')
ax.legend()

# Hold time comparison
ax = axes[1]
ax.hist(real_typing['DU.key1.key1'].clip(0, 0.3), bins=80, density=True,
        alpha=0.5, color='#58a6ff', label='Real Human', edgecolor='none')
ax.hist(np.clip(sim_hold, 0, 0.3), bins=40, density=True,
        alpha=0.5, color='#7ee787', label='Simulated', edgecolor='none')
ax.set_xlabel('Hold Time (seconds)')
ax.set_ylabel('Density')
ax.set_title('Hold Time: Real vs Simulated', fontsize=13, fontweight='bold')
ax.legend()

# QQ-like comparison
ax = axes[2]
real_dd_sorted = np.sort(real_typing['DD.key1.key2'].clip(0, 1).sample(min(1000, len(sim_dd))).values)
sim_dd_sorted = np.sort(np.clip(sim_dd[:len(real_dd_sorted)], 0, 1))
# Ensure same length
min_len = min(len(real_dd_sorted), len(sim_dd_sorted))
real_dd_sorted = real_dd_sorted[:min_len]
sim_dd_sorted = sim_dd_sorted[:min_len]

ax.scatter(real_dd_sorted, sim_dd_sorted, alpha=0.3, s=10, color='#d2a8ff')
ax.plot([0, 1], [0, 1], '--', color='#ffa657', linewidth=2, label='Perfect match')
ax.set_xlabel('Real Human DD (seconds)')
ax.set_ylabel('Simulated DD (seconds)')
ax.set_title('Q-Q Plot: DD Flight Times', fontsize=13, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/plots/15_timing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 15] Saved: outputs/plots/15_timing_comparison.png')

# ─── SUMMARY ─────────────────────────────────────────────
log('\n' + '='*60)
log('EVALUATION SUMMARY')
log('='*60)
for name in model_names:
    r = eval_results[name]
    log(f'{name}:')
    log(f'  Simulation: {r["simulation"]["classified_as_human"]*100:.1f}% human (prob: {r["simulation"]["avg_human_probability"]:.3f})')
    log(f'  Naive:      {r["naive_synthetic"]["classified_as_human"]*100:.1f}% human (prob: {r["naive_synthetic"]["avg_human_probability"]:.3f})')

log('\n[DONE] Phase 8 complete. Evaluation results ready.')
save_log()
print('\n[LOG] Updated decision_log.txt')
