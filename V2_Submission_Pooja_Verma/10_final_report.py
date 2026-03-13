"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 10: Final Report & Paper-Ready Outputs
==============================================
- Generate final summary statistics
- Create paper-ready comparison table
- Update decision log with conclusions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
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

print('='*70)
print('PHASE 10: FINAL REPORT & PAPER-READY OUTPUTS')
print('='*70)

# ─── Load all results ────────────────────────────────────
with open('outputs/texts/phase1_summary.json', 'r') as f:
    phase1 = json.load(f)

with open('outputs/texts/phase2_eda_summary.json', 'r') as f:
    phase2 = json.load(f)

with open('outputs/texts/phase5_model_results.json', 'r') as f:
    phase5 = json.load(f)

with open('outputs/texts/phase8_evaluation_results.json', 'r') as f:
    phase8 = json.load(f)

with open('outputs/texts/phase6_cluster_summary.json', 'r') as f:
    phase6 = json.load(f)

# ─── Final Summary Table ─────────────────────────────────
print('\n[TABLE] Generating final summary tables...')

# Model performance table
model_table = []
for name, results in phase5.items():
    model_table.append({
        'Model': name,
        'Accuracy': f'{results["accuracy"]:.4f}',
        'F1 Score': f'{results["f1_score"]:.4f}',
        'AUC-ROC': f'{results["auc_roc"]:.4f}',
        'CV F1 (mean±std)': f'{results["cv_f1_mean"]:.4f}±{results["cv_f1_std"]:.4f}',
    })
model_df = pd.DataFrame(model_table)
model_df.to_csv('outputs/texts/final_model_comparison.csv', index=False)
print(model_df.to_string(index=False))

# Evaluation results table
print('\n\nSimulation Evaluation Results:')
eval_table = []
for name, results in phase8.items():
    eval_table.append({
        'Model': name,
        'Our Sim → Human (%)': f'{results["simulation"]["classified_as_human"]*100:.1f}%',
        'Our Sim Prob': f'{results["simulation"]["avg_human_probability"]:.3f}',
        'Naive → Human (%)': f'{results["naive_synthetic"]["classified_as_human"]*100:.1f}%',
        'Naive Prob': f'{results["naive_synthetic"]["avg_human_probability"]:.3f}',
    })
eval_df = pd.DataFrame(eval_table)
eval_df.to_csv('outputs/texts/final_evaluation_comparison.csv', index=False)
print(eval_df.to_string(index=False))

# ─── PLOT 16: Final Research Summary ─────────────────────
print('\n[PLOT 16] Creating final research summary visualization...')

fig = plt.figure(figsize=(20, 14))
fig.suptitle('Human Keystroke Simulation Research — Key Results', 
             fontsize=20, fontweight='bold', y=0.98, color='white')

gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Data overview (pie)
ax = fig.add_subplot(gs[0, 0])
key_types = phase2.get('key_type_distribution', {})
top_types = sorted(key_types.items(), key=lambda x: x[1], reverse=True)[:5]
labels = [t[0] for t in top_types]
sizes = [t[1] for t in top_types]
colors = ['#58a6ff', '#7ee787', '#f778ba', '#ffa657', '#d2a8ff']
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize': 9})
for text in texts:
    text.set_color('#c9d1d9')
for text in autotexts:
    text.set_color('white')
    text.set_fontsize(8)
ax.set_title('Key Type Distribution', fontsize=12, fontweight='bold')

# 2. Model accuracy bars
ax = fig.add_subplot(gs[0, 1])
model_names = list(phase5.keys())
accs = [phase5[m]['f1_score'] for m in model_names]
colors_m = ['#58a6ff', '#7ee787', '#f778ba']
bars = ax.bar([n.replace(' ', '\n') for n in model_names], accs, color=colors_m[:len(model_names)], alpha=0.8)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.4f}', 
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_title('ML Model Performance\n(Human vs Synthetic)', fontsize=12, fontweight='bold')
ax.set_ylim(0.99, 1.002)

# 3. Simulation vs Naive comparison
ax = fig.add_subplot(gs[0, 2])
sim_rates = [phase8[m]['simulation']['classified_as_human']*100 for m in phase8.keys()]
naive_rates = [phase8[m]['naive_synthetic']['classified_as_human']*100 for m in phase8.keys()]
x = np.arange(len(list(phase8.keys())))
width = 0.35
ax.bar(x - width/2, sim_rates, width, label='Our Simulation', color='#7ee787', alpha=0.8)
ax.bar(x + width/2, naive_rates, width, label='Naive Synthetic', color='#f778ba', alpha=0.8)
ax.axhline(50, color='#ffa657', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in phase8.keys()], fontsize=8)
ax.set_ylabel('% Classified as Human')
ax.set_title('Simulation Evaluation\nvs Naive Baseline', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# 4. Timing distribution (DD)
ax = fig.add_subplot(gs[1, 0:2])
df = pd.read_csv('outputs/texts/free_text_cleaned.csv')
typing_keys = set(list('abcdefghijklmnopqrstuvwxyz') + ['Space', '.', ','])
df_typing = df[df['key1'].isin(typing_keys) & df['key2'].isin(typing_keys)]
dd_data = df_typing['DD.key1.key2'].clip(0, 0.8)
hold_data = df_typing['DU.key1.key1'].clip(0, 0.3)

ax.hist(dd_data, bins=100, density=True, alpha=0.6, color='#58a6ff', label='DD Flight Time', edgecolor='none')
ax.hist(hold_data, bins=80, density=True, alpha=0.6, color='#f778ba', label='Hold Time', edgecolor='none')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Density')
ax.set_title('Overall Keystroke Timing Distributions (559K samples)', fontsize=12, fontweight='bold')
ax.legend()

# 5. Participant clustering
ax = fig.add_subplot(gs[1, 2])
cluster_sizes = phase6.get('cluster_sizes', {})
archetypes = phase6.get('archetypes', {})
c_labels = [archetypes.get(str(k), f'C{k}').split('(')[0].strip() for k in sorted(cluster_sizes.keys())]
c_sizes = [cluster_sizes[k] for k in sorted(cluster_sizes.keys())]
c_colors = ['#58a6ff', '#7ee787', '#f778ba', '#ffa657']
ax.bar(range(len(c_labels)), c_sizes, color=c_colors[:len(c_labels)], alpha=0.8)
ax.set_xticks(range(len(c_labels)))
ax.set_xticklabels(c_labels, fontsize=8, rotation=15, ha='right')
ax.set_ylabel('Participants')
ax.set_title('Typing Archetypes\n(K-Means Clusters)', fontsize=12, fontweight='bold')

# 6. Key findings text
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
findings = [
    "1. Dataset: 559,485 cleaned keystrokes from 99 participants across 2 sessions, covering 2,005 unique key bigrams",
    "2. Typing Speed Range: 39–143 WPM (estimated), with median 71 WPM across participants",
    "3. Distribution Modeling: Weibull distributions best fit keystroke timing data at per-bigram level",
    "4. ML Models: Random Forest (F1=0.9996), Gradient Boosting (F1=0.9999), AdaBoost (F1=0.9989) — near-perfect detection of naive synthetic data",
    "5. Simulation Success: Our engine achieves 99.5–100% classification as human across all models — indistinguishable from real typing",
    "6. Key Insight: Natural variability from fitted distributions is critical — fixed/random delays are instantly detectable",
    "7. Overlap Typing: 17.2% of keystrokes show rollover typing (negative UD), essential for realistic simulation",
    "8. Context Effects: Word-start transitions are ~52% slower than mid-word transitions — key for natural rhythm",
]

text = '\n'.join(findings)
ax.text(0.02, 0.95, 'KEY RESEARCH FINDINGS', fontsize=14, fontweight='bold',
        transform=ax.transAxes, va='top', color='#58a6ff')
ax.text(0.02, 0.82, text, fontsize=10, transform=ax.transAxes, va='top', color='#c9d1d9',
        linespacing=1.6, fontfamily='sans-serif')

plt.savefig('outputs/plots/16_research_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('[PLOT 16] Saved: outputs/plots/16_research_summary.png')

# ─── Final Decision Log ──────────────────────────────────
print('\n[LOG] Writing final decision log...')

with open('decision_log.txt', 'a') as f:
    f.write('\n' + '='*80 + '\n')
    f.write('PHASE 10: FINAL CONCLUSIONS\n')
    f.write('='*80 + '\n')
    f.write(f'\nDataset: {phase1["original_rows"]:,} raw rows → {phase1["after_cleaning"]:,} clean rows\n')
    f.write(f'Participants: {phase1["participants"]}\n')
    f.write(f'Unique Key Bigrams: {phase1["unique_key_pairs"]:,}\n\n')
    
    f.write('RESEARCH QUESTION: Can we make LLM text appear human-typed?\n')
    f.write('ANSWER: YES — with distribution-based keystroke simulation.\n\n')
    
    f.write('METHOD SUMMARY:\n')
    f.write('1. Analyzed 559K keystrokes from 99 real participants\n')
    f.write('2. Fitted Weibull/lognormal/gamma distributions per key bigram\n')
    f.write('3. Built context-aware simulation (word boundaries, thinking pauses, fatigue drift)\n')
    f.write('4. Trained ML classifiers achieving 99.99% accuracy on naive synthetic\n')
    f.write('5. Our simulation fools ALL classifiers — classified as human 99.5-100% of the time\n\n')
    
    f.write('KEY FINDINGS:\n')
    for finding in findings:
        f.write(f'  {finding}\n')
    
    f.write('\nPROJECT OUTPUTS:\n')
    f.write('  - 16 research-quality visualizations in outputs/plots/\n')
    f.write('  - ML models (RF, GBM, AdaBoost) saved in outputs/models/\n')
    f.write('  - Distribution parameters for 100+ bigrams in outputs/texts/\n')
    f.write('  - Interactive web interface at localhost:5001\n')
    f.write('  - Simulation engine in 07_simulation_engine.py\n')
    f.write('  - Complete decision log for research paper\n')

# ─── List all outputs ────────────────────────────────────
print('\n' + '='*70)
print('COMPLETE OUTPUT INVENTORY')
print('='*70)

# Plots
print('\n📊 PLOTS (outputs/plots/):')
for f in sorted(os.listdir('outputs/plots')):
    if f.endswith('.png'):
        size = os.path.getsize(f'outputs/plots/{f}')
        print(f'  {f} ({size/1024:.0f} KB)')

# Text outputs
print('\n📝 TEXT OUTPUTS (outputs/texts/):')
for f in sorted(os.listdir('outputs/texts')):
    size = os.path.getsize(f'outputs/texts/{f}')
    print(f'  {f} ({size/1024:.0f} KB)')

# Models
print('\n🤖 MODELS (outputs/models/):')
for f in sorted(os.listdir('outputs/models')):
    size = os.path.getsize(f'outputs/models/{f}')
    print(f'  {f} ({size/1024:.0f} KB)')

# Python scripts
print('\n🐍 PYTHON SCRIPTS:')
for f in sorted(os.listdir('.')):
    if f.endswith('.py') and not f.startswith('.'):
        print(f'  {f}')

print(f'\n✅ decision_log.txt (running log)')
print(f'\n🌐 Web Interface: 09_web_interface/')
print(f'   Run: python3 09_web_interface/app.py')
print(f'   Open: http://localhost:5001')

print('\n' + '='*70)
print('RESEARCH PROJECT COMPLETE')
print('='*70)
