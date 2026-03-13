"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 3: Feature Engineering for ML Models
============================================
- Engineer bigram-level, key-level, and context features
- Test statistical distribution fits per bigram
- Build participant profiles for classification
"""

import pandas as pd
import numpy as np
import os
import json
from scipy import stats

os.makedirs('outputs/texts', exist_ok=True)

log_entries = []
def log(msg):
    log_entries.append(msg)
    print(msg)

def save_log():
    with open('decision_log.txt', 'a') as f:
        f.write('\n' + '='*80 + '\n')
        f.write('PHASE 3: FEATURE ENGINEERING\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')

# ─── LOAD DATA ───────────────────────────────────────────
log('[LOAD] Loading cleaned dataset...')
df = pd.read_csv('outputs/texts/free_text_cleaned.csv')
log(f'[LOAD] Shape: {df.shape}')

# Focus on typing-relevant keys
typing_keys = set(list('abcdefghijklmnopqrstuvwxyz') + ['Space', 'Backspace', '.', ',', 'Shift', 'Enter'])
df_typing = df[df['key1'].isin(typing_keys) & df['key2'].isin(typing_keys)].copy()
log(f'[FILTER] Typing-relevant subset: {df_typing.shape}')

# ─── 1. BIGRAM-LEVEL FEATURES ────────────────────────────
log('\n[FEAT] Computing bigram-level statistics...')

bigram_features = df_typing.groupby(['key1', 'key2']).agg(
    count=('DD.key1.key2', 'count'),
    # Hold time stats
    hold_mean=('DU.key1.key1', 'mean'),
    hold_median=('DU.key1.key1', 'median'),
    hold_std=('DU.key1.key1', 'std'),
    hold_iqr=('DU.key1.key1', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    hold_skew=('DU.key1.key1', 'skew'),
    # DD flight stats
    dd_mean=('DD.key1.key2', 'mean'),
    dd_median=('DD.key1.key2', 'median'),
    dd_std=('DD.key1.key2', 'std'),
    dd_iqr=('DD.key1.key2', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    dd_skew=('DD.key1.key2', 'skew'),
    dd_q10=('DD.key1.key2', lambda x: x.quantile(0.10)),
    dd_q90=('DD.key1.key2', lambda x: x.quantile(0.90)),
    # UD flight stats
    ud_mean=('UD.key1.key2', 'mean'),
    ud_median=('UD.key1.key2', 'median'),
    ud_std=('UD.key1.key2', 'std'),
    # Overlap ratio
    overlap_ratio=('UD.key1.key2', lambda x: (x < 0).mean()),
).reset_index()

bigram_features = bigram_features[bigram_features['count'] >= 10]  # minimum 10 samples
log(f'[FEAT] Bigram features computed: {len(bigram_features)} bigrams with ≥10 samples')

bigram_features.to_csv('outputs/texts/bigram_features.csv', index=False)
log('[SAVE] Saved: outputs/texts/bigram_features.csv')

# ─── 2. KEY-LEVEL HOLD TIME FEATURES ─────────────────────
log('\n[FEAT] Computing per-key hold time features...')

key_hold = df_typing.groupby('key1').agg(
    count=('DU.key1.key1', 'count'),
    hold_mean=('DU.key1.key1', 'mean'),
    hold_median=('DU.key1.key1', 'median'),
    hold_std=('DU.key1.key1', 'std'),
    hold_q25=('DU.key1.key1', lambda x: x.quantile(0.25)),
    hold_q75=('DU.key1.key1', lambda x: x.quantile(0.75)),
).reset_index()

key_hold.to_csv('outputs/texts/key_hold_features.csv', index=False)
log(f'[FEAT] Per-key hold features: {len(key_hold)} keys')

# ─── 3. CONTEXT FEATURES ─────────────────────────────────
log('\n[FEAT] Computing context-dependent features...')

# Word position analysis: track where in a word each keystroke is
# A "word" is a sequence between Space keys
df_typing = df_typing.copy()
df_typing['at_word_start'] = df_typing['key1'] == 'Space'
df_typing['at_word_end'] = df_typing['key2'] == 'Space'
df_typing['mid_word'] = ~df_typing['at_word_start'] & ~df_typing['at_word_end']

# Context timing: how does position in word affect timing?
context_stats = {}
for context, label in [('at_word_start', 'Word Start'), ('at_word_end', 'Word End'), ('mid_word', 'Mid Word')]:
    subset = df_typing[df_typing[context]]
    context_stats[label] = {
        'count': int(len(subset)),
        'dd_mean': float(subset['DD.key1.key2'].mean()),
        'dd_median': float(subset['DD.key1.key2'].median()),
        'dd_std': float(subset['DD.key1.key2'].std()),
        'hold_mean': float(subset['DU.key1.key1'].mean()),
        'hold_median': float(subset['DU.key1.key1'].median()),
    }
    log(f'  {label}: count={len(subset)}, dd_median={context_stats[label]["dd_median"]:.4f}s')

# ─── 4. PARTICIPANT PROFILE FEATURES ─────────────────────
log('\n[FEAT] Computing participant-level profile features...')

participant_features = df_typing.groupby('participant').agg(
    total_keystrokes=('DD.key1.key2', 'count'),
    # Speed metrics
    wpm_estimate=('DD.key1.key2', lambda x: 12.0 / x.median()),  # rough WPM (avg word = 5 chars)
    # Hold patterns
    hold_mean=('DU.key1.key1', 'mean'),
    hold_std=('DU.key1.key1', 'std'),
    hold_cv=('DU.key1.key1', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    # Flight patterns 
    dd_mean=('DD.key1.key2', 'mean'),
    dd_std=('DD.key1.key2', 'std'),
    dd_cv=('DD.key1.key2', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    dd_skew=('DD.key1.key2', 'skew'),
    dd_kurtosis=('DD.key1.key2', 'kurt'),
    # UD patterns
    ud_mean=('UD.key1.key2', 'mean'),
    ud_std=('UD.key1.key2', 'std'),
    overlap_ratio=('UD.key1.key2', lambda x: (x < 0).mean()),
    # Consistency
    dd_iqr=('DD.key1.key2', lambda x: x.quantile(0.75) - x.quantile(0.25)),
).reset_index()

participant_features.to_csv('outputs/texts/participant_features.csv', index=False)
log(f'[FEAT] Participant features: {len(participant_features)} profiles')

# Print WPM range
log(f'  Estimated WPM range: {participant_features["wpm_estimate"].min():.0f} - {participant_features["wpm_estimate"].max():.0f}')
log(f'  Median WPM: {participant_features["wpm_estimate"].median():.0f}')

# ─── 5. SEQUENCE FEATURES (for ML classification) ────────
log('\n[FEAT] Computing windowed sequence features for ML...')

# For each participant-session, create sliding window features
window_size = 20  # 20 keystrokes per window
windows = []

for (participant, session), group in df_typing.groupby(['participant', 'session']):
    group = group.reset_index(drop=True)
    n_windows = len(group) // window_size
    
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window = group.iloc[start:end]
        
        feat = {
            'participant': participant,
            'session': session,
            'window_id': w,
            # Timing features
            'dd_mean': window['DD.key1.key2'].mean(),
            'dd_std': window['DD.key1.key2'].std(),
            'dd_median': window['DD.key1.key2'].median(),
            'dd_iqr': window['DD.key1.key2'].quantile(0.75) - window['DD.key1.key2'].quantile(0.25),
            'dd_min': window['DD.key1.key2'].min(),
            'dd_max': window['DD.key1.key2'].max(),
            'dd_range': window['DD.key1.key2'].max() - window['DD.key1.key2'].min(),
            'dd_skew': window['DD.key1.key2'].skew(),
            # Hold time features
            'hold_mean': window['DU.key1.key1'].mean(),
            'hold_std': window['DU.key1.key1'].std(),
            'hold_median': window['DU.key1.key1'].median(),
            # UD features
            'ud_mean': window['UD.key1.key2'].mean(),
            'ud_std': window['UD.key1.key2'].std(),
            'ud_neg_ratio': (window['UD.key1.key2'] < 0).mean(),
            # UU features
            'uu_mean': window['UU.key1.key2'].mean(),
            'uu_std': window['UU.key1.key2'].std(),
            # Ratios and derived
            'hold_to_dd_ratio': window['DU.key1.key1'].mean() / max(window['DD.key1.key2'].mean(), 0.001),
            'cv_dd': window['DD.key1.key2'].std() / max(window['DD.key1.key2'].mean(), 0.001),
            'cv_hold': window['DU.key1.key1'].std() / max(window['DU.key1.key1'].mean(), 0.001),
        }
        windows.append(feat)

window_df = pd.DataFrame(windows)
window_df.to_csv('outputs/texts/window_features.csv', index=False)
log(f'[FEAT] Windowed features: {len(window_df)} windows of {window_size} keystrokes each')
log(f'  Feature columns: {len(window_df.columns) - 3} (excluding ID columns)')

# ─── 6. SAVE CONTEXT ANALYSIS ────────────────────────────
with open('outputs/texts/context_features.json', 'w') as f:
    json.dump(context_stats, f, indent=2)

log('\n[DONE] Phase 3 complete. All feature sets ready for modeling.')
save_log()
print('\n[LOG] Updated decision_log.txt')
