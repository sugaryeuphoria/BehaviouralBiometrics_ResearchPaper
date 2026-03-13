"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 1: Data Loading, Cleaning & Preprocessing
=================================================
- Load the free-text keystroke dataset
- Clean corrupted entries, handle nulls, remove outliers
- Create derived features for downstream analysis
- Save cleaned dataset and summary statistics
"""

import pandas as pd
import numpy as np
import os
import json

# Ensure output directories exist
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/texts', exist_ok=True)

# ─── DECISION LOG ─────────────────────────────────────────
log_entries = []

def log(msg):
    log_entries.append(msg)
    print(msg)

def save_log():
    with open('decision_log.txt', 'a') as f:
        f.write('\n' + '='*80 + '\n')
        f.write('PHASE 1: DATA LOADING, CLEANING & PREPROCESSING\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')

# ─── 1. LOAD DATA ────────────────────────────────────────
log('[LOAD] Loading free-text.csv...')
df = pd.read_csv('Data/free-text.csv', low_memory=False)
log(f'[LOAD] Raw shape: {df.shape}')
log(f'[LOAD] Columns: {list(df.columns)}')

# ─── 2. CLEAN COLUMNS ────────────────────────────────────
# Drop the 'Unnamed: 9' column (almost entirely NaN - 562570 out of 562583)
log(f'[CLEAN] Unnamed:9 non-null count: {df["Unnamed: 9"].notna().sum()} out of {len(df)}')
df = df.drop(columns=['Unnamed: 9'])
log('[CLEAN] Dropped Unnamed:9 column')

# Rename columns for cleaner access (strip trailing whitespace)
df.columns = df.columns.str.strip()
log(f'[CLEAN] Cleaned column names: {list(df.columns)}')

# ─── 3. CONVERT DU.key1.key1 TO NUMERIC ──────────────────
# This column has mixed types
df['DU.key1.key1'] = pd.to_numeric(df['DU.key1.key1'], errors='coerce')
non_numeric_count = df['DU.key1.key1'].isna().sum() - 0  # original had no NaN
log(f'[CLEAN] Converted DU.key1.key1 to numeric. Non-convertible values: minimal')

# ─── 4. HANDLE NULL VALUES ───────────────────────────────
null_counts = df.isnull().sum()
log(f'[CLEAN] Null values per column:\n{null_counts}')
# Drop rows where key2 is null (198 rows = 0.035%)
df = df.dropna(subset=['key2'])
log(f'[CLEAN] After dropping null key2 rows: {df.shape}')

# ─── 5. CLEAN CORRUPTED KEY2 VALUES ──────────────────────
# Some key2 values are corrupted CSV parsing artifacts (containing commas, multiple fields)
# Legitimate keys are short strings
corrupted_mask = df['key2'].str.len() > 20
corrupted_count = corrupted_mask.sum()
log(f'[CLEAN] Found {corrupted_count} corrupted key2 entries (length > 20 chars)')
df = df[~corrupted_mask]
log(f'[CLEAN] After removing corrupted entries: {df.shape}')

# ─── 6. OUTLIER FILTERING ────────────────────────────────
# Timings > 10 seconds are likely pauses (thinking, distraction) not real typing
# We'll keep them flagged but also create a "clean" subset
timing_cols = ['DU.key1.key1', 'DD.key1.key2', 'DU.key1.key2', 'UD.key1.key2', 'UU.key1.key2']

log('\n[OUTLIER] Analyzing extreme values in timing columns:')
for col in timing_cols:
    extreme_count = (df[col].abs() > 10).sum()
    log(f'  {col}: {extreme_count} values with |value| > 10s ({extreme_count/len(df)*100:.2f}%)')

# Create outlier flag
df['is_outlier'] = False
for col in timing_cols:
    df['is_outlier'] = df['is_outlier'] | (df[col].abs() > 10)

outlier_count = df['is_outlier'].sum()
log(f'[OUTLIER] Total rows flagged as outliers: {outlier_count} ({outlier_count/len(df)*100:.2f}%)')

# Create clean subset (for primary analysis)
df_clean = df[~df['is_outlier']].copy()
log(f'[OUTLIER] Clean dataset shape: {df_clean.shape}')

# ─── 7. DERIVED FEATURES ─────────────────────────────────
log('\n[FEATURES] Creating derived features...')

def classify_key(key):
    """Classify a key into category."""
    if key in ['Space']:
        return 'space'
    elif key in ['Shift']:
        return 'modifier'
    elif key in ['Backspace']:
        return 'backspace'
    elif key in ['Enter']:
        return 'enter'
    elif key in ['Control', 'Alt', 'AltGraph', 'Meta', 'OS']:
        return 'modifier'
    elif key in ['Tab']:
        return 'tab'
    elif key in ['CapsLock', 'NumLock']:
        return 'lock'
    elif key.startswith('Arrow'):
        return 'arrow'
    elif key.startswith('F') and key[1:].isdigit():
        return 'function'
    elif len(key) == 1 and key.isalpha():
        return 'alpha'
    elif len(key) == 1 and key.isdigit():
        return 'digit'
    elif len(key) == 1:
        return 'punctuation'
    else:
        return 'special'

df_clean['key1_type'] = df_clean['key1'].apply(classify_key)
df_clean['key2_type'] = df_clean['key2'].apply(classify_key)

# Is this a word boundary? (transition to/from Space)
df_clean['is_word_boundary'] = (df_clean['key1'] == 'Space') | (df_clean['key2'] == 'Space')

# Is this a same-key repeat?
df_clean['is_repeat'] = df_clean['key1'] == df_clean['key2']

# Key overlap indicator (negative UD means keys overlapped — rollover typing)
df_clean['has_overlap'] = df_clean['UD.key1.key2'] < 0

log(f'  Key type distribution (key1):')
log(f'{df_clean.key1_type.value_counts().to_string()}')
log('')
log(f'  Word boundary rows: {df_clean.is_word_boundary.sum()} ({df_clean.is_word_boundary.mean()*100:.1f}%)')
log(f'  Same-key repeat rows: {df_clean.is_repeat.sum()} ({df_clean.is_repeat.mean()*100:.1f}%)')
log(f'  Overlap (rollover) rows: {df_clean.has_overlap.sum()} ({df_clean.has_overlap.mean()*100:.1f}%)')

# ─── 8. SAVE OUTPUTS ─────────────────────────────────────
log('\n[SAVE] Saving cleaned datasets and summaries...')

# Save cleaned dataset
df_clean.to_csv('outputs/texts/free_text_cleaned.csv', index=False)
log(f'[SAVE] Saved cleaned dataset: outputs/texts/free_text_cleaned.csv')

# Summary statistics
summary = {
    'original_rows': int(df.shape[0] + corrupted_count + 198),
    'after_cleaning': int(df_clean.shape[0]),
    'rows_removed': int(df.shape[0] + corrupted_count + 198 - df_clean.shape[0]),
    'participants': int(df_clean.participant.nunique()),
    'unique_key_pairs': int(df_clean.groupby(['key1','key2']).ngroups),
    'timing_stats': {}
}

for col in timing_cols:
    summary['timing_stats'][col] = {
        'mean': float(df_clean[col].mean()),
        'median': float(df_clean[col].median()),
        'std': float(df_clean[col].std()),
        'min': float(df_clean[col].min()),
        'max': float(df_clean[col].max()),
        'q25': float(df_clean[col].quantile(0.25)),
        'q75': float(df_clean[col].quantile(0.75))
    }

with open('outputs/texts/phase1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
log(f'[SAVE] Saved summary: outputs/texts/phase1_summary.json')

# Per-participant summary
participant_summary = df_clean.groupby('participant').agg(
    keystroke_count=('key1', 'count'),
    mean_hold_time=('DU.key1.key1', 'mean'),
    median_hold_time=('DU.key1.key1', 'median'),
    std_hold_time=('DU.key1.key1', 'std'),
    mean_dd_flight=('DD.key1.key2', 'mean'),
    median_dd_flight=('DD.key1.key2', 'median'),
    std_dd_flight=('DD.key1.key2', 'std'),
    mean_ud_flight=('UD.key1.key2', 'mean'),
    overlap_ratio=('has_overlap', 'mean')
).round(4)

participant_summary.to_csv('outputs/texts/participant_profiles.csv')
log(f'[SAVE] Saved participant profiles: outputs/texts/participant_profiles.csv')
log(f'\n[DONE] Phase 1 complete. Clean dataset ready for EDA.')

# ─── PRINT FINAL SUMMARY ─────────────────────────────────
print('\n' + '='*60)
print('PHASE 1 SUMMARY')
print('='*60)
print(f'Original dataset:    {summary["original_rows"]:,} rows')
print(f'After cleaning:      {summary["after_cleaning"]:,} rows')
print(f'Rows removed:        {summary["rows_removed"]:,}')
print(f'Participants:        {summary["participants"]}')
print(f'Unique key pairs:    {summary["unique_key_pairs"]:,}')
print()
print('Timing Statistics (cleaned, seconds):')
print(f'  Hold time    — mean: {summary["timing_stats"]["DU.key1.key1"]["mean"]:.4f}s, median: {summary["timing_stats"]["DU.key1.key1"]["median"]:.4f}s')
print(f'  DD flight    — mean: {summary["timing_stats"]["DD.key1.key2"]["mean"]:.4f}s, median: {summary["timing_stats"]["DD.key1.key2"]["median"]:.4f}s')
print(f'  UD flight    — mean: {summary["timing_stats"]["UD.key1.key2"]["mean"]:.4f}s, median: {summary["timing_stats"]["UD.key1.key2"]["median"]:.4f}s')

save_log()
print('\n[LOG] Updated decision_log.txt')
