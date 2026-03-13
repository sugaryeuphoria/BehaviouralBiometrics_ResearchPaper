"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 5: ML Classification Models — Human vs. Synthetic
=========================================================
- Train classifiers to distinguish human from synthetic keystrokes
- Generate naive synthetic data as negative class
- Multiple model comparison (RF, XGBoost, SVM)
- These models will later evaluate our simulation engine
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                              roc_auc_score, roc_curve, classification_report, confusion_matrix)
from sklearn.ensemble import AdaBoostClassifier
import os
import json
import joblib
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
os.makedirs('outputs/models', exist_ok=True)

log_entries = []
def log(msg):
    log_entries.append(msg)
    print(msg)

def save_log():
    with open('decision_log.txt', 'a') as f:
        f.write('\n' + '='*80 + '\n')
        f.write('PHASE 5: ML CLASSIFICATION MODELS\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')

# ─── LOAD REAL HUMAN DATA ────────────────────────────────
log('[LOAD] Loading windowed human features...')
human_df = pd.read_csv('outputs/texts/window_features.csv')
log(f'[LOAD] Human windows: {len(human_df)}')

# Feature columns (exclude ID columns)
feature_cols = [c for c in human_df.columns if c not in ['participant', 'session', 'window_id']]
log(f'[LOAD] Feature columns ({len(feature_cols)}): {feature_cols}')

# ─── GENERATE SYNTHETIC DATA ─────────────────────────────
log('\n[SYNTH] Generating synthetic keystroke data (3 types)...')

n_synthetic = len(human_df)

def generate_fixed_delay(n, feature_cols):
    """Naive: fixed delay per keystroke (constant timing)."""
    data = {}
    for col in feature_cols:
        if 'mean' in col or 'median' in col:
            data[col] = np.random.uniform(0.1, 0.3, n)  # fixed range
        elif 'std' in col or 'iqr' in col:
            data[col] = np.full(n, 0.01)  # very low variance (unrealistic)
        elif 'skew' in col:
            data[col] = np.random.normal(0, 0.1, n)
        elif 'neg_ratio' in col:
            data[col] = np.zeros(n)  # no overlaps
        elif 'ratio' in col or 'cv' in col:
            data[col] = np.random.uniform(0.2, 0.6, n)
        elif 'min' in col:
            data[col] = np.random.uniform(0.08, 0.15, n)
        elif 'max' in col:
            data[col] = np.random.uniform(0.2, 0.4, n)
        elif 'range' in col:
            data[col] = np.random.uniform(0.05, 0.15, n)
        else:
            data[col] = np.random.uniform(0.1, 0.3, n)
    return pd.DataFrame(data)

def generate_random_delay(n, feature_cols):
    """Random delays from uniform distribution."""
    data = {}
    for col in feature_cols:
        if 'mean' in col or 'median' in col:
            data[col] = np.random.uniform(0.05, 0.5, n)
        elif 'std' in col:
            data[col] = np.random.uniform(0.02, 0.15, n)
        elif 'iqr' in col:
            data[col] = np.random.uniform(0.03, 0.2, n)
        elif 'skew' in col:
            data[col] = np.random.uniform(-1, 3, n)
        elif 'neg_ratio' in col:
            data[col] = np.random.uniform(0, 0.05, n)  # very low overlap
        elif 'ratio' in col or 'cv' in col:
            data[col] = np.random.uniform(0.1, 1.0, n)
        elif 'min' in col:
            data[col] = np.random.uniform(0.02, 0.1, n)
        elif 'max' in col:
            data[col] = np.random.uniform(0.3, 1.0, n)
        elif 'range' in col:
            data[col] = np.random.uniform(0.1, 0.5, n)
        else:
            data[col] = np.random.uniform(0.05, 0.5, n)
    return pd.DataFrame(data)

def generate_statistical_mimic(n, feature_cols, human_df):
    """Mimics overall human statistics but without per-bigram context."""
    data = {}
    for col in feature_cols:
        mean = human_df[col].mean()
        std = human_df[col].std()
        # Sample from normal with same mean/std
        data[col] = np.random.normal(mean, std, n)
        # Clip to reasonable range
        data[col] = np.clip(data[col], human_df[col].quantile(0.01), human_df[col].quantile(0.99))
    return pd.DataFrame(data)

# Generate each type
synth_fixed = generate_fixed_delay(n_synthetic // 3, feature_cols)
synth_random = generate_random_delay(n_synthetic // 3, feature_cols)
synth_mimic = generate_statistical_mimic(n_synthetic // 3, feature_cols, human_df)

synth_all = pd.concat([synth_fixed, synth_random, synth_mimic], ignore_index=True)
synth_all['synth_type'] = (['fixed'] * (n_synthetic // 3) + 
                            ['random'] * (n_synthetic // 3) + 
                            ['statistical'] * (n_synthetic // 3))

log(f'[SYNTH] Generated {len(synth_all)} synthetic windows')
log(f'  Fixed delay: {n_synthetic // 3}')
log(f'  Random delay: {n_synthetic // 3}')
log(f'  Statistical mimic: {n_synthetic // 3}')

# ─── PREPARE TRAINING DATA ───────────────────────────────
log('\n[TRAIN] Preparing training data...')

X_human = human_df[feature_cols].values
X_synth = synth_all[feature_cols].values

y_human = np.ones(len(X_human))   # 1 = human
y_synth = np.zeros(len(X_synth))  # 0 = synthetic

X = np.vstack([X_human, X_synth])
y = np.concatenate([y_human, y_synth])

# Handle NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

log(f'[TRAIN] Train: {len(X_train)}, Test: {len(X_test)}')
log(f'[TRAIN] Class balance - Train: {y_train.mean():.3f} human, Test: {y_test.mean():.3f} human')

# ─── TRAIN MODELS ────────────────────────────────────────
log('\n[MODEL] Training multiple classifiers...')

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
}

results = {}
model_objects = {}

for name, model in models.items():
    log(f'\n  Training {name}...')
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc_roc': float(auc),
    }
    
    model_objects[name] = {
        'model': model,
        'y_prob': y_prob,
        'y_pred': y_pred,
    }
    
    log(f'  {name}: acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    results[name]['cv_f1_mean'] = float(cv_scores.mean())
    results[name]['cv_f1_std'] = float(cv_scores.std())
    log(f'  {name} CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Save results
with open('outputs/texts/phase5_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
log('\n[SAVE] Saved: outputs/texts/phase5_model_results.json')

# ─── FEATURE IMPORTANCE ──────────────────────────────────
log('\n[FEAT] Extracting feature importance...')

# Use Random Forest feature importance
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feat_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

feat_importance.to_csv('outputs/texts/feature_importance.csv', index=False)
log('[SAVE] Saved: outputs/texts/feature_importance.csv')

log('\nTop 10 features for distinguishing human vs synthetic:')
for _, row in feat_importance.head(10).iterrows():
    log(f'  {row["feature"]}: {row["importance"]:.4f}')

# ─── PLOT 11: Model Comparison & ROC Curves ──────────────
log('\n[PLOT 11] Creating model comparison plots...')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ROC curves
ax = axes[0]
for name, obj in model_objects.items():
    fpr, tpr, _ = roc_curve(y_test, obj['y_prob'])
    auc = results[name]['auc_roc']
    color = {'Random Forest': '#58a6ff', 'Gradient Boosting': '#7ee787', 'AdaBoost': '#f778ba'}[name]
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

ax.plot([0, 1], [0, 1], 'w--', alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Human vs Synthetic', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

# Accuracy comparison bar chart
ax = axes[1]
model_names = list(results.keys())
metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
x = np.arange(len(model_names))
width = 0.2
colors_m = ['#58a6ff', '#7ee787', '#ffa657', '#d2a8ff']

for i, metric in enumerate(metrics_to_plot):
    values = [results[name][metric] for name in model_names]
    ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), 
           color=colors_m[i], alpha=0.8)

ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=10)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.5, 1.05)

# Feature importance
ax = axes[2]
top_feats = feat_importance.head(12)
ax.barh(range(len(top_feats)), top_feats['importance'].values, color='#d2a8ff', alpha=0.8)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats['feature'].values, fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Top Features for Detection', fontsize=14, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/plots/11_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 11] Saved: outputs/plots/11_model_comparison.png')

# ─── PLOT 12: Confusion Matrices ─────────────────────────
log('\n[PLOT 12] Creating confusion matrices...')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, (name, obj) in enumerate(model_objects.items()):
    ax = axes[idx]
    cm = confusion_matrix(y_test, obj['y_pred'])
    
    im = ax.imshow(cm, cmap='Blues', alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Synthetic', 'Human'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Synthetic', 'Human'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=14, fontweight='bold')

plt.suptitle('Confusion Matrices: Human vs Synthetic Detection', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/plots/12_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
log('[PLOT 12] Saved: outputs/plots/12_confusion_matrices.png')

# ─── SAVE MODELS ─────────────────────────────────────────
log('\n[SAVE] Saving trained models...')

for name, model in models.items():
    filename = f'outputs/models/{name.lower().replace(" ", "_")}_model.joblib'
    joblib.dump(model, filename)
    log(f'  Saved: {filename}')

joblib.dump(scaler, 'outputs/models/scaler.joblib')
log('  Saved: outputs/models/scaler.joblib')

# Save feature column names
with open('outputs/models/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)
log('  Saved: outputs/models/feature_columns.json')

# ─── SUMMARY ─────────────────────────────────────────────
log('\n[SUMMARY] Key Findings:')
best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
log(f'  Best model: {best_model} (F1={results[best_model]["f1_score"]:.4f})')
log(f'  All models achieve very high accuracy detecting naive synthetic data')
log(f'  Key distinguishing features: {", ".join(feat_importance.head(5)["feature"].tolist())}')
log(f'  This baseline will be used to evaluate our simulation engine in Phase 8')
log(f'  Goal: our simulated data should DROP these accuracy scores toward 50%')

log('\n[DONE] Phase 5 complete. Models trained and saved.')
save_log()
print('\n[LOG] Updated decision_log.txt')
