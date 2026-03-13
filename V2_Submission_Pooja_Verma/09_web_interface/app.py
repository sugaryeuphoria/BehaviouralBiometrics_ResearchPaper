"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 9: Web Interface — Flask Backend
========================================
- Serves the typing simulation web interface
- API endpoint for generating keystroke sequences
- API endpoint for evaluation metrics
"""

from flask import Flask, jsonify, request, send_from_directory
import sys
import os
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from importlib.machinery import SourceFileLoader
sim_module = SourceFileLoader("sim", os.path.join(PROJECT_ROOT, "07_simulation_engine.py")).load_module()
HumanKeystrokeSimulator = sim_module.HumanKeystrokeSimulator

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Pre-load simulators for each speed profile
simulators = {}
for speed in ['slow', 'medium', 'fast']:
    try:
        simulators[speed] = HumanKeystrokeSimulator(
            data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'texts'),
            speed_profile=speed
        )
    except Exception as e:
        print(f"Warning: Could not load {speed} simulator: {e}")

print(f"Loaded {len(simulators)} simulator profiles: {list(simulators.keys())}")


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Generate keystroke sequence for given text."""
    data = request.json
    text = data.get('text', '')
    speed = data.get('speed', 'medium')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if len(text) > 5000:
        return jsonify({'error': 'Text too long (max 5000 chars)'}), 400
    
    if speed not in simulators:
        speed = 'medium'
    
    sim = simulators[speed]
    keystrokes = sim.simulate(text)
    metrics = sim.get_metrics(keystrokes)
    
    return jsonify({
        'keystrokes': keystrokes,
        'metrics': metrics,
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate keystroke sequence against ML models."""
    try:
        import joblib
        import numpy as np
        import pandas as pd
        
        data = request.json
        keystrokes = data.get('keystrokes', [])
        
        if len(keystrokes) < 5:
            return jsonify({'error': 'Not enough keystrokes for evaluation'}), 400
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load models
        rf = joblib.load(os.path.join(base_dir, 'outputs/models/random_forest_model.joblib'))
        gb = joblib.load(os.path.join(base_dir, 'outputs/models/gradient_boosting_model.joblib'))
        scaler = joblib.load(os.path.join(base_dir, 'outputs/models/scaler.joblib'))
        
        with open(os.path.join(base_dir, 'outputs/models/feature_columns.json'), 'r') as f:
            feature_cols = json.load(f)
        
        # Extract features from the first window
        dd_times = [k['dd_time'] for k in keystrokes[1:]]
        hold_times = [k['hold_time'] for k in keystrokes]
        
        ud_times = []
        uu_times = []
        for j in range(1, len(keystrokes)):
            ud = keystrokes[j]['dd_time'] - keystrokes[j-1]['hold_time']
            uu = keystrokes[j]['dd_time'] + keystrokes[j]['hold_time'] - keystrokes[j-1]['hold_time']
            ud_times.append(ud)
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
        
        X = np.array([[feat.get(c, 0.0) for c in feature_cols]])
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        X_scaled = scaler.transform(X)
        
        rf_prob = float(rf.predict_proba(X_scaled)[0, 1])
        gb_prob = float(gb.predict_proba(X_scaled)[0, 1])
        
        return jsonify({
            'random_forest': {
                'human_probability': rf_prob,
                'verdict': 'Human' if rf_prob > 0.5 else 'Synthetic'
            },
            'gradient_boosting': {
                'human_probability': gb_prob,
                'verdict': 'Human' if gb_prob > 0.5 else 'Synthetic'
            },
            'overall_verdict': 'Human-like' if (rf_prob + gb_prob) / 2 > 0.5 else 'Synthetic-like',
            'confidence': round((rf_prob + gb_prob) / 2 * 100, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Human Keystroke Simulator Web Interface...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=False, port=5001, host='0.0.0.0')
