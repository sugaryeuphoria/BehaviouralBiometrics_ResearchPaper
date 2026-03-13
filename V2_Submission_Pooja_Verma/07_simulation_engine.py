"""
Author: Pooja Verma
TRU ID: T00729545
Course: COMP 4980 — Behavioral Biometrics

Phase 7: Human Keystroke Simulation Engine (Core Function)
============================================================
- Takes any input text and generates realistic keystroke timings
- Uses distribution parameters learned from real human data
- Incorporates natural variability, context effects, and typing patterns
"""

import pandas as pd
import numpy as np
from scipy.stats import lognorm, gamma, weibull_min, norm
import json
import os

# ─── LOAD DISTRIBUTION PARAMETERS ────────────────────────

def load_simulation_data():
    """Load all pre-computed distribution parameters."""
    data = {}
    
    # Bigram distribution parameters
    with open('outputs/texts/bigram_distribution_params.json', 'r') as f:
        data['bigram_params'] = json.load(f)
    
    # Per-key hold time distributions
    with open('outputs/texts/key_hold_distributions.json', 'r') as f:
        data['key_hold_params'] = json.load(f)
    
    # Context distributions
    with open('outputs/texts/context_distributions.json', 'r') as f:
        data['context_params'] = json.load(f)
    
    # Global distribution fits
    with open('outputs/texts/global_distribution_fits.json', 'r') as f:
        data['global_params'] = json.load(f)
    
    # Bigram features (for fallback stats)
    data['bigram_features'] = pd.read_csv('outputs/texts/bigram_features.csv')
    
    return data

DIST_MAP = {
    'lognormal': lognorm,
    'gamma': gamma,
    'weibull': weibull_min,
    'normal': norm,
}


class HumanKeystrokeSimulator:
    """
    Simulates realistic human typing behavior for any input text.
    
    Uses distribution parameters learned from real keystroke data to generate
    timings that include natural variability, context effects, and typing patterns.
    """
    
    def __init__(self, data_dir='outputs/texts', speed_profile='medium'):
        """
        Initialize simulator with learned parameters.
        
        Args:
            data_dir: Directory containing distribution parameter files
            speed_profile: 'slow', 'medium', or 'fast' typing speed
        """
        self.data = load_simulation_data()
        self.speed_profile = speed_profile
        
        # Speed multipliers
        self.speed_multipliers = {
            'slow': 1.4,
            'medium': 1.0,
            'fast': 0.7,
        }
        self.speed_mult = self.speed_multipliers[speed_profile]
        
        # Build lookup dictionaries for fast access
        self._build_lookups()
        
        # State variables for drift modeling
        self._fatigue_factor = 1.0
        self._keystroke_count = 0
        self._momentum = 0.0  # typing rhythm momentum
    
    def _build_lookups(self):
        """Build efficient lookup structures from loaded data."""
        # Bigram -> distribution params
        self.bigram_dd_lookup = {}
        self.bigram_hold_lookup = {}
        self.bigram_ud_lookup = {}
        
        for bigram_key, params in self.data['bigram_params'].items():
            k1, k2 = bigram_key.split('->')
            if 'dd' in params:
                self.bigram_dd_lookup[(k1, k2)] = params['dd']
            if 'hold' in params:
                self.bigram_hold_lookup[(k1, k2)] = params['hold']
            if 'ud' in params:
                self.bigram_ud_lookup[(k1, k2)] = params['ud']
        
        # Key -> hold time params
        self.key_hold_lookup = {}
        for key, params in self.data['key_hold_params'].items():
            self.key_hold_lookup[key] = params
        
        # Global fallback stats
        bf = self.data['bigram_features']
        self.global_dd_mean = bf['dd_median'].median()
        self.global_dd_std = bf['dd_std'].median()
        self.global_hold_mean = bf['hold_median'].median()
        self.global_hold_std = 0.03
        
        # Context multipliers (learned from data)
        ctx = self.data['context_params']
        self.word_start_multiplier = (ctx.get('word_start_dd', {}).get('data_stats', {}).get('median', 0.24) / 
                                       ctx.get('mid_word_dd', {}).get('data_stats', {}).get('median', 0.16))
        self.word_end_multiplier = (ctx.get('word_end_dd', {}).get('data_stats', {}).get('median', 0.16) / 
                                     ctx.get('mid_word_dd', {}).get('data_stats', {}).get('median', 0.16))
    
    def _sample_from_distribution(self, params_dict, clip_min=0.02, clip_max=2.0):
        """Sample a single value from a fitted distribution with realistic variance."""
        dist_name = params_dict.get('best_dist', 'lognormal')
        params = params_dict.get('params', None)
        data_stats = params_dict.get('data_stats', {})
        
        if params and dist_name in DIST_MAP:
            dist = DIST_MAP[dist_name]
            try:
                sample = dist.rvs(*params, size=1)[0]
                # Clip to reasonable range
                sample = np.clip(sample, clip_min, clip_max)
                return float(sample)
            except Exception:
                pass
        
        # Fallback: sample from normal based on data stats
        mean = data_stats.get('median', 0.15)
        std = data_stats.get('std', 0.05)
        q25 = data_stats.get('q25', mean - std)
        q75 = data_stats.get('q75', mean + std)
        iqr = q75 - q25
        
        # Use truncated normal around median
        sample = np.random.normal(mean, iqr * 0.5)
        sample = np.clip(sample, clip_min, clip_max)
        return float(sample)
    
    def _get_dd_time(self, key1, key2, context='mid_word'):
        """Get realistic Down-Down flight time for a key pair."""
        
        # 1. Try bigram-specific distribution
        if (key1, key2) in self.bigram_dd_lookup:
            dd = self._sample_from_distribution(self.bigram_dd_lookup[(key1, key2)])
        
        # 2. Try case-insensitive lookup
        elif (key1.lower(), key2.lower()) in self.bigram_dd_lookup:
            dd = self._sample_from_distribution(self.bigram_dd_lookup[(key1.lower(), key2.lower())])
        
        # 3. Fallback: use global stats with noise
        else:
            dd = np.random.lognormal(
                np.log(self.global_dd_mean), 
                0.4
            )
            dd = np.clip(dd, 0.04, 1.5)
        
        # Apply context multiplier
        if context == 'word_start':
            dd *= self.word_start_multiplier
        elif context == 'word_end':
            dd *= self.word_end_multiplier
        
        # Apply speed profile
        dd *= self.speed_mult
        
        # Apply fatigue drift (subtle)
        dd *= self._fatigue_factor
        
        # Apply momentum (smoothing with previous timing)
        if self._momentum > 0:
            dd = 0.7 * dd + 0.3 * self._momentum
        self._momentum = dd
        
        return float(dd)
    
    def _get_hold_time(self, key):
        """Get realistic hold time for a key press."""
        
        # 1. Try key-specific distribution
        if key.lower() in self.key_hold_lookup:
            hold = self._sample_from_distribution(self.key_hold_lookup[key.lower()], clip_max=0.5)
        
        # 2. Fallback based on key type
        elif key == ' ':
            hold = np.random.lognormal(np.log(0.08), 0.25)
        elif key in '.,!?;:':
            hold = np.random.lognormal(np.log(0.10), 0.3)
        else:
            hold = np.random.lognormal(np.log(self.global_hold_mean), 0.3)
        
        # Apply speed profile  
        hold *= self.speed_mult * 0.8  # Hold times less affected by speed
        
        return float(np.clip(hold, 0.02, 0.4))
    
    def _char_to_key(self, char):
        """Convert a character to its key name in the dataset format."""
        if char == ' ':
            return 'Space'
        elif char == '\n':
            return 'Enter'
        elif char == '\t':
            return 'Tab'
        elif char == '.':
            return '.'
        elif char == ',':
            return ','
        elif char.isupper():
            return char.upper()
        else:
            return char.lower()
    
    def _needs_shift(self, char):
        """Check if a character requires Shift key."""
        return char.isupper() or char in '!@#$%^&*()_+{}|:"<>?~'
    
    def _add_thinking_pause(self, current_pos, text):
        """Model natural thinking pauses at sentence/paragraph boundaries."""
        pause = 0.0
        
        if current_pos > 0:
            prev_char = text[current_pos - 1]
            
            # Sentence boundary pause
            if prev_char in '.!?':
                pause = np.random.lognormal(np.log(0.5), 0.5)
                pause = np.clip(pause, 0.2, 3.0)
            
            # Comma/semicolon pause
            elif prev_char in ',;:':
                pause = np.random.lognormal(np.log(0.15), 0.4)
                pause = np.clip(pause, 0.05, 0.8)
            
            # Paragraph boundary
            elif prev_char == '\n':
                pause = np.random.lognormal(np.log(1.0), 0.6)
                pause = np.clip(pause, 0.3, 5.0)
        
        return float(pause)
    
    def _update_fatigue(self):
        """Update fatigue factor (typing speed gradually changes)."""
        self._keystroke_count += 1
        
        # Gradual fatigue: slight slowdown over time with random recovery
        if self._keystroke_count % 50 == 0:
            # Small random drift
            drift = np.random.normal(0, 0.02)
            self._fatigue_factor = np.clip(self._fatigue_factor + drift, 0.9, 1.15)
        
        # Occasional "burst" of faster/slower typing
        if np.random.random() < 0.01:
            self._fatigue_factor = np.random.uniform(0.85, 1.1)
    
    def simulate(self, text):
        """
        Generate realistic keystroke sequence for given text.
        
        Args:
            text: Input text string to simulate typing for
            
        Returns:
            List of dicts with keys: 'char', 'key', 'key_down', 'key_up', 
            'hold_time', 'dd_time', 'elapsed'
        """
        # Reset state
        self._fatigue_factor = 1.0
        self._keystroke_count = 0
        self._momentum = 0.0
        
        keystrokes = []
        current_time = 0.0
        prev_key = None
        
        for i, char in enumerate(text):
            key = self._char_to_key(char)
            
            # Determine context
            if i == 0:
                context = 'word_start'
            elif char == ' ':
                context = 'word_end' 
            elif i > 0 and text[i-1] == ' ':
                context = 'word_start'
            else:
                context = 'mid_word'
            
            # Calculate timing
            if prev_key is not None:
                # Down-Down flight time
                dd_time = self._get_dd_time(prev_key, key, context)
                
                # Add thinking pause if applicable
                pause = self._add_thinking_pause(i, text)
                dd_time += pause
            else:
                dd_time = 0.0
            
            # Hold time
            hold_time = self._get_hold_time(key)
            
            # Calculate key events
            key_down = current_time + dd_time
            key_up = key_down + hold_time
            
            keystroke = {
                'char': char,
                'key': key,
                'key_down': round(key_down, 4),
                'key_up': round(key_up, 4),
                'hold_time': round(hold_time, 4),
                'dd_time': round(dd_time, 4),
                'elapsed': round(key_down, 4),
                'context': context,
            }
            keystrokes.append(keystroke)
            
            # Update state
            current_time = key_down
            prev_key = key
            self._update_fatigue()
        
        return keystrokes
    
    def get_metrics(self, keystrokes):
        """Calculate typing metrics from a keystroke sequence."""
        if len(keystrokes) < 2:
            return {}
        
        dd_times = [k['dd_time'] for k in keystrokes[1:]]
        hold_times = [k['hold_time'] for k in keystrokes]
        
        total_time = keystrokes[-1]['key_up'] - keystrokes[0]['key_down']
        char_count = len(keystrokes)
        word_count = sum(1 for k in keystrokes if k['char'] == ' ') + 1
        
        return {
            'total_time_s': round(total_time, 2),
            'characters': char_count,
            'words': word_count,
            'wpm': round(word_count / (total_time / 60), 1) if total_time > 0 else 0,
            'cpm': round(char_count / (total_time / 60), 1) if total_time > 0 else 0,
            'avg_dd_ms': round(np.mean(dd_times) * 1000, 1),
            'median_dd_ms': round(np.median(dd_times) * 1000, 1),
            'std_dd_ms': round(np.std(dd_times) * 1000, 1),
            'avg_hold_ms': round(np.mean(hold_times) * 1000, 1),
            'median_hold_ms': round(np.median(hold_times) * 1000, 1),
            'std_hold_ms': round(np.std(hold_times) * 1000, 1),
            'min_dd_ms': round(np.min(dd_times) * 1000, 1),
            'max_dd_ms': round(np.max(dd_times) * 1000, 1),
        }
    
    def to_json(self, keystrokes):
        """Export keystrokes as JSON for web interface."""
        return json.dumps(keystrokes, indent=2)


# ─── MAIN: TEST THE SIMULATOR ────────────────────────────
if __name__ == '__main__':
    log_entries = []
    def log(msg):
        log_entries.append(msg)
        print(msg)
    
    log('='*60)
    log('PHASE 7: HUMAN KEYSTROKE SIMULATION ENGINE')
    log('='*60)
    
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog. This is a test of the human keystroke simulation engine."
    
    # Test all speed profiles
    for speed in ['slow', 'medium', 'fast']:
        log(f'\n--- Speed Profile: {speed} ---')
        sim = HumanKeystrokeSimulator(speed_profile=speed)
        keystrokes = sim.simulate(test_text)
        metrics = sim.get_metrics(keystrokes)
        
        log(f'  Text length: {len(test_text)} chars')
        log(f'  WPM: {metrics["wpm"]}')
        log(f'  Avg DD: {metrics["avg_dd_ms"]}ms')
        log(f'  Median DD: {metrics["median_dd_ms"]}ms')
        log(f'  DD Std: {metrics["std_dd_ms"]}ms')
        log(f'  Avg Hold: {metrics["avg_hold_ms"]}ms')
        log(f'  Total time: {metrics["total_time_s"]}s')
    
    # Detailed output for medium profile
    log('\n--- Detailed keystroke sequence (first 20, medium) ---')
    sim = HumanKeystrokeSimulator(speed_profile='medium')
    keystrokes = sim.simulate(test_text)
    
    for i, ks in enumerate(keystrokes[:20]):
        log(f'  [{i:3d}] char="{ks["char"]}" key={ks["key"]:8s} '
            f'dd={ks["dd_time"]*1000:6.1f}ms hold={ks["hold_time"]*1000:5.1f}ms '
            f'context={ks["context"]}')
    
    # Save sample output
    sample_output = {
        'input_text': test_text,
        'speed_profile': 'medium',
        'keystrokes': keystrokes,
        'metrics': sim.get_metrics(keystrokes),
    }
    with open('outputs/texts/phase7_sample_output.json', 'w') as f:
        json.dump(sample_output, f, indent=2)
    log('\n[SAVE] Saved sample output: outputs/texts/phase7_sample_output.json')
    
    # Update decision log
    with open('decision_log.txt', 'a') as f:
        f.write('\n' + '='*80 + '\n')
        f.write('PHASE 7: HUMAN KEYSTROKE SIMULATION ENGINE\n')
        f.write('='*80 + '\n')
        for entry in log_entries:
            f.write(entry + '\n')
    
    log('\n[DONE] Phase 7 complete. Simulation engine ready for evaluation.')
