# Core Project Code Snippets

This document contains the most important code segments from the research project. These snippets represent the core logic for statistical modeling, machine learning detection, and the final simulation engine.

---

### 1. Statistical Modeling: Fitting Distributions to Typing Data
This code identifies the best-fit mathematical distribution for the typing patterns of specific key pairs (like "t to h"). This allows the simulation to mimic the exact timing behavior observed in real humans.

```python
# We iterate through the top 100 most common key pairs (bigrams)
for key1, key2 in top_bigrams:
    # Get all timing data for this specific sequence (e.g., 'A' followed by 'S')
    bigram_data = df_typing[(df_typing['key1'] == key1) & (df_typing['key2'] == key2)]
    
    # Try fitting log-normal, gamma, and Weibull distributions to the data
    # We choose the 'best' one based on how well it matches the real data points
    result = fit_and_evaluate(bigram_data['DD.key1.key2'].values)
    
    if result:
        best_dist = result['best_dist']
        params = result[best_dist]['params']
        
        # Save these parameters so the simulation engine can 'sample' from them later
        bigram_dist_params[f"{key1}->{key2}"] = {
            'best_dist': best_dist,
            'params': params
        }
```

---

### 2. Detection Engine: Training Machine Learning Models
This code trains ensemble classifiers (Random Forest, Gradient Boosting) to distinguish between real human typing and fake (synthetic) typing. This establishes the benchmark that the simulation engine must try to bypass.

```python
# We test multiple models to see which one detects synthetic typing best
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

# Train and evaluate each model
for name, model in models.items():
    # 'X_train' contains features like timing variability and rhythm
    # 'y_train' tells the model if the data is Human (1) or Synthetic (0)
    model.fit(X_train_scaled, y_train)
    
    # Predict on unseen data to test accuracy
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate more advanced metrics like AUC (Area Under the Curve)
    # A perfect score is 1.0, though real-world scores are usually slightly lower
    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print(f"{name}: F1 Score = {f1:.4f}, AUC = {auc:.4f}")
```

---

### 3. Simulation Engine: Generating Human-Like Typing
This is the heart of the project. It takes a piece of text and generates human-like timing by sampling from the previously learned distributions and applying "contextual rules" (like pausing at the start of a word).

```python
def simulate_keystroke(self, current_key, next_key, position_type):
    # 1. Look up the statistical profile for this specific key pair
    params = self.bigram_dd_lookup.get((current_key, next_key))
    
    # 2. Generate a random 'flight time' (delay) based on the human distribution
    # This ensures the timing isn't identical every time, just like a real person
    raw_delay = self._sample_from_distribution(params)
    
    # 3. Apply a 'context multiplier' 
    # For example, humans are 50% slower when starting a new word
    if position_type == 'word_start':
        raw_delay *= self.word_start_multiplier
    
    # 4. Add 'momentum' to keep the rhythm steady
    # A fast key press usually follows another fast key press
    final_delay = (raw_delay * 0.7) + (self.previous_delay * 0.3)
    self.previous_delay = final_delay
    
    return final_delay
```
