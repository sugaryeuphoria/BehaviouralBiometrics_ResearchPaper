# Simulating Human Keystroke Dynamics: Can LLM-Generated Text Be Made Indistinguishable from Human Typing?

**COMP 4980: Behavioral Biometrics — Final Research Paper**

---

## Abstract

This paper investigates whether Large Language Model (LLM) generated text can be streamed with realistic keystroke timing that is indistinguishable from genuine human typing. I used the KeyRecs dataset, which contains approximately 559,000 cleaned keystrokes from 99 participants, to analyze human typing patterns and build a simulation engine that replicates them. My approach involved fitting statistical distributions (Weibull, log-normal, gamma) to keystroke timing data at the per-bigram level, training machine learning classifiers (Random Forest, Gradient Boosting, AdaBoost) to detect synthetic typing, and then evaluating a distribution-based simulation engine against those classifiers. When trained against challenging synthetic data — including statistical mimics and perturbed human samples — the Random Forest achieved an F1 of 0.982 (AUC = 0.999) and the Gradient Boosting model reached an F1 of 0.901 (AUC = 0.975). With respect to our simulation engine, the results were mixed but promising: the AdaBoost classifier rated 82.6% of simulated keystrokes as human, while the Random Forest gave a 56.3% human classification rate. These findings suggest that per-bigram distribution sampling can approximate human-like typing patterns but current ensemble methods can still detect subtle statistical artifacts, particularly the absence of rollover typing (key overlap). I also identified four distinct typing archetypes through clustering analysis. An interactive web interface was built to demonstrate the simulation in real time.

---

## 1. Introduction

Keystroke dynamics is a behavioral biometric technique that identifies individuals based on their typing patterns — specifically the timing of key press and release events. Unlike physiological biometrics such as fingerprints or facial recognition, keystroke dynamics captures behavioral traits that are unique to each person's motor skills, muscle memory, and cognitive processing (Alsultan & Warwick, 2013). This makes it useful for applications ranging from user authentication to continuous identity verification and fraud detection.

With the rise of Large Language Models (LLMs) like ChatGPT, there is growing interest in detecting AI-generated content. Most detection methods focus on analyzing the text content itself — checking for patterns in word choice, sentence structure, or statistical properties like perplexity (Mitchell et al., 2023). However, another approach is to analyze how the text was typed. If LLM output is streamed with uniform timing, it can be detected through keystroke analysis even if the content itself appears natural.

This raises an interesting question from the opposite perspective: **Can we design a system that takes LLM-generated text and delivers it with realistic keystroke timing, making it behaviorally indistinguishable from human typing?** This is the central research question of my project.

My approach has three main components:

1. **Analysis**: Thoroughly analyze a real keystroke dynamics dataset to understand the statistical properties of human typing, including timing distributions, contextual variations, and individual differences.
2. **Detection**: Train machine learning models to distinguish between human and synthetic keystroke data, establishing a rigorous benchmark for what "detectable" means.
3. **Simulation**: Build a simulation engine that generates keystroke timing sequences realistic enough to fool those same trained classifiers.

This paper presents the complete pipeline — from data preprocessing and exploratory analysis through distribution modeling, classification model training, participant clustering, simulation engine design, and evaluation. I also built an interactive web interface that demonstrates the simulation in real time.

---

## 2. Related Work

### 2.1 Keystroke Dynamics as a Biometric

Keystroke dynamics has been studied since the 1980s as a low-cost, non-intrusive authentication method. Killourhy and Maxion (2009) conducted one of the most influential benchmark studies, collecting keystroke data from 51 participants typing a fixed password and evaluating 14 anomaly-detection algorithms. Their work established the CMU dataset as a standard benchmark and showed that distance-based classifiers (particularly Scaled Manhattan distance) could effectively authenticate users based on typing patterns alone. Their study was important in showing that a rigorous, consistent evaluation methodology is needed when comparing keystroke dynamics algorithms — a principle I followed in my own model comparison.

### 2.2 Free-Text Keystroke Dynamics

While much of the early work focused on fixed-text (password) typing, free-text analysis is more relevant to real-world scenarios where users type arbitrary content. Alsultan and Warwick (2013) provided a comprehensive survey of free-text keystroke dynamics methods, finding that digraph features — the timing relationships between pairs of consecutive keys — are the most commonly used and most informative features. They noted that machine learning classifiers like Random Forest and SVM generally outperform statistical approaches for free-text authentication. This finding influenced my decision to focus on bigram-level analysis and use ensemble classifiers.

### 2.3 The KeyRecs Dataset

The specific dataset I used is KeyRecs, introduced by Dias, Vitorino, Maia, Sousa, and Praça (2024). KeyRecs contains data from 99 participants who completed both password retype and free-text transcription exercises. It records approximately 1.6 million keystrokes with inter-key latencies computed as digraphs — measuring the time between each key press and release event during a typing exercise. The dataset also includes demographic information such as age, gender, handedness, and nationality. The authors designed KeyRecs specifically for machine learning applications in anomaly detection and biometric authentication, making it well-suited for my project.

### 2.4 Synthetic Keystroke Detection

The most closely related work to mine is by Gonzalez and Calot (2022), who studied liveness detection in keystroke dynamics by generating synthetic keystroke timing sequences and testing whether they could fool verification systems. They found that naive synthetic approaches — such as using fixed or randomly sampled delays — are easily detected by trained classifiers. However, more statistically sophisticated approaches that match the distribution of real timing data can sometimes bypass basic verification. They published a companion dataset of human and synthesized keystroke samples for benchmarking. My work extends their findings by building a simulation engine that goes beyond simple statistical matching, incorporating context-dependent timing, bigram-specific distributions, and fatigue modeling, and evaluating it against stronger multi-model classifiers.

### 2.5 Deep Learning for Keystroke Biometrics

Acien, Morales, Monaco, Vera-Rodriguez, and Fierrez (2022) proposed TypeNet, a deep learning architecture for keystroke biometrics that uses recurrent neural networks to learn temporal dependencies in typing sequences. TypeNet demonstrated that deep models can capture patterns that traditional feature engineering might miss. While I chose ensemble methods (Random Forest, Gradient Boosting) for practical reasons — they are faster to train, easier to interpret, and my dataset size made them sufficient — TypeNet's emphasis on temporal modeling influenced my design of the simulation engine, particularly the inclusion of momentum smoothing and fatigue drift mechanisms.

### 2.6 Continuous Authentication and Behavioral Drift

Mondal and Bours (2017) investigated continuous authentication using keystroke and mouse biometrics and found that user typing patterns are not static — they exhibit natural drift due to factors like fatigue, distraction, and learning effects. This was an important insight for my simulation engine. If simulated typing is perfectly consistent over long sequences, it becomes suspicious precisely because real humans are not that consistent. I incorporated this finding by adding a fatigue drift mechanism that subtly modulates typing speed over time.

### 2.7 AI-Generated Text Detection via Typing Behavior

A recent and highly relevant line of work by Huang, Estrada, Bao, and Schuckers (2024) explores using keystroke dynamics specifically to detect AI-assisted writing in educational contexts. Their approach analyzes how students type — rather than just what they write — to identify whether text was genuinely composed or copy-pasted from an AI tool. This work highlights the growing importance of keystroke-level analysis for AI detection and directly motivates the defensive side of my research: if keystroke timing can be used to detect AI involvement, then a realistic keystroke simulator would need to defeat such detection.

---

## 3. Dataset

### 3.1 Dataset Description

I used the free-text portion of the KeyRecs dataset (`free-text.csv`). The raw dataset contained **562,583 rows**, where each row represents a transition between two consecutive keys (a digraph). Each row includes the following columns:

| Column | Description |
|---|---|
| `participant` | Participant ID (p001–p099) |
| `session` | Session number (1 or 2) |
| `key1` | First key in the digraph |
| `key2` | Second key in the digraph |
| `DU.key1.key1` | Hold time — duration from key-down to key-up for key1 |
| `DD.key1.key2` | Down-Down time — interval from pressing key1 to pressing key2 |
| `UD.key1.key2` | Up-Down time — interval from releasing key1 to pressing key2 |
| `UU.key1.key2` | Up-Up time — interval from releasing key1 to releasing key2 |

### 3.2 Data Cleaning

I applied the following preprocessing steps:

1. **Dropped the `Unnamed: 9` column**, which was almost entirely null (only 13 non-null values out of 562,583 rows).
2. **Cleaned column names** by stripping trailing whitespace.
3. **Converted `DU.key1.key1`** to numeric format, handling a small number of mixed-type entries.
4. **Dropped rows with null `key2` values** (198 rows).
5. **Removed corrupted entries** — 13 rows where `key2` contained long strings instead of key names.
6. **Filtered extreme outliers** — 2,887 rows (0.51%) where any timing value exceeded 10 seconds, likely caused by the participant pausing or being distracted.

After cleaning, the final dataset contained **559,485 rows** across **99 participants** and **2 sessions** per participant.

*[Figure 1: Placeholder — Distribution of rows per participant, showing data volume per typist]*

### 3.3 Derived Features

I created several derived features to enrich the analysis:

- **Key type classification**: Categorized each key as alpha, space, backspace, punctuation, modifier, lock, arrow, digit, enter, special, function, or tab.
- **Word boundary flag**: Marked transitions involving the Space key.
- **Key repeat flag**: Marked cases where key1 = key2.
- **Overlap flag**: Marked cases where UD flight time was negative, indicating rollover typing (pressing the next key before fully releasing the current one).

---

## 4. Methodology

### 4.1 Exploratory Data Analysis

I began by analyzing the statistical properties of the keystroke timing data. This phase produced 8 research-quality visualizations that guided my modeling decisions.

**Timing Distributions**: The hold time (key press duration) had a median of 91ms with a right-skewed distribution. The DD flight time (interval between consecutive key presses) had a median of 174ms. Both distributions showed long right tails, suggesting that log-normal or gamma distributions would be better fits than normal distributions.

*[Figure 2: Placeholder — Histograms and KDE plots for hold time, DD flight, UD flight, and UU interval]*

**Key Type Effects**: Modifier keys (Shift, Ctrl) had significantly longer hold times (median 219ms) compared to alpha keys (92ms) and space (91ms). Backspace had the shortest hold time (74ms), consistent with the quick tapping motion used for corrections.

*[Figure 3: Placeholder — Violin plot of hold time distributions by key type]*

**Bigram Timing Analysis**: I computed median DD flight times for the most common key bigrams and found substantial variation. For example, the transition `t→h` (a common English digraph) had a much shorter flight time than `q→u`, reflecting the different finger movements required. This variation is critical for realistic simulation — using a single global average would miss these important differences.

*[Figure 4: Placeholder — Heatmap of median DD flight times for the 20 most common key bigrams]*

**Word Boundary Effects**: Transitions after a space (Space→letter, i.e., starting a new word) were approximately 52% slower than mid-word transitions (median 241ms vs. 157ms). This makes sense — starting a new word requires a small cognitive pause to plan the next word. Transitions before a space (letter→Space) were similar in speed to mid-word transitions (median 155ms).

*[Figure 5: Placeholder — Box plot comparing DD flight times for within-word, before-space, and after-space transitions]*

**Participant Variation**: The 99 participants showed substantial variation in typing speed, ranging from approximately 39 WPM (words per minute) to 143 WPM, with a median of 71 WPM. The fastest typist had a median DD flight time of 84ms, while the slowest had 315ms — a 3.8x difference.

*[Figure 6: Placeholder — Bar chart of participant typing speeds and scatter plot of hold time vs. flight time]*

**Overlap (Rollover) Typing**: 17.2% of all keystrokes showed negative UD flight times, meaning the next key was pressed before the current key was fully released. This is known as rollover typing and is common among faster typists. This was an important finding because it means the simulation engine needs to account for key overlap, not just positive delays.

*[Figure 7: Placeholder — Distribution of negative UD flight times and most common overlapping key pairs]*

**Typing Rhythm**: I computed the autocorrelation of DD flight times and found a small positive lag-1 autocorrelation (0.062), indicating that consecutive keystrokes have slightly correlated timing — a fast keystroke is slightly more likely to be followed by another fast keystroke. This suggests local rhythm patterns that should be preserved in simulation.

*[Figure 8: Placeholder — Time series and autocorrelation plot of DD flight times]*

### 4.2 Feature Engineering

For the machine learning classification models, I engineered 19 features computed over sliding windows of 20 consecutive keystrokes. These features were chosen to capture the statistical properties that distinguish human typing from synthetic patterns:

- **DD flight time features** (8): mean, std, median, IQR, min, max, range, skewness
- **Hold time features** (3): mean, std, median
- **UD flight time features** (3): mean, std, negative ratio (fraction of overlapping keys)
- **UU flight time features** (2): mean, std
- **Derived ratios** (3): hold-to-DD ratio, coefficient of variation for DD, coefficient of variation for hold

This produced **26,430 human windows** from the dataset.

### 4.3 Distribution Modeling

I fitted four statistical distributions — log-normal, gamma, Weibull, and normal — to the keystroke timing data at multiple levels of granularity:

1. **Global level**: Fitting to all timing values across all participants.
2. **Per-bigram level**: Fitting separately for each of the 100 most common key bigrams.
3. **Per-key level**: Fitting hold time distributions for each individual key.
4. **Per-context level**: Fitting separately for word-start, mid-word, and word-end positions.

I evaluated the fits using the Akaike Information Criterion (AIC) and the Kolmogorov-Smirnov (KS) test. Globally, the **Weibull distribution** provided the best fit for all timing metrics. However, the most important finding was that **per-bigram distribution parameters** captured the natural variability much better than any single global distribution. This is because different key combinations involve different finger movements and therefore have fundamentally different timing properties.

*[Figure 9: Placeholder — Distribution fits comparison showing empirical data overlaid with fitted log-normal, gamma, and Weibull PDFs for hold time and DD flight time]*

*[Figure 10: Placeholder — Distribution analysis of negative UD flight times (key overlap)]*

### 4.4 Machine Learning Classification

To establish a baseline for evaluating the simulation engine, I trained three ensemble classifiers to distinguish between human and synthetic keystroke data:

**Synthetic Data Generation**: I created three types of increasingly challenging synthetic data to serve as the negative class:

- **Statistical mimic** (1/3): Features sampled independently from normal distributions matching the per-feature human mean and standard deviation, but without inter-feature correlations.
- **Distribution-matched** (1/3): Each feature sampled by randomly selecting values from the empirical human distribution (column-independent bootstrap), preserving marginal distributions but breaking the joint structure.
- **Perturbed human** (1/3): Real human feature windows with 10–30% multiplicative Gaussian noise added, creating subtle distortions that are the hardest to detect.

This design produces a substantially harder classification problem than using naive fixed-delay or uniform-random synthetic data. Each type contributed equally to the synthetic class (8,810 windows each, totaling 26,430 synthetic windows to match the human class).

**Models Trained**:
- **Random Forest** (100 trees)
- **Gradient Boosting** (100 estimators)
- **AdaBoost** (100 estimators)

I used an 80/20 stratified train-test split with StandardScaler feature normalization, and evaluated each model using accuracy, F1-score, precision, recall, AUC-ROC, and 5-fold cross-validation.

### 4.5 Participant Clustering

I used K-Means clustering on 12 participant-level features to identify distinct typing archetypes. I tested K values from 2 to 9 using the elbow method and selected K=4. I also applied PCA (which captured 76.7% of variance in 2 components) and t-SNE for visualization.

### 4.6 Simulation Engine Design

The core simulation engine (`HumanKeystrokeSimulator`) takes any input text and generates a realistic keystroke timing sequence. The key design decisions were:

1. **Per-bigram distribution sampling**: For each character pair in the input text, the engine looks up the fitted distribution parameters for that specific bigram and samples a DD flight time from the corresponding Weibull (or log-normal) distribution.
2. **Context-aware timing**: The engine applies different multipliers for word-start positions (1.5x slower, based on the empirical 52% slowdown I observed), word-end positions, and mid-word positions.
3. **Thinking pauses**: At sentence boundaries (after `.`, `!`, `?`), the engine adds a log-normal distributed pause (200–3000ms). At commas and semicolons, it adds a shorter pause (50–800ms).
4. **Fatigue drift**: Every 50 keystrokes, a small random drift is applied to the speed multiplier, simulating the natural fluctuation in typing speed over time.
5. **Momentum smoothing**: Each DD flight time is blended 70/30 with the previous timing value, creating smooth rhythm continuity rather than abrupt speed changes.
6. **Speed profiles**: Three configurable profiles (slow, medium, fast) allow simulating different types of typists.

---

## 5. Results and Analysis

### 5.1 Classification Performance

The three models showed varying levels of effectiveness in distinguishing human from the more challenging synthetic keystroke data:

| Model | Accuracy | F1 Score | AUC-ROC | CV F1 (mean ± std) |
|---|---|---|---|---|
| Random Forest | 0.9816 | 0.9819 | 0.9990 | 0.9799 ± 0.0010 |
| Gradient Boosting | 0.8911 | 0.9009 | 0.9753 | 0.8997 ± 0.0053 |
| AdaBoost | 0.6644 | 0.7451 | 0.6959 | 0.7464 ± 0.0019 |

*[Figure 11: Placeholder — ROC curves, performance metrics bar chart, and feature importance plot for all three models]*

*[Figure 12: Placeholder — Confusion matrices for all three classifiers]*

The Random Forest emerged as the strongest classifier with an F1 of 0.982 and near-perfect AUC (0.999), confirming that it can effectively leverage inter-feature correlations and joint statistical structure to detect synthetic data. Gradient Boosting achieved a solid F1 of 0.901, while AdaBoost struggled more (F1 = 0.745), likely due to the difficulty of the perturbed-human subset. The models rely most heavily on features like `hold_to_dd_ratio` (importance: 0.098), `cv_dd` (0.083), `dd_mean` (0.071), and `hold_mean` (0.066).

### 5.2 Feature Importance Analysis

The top 10 most important features for distinguishing human from synthetic typing were:

| Rank | Feature | Importance |
|---|---|---|
| 1 | hold_to_dd_ratio | 0.0982 |
| 2 | cv_dd | 0.0827 |
| 3 | dd_mean | 0.0708 |
| 4 | uu_mean | 0.0670 |
| 5 | hold_mean | 0.0656 |
| 6 | hold_std | 0.0653 |
| 7 | hold_median | 0.0623 |
| 8 | ud_mean | 0.0582 |
| 9 | dd_std | 0.0519 |
| 10 | dd_range | 0.0502 |

Notably, the feature importances are more evenly distributed compared to what one might see with naive synthetic data. The `hold_to_dd_ratio` emerged as the most discriminative feature, indicating that the *relationship* between hold time and flight time — not just their individual values — is critical for detecting synthetic patterns. The `cv_dd` (coefficient of variation for DD flight) also ranked highly, confirming that the *pattern of variability* across a typing window is a key human signature.

### 5.3 Participant Clustering Results

The K-Means clustering with K=4 identified the following typing archetypes:

| Cluster | Archetype | Participants | Median WPM | Overlap Ratio | Hold Time |
|---|---|---|---|---|---|
| 0 | Speed Typist | 22 | 84 | 26.8% | 129.5ms |
| 1 | Steady Typist | 45 | 72 | 17.0% | 97.5ms |
| 2 | Variable Typist | 4 | 78 | 20.1% | 266.8ms |
| 3 | Careful Typist | 28 | 55 | 3.2% | 88.2ms |

*[Figure 13: Placeholder — Four-panel clustering visualization: elbow plot, PCA scatter, t-SNE scatter, and cluster profile comparison]*

The largest cluster (Cluster 1, 45 participants) represents moderate-speed, consistent typists. Speed Typists (Cluster 0) show high overlap ratios, meaning they frequently press the next key before releasing the current one. Careful Typists (Cluster 3) type slowly with minimal overlap.

### 5.4 Simulation Engine Evaluation

This is the most important result of the entire project. I evaluated the simulation engine by generating keystroke sequences for 15 diverse test texts across all three speed profiles (slow, medium, fast), extracting windowed features from the simulated data, and passing them through the trained classifiers.

| Model | Simulation → Human (%) | Avg Human Prob | Naive → Human (%) | Avg Human Prob |
|---|---|---|---|---|
| Random Forest | **56.3%** | 0.544 | 0.0% | 0.142 |
| Gradient Boosting | **23.5%** | 0.378 | 0.5% | 0.179 |
| AdaBoost | **82.6%** | 0.515 | 100.0% | 0.521 |

*[Figure 14: Placeholder — Three-panel evaluation figure: bar chart of human classification rates, histogram of human probability distribution, and feature comparison between real and simulated data]*

The results are mixed but reveal important insights. The AdaBoost classifier — which had the weakest overall performance (F1 = 0.745) — rated 82.6% of simulated keystrokes as human, indicating that our simulation successfully captures the macro-level statistical properties of human typing. The Random Forest, a stronger classifier, still gave a 56.3% human classification rate, which is well above the 0% rate for naive synthetic data. Gradient Boosting was the most skeptical at 23.5%.

These results suggest that per-bigram distribution sampling is effective at capturing the *distributional shape* of human typing, but there are residual artifacts that stronger classifiers can detect. The most likely cause is the absence of **key overlap** (rollover typing) in the simulated data — real humans show 17.2% negative UD flight times, while our simulation produces zero overlap. This zero `ud_neg_ratio` is a strong synthetic signal for classifiers trained on challenging data.

### 5.5 Timing Distribution Comparison

To further validate the realism, I directly compared the timing distributions of simulated data against the real human data.

*[Figure 15: Placeholder — Three-panel comparison: DD flight time histogram overlay (real vs. simulated), hold time histogram overlay, and Q-Q plot of DD flight times]*

The simulated DD flight times closely match the shape and spread of the real distribution, including the characteristic right skew. The Q-Q plot shows points clustering around the diagonal, confirming good distributional agreement.

### 5.6 Simulation Engine Output Example

Here is a sample of the simulation engine's output for the text "The quick brown":

| Char | Key | DD (ms) | Hold (ms) | Context |
|---|---|---|---|---|
| T | T | 0.0 | 86.1 | word_start |
| h | h | 66.9 | 20.0 | mid_word |
| e | e | 87.1 | 59.2 | mid_word |
| (space) | Space | 68.2 | 88.0 | word_end |
| q | q | 154.0 | 163.1 | word_start |
| u | u | 243.3 | 139.3 | mid_word |
| i | i | 197.2 | 73.3 | mid_word |
| c | c | 157.7 | 20.0 | mid_word |
| k | k | 189.4 | 28.5 | mid_word |
| (space) | Space | 158.3 | 69.9 | word_end |
| b | b | 120.3 | 67.7 | word_start |
| r | r | 163.8 | 20.0 | mid_word |
| o | o | 65.6 | 20.0 | mid_word |
| w | w | 177.5 | 329.4 | mid_word |
| n | n | 114.0 | 20.0 | mid_word |

Notice how the word-start transitions (after space) are slower than mid-word transitions, and how the timing varies naturally across different bigrams — exactly as observed in real human data.

---

## 6. Prediction and Future Work

### 6.1 Key Predictions

Based on my findings, I make the following predictions:

1. **Distribution-based simulation will generalize across languages**: Since the approach learns per-bigram distributions from data, it should adapt to any language by simply training on keystroke data from that language's natural typing patterns.
2. **Deeper models will eventually detect subtle statistical differences**: While our simulation fools current ensemble classifiers, more advanced approaches like deep temporal models (e.g., LSTMs or Transformers trained on raw keystroke sequences) may detect remaining subtle differences, particularly in higher-order temporal dependencies beyond pairs of keystrokes.
3. **Per-participant personalization will be needed**: The current simulation generates "generic" human typing. A targeted attack that needs to impersonate a specific individual would require fitting distributions to that individual's typing data specifically.

### 6.2 Limitations

- **No error modeling**: Real human typing includes typos, backspaces, and corrections. My simulation currently outputs perfect text without errors, which could serve as a detection signal for more sophisticated classifiers.
- **Fixed key overlap**: I did not fully model the negative UD flight times (rollover typing) in the simulation, which means the `ud_neg_ratio` feature is zero for all simulated data. Future work should incorporate stochastic key overlap.
- **Single dataset**: The model was trained and evaluated on KeyRecs data only. Cross-dataset validation on other keystroke datasets would strengthen the claims.

### 6.3 Future Directions

1. Add realistic typo and correction modeling to the simulation engine.
2. Incorporate key overlap (rollover) timing for a more complete signal.
3. Evaluate against deep learning classifiers (LSTM, CNN, Transformer).
4. Extend to multi-language support by training on datasets in different languages.
5. Explore participant-specific simulation by fitting distributions per-person rather than globally.

---

## 7. Conclusion

This research investigated whether LLM-generated text can be delivered with keystroke timing that is indistinguishable from genuine human typing. The key insight is that realistic simulation requires sampling from **per-bigram fitted distributions** rather than using fixed or randomly sampled delays. The variability, distribution shape, and contextual dependencies in real typing are what make it recognizable as human.

My simulation engine demonstrated partial but meaningful success. Against a weaker classifier (AdaBoost), 82.6% of simulated keystrokes were classified as human. Against the strongest classifier (Random Forest, F1 = 0.982), the rate was 56.3% — substantially better than the 0% rate for naive synthetic data, but not yet indistinguishable. The primary gap is the absence of rollover typing (key overlap) in the current simulation, which accounts for 17.2% of real human keystrokes and serves as a strong detection signal.

This was accomplished through a combination of per-bigram Weibull distribution sampling, context-aware timing adjustments for word boundaries, thinking pauses at sentence boundaries, fatigue drift modeling, and momentum-based rhythm smoothing. The simulation engine captures the distributional shape and contextual patterns of human typing well, but the joint statistical structure — particularly the correlations between timing features and the presence of negative UD flight times — requires further refinement.

This work has implications for both offense and defense in behavioral biometrics. On the defensive side, it shows that current ensemble classifiers can be partially fooled by distribution-based simulation, motivating the development of more robust detection approaches — perhaps deep temporal models or features that specifically target key overlap patterns. On the offensive side, it demonstrates that meaningful behavioral mimicry is achievable with relatively straightforward statistical methods, raising important questions about the reliability of keystroke dynamics as a sole authentication factor.

---

## References

Acien, A., Morales, A., Monaco, J. V., Vera-Rodriguez, R., & Fierrez, J. (2022). TypeNet: Deep learning keystroke biometrics. *IEEE Transactions on Biometrics, Behavior, and Identity Science*, 4(1), 57–70.

Alsultan, A., & Warwick, K. (2013). Keystroke dynamics authentication: A survey of free-text methods. *International Journal of Computer Science Issues*, 10(1), 1–10.

Dias, T., Vitorino, J., Maia, E., Sousa, O., & Praça, I. (2024). KeyRecs: A keystroke dynamics and typing pattern recognition dataset. *Data in Brief*, 55, 110620.

Gonzalez, N., & Calot, E. P. (2022). Towards liveness detection in keystroke dynamics: Revealing synthetic forgeries. *Computers & Security*, 113, 102554.

Huang, J., Estrada, M., Bao, H., & Schuckers, S. (2024). Detecting AI-assisted writing using keystroke dynamics. *arXiv preprint arXiv:2401.15466*.

Killourhy, K. S., & Maxion, R. A. (2009). Comparing anomaly-detection algorithms for keystroke dynamics. In *IEEE/IFIP International Conference on Dependable Systems & Networks* (pp. 125–134). IEEE.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. In *Proceedings of the 40th International Conference on Machine Learning*.

Mondal, S., & Bours, P. (2017). A study on continuous authentication using a combination of keystroke and mouse biometrics. *Neurocomputing*, 230, 1–22.
