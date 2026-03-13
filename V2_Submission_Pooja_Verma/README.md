# Keystroke Dynamics: Human Typing Simulation & Detection
**Course:** COMP 4980 — Behavioral Biometrics  
**Author:** Pooja Verma (T00729545)  
**Professor:** Thompson Rivers University  

---

## Project Overview
This project investigates the feasibility of simulating human-like keystroke timing for AI-generated text. The goal is to create a simulation engine that can "fool" machine learning classifiers by mimicking the statistical distributions and contextual patterns of real human typists.

### Key Research Questions
1. Can we statistically model the "rhythm" of human typing at the per-bigram level?
2. How effective are ensemble classifiers at detecting synthetic timing patterns?
3. Can a distribution-based simulation engine bypass these detectors?

---

## Folder Structure

- `01_data_preprocessing.py`: Cleans raw KeyRecs data and handles outliers.
- `02_eda.py`: Comprehensive Exploratory Data Analysis with TRU-themed visualizations.
- `03_feature_engineering.py`: Extracts 19 statistical features over 20-keystroke windows.
- `04_distribution_modeling.py`: Fits Weibull/Log-normal distributions per bigram.
- `05_ml_models.py`: Trains Random Forest, Gradient Boosting, and AdaBoost classifiers.
- `06_clustering.py`: Identifies four distinct typing archetypes using K-Means.
- `07_simulation_engine.py`: The core human keystroke simulator.
- `08_evaluation.py`: Evaluates the simulation against the trained detectors.
- `10_final_report.py`: Generates summary tables and metrics.
- `09_web_interface/`: A Flask-based web demo for real-time simulation testing.
- `Data/`: Contains the cleaned `free-text.csv` dataset.
- `outputs/`: Pre-generated plots, trained models, and statistical reports.
- `tru_theme.py`: Global styling configuration using official TRU brand colors.

---

## How to Run

### 1. Environment Setup
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running the Full Pipeline
You can regenerate all results and plots by running:
```bash
python3 regenerate_all.py
```
This script will:
- Load the data
- Train the models against realistic synthetic data
- Generate all 16 research plots in the `outputs/plots` folder

### 3. Launching the Web Interface
To see the simulation in action:
```bash
cd 09_web_interface
python3 app.py
```
Then open your browser to the local address provided.

---

## Key Findings
- **Realistic Detection**: By using challenging synthetic data (statistical mimics and perturbed human samples), we achieved an F1-score of **0.982** and AUC of **0.999** with Random Forest.
- **Simulation Efficacy**: Our simulation engine successfully bypassed weaker classifiers (82% human rate on AdaBoost) but shown susceptibility to stronger ones (56% on RF) due to the absence of key overlap (rollover) modeling.
- **Contextual Importance**: Analysis confirmed that word boundaries and specific bigram transitions are the most significant factors in human-like timing rhythm.

---
Thank you for reviewing my project!
