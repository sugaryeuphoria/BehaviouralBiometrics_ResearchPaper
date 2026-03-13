
I want you to act like an agent who is really good at Machine Learning and Artificial Intelligence. Act like an expert in ML with access to all your coding abilities.

I will provide a dataset in the form of a **CSV file located in the root directory** called "free-text.csv". I want to research this dataset, apply machine learning algorithms to it, and ultimately accomplish everything as per the user's goal based on it. Consider this your task—do everything end-to-end for me.


---

## Core Workflow Requirements
- You are allowed to **write Python scripts and run them**, observe the terminal outputs, and make decisions based on what you see.
- Start by inspecting the dataset (example: write a Python script to print the **first 50 rows**).
- After analyzing outputs, do planning and decision-making properly following the user's goal.

---

## Decision Log (Mandatory)
- Maintain a **TXT file in the root directory** as a running log.
- Continuously append:
  - what you observed,
  - what decision you took,
  - why you took that decision (rationale),
  - what you will try next.
- Write this in clear, natural language so it can later be used directly in the research paper.

---

## Dynamic + Human-Like Approach (Mandatory)
I want you to work **naturally, like a human researcher would**:
- First: read and understand the dataset
- Then: analyze patterns, issues, and structure
- Then: clean, manipulate, format, and validate the data
- Then: build models, compare approaches, iterate intelligently
- Finally: write a strong conclusion with clear outcomes and evidence

Do **not** lock yourself into a single path. Be exploratory, iterative, and adaptive.

---

## Text / Mathematical Outputs for Decision-Making (Very Important)
Since you cannot visually inspect charts directly, you must produce **text-based or mathematical outputs** that you can read and use to guide decisions.

- These outputs can be in **any form that is useful**, such as:
  - CSV summaries / tables
  - TXT reports
  - JSON outputs
  - metric dumps (accuracy, F1, RMSE, etc.)
  - computed values like correlation coefficients, slopes of lines, trend values, feature importance rankings, etc.
- Store these in: `outputs/texts/`
- Your workflow should explicitly **refer back to these saved outputs** to decide what to try next (example: “the slope indicates a strong upward trend,” or “feature importance shows X dominates,” etc.).

---

## Code Structure (Mandatory)
You must create **6–7 Python files** that organize the full workflow. As per the user's goal, you must follow the structure of the files.
---

## Visuals for Research Paper (High Priority)
- Strongly focus on producing **high-quality visualizations** using **matplotlib** that can be used directly in the research paper.
- I am looking for about 10-15 visualizations.
- Save all generated plots to: `outputs/plots/`

---

## Outputs Folder Structure (Mandatory)
Create and maintain a clean outputs structure like:
- `outputs/plots/` → saved images (paper-ready)
- `outputs/texts/` → text + mathematical outputs (CSV/TXT/JSON/metrics/tables) used for decision-making

---

## Multi-Outcome Requirement (Mandatory)
I want you to come up with great results for this dataset, such as:
- multiple strong model candidates with different strengths numbers for keylogs.
- multiple insights/hypotheses supported by evidence,
- different problem framings (e.g., classification vs regression, clustering insights, forecasting trends) if applicable.

Each outcome should be well-supported by:
- saved plots,
- saved metrics/tables in `outputs/texts/`,
- and clear reasoning in the TXT decision log.

---

## Phase-Based Execution
1. Inspect and understand the data
2. Clean and format it
3. Perform EDA + visuals
4. Build baseline model and draft an implementation plan (to be modified dynamically if needed)
5. Run multiple model experiments + improvements
6. Evaluate, compare, and finalize
7. Produce a final implementation plan and deliver paper-ready outputs + conclusions

IMPORTANT: At each phase, keep updating the root TXT decision log with everything you did, what you learned, and why you chose the next step—so the reasoning can be referenced later in the research paper.  
