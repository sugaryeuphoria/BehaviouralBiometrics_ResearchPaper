# Image Generation Prompt: Research Pipeline Diagram

**Purpose:** Use this prompt with an AI image generation model (e.g., DALL-E, Midjourney, Stable Diffusion) to produce a clean, professional pipeline/workflow diagram suitable for inclusion in a LaTeX research paper.

---

## Prompt

Generate a professional, clean research pipeline diagram illustrating the workflow of a keystroke dynamics research project. The diagram should be rendered in a polished, flat-design infographic style with crisp edges, consistent iconography, and a white background suitable for inclusion in an academic research paper. The overall layout should flow from left to right (or top to bottom) in a clear, linear sequence with labeled stages connected by directional arrows, using a modern sans-serif font for all text labels.

The pipeline begins with a **Dataset Ingestion** stage on the far left, depicted as a document or database icon labeled "KeyRecs Dataset" with a subtitle "562K raw keystrokes from 99 participants." An arrow leads to the next stage, **Data Preprocessing**, shown as a funnel or filter icon labeled "Cleaning & Filtering" with annotations like "Outlier removal," "Null handling," and "Derived features," and a note beneath it saying "→ 559K cleaned rows." A second arrow connects this to the **Exploratory Data Analysis (EDA)** stage, visualized as a set of small chart icons (histogram, heatmap, scatter plot) with the label "Statistical Analysis" and bullet points including "Timing distributions," "Bigram patterns," "Word boundary effects," and "Participant variation."

The next stage is **Feature Engineering & Distribution Modeling**, depicted as a set of interlocking gears or a mathematical function symbol, labeled "Per-Bigram Distribution Fitting" with annotations such as "Weibull / Log-normal / Gamma fits," "Context-dependent parameters," and "100+ bigram profiles." An arrow connects it to the **Machine Learning Classification** stage, shown as a brain or neural network icon labeled "Human vs. Synthetic Detection" with three model labels beneath it — "Random Forest," "Gradient Boosting," and "AdaBoost" — along with a small bar chart showing performance metrics. The pipeline then flows to the **Simulation Engine** stage, the central and largest element of the diagram, depicted as a prominent engine or cog icon with glowing emphasis, labeled "Human Keystroke Simulator" with annotations like "Distribution sampling," "Context-aware timing," "Fatigue drift," and "Momentum smoothing."

From the simulation engine, two branches emerge. One arrow points to the **Evaluation** stage, shown as a checklist or verification icon labeled "ML Model Evaluation" with a key result annotation "99.5–100% classified as human." The other arrow points to the **Web Interface** stage, depicted as a browser or monitor icon labeled "Interactive Demo" with a subtitle "Real-time typing simulation." Beneath the entire pipeline, include a thin horizontal bar spanning the full width, labeled "Decision Log — Transparent Research Process," indicating that every decision is documented throughout all stages. The color scheme should use the official Thompson Rivers University (TRU) brand palette: deep navy blue (#003E51) as the primary color for headers and major elements, teal (#00B0B9) for secondary elements and connecting arrows, sage green (#BAD1BA) for background accents or subtle fills, warm yellow (#FFCD00) for highlights and key result callouts, and orange (#F88F23) sparingly for emphasis on the most important findings. The overall aesthetic should feel academic yet modern — similar to a Nature or IEEE journal figure — with no unnecessary decoration, a generous use of whitespace, and a balanced composition that guides the reader's eye through the full research workflow from data to results.

---

## Style Notes

- **Background:** Pure white (#FFFFFF) — no dark mode
- **Primary font:** Sans-serif (Helvetica, Inter, or similar)
- **Icon style:** Flat, minimalist with consistent stroke weight
- **Arrow style:** Clean, directional, with subtle teal (#00B0B9) color
- **Emphasis:** The Simulation Engine should be visually prominent (larger, slightly glowing border)
- **Dimensions:** Landscape orientation, approximately 16:9 aspect ratio (suitable for full-page-width in a LaTeX document)
- **Resolution:** High resolution (at least 300 DPI) for print quality

## TRU Color Palette Reference

| Color        | Hex       | Usage                                   |
|-------------|-----------|------------------------------------------|
| TRU Blue    | `#003E51` | Headers, primary elements, text labels   |
| TRU Teal    | `#00B0B9` | Arrows, secondary elements, borders      |
| TRU Sage    | `#BAD1BA` | Background accents, subtle fills         |
| TRU Yellow  | `#FFCD00` | Highlights, key results, callout badges  |
| TRU Orange  | `#F88F23` | Sparingly for emphasis, important metrics |
| White       | `#FFFFFF` | Background                               |
| Dark Text   | `#1A1A1A` | Body text, labels                        |
