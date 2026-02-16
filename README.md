# Swim Meet Simulator

A **stochastic simulator** for high school swimming championships. It uses a Markov performance-state model and Monte Carlo simulation to estimate team score distributions, win probabilities, and individual swimmer/event outcomes.

## What it does

- **Loads seed data** from MSHSAA HTML exports (swimming + dive) or psych sheet PDFs.
- **Models performance** with a Markov chain (Good / Average / Bad) and team momentum.
- **Optimizes entries** with ILP (PuLP) or uses **actual psych sheet** entries.
- **Runs Monte Carlo** and writes all results under `data/processed/` and `results/`.

## Directory layout

- **`data/raw/`** — Raw inputs: `MSHSAA Swimming Performance List.html`, `MSHSAA Dive Performance Listing.html`, `school_class_assignments.csv`, psych sheet PDFs (e.g. `class1psychsheets.pdf`).
- **`data/processed/`** — Processed data: `swimming_performance.csv`, `dive_performance.csv`, `classN_dive.csv`, `classN_psych_entries.csv`, `classN_assignments.csv`.
- **`results/class1_ilp/`**, **`results/class1_psych/`**, **`results/class2_ilp/`**, **`results/class2_psych/`** — Simulation outputs: `team_scores.csv`, `swimmer_results.csv`.
- **`results/reports/`** — PDF reports from `generate_report.py`.

## How to run it

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Import data** (writes to `data/processed/`)
   ```bash
   python import_data.py
   ```
   Place HTML files in `data/raw/` (or project root). Place **school_class_assignments.csv** (columns: School, Class) in `data/raw/` for ILP.

3. **Psych sheets (optional)** — parse PDFs into `data/processed/`
   ```bash
   python psych_sheet_parser.py data/raw/class1psychsheets.pdf
   python psych_sheet_parser.py data/raw/class2psychsheets.pdf
   ```

4. **Run simulations** (reads `data/processed/`, writes `results/`)
   ```bash
   python run_simulations.py --class 1 --mode ilp -n 67
   python run_simulations.py --class 1 --mode psych -n 67
   python run_simulations.py --class 2 --mode ilp -n 67
   python run_simulations.py --class 2 --mode psych -n 67
   ```

5. **Explore results**
   - **Explorer notebook:** Open `class_1_swim_meet_simulator_explorer_notebook.ipynb` — full pipeline with graphics, visualizations, and descriptions (reads `data/processed/` and `data/raw/`, can export to `results/`).
   - **Streamlit app:** `streamlit run swimmer_explorer.py` — team summaries, score distributions, Assignments vs Psych comparison.
   - **PDF report:** `python generate_report.py` (reads from `results/classN_ilp/`, writes to `results/reports/` by default).

## Main files

| File | Purpose |
|------|--------|
| `import_data.py` | Scrape MSHSAA swimming + dive HTML → `data/processed/`. |
| `run_simulations.py` | Run ILP or psych simulation; `--class 1\|2`, `--mode ilp\|psych`, `-n N`. Reads `data/processed/`, writes `results/`. |
| `psych_sheet_parser.py` | Parse psych sheet PDFs → `data/processed/classN_psych_entries.csv`. |
| `class_1_swim_meet_simulator_explorer_notebook.ipynb` | Full pipeline with graphics, visualizations, and descriptions (reads/writes `data/processed/`, `data/raw/`, `results/`). |
| `swimmer_explorer.py` | Streamlit app: team/swimmer results, Assignments vs Psych. |
| `generate_report.py` | Build PDF report; reads `results/classN_ilp/team_scores.csv`, writes `results/reports/`. |

---

Made by Serena.
