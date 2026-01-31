# Swim Meet Simulator

A **stochastic simulator** for high school swimming championships. It uses a Markov performance-state model and Monte Carlo simulation to estimate team score distributions, win probabilities, and individual swimmer/event outcomes.

## What it does

- **Loads seed data** from an MSHSAA-style export (CSV with events, times, schools).
- **Models performance** with a simple Markov chain: each swimmer (or relay) is in a state (Good / Average / Bad) that affects their time, and state can change between events.
- **Adds team momentum**: when a team does better than seed, later swimmers from that team get a small time boost.
- **Optimizes entries** with integer linear programming (PuLP): assigns each swimmer to at most 2 individual events to maximize expected team points.
- **Runs many simulated meets** (Monte Carlo) and collects team total scores and per-event results.
- **Outputs**: distributions of team scores, win and podium probabilities, average team placement, and per-swimmer medal/time distributions.

## How to run it

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   (Key packages: pandas, numpy, streamlit, pulp, seaborn, matplotlib, reportlab.)

2. **Run the notebook**  
   Open `swim_meet_simulator_notebook.ipynb`, run all cells. The notebook loads data, runs the performance model and ILP, then runs Monte Carlo (e.g. 500–1000 simulations). At the end, export cells write gzipped CSVs (e.g. `team_scores.csv.gz`, `swimmer_results.csv.gz`).

3. **Explore results (optional)**  
   - **Streamlit app:** `streamlit run swimmer_explorer.py` — team summaries, score distributions, swimmer stats, medal probabilities.  
   - **PDF report:** `python generate_report.py` — uses `team_scores.csv` or `team_scores.csv.gz` to generate `swim_meet_simulation_report.pdf`.

## Main files

| File | Purpose |
|------|--------|
| `swim_meet_simulator_notebook.ipynb` | Full pipeline: load data, Markov model, ILP, single-meet sim, Monte Carlo, export. |
| `swimmer_explorer.py` | Streamlit app to explore team and swimmer results. |
| `generate_report.py` | Builds a PDF report from team score CSVs. |
| `import_data.ipynb` | Data prep (e.g. class split, schools). |

Input data can come from an MSHSAA export (e.g. `Girls Swimming Export.csv`) or from preprocessed CSVs like `swimming_performance.csv` and `school_class_assignments.csv`.

## Outputs

- **Team scores** (`team_scores.csv` or `team_scores.csv.gz`): one row per team per simulation (total points).
- **Swimmer results** (`swimmer_results.csv` or `swimmer_results.csv.gz`): one row per swimmer-event per simulation (time, place, points).  
Using `.csv.gz` keeps the same data with much smaller file size.

---

Made by Serena.
