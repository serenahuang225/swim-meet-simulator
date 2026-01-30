#!/usr/bin/env python3
"""Add subsection markdown cells and fix ILP cell in swim_meet_simulator_notebook.ipynb."""
import json

NOTEBOOK_PATH = "swim_meet_simulator_notebook.ipynb"

with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# 1) Fix ILP markdown cell: replace long paragraph with short bullets
for i, c in enumerate(cells):
    if c["cell_type"] != "markdown" or not c.get("source"):
        continue
    src = "".join(c["source"])
    if "## 4) Event optimization (ILP)" in src and "Objective: maximize total expected" in src:
        c["source"] = [
            "## 4) Event optimization (ILP)\n",
            "\n",
            "Use **PuLP** to assign each swimmer to at most 2 individual events to maximize expected team points.\n",
            "\n",
            "- **Objective:** Maximize sum of expected points (from seed rank).\n",
            "- **Variables:** Binary \\(x[swimmer, event]\\) = 1 if swimmer enters that event.\n",
            "- **Constraints:** Each swimmer ≤ 2 events; each event has 16 swimmers; conflict pairs (e.g. 200 Free & 200 IM) mutually exclusive.\n",
        ]
        print(f"Fixed ILP cell at index {i}")
        break

# 2) Insert new markdown cells (insert in reverse order so indices stay valid)
def mk_md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

inserts = [
    (8, ["### 1.1 Class split and data prep", "", "Load performance data by class, merge school/class info, reseed ranks within class+event, and filter to Class 1 for simulation."]),
    (31, ["### 6.1 Event assignments", "", "Build individual event assignments (ILP) and relay assignments from the result DataFrame."]),
    (33, ["### 6.2 Run Monte Carlo (quick check)", "", "Run a small number of simulations (e.g. 100) to inspect outputs before a full run."]),
    (36, ["### 6.4 Team score statistics and win probabilities", "", "Summarize team score distributions and compute win and placement probabilities from `df_scores`."]),
    (50, ["### 6.6 Individual swimmer & medal probabilities", "", "From `df_results`, compute medal (top 3) and event-strength probabilities per swimmer/event."]),
    (55, ["### 6.8 Team points breakdown", "", "Break down team points into individual vs relay contribution (relays count double)."]),
]

# Insert in reverse order by index so we don't shift earlier indices
inserts_sorted = sorted(inserts, key=lambda x: -x[0])
for insert_at, lines in inserts_sorted:
    new_cell = mk_md(lines)
    cells.insert(insert_at, new_cell)
    print(f"Inserted markdown at index {insert_at}: {lines[0][:50]}...")

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=2)

print("Done.")
