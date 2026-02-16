#!/usr/bin/env python3
"""
Run swim meet simulations: ILP (optimized assignments) or Psych (actual entries).
Reads from data/processed/, writes to results/ (results/classN_ilp/ or results/classN_psych/).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from tqdm import tqdm

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
POINTS = [20, 17, 16, 15, 14, 13, 12, 11, 9, 7, 6, 5, 4, 3, 2, 1]
PERF_STATES = ["Good", "Average", "Bad"]
transition_matrix = {
    "Good": {"Good": 0.7, "Average": 0.25, "Bad": 0.05},
    "Average": {"Good": 0.15, "Average": 0.7, "Bad": 0.15},
    "Bad": {"Good": 0.05, "Average": 0.25, "Bad": 0.7},
}
sigma_mult = {"Good": 0.005, "Average": 0.01, "Bad": 0.02}


def convert_swim_time(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return np.nan, ""
    time_str = time_str.strip()
    if not time_str:
        return np.nan, ""
    try:
        parts = time_str.split(":")
        if len(parts) == 1:
            seconds = float(parts[0])
            formatted = f"{int(seconds // 60)}:{seconds % 60:05.2f}" if seconds >= 60 else f"{seconds:.2f}"
            return seconds, formatted
        if len(parts) == 2:
            total_seconds = float(parts[0]) * 60 + float(parts[1])
            formatted = f"{int(parts[0])}:{float(parts[1]):05.2f}" if float(parts[0]) > 0 else f"{float(parts[1]):.2f}"
            return total_seconds, formatted
        if len(parts) == 3:
            total_seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return total_seconds, time_str
    except (ValueError, AttributeError):
        pass
    return np.nan, str(time_str)


def add_time_seconds_column(df, time_col="best_time"):
    results = df[time_col].apply(lambda x: convert_swim_time(x) if pd.notna(x) else (np.nan, ""))
    df = df.copy()
    df[time_col] = [r[0] for r in results]
    df["time_formatted"] = [r[1] for r in results]
    return df


def seed_to_points(rank):
    if rank <= 16:
        return POINTS[rank - 1]
    return 16 / rank


def optimize_event_assignments(df, max_events_per_swimmer=2, swimmers_per_event=32, max_swimmers_per_school_per_event=4):
    df = df.copy().drop_duplicates(subset=["name", "event"]).reset_index(drop=True)
    df["expected_points"] = df["seed_rank"].apply(seed_to_points)
    df = df.sort_values(["name", "expected_points", "seed_rank", "event"], ascending=[True, False, True, True]).reset_index(drop=True)
    swimmers = df["name"].unique()
    events = df["event"].unique()
    model = pulp.LpProblem("SwimMeetOptimization", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ((r["name"], r["event"]) for _, r in df.iterrows()), cat="Binary")
    model += pulp.lpSum(r["expected_points"] * x[(r["name"], r["event"])] for _, r in df.iterrows())
    for s in swimmers:
        events_for_s = df[df["name"] == s]["event"].unique()
        model += pulp.lpSum(x[(s, e)] for e in events_for_s) <= max_events_per_swimmer
    if "team" in df.columns:
        for e in events:
            event_df = df[df["event"] == e]
            for team in event_df["team"].unique():
                team_rows = event_df[event_df["team"] == team]
                model += pulp.lpSum(x[(r["name"], e)] for _, r in team_rows.iterrows()) <= max_swimmers_per_school_per_event
    for e in events:
        event_swimmers = df[df["event"] == e]
        if "team" in df.columns:
            max_possible = sum(
                min(max_swimmers_per_school_per_event, len(event_swimmers[event_swimmers["team"] == t]))
                for t in event_swimmers["team"].unique()
            )
            target = min(swimmers_per_event, max_possible, len(event_swimmers))
        else:
            target = min(swimmers_per_event, len(event_swimmers))
        model += pulp.lpSum(x[(r["name"], e)] for _, r in event_swimmers.iterrows()) == target
    conflict_pairs = [
        ("200 Free", "200 IM"), ("200 Freestyle", "200 IM"),
        ("50 Free", "200 IM"), ("50 Freestyle", "200 IM"),
        ("100 Free", "100 Fly"), ("100 Freestyle", "100 Butterfly"),
        ("100 Back", "100 Breast"), ("100 Backstroke", "100 Breaststroke"),
        ("500 Free", "100 Free"), ("500 Freestyle", "100 Freestyle"),
    ]
    for s in swimmers:
        swimmer_events = set(df[df["name"] == s]["event"].unique())
        for e1, e2 in conflict_pairs:
            if e1 in swimmer_events and e2 in swimmer_events:
                model += x[(s, e1)] + x[(s, e2)] <= 1
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = pd.DataFrame([r for _, r in df.iterrows() if pulp.value(x[(r["name"], r["event"])]) > 0.5])
    selected = selected.sort_values(["event", "best_time", "name"]).reset_index(drop=True)
    selected["new_seed_rank"] = selected.groupby("event").cumcount() + 1
    return selected


def step_state(current_state):
    probs = list(transition_matrix[current_state].values())
    return np.random.choice(list(transition_matrix[current_state].keys()), p=probs)


def sample_time(best_time, state):
    sigma = best_time * sigma_mult[state]
    return np.random.normal(best_time, sigma)


def apply_team_momentum(team_momentum, base_time, team, momentum_factor=0.001):
    momentum = team_momentum.get(team, 0.0)
    return base_time - (momentum * momentum_factor)


def simulate_one_meet(df, assignments, relay_assignments=None, dive_df=None):
    swimmer_states = {name: np.random.choice(PERF_STATES, p=[0.2, 0.6, 0.2]) for name in df[df["is_relay"] == False]["name"].unique()}
    team_relay_states = {team: np.random.choice(PERF_STATES, p=[0.2, 0.6, 0.2]) for team in df[df["is_relay"] == True]["team"].unique()}
    results = []
    team_momentum = {}
    events_from_df = df["event"].unique().tolist()
    events_from_assignments = sorted(set(e for evs in assignments.values() for e in evs))
    events_order = list(events_from_df)
    for e in events_from_assignments:
        if e not in events_order:
            events_order.append(e)
    for event in events_order:
        event_rows = df[df["event"] == event]
        is_relay = event_rows["is_relay"].iloc[0] if len(event_rows) > 0 else False
        if is_relay:
            entrants = [t for t, evs in relay_assignments.items() if event in evs] if relay_assignments else event_rows["team"].unique().tolist()
            rows = df[df["team"].isin(entrants) & (df["event"] == event)]
            sim_rows = []
            for _, r in rows.iterrows():
                team, base = r["team"], r["best_time"]
                cur = team_relay_states.get(team, "Average")
                nxt = step_state(cur)
                team_relay_states[team] = nxt
                adj = apply_team_momentum(team_momentum, base, team)
                sim_rows.append({"name": team, "team": team, "event": event, "time": sample_time(adj, nxt), "state": nxt, "is_relay": True})
        else:
            entrants = [s for s, evs in assignments.items() if event in evs]
            rows = df[df["name"].isin(entrants) & (df["event"] == event)]
            if rows.empty and entrants:
                ev_in_df = [e for e in df["event"].unique() if str(e).strip() == str(event).strip()]
                if ev_in_df:
                    rows = df[df["name"].isin(entrants) & (df["event"] == ev_in_df[0])]
            sim_rows = []
            for _, r in rows.iterrows():
                name, team, base = r["name"], r["team"], r["best_time"]
                cur = swimmer_states.get(name, "Average")
                nxt = step_state(cur)
                swimmer_states[name] = nxt
                adj = apply_team_momentum(team_momentum, base, team)
                sim_rows.append({"name": name, "team": team, "event": event, "time": sample_time(adj, nxt), "state": nxt, "is_relay": False})
        if not sim_rows:
            continue
        sim_df = pd.DataFrame(sim_rows).sort_values("time").reset_index(drop=True)
        sim_df["place"] = sim_df.index + 1
        sim_df["points"] = sim_df["place"].apply(lambda p: POINTS[p - 1] * 2 if is_relay and p <= len(POINTS) else (POINTS[p - 1] if p <= len(POINTS) else 0))
        for _, row in sim_df.iterrows():
            seed_row = df[(df["name"] == row["name"]) & (df["event"] == event)]
            if not seed_row.empty:
                seed_rank = int(seed_row["seed_rank"].iloc[0])
                unexpected = max(0, seed_rank - row["place"])
                team_momentum[row["team"]] = team_momentum.get(row["team"], 0.0) + unexpected * 0.2
        results.append(sim_df)
    if dive_df is not None and not dive_df.empty:
        d = dive_df.copy()
        d["noisy_score"] = d["total_score"] + np.random.normal(0, d["total_score"].mean() * 0.025, size=len(d))
        d = d.sort_values("noisy_score", ascending=False).reset_index(drop=True)
        d["place"] = d.index + 1
        d["points"] = d["place"].apply(lambda p: POINTS[p - 1] if p <= len(POINTS) else 0)
        results.append(pd.DataFrame([{"name": row["name"], "team": row["team"], "event": "Diving", "time": row["total_score"], "state": "Average", "is_relay": False, "place": row["place"], "points": row["points"]} for _, row in d.iterrows()]))
    full = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["name", "team", "event", "time", "place", "points", "is_relay"])
    return full, full.groupby("team")["points"].sum().to_dict()


def run_monte_carlo(df, assignments, relay_assignments=None, dive_df=None, n_sims=500):
    team_rows = []
    all_results = []
    for i in tqdm(range(n_sims), desc="Monte Carlo"):
        full, team_scores = simulate_one_meet(df, assignments, relay_assignments, dive_df)
        full = full.copy()
        full["simulation_id"] = i
        all_results.append(full)
        for team, score in team_scores.items():
            team_rows.append({"simulation_id": i, "team": team, "points": score})
    return pd.DataFrame(team_rows), pd.concat(all_results, ignore_index=True)


def run_ilp(class_num: str, data_dir: Path, results_dir: Path, n_sims: int, no_dive: bool):
    swimmers = pd.read_csv(data_dir / "swimming_performance.csv")
    # school_class_assignments can be in data/raw or data/processed
    schools_path = data_dir / "school_class_assignments.csv"
    if not schools_path.exists():
        schools_path = Path("data/raw") / "school_class_assignments.csv"
    if not schools_path.exists():
        raise SystemExit(f"Missing school_class_assignments.csv. Add to data/processed/ or data/raw/.")
    schools = pd.read_csv(schools_path.resolve())
    swimmers = swimmers.merge(schools[["School", "Class"]], left_on="team", right_on="School", how="left", validate="m:1").drop(columns=["School"])
    swimmers = add_time_seconds_column(swimmers)
    swimmers = swimmers.sort_values(["Class", "event", "best_time", "name"], ascending=[True, True, True, True])
    swimmers["seed_rank"] = swimmers.groupby(["Class", "event"]).cumcount() + 1
    class_swimmers = swimmers[swimmers["Class"] == int(class_num)].copy()
    class_swimmers = class_swimmers[pd.to_numeric(class_swimmers["best_time"], errors="coerce").notna()]
    class_swimmers["best_time"] = class_swimmers["best_time"].astype(float)

    assignments_df = optimize_event_assignments(class_swimmers[class_swimmers["is_relay"] == False])
    out_subdir = results_dir / f"class{class_num}_ilp"
    out_subdir.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(data_dir / f"class{class_num}_assignments.csv", index=False)
    assignments = assignments_df.groupby("name")["event"].apply(list).to_dict()
    relay_assignments = {}
    for team in class_swimmers[class_swimmers["is_relay"] == True]["team"].unique():
        evs = class_swimmers[(class_swimmers["team"] == team) & (class_swimmers["is_relay"] == True)]["event"].tolist()
        if evs:
            relay_assignments[team] = evs

    dive_df = None
    if not no_dive:
        for p in [data_dir / f"class{class_num}_dive.csv", data_dir / "dive_performance.csv"]:
            if p.exists():
                dive_df = pd.read_csv(p)
                if p.name == "dive_performance.csv":
                    dive_df = dive_df[dive_df["class"].astype(str) == class_num]
                if "total_score" not in dive_df.columns and "score" in dive_df.columns:
                    dive_df["total_score"] = dive_df["score"]
                break

    df_scores, df_results = run_monte_carlo(class_swimmers, assignments, relay_assignments, dive_df=dive_df, n_sims=n_sims)
    df_scores.to_csv(out_subdir / "team_scores.csv", index=False)
    df_results.to_csv(out_subdir / "swimmer_results.csv", index=False)
    print(f"Wrote {out_subdir / 'team_scores.csv'}, {out_subdir / 'swimmer_results.csv'}")


def run_psych(class_num: str, data_dir: Path, results_dir: Path, n_sims: int, no_dive: bool):
    psych_path = data_dir / f"class{class_num}_psych_entries.csv"
    if not psych_path.exists():
        raise SystemExit(f"Psych entries not found: {psych_path}. Run psych_sheet_parser.py and save to data/.")
    df = pd.read_csv(psych_path)
    df = df[df["event"] != "Diving"]
    df = df[pd.to_numeric(df["best_time"], errors="coerce").notna()].copy()
    df["best_time"] = df["best_time"].astype(float)
    assignments = df[df["is_relay"] == False].groupby("name")["event"].apply(list).to_dict()
    relay_assignments = {}
    for team in df[df["is_relay"] == True]["team"].unique():
        evs = df[(df["team"] == team) & (df["is_relay"] == True)]["event"].tolist()
        if evs:
            relay_assignments[team] = evs

    dive_df = None
    if not no_dive:
        for p in [data_dir / f"class{class_num}_dive.csv", data_dir / "dive_performance.csv"]:
            if p.exists():
                dive_df = pd.read_csv(p)
                if p.name == "dive_performance.csv":
                    dive_df = dive_df[dive_df["class"].astype(str) == class_num]
                if "total_score" not in dive_df.columns and "score" in dive_df.columns:
                    dive_df["total_score"] = dive_df["score"]
                break

    out_subdir = results_dir / f"class{class_num}_psych"
    out_subdir.mkdir(parents=True, exist_ok=True)
    df_scores, df_results = run_monte_carlo(df, assignments, relay_assignments, dive_df=dive_df, n_sims=n_sims)
    df_scores.to_csv(out_subdir / "team_scores.csv", index=False)
    df_results.to_csv(out_subdir / "swimmer_results.csv", index=False)
    print(f"Wrote {out_subdir / 'team_scores.csv'}, {out_subdir / 'swimmer_results.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Run swim meet simulation (ILP or Psych). Reads data/, writes results/.")
    parser.add_argument("--class", dest="class_num", choices=["1", "2"], required=True, help="Class 1 or 2")
    parser.add_argument("--mode", choices=["ilp", "psych"], required=True, help="ILP (optimized) or psych (actual entries)")
    parser.add_argument("-n", "--n-sims", type=int, default=500, help="Number of Monte Carlo runs")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Data directory")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Results directory")
    parser.add_argument("--no-dive", action="store_true", help="Skip diving")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "ilp":
        run_ilp(args.class_num, args.data_dir, args.results_dir, args.n_sims, args.no_dive)
    else:
        run_psych(args.class_num, args.data_dir, args.results_dir, args.n_sims, args.no_dive)


if __name__ == "__main__":
    main()
