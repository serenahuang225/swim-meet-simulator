#!/usr/bin/env python3
"""
Streamlit app to explore swim meet simulation: individual swimmer results and team scores.

Usage:
  1. Run the notebook and export:
       df_scores.to_csv('team_scores.csv', index=False)   # or simulation_results.csv
       df_results.to_csv('swimmer_results.csv', index=False)
  2. Launch: streamlit run swimmer_explorer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(
    page_title="Swim Meet Explorer",
    page_icon="🏊",
    layout="wide",
)

SWIMMER_FILE = "swimmer_results.csv"
TEAM_FILE = "team_scores.csv"  # fallback: simulation_results.csv


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_team_scores():
    """Load team scores from team_scores.csv or simulation_results.csv."""
    for path in [TEAM_FILE, "simulation_results.csv"]:
        try:
            return load_csv(path)
        except FileNotFoundError:
            continue
    return None


def compute_team_stats(df_scores: pd.DataFrame):
    """Compute team summary, win probs, podium probs, placement probs."""
    n_sims = df_scores["simulation_id"].nunique()

    team_stats = (
        df_scores.groupby("team")["points"]
        .agg(["mean", "std", "min", "max", "median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .round(1)
    )
    team_stats.columns = ["mean", "std", "min", "max", "median", "q25", "q75"]
    team_stats = team_stats.sort_values("mean", ascending=False)

    winners = df_scores.loc[df_scores.groupby("simulation_id")["points"].idxmax()]
    win_counts = winners["team"].value_counts()
    win_probs = (win_counts / n_sims * 100).round(2)
    win_df = pd.DataFrame(
        {"team": win_probs.index, "wins": win_counts.values, "win_%": win_probs.values}
    ).sort_values("win_%", ascending=False)

    team_totals = df_scores.groupby(["simulation_id", "team"])["points"].sum().reset_index()
    team_totals["rank"] = team_totals.groupby("simulation_id")["points"].rank(
        ascending=False, method="min"
    )
    team_totals["on_podium"] = (team_totals["rank"] <= 3).astype(int)
    podium = (
        team_totals.groupby("team")
        .agg(podium_prob=("on_podium", "mean"), avg_rank=("rank", "mean"))
        .round(3)
    )
    podium = podium.sort_values("podium_prob", ascending=False)

    place_probs = []
    for team in df_scores["team"].unique():
        sub = team_totals[team_totals["team"] == team]
        place_probs.append({
            "team": team,
            "place_1_%": (sub["rank"] == 1).sum() / n_sims * 100,
            "place_2_%": (sub["rank"] == 2).sum() / n_sims * 100,
            "place_3_%": (sub["rank"] == 3).sum() / n_sims * 100,
            "podium_%": (sub["rank"] <= 3).sum() / n_sims * 100,
        })
    place_df = pd.DataFrame(place_probs).sort_values("place_1_%", ascending=False)

    summary = team_stats.merge(
        win_df.set_index("team")[["win_%", "wins"]],
        left_index=True,
        right_index=True,
        how="left",
    ).fillna(0)
    summary = summary.merge(
        podium[["podium_prob"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    summary = summary.sort_values("mean", ascending=False)

    return {
        "n_sims": n_sims,
        "team_stats": team_stats,
        "summary": summary,
        "podium": podium,
        "place_df": place_df,
        "team_totals": team_totals,
        "df_scores": df_scores,
    }


def compute_swimmer_stats(df: pd.DataFrame) -> pd.DataFrame:
    n_sims = df["simulation_id"].nunique()
    agg = df.groupby(["name", "team", "event", "is_relay"]).agg(
        avg_time=("time", "mean"),
        std_time=("time", "std"),
        min_time=("time", "min"),
        max_time=("time", "max"),
        avg_place=("place", "mean"),
        avg_points=("points", "mean"),
        total_points=("points", "sum"),
        gold_count=("place", lambda x: (x == 1).sum()),
        silver_count=("place", lambda x: (x == 2).sum()),
        bronze_count=("place", lambda x: (x == 3).sum()),
        appearances=("place", "count"),
    ).reset_index()
    agg["gold_prob"] = (agg["gold_count"] / n_sims * 100).round(1)
    agg["silver_prob"] = (agg["silver_count"] / n_sims * 100).round(1)
    agg["bronze_prob"] = (agg["bronze_count"] / n_sims * 100).round(1)
    agg["medal_prob"] = agg["gold_prob"] + agg["silver_prob"] + agg["bronze_prob"]
    for c in ["avg_time", "std_time", "min_time", "max_time", "avg_place", "avg_points"]:
        agg[c] = agg[c].round(2)
    return agg.sort_values(["avg_points", "avg_place"], ascending=[False, True])


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.ss or SS.ss for display."""
    if pd.isna(seconds):
        return ""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
    return f"{seconds:.2f}"


def format_time_series(series: pd.Series) -> pd.Series:
    """Format a Series of seconds as MM:SS.ss for display in tables."""
    return series.apply(format_time)


def render_team_section():
    st.header("🏆 Team Results")
    df_scores = load_team_scores()
    if df_scores is None:
        st.error("Team scores file not found. Export from the notebook:")
        st.code("df_scores.to_csv('team_scores.csv', index=False)\ndf_scores.to_csv('simulation_results.csv', index=False)")
        return

    if "points" not in df_scores.columns or "simulation_id" not in df_scores.columns:
        st.error("CSV must have columns: simulation_id, team, points")
        return

    data = compute_team_stats(df_scores)
    n_sims = data["n_sims"]
    st.caption(f"Based on **{n_sims}** simulations • **{data['df_scores']['team'].nunique()}** teams")

    tab_scores, tab_placement, tab_dist, tab_box = st.tabs([
        "📊 Overall Team Scores",
        "📋 Placement Probabilities",
        "📈 Score Distributions",
        "📦 Box Plot (Top Teams)",
    ])

    with tab_scores:
        st.subheader("Team summary (mean, std, min, max, win %, podium)")
        summary = data["summary"].head(25).reset_index()
        cols = [c for c in ["team", "mean", "std", "min", "max", "median", "win_%", "wins", "podium_prob"] if c in summary.columns]
        st.dataframe(summary[cols].rename(columns={"team": "Team", "mean": "Avg Pts", "std": "Std", "min": "Min", "max": "Max", "median": "Median", "win_%": "Win %", "podium_prob": "Podium Prob"}), use_container_width=True, hide_index=True)

    with tab_placement:
        st.subheader("Placement probabilities (%)")
        pf = data["place_df"].head(20).reset_index(drop=True)
        st.dataframe(pf.rename(columns={"team": "Team", "place_1_%": "1st %", "place_2_%": "2nd %", "place_3_%": "3rd %", "podium_%": "Podium %"}), use_container_width=True, hide_index=True)
        st.markdown("#### Win probability (1st place)")
        top_win = data["place_df"].head(12).sort_values("place_1_%", ascending=True)
        if not top_win.empty:
            st.bar_chart(top_win.set_index("team")["place_1_%"])

    with tab_dist:
        st.subheader("Score distribution by team (top 10)")
        top_teams = data["team_stats"].head(10).index.tolist()
        df_top = data["df_scores"][data["df_scores"]["team"].isin(top_teams)]
        fig, ax = plt.subplots(figsize=(10, 5))
        for team in top_teams:
            pts = df_top[df_top["team"] == team]["points"].values
            ax.hist(pts, bins=min(30, max(5, len(pts)//10)), alpha=0.5, label=team, density=True)
        ax.set_xlabel("Total points")
        ax.set_ylabel("Density")
        ax.set_title(f"Score distribution (top 10 teams, {n_sims} sims)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_box:
        st.subheader("Box plot: score distribution (top 10 teams)")
        top_teams_box = data["team_stats"].head(10).index.tolist()
        df_top_box = data["df_scores"][data["df_scores"]["team"].isin(top_teams_box)].copy()
        df_top_box["team"] = pd.Categorical(df_top_box["team"], categories=top_teams_box, ordered=True)
        df_top_box = df_top_box.sort_values("team")
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = []
        data_list = []
        for i, team in enumerate(top_teams_box):
            pts = df_top_box[df_top_box["team"] == team]["points"].values
            data_list.append(pts)
            positions.append(i + 1)
        bp = ax.boxplot(data_list, positions=positions, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_xticks(positions)
        ax.set_xticklabels(top_teams_box, rotation=45, ha="right")
        ax.set_ylabel("Total points")
        ax.set_title(f"Score distribution — top 10 teams ({n_sims} simulations)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_swimmer_section():
    st.header("🏊 Swimmer Explorer")
    try:
        df = load_csv(SWIMMER_FILE)
    except FileNotFoundError:
        st.error(f"**File not found:** `{SWIMMER_FILE}`")
        st.info("Export from the notebook: `df_results.to_csv('swimmer_results.csv', index=False)`")
        return

    n_sims = df["simulation_id"].nunique()
    st.caption(f"Data from **{n_sims}** simulations • {len(df):,} total results")

    st.sidebar.header("Filters")
    all_teams = sorted(df["team"].unique())
    selected_teams = st.sidebar.multiselect("Team(s)", all_teams, default=[])
    all_events = sorted(df["event"].unique())
    selected_events = st.sidebar.multiselect("Event(s)", all_events, default=[])
    show_relays = st.sidebar.checkbox("Include Relays", value=True)
    show_individual = st.sidebar.checkbox("Include Individual", value=True)
    swimmer_search = st.sidebar.text_input("Search swimmer")

    filtered = df.copy()
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]
    if selected_events:
        filtered = filtered[filtered["event"].isin(selected_events)]
    if not show_relays:
        filtered = filtered[filtered["is_relay"] == False]
    if not show_individual:
        filtered = filtered[filtered["is_relay"] == True]
    if swimmer_search:
        filtered = filtered[filtered["name"].str.contains(swimmer_search, case=False, na=False)]

    if filtered.empty:
        st.warning("No results match the filters.")
        return

    stats = compute_swimmer_stats(filtered)
    tab1, tab2, tab3 = st.tabs(["📊 Swimmer Stats", "🏅 Medal Probabilities", "📈 Distributions"])

    with tab1:
        st.subheader("Swimmer performance summary")
        sort_col = st.selectbox(
            "Sort by",
            ["avg_points", "avg_place", "medal_prob", "gold_prob", "avg_time"],
            format_func=lambda x: {"avg_points": "Avg Points", "avg_place": "Avg Place", "medal_prob": "Medal %", "gold_prob": "Gold %", "avg_time": "Avg Time"}.get(x, x),
        )
        ascending = sort_col in ["avg_place", "avg_time"]
        display_stats = stats.sort_values(sort_col, ascending=ascending)
        # Format avg_time as MM:SS.ss for display
        display_df = display_stats[
            ["name", "team", "event", "avg_time", "avg_place", "avg_points", "gold_prob", "silver_prob", "bronze_prob", "medal_prob"]
        ].copy()
        display_df["avg_time"] = format_time_series(display_df["avg_time"])
        st.dataframe(
            display_df.rename(columns={
                "name": "Swimmer", "team": "Team", "event": "Event", "avg_time": "Avg Time", "avg_place": "Avg Place",
                "avg_points": "Avg Pts", "gold_prob": "Gold %", "silver_prob": "Silver %", "bronze_prob": "Bronze %", "medal_prob": "Medal %",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.subheader("Top medal contenders")
        top_gold = stats[stats["is_relay"] == False].nlargest(15, "gold_prob")
        if not top_gold.empty:
            st.bar_chart(top_gold.set_index("name")["gold_prob"])
        top_medal = stats[stats["is_relay"] == False].nlargest(15, "medal_prob")
        if not top_medal.empty:
            st.bar_chart(top_medal.set_index("name")["medal_prob"])

    with tab3:
        st.subheader("Time / place distributions")
        swimmer_event_options = stats[["name", "event"]].drop_duplicates()
        swimmer_event_options["label"] = swimmer_event_options["name"] + " — " + swimmer_event_options["event"]
        selected_label = st.selectbox("Swimmer + Event", swimmer_event_options["label"].tolist())
        if selected_label:
            sel_name, sel_event = selected_label.split(" — ", 1)
            subset = filtered[(filtered["name"] == sel_name) & (filtered["event"] == sel_event)]
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(subset["time"].value_counts().sort_index())
            with c2:
                st.bar_chart(subset["place"].value_counts().sort_index())
            st.metric("Avg Time", format_time(subset["time"].mean()))
            st.metric("Avg Place", f"{subset['place'].mean():.1f}")


def main():
    st.title("🏊 Swim Meet Simulation Explorer")
    st.markdown("Explore **team scores** and **individual swimmer** results from Monte Carlo simulations.")

    mode = st.sidebar.radio(
        "View",
        ["🏆 Team Results", "🏊 Swimmer Explorer"],
        label_visibility="collapsed",
    )

    if mode == "🏆 Team Results":
        render_team_section()
    else:
        render_swimmer_section()

    st.markdown("---")
    st.caption("Stochastic Swim Meet Simulator • Export team_scores.csv & swimmer_results.csv from the notebook")


if __name__ == "__main__":
    main()
