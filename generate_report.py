#!/usr/bin/env python3
"""
Generate a PDF report from swim meet simulation results.

Usage:
  1. Run the swim_meet_simulator_notebook.ipynb and ensure the Monte Carlo cell
     has been executed (df_scores is in memory).
  2. Export results to CSV by running in the notebook:
       df_scores.to_csv('team_scores.csv', index=False)
  3. Run this script:
       python generate_report.py [path/to/team_scores.csv] [-o output.pdf]
       python generate_report.py -c 1          # Class 1 (default)
       python generate_report.py -c 2          # Class 2 → class2_team_scores.csv, ..._class2.pdf
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def load_and_validate_csv(path: Path) -> pd.DataFrame:
    """Load simulation results CSV; must have simulation_id, team, points. Tries .csv.gz if .csv missing."""
    p = Path(path)
    if not p.exists():
        p = Path(str(p).replace(".csv", "") + ".csv.gz")
    if not p.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        print("\nTo create it, run the swim meet simulator notebook, then export (optionally gzipped):",
              file=sys.stderr)
        print("  df_scores.to_csv('team_scores.csv', index=False)", file=sys.stderr)
        print("  # or: df_scores.to_csv('team_scores.csv.gz', index=False, compression='gzip')", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(p)
    for col in ['simulation_id', 'team', 'points']:
        if col not in df.columns:
            print(f"Error: CSV must have column '{col}'. Found: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
    return df


def compute_summary_tables(df_scores: pd.DataFrame):
    """Compute team_stats, summary (with win %), podium_probs, and place probabilities."""
    n_sims = df_scores['simulation_id'].nunique()

    team_stats = df_scores.groupby('team')['points'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
    ]).round(1)
    team_stats = team_stats.sort_values('mean', ascending=False)

    winners = df_scores.loc[df_scores.groupby('simulation_id')['points'].idxmax()]
    win_counts = winners['team'].value_counts()
    win_probs = (win_counts / n_sims * 100).round(2)
    win_prob_df = pd.DataFrame({
        'team': win_probs.index,
        'wins': win_counts.values,
        'win_probability_%': win_probs.values,
    }).sort_values('win_probability_%', ascending=False)

    summary = team_stats.merge(
        win_prob_df.set_index('team')[['win_probability_%', 'wins']],
        left_index=True, right_index=True, how='left'
    ).fillna(0).sort_values('mean', ascending=False)

    team_totals = df_scores.groupby(['simulation_id', 'team'])['points'].sum().reset_index()
    team_totals['rank'] = team_totals.groupby('simulation_id')['points'].rank(
        ascending=False, method='min'
    )
    team_totals['on_podium'] = (team_totals['rank'] <= 3).astype(int)
    podium_probs = team_totals.groupby('team').agg({
        'on_podium': 'mean',
        'rank': 'mean',
    }).round(3)
    podium_probs.columns = ['podium_probability', 'avg_rank']
    podium_probs = podium_probs.sort_values('podium_probability', ascending=False)

    # Place probabilities: rank is already in team_totals per (sim, team)
    place_results = []
    for team in df_scores['team'].unique():
        subset = team_totals[team_totals['team'] == team]
        if subset.empty:
            place_results.append({
                'team': team, 'place_1%': 0, 'place_2%': 0, 'place_3%': 0, 'place_4%': 0,
            })
            continue
        place_results.append({
            'team': team,
            'place_1%': (subset['rank'] == 1).sum() / n_sims * 100,
            'place_2%': (subset['rank'] == 2).sum() / n_sims * 100,
            'place_3%': (subset['rank'] == 3).sum() / n_sims * 100,
            'place_4%': (subset['rank'] == 4).sum() / n_sims * 100,
        })
    prob_df = pd.DataFrame(place_results)
    prob_df['podium%'] = (
        prob_df['place_1%'] + prob_df['place_2%'] +
        prob_df['place_3%'] + prob_df['place_4%']
    )
    prob_df = prob_df.sort_values(['place_1%', 'podium%'], ascending=False).reset_index(drop=True)

    return {
        'n_sims': n_sims,
        'n_teams': df_scores['team'].nunique(),
        'team_stats': team_stats,
        'summary': summary,
        'podium_probs': podium_probs,
        'prob_df': prob_df,
    }


def plot_win_probability(prob_df: pd.DataFrame, n_sims: int, out_path: Path) -> None:
    """Top 12 teams by 1st-place probability (horizontal bar)."""
    top = prob_df.sort_values('place_1%', ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(top))
    colors_bar = plt.cm.YlOrBr((top['place_1%'].values / max(top['place_1%'].max(), 0.01)) * 0.8 + 0.2)
    bars = ax.barh(y_pos, top['place_1%'], color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['team'], fontsize=9)
    ax.set_xlabel('Probability (%)')
    ax.set_title(f'Win Probability (1st Place) — {n_sims} Simulations')
    ax.set_xlim(0, max(top['place_1%']) * 1.15)
    for bar, val in zip(bars, top['place_1%']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_score_distribution(
    df_scores: pd.DataFrame, team_stats: pd.DataFrame, n_sims: int, out_path: Path
) -> None:
    """Box plot of score distribution for top 10 teams."""
    top_teams = team_stats.head(10).index.tolist()
    df_top = df_scores[df_scores['team'].isin(top_teams)].copy()
    df_top['team'] = pd.Categorical(df_top['team'], categories=top_teams, ordered=True)
    df_top = df_top.sort_values('team')
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    data_list = []
    labels = []
    for i, team in enumerate(top_teams):
        pts = df_top[df_top['team'] == team]['points'].values
        data_list.append(pts)
        positions.append(i + 1)
        labels.append(team)
    bp = ax.boxplot(data_list, positions=positions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Total Points')
    ax.set_title(f'Score Distribution — Top 10 Teams ({n_sims} Simulations)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def df_to_table_data(df: pd.DataFrame, max_rows: int = 20) -> list:
    """Convert DataFrame to list of lists for reportlab Table; limit rows."""
    df = df.head(max_rows)
    if isinstance(df.columns, pd.MultiIndex):
        headers = [str(c) for c in df.columns]
    else:
        headers = list(df.columns)
    if df.index.name and df.index.name not in headers:
        headers = [df.index.name or ''] + headers
        data = [[str(i)] + [str(v) for v in row] for i, row in df.iterrows()]
    else:
        data = [[str(v) for v in row] for _, row in df.iterrows()]
    return [headers] + data


def add_commentary(story, data: dict) -> None:
    """Append analysis/commentary paragraphs to the story."""
    style = getSampleStyleSheet()['Normal']
    summary = data['summary']
    prob_df = data['prob_df']
    n_sims = data['n_sims']

    story.append(Paragraph("Analysis & Commentary", getSampleStyleSheet()['Heading2']))
    story.append(Spacer(1, 0.2 * inch))

    top_team = summary.index[0]
    win_pct = summary.loc[top_team, 'win_probability_%']
    avg_pts = summary.loc[top_team, 'mean']
    story.append(Paragraph(
        f"Based on {n_sims} simulated meets, <b>{top_team}</b> is the clear favorite, "
        f"winning approximately <b>{win_pct:.1f}%</b> of simulations with an average "
        f"total score of <b>{avg_pts:.1f} points</b>. The model reflects seed times, "
        "event assignments, and a Markov-style performance state with team momentum. "
        "Total scores include both swimming and diving events.",
        style
    ))
    story.append(Spacer(1, 0.15 * inch))

    if len(summary) >= 2:
        second = summary.index[1]
        second_win = summary.loc[second, 'win_probability_%']
        story.append(Paragraph(
            f"The primary challenger is <b>{second}</b>, with a <b>{second_win:.1f}%</b> "
            "chance of winning. The gap between the top two teams indicates that "
            "the state title is likely a two-school race unless seed data or "
            "day-of-meet variance shifts significantly.",
            style
        ))
        story.append(Spacer(1, 0.15 * inch))

    podium = data['podium_probs']
    top3_podium = podium.head(3)
    names = ", ".join(top3_podium.index.tolist())
    story.append(Paragraph(
        f"The most likely podium (top 3) consists of <b>{names}</b>, with each of these "
        "teams having a high probability of finishing in the top three. Coaches and "
        "athletes can use these probabilities to set realistic goals (e.g., contending "
        "for a podium spot vs. winning the meet).",
        style
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(
        "This report is generated from a stochastic simulator and should be interpreted "
        "as a probabilistic outlook, not a prediction of a single outcome. Actual results "
        "depend on many factors not in the model (injuries, taper, relay order, etc.).",
        style
    ))


def build_pdf(
    data: dict,
    fig_win_path: Path,
    fig_dist_path: Path,
    out_path: Path,
    title: str = "Swim Meet Simulation Report",
) -> None:
    """Assemble the PDF using reportlab."""
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        f"Based on {data['n_sims']} Monte Carlo simulations across {data['n_teams']} teams.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Methodology", styles['Heading2']))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "The simulator uses seed times from the state performance list, event assignments "
        "(up to 2 individual events per swimmer, plus relays), and a Markov-style performance "
        "state (Good/Average/Bad) with team momentum. Each simulation samples swim times "
        "and ranks finishers to compute team points; win and podium probabilities are "
        "estimated from the fraction of simulations in which each team placed 1st or top 3.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.4 * inch))

    story.append(Paragraph("Team Summary (Mean, Std, Min, Max, Win %)", styles['Heading2']))
    story.append(Spacer(1, 0.15 * inch))
    sum_df = data['summary'].head(15).reset_index()
    sum_cols = ['team', 'mean', 'std', 'min', 'max', 'win_probability_%']
    sum_df = sum_df[[c for c in sum_cols if c in sum_df.columns]]
    table_data = [list(sum_df.columns)] + sum_df.values.tolist()
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Podium Probability (Top 3 Finish)", styles['Heading2']))
    story.append(Spacer(1, 0.15 * inch))
    pod_df = data['podium_probs'].head(12).reset_index()
    pod_data = [list(pod_df.columns)] + pod_df.values.tolist()
    t2 = Table(pod_data, repeatRows=1)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70AD47')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4 * inch))

    if fig_win_path.exists():
        story.append(Paragraph("Win Probability (1st Place)", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        img = Image(str(fig_win_path), width=6 * inch, height=3.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.3 * inch))

    if fig_dist_path.exists():
        story.append(Paragraph("Score Distribution (Top 10 Teams)", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        img2 = Image(str(fig_dist_path), width=6 * inch, height=3 * inch)
        story.append(img2)
        story.append(Spacer(1, 0.3 * inch))

    add_commentary(story, data)

    doc.build(story)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a PDF report from swim meet simulation results."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help="Path to team_scores CSV (default: results/classN_ilp/team_scores.csv when -c/--class is set)",
    )
    parser.add_argument(
        "-c", "--class",
        dest="class_num",
        choices=["1", "2"],
        default="1",
        help="Class 1 or Class 2 data (default: 1). Sets default input/output when input_csv is not given.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PDF path (default: swim_meet_simulation_report.pdf or ..._class2.pdf for class 2)",
    )
    args = parser.parse_args()

    default_inputs = {
        "1": "results/class1_ilp/team_scores.csv",
        "2": "results/class2_ilp/team_scores.csv",
    }
    default_outputs = {
        "1": "results/reports/swim_meet_simulation_report.pdf",
        "2": "results/reports/swim_meet_simulation_report_class2.pdf",
    }
    csv_path = Path(args.input_csv or default_inputs[args.class_num]).resolve()
    out_path = Path(args.output or default_outputs[args.class_num]).resolve()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = "Swim Meet Simulation Report"
    if args.class_num == "2":
        title = "Swim Meet Simulation Report — Class 2"

    df_scores = load_and_validate_csv(csv_path)
    data = compute_summary_tables(df_scores)

    fig_dir = out_path.parent
    fig_win = fig_dir / "_report_win_prob.png"
    fig_dist = fig_dir / "_report_score_dist.png"

    plot_win_probability(data['prob_df'], data['n_sims'], fig_win)
    plot_score_distribution(df_scores, data['team_stats'], data['n_sims'], fig_dist)

    build_pdf(data, fig_win, fig_dist, out_path, title=title)

    for f in (fig_win, fig_dist):
        if f.exists():
            f.unlink()

    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
