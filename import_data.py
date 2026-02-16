#!/usr/bin/env python3
"""
Import MSHSAA swimming and dive performance data from HTML exports.
Writes all outputs to data/processed/. Looks for HTML in data/raw/ or project root.
"""

import re
import argparse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

EVENT_ORDER = [
    "200 Medley Relay",
    "200 Free",
    "200 IM",
    "50 Free",
    "100 Fly",
    "100 Free",
    "500 Free",
    "200 Free Relay",
    "100 Back",
    "100 Breast",
    "400 Free Relay",
]


def clean_time(time_str):
    if not time_str:
        return time_str
    return " ".join(time_str.replace("*", "").split()).strip()


def scrape_swimming(html_path: Path, output_path: Path) -> pd.DataFrame:
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    all_data = []
    fs_grid_tables = soup.find_all("table", class_="fs_grid")
    if len(fs_grid_tables) != len(EVENT_ORDER):
        print(f"Warning: found {len(fs_grid_tables)} tables, expected {len(EVENT_ORDER)}")
    for table_index, table in enumerate(fs_grid_tables):
        event_name = EVENT_ORDER[table_index] if table_index < len(EVENT_ORDER) else f"Event {table_index + 1}"
        rows = table.find_all("tr")
        is_relay_event = "Relay" in event_name
        for row_index, row in enumerate(rows):
            if row_index == 0:
                continue
            tds = row.find_all("td")
            if not tds:
                continue
            is_relay_table = len(tds) == 5
            if is_relay_table:
                seed_rank = tds[0].text.strip().replace("#", "")
                team = tds[1].text.strip()
                name = team
                best_time = tds[4].text.strip() if len(tds) > 4 else ""
                grade = ""
            else:
                seed_rank = tds[0].text.strip().replace("#", "")
                name_cell = tds[1]
                name_text = name_cell.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in name_text.split("\n") if l.strip()]
                name = lines[0] if lines else name_cell.text.strip()
                grade = tds[2].text.strip() if len(tds) > 2 else ""
                team = tds[3].text.strip() if len(tds) > 3 else ""
                if not team and len(lines) > 1:
                    team = lines[1]
                best_time = tds[6].text.strip() if len(tds) > 6 else ""
            name = " ".join(name.split())
            team = " ".join(team.split())
            best_time = clean_time(best_time)
            all_data.append({
                "name": name,
                "team": team,
                "event": event_name,
                "best_time": best_time,
                "seed_rank": seed_rank,
                "is_relay": is_relay_table,
            })
    df = pd.DataFrame(all_data)[["name", "team", "event", "best_time", "seed_rank", "is_relay"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")
    return df


def scrape_dive(html_path: Path, output_path: Path, data_dir: Path) -> pd.DataFrame:
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    tables = soup.find_all("table", class_="fs_grid")
    if len(tables) < 2:
        print(f"Warning: expected 2 dive tables, found {len(tables)}")
    rows_all = []
    for table_idx, table in enumerate(tables):
        class_num = table_idx + 1
        for tr in table.find_all("tr"):
            if "hide" in (tr.get("class") or []):
                continue
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            rank_text = tds[0].get_text(strip=True)
            if not rank_text or not rank_text.replace("#", "").isdigit():
                continue
            student_cell = tds[1]
            lines = [l.strip() for l in student_cell.get_text(separator="\n", strip=True).split("\n") if l.strip()]
            name = lines[0] if lines else ""
            team = lines[1] if len(lines) > 1 else ""
            if not team:
                school_div = student_cell.find("div", class_=lambda c: c and ("gray" in (c if isinstance(c, list) else [])))
                if school_div:
                    team = school_div.get_text(strip=True)
            name = " ".join(name.split())
            team = " ".join(team.split())
            score_text = tds[3].get_text(strip=True)
            m = re.search(r"[\d.]+", score_text)
            total_score = float(m.group()) if m else None
            if total_score is None:
                continue
            try:
                difficulty = float(tds[4].get_text(strip=True))
            except (ValueError, IndexError):
                difficulty = None
            rows_all.append({"name": name, "team": team, "total_score": total_score, "difficulty": difficulty, "class": class_num})
    df = pd.DataFrame(rows_all)
    df = df.sort_values(["class", "total_score"], ascending=[True, False]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} dive records to {output_path}")
    print(f"  Class 1: {len(df[df['class']==1])}, Class 2: {len(df[df['class']==2])}")
    for c in [1, 2]:
        subset = df[df["class"] == c]
        if not subset.empty:
            out = data_dir / f"class{c}_dive.csv"
            subset.to_csv(out, index=False)
            print(f"  {out}: {len(subset)} rows")
    return df


def _find_input(path: Path, raw_dir: Path) -> Path:
    """Resolve input path: use as-is if exists, else try data/raw/."""
    if path.exists():
        return path
    alt = raw_dir / path.name
    return alt if alt.exists() else path


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    parser = argparse.ArgumentParser(description="Import swimming and dive data from MSHSAA HTML to data/processed/")
    parser.add_argument("--data-dir", type=Path, default=processed_dir, help="Output directory (default: data/processed)")
    parser.add_argument("--raw-dir", type=Path, default=raw_dir, help="Directory to look for HTML inputs (default: data/raw)")
    parser.add_argument("--swimming-html", type=Path, default=Path("MSHSAA Swimming Performance List.html"), help="Swimming HTML path")
    parser.add_argument("--dive-html", type=Path, default=Path("MSHSAA Dive Performance Listing.html"), help="Dive HTML path")
    parser.add_argument("--swimming-only", action="store_true", help="Only scrape swimming")
    parser.add_argument("--dive-only", action="store_true", help="Only scrape dive")
    args = parser.parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    swimming_path = _find_input(args.swimming_html, args.raw_dir)
    dive_path = _find_input(args.dive_html, args.raw_dir)

    if not args.dive_only and swimming_path.exists():
        out = data_dir / "swimming_performance.csv"
        scrape_swimming(swimming_path, out)
    elif not args.dive_only:
        print(f"Swimming HTML not found: {args.swimming_html} or {args.raw_dir / args.swimming_html.name}")

    if not args.swimming_only and dive_path.exists():
        out = data_dir / "dive_performance.csv"
        scrape_dive(dive_path, out, data_dir)
    elif not args.swimming_only:
        print(f"Dive HTML not found: {args.dive_html} or {args.raw_dir / args.dive_html.name}")


if __name__ == "__main__":
    main()
