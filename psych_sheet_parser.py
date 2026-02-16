#!/usr/bin/env python3
"""
Parse MSHSAA psych sheet PDFs into CSV with columns: name, team, event, best_time, seed_rank, is_relay.
Output matches the schema expected by the swim meet simulator.
"""

import re
import csv
from pathlib import Path
from typing import Optional

import pdfplumber
import numpy as np


# Map PDF event titles to simulator canonical names (match class1_assignments / swimming_performance)
EVENT_NAME_MAP = {
    "200 Yard Medley Relay": "200 Medley Relay",
    "200 Yard Freestyle": "200 Free",
    "200 Yard Individual Medley": "200 IM",
    "200 Yard IM": "200 IM",
    "50 Yard Freestyle": "50 Free",
    "100 Yard Butterfly": "100 Butterfly",
    "100 Yard Freestyle": "100 Free",
    "500 Yard Freestyle": "500 Free",
    "200 Yard Freestyle Relay": "200 Freestyle Relay",
    "100 Yard Backstroke": "100 Back",
    "100 Yard Breaststroke": "100 Breast",
    "400 Yard Freestyle Relay": "400 Freestyle Relay",
    "1 mtr Diving": "Diving",
}


def normalize_event_name(pdf_title: str) -> Optional[str]:
    """Extract and normalize event name from 'Event N ...(Girls X Yard Y)' or 'Event N Girls X Yard Y'."""
    # Match "Girls 200 Yard Medley Relay" or "Girls 50 Yard Freestyle" etc.
    m = re.search(r"Girls\s+(.+?)(?:\s*\)|\s*$)", pdf_title, re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r"Girls\s+(.+)", pdf_title, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    return EVENT_NAME_MAP.get(raw, raw)


def convert_swim_time(time_str: str):
    """Convert time string to seconds. Returns (seconds_float, True) or (np.nan, False) for invalid."""
    if not time_str or not isinstance(time_str, str):
        return np.nan, False
    time_str = time_str.strip()
    try:
        parts = time_str.split(":")
        if len(parts) == 1:
            return float(parts[0]), True
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1]), True
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]), True
    except ValueError:
        pass
    return np.nan, False


def parse_individual_line(line: str) -> Optional[tuple]:
    """Parse a line like '2 Avery Frick JR St. Michael the 23.81'. Returns (rank, name, school, time_sec) or None."""
    line = line.strip()
    # Rank at start (digits)
    m = re.match(r"^(\d+)\s+", line)
    if not m:
        return None
    rank = int(m.group(1))
    rest = line[m.end() :].strip()
    # Year: FR, SO, JR, SR
    yr_m = re.search(r"\s(FR|SO|JR|SR)\s", rest)
    if not yr_m:
        return None
    name = rest[: yr_m.start()].strip()
    after_yr = rest[yr_m.end() :].strip()
    # Time at end: digits.digits or digits:digits.digits
    time_m = re.search(r"\s([\d]+:\d+\.\d+|[\d]+\.\d+)\s*$", after_yr)
    if not time_m:
        return None
    time_str = time_m.group(1).strip()
    school = after_yr[: time_m.start()].strip()
    sec, ok = convert_swim_time(time_str)
    if not ok:
        return None
    return (rank, name, school, sec)


def parse_relay_line(line: str) -> Optional[tuple]:
    """Parse a line like '1 Visitation Acade 1:50.53' or '1 St. Teresa's Aca 1:40.94'. Returns (rank, team, time_sec) or None."""
    line = line.strip()
    m = re.match(r"^(\d+)\s+(.+?)\s+([\d]+:\d+\.\d+)\s*$", line)
    if not m:
        return None
    rank = int(m.group(1))
    team = m.group(2).strip()
    sec, ok = convert_swim_time(m.group(3))
    if not ok:
        return None
    return (rank, team, sec)


def extract_column_text(page, x0: float, x1: float) -> list[str]:
    """Extract text from a vertical slice of the page, line by line."""
    crop = page.crop((x0, 0, x1, page.height))
    text = crop.extract_text()
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Parse psych sheet PDF into list of records: name, team, event, best_time, seed_rank, is_relay."""
    records = []
    # Column split (approximate from inspection)
    mid = 300

    with pdfplumber.open(pdf_path) as pdf:
        current_event = None
        current_is_relay = False
        current_is_diving = False

        for page in pdf.pages:
            left_lines = extract_column_text(page, 0, mid)
            right_lines = extract_column_text(page, mid, page.width)
            all_lines = left_lines + right_lines

            for line in all_lines:
                # Event header: "Event 4 ...(Girls 50 Yard Freestyle)" or "Event 6 Girls 100 Yard Butterfly"
                event_m = re.match(r"Event\s+\d+\s+\.{0,3}\s*\(?Girls\s+", line, re.IGNORECASE) or re.match(
                    r"Event\s+\d+\s+Girls\s+", line, re.IGNORECASE
                )
                if event_m:
                    norm = normalize_event_name(line)
                    if norm:
                        current_event = norm
                        current_is_relay = "Relay" in current_event
                        current_is_diving = "Diving" in current_event
                    continue

                if current_event is None:
                    continue

                # Skip header lines
                if "Name" in line and "School" in line and "Seed" in line:
                    continue
                if "Team" in line and "Relay" in line and "Seed" in line:
                    continue

                if current_is_diving:
                    # Diving: optional to include; skip for now to match individual swimming events only
                    continue

                if current_is_relay:
                    parsed = parse_relay_line(line)
                    if parsed:
                        rank, team, time_sec = parsed
                        records.append({
                            "name": team,
                            "team": team,
                            "event": current_event,
                            "best_time": time_sec,
                            "seed_rank": rank,
                            "is_relay": True,
                        })
                else:
                    parsed = parse_individual_line(line)
                    if parsed:
                        rank, name, school, time_sec = parsed
                        records.append({
                            "name": name,
                            "team": school,
                            "event": current_event,
                            "best_time": time_sec,
                            "seed_rank": rank,
                            "is_relay": False,
                        })

    return records


def write_psych_entries_csv(records: list[dict], out_path: Path) -> None:
    """Write records to CSV with columns name, team, event, best_time, seed_rank, is_relay."""
    out_path = Path(out_path)
    fieldnames = ["name", "team", "event", "best_time", "seed_rank", "is_relay"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = {k: r[k] for k in fieldnames}
            t = r["best_time"]
            row["best_time"] = round(t, 2) if np.isfinite(t) else ""
            w.writerow(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse psych sheet PDF to CSV (writes to data/processed/ by default)")
    parser.add_argument("pdf", type=Path, help="Path to psych sheet PDF (e.g. data/raw/class1psychsheets.pdf)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path (default: data/processed/classN_psych_entries.csv)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Output directory (default: data/processed)")
    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")
    records = parse_pdf(pdf_path)
    out = args.output
    if out is None:
        data_dir = args.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        name = pdf_path.stem.lower()
        if "class1" in name or "class 1" in name:
            out = data_dir / "class1_psych_entries.csv"
        elif "class2" in name or "class 2" in name:
            out = data_dir / "class2_psych_entries.csv"
        else:
            out = data_dir / (pdf_path.stem + "_psych_entries.csv")
    else:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
    write_psych_entries_csv(records, out)
    print(f"Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
