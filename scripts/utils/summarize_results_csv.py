#!/usr/bin/env python3
"""Summarize rollout CSVs; count zero-steering episodes in per-episode means.

Visual inputs use Traj (img_overlay); global inputs use global cmd.
Prompt alignment defaults to sum(PAT)/sum(ST).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROGRESS = "Task Progress (TR) %"
STEERING = "Steering Times (ST)"
VISUAL = "Traj"
GLOBAL = "global cmd"
ALIGNED = "Prompt Aligned Times(PAT)"
COLUMNS = (PROGRESS, STEERING, VISUAL, GLOBAL, ALIGNED)
ALIASES = {
    PROGRESS: "task_progress",
    STEERING: "prompt_counts.total",
    VISUAL: "prompt_counts.img_overlay",
    GLOBAL: "prompt_counts.global_action",
    ALIGNED: "PAT",
}


def summarize(path: Path, alignment: str = "overall") -> dict:
    if alignment not in {"overall", "episode", "count"}:
        raise ValueError(f"Unknown alignment method: {alignment}")
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or [])
        source_columns = {
            column: column if column in headers else ALIASES[column]
            for column in COLUMNS
        }
        missing = {column for column, source in source_columns.items() if source not in headers}
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            values = {}
            for column in COLUMNS:
                raw = (row.get(source_columns[column]) or "").strip()
                if column == PROGRESS:
                    raw = raw.removesuffix("%").strip()
                try:
                    value = float(raw)
                except ValueError:
                    raise ValueError(f"{path}:{reader.line_num}: invalid {column}: {raw!r}") from None
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path}:{reader.line_num}: invalid {column}: {raw!r}")
                if (column == PROGRESS and value > 100) or (column != PROGRESS and not value.is_integer()):
                    raise ValueError(f"{path}:{reader.line_num}: out-of-range progress or non-integer count: {column}")
                values[column] = value
            if values[ALIGNED] > values[STEERING]:
                raise ValueError(f"{path}:{reader.line_num}: PAT exceeds ST")
            rows.append(values)
    if not rows:
        raise ValueError(f"{path}: no episode rows")

    totals = {column: sum(row[column] for row in rows) for column in COLUMNS}
    ratios = [row[ALIGNED] / row[STEERING] for row in rows if row[STEERING] > 0]
    episode_alignment = 100 * sum(ratios) / len(ratios) if ratios else None
    overall_alignment = 100 * totals[ALIGNED] / totals[STEERING] if totals[STEERING] else None
    aligned_mean = totals[ALIGNED] / len(rows)
    alignment_values = {"episode": episode_alignment, "overall": overall_alignment, "count": aligned_mean}
    return {
        "source": str(path),
        "episodes": len(rows),
        "episodes_with_steering": len(ratios),
        "average_progress_percent": totals[PROGRESS] / len(rows),
        "average_steering_times": totals[STEERING] / len(rows),
        "average_visual_input": totals[VISUAL] / len(rows),
        "average_global_input": totals[GLOBAL] / len(rows),
        "average_prompt_alignment": alignment_values[alignment],
        "prompt_alignment_method": alignment,
        "prompt_alignment_unit": "times" if alignment == "count" else "percent",
        "episode_prompt_alignment_percent": episode_alignment,
        "overall_prompt_alignment_percent": overall_alignment,
        "average_prompt_aligned_times": aligned_mean,
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Rollout CSV path")
    parser.add_argument("--alignment", choices=("episode", "overall", "count"), default="overall",
                        help="episode: mean(PAT/ST), excluding ST=0; overall: sum(PAT)/sum(ST); count: mean(PAT)")
    parser.add_argument("-o", "--output", type=Path, help="Also save the summary as JSON")
    args = parser.parse_args()
    try:
        result = summarize(args.input, args.alignment)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        if args.output.resolve() == args.input.resolve():
            parser.error("Output must differ from input CSV")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
