"""
Aggregate all benchmark CSVs into a single summary table.

Usage:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --root /path/to/project
    python scripts/summarize_results.py --collect-dir trajectories/  # include per-ep JSON
"""
import argparse
import csv
import glob
import json
import os
import sys


# Maps CSV filename → (mode_label, tier)
# Single-egg CSVs use 6-column schema; two-egg use 20-column schema.
_KNOWN_CSVS = {
    "results.csv":                ("single",          "medium"),
    "results_stress.csv":         ("single",          "stress"),
    "results_extreme.csv":        ("single",          "extreme"),
    "results_two_egg_easy.csv":   ("two-egg (static)", "easy"),
    "results_two_egg_medium.csv": ("two-egg (static)", "medium"),
    "results_two_egg_stress.csv": ("two-egg (static)", "stress"),
    "results_two_egg_extreme.csv":("two-egg (static)", "extreme"),
}


def _parse_csv(path, mode):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _agg_single(rows):
    n = len(rows)
    success = sum(1 for r in rows if r["result"] == "SUCCESS")
    dropped = sum(1 for r in rows if "DROPPED" in r["result"] or "TIMEOUT" in r["result"])
    overq   = sum(1 for r in rows if "OVER" in r["result"])
    peak    = max(float(r["grip_peak"]) for r in rows) if rows else 0.0
    return {
        "n": n, "success": success,
        "dist_disturbed": 0, "dropped": dropped, "over_squeezed": overq,
        "peak_grip": peak,
    }


def _agg_two_egg(rows):
    n = len(rows)
    success   = sum(1 for r in rows if r["result"] == "SUCCESS")
    disturbed = sum(1 for r in rows if "DISTRACTOR_DISTURBED" in r["result"])
    rolling   = sum(1 for r in rows if "DISTRACTOR_ROLLING"   in r["result"])
    dropped   = sum(1 for r in rows if "DROPPED" in r["result"] or "TIMEOUT" in r["result"])
    overq     = sum(1 for r in rows if "OVER" in r["result"])
    peak      = max(float(r["peak_grip"]) for r in rows) if rows else 0.0
    return {
        "n": n, "success": success,
        "dist_disturbed": disturbed + rolling, "dropped": dropped,
        "over_squeezed": overq, "peak_grip": peak,
    }


def _md_table(table_rows):
    """Render list-of-dicts as a markdown table. First dict defines column order."""
    if not table_rows:
        return ""
    headers = list(table_rows[0].keys())
    widths  = [max(len(h), max(len(str(r.get(h, ""))) for r in table_rows))
               for h in headers]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr  = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    lines = [hdr, sep]
    for r in table_rows:
        lines.append("| " + " | ".join(str(r.get(h, "")).ljust(w)
                                        for h, w in zip(headers, widths)) + " |")
    return "\n".join(lines)


def _load_summaries(collect_dir):
    """Return list of per-episode summary dicts loaded from JSON files."""
    pattern = os.path.join(collect_dir, "summary_ep*.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate benchmark CSVs into a summary table")
    p.add_argument("--root",        default=".", help="project root (default: .)")
    p.add_argument("--collect-dir", default=None,
                   help="also aggregate per-episode summary JSONs from this directory")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    table_rows = []

    for filename, (mode, tier) in _KNOWN_CSVS.items():
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            continue
        rows = _parse_csv(path, mode)
        if not rows:
            continue
        agg = _agg_two_egg(rows) if mode.startswith("two") else _agg_single(rows)
        n   = agg["n"]
        suc = agg["success"]
        pct = f"{100 * suc // n}%" if n else "—"
        dist = str(agg["dist_disturbed"]) if mode.startswith("two") else "—"
        grip = f"{agg['peak_grip']:.3f} N"
        table_rows.append({
            "mode":            mode,
            "tier":            tier,
            "N":               str(n),
            "success":         f"{suc}/{n} ({pct})",
            "DIST_DISTURBED":  dist,
            "DROPPED/TIMEOUT": str(agg["dropped"]),
            "OVER-SQUEEZED":   str(agg["over_squeezed"]),
            "peak_grip":       grip,
        })

    print("## Benchmark Summary — seed 42, 10 episodes per row\n")
    print(_md_table(table_rows))

    if args.collect_dir:
        summaries = _load_summaries(args.collect_dir)
        if not summaries:
            print(f"\nNo summary JSONs found in {args.collect_dir}")
            return
        print(f"\n### Per-episode summaries from {args.collect_dir}\n")
        ep_rows = []
        for s in summaries:
            dist_r = s.get("distractor_result") or {}
            ep_rows.append({
                "ep":            str(s["episode_id"]),
                "tier":          s["tier"],
                "mode":          s["mode"],
                "result":        s["result"],
                "steps":         str(s["steps"]),
                "peak_grip":     f"{s['peak_grip']:.4f} N",
                "contact_count": str(s["contact_count"]),
                "target_ok":     str(s["target_success"]),
                "dist_stable":   str(dist_r.get("stable", "—")),
                "dist_disp_mm":  f"{dist_r.get('displacement_mm', '—')}",
            })
        print(_md_table(ep_rows))


if __name__ == "__main__":
    main()
