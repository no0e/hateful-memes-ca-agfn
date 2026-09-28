"""Collect results/<variant>/seed*.json into one table.

    python scripts/aggregate.py

Writes results/summary.json and results/summary.md and prints the table. Each
number is a mean over seeds with its standard deviation. Smoke runs are
ignored.
"""
import argparse
import json
from pathlib import Path

from ca_agfn.config import ROOT
from ca_agfn.results import cell, load, summarise, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args()

    summary = summarise(load(args.results))
    if not summary:
        raise SystemExit(f"No results under {args.results}.")
    out = Path(args.results)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    markdown = table(summary)
    (out / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)

    full = summary.get("full")
    if full and full["r_gate_entropy"]:
        print(f"\nFull model: gate against entropy r = "
              f"{cell(full['r_gate_entropy'])}, entropy std across memes "
              f"{cell(full['entropy_std'])}")


if __name__ == "__main__":
    main()
