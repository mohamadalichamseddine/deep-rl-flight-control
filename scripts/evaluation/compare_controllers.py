"""Compare tested controllers side by side, from the summaries in runs/test/.

Run evaluate_controller.py for each controller first.

    uv run python scripts/evaluation/compare_controllers.py   # level 0, all evaluated controllers
    uv run python scripts/evaluation/compare_controllers.py --level 2
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROWS = [  # (label, how to read it from a summary, format)
    ("landed", lambda s: s["landing_rate"], "{:.1%}"),
    (
        "touchdown speed, mean [m/s]",
        lambda s: s["touchdown"]["vertical speed [m/s]"]["mean"],
        "{:.2f}",
    ),
    (
        "touchdown speed, worst [m/s]",
        lambda s: s["touchdown"]["vertical speed [m/s]"]["max"],
        "{:.2f}",
    ),
    ("off center, worst [m]", lambda s: s["touchdown"]["off center [m]"]["max"], "{:.2f}"),
    ("tilt, worst [deg]", lambda s: s["touchdown"]["tilt [deg]"]["max"], "{:.2f}"),
    ("flight time, mean [s]", lambda s: s["duration [s]"]["mean"], "{:.1f}"),
    ("fuel, mean [s of full thrust]", lambda s: s["fuel [s of full thrust]"]["mean"], "{:.1f}"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    test = ROOT / "runs" / "test"
    ending = f"tested_on_level{args.level}.json"
    files = sorted(test.rglob(f"*{ending}"))
    if not files:
        raise SystemExit("no test results found: run evaluate_controller.py first")
    # Each result is named after where it is, e.g. pd or <experiment>/seed0/level0/best_network.
    results = {
        (f.parent.relative_to(test) / f.name.removesuffix(ending).rstrip("_")).as_posix(): f
        for f in files
    }
    names = sorted(results, key=lambda name: name != "pd")  # the baseline first
    summaries = [json.loads(results[name].read_text(encoding="utf-8")) for name in names]

    # The names are long, so the columns are numbered.
    print(f"tested on level {args.level}\n")
    for i, name in enumerate(names, 1):
        print(f"  [{i}] {name}")
    width = 10
    print(f"\n{'':<32}" + "".join(f"{f'[{i}]':>{width}}" for i in range(1, len(names) + 1)))
    for label, read, fmt in ROWS:
        cells = []
        for summary in summaries:
            try:
                cells.append(fmt.format(read(summary)))
            except KeyError:  # e.g. no touchdowns at all
                cells.append("-")
        print(f"{label:<32}" + "".join(f"{c:>{width}}" for c in cells))


if __name__ == "__main__":
    main()
