"""Compare evaluated controllers side by side, from the summaries in runs/evaluation/.

Run evaluate_controller.py for each controller first.

    uv run python scripts/compare_controllers.py              # level 0, every evaluated controller
    uv run python scripts/compare_controllers.py --level 2
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
    ("off center, worst [m]", lambda s: s["touchdown"]["x error [m]"]["max"], "{:.2f}"),
    ("tilt, worst [deg]", lambda s: s["touchdown"]["tilt [deg]"]["max"], "{:.2f}"),
    ("flight time, mean [s]", lambda s: s["duration [s]"]["mean"], "{:.1f}"),
    ("fuel, mean [s of full thrust]", lambda s: s["fuel [s of full thrust]"]["mean"], "{:.1f}"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    suffix = f"_level{args.level}.json"
    files = sorted((ROOT / "runs" / "evaluation").glob(f"*{suffix}"))
    files.sort(key=lambda f: not f.name.startswith("pd_"))  # the baseline first
    if not files:
        raise SystemExit("no evaluations found: run scripts/evaluate_controller.py first")
    names = [f.name.removesuffix(suffix) for f in files]
    summaries = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    width = max(len(n) for n in names) + 2
    print(f"level {args.level}, test set\n")
    print(f"{'':<32}" + "".join(f"{n:>{width}}" for n in names))
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
