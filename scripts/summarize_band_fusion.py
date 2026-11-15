#!/usr/bin/env python3
"""Make SUMMARY.csv from the band/fusion runs (scripts/band_fusion.sh).

    python3 scripts/summarize_band_fusion.py [results_local/band_fusion]

For each timing CSV: number of frames, median / min / max / IQR of the pipeline time.
If the Cachegrind file cg_<name>.out exists, also the D1 and LL data misses
(the Cachegrind window is 2 frames, so we divide by 2 to get misses per frame).
"""
import csv
import pathlib
import statistics
import sys


def cg_events(path):
    """totals of a cachegrind .out file, as a dict event -> count"""
    names = summary = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("events: "):
            names = line.split()[1:]
        elif line.startswith("summary: "):
            summary = [int(x) for x in line.split()[1:]]
    if names is None or summary is None or len(names) != len(summary):
        raise ValueError(f"cannot read the Cachegrind events of {path}")
    return dict(zip(names, summary))


def main(folder):
    folder = pathlib.Path(folder)
    rows = []
    for p in sorted(folder.glob("*.csv")):
        if p.name.startswith(("cg_", "perf_")):
            continue
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            if "pipeline_ms" not in (reader.fieldnames or []):
                continue  # SUMMARY.csv itself, or another kind of file
            values = [float(r["pipeline_ms"]) for r in reader]
        if not values:
            continue
        q = statistics.quantiles(values, n=4) if len(values) > 3 else None
        row = {
            "test": p.stem,
            "frames": len(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "iqr_ms": q[2] - q[0] if q else "",
        }
        cg = folder / f"cg_{p.stem}.out"
        if cg.exists():
            try:
                c = cg_events(cg)
                row["cg_D1misses_2frames"] = c.get("D1mr", 0) + c.get("D1mw", 0)
                row["cg_LLdmisses_2frames"] = c.get("DLmr", 0) + c.get("DLmw", 0)
                row["cg_D1misses_per_frame"] = row["cg_D1misses_2frames"] / 2
                row["cg_LLdmisses_per_frame"] = row["cg_LLdmisses_2frames"] / 2
            except ValueError as e:
                print(f"warning: {e}", file=sys.stderr)
        rows.append(row)

    if not rows:
        print("no timing CSV found in", folder, file=sys.stderr)
        sys.exit(1)

    fields = ["test", "frames", "median_ms", "min_ms", "max_ms", "iqr_ms",
              "cg_D1misses_2frames", "cg_LLdmisses_2frames",
              "cg_D1misses_per_frame", "cg_LLdmisses_per_frame"]
    out = folder / "SUMMARY.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} runs -> {out}")
    for r in rows:
        line = f"{r['test']:27s} median={r['median_ms']:.3f} ms"
        if "cg_D1misses_per_frame" in r:
            line += f"  D1/frame={r['cg_D1misses_per_frame']:.0f}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_local/band_fusion")
