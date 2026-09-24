"""Inline the precomputed grid into a single self-contained dashboard file.

    python dashboard/build.py

The dashboard has to open from disk and survive being handed to someone as one
file, so the grid is inlined rather than fetched: a browser opening
``file://.../index.html`` will not fetch a sibling JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "dashboard" / "static" / "index.template.html"
GRID = ROOT / "results" / "grid.json"
OUT = ROOT / "dashboard" / "static" / "index.html"


def main() -> None:
    if not GRID.exists():
        raise SystemExit(
            f"{GRID} not found. Run: python -m payments_shock.run_grid --out results/grid.json"
        )
    data = json.loads(GRID.read_text())
    # Keep the payload out of the JSON-in-HTML failure modes.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__GRID_JSON__", blob)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
