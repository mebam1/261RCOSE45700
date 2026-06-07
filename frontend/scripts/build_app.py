from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
FILES = ("index.html", "styles.css", "app-config.js", "app.js")


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    for filename in FILES:
        shutil.copy2(ROOT / filename, DIST / filename)

    print(f"Built {len(FILES)} frontend files into {DIST}")


if __name__ == "__main__":
    main()
