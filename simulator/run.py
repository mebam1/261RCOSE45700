from __future__ import annotations

from pathlib import Path

from simulator.ui import launch


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    launch(project_root)


if __name__ == "__main__":
    main()
