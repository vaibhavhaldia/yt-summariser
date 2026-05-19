"""
run.py — Desktop app launcher for yt-summariser.

Usage:
    python run.py          # launch GUI
    python run.py --cli <youtube_url> [options]   # use CLI directly
"""

import sys
from pathlib import Path

# Ensure the zed/ directory is on the path so both
# `app` and `summarise` packages resolve correctly.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def _launch_gui():
    from app.app import run

    run()


def _launch_cli():
    # Strip the leading --cli flag then hand off to the summarise CLI
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    from summarise.summarise import main

    main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        _launch_cli()
    else:
        _launch_gui()
