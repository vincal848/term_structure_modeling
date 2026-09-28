"""Refresh the cached FRED data in data/.

Usage: python scripts/fetch_data.py [start-date]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from termstructure.data import DATA_DIR, refresh  # noqa: E402

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "1990-01-01"
    refresh(start)
    print(f"Saved Treasury CMT curve and DTB3 from {start} to {DATA_DIR}")
