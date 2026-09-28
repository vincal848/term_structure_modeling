"""Re-execute the notebook in place, regenerating every figure in figures/.

Usage: python scripts/make_figures.py
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "term_structure.ipynb"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=900", str(NOTEBOOK)],
        check=True,
        cwd=NOTEBOOK.parent,
    )
    print(f"Executed {NOTEBOOK.name}; figures written to figures/")
