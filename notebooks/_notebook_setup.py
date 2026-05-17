from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import seaborn as sns


def find_project_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "src").exists():
        return root

    for parent in root.parents:
        if (parent / "src").exists():
            return parent

    raise FileNotFoundError("Could not locate the project root containing the src/ folder.")


def bootstrap_notebook() -> Path:
    # This helper keeps notebook setup short and identical across the project.
    root = find_project_root()
    src_path = root / "src"
    data_path = root / "data" / "raw" / "Data.xlsx"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    if not data_path.exists():
        raise FileNotFoundError(
            "Could not find data/raw/Data.xlsx from the notebook location. "
            "Please place the project workbook there before running the notebooks."
        )

    sns.set_theme(style="whitegrid", context="talk")
    pd.options.display.float_format = "{:.4f}".format
    return root
