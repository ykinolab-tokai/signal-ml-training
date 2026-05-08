#!/usr/bin/env python3
"""Replay code cells for chapter 08 and save figure outputs.

This script was generated from the notebook before notebook deletion.
It requires the original runtime dependencies such as NumPy and matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_CELLS = [
"# %matplotlib inline\n# import matplotlib.pyplot as plt",
"import pandas as pd\n\ndf = pd.read_csv('sample_data/california_housing_train.csv')\n\ndf.head(5)",
"plt.scatter(df['median_income'], df['median_house_value'])",
"plt.scatter(df['population'], df['median_house_value'])",
"plt.hist(df['median_house_value'])",
"# bins \u5f15\u6570\u306b\u5024\u3092\u6307\u5b9a\u3059\u308b\u3053\u3068\u3067\u3001\u30d3\u30f3\u306e\u6570\u3092\u6307\u5b9a\u3067\u304d\u307e\u3059\nplt.hist(df['median_house_value'], bins=50)",
"plt.boxplot(df['median_house_value'])",
"# \u8907\u6570\u6307\u5b9a\u3059\u308b\u5834\u5408\u306f\u3001\u30bf\u30d7\u30eb\u3092\u7528\u3044\u307e\u3059\nplt.boxplot((df['total_bedrooms'], df['population']))",
"import numpy as np\n\n# [0,10]\u306e\u9593\u3092100\u5206\u5272\u3057\u3066\u6570\u5024\u3092\u8fd4\u3059\nx = np.linspace(0, 10, 100)\n\n# x \u306e\u5024\u306b\u30e9\u30f3\u30c0\u30e0\u30ce\u30a4\u30ba\u3092\u52a0\u3048\u308b\ny = x + np.random.randn(100)",
"plt.plot(y)",
"plt.plot(x, y)",
"import seaborn as sns",
"sns.distplot(df['population'])",
"sns.pairplot(df)"
]
FIGURE_JOBS = [
    (2, ["figure-01.png"]),
    (3, ["figure-02.png"]),
    (4, ["figure-03.png"]),
    (5, ["figure-04.png"]),
    (6, ["figure-05.png"]),
    (7, ["figure-06.png"]),
    (9, ["figure-07.png"]),
    (10, ["figure-08.png"]),
    (12, ["figure-09.png"]),
    (13, ["figure-10.png"])
]


def main() -> None:
    """Execute notebook code cells sequentially and save any figures."""
    namespace: dict[str, object] = {}
    output_dir = Path(__file__).resolve().parent.parent / "generated" / "08-applied-matplotlib"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_job_map = dict(FIGURE_JOBS)

    plt.show = lambda *args, **kwargs: None
    plt.savefig = lambda *args, **kwargs: None

    for cell_index, code in enumerate(CODE_CELLS):
        exec(code, namespace)
        expected_files = figure_job_map.get(cell_index, [])
        fig_numbers = list(plt.get_fignums())
        if len(fig_numbers) != len(expected_files):
            raise RuntimeError(
                f"Cell {cell_index} produced {len(fig_numbers)} figure(s), expected {len(expected_files)}"
            )
        for fig_number, filename in zip(fig_numbers, expected_files):
            fig = plt.figure(fig_number)
            fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    main()
