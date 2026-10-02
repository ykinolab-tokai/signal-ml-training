#!/usr/bin/env python3
"""Replay code cells for chapter 08 and save figure outputs.

This legacy replay uses deterministic synthetic housing data.
It needs only the shared NumPy, pandas, and Matplotlib dependencies.
Replayed figures replace the historical reference figures when explicitly run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_CELLS = [
    '# %matplotlib inline\n# import matplotlib.pyplot as plt',
    'import numpy as np\nimport pandas as pd\nrng = np.random.default_rng(0)\nn = 1000\nincome = rng.uniform(1.0, 10.0, n)\nhouseholds = rng.integers(50, 1000, n)\ndf = pd.DataFrame({\n    "median_income": income,\n    "median_house_value": np.clip(50000 * income + rng.normal(0, 50000, n), 0, None),\n    "households": households,\n    "population": households * rng.integers(1, 6, n),\n    "total_bedrooms": households * rng.uniform(1.0, 3.0, n),\n})\n',
    "plt.scatter(df['median_income'], df['median_house_value'])",
    "plt.scatter(df['population'], df['median_house_value'])",
    "plt.hist(df['median_house_value'])",
    "# bins 引数に値を指定することで、ビンの数を指定できます\nplt.hist(df['median_house_value'], bins=50)",
    "plt.boxplot(df['median_house_value'])",
    "# 複数指定する場合は、タプルを用います\nplt.boxplot((df['total_bedrooms'], df['population']))",
    'import numpy as np\n\n# [0,10]の間を100分割して数値を返す\nx = np.linspace(0, 10, 100)\n\n# x の値にランダムノイズを加える\ny = x + np.random.default_rng(0).normal(size=100)',
    'plt.plot(y)',
    'plt.plot(x, y)',
    'from pandas.plotting import scatter_matrix',
    'plt.hist(df["population"], bins=30)',
    'scatter_matrix(df[["median_income", "median_house_value", "population", "households"]], figsize=(10, 10), diagonal="hist")',
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
