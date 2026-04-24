#!/usr/bin/env python3
"""Replay code cells for chapter 07 and save figure outputs.

This script was generated from the notebook before notebook deletion.
It requires the original runtime dependencies such as NumPy and matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_CELLS = [
"import pandas as pd",
"# \u30c7\u30fc\u30bf\u30bb\u30c3\u30c8\u306e\u8aad\u307f\u8fbc\u307f\ndf = pd.read_csv('sample_data/california_housing_train.csv')",
"# \u578b\u306e\u78ba\u8a8d\ntype(df)",
"df",
"df.head()",
"df.head(3)",
"df['longitude'].head(3)",
"df.to_csv('sample.csv')",
"# !ls sample.csv",
"# \u5f62\u306e\u78ba\u8a8d\ndf.shape",
"# \u5e73\u5747\ndf.mean()",
"# \u5206\u6563\ndf.var()",
"# \u5404\u5217\u306e None, NaN, NaT \u306e\u3044\u305a\u308c\u3067\u3082\u306a\u3044\u5024\u306e\u6570\ndf.count()",
"# \u30c7\u30fc\u30bf\u306e\u6982\u8981\ndf.describe()",
"# \u76f8\u95a2\u4fc2\u6570\u306e\u7b97\u51fa\ndf.corr()",
"# total_rooms \u5217\u306e\u5024\u3092\u6607\u9806\u306b\u4e26\u3079\u66ff\u3048\ndf_as = df.sort_values(by='total_rooms')",
"df_as.head()",
"# total_rooms \u306e\u5217\u306e\u5024\u3092\u964d\u9806\u306b\u4e26\u3079\u66ff\u3048\ndf_de = df.sort_values(by='total_rooms', ascending=False)",
"df_de.head()",
"# \u30c7\u30fc\u30bf\u306e\u78ba\u8a8d\ndf.head(3)",
"# df.iloc[\u884c, \u5217]\n# 0 \u884c\u76ee longitude \u5217\u306e\u9078\u629e\ndf.iloc[0, 0]",
"# 1 \u884c\u76ee latitude \u5217\u306e\u9078\u629e\ndf.iloc[1, 1]",
"# \u3059\u3079\u3066\u306e\u884c\u306e\u3001\u6700\u5f8c\u306e\u5217\u3092\u9078\u629e\nt = df.iloc[:, -1]",
"# \u5148\u982d3\u4ef6\u306e\u8868\u793a\nt.head(3)",
"# \u578b\u306e\u78ba\u8a8d\ntype(t)",
"# \u3059\u3079\u3066\u306e\u884c\u306e\u3001\u5148\u982d\u306e\u5217\u304b\u3089\u672b\u5c3e\u306e\u5217\u306e\u3072\u3068\u3064\u624b\u524d\u307e\u3067\u3092\u9078\u629e\nx = df.iloc[:, 0:-1]",
"# \u5148\u982d\u306e3\u4ef6\u306e\u8868\u793a\nx.head(3)",
"# \u3059\u3079\u3066\u306e\u884c\u306e\u3001\u5148\u982d\u306e\u5217\u304b\u3089\u672b\u5c3e\u306e\u5217\u306e\u3072\u3068\u3064\u624b\u524d\u307e\u3067\u3092\u9078\u629e\nx = df.iloc[:, :-1]",
"# \u5148\u982d\u306e3\u4ef6\u306e\u8868\u793a\nx.head(3)",
"# \u578b\u306e\u78ba\u8a8d\ntype(x)",
"# median_house_value \u5217\u3092\u9078\u629e\u3057\u3001\u5168\u8981\u7d20\u306b\u5bfe\u3057 70000 \u3088\u308a\u5927\u304d\u3044\u304b\u3069\u3046\u304b\u3092\u8a08\u7b97\nmask = df['median_house_value'] > 70000",
"mask.head()",
"# df[mask] \u306e\u5148\u982d 5 \u4ef6\u3092\u8868\u793a\ndf[mask].head()",
"# 70000 \u3088\u308a\u5c0f\u3055\u3044 \u307e\u305f\u306f 80000 \u3088\u308a\u5927\u304d\u3044\nmask2 = (df['median_house_value'] < 70000) | (df['median_house_value'] > 80000)",
"mask2.head()",
"df[mask2].head()",
"# 70000 \u3088\u308a\u5927\u304d\u3044 \u304b\u3064 80000 \u3088\u308a\u5c0f\u3055\u3044\nmask3 = (df['median_house_value'] > 70000) & (df['median_house_value'] < 80000)",
"mask3.head()",
"df[mask3].head()",
"df[(df['median_house_value'] > 70000) & (df['median_house_value'] < 80000)].head()",
"# \u65b0\u3057\u3044\u5217 target \u3092 None \u3067\u521d\u671f\u5316\ndf['target'] = None",
"df.head()",
"mask1 = df['median_house_value'] < 60000\nmask2 = (df['median_house_value'] >= 60000) & (df['median_house_value'] < 70000)\nmask3 = (df['median_house_value'] >= 70000) & (df['median_house_value'] < 80000)\nmask4 = df['median_house_value'] >= 80000",
"df.loc[mask1, 'target'] = 0\ndf.loc[mask2, 'target'] = 1\ndf.loc[mask3, 'target'] = 2\ndf.loc[mask4, 'target'] = 3",
"# \u5148\u982d\u304b\u3089 5 \u756a\u76ee\u307e\u3067\u3092\u8868\u793a\ndf.head()",
"# \u6b20\u640d\u5024\u3092\u4eba\u70ba\u7684\u306b\u4f5c\u6210\ndf.iloc[0, 0] = None",
"# (0, 'longitude') \u306e\u8981\u7d20\u304c NaN \u306b\u306a\u3063\u3066\u3044\u308b\u3053\u3068\u3092\u78ba\u8a8d\ndf.head(3)",
"# \u6b20\u640d\u5024\u306e\u3042\u308b\u30ec\u30b3\u30fc\u30c9\u3092\u524a\u9664\ndf_dropna = df.dropna()\n\n# \u5148\u982d\u304b\u3089 3 \u4ef6\u3092\u8868\u793a\ndf_dropna.head(3)",
"mean = df.mean()\nmean",
"# \u6b20\u640d\u5024\u3092 mean \u3067\u88dc\u5b8c\ndf_fillna =  df.fillna(mean)\n\n# \u5148\u982d\u304b\u3089 3 \u4ef6\u3092\u8868\u793a\ndf_fillna.head(3)",
"type(df)",
"type(df.values)",
"df.values",
"type(df['longitude'])",
"type(df['longitude'].values)",
"import numpy as np\n\n# ndarray -> pd.DataFrame\ndf = pd.DataFrame(\n    data=np.random.randn(10, 10)\n)\n\ndf",
"# \u30b0\u30e9\u30d5\u306e\u63cf\u753b\ndf.plot()"
]
FIGURE_JOBS = [
    (56, ["figure-01.png"])
]


def main() -> None:
    """Execute notebook code cells sequentially and save any figures."""
    namespace: dict[str, object] = {}
    output_dir = Path(__file__).resolve().parent.parent / "generated" / "07-introduction-to-pandas"
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
