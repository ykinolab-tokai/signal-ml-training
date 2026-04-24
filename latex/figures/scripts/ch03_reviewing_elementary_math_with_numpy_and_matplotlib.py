#!/usr/bin/env python3
"""Replay code cells for chapter 03 and save figure outputs.

This script was generated from the notebook before notebook deletion.
It requires the original runtime dependencies such as NumPy and matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_CELLS = [
"import numpy as np",
"# 1\u6b21\u5143\u914d\u5217\u306e\u5b9a\u7fa9\na = np.array([1, 2, 3])\n\na",
"a.shape",
"a.ndim",
"# 2\u6b21\u5143\u914d\u5217\u306e\u5b9a\u7fa9\nb = np.array(\n    [[1, 2, 3],\n     [4, 5, 6],\n     [7, 8, 9]]\n)\n\nprint(b)",
"print('Shape:', b.shape)\nprint('Rank:', b.ndim)",
"b.size",
"# \u5f62\u3092\u6307\u5b9a\u3057\u3066\u3001\u8981\u7d20\u304c\u5168\u3066 0 \u3067\u57cb\u3081\u3089\u308c\u305f ndarray \u3092\u4f5c\u308b\na = np.zeros((3, 3))\n\na",
"# \u5f62\u3092\u6307\u5b9a\u3057\u3066\u3001\u8981\u7d20\u304c\u5168\u3066 1 \u3067\u57cb\u3081\u3089\u308c\u305f ndarray \u3092\u4f5c\u308b\nb = np.ones((2, 3))\n\nb",
"# \u5f62\u3068\u5024\u3092\u6307\u5b9a\u3057\u3066\u3001\u8981\u7d20\u304c\u6307\u5b9a\u3057\u305f\u5024\u3067\u57cb\u3081\u3089\u308c\u305f ndarray \u3092\u4f5c\u308b\nc = np.full((3, 2), 9)\n\nc",
"# \u6307\u5b9a\u3055\u308c\u305f\u5927\u304d\u3055\u306e\u5358\u4f4d\u884c\u5217\u3092\u8868\u3059 ndarray \u3092\u4f5c\u308b\nd = np.eye(5)\n\nd",
"# \u5f62\u3092\u6307\u5b9a\u3057\u3066\u3001 0 ~ 1 \u306e\u9593\u306e\u4e71\u6570\u3067\u8981\u7d20\u3092\u57cb\u3081\u305f ndarray \u3092\u4f5c\u308b\ne = np.random.random((4, 5))\n\ne",
"# 3 \u304b\u3089\u59cb\u307e\u308a 10 \u306b\u306a\u308b\u307e\u3067 1 \u305a\u3064\u5897\u52a0\u3059\u308b\u6570\u5217\u3092\u4f5c\u308b\uff0810 \u306f\u542b\u307e\u306a\u3044\uff09\nf = np.arange(3, 10, 1)\n\nf",
"val = e[0, 1]\n\nval",
"# 4 x 5 \u884c\u5217 e \u306e\u771f\u3093\u4e2d\u306e 2 x 3 = 6 \u500b\u306e\u5024\u3092\u53d6\u308a\u51fa\u3059\ncenter = e[1:3, 1:4]\n\ncenter",
"print('Shape of e:', e.shape)\nprint('Shape of center:', center.shape)",
"# \u5148\u7a0b\u306e\u771f\u3093\u4e2d\u306e 6 \u500b\u306e\u5024\u3092 0 \u306b\u3059\u308b\ne[1:3, 1:4] = 0\n\ne",
"a = np.array(\n    [[1, 2, 3],\n     [4, 5, 6],\n     [7, 8, 9]]\n)\n\na",
"np.array([a[0, 1], a[2, 1], a[1, 0]])",
"a[[0, 2, 1], [1, 1, 0]]",
"# \u6574\u6570\uff08Python \u306e int \u578b\uff09\u306e\u8981\u7d20\u3092\u3082\u3064\u30ea\u30b9\u30c8\u3092\u4e0e\u3048\u305f\u5834\u5408\nx = np.array([1, 2, 3])\n\nx.dtype",
"# \u6d6e\u52d5\u5c0f\u6570\u70b9\u6570\uff08Python \u306e float \u578b\uff09\u306e\u8981\u7d20\u3092\u3082\u3064\u30ea\u30b9\u30c8\u3092\u4e0e\u3048\u305f\u5834\u5408\nx = np.array([1., 2., 3.])\n\nx.dtype",
"x = np.array([1, 2, 3], dtype=np.float32)\n\nx.dtype",
"x = np.array([1, 2, 3], dtype='float32')\n\nx.dtype",
"x = np.array([1, 2, 3], dtype='f')\n\nx.dtype",
"x = x.astype(np.float64)\n\nx.dtype",
"# \u540c\u3058\u5f62 (3 x 3) \u306e\u914d\u5217\u3092 2 \u3064\u5b9a\u7fa9\u3059\u308b\na = np.array([\n    [0, 1, 2],\n    [3, 4, 5],\n    [6, 7, 8]\n])\n\nb = np.array([\n    [1, 2, 3],\n    [4, 5, 6],\n    [7, 8, 9]\n])",
"# \u8db3\u3057\u7b97\nc = a + b\n\nc",
"# \u5f15\u304d\u7b97\nc = a - b\n\nc",
"# \u639b\u3051\u7b97\nc = a * b\n\nc",
"# \u5272\u308a\u7b97\nc = a / b\n\nc",
"# \u8981\u7d20\u3054\u3068\u306b\u5e73\u65b9\u6839\u3092\u8a08\u7b97\u3059\u308b\nc = np.sqrt(b)\n\nc",
"# \u8981\u7d20\u3054\u3068\u306b\u5024\u3092 n \u4e57\u3059\u308b\nn = 2\nc = np.power(b, n)\n\nc",
"c ** n",
"import numpy as np\n\na = 2 # \u50be\u304d\nb = 1 # \u5207\u7247\n\nx = np.arange(-5, 6) # -5 \u304b\u3089 6 \u307e\u3067 (6\u306f\u542b\u307e\u306a\u3044) \u306e\u6574\u6570\u3092\u4e26\u3079\u305f1\u6b21\u5143\u914d\u5217\ny = a * x + b\n\nfor i in range(len(x)):\n    print(f\"x: {x[i]:3}, y: {y[i]:3}\")\n",
"import matplotlib.pyplot as plt\n\nplt.scatter(x, y) # \u6563\u5e03\u56f3\u30b0\u30e9\u30d5\u3092\u63cf\u753b\nplt.show()",
"import matplotlib.pyplot as plt\n\nplt.scatter(x, y) # \u6563\u5e03\u56f3\u30b0\u30e9\u30d5\u3092\u63cf\u753b\nplt.savefig(\"scatter.png\")",
"plt.plot(x, y)\nplt.show()",
"a = 1/2\nb = -1\nc = 1\n\nx = np.arange(-5, 6)\ny = a * x**2 + b * x + c\n\nplt.plot(x, y)\nplt.show()",
"x = np.linspace(-5, 5, 100) # -5 \u304b\u3089 5 \u307e\u3067\u306e\u7bc4\u56f2\u3092 100 \u7b49\u5206\u3057\u305f1\u6b21\u5143\u914d\u5217\ny = a * x**2 + b * x + c\n\nplt.plot(x, y)\nplt.show()",
"alpha = -1\n\nx = np.linspace(-5, 5, 100)\ny1 = a * x**2 + b * x + c\ny2 = a * (x - alpha) **2 + b * (x - alpha) + c\n\nplt.plot(x, y1)\nplt.plot(x, y2)\nplt.show()",
"beta = 2\n\nx = np.linspace(-5, 5, 100)\ny1 = a * x**2 + b * x + c\ny2 = a * x**2 + b * x + (c + beta)\n\nplt.plot(x, y1)\nplt.plot(x, y2)\nplt.show()",
"import numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.linspace(0, 4 * np.pi, 100) # 0 \u304b\u3089 2\u03c0 \u307e\u3067\u306e\u7bc4\u56f2\u3092 100 \u7b49\u5206\u3057\u305f1\u6b21\u5143\u914d\u5217\ny1 = np.sin(x)\ny2 = np.cos(x)\n\nplt.plot(x, y1)\nplt.plot(x, y2)\nplt.show()",
"x = np.linspace(0, 4 * np.pi, 1000) # 0 \u304b\u3089 4\u03c0 \u307e\u3067\u306e\u7bc4\u56f2\u3092 1000 \u7b49\u5206\u3057\u305f1\u6b21\u5143\u914d\u5217\ny = np.tan(x)\nthreshold = 10\ny[y > threshold] = np.inf\ny[y < -threshold] = -np.inf\n\nplt.plot(x, y)\nplt.show()",
"x = np.linspace(-5, 5, 100) # -5 \u304b\u3089 5 \u307e\u3067\u306e\u7bc4\u56f2\u3092 100 \u7b49\u5206\u3057\u305f1\u6b21\u5143\u914d\u5217\na = 2\ny = a**x\n\nplt.plot(x, y)\nplt.show()",
"x = np.linspace(-5, 5, 100) # -5 \u304b\u3089 5 \u307e\u3067\u306e\u7bc4\u56f2\u3092 100 \u7b49\u5206\u3057\u305f1\u6b21\u5143\u914d\u5217\ny = np.exp(x)\n\nplt.plot(x, y)\nplt.show()",
"x = np.linspace(0.01, 5, 100) # -5 \u304b\u3089 5 \u307e\u3067\u306e\u7bc4\u56f2\u3092 100 \u7b49\u5206\u3057\u305f1\u6b21\u5143\u914d\u5217\ny = np.log2(x)\n\nplt.plot(x, y)\nplt.show()",
"a = np.array([1, 2, 3])\n\na",
"b = np.array([2, 2, 2])\n\nc = a * b\n\nc",
"c = a * 2\n\nc",
"# 0 ~ 9 \u306e\u7bc4\u56f2\u306e\u5024\u3092\u30e9\u30f3\u30c0\u30e0\u306b\u7528\u3044\u3066\u57cb\u3081\u3089\u308c\u305f (2, 1, 3) \u3068 (3, 1) \u3068\u3044\u3046\u5927\u304d\u3055\u306e\u914d\u5217\u3092\u4f5c\u308b\na = np.random.randint(0, 10, (2, 1, 3))\nb = np.random.randint(0, 10, (3, 1))\n\nprint('a:\\n', a)\nprint('\\na.shape:', a.shape)\nprint('\\nb:\\n', b)\nprint('\\nb.shape:', b.shape)\n\n# \u52a0\u7b97\nc = a + b\n\nprint('\\na + b:\\n', c)\nprint('\\n(a + b).shape:', c.shape)",
"print('Original shape:', b.shape)\n\nb_expanded = b[np.newaxis, :, :]\n\nprint('Added new axis to the top:', b_expanded.shape)\n\nb_expanded2 = b[:, np.newaxis, :]\n\nprint('Added new axis to the middle:', b_expanded2.shape)",
"b",
"b_expanded",
"b_expanded2",
"a = np.array([\n    [0, 1, 2, 1, 0],\n    [3, 4, 5, 4, 3],\n    [6, 7, 8, 7, 6],\n    [3, 4, 5, 4, 4],\n    [0, 1, 2, 1, 0]\n])\n\nb = np.array([1, 2, 3, 4, 5])\n\n# \u7d50\u679c\u3092\u683c\u7d0d\u3059\u308b\u914d\u5217\u3092\u5148\u306b\u4f5c\u308b\nc = np.empty((5, 5))",
"# %%timeit\n# for i in range(a.shape[0]):\n#     c[i, :] = a[i, :] + b",
"c",
"# %%timeit\n# c = a + b",
"c",
"x = np.random.randint(0, 10, (8, 10))\n\nx",
"# \u5e73\u5747\u5024\nx.mean()",
"# \u5206\u6563\nx.var()",
"# \u6a19\u6e96\u504f\u5dee\nx.std()",
"# \u6700\u5927\u5024\nx.max()",
"# \u6700\u5c0f\u5024\nx.min()",
"x.mean(axis=1)",
"np.array([\n    x[0, :].mean(),\n    x[1, :].mean(),\n    x[2, :].mean(),\n    x[3, :].mean(),\n    x[4, :].mean(),\n    x[5, :].mean(),\n    x[6, :].mean(),\n    x[7, :].mean(),\n])"
]
FIGURE_JOBS = [
    (35, ["figure-01.png"]),
    (36, ["figure-02.png"]),
    (37, ["figure-03.png"]),
    (38, ["figure-04.png"]),
    (39, ["figure-05.png"]),
    (40, ["figure-06.png"]),
    (41, ["figure-07.png"]),
    (42, ["figure-08.png"]),
    (43, ["figure-09.png"]),
    (44, ["figure-10.png"]),
    (45, ["figure-11.png"]),
    (46, ["figure-12.png"])
]


def main() -> None:
    """Execute notebook code cells sequentially and save any figures."""
    namespace: dict[str, object] = {}
    output_dir = Path(__file__).resolve().parent.parent / "generated" / "03-reviewing-elementary-math-with-numpy-and-matplotlib"
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
