# 第16回 Python・NumPy・matplotlib の基礎

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- Python script を 1 本作り、NumPy 配列の `dtype` と `shape` を確認しながら実行できる。
- indexing と broadcasting をコード中で使い、その結果を説明できる。
- 複数の関数グラフを 1 枚の図へ保存し、見え方を解釈できる。

## ミニ講義
- 配列計算では、値そのものだけでなく `dtype` と `shape` が重要である。どんな演算ができるかは shape に依存し、結果の型は dtype に依存する。
- indexing は必要な要素を取り出す操作、broadcasting は shape の違う配列を規則に従って拡張して演算する仕組みである。両者を区別すると、NumPy の挙動が読みやすくなる。
- 複数の曲線を同じ図へ描くときは、凡例、軸ラベル、タイトルがないと比較できない。図は「保存できたか」だけでなく「読み取れるか」で評価する。

## 演習
### 基礎レベル
1. `session16_numpy_matplotlib_basics.py` を作成し、`from pathlib import Path`, `import numpy as np`, `import matplotlib.pyplot as plt` の 3 行を書く。
2. `outputs/figures/` を作るコードを入れる。
```python
out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
```
3. 次の 2 つの配列をそのまま定義する。
```python
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
b = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
```
4. `a` と `b` の `dtype` と `shape` を `print()` で表示する。
5. indexing として `a[1, 2]` と `a[:, 1]` を表示する。
6. broadcasting として次を実行し、結果 `c` を表示する。
```python
c = a + np.array([10, 20, 30], dtype=np.int64)
```
7. `x = np.linspace(-2.0, 2.0, 400, dtype=np.float64)` を使い、`sin(x)`, `cos(x)`, `exp(x)` の 3 本を 1 枚の図に描く。
8. タイトル `Session 16 Functions`、x 軸ラベル `x`、y 軸ラベル `y`、凡例ラベル `sin(x)`, `cos(x)`, `exp(x)` を必ず入れ、`outputs/figures/session16_functions.png` に保存する。
9. `session16_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## dtype と shape`
   - `## indexing と broadcasting`
   - `## 保存した図の説明`

### 発展レベル
1. `session16_report.md` に `## 同じ図に重ねたときの見え方` を追加し、`sin(x)`、`cos(x)`、`exp(x)` を同じ y 軸で描いたときに見えやすい曲線と見えにくい曲線を 2 行以内で書く。
2. `x > 0` の範囲で `exp(x)` が他の 2 本より大きくなりやすい理由を、値の増え方に触れて 2 行以内で説明する。
3. `a[:, 1]` と `c` の結果を見比べて、「indexing は何を取り出し、broadcasting は何を広げているのか」を 2 行で説明する。

## 確認ポイント
- `a` の `dtype` が `int64`、`shape` が `(2, 3)` である。
- `b` の `dtype` が `float64`、`shape` が `(2, 3)` である。
- `a[1, 2]` が `6`、`a[:, 1]` が `[2 5]` である。
- `c` が `[[11 22 33] [14 25 36]]` になる。
- report に、図の保存確認だけでなく、曲線の見え方に関する説明がある。

## 詰まったときに見る資料
- [`../../latex/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md`](../../latex/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md)
- [`../../latex/markdown/ch08-applied-matplotlib.md`](../../latex/markdown/ch08-applied-matplotlib.md)
