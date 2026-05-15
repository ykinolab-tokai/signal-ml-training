# 第03回 数学基礎と NumPy, Matplotlib

- 初等関数を NumPy で計算し，Matplotlib でグラフにする。
- グラフの平行移動を式と図の両方で確認する。
- NumPy 配列の shape，dtype，スライスを確認する。
- 計算結果と図を file として保存する。

## 解説
- 初等関数は，入力 `x` に対して出力 `y` を返す対応として見る。NumPy 配列を入力にすると，多数の点で同時に関数値を計算できる。
- グラフの平行移動は，式の中の `x - a` と外側の `+ b` が，横方向と縦方向の移動に対応することを図で確認すると理解しやすい。
- NumPy 配列では，値だけでなく `shape`，`dtype`，次元数を確認する。後の信号処理と画像処理では，shape の読み間違いが実装ミスにつながる。
- Matplotlib の図は画面表示だけで終えず，`savefig` で保存する。保存された図と script を対応づけることで，結果を再現しやすくなる。

## 演習
### 基礎レベル（7問）
1. `scripts/`，`outputs/session03/`，`outputs/figures/`，`outputs/data/` を作成し，`scripts/session03_math_numpy_matplotlib.py` で `np.linspace(-5, 5, 501)` の `x` を作る。
2. `y = x`，`y = x**2`，`y = np.sin(x)`，`y = np.exp(-x**2)` を計算し，それぞれの先頭5要素，shape，dtype を表示する。
3. 4 つの関数を1枚の図に描き，凡例，軸ラベル，grid を付けて `outputs/figures/session03_elementary_functions.png` に保存する。
4. `f(x) = x**2`，`g(x) = (x - 2)**2 + 1`，`h(x) = (x + 1)**2 - 2` を描き，平行移動の方向を確認する。
5. `np.array([[1, 2, 3], [4, 5, 6]])` を作り，shape，dtype，1行目，2列目，右下の値を取り出す。
6. Python list に対する `+` と NumPy 配列に対する `+` の違いを，短い例で確認する。
7. `outputs/session03/session03_report.md` に，作成した図の file 名，配列の shape，平行移動の読み取りを書く。

### 発展レベル（7問）
1. `np.log(x)` と `np.sqrt(x)` を扱うとき，定義域の外で何が起きるか確認し，warning や `nan` の意味を書く。
2. `np.linspace` の点数を 21，101，1001 に変え，`sin(x)` の図の滑らかさがどう変わるか比較する。
3. `a = np.arange(12).reshape(3, 4)` を作り，行方向の平均，列方向の平均，全体平均を計算する。
4. broadcasting を使って，`a` の各列から列平均を引く。結果の各列平均が 0 に近いことを確認する。
5. 同じ関数を複数の subplot に分けて描き，1枚の図として保存する。
6. `np.savetxt` を使って，`x` と `sin(x)` の対応表を `outputs/data/session03_sin_table.csv` に保存する。
7. 第4回で数値微分に使えるように，`f(x)` を受け取って `x` と `y` を返す関数を1つ定義する。

## 詰まったときに見る資料
- [`../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md`](../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md)
- [`../textbook/markdown/ch08-applied-matplotlib.md`](../textbook/markdown/ch08-applied-matplotlib.md)
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)
