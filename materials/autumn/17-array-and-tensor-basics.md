# 第17回 NumPy・PyTorch Tensor・matplotlib の基礎

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- NumPy 配列と PyTorch Tensor の `shape` と `dtype` を確認できる。
- indexing、broadcasting、reshape、転置の結果を説明できる。
- 計算結果を matplotlib で図にして保存し、図から読み取れることを説明できる。

## ミニ講義
- 配列計算では、値そのものだけでなく `shape` と `dtype` が重要である。どんな演算ができるかは shape に依存し、結果の型は dtype に依存する。
- indexing は必要な要素を取り出す操作、broadcasting は shape の違う配列を規則に従って拡張して演算する仕組みである。両者を区別すると、NumPy と PyTorch の挙動が読みやすくなる。
- 第18回以降では `torch.Tensor` を使って線形代数、統計、信号処理を扱う。ここでは `@`, reshape, 転置, plot 保存までを入口として確認する。

## 演習
### 基礎レベル
1. `session17_array_tensor_basics.py` を作成し、同じ値 `[[1, 2, 3], [4, 5, 6]]` から `np.ndarray` と `torch.Tensor` を作る。両方の `shape` と `dtype` を表示する。
2. indexing で 1 行目、2 列目、右下の値を取り出し、NumPy と PyTorch の結果を並べて表示する。
3. NumPy では `arr + np.array([10, 20, 30])`、PyTorch では `tensor + torch.tensor([10, 20, 30])` を計算し、同じ broadcasting が起きることを確認する。結果の shape と値を表示する。
4. `m = torch.arange(6).reshape(2, 3)` を作り、`m.T`、`m @ m.T`、`m.flatten()` の shape と値を表示する。
5. `x = np.linspace(0, 2 * np.pi, 100)` に対して `sin(x)` と `cos(x)` を同じ図に描き、`outputs/figures/session17_sin_cos.png` に保存する。
6. `session17_array_tensor_report.md` に `## ndarray と Tensor`, `## shape と dtype`, `## indexing と broadcasting`, `## 保存した図の解釈` を書く。

### 発展レベル
1. NumPy と PyTorch で同じ計算を書いたとき、型変換が必要になる場面を 1 つ作り、report に `## 型変換で詰まりやすい点` として説明する。
2. `reshape(3, 2)` に変えた場合、転置と行列積で何が変わるかを比較する。

## 確認ポイント
- NumPy 配列と PyTorch Tensor の `shape` と `dtype` が表示されている。
- indexing、broadcasting、reshape、転置、行列積の結果が表示されている。
- `outputs/figures/session17_sin_cos.png` が保存されている。
- report に、shape と dtype、保存した図の解釈が書かれている。

## 詰まったときに見る資料
- [`../../README.md`](../../README.md)
- [`../../latex/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md`](../../latex/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md)
- [`../../latex/markdown/ch08-applied-matplotlib.md`](../../latex/markdown/ch08-applied-matplotlib.md)
