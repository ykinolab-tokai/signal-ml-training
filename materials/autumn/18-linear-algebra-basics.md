# 第18回 線形代数の基礎：ベクトル・内積・行列積・ノルム

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- ベクトル、行列、内積、行列積、ノルムをコードで計算できる。
- `shape` を見て、演算できる組合せとできない組合せを説明できる。
- 行列積の順序によって結果が変わる理由を説明できる。

## ミニ講義
- 線形代数の計算では、値を見る前に shape を見ると間違えにくい。特に行列積は、内側の次元が一致するかで可否が決まる。
- ベクトルノルムは「大きさ」、行列ノルムは「行列全体の大きさの代表値」と考えると入りやすい。今回のような小さな例では、値そのものを計算して感覚を掴むことが重要である。
- 行列積は一般に可換ではない。`A @ B` と `B @ A` は shape が作れても同じ値になるとは限らない。

## 演習
### 基礎レベル
1. `session18_linear_algebra_basics.py` を作成し、指定された `x`, `y`, `A`, `B`, `v` の tensor を定義して shape を表示する。
2. `torch.dot`, `A @ v`, `A @ B`, vector norm, matrix norm を計算し、値と shape を対応づけて表示する。
3. `A @ x` を `try` / `except` で実行し、shape mismatch が起きる理由を確認する。
4. `session18_linear_algebra_report.md` に `## shape 一覧`, `## 計算結果`, `## shape mismatch の原因` を書く。

### 発展レベル
1. `B @ A`, `A @ v`, `B @ v` を追加で計算する。
2. report に `## 行列積の順序` を追加し、`AB` と `BA` の値が違う理由、`A` と `B` がベクトルへ与える変化の違いを説明する。

## 確認ポイント
- `x` と `y` の shape が `(3,)`、`A` と `B` の shape が `(2, 2)`、`v` の shape が `(2,)` である。
- `dot_xy` が `32.0`、`Av` が `tensor([-1., -1.])`、`AB` が `tensor([[2., 1.], [4., 3.]])` である。
- report に、shape mismatch の原因と、行列積の順序の違いに関する説明がある。

## 詰まったときに見る資料
- [`17-array-and-tensor-basics.md`](17-array-and-tensor-basics.md)
- [`../../latex/markdown/ch06a-basics-of-linear-algebra.md`](../../latex/markdown/ch06a-basics-of-linear-algebra.md)
