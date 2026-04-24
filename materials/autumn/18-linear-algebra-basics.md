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
1. `session18_linear_algebra_basics.py` を作成し、次の tensor をそのまま定義する。
```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
v = torch.tensor([1.0, -1.0])
```
2. `x`, `y`, `A`, `B`, `v` の shape を表示する。
3. 次の 5 つを計算し、変数名付きで表示する。
```python
dot_xy = torch.dot(x, y)
Av = A @ v
AB = A @ B
x_norm = torch.linalg.vector_norm(x)
A_norm = torch.linalg.matrix_norm(A)
```
4. `bad = A @ x` を `try` / `except` で囲み、shape mismatch のエラーメッセージを表示する。
5. `session18_linear_algebra_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## shape 一覧`
   - `## 計算結果`
   - `## shape mismatch の原因`
6. `## shape mismatch の原因` には `A @ x` が失敗する理由を 2 行以内で書く。

### 発展レベル
1. `BA = B @ A` を追加で計算する。
2. `session18_linear_algebra_report.md` に `## 行列積の順序` を追加し、`AB` と `BA` の値を書く。
3. そのうえで、「どちらも計算できるのに結果が違う理由」を 3 行以内で説明する。
4. `Av` と `B @ v` を見比べて、`A` と `B` がベクトルへ与えている変化の違いを 2 行で書く。

## 確認ポイント
- `x` と `y` の shape が `(3,)`、`A` と `B` の shape が `(2, 2)`、`v` の shape が `(2,)` である。
- `dot_xy` が `32.0`、`Av` が `tensor([-1., -1.])`、`AB` が `tensor([[2., 1.], [4., 3.]])` である。
- report に、shape mismatch の原因と、行列積の順序の違いに関する説明がある。

## 詰まったときに見る資料
- [`16-python-numpy-matplotlib-basics.md`](16-python-numpy-matplotlib-basics.md)
- [`../../latex/markdown/ch06a-basics-of-linear-algebra.md`](../../latex/markdown/ch06a-basics-of-linear-algebra.md)
