# 第05回 線形代数

## この回の目標

ベクトル・行列の演算を NumPy で実装し，幾何学的な意味を説明できる。

## 解説

この回では，第3回で扱った NumPy と Matplotlib を使い，ベクトルと行列の演算を計算して，その幾何学的な意味を図で確かめる。
線形代数は，信号処理や機械学習における基礎的なツールであり，極めて重要である。
スクリプトの先頭には，第3回と同様に次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figure_dir = Path("outputs/exc05")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. ベクトルと行列

#### ベクトル

**ベクトル (vector)** は，数を一列に並べたものである。
並べた数をベクトルの **成分 (component)**，成分の個数をベクトルの **次元 (dimension)** という。
次式の $\boldsymbol{v}$ のように成分を縦に並べたベクトルを **列ベクトル (column vector)** といい，
$\boldsymbol{u}$ のように成分を横に並べたベクトルを **行ベクトル (row vector)** という。
この資料では，ベクトルを太字の小文字で表し，特に明記しない限り列ベクトルとして扱う。

$$
\boldsymbol{v} = \begin{pmatrix} v_1 \\ v_2 \end{pmatrix}, \quad
\boldsymbol{u} = (u_1, u_2)
$$

成分が実数である $n$ 次元のベクトル全体を $\mathbb{R}^n$ と書く。
2 次元のベクトル $(v_1, v_2)^\top$ は，平面上で原点から点 $(v_1, v_2)$ へ向かう矢印として図示できる。

NumPy では，ベクトルを 1 次元配列で表す。
例えば，`v = np.array([2, 1])` の形は `(2,)` である。
第3回で見たように，1 次元配列は行と列の区別を持たない。
縦と横を区別する必要がある場合は，`v.reshape(2, 1)`（形 `(2, 1)` の列ベクトル）や `v.reshape(1, 2)`（形 `(1, 2)` の行ベクトル）のように 2 次元配列にする。

#### 行列と転置

**行列 (matrix)** は，以下の $\boldsymbol{A}$ のように数を縦横の2方向に並べたものである。
横の並びを **行 (row)**，縦の並びを **列 (column)** といい，$m$ 個の行と $n$ 個の列を持つ行列を $m\times n$ 行列という。
上述した列ベクトルは $m \times 1$ の行列として，行ベクトルは $1 \times n$ の行列としてみなすことができる。
この資料では，行列を太字の大文字で表し，$i$ 行 $j$ 列の成分を $a_{ij}$ と書く。

$$
\boldsymbol{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{pmatrix}
$$

行列の行と列を入れ替える操作を **転置 (transpose)** といい，$\boldsymbol{A}$ の転置を $\boldsymbol{A}^\top$ と書く。
$m\times n$ 行列の転置は $n\times m$ 行列であり，その $i$ 行 $j$ 列の成分は $a_{ji}$ である。
転置を用いると列ベクトルを $\boldsymbol{v} = (v_1, v_2)^\top$ として表記できる。
この記法は，列ベクトルを縦方向の長さを抑えて記述できるため，よく用いられる。

NumPy では，行列を 2 次元配列で表し，`shape` は `(行数, 列数)` となる。
NumPy の添字は 0 から始まるため，数式の $a_{ij}$ は `A[i - 1, j - 1]` に対応する。
また，NumPy では転置を `A.T` で計算する。

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape, A.T.shape)  # (2, 3) (3, 2)
print(A.T)
```

```text
[[1 4]
 [2 5]
 [3 6]]
```

NumPy の 1 次元配列は行と列の区別がないため，
1 次元配列 `v` に対する `v.T` は `v` と同じ形 `(2,)` の配列を返すことに注意が必要である。

#### 行列が等しい条件

2 つの行列の行数と列数がそれぞれ同じで対応する位置の成分がすべて等しいとき，それらの **行列は等しい** といい，$\boldsymbol{A}=\boldsymbol{B}$ と書く。
すなわち，ともに $m\times n$ の行列 $\boldsymbol{A}$，$\boldsymbol{B}$ について，

$$
\boldsymbol{A}=\boldsymbol{B}
\quad\Longleftrightarrow\quad
a_{ij}=b_{ij}\quad(i=1,\ldots,m,\ j=1,\ldots,n)
$$

である。
例えば，$\begin{pmatrix}a&2\\3&b\end{pmatrix}=\begin{pmatrix}1&2\\3&4\end{pmatrix}$ が成り立つのは，$a=1$ かつ $b=4$ のときである。

#### 和・スカラー倍・線形結合

同じ形のベクトルや行列の **和** と **差** は，対応する成分どうしの和と差として次式の通り定義される。
また，スカラー $c$ をベクトルや行列に掛ける **スカラー倍** は，すべての成分を $c$ 倍する演算として定義される。

$$
\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
+ \begin{pmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{pmatrix}
= \begin{pmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{pmatrix}
$$

$$
c\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
= \begin{pmatrix} c\,a_{11} & c\,a_{12} \\ c\,a_{21} & c\,a_{22} \end{pmatrix}
$$

第3回で扱った通り，NumPy の計算は基本的に要素ごとの計算であるため，ベクトルや行列の和やスカラー倍は `A + B` や `c * A` で計算できる。

```python
A = np.array([[2, 1], [0, 3]])
B = np.array([[-1, 2], [4, -2]])
print(A + B)
print(2 * A)
print(A - 2 * B)
```

```text
[[1 3]
 [4 1]]
[[4 2]
 [0 6]]
[[ 4 -3]
 [-8  7]]
```

いくつかのベクトルをそれぞれスカラー倍して足し合わせた $c_1\boldsymbol{v}_1 + c_2\boldsymbol{v}_2 + \cdots + c_k\boldsymbol{v}_k$ を，$\boldsymbol{v}_1, \ldots, \boldsymbol{v}_k$ の **線形結合 (linear combination)** という。
例えば，$\boldsymbol{e}_1 = (1, 0)^\top$，$\boldsymbol{e}_2 = (0, 1)^\top$ とすると，任意の 2 次元ベクトルは $(v_1, v_2)^\top = v_1\boldsymbol{e}_1 + v_2\boldsymbol{e}_2$ という線形結合で表せる。

次のコードでは，係数がともに 1 の線形結合 `a + b` を図示する。

```python
a = np.array([2, 1])
b = np.array([-1, 2])
s = a + b
fig, ax = plt.subplots()
ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="tab:blue", label="a")
ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="tab:orange", label="b")
ax.quiver(0, 0, s[0], s[1], angles="xy", scale_units="xy", scale=1, color="tab:green", label="a + b")
ax.set_xlim(-2, 4)
ax.set_ylim(-1, 4)
ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session05_example_vectors.png")
plt.show()
plt.close(fig)
```

`ax.quiver(x0, y0, u, v, angles="xy", scale_units="xy", scale=1)` は，点 $(x_0, y_0)$ から成分 $(u, v)$ の矢印を描く。
`angles="xy"`，`scale_units="xy"`，`scale=1` を指定すると，矢印の成分が座標軸の目盛りと同じ単位で扱われ，矢印の終点は $(x_0+u, y_0+v)$ になる。
`quiver` は矢印の終点に合わせて表示範囲を広げないため，`set_xlim` と `set_ylim` で範囲を指定している。
`ax.set_aspect("equal")` は，縦軸と横軸の目盛りの間隔をそろえる。
目盛りの間隔が縦と横で異なると，図の上で矢印の長さや角度が正しく表示されないことに注意する。

### 2. 内積とノルム

#### 内積

同じ次元の 2 つの実ベクトル $\boldsymbol{u}$，$\boldsymbol{v} \in \mathbb{R}^n$ の **内積 (inner product)** を，対応する成分どうしの積の和として次式の通り定義する。

$$
\boldsymbol{u}\cdot\boldsymbol{v} = u_1v_1 + u_2v_2 + \cdots + u_n v_n = \sum_{i=1}^{n} u_i v_i
$$

内積は，$\langle\boldsymbol{u}, \boldsymbol{v}\rangle$ と表記されることもある。
内積は2つのベクトル間の演算であるが，その結果はベクトルではなくスカラーとなる。

NumPy では，内積を `np.dot(u, v)` で計算する。
`u * v` は成分ごとの積を並べた配列となるため，その和 `np.sum(u * v)` として内積を計算することもできる。

#### ノルムと単位ベクトル

ベクトルの長さを **ノルム (norm)** といい，$\lVert\boldsymbol{v}\rVert$ と書く。
ノルムは，任意の実ベクトル $\boldsymbol{u},\boldsymbol{v}\in\mathbb{R}^n$ と任意の実数 $c$ に対して，次の 3 つの条件（**ノルムの公理**）を満たす。
ここで，$\boldsymbol{0}$ はすべての成分が 0 の零ベクトルを表す。

1. **正値性**：値は 0 以上であり，0 になるのは零ベクトルのときに限る。

    $$
    \lVert\boldsymbol{v}\rVert\ge0,\qquad
    \lVert\boldsymbol{v}\rVert=0\quad\Longleftrightarrow\quad\boldsymbol{v}=\boldsymbol{0}
    $$

2. **絶対斉次性**：ベクトルを $c$ 倍すると，ノルムは $|c|$ 倍になる。

    $$
    \lVert c\boldsymbol{v}\rVert=|c|\,\lVert\boldsymbol{v}\rVert
    $$

3. **三角不等式**：和のノルムは，それぞれのノルムの和以下である。

    $$
    \lVert\boldsymbol{u}+\boldsymbol{v}\rVert\le\lVert\boldsymbol{u}\rVert+\lVert\boldsymbol{v}\rVert
    $$

自分自身との内積の平方根 $\sqrt{\boldsymbol{v} \cdot \boldsymbol{v}}$ は，これらの公理を満たすノルムの 1 つである（内積から導かれるノルム）。
前述の内積から導かれるノルムは次式で与えられ，ユークリッドノルムやL2ノルムと呼ばれる。

$$
\lVert\boldsymbol{v}\rVert = \sqrt{\boldsymbol{v}\cdot\boldsymbol{v}}
= \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

NumPy では，ノルムを `np.linalg.norm(v)` で計算する。
ノルムが 1 のベクトルを **単位ベクトル (unit vector)** という。
零ベクトルでない $\boldsymbol{v}$ をそのノルムで割った $\boldsymbol{v}/\lVert\boldsymbol{v}\rVert$ は，$\boldsymbol{v}$ と同じ向きの単位ベクトルになる。

#### なす角と直交

零ベクトルでない 2 つのベクトルのなす角 $\theta$（$0\le\theta\le\pi$）を，内積が導くノルムを用いて次式を満たすものとして定義する。

$$
\cos\theta = \frac{\boldsymbol{u}\cdot\boldsymbol{v}}{\lVert\boldsymbol{u}\rVert\,\lVert\boldsymbol{v}\rVert}
$$

ノルムは正なので，内積の符号は $\cos\theta$ の符号と一致する。
内積が正なら $0\le\theta<\pi/2$，負なら $\pi/2<\theta\le\pi$，0 なら $\theta=\pi/2$ である。
内積が 0 になる 2 つのベクトルは **直交する (orthogonal)** という。

$\theta=0$ のとき，2 つのベクトルは同じ向きであり，$\cos\theta=1$ となる。
$\theta=\pi$（$180^\circ$）のとき，2 つのベクトルは平行で向きが反対であり，$\cos\theta=-1$ となる。
例えば，$\boldsymbol{u}=(1,0)^\top$ と $\boldsymbol{v}=(-2,0)^\top$ のなす角は $\pi$ である。

$\cos\theta$ は，2 つのベクトルの近さをそれらの長さによらずに測る量と考えることができ，これを **コサイン類似度 (cosine similarity)** と呼ぶことがある。

```python
a = np.array([2, 1])
b = np.array([-1, 2])
c = np.array([3, 4])
print(np.dot(a, b), np.dot(a, c))  # 0 10
print(np.linalg.norm(c))           # 5.0
cos = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
print(cos)                         # 約 0.894
print(np.degrees(np.arccos(cos)))  # 約 26.57
print(c / np.linalg.norm(c))       # [0.6 0.8]
```

`np.dot(a, b)` が 0 なので，`a` と `b` は直交する。
`np.arccos` は $\cos\theta$ の値から角度 $\theta$ をラジアンで返し，`np.degrees` はラジアンを度数法に変換する。
丸め誤差によって `cos` が 1 をわずかに超えると，`np.arccos` は `nan` を返す。
このような場合は，`np.clip(cos, -1, 1)` で値を $[-1, 1]$ の範囲に収めてから `np.arccos` に渡す。


### 3. 行列積

#### 行列積の定義

$l\times m$ 行列 $\boldsymbol{A}$ と $m\times n$ 行列 $\boldsymbol{B}$ の **行列積 (matrix product)** を，
$\boldsymbol{A}$ の第 $i$ 行と $\boldsymbol{B}$ の第 $j$ 列の内積をすべての $i, j$ について計算する演算として定義し，
$\boldsymbol{AB}$ と書く。
行列積 $\boldsymbol{A}\boldsymbol{B}$ は $l\times n$ 行列であり，
その $i$ 行 $j$ 列の成分 $(\boldsymbol{AB})_{ij}$ は次式で与えられる。

$$
(\boldsymbol{A}\boldsymbol{B})_{ij} = \sum_{k=1}^{m} a_{ik}b_{kj}
$$

行列積が定義されるのは，左の行列の行ベクトルと右の行列の列ベクトルの内積が計算できるとき，すなわち，
左の行列の列数と右の行列の行数が等しいときだけである。

![行列積の計算例：A の第 1 行と B の第 1 列の内積が AB の 1 行 1 列の成分になる](../textbook/figures/source/06a-basics-of-linear-algebra/05_matrix_product.png)

NumPy では，行列積を計算する演算子として `@` が用意されており，`A @ B` のように行列積を計算できる。
形 `(l, m)` と `(m, n)` の配列の `@` の結果の形は `(l, n)` となる。
行列積が定義されない配列の組が与えられたときには `ValueError` が発生する。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)
print(B @ A)
print(A * B)
```


```text
[[19 22]
 [43 50]]
[[23 34]
 [31 46]]
[[ 5 12]
 [21 32]]
```

上の例からわかるように，一般に $\boldsymbol{A}\boldsymbol{B}\neq\boldsymbol{B}\boldsymbol{A}$ であり，行列積では掛ける順序を入れ替えられない。
また，`A * B` は要素ごとの積であり，行列積とは異なる。

転置と行列積の間には，$(\boldsymbol{A}\boldsymbol{B})^\top = \boldsymbol{B}^\top\boldsymbol{A}^\top$ という関係が成り立つ。

#### 行列・ベクトル積

$n$ 次元の列ベクトル $\boldsymbol{x}$ は $n\times1$ 行列なので，$m\times n$ 行列 $\boldsymbol{A}$ との積 $\boldsymbol{A}\boldsymbol{x}$ も，上で定義した行列積として計算できる。
結果は $m\times1$ 行列，すなわち $m$ 次元の列ベクトルになる。
$\boldsymbol{A}$ の列ベクトルを $\boldsymbol{a}_1,\ldots,\boldsymbol{a}_n$ とすると，この積は次の線形結合として表せる。

$$
\boldsymbol{A}\boldsymbol{x}
= \begin{pmatrix}\boldsymbol{a}_1 & \cdots & \boldsymbol{a}_n\end{pmatrix}\boldsymbol{x}
= x_1\boldsymbol{a}_1+x_2\boldsymbol{a}_2+\cdots+x_n\boldsymbol{a}_n
$$

例えば，2×2 行列の場合は，行列積の各成分を列ごとにまとめ直すと，次のようになる。

$$
\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
= \begin{pmatrix} a_{11}x_1+a_{12}x_2 \\ a_{21}x_1+a_{22}x_2 \end{pmatrix}
= x_1\begin{pmatrix}a_{11}\\a_{21}\end{pmatrix}
+ x_2\begin{pmatrix}a_{12}\\a_{22}\end{pmatrix}
$$

つまり，$\boldsymbol{A}\boldsymbol{x}$ は，$\boldsymbol{A}$ の各列を，$\boldsymbol{x}$ の対応する成分でスカラー倍して足し合わせたものだと見なすことができる。

同じ次元の列ベクトル $\boldsymbol{u}$，$\boldsymbol{v}$ に対して，$1\times n$ 行列 $\boldsymbol{u}^\top$ と $n\times1$ 行列 $\boldsymbol{v}$ の積は，内積 $\boldsymbol{u}\cdot\boldsymbol{v}$ を唯一の成分とする $1\times1$ 行列になる。
このため，内積は $\boldsymbol{u}^\top\boldsymbol{v}$ とも書くことができ，以降ではこの表記も用いる。

### 4. 線形変換

#### 行列による平面の変換

2×2 行列 $\boldsymbol{A}$ を使って，平面上の点 $\boldsymbol{x}$ を点 $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}$ に移す対応を考える。
行列積の定義から，任意のベクトル $\boldsymbol{x}_1$，$\boldsymbol{x}_2$ と数 $c_1$，$c_2$ について次の式が成り立つ。

$$
\boldsymbol{A}(c_1\boldsymbol{x}_1 + c_2\boldsymbol{x}_2) = c_1\boldsymbol{A}\boldsymbol{x}_1 + c_2\boldsymbol{A}\boldsymbol{x}_2
$$

この性質を **線形性 (linearity)** といい，線形性を持つ対応を **線形変換 (linear transformation)** という。
第4回で扱った微分の線形性と同じ性質である。
線形変換では，原点は原点に移る。
また，逆行列を持つ（第6節）行列による変換では，直線は直線に，平行な直線は平行な直線に移る。

$\boldsymbol{x} = x_1\boldsymbol{e}_1 + x_2\boldsymbol{e}_2$ と書くと，線形性から $\boldsymbol{A}\boldsymbol{x} = x_1\boldsymbol{A}\boldsymbol{e}_1 + x_2\boldsymbol{A}\boldsymbol{e}_2$ となる。
$\boldsymbol{A}\boldsymbol{e}_1$ と $\boldsymbol{A}\boldsymbol{e}_2$ は，それぞれ $\boldsymbol{A}$ の第 1 列と第 2 列である。
つまり，行列の各列は $\boldsymbol{e}_1$ と $\boldsymbol{e}_2$ の移り先を表し，その 2 つが分かれば平面全体の移り方が決まる。

#### 回転・拡大縮小・せん断

原点を中心に角度 $\theta$ だけ反時計回りに回転する変換 $\boldsymbol{R}(\theta)$，横方向に $s_1$ 倍・縦方向に $s_2$ 倍する変換 $\boldsymbol{S}$，各点を高さに比例して横にずらす **せん断 (shear)** $\boldsymbol{H}$ は，それぞれ次の行列で表される。

$$
\boldsymbol{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix},\qquad
\boldsymbol{S} = \begin{pmatrix} s_1 & 0 \\ 0 & s_2 \end{pmatrix},\qquad
\boldsymbol{H} = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}
$$

$\boldsymbol{R}(\theta)$ の第 1 列 $(\cos\theta, \sin\theta)^\top$ は，第3回の単位円の上で，$\boldsymbol{e}_1$ を角度 $\theta$ だけ回転した点である。
第 2 列 $(-\sin\theta, \cos\theta)^\top$ は，$\boldsymbol{e}_2$ を同じ角度だけ回転した点である。
$\boldsymbol{H}$ は $\boldsymbol{e}_1$ を動かさず，$\boldsymbol{e}_2$ を $(k, 1)^\top$ に移すため，点 $(x_1, x_2)$ を横に $kx_2$ だけずらす。
回転は長さと角度を変えないが，拡大縮小とせん断は一般に長さや角度を変える。

多数の点をまとめて変換するには，点を行として並べた形 `(N, 2)` の配列を使う。

```python
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])  # 単位正方形の頂点（最後に始点を繰り返す）
A = np.array([[2, 1], [1, 1]])
moved = square @ A.T
print(moved)

fig, ax = plt.subplots()
ax.plot(square[:, 0], square[:, 1], label="original")
ax.plot(moved[:, 0], moved[:, 1], label="transformed")
ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session05_example_transform.png")
plt.show()
plt.close(fig)
```

```text
[[0 0]
 [2 1]
 [3 2]
 [1 1]
 [0 0]]
```

各点 $\boldsymbol{p}$ を行ベクトル $\boldsymbol{p}^\top$ として並べた配列 `P` に対して，すべての点に $\boldsymbol{A}$ を掛けた結果は `P @ A.T` で計算できる。
これは，各行について $(\boldsymbol{A}\boldsymbol{p})^\top = \boldsymbol{p}^\top\boldsymbol{A}^\top$ が成り立つためである。
`A @ square` と書くと，形 `(2, 2)` と `(5, 2)` の内側の長さがそろわないため，`ValueError` になる。
この例では，単位正方形が，$\boldsymbol{A}$ の 2 つの列 $(2, 1)^\top$，$(1, 1)^\top$ を 2 辺とする平行四辺形に移る。

#### 合成変換

変換 $\boldsymbol{A}$ を施した後に変換 $\boldsymbol{B}$ を施すと，$\boldsymbol{B}(\boldsymbol{A}\boldsymbol{x}) = (\boldsymbol{B}\boldsymbol{A})\boldsymbol{x}$ となる。
つまり，2 つの線形変換を続けて施す **合成変換** は，行列積 $\boldsymbol{B}\boldsymbol{A}$ で表される。
先に施す変換の行列が右に来ることに注意する。
行列積では順序を入れ替えられないため，変換の順序を入れ替えると，一般に結果も変わる。

#### 行列式と面積

2×2 行列に対して，

$$
\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
$$

を **行列式 (determinant)** という。
単位正方形は，$\boldsymbol{A}$ による変換で，$\boldsymbol{A}$ の 2 つの列を 2 辺とする平行四辺形に移る。
その面積は $\lvert\det\boldsymbol{A}\rvert$ である。
一般に，$\boldsymbol{A}$ による変換で，図形の面積は $\lvert\det\boldsymbol{A}\rvert$ 倍になる。
$\det\boldsymbol{A}<0$ のときは，図形が裏返る（頂点を反時計回りにたどる順序が，変換後は時計回りになる）。
$\det\boldsymbol{A}=0$ のときは，平面全体が直線または 1 点につぶれる。
NumPy では，行列式を `np.linalg.det(A)` で計算する。
例えば，上の例の `np.linalg.det(A)` は `1.0` であり，平行四辺形の面積は単位正方形と同じ 1 である。
`np.linalg.det` の結果は浮動小数点数であり，`np.linalg.det(np.array([[1, 2], [3, 4]]))` が `-2.0000000000000004` になるように，丸め誤差を含むことがある。

### 5. 射影

#### ベクトルへの射影

零ベクトルでないベクトル $\boldsymbol{u}$ の向きに伸びる，原点を通る直線を考える。
ベクトル $\boldsymbol{v}$ の終点からこの直線に垂線を下ろしたとき，原点から垂線の足へ向かうベクトルを，$\boldsymbol{v}$ の $\boldsymbol{u}$ への **射影 (projection)** という。
射影は直線上にあるので，数 $c$ を使って $c\,\boldsymbol{u}$ と書ける。
$\boldsymbol{v}$ と射影の差 $\boldsymbol{r} = \boldsymbol{v} - c\,\boldsymbol{u}$ を **残差 (residual)** という。
垂線を下ろしたので，残差は $\boldsymbol{u}$ と直交する。
この条件から $c$ が求まる。

$$
\boldsymbol{u}^\top(\boldsymbol{v} - c\,\boldsymbol{u}) = 0
\quad\Longrightarrow\quad
c = \frac{\boldsymbol{u}^\top\boldsymbol{v}}{\boldsymbol{u}^\top\boldsymbol{u}},
\qquad
\mathrm{proj}_{\boldsymbol{u}}\boldsymbol{v} = \frac{\boldsymbol{u}^\top\boldsymbol{v}}{\boldsymbol{u}^\top\boldsymbol{u}}\,\boldsymbol{u}
$$

このとき，$\boldsymbol{v} = \mathrm{proj}_{\boldsymbol{u}}\boldsymbol{v} + \boldsymbol{r}$ であり，$\boldsymbol{v}$ は「$\boldsymbol{u}$ の向きの成分」と「$\boldsymbol{u}$ に直交する成分」に分解される。
射影は，直線上の点のうち $\boldsymbol{v}$ に最も近い点でもある。
実際，直線上の点 $t\,\boldsymbol{u}$ と $\boldsymbol{v}$ の距離の 2 乗 $\lVert\boldsymbol{v} - t\,\boldsymbol{u}\rVert^2 = \boldsymbol{v}^\top\boldsymbol{v} - 2t\,\boldsymbol{u}^\top\boldsymbol{v} + t^2\,\boldsymbol{u}^\top\boldsymbol{u}$ は $t$ の 2 次関数であり，第4回のように $t$ で微分して 0 とおくと，上と同じ条件が得られる。
したがって，残差のノルム $\lVert\boldsymbol{r}\rVert$ は，$\boldsymbol{v}$ の終点と直線との距離である。
「最も近い点を，残差が直交するという条件から求める」という考え方は，第16回の最小二乗法で使う。

```python
u = np.array([3, 1])
v = np.array([1, 2])
c = (u @ v) / (u @ u)
proj = c * u
r = v - proj
print(c)                  # 0.5
print(proj, r)            # [1.5 0.5] [-0.5  1.5]
print(u @ r)              # 0.0
print(np.linalg.norm(r))  # 約 1.581
```

`u @ r` が 0 になることから，残差が $\boldsymbol{u}$ と直交していることを確かめられる。
一般には，丸め誤差のため `u @ r` は厳密な 0 ではなく 0 に近い値になるので，`np.isclose(u @ r, 0)` のように判定する。

#### 射影行列

射影の式は，次のように行列とベクトルの積に書き直せる。

$$
\mathrm{proj}_{\boldsymbol{u}}\boldsymbol{v}
= \boldsymbol{u}\,\frac{\boldsymbol{u}^\top\boldsymbol{v}}{\boldsymbol{u}^\top\boldsymbol{u}}
= \frac{\boldsymbol{u}\boldsymbol{u}^\top}{\boldsymbol{u}^\top\boldsymbol{u}}\,\boldsymbol{v}
= \boldsymbol{P}\boldsymbol{v},
\qquad
\boldsymbol{P} = \frac{\boldsymbol{u}\boldsymbol{u}^\top}{\boldsymbol{u}^\top\boldsymbol{u}}
$$

$\boldsymbol{P}$ を，$\boldsymbol{u}$ の向きの直線への **射影行列 (projection matrix)** という。
$\boldsymbol{u}\boldsymbol{u}^\top$ は，$n\times1$ 行列（列ベクトル）と $1\times n$ 行列（行ベクトル）の行列積なので，$n\times n$ 行列である。
1 次元配列 `u` から $\boldsymbol{u}\boldsymbol{u}^\top$ を作るには，`np.outer(u, u)` を使う。
`u @ u.T` と書くと，`u.T` が `u` と同じ 1 次元配列なので，内積 $\boldsymbol{u}^\top\boldsymbol{u}$ が計算される。

```python
P = np.outer(u, u) / (u @ u)
print(P @ v)  # [1.5 0.5]
print(P)
```

```text
[[0.9 0.3]
 [0.3 0.1]]
```

`P @ v` は，上で求めた射影 `proj` と一致する。

### 6. 逆行列と連立一次方程式

#### 単位行列と逆行列

対角成分（$i$ 行 $i$ 列の成分）がすべて 1 で，それ以外の成分がすべて 0 の正方行列を **単位行列 (identity matrix)** といい，$\boldsymbol{I}$ と書く。
任意のベクトル $\boldsymbol{x}$ に対して $\boldsymbol{I}\boldsymbol{x} = \boldsymbol{x}$ であり，単位行列は点を動かさない変換を表す。
NumPy では，第3回で扱った `np.eye(n)` で $n\times n$ の単位行列を作る。

正方行列 $\boldsymbol{A}$ に対して

$$
\boldsymbol{A}\boldsymbol{A}^{-1} = \boldsymbol{A}^{-1}\boldsymbol{A} = \boldsymbol{I}
$$

を満たす行列 $\boldsymbol{A}^{-1}$ を，$\boldsymbol{A}$ の **逆行列 (inverse matrix)** という。
$\boldsymbol{A}$ で点 $\boldsymbol{x}$ を $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}$ に移したとき，$\boldsymbol{A}^{-1}\boldsymbol{y} = \boldsymbol{A}^{-1}\boldsymbol{A}\boldsymbol{x} = \boldsymbol{x}$ となる。
つまり，逆行列は，変換を元に戻す変換を表す。
2×2 行列の逆行列は，行列式を使って次のように書ける。

$$
\boldsymbol{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix},\qquad
\boldsymbol{A}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}\qquad(ad - bc\neq0)
$$

右辺に $\boldsymbol{A}$ を掛けると $\boldsymbol{I}$ になることは，行列積の定義から確かめられる。
NumPy では，逆行列を `np.linalg.inv(A)` で計算する。

#### 逆行列を持たない行列

行列式が 0 の正方行列は逆行列を持たない。
このような行列を **特異行列 (singular matrix)** という。
2×2 の特異行列による変換は平面全体を直線または 1 点につぶすため，異なる点が同じ点に移る。
このため，移った先の点から元の点を 1 つに決めることができない。

いくつかのベクトルのどれも残りのベクトルの線形結合で表せないとき，それらのベクトルは **線形独立 (linearly independent)** であるという。
行列の列ベクトルから選べる線形独立なベクトルの最大個数を，行列の **階数 (rank)** という。
階数は，変換後の点全体が広がる範囲の次元に等しい。
2×2 行列では，階数が 2 なら平面全体に，1 なら原点を通る直線上に，0 なら原点だけに点が移る。
NumPy では，階数を `np.linalg.matrix_rank(A)` で計算する。
$n\times n$ 行列が逆行列を持つのは，階数が $n$ のとき，すなわち行列式が 0 でないときである。
`np.linalg.inv` に特異行列を渡すと，`numpy.linalg.LinAlgError` が発生する。

#### 連立一次方程式

連立一次方程式

$$
\begin{cases} 3x_1 + x_2 = 9 \\ x_1 + 2x_2 = 8 \end{cases}
$$

は，係数を並べた行列 $\boldsymbol{A} = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$ と，$\boldsymbol{x} = (x_1, x_2)^\top$，$\boldsymbol{b} = (9, 8)^\top$ を使って，$\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ と書ける。
$\boldsymbol{A}$ が逆行列を持つなら，両辺に左から $\boldsymbol{A}^{-1}$ を掛けて，解 $\boldsymbol{x} = \boldsymbol{A}^{-1}\boldsymbol{b}$ が得られる。
NumPy では，連立一次方程式を `np.linalg.solve(A, b)` で解く。

```python
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
print(np.linalg.det(A))          # 約 5.0
print(np.linalg.matrix_rank(A))  # 2
x = np.linalg.solve(A, b)
print(x)                         # [2. 3.]
print(np.allclose(A @ x, b))     # True
print(np.linalg.inv(A) @ b)      # [2. 3.]
print(np.linalg.inv(A))
```

```text
[[ 0.4 -0.2]
 [-0.2  0.6]]
```

`np.allclose(p, q)` は，2 つの配列の対応する要素がすべてほぼ等しいときに `True` を返す。
第3回の `np.isclose` の判定を，すべての要素についてまとめて行うものである。
求めた解は，このように元の方程式に代入して確かめる。
`np.linalg.inv(A) @ b` でも同じ解が得られるが，連立一次方程式を解く目的では `np.linalg.solve` を使う。
`solve` は逆行列全体を求めずに方程式を解くため，計算量が少なく，丸め誤差の影響も受けにくい。
`solve` に特異行列を渡した場合も，`LinAlgError` が発生する。

### 7. 固有値と固有ベクトル

#### 変換で自分自身のスカラー倍になるベクトル

正方行列 $\boldsymbol{A}$ による変換で，零ベクトルでない $\boldsymbol{v}$ が自分自身のスカラー倍になる場合を考える。

$$
\boldsymbol{A}\boldsymbol{v} = \lambda\boldsymbol{v},\qquad \boldsymbol{v}\neq\boldsymbol{0}
$$

このとき，$\lambda$ を $\boldsymbol{A}$ の **固有値 (eigenvalue)**，$\boldsymbol{v}$ をその固有値に対応する **固有ベクトル (eigenvector)** という。
零ベクトルはどの $\lambda$ でも上の式を満たしてしまうため，固有ベクトルから除く。
固有値と固有ベクトルが実数の場合，$\lambda>0$ なら向きは同じで長さが $\lambda$ 倍になり，$\lambda<0$ なら向きが反転して長さが $|\lambda|$ 倍になる。
$\lambda=0$ なら，その固有ベクトルは零ベクトルに移る。
したがって，実数の固有ベクトルがつくる直線上の点は，変換後も同じ直線上にある。

例えば，次の行列とベクトルを考える。

$$
\boldsymbol{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix},\qquad
\boldsymbol{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix},\qquad
\boldsymbol{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}
$$

行列とベクトルの積を計算すると，$\boldsymbol{A}\boldsymbol{v}_1 = 3\boldsymbol{v}_1$，$\boldsymbol{A}\boldsymbol{v}_2 = \boldsymbol{v}_2$ である。
つまり，$\boldsymbol{v}_1$ の方向には 3 倍に伸ばし，$\boldsymbol{v}_2$ の方向には長さを変えない変換であり，対応する固有値はそれぞれ 3 と 1 である。
固有ベクトルの 0 でないスカラー倍も，同じ固有値に対応する固有ベクトルである。
実際，$\boldsymbol{A}(c\boldsymbol{v}) = c\boldsymbol{A}\boldsymbol{v} = \lambda(c\boldsymbol{v})$ なので，固有ベクトルの長さや符号は一意には決まらない。

#### 固有値と固有ベクトルの求め方

固有値の定義を移項すると，$(\boldsymbol{A}-\lambda\boldsymbol{I})\boldsymbol{v}=\boldsymbol{0}$ となる。
$\boldsymbol{A}-\lambda\boldsymbol{I}$ が逆行列を持つなら，両辺にその逆行列を掛けると $\boldsymbol{v}=\boldsymbol{0}$ となり，固有ベクトルの条件を満たさない。
零ベクトルでない解を持つ条件は，次の式で表せる。

$$
\det(\boldsymbol{A}-\lambda\boldsymbol{I})=0
$$

これを $\lambda$ について解き，各固有値を $(\boldsymbol{A}-\lambda\boldsymbol{I})\boldsymbol{v}=\boldsymbol{0}$ に代入して，零ベクトルでない解を求める。
上の例では，$\det(\boldsymbol{A}-\lambda\boldsymbol{I})=(2-\lambda)^2-1=(\lambda-3)(\lambda-1)$ なので，固有値は 3 と 1 である。
$\lambda=3$ を代入すると $v_2=v_1$，$\lambda=1$ を代入すると $v_2=-v_1$ となり，それぞれ $(1,1)^\top$，$(1,-1)^\top$ を固有ベクトルとして選べる。
この連立方程式の係数行列は特異なので，第6節の `np.linalg.solve(A - lam * np.eye(2), np.zeros(2))` で固有ベクトルを求めることはできない。

NumPy では，`np.linalg.eig(A)` で固有値と固有ベクトルをまとめて計算する。

```python
A = np.array([[2.0, 1.0], [1.0, 2.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues.shape, eigenvectors.shape)  # (2,) (2, 2)
print(np.sort(eigenvalues))                   # [1. 3.]
for i in range(2):
    v = eigenvectors[:, i]
    print(np.allclose(A @ v, eigenvalues[i] * v))  # どちらも True
```

$n\times n$ 行列に対して，`eigenvalues` の形は `(n,)`，`eigenvectors` の形は `(n, n)` である。
`eigenvalues[i]` に対応する固有ベクトルは，`eigenvectors` の **第 `i` 列** `eigenvectors[:, i]` に入る。
返される固有ベクトルのノルムは 1 なので，この例の手計算のベクトルとは長さが異なる。
固有値の返される順序は決まっておらず，上の `np.sort` は表示するときだけ値を昇順に並べている。
計算に使う固有値の順序を変える場合は，対応する固有ベクトルの列も同じ順序で並べ替える必要がある。
固有ベクトルの成分が手計算と完全に一致するかではなく，$\boldsymbol{A}\boldsymbol{v}=\lambda\boldsymbol{v}$ を満たすかで確かめる。

### 8. 対角化

#### 基底と座標

$n$ 個の $n$ 次元ベクトル $\boldsymbol{v}_1,\ldots,\boldsymbol{v}_n$ が線形独立なら，任意の $n$ 次元ベクトルを，それらの線形結合でただ 1 通りに表せる。
このようなベクトルの組を **基底 (basis)** といい，線形結合の係数を，その基底に関する **座標 (coordinates)** という。
各ベクトルを列に並べた行列を $\boldsymbol{P}$ とすると，次のように書ける。

$$
\boldsymbol{P} = \begin{pmatrix} \boldsymbol{v}_1 & \cdots & \boldsymbol{v}_n \end{pmatrix},\qquad
\boldsymbol{x} = c_1\boldsymbol{v}_1+\cdots+c_n\boldsymbol{v}_n = \boldsymbol{P}\boldsymbol{c},\qquad
\boldsymbol{c}=\boldsymbol{P}^{-1}\boldsymbol{x}
$$

列が線形独立なので $\boldsymbol{P}$ の階数は $n$ であり，第6節より逆行列が存在する。
$\boldsymbol{P}^{-1}$ は，元の成分表示 $\boldsymbol{x}$ を新しい基底での座標 $\boldsymbol{c}$ に直し，$\boldsymbol{P}$ はその座標を元の成分表示に戻す。

#### 固有ベクトルを基底にする

$n\times n$ 行列 $\boldsymbol{A}$ に，線形独立な固有ベクトルが $n$ 個あるとする。
それらを列に並べた行列を $\boldsymbol{P}$，対応する固有値を同じ順序で対角成分に並べた行列を $\boldsymbol{D}$ とする。
対角成分以外がすべて 0 の行列を **対角行列 (diagonal matrix)** といい，ここでは $\boldsymbol{D}=\mathrm{diag}(\lambda_1,\ldots,\lambda_n)$ と書く。
各列について $\boldsymbol{A}\boldsymbol{v}_i=\lambda_i\boldsymbol{v}_i$ なので，次の関係が成り立つ。

$$
\boldsymbol{A}\boldsymbol{P}=\boldsymbol{P}\boldsymbol{D},\qquad
\boldsymbol{P}^{-1}\boldsymbol{A}\boldsymbol{P}=\boldsymbol{D},\qquad
\boldsymbol{A}=\boldsymbol{P}\boldsymbol{D}\boldsymbol{P}^{-1}
$$

このように，基底を取り替えて行列を対角行列で表すことを **対角化 (diagonalization)** という。
$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{P}\boldsymbol{D}\boldsymbol{P}^{-1}\boldsymbol{x}$ は，右から順に次の 3 つの操作を表す。

1. $\boldsymbol{P}^{-1}$ で，固有ベクトルを基底とする座標に直す。
2. $\boldsymbol{D}$ で，第 $i$ 成分を固有値 $\lambda_i$ 倍する。
3. $\boldsymbol{P}$ で，元の成分表示に戻す。

第7節の行列では，次の $\boldsymbol{P}$ と $\boldsymbol{D}$ を使える。

$$
\boldsymbol{P}=\begin{pmatrix}1&1\\1&-1\end{pmatrix},\qquad
\boldsymbol{D}=\begin{pmatrix}3&0\\0&1\end{pmatrix},\qquad
\boldsymbol{P}^{-1}=\frac{1}{2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}
$$

```python
P = np.array([[1.0, 1.0], [1.0, -1.0]])
D = np.diag([3.0, 1.0])
print(np.allclose(A @ P, P @ D))                    # True
print(np.allclose(np.linalg.inv(P) @ A @ P, D))     # True
print(np.allclose(P @ D @ np.linalg.inv(P), A))     # True
x = np.array([2.0, 0.0])
c = np.linalg.solve(P, x)
print(c, D @ c)                                    # [1. 1.] [3. 1.]
print(P @ (D @ c), A @ x)                          # [4. 2.] [4. 2.]
```

`np.diag` に数のリストを渡すと，それらを対角成分に並べた対角行列を作る。
この例では $\boldsymbol{x}=\boldsymbol{v}_1+\boldsymbol{v}_2$ が，変換によって $3\boldsymbol{v}_1+\boldsymbol{v}_2$ に変わる。
式の確認には逆行列を使ったが，特定の $\boldsymbol{x}$ の座標 $\boldsymbol{c}$ を求める計算には，第6節と同様に `np.linalg.solve(P, x)` を使える。

#### 対角化できる条件

対角化できるのは，線形独立な固有ベクトルを $n$ 個選べる場合である。
$n$ 個の固有値がすべて異なるなら，対応する固有ベクトルは線形独立になり，対角化できる。
同じ固有値が重複する場合は，独立な固有ベクトルを必要な個数だけ選べるかを確認する。
例えば，単位行列はすべての固有値が 1 だが，すでに対角行列である。
一方，せん断行列 $\boldsymbol{H}=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ は，$\det(\boldsymbol{H}-\lambda\boldsymbol{I})=(1-\lambda)^2$ より固有値が 1 だけである。
$(\boldsymbol{H}-\boldsymbol{I})\boldsymbol{v}=\boldsymbol{0}$ を解くと $v_2=0$ なので，固有ベクトルはすべて $(1,0)^\top$ のスカラー倍であり，独立な固有ベクトルを 2 個選べず，対角化できない。
$\boldsymbol{H}$ 自体は行列式が 1 で逆行列を持つため，「逆行列を持つこと」と「対角化できること」は異なる条件である。

実数の行列でも，固有値や固有ベクトルが複素数になることがある。
その場合も $\boldsymbol{A}\boldsymbol{v}=\lambda\boldsymbol{v}$ という定義は同じであり，複素数の固有ベクトルを使って基底を作れれば，複素数の範囲で対角化できる。
例えば，平面の $90^\circ$ 回転には実数の固有ベクトルがないが，複素数まで許すと固有値は $j$ と $-j$（$j^2=-1$）となり，対角化できる。
第8回では複素正弦波を基底とする DFT を扱い，第10回では，巡回畳み込みを表す行列がその基底で対角化できることを確かめる。
そこでの「DFT で座標を直す → 周波数ごとに掛け算する → 逆 DFT で戻す」という計算は，上の 3 つの操作に対応する。

## 演習

演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第5回の第1問であれば，`exercise/exc05_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc05`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第2問の図であれば，`outputs/exc05/exc05_02.png` とします。
ベクトルや図形を描く図では，`ax.set_aspect("equal")` で縦軸と横軸の目盛りの間隔をそろえてください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえで NumPy で計算した結果と比較してください。

### 基礎レベル（10問）
1. ベクトルや行列を NumPy の配列で表すときは，形の違いに注意する必要があります。次の各式が作る配列の `shape` を予想してください。`exercise/exc05_01.py` に各式を書き，実際に表示して予想と比較してください。

    - `np.array([3, 4])`
    - `np.array([3, 4]).reshape(2, 1)`
    - `np.array([3, 4]).reshape(1, 2)`
    - `np.array([3, 4]).T`
    - `np.array([3, 4]).reshape(2, 1).T`
    - `np.array([[1, 2, 3], [4, 5, 6]]).T`
    - `np.array([[1, 2, 3], [4, 5, 6]])[:, 1]`

    さらに，次の 2 つにも取り組んでください。

    1. 4 つ目の式で，転置しても形が変わらない理由を説明してください。
    2. `M = np.array([[1, 2, 3], [4, 5, 6]])` について，`M.T[2, 0]` の値を予想してから表示してください。数式の成分の関係 $(\boldsymbol{M}^\top)_{ij} = m_{ji}$ と，NumPy の添字の対応を説明してください。

2. ベクトルの和とスカラー倍を，式と図の両方で確かめます。$\boldsymbol{u} = (1, 2)^\top$，$\boldsymbol{v} = (3, -1)^\top$ とします。

    1. $\boldsymbol{u} + \boldsymbol{v}$，$\boldsymbol{u} - \boldsymbol{v}$，$2\boldsymbol{u} - 3\boldsymbol{v}$ を手計算してください。
    2. `exercise/exc05_02.py` で，1 の 3 つのベクトルを NumPy で計算し，手計算の結果と比較してください。
    3. $\boldsymbol{u}$，$\boldsymbol{v}$，$\boldsymbol{u} + \boldsymbol{v}$ を，原点を始点とする矢印として 1 枚の図に描いてください。さらに，$\boldsymbol{u}$ の終点を始点として $\boldsymbol{v}$ と同じ成分の矢印を描き加え，`outputs/exc05/exc05_02.png` に保存してください。
    4. 図を見て，$\boldsymbol{u} + \boldsymbol{v}$ の矢印が，$\boldsymbol{u}$ と $\boldsymbol{v}$ を 2 辺とする平行四辺形の対角線になっていることを確認してください。$\boldsymbol{u}$ の終点に $\boldsymbol{v}$ をつないでも，$\boldsymbol{v}$ の終点に $\boldsymbol{u}$ をつないでも同じ点に着く理由を，成分の計算から説明してください。
    5. $\boldsymbol{w} = (5, 3)^\top$ を $\boldsymbol{u}$ と $\boldsymbol{v}$ の線形結合 $c_1\boldsymbol{u} + c_2\boldsymbol{v}$ で表すときの係数 $c_1$，$c_2$ を手計算してください。成分ごとに式を立てると，$c_1$，$c_2$ についての連立方程式になります。求めた係数で $c_1\boldsymbol{u} + c_2\boldsymbol{v}$ を計算し，$\boldsymbol{w}$ と一致することを確かめてください。

3. 内積とノルムから，2 つのベクトルのなす角を求めます。基礎第2問と同じく，$\boldsymbol{u} = (1, 2)^\top$，$\boldsymbol{v} = (3, -1)^\top$ とします。

    1. $\boldsymbol{u}^\top\boldsymbol{v}$，$\lVert\boldsymbol{u}\rVert$，$\lVert\boldsymbol{v}\rVert$，$\cos\theta$ を手計算してください。根号は残したままでかまいません。また，内積の符号から，なす角 $\theta$ が鋭角と鈍角のどちらかを答えてください。
    2. `exercise/exc05_03.py` で，`@` と `np.linalg.norm` を使って 1 の値を計算し，比較してください。さらに，`np.arccos` と `np.degrees` を使って $\theta$ を度数法で表示し，1 の答えと矛盾しないことを確かめてください。
    3. $\boldsymbol{u}$ と直交する単位ベクトルを 2 つ手計算で求めてください。求めた 2 つのベクトルについて，$\boldsymbol{u}$ との内積が 0，ノルムが 1 であることを NumPy で確かめてください。
    4. $3\boldsymbol{u}$ と $\boldsymbol{v}$ のなす角の $\cos$ の値を予想してください。実際に計算して確かめ，$\cos\theta$ がベクトルの長さによらない理由を，内積とノルムの式から説明してください。

4. 行列積の定義と，計算できる形の規則を確かめます。$\boldsymbol{A} = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$，$\boldsymbol{B} = \begin{pmatrix} 0 & 1 \\ 2 & 1 \end{pmatrix}$，$\boldsymbol{x} = (1, -1)^\top$ とします。

    1. $\boldsymbol{A}\boldsymbol{x}$，$\boldsymbol{A}\boldsymbol{B}$，$\boldsymbol{B}\boldsymbol{A}$ を手計算してください。
    2. `exercise/exc05_04.py` で `A @ x`，`A @ B`，`B @ A`，`A * B` を計算し，1 の結果と比較してください。`A @ B` と `B @ A` が異なること，`A * B` が行列積と異なることを確認し，`A * B` が何を計算しているかを説明してください。
    3. `A @ x` が，`A` の各行と `x` の内積を並べたベクトル，および `A` の列の線形結合 `x[0] * A[:, 0] + x[1] * A[:, 1]` の両方と一致することを確かめてください。
    4. 次の形の配列どうしの `@` の結果の形を予想してください。計算できない場合は，その理由を答えてください。`np.ones` で各形の配列を作って確かめてください。計算できない組み合わせは，他の組み合わせとは分けて実行してください。

        | 左の配列の形 | 右の配列の形 |
        | --- | --- |
        | `(3, 2)` | `(2, 4)` |
        | `(4,)` | `(4, 2)` |
        | `(2, 3)` | `(3,)` |
        | `(5,)` | `(5,)` |
        | `(3, 2)` | `(3, 2)` |

    5. $(\boldsymbol{A}\boldsymbol{B})^\top$ が $\boldsymbol{B}^\top\boldsymbol{A}^\top$ と一致し，$\boldsymbol{A}^\top\boldsymbol{B}^\top$ とは一致しないことを確かめてください。`np.array_equal(p, q)` は，2 つの配列の形とすべての要素が等しいときに `True` を返します。

5. 行列による平面の変換を図で確かめます。回転 $\boldsymbol{R} = \boldsymbol{R}(30^\circ)$，拡大縮小 $\boldsymbol{S} = \begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix}$，せん断 $\boldsymbol{H} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ の 3 つの変換を考えます。

    1. 各行列による $\boldsymbol{e}_1 = (1, 0)^\top$ と $\boldsymbol{e}_2 = (0, 1)^\top$ の移り先を手計算してください。$\cos 30^\circ = \sqrt{3}/2$，$\sin 30^\circ = 1/2$ を使ってください。
    2. 3 つの行列の行列式を手計算し，それぞれの変換で面積が何倍になるかを答えてください。また，3 つの変換のうち，長さと角度をどちらも保つ変換はどれかを予想してください。
    3. `exercise/exc05_05.py` で，3 つの行列を作ってください。角度はラジアンで `np.pi / 6` と書けます。さらに，解説の単位正方形 `square` と，単位円周上の 100 点を作ってください。単位円周上の点は，`t = np.linspace(0, 2 * np.pi, 100)` に対して `np.column_stack([np.cos(t), np.sin(t)])` とすると，形 `(100, 2)` の配列として作れます。
    4. `plt.subplots(1, 3, figsize=(12, 4))` を使い，変換ごとに 1 つの subplot に，変換前の正方形と円，変換後の正方形と円を重ねて描いてください。図は `outputs/exc05/exc05_05.png` に保存してください。
    5. `np.linalg.det` で 3 つの行列の行列式を計算し，2 の手計算の結果と比較してください。
    6. 図を見て，2 の予想が正しかったかを確かめてください。また，面積の倍率だけでは長さや角度が保たれるかを判断できない理由を，1 で求めた $\boldsymbol{e}_1$，$\boldsymbol{e}_2$ の移り先の長さと，2 つの移り先のなす角を使って説明してください。

6. 2 つの変換を続けて施すとき，順序によって結果が変わることを確かめます。$\boldsymbol{R} = \boldsymbol{R}(90^\circ)$，$\boldsymbol{S} = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$ とします。

    1. $\boldsymbol{R}$ の各成分を手計算で求め，$\boldsymbol{R}\boldsymbol{S}$ と $\boldsymbol{S}\boldsymbol{R}$ を手計算してください。
    2. 「先に $\boldsymbol{S}$ で横に 2 倍してから，$\boldsymbol{R}$ で回転する」合成変換は，$\boldsymbol{R}\boldsymbol{S}$ と $\boldsymbol{S}\boldsymbol{R}$ のどちらで表されるかを答えてください。
    3. `exercise/exc05_06.py` で，4 点 $(1, 0)$，$(1, 2)$，$(2, 1.5)$，$(1, 1)$ を行として並べた配列 `flag` を作ってください。この 4 点を順に結ぶと，旗の形になります。
    4. `flag` を $\boldsymbol{R}\boldsymbol{S}$ と $\boldsymbol{S}\boldsymbol{R}$ のそれぞれで変換し，元の旗と重ねて 1 枚の図に描いて，`outputs/exc05/exc05_06.png` に保存してください。描く前に，2 つの変換後の旗で，旗竿（最初の 2 点を結ぶ線分）の長さと旗の部分の大きさがそれぞれどうなるかを予想してください。
    5. 図で見られた違いを，「どちらの変換を先に施したか」を使って説明してください。また，`flag @ S.T @ R.T` と `flag @ (R @ S).T` が一致することを `np.allclose` で確かめ，その理由を説明してください。

7. ベクトルを，ある向きの成分とそれに直交する成分に分解します。$\boldsymbol{u} = (2, 1)^\top$，$\boldsymbol{v} = (1, 3)^\top$ とします。

    1. $\boldsymbol{v}$ の $\boldsymbol{u}$ への射影 $\mathrm{proj}_{\boldsymbol{u}}\boldsymbol{v}$ と，残差 $\boldsymbol{r} = \boldsymbol{v} - \mathrm{proj}_{\boldsymbol{u}}\boldsymbol{v}$ を手計算してください。また，$\boldsymbol{u}^\top\boldsymbol{r} = 0$ を確かめ，点 $(1, 3)$ と $\boldsymbol{u}$ の向きの直線との距離を求めてください。
    2. `exercise/exc05_07.py` で 1 の値を NumPy で計算し，比較してください。
    3. 射影行列 $\boldsymbol{P}$ を手計算してください。`np.outer` を使って `P` を計算し，`P @ v` が 1 の射影と一致すること，`P @ P` が `P` と一致することを `np.allclose` で確かめてください。
    4. $\boldsymbol{P}\boldsymbol{P} = \boldsymbol{P}$ となる理由を，「直線上にある点を射影すると何が起きるか」という観点から説明してください。
    5. `np.outer(u, u)` の代わりに `u @ u.T` と書いたときの結果を予想し，実際に表示して確かめてください。
    6. $\boldsymbol{u}$ の向きの直線，$\boldsymbol{v}$，射影，残差を 1 枚の図に描き，`outputs/exc05/exc05_07.png` に保存してください。直線は，`t = np.linspace(-1, 2, 2)` の各値 $t$ に対する点 $t\,\boldsymbol{u}$ を結んで描けます。残差は，射影の終点を始点とする矢印として描いてください。図で，残差の矢印が直線と直交し，$\boldsymbol{v}$ の終点に届いていることを確認してください。

8. 逆行列を使って連立一次方程式を解き，逆行列を持たない行列を調べます。

    1. 連立一次方程式 $2x_1 + x_2 = 5$，$x_1 + 3x_2 = 10$ を $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ の形で書いてください。$\boldsymbol{A}$ の行列式と逆行列を手計算し，その逆行列を使って解 $\boldsymbol{x}$ を手計算してください。
    2. `exercise/exc05_08.py` で，`np.linalg.inv` と `np.linalg.solve` のそれぞれで解を求め，1 の結果と比較してください。また，`A @ np.linalg.inv(A)` が単位行列とほぼ等しいことを，`np.allclose` と `np.eye` で確かめてください。
    3. $\boldsymbol{S} = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$ の行列式と階数を手計算で求め，`np.linalg.det` と `np.linalg.matrix_rank` で確かめてください。
    4. `np.linalg.inv(S)` を実行すると何が起きるかを予想してください。他の処理とは分けて実行し，結果を確かめてください。
    5. 基礎第5問と同じ単位円周上の 100 点を $\boldsymbol{S}$ で変換し，変換前の円と変換後の点を 1 枚の図に描いて，`outputs/exc05/exc05_08.png` に保存してください。変換後の点がどのような図形の上に並ぶかを確かめ，その図形を式で表してください。
    6. $\boldsymbol{S}\boldsymbol{x} = (1, 2)^\top$ を満たす $\boldsymbol{x}$ を 3 つ見つけ，NumPy で確かめてください。また，$\boldsymbol{S}\boldsymbol{x} = (1, 0)^\top$ を満たす $\boldsymbol{x}$ が存在するかを，5 の図から答えてください。これらの結果を使って，$\boldsymbol{S}$ が逆行列を持たない理由を説明してください。

9. 固有値と固有ベクトルを求め，変換の意味を確かめます。$\boldsymbol{B}=\begin{pmatrix}1&2\\2&1\end{pmatrix}$ とします。

    1. $\det(\boldsymbol{B}-\lambda\boldsymbol{I})=0$ を手計算で解き，2 つの固有値を求めてください。それぞれについて，$(\boldsymbol{B}-\lambda\boldsymbol{I})\boldsymbol{v}=\boldsymbol{0}$ を満たす零ベクトルでない $\boldsymbol{v}$ を 1 つずつ求めてください。
    2. `exercise/exc05_09.py` で，`np.linalg.eig` による固有値と固有ベクトルを表示し，1 と比較してください。各列について `B @ v` と `lam * v` がほぼ等しいことを確かめてください。
    3. 手計算で求めた各固有ベクトル $\boldsymbol{v}$ と，変換後の $\boldsymbol{B}\boldsymbol{v}$ を，原点を始点とする矢印として描いてください。固有値ごとに `plt.subplots(1, 2)` の別の subplot を使い，変換前後の矢印を色で区別して凡例を付け，`outputs/exc05/exc05_09.png` に保存してください。描く前に，矢印の向きと長さがどう変わるかを予想し，図と比較してください。
    4. $-2\boldsymbol{v}$ も同じ固有値に対応する固有ベクトルであることを式と NumPy で確かめてください。また，零ベクトルを固有ベクトルとしない理由を説明してください。

10. 固有ベクトルを基底として，第9問の $\boldsymbol{B}$ を対角化します。固有値の大きい方に対応するベクトルを $\boldsymbol{v}_1=(1,1)^\top$，小さい方に対応するベクトルを $\boldsymbol{v}_2=(1,-1)^\top$ とします。

    1. $\boldsymbol{v}_1$，$\boldsymbol{v}_2$ を列に並べた $\boldsymbol{P}$ と，対応する固有値を同じ順序で並べた対角行列 $\boldsymbol{D}$ を書いてください。$\boldsymbol{P}^{-1}$ を手計算し，$\boldsymbol{P}^{-1}\boldsymbol{B}\boldsymbol{P}=\boldsymbol{D}$ を確かめてください。
    2. $\boldsymbol{x}=(3,1)^\top$ について，$\boldsymbol{c}=\boldsymbol{P}^{-1}\boldsymbol{x}$，$\boldsymbol{D}\boldsymbol{c}$，$\boldsymbol{P}\boldsymbol{D}\boldsymbol{c}$ を順に手計算してください。最後の結果が $\boldsymbol{B}\boldsymbol{x}$ と一致することを確かめ，各段階の意味を説明してください。
    3. `exercise/exc05_10.py` で，`np.diag` を使って $\boldsymbol{D}$ を作り，1 と 2 の結果を確かめてください。$\boldsymbol{c}$ は `np.linalg.solve` で求め，`B @ P` と `P @ D`，`B` と `P @ D @ np.linalg.inv(P)` がそれぞれほぼ等しいことも確かめてください。
    4. $\boldsymbol{P}$ の 2 つの列を交換した場合に，$\boldsymbol{D}$ をどう変えれば同じ $\boldsymbol{B}$ を表せるかを予想し，NumPy で確かめてください。また，$\boldsymbol{P}$ を転置するだけでは逆行列にならないことを計算で確認し，第8節の $\boldsymbol{P}^{-1}$ の式を使って理由を説明してください。

### 発展レベル（5問）
1. 係数のわずかな違いで解が大きく変わる連立一次方程式を調べます。$\boldsymbol{A} = \begin{pmatrix} 1 & 1 \\ 1 & 1.001 \end{pmatrix}$ とします。

    1. $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ の解を，$\boldsymbol{b} = (2, 2.001)^\top$ と $\boldsymbol{b}' = (2, 2.002)^\top$ のそれぞれについて手計算してください。右辺の変化量と解の変化量を比べてください。
    2. `exercise/exc05_11.py` で，2 つの右辺について `np.linalg.solve` で解を求め，1 の結果と比較してください。
    3. $\boldsymbol{A}$ の行列式と，$\boldsymbol{A}$ の 2 つの列ベクトルのなす角（度数法）を計算して表示してください。2 つの列がほぼ平行であることと，右辺のわずかな変化で解が大きく変わることの関係を，第4節の「各列は $\boldsymbol{e}_1$，$\boldsymbol{e}_2$ の移り先を表す」という見方を使って説明してください。
    4. 右辺の変化が解の変化として何倍に拡大されるかの目安を，**条件数 (condition number)** といいます。条件数 $\kappa(\boldsymbol{A})$ を使うと，右辺を $\boldsymbol{b}$ から $\boldsymbol{b} + \Delta\boldsymbol{b}$ に変えたときの解の変化 $\Delta\boldsymbol{x}$ について，次の不等式が成り立ちます。

        $$
        \frac{\lVert\Delta\boldsymbol{x}\rVert}{\lVert\boldsymbol{x}\rVert} \le \kappa(\boldsymbol{A})\,\frac{\lVert\Delta\boldsymbol{b}\rVert}{\lVert\boldsymbol{b}\rVert}
        $$

        `np.linalg.cond` で，この $\boldsymbol{A}$ と基礎第8問の $\boldsymbol{A}$ の条件数を計算してください。また，1 の $\boldsymbol{b}$ と $\boldsymbol{b}'$ について，拡大率 $\dfrac{\lVert\Delta\boldsymbol{x}\rVert/\lVert\boldsymbol{x}\rVert}{\lVert\Delta\boldsymbol{b}\rVert/\lVert\boldsymbol{b}\rVert}$ を計算し，条件数を超えないことを確かめてください。$\boldsymbol{x}$ は $\boldsymbol{b}$ に対する解とします。
    5. 測定値から右辺 $\boldsymbol{b}$ を作り，$\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ を解いて $\boldsymbol{x}$ を求める場面を考えます。条件数の大きい $\boldsymbol{A}$ を使うと，どのような困難が生じるかを考察してください。

2. 回転行列 $\boldsymbol{R}(\theta)$ の性質を，式と数値計算の両方で確かめます。三角関数の加法定理 $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$，$\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$ を使ってかまいません。

    1. $\boldsymbol{R}(\alpha)\boldsymbol{R}(\beta) = \boldsymbol{R}(\alpha+\beta)$ を手計算で示し，この式が表す変換の意味を説明してください。
    2. $\boldsymbol{R}(\theta)^\top\boldsymbol{R}(\theta) = \boldsymbol{I}$ と $\det\boldsymbol{R}(\theta) = 1$ を手計算で示してください。この結果を使って，$\boldsymbol{R}(\theta)$ の逆行列を $\boldsymbol{R}(-\theta)$ と $\boldsymbol{R}(\theta)^\top$ の 2 通りで表し，逆行列が表す変換の意味を説明してください。
    3. `exercise/exc05_12.py` で，角度 $\theta$（ラジアン）を受け取って $\boldsymbol{R}(\theta)$ を返す関数 `rotation(theta)` を定義してください。`np.linspace(0, 2 * np.pi, 13)` の各 $\theta$ と $\beta = \pi/5$ について，1 と 2 の関係がすべて成り立つことを `np.allclose` と `np.isclose` で確かめてください。
    4. $\boldsymbol{a} = (3, 1)^\top$，$\boldsymbol{b} = (-1, 2)^\top$ を $\boldsymbol{R}(\pi/7)$ で回転し，回転の前後でノルムと内積が変わらないことを確かめてください。$\boldsymbol{R}^\top\boldsymbol{R} = \boldsymbol{I}$ と $(\boldsymbol{R}\boldsymbol{a})^\top = \boldsymbol{a}^\top\boldsymbol{R}^\top$ を使って，その理由を式で説明してください。

3. 第5節の射影を，2 つのベクトルの線形結合全体がつくる平面への射影に広げます。この考え方は，第16回の最小二乗法で使います。3 次元のベクトル $\boldsymbol{a}_1 = (1, 0, 1)^\top$，$\boldsymbol{a}_2 = (0, 1, 1)^\top$，$\boldsymbol{y} = (2, 3, 8)^\top$ とし，$\boldsymbol{a}_1$，$\boldsymbol{a}_2$ を列として並べた $3\times2$ 行列を $\boldsymbol{A}$ とします。$\boldsymbol{a}_1$ と $\boldsymbol{a}_2$ の線形結合 $c_1\boldsymbol{a}_1 + c_2\boldsymbol{a}_2$ は，$\boldsymbol{c} = (c_1, c_2)^\top$ を使って $\boldsymbol{A}\boldsymbol{c}$ と書けます。

    1. 残差 $\boldsymbol{r} = \boldsymbol{y} - \boldsymbol{A}\boldsymbol{c}$ が $\boldsymbol{a}_1$ と $\boldsymbol{a}_2$ の両方と直交するという条件が，$\boldsymbol{A}^\top\boldsymbol{r} = \boldsymbol{0}$ と書けることを説明してください。この条件を変形して，$\boldsymbol{A}^\top\boldsymbol{A}\boldsymbol{c} = \boldsymbol{A}^\top\boldsymbol{y}$ を導いてください。
    2. $\boldsymbol{A}^\top\boldsymbol{A}$ と $\boldsymbol{A}^\top\boldsymbol{y}$ を手計算し，$\boldsymbol{c}$，射影 $\boldsymbol{A}\boldsymbol{c}$，残差 $\boldsymbol{r}$ を求めてください。
    3. `exercise/exc05_13.py` で，$\boldsymbol{A}$ を作り，`np.linalg.solve` で $\boldsymbol{c}$ を求めてください。`np.column_stack([a1, a2])` は，1 次元配列 `a1`，`a2` を列として並べた 2 次元配列を返します。射影と残差を表示し，`A.T @ r` が零ベクトルとほぼ等しいことを確かめてください。
    4. $c_1$，$c_2$ を，3 で求めた値を中心に $\pm0.5$ の範囲で 0.1 刻みに動かした $11\times11$ 通りの組 $\boldsymbol{c}'$ について，距離 $\lVert\boldsymbol{y} - \boldsymbol{A}\boldsymbol{c}'\rVert$ を計算してください。距離が最小となる組が，3 の $\boldsymbol{c}$ と一致することを確かめてください。
    5. 任意の $\boldsymbol{c}'$ について $\boldsymbol{y} - \boldsymbol{A}\boldsymbol{c}' = \boldsymbol{r} + \boldsymbol{A}(\boldsymbol{c} - \boldsymbol{c}')$ と書けることを使い，射影が平面上で $\boldsymbol{y}$ に最も近い点である理由を，残差の直交性と三平方の定理から説明してください。

4. 連立一次方程式を解くときに `np.linalg.solve` を使う理由を，精度と計算時間の両面から確かめます。

    1. $n\times n$ の **ヒルベルト行列 (Hilbert matrix)** $\boldsymbol{H}$ は，$i$ 行 $j$ 列の成分が $1/(i+j-1)$（$i, j = 1, \ldots, n$）の行列です。`exercise/exc05_14.py` で，`i = np.arange(1, n + 1)` と第3回のブロードキャストを使い，$n = 10$ のヒルベルト行列を `for` 文を使わずに作ってください。また，`b = H @ np.ones(n)` とします。このとき，$\boldsymbol{H}\boldsymbol{x} = \boldsymbol{b}$ の真の解は，すべての成分が 1 のベクトルです。
    2. `np.linalg.solve(H, b)` と `np.linalg.inv(H) @ b` の 2 通りで解を求め，それぞれについて，残差のノルム `np.linalg.norm(H @ x - b)` と，真の解との差のノルムを表示してください。どちらの方法の残差が小さくなるかを，実行前に予想してください。
    3. $n = 6, 8, 10, 12$ のそれぞれについて，`np.linalg.cond(H)` と 2 の 4 つの値を 1 行にまとめて表示してください。$n$ が大きくなると，2 つの方法の残差と真の解との差がそれぞれどう変わるかを説明してください。また，残差が小さくても真の解との差が大きくなる場合がある理由を，発展第1問の条件数の不等式を使って考察してください。
    4. 計算時間を比べます。`rng = np.random.default_rng(0)`，`A = rng.normal(size=(1000, 1000))`，`b = rng.normal(size=1000)` として，成分が乱数の行列とベクトルを作ってください（乱数の生成は第6回で扱います）。`import time` を書き，`time.perf_counter()` を使って，2 つの方法の計算時間をそれぞれ 5 回ずつ測り，平均を表示してください。
    5. 2〜4 の結果をもとに，連立一次方程式を解くときに `np.linalg.solve` を優先すべき理由を説明してください。

5. $\lvert\det\boldsymbol{A}\rvert$ が面積の倍率であることを，円を変換して確かめます。$\boldsymbol{A} = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$，$\boldsymbol{B} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ とします。

    1. 単位円（面積 $\pi$）を $\boldsymbol{A}$ で変換した図形の面積を，行列式を使って手計算してください。
    2. `exercise/exc05_15.py` で，`t = np.linspace(0, 2 * np.pi, 1000)` に対する単位円周上の 1000 点を $\boldsymbol{A}$ で変換し，変換前後の図形を 1 枚の図に描いて，`outputs/exc05/exc05_15.png` に保存してください。
    3. 頂点 $(x_0, y_0), \ldots, (x_{N-1}, y_{N-1})$ を順に結んだ多角形の面積は，次の **靴ひも公式 (shoelace formula)** で計算できます。ここで，$(x_N, y_N) = (x_0, y_0)$ とします。

        $$
        S = \frac{1}{2}\sum_{k=0}^{N-1}(x_ky_{k+1} - x_{k+1}y_k)
        $$

        $S$ は，頂点を反時計回りにたどると正，時計回りにたどると負になる符号付きの面積です。`np.roll(x, -1)` が `x` の要素を 1 つずつ前にずらした配列 `[x[1], x[2], ..., x[N-1], x[0]]` を返すことを使って，形 `(N, 2)` の頂点の配列から $S$ を返す関数を定義してください。
    4. 2 の変換前の 1000 点が作る多角形と，変換後の 1000 点が作る多角形の面積を計算し，その比を表示してください。比が $\lvert\det\boldsymbol{A}\rvert$ とほぼ一致することを確かめてください。また，変換前の多角形の面積が $\pi$ よりわずかに小さくなる理由を説明してください。
    5. $\boldsymbol{B}$ についても同じ計算を行い，符号付きの面積の比が $\det\boldsymbol{B}$ とほぼ一致することを確かめてください。$\det\boldsymbol{B}<0$ のとき，変換後の多角形の頂点をたどる向きがどう変わるかを説明してください。

## 詰まったときに見る資料
- [`../textbook/markdown/ch06a-basics-of-linear-algebra.md`](../textbook/markdown/ch06a-basics-of-linear-algebra.md)
- [NumPy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [NumPy `linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)
- [Matplotlib `Axes.quiver`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html)
