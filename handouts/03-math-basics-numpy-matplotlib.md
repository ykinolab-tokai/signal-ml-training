# 第03回 数学基礎と NumPy, Matplotlib

## この回の目標

NumPy 配列の作成・選択・要素ごとの計算・ブロードキャストを使い，初等関数とその平行移動を計算して Matplotlib で可視化できる。

## 解説

**ライブラリ** は，再利用できる関数やクラスなどをまとめたものである。
Python では，ライブラリをモジュールやパッケージとして読み込んで利用する。
`import ... as ...` と書くと，読み込んだモジュール（パッケージ）に短い別名を付けられる。
**NumPy** は多数の数値を配列にまとめて計算するためのライブラリであり，
**Matplotlib** は数値をグラフとして可視化するためのライブラリである。

NumPy と Matplotlib をインポートするには，スクリプトの先頭に以下の通り記述する。

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

`numpy` は `np`，描画用モジュールの `matplotlib.pyplot` は `plt` という別名で使うのが慣例となっており，
それに従ってインポートしている。
`from pathlib import Path` は，Python の標準ライブラリ `pathlib` から，
ファイルやフォルダのパスを扱う `Path` をインポートするためのものである。

また， `matplotlib` で作成した図を保存するために，次のように保存先ディレクトリを用意する。

```python
figure_dir = Path("outputs/exc03")
figure_dir.mkdir(parents=True, exist_ok=True)
```

`figure_dir` は作業リポジトリ内の保存先を表すパスであり，
`mkdir` はそのパスにディレクトリを作成するメソッドである．

### 1. 多次元配列の作成と形

**多次元配列 (multidimensional array)** は，数を並べたものである。
特に，数を一列に並べたもの（ベクトル）を 1 次元配列，数を行と列に並べたもの（行列）を 2 次元配列という。
NumPy では，多次元配列を `ndarray` という型で表す。
配列の **軸 (axis)** には，先頭から 0, 1, 2, … と番号を付ける。
`ndarray` の属性には，
軸ごとの要素数を並べたタプルである `shape` ，
軸の数を表す `ndim` ，配列全体の要素数を表す `size` などがある。
`ndim` は `len(shape)` に等しく，`size` は `shape` に並んだ数の積に等しい。

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape, a.ndim, a.size)  # (3,) 1 3
print(b.shape, b.ndim, b.size)  # (2, 3) 2 6
```

形が `(2, 3)` の配列は，軸 0 の長さが 2（行数），軸 1 の長さが 3（列数）である。
形が `(3,)` の配列は 1 次元であり，2 次元の `(1, 3)` や `(3, 1)` とは区別する。

#### 値や範囲を指定した作成

様々な配列を作成するための関数が用意されている。

| 書き方 | 作られる配列 |
| --- | --- |
| `np.zeros((2, 3))` | 全要素が 0 の 2 行 3 列の配列 |
| `np.ones((2, 3))` | 全要素が 1 の 2 行 3 列の配列 |
| `np.full((2, 3), 9)` | 全要素が 9 の 2 行 3 列の配列 |
| `np.eye(3)` | 対角要素が 1，それ以外が 0 の 3 行 3 列の単位行列 |
| `np.arange(3, 10, 2)` | 開始値 3 から刻み幅 2 で並べた `[3, 5, 7, 9]`（終了値 10 を含まない） |
| `np.linspace(-5, 5, 101)` | −5 から 5 までを，両端を含めて等間隔に並べた 101 点 |

`arange` と `linspace` は共に等差数列を作成するが，指定するパラメータ（刻み幅か点数か）や
既定で端点を含むか否か（`arrange` は含まないが `linspace` は含む）などに違いがある。

配列の形を変えるためのメソッドとして `reshape` がある。
以下の例では，0 から 11 までを並べた 1 次元配列を，3 行 4 列の配列に変換する。

```python
a = np.arange(12).reshape(3, 4)
print(a)
```

```text
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
```

### 2. 配列の要素の選択と代入

#### 整数とスライスによる選択

配列の要素の指定には，リストと同じく 0 から始まるインデックスを用いる。
2 次元配列では `a[行の添字, 列の添字]` と書き，
添字に `:` だけを書くとその軸の全要素を選択できる。

```python
a = np.arange(12).reshape(3, 4)
print(a[0, 1])       # 1：1 行 2 列目
print(a[-1, -1])     # 11：最後の行の最後の列
print(a[0, :])       # [0 1 2 3]：1 行目
print(a[:, 1])       # [1 5 9]：2 列目
print(a[1:3, 1:4])   # [[ 5  6  7]：スライス
                     #  [ 9 10 11]]
print(a[0, :].shape, a[0:1, :].shape)  # (4,) (1, 4)

a[1:3, 1:4] = 0     # 指定した範囲を 0 にする
print(a)
```

整数で指定した軸は結果から除かれ，スライスで指定した軸は結果に残る。
このため，`a[0, :]` の形は `(4,)`，`a[0:1, :]` の形は `(1, 4)` になる。
`a[1:3, 1:4] = 0` を実行すると，2〜3 行目の 2〜4 列目がすべて 0 になる。

基本的なスライスで得た配列は元の配列とデータを共有する **ビュー (view)** である．
元の配列と独立したデータを持つ配列を作るには， `a[1:3, 1:4].copy()` のように明示的にコピーする必要がある。

#### 整数配列による選択

行の添字と列の添字をそれぞれ配列で渡すと，同じ位置にある添字を組にして要素を選択できる。
例えば，`a[0, 1]`，`a[2, 1]`，`a[1, 0]` を順に取り出す操作は，以下のように記述できる。

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a[[0, 2, 1], [1, 1, 0]])  # [2 8 4]
```


### 3. 配列のデータ型

1 つの数値配列のすべての要素は，共通の **データ型 (dtype)** を持つ。
整数には `int32` や `int64`，浮動小数点数には `float32` や `float64` などを使う。
型名の末尾の数は，1 要素を表すビット数である。
`dtype` を省略すると，型は入力から推測される。
型を固定したい場合は，`dtype` を明示する。

```python
x = np.array([1, 2, 3], dtype=np.int64)
y = np.array([1, 2, 3], dtype=np.float32)
z = y.astype(np.float64)
print(x.dtype, y.dtype, z.dtype)  # int64 float32 float64
```

`dtype=np.float32` は，`dtype="float32"` や `dtype="f"` とも書ける。
`astype` は，指定した型に変換した配列を返す。
変換結果を使うには，返された配列を変数に代入する。
整数配列に小数を代入すると，小数部分が失われる。
このため，小数を保存する配列は初めから浮動小数点数型で作る。
ただし，浮動小数点数にも表現精度の限界があり，計算結果には丸め誤差が生じる。

### 4. 配列を使った要素ごとの計算

同じ形の配列同士に `+`，`-`，`*`，`/` を使うと，対応する位置の要素同士で加減乗除を計算する。
特に，`*` は要素ごとの積であり，行列積とは異なる。
1 つの数値だけを表す値を **スカラー** という。
スカラーと配列の演算では，そのスカラーを配列の各要素との演算に使う。

```python
a = np.array([1, 2, 4])
b = np.array([2, 4, 8])
print(a + b)           # [ 3  6 12]
print(a - b)           # [-1 -2 -4]
print(a * b)           # [ 2  8 32]
print(a / b)           # [0.5 0.5 0.5]
print(a * 2)           # [2 4 8]
print(a**2)            # [ 1  4 16]
print(np.power(a, 2))  # [ 1  4 16]
print(np.sqrt(a))      # [1.         1.41421356 2.        ]

print([1, 2] + [3, 4])                  # [1, 2, 3, 4]：リストの連結
print(np.array([1, 2]) + np.array([3, 4]))  # [4 6]：要素ごとの和
```

数学の $x^2$ は，Python では `x**2` と書く。
Python の `^` はべき乗を表さない。
`np.sqrt` のような NumPy の関数も配列の各要素に作用する。
このため，Python の `for` 文を書かずに，多数の関数値を計算できる。

### 5. 直線のグラフと Matplotlib

**関数** $y=f(x)$ は，入力 $x$ に対して出力 $y$ を一意に定める対応である。
直線を表す関数

$$
y = ax + b
$$

では，$a$ を **傾き**，$b$ を **切片** という。
$x$ が 1 増えると $y$ は $a$ だけ増え，$x=0$ のとき $y=b$ となる。

グラフを描くには，まず入力の配列 `x` を作り，各入力に対する出力の配列 `y` を計算する。
`scatter(x, y)` は各点を描き，`plot(x, y)` は点を与えた順に線で結ぶ。

```python
x = np.arange(-5, 6)
y = 2 * x + 1
print(x.shape, y.shape)  # (11,) (11,)

fig, ax = plt.subplots()
ax.scatter(x, y, label="sample points")
ax.plot(x, y, label="y = 2x + 1")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session03_example_line.png")
plt.show()
plt.close(fig)
```

`plt.subplots()` が返す `fig` は図全体を，`ax` はグラフを描く領域を表す。
`ax.plot` や `ax.scatter` は描画先の `ax` を明示する書き方であり，それぞれ `plt.plot` と `plt.scatter` に対応する。
`label` で付けた名前は，`legend()` を呼ぶと凡例に表示される。
図の保存には `fig.savefig`（または現在の図に対する `plt.savefig`）を，画面表示には `plt.show` を使う。
表示機能のない環境では `show` を省略し，保存した PNG を開いて確認する。
保存は `show` より前に行い，描画が終わったら `close` で図を閉じる。

### 6. 2 次関数とグラフの平行移動

**2 次関数** $y=ax^2+bx+c$（$a\neq0$）のグラフは放物線になる。
平方完成すると，頂点の位置が分かる。

$$
ax^2+bx+c
=a\left(x+\frac{b}{2a}\right)^2+c-\frac{b^2}{4a}
$$

例えば，$f(x)=\frac{1}{2}x^2-x+1=\frac{1}{2}(x-1)^2+\frac{1}{2}$ の頂点は $(1,\frac{1}{2})$ である。
`plot` は離散的な点を線で結ぶ。
このため，曲線を滑らかに見せるには，入力の点の間隔を細かくする。

#### 平行移動の式

一般の関数 $f$ に対して，関数

$$
g(x)=f(x-\alpha)+\beta
$$

のグラフは，$f$ のグラフを横に $\alpha$，縦に $\beta$ だけ平行移動したものである。
実際，$g(u+\alpha)=f(u)+\beta$ なので，$f$ のグラフ上の点 $(u,f(u))$ は $g$ のグラフ上の点 $(u+\alpha,f(u)+\beta)$ に対応する。
$\alpha>0$ なら右，$\alpha<0$ なら左，$\beta>0$ なら上，$\beta<0$ なら下へ動く。
横方向に移動するには，式に現れるすべての $x$ を $x-\alpha$ に置き換える。

```python
def f(x):
    return 0.5 * x**2 - x + 1

x = np.linspace(-5, 5, 501)
fig, ax = plt.subplots()
ax.plot(x, f(x), label="f(x)")
ax.plot(x, f(x - 2), label="f(x - 2)")
ax.plot(x, f(x) + 1, label="f(x) + 1")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session03_example_shift.png")
plt.show()
plt.close(fig)
```

`f(x)`，`f(x - 2)`，`f(x) + 1` の頂点は，それぞれ $(1,0.5)$，$(3,0.5)$，$(1,1.5)$ である。
この `f` は数値だけでなく NumPy 配列も受け取り，配列を受け取った場合は同じ形の関数値の配列を返す。

### 7. 三角関数

#### 単位円とラジアン

原点を中心とする半径 1 の円を **単位円** という。
単位円上で，点 $(1,0)$ から反時計回りに角度 $\theta$ だけ回転した点を $P=(u,v)$ とする。
**三角関数** は，この点の座標を使って次のように定義される。

$$
\cos\theta=u,\qquad
\sin\theta=v,\qquad
\tan\theta=\frac{v}{u}=\frac{\sin\theta}{\cos\theta}\quad(u\neq0)
$$

角度の表し方には，1 周を $360^\circ$ とする **度数法** と，1 周を $2\pi$ rad とする **弧度法** がある。
rad はラジアンと読む。
度数法の角度 $\phi$ は，次式で弧度法の角度 $\theta$ に変換できる。

$$
\theta = \frac{\pi}{180}\phi
$$

| 度数法 | $0^\circ$ | $30^\circ$ | $45^\circ$ | $60^\circ$ | $90^\circ$ | $180^\circ$ | $360^\circ$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 弧度法 [rad] | $0$ | $\pi/6$ | $\pi/4$ | $\pi/3$ | $\pi/2$ | $\pi$ | $2\pi$ |

NumPy の `np.sin`，`np.cos`，`np.tan` には，角度をラジアンで渡す。
例えば，$90^\circ$ の正弦は `np.sin(np.pi / 2)` で計算する。

#### 周期と未定義点

$\sin$ と $\cos$ の値は $[-1,1]$ に収まる。
また，$\sin$ と $\cos$ は，$2\pi$ ごとに同じ値を繰り返す **周期関数** である。
$\sin\theta=\cos(\theta-\pi/2)$ が成り立つので，$\sin$ と $\cos$ のグラフは横方向の平行移動で互いに重なる。
$\tan$ の周期は $\pi$ である。
次の点では分母の $\cos\theta$ が 0 になるため，$\tan\theta$ は定義されない。

$$
\theta=\frac{\pi}{2}+k\pi,\qquad k\in\mathbb{Z}
$$

```python
theta = np.linspace(0, 4 * np.pi, 1001)
tan_values = np.tan(theta)
tan_values[np.abs(tan_values) > 10] = np.nan

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(theta, np.sin(theta), label="sin")
axes[0].plot(theta, np.cos(theta), label="cos")
axes[1].plot(theta, tan_values, label="tan")
axes[1].set_ylim(-10, 10)
for ax in axes:
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True)
axes[1].set_xlabel("theta [rad]")
fig.tight_layout()
fig.savefig(figure_dir / "session03_example_trigonometry.png")
plt.show()
plt.close(fig)
```

`plt.subplots(2, 1)` は，図を縦 2 段のグラフ領域に分ける。
`np.abs` は各要素の絶対値を計算する。
`tan_values > 10` のような比較は，要素ごとの真偽値を並べた配列を返す。
この真偽値の配列を添字に使っても，要素を選択できる。
上の例では，絶対値が 10 を超える要素を `np.nan` に置き換え，その部分で線を途切れさせている。
この置き換えは，未定義点をまたぐ線を避けるための表示上の処理である。
また，浮動小数点数で表した $\pi/2$ は誤差を含む近似値である。
このため，`np.tan(np.pi / 2)` が大きな有限値を返すことがある。

### 8. 指数関数

$a^x$ の $a$ を **底**，$x$ を **指数** という。
指数が自然数の場合，$a^x$ は $a$ を繰り返し掛けた値として定義できる。
底が正の場合，次の **指数法則** が成り立つように定義を広げると，指数を実数まで拡張できる。

$$
a^m a^n=a^{m+n},\qquad
(a^m)^n=a^{mn},\qquad
(ab)^n=a^n b^n\quad(a,b>0)
$$

この拡張により，例えば $a^0=1$，$a^{-1}=1/a$，$a^{1/2}=\sqrt{a}$ となる。
**指数関数** $y=a^x$（$a>0,\ a\neq1$）は，すべての実数 $x$ に対して正の値をとり，点 $(0,1)$ を通る。
$a>1$ なら単調増加し，$0<a<1$ なら単調減少する。
特に，ネイピア数 $e=\lim_{n \rightarrow \infty} \left(1 + \frac{1}{n}\right)^n \simeq 2.71828\ldots$ を底とする $e^x=\exp(x)$ は，`np.exp(x)` で計算できる。

```python
x = np.linspace(-3, 3, 301)
fig, ax = plt.subplots()
ax.plot(x, 2.0**x, label="2**x")
ax.plot(x, 0.5**x, label="0.5**x")
ax.plot(x, np.exp(x), label="exp(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session03_example_exponential.png")
plt.show()
plt.close(fig)
```

`np.exp(-x**2)` のように，指数に別の式を入れることもできる。
この関数は $x=0$ で 1 となり，$|x|$ が大きくなるほど 0 に近づく。

### 9. 対数関数と定義域

**対数関数** $y=\log_a x$ は，「$a$ を何乗すると $x$ になるか」を返す関数である。
対数関数は，指数関数の入力と出力の対応を逆にした **逆関数** であり，次の関係を満たす。

$$
y=\log_a x\quad\Longleftrightarrow\quad x=a^y,
\qquad a>0,\ a\neq1,\ x>0
$$

例えば，$2^3=8$ なので $\log_2 8=3$ である。
関数の入力として許される範囲を **定義域** という。
実数の対数関数の定義域は $x>0$ である。

#### 対数法則と底の変換

$M,N>0$，$p\in\mathbb{R}$ のとき，次の **対数法則** が成り立つ。

$$
\log_a(MN)=\log_a M+\log_a N,\qquad
\log_a\frac{M}{N}=\log_a M-\log_a N,\qquad
\log_a(M^p)=p\log_a M
$$

このため，積の対数は対数の和として計算できる。
底が $e$ の対数を **自然対数**，底が 10 の対数を **常用対数** と呼ぶ。
NumPy では，自然対数を `np.log`，常用対数を `np.log10`，底が 2 の対数を `np.log2` で計算する。
それ以外の底の対数は，次の底の変換公式で計算できる。

$$
\log_a x=\frac{\log_b x}{\log_b a},\qquad b>0,\ b\neq1
$$

```python
print(np.log2(8.0))                 # 3.0
print(np.log10(100.0))              # 2.0
print(np.log(np.e))                 # 約 1.0
print(np.log(9.0) / np.log(3.0))    # 約 2.0：底が 3 の対数

x = np.linspace(0.01, 5, 500)        # 0 を含めない
fig, ax = plt.subplots()
ax.plot(x, np.log2(x), label="log2(x)")
ax.plot(x, np.log(x), label="log(x)")
ax.plot(x, np.log10(x), label="log10(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session03_example_logarithm.png")
plt.show()
plt.close(fig)
```

3 つのグラフはいずれも点 $(1,0)$ を通る。
この例のように底が 1 より大きければ，対数関数は単調増加する。

実数の配列に定義域の外の値を与えると，NumPy は次の値を返し，通常は警告を出す。

- `np.log(0.0)` は `-inf`（負の無限大）を返す。
- 負の数に対する `np.log` や `np.sqrt` は `nan`（数値として表せない結果）を返す。

`np.log(0.0)` が `-inf` となるのは $x\to+0$ での極限に相当する。

### 10. ブロードキャスト

**ブロードキャスト (broadcasting)** は，形の異なる配列同士で要素ごとの演算を行うための仕組みである。
例えば，配列にスカラー 2 を掛けると，全要素に 2 を掛けた配列が得られる。

#### 形をそろえる規則

2 つの配列の次元数が異なる場合は，まず次元数の少ない方の `shape` の先頭に長さ 1 の軸を補う。
そのうえで `shape` を末尾の軸から比較し，各軸が次のいずれかを満たせば演算できる。

- 軸の長さが同じである。
- どちらかの軸の長さが 1 である。

長さ 1 の軸では，同じ値を繰り返し演算に用いる。
条件を満たさない場合は `ValueError` になる。

| 配列Aの形 | 配列Bの形 | 小さい方（B）の次元を補ったもの | 結果の形 |
| --- | --- | --- | --- |
| `(3, 4)` | `(4,)` | `(1, 4)` | `(3, 4)` |
| `(2, 1, 3)` | `(3, 1)` | `(1, 3, 1)` | `(2, 3, 3)` |
| `(3, 4)` | `(3,)` | `(1, 3)` | 演算できない：末尾の 4 と 3 が合わない |

```python
a = np.arange(12).reshape(3, 4)
b = np.array([10, 20, 30, 40])
print(a + b)  # 各行に b を加える

p = np.arange(6).reshape(2, 1, 3)
q = np.array([[10], [20], [30]])
print((p + q).shape)  # (2, 3, 3)

row_offsets = np.array([10, 20, 30])
print(row_offsets[:, np.newaxis].shape)  # (3, 1)
print(a + row_offsets[:, np.newaxis])   # 各行に別々の値を加える
```

`a + b` の結果の 1 行目は `[10, 21, 32, 43]` になる。
`np.newaxis` は，指定した位置に長さ 1 の軸を挿入する。
軸を挿入しても要素数は変わらない。
`row_offsets[:, np.newaxis]` の形は `(3, 1)` だが，`row_offsets[np.newaxis, :]` の形は `(1, 3)` になる。
このように，軸を挿入する位置によって形が変わるため，挿入位置が重要である。

## 演習

演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第3回の第1問であれば，`exercise/exc03_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc03`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第5問の図であれば，`outputs/exc03/exc03_05.png` とします。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえで NumPy で計算した結果と比較してください。

### 基礎レベル（9問）
1. 配列の形を読めるようにしておくと，形の食い違いによるエラーを避けられます。次の各式が作る配列の `shape`，`ndim`，`size` を予想してください。`exercise/exc03_01.py` に各式を書き，実際にそれらを表示して予想と比較してください。

    - `np.zeros((2, 3))`
    - `np.arange(3, 10, 2)`
    - `np.linspace(0, 1, 5)`
    - `np.arange(12).reshape(2, 3, 2)`
    - `np.array([[1, 2, 3]])`
    - `np.array([[1], [2], [3]])`
    - `np.eye(3)[0]`

    さらに，次の 2 つにも取り組んでください。

    1. `[0.0, 0.25, 0.5, 0.75, 1.0]` という配列を，`np.arange` と `np.linspace` のそれぞれで作ってください。
    2. `np.arange(12).reshape(5, 3)` を実行すると何が起きるか予想してください。実行して結果を確かめ，理由を書いてください。

2. `a = np.arange(12).reshape(3, 4)` という配列に対して，以下の結果がそれぞれ得られるように要素を選択するコードを `exercise/exc03_02.py` に書いてください。

    ```python
    # 以下の ... という部分を添字に置き換えてください。

    a = np.arange(12).reshape(3, 4)
    res = a[...]
    print(res)
    ```

    - 2 行 3 列目の要素 `6`
    - 4 列目 `[ 3  7 11]`
    - 2〜3 行目の 2〜3 列目 `[[ 5  6] [ 9 10]]`
    - 1 列目と 3 列目 `[[ 0  2] [ 4  6] [ 8 10]]`（スライスの刻み幅を使ってください）
    - 対角に並ぶ要素 `[ 0  5 10]`（整数配列による選択を使ってください）
    - 6 より大きい要素 `[ 7  8  9 10 11]`（比較で得た真偽値の配列を使ってください）

    最後に，`a` の 2 列目をすべて `-1` に置き換えてから `a` を表示してください。

3. スライスで得た配列が元の配列とデータを共有するかどうかを確かめます。次のコードを実行したとき，`a`，`b`，`c` がそれぞれどう表示されるか予想してください。`exercise/exc03_03.py` にコードを書き，予想と一致しているか確認し，なぜそのような結果になったのか考察してください。第2回の「再代入とデータの変更」と比べてもかまいません。

    ```python
    a = np.arange(6)
    b = a[1:4]
    c = a[1:4].copy()
    b[0] = 100
    c[1] = 200
    print(a)
    print(b)
    print(c)
    ```

4. 次の各式の結果と，配列の場合はさらに `dtype` を予想してください。`exercise/exc03_04.py` にコードを書き，実際に実行して予想と比較してください。

    - `[1, 2] * 3`
    - `np.array([1, 2]) * 3`
    - `np.array([1, 2]) * np.array([3, 4])`
    - `2 ^ 3`
    - `np.array([1, 2, 3]) / 2`
    - `np.array([1.5, 2.5, -1.7]).astype(np.int64)`
    - `x = np.array([1, 2, 3])` としたあとで `x[0] = 2.7` を実行したときの `x`
    - `0.1 + 0.2 == 0.3`

    最後の式の結果を確認したら，[`np.isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) を使って `0.1 + 0.2` と `0.3` がほぼ等しいことを確かめてください。

5. 初等関数をまとめて計算し，1 枚の図に描きます。`exercise/exc03_05.py` に以下の処理を書いてください。

    1. `x = np.linspace(-5, 5, 501)` の隣り合う点の間隔を手計算し，`x` の先頭 5 要素を予想してください。予想を書いてから，実際に表示して確かめてください。
    2. `y = x`，`y = x**2`，`y = np.sin(x)`，`y = np.exp(-x**2)` を計算し，それぞれの先頭 5 要素，`shape`，`dtype` を表示してください。
    3. 4 つの関数を 1 枚の図に描き，凡例，軸ラベル，グリッドを付けて `outputs/exc03/exc03_05.png` に保存してください。

6. 直線と 2 次関数の式を手で求め，グラフで確かめます。

    1. 2 点 $(-1,-1)$ と $(2,5)$ を通る直線 $y=ax+b$ の傾き $a$ と切片 $b$ を手計算してください。
    2. $f(x)=2x^2-4x+5$ を平方完成し，頂点の座標を手計算してください。
    3. `exercise/exc03_06.py` で，1 の直線と 2 の放物線を $-3\le x\le3$ の範囲で 1 枚の図に描き，`outputs/exc03/exc03_06.png` に保存してください。2 点 $(-1,-1)$，$(2,5)$ と 2 の頂点は `scatter` で重ねて描いてください。
    4. [`np.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html) は，配列の最小値を持つ要素のインデックスを返します。これを使って $f(x)$ が最小となる `x` の値を求め，2 の頂点の $x$ 座標と比較してください。

7. グラフの平行移動を式から読み取ります。$f(x)=x^2$ とします。

    1. $g(x)=(x-2)^2+1$ と $h(x)=(x+1)^2-2$ のグラフは，$f$ のグラフをそれぞれどちらの方向にいくつ平行移動したものか答えてください。また，それぞれの頂点の座標を答えてください。
    2. $f$ のグラフを左に 3，下に 2 だけ平行移動したグラフを表す関数 $k(x)$ を求め，展開した形で答えてください。
    3. `exercise/exc03_07.py` で `f` を Python の関数として定義し，`g`，`h`，`k` を `f` を使って計算してください（例：`f(x - 2) + 1`）。4 つのグラフを 1 枚の図に描き，`outputs/exc03/exc03_07.png` に保存してください。図から 1 と 2 の答えが正しいことを確認してください。

8. 三角関数の値を単位円から求めます。

    1. $120^\circ$，$225^\circ$，$-90^\circ$ をそれぞれ弧度法に変換してください。
    2. 1 の各角度に対する $\sin\theta$，$\cos\theta$，$\tan\theta$ の値を，単位円を使って手計算してください。定義されない値があれば，そのことを答えてください。
    3. `exercise/exc03_08.py` で，度数法の角度を並べた配列を弧度法に変換し，`np.sin`，`np.cos`，`np.tan` で 2 の値を計算して比較してください。
    4. $\tan(-90^\circ)$ に対して NumPy がどのような値を返したかを確認し，その理由を考察してください。

9. 指数法則と対数法則を使って，次の値を手計算してください。`exercise/exc03_09.py` で同じ値を NumPy で計算し，手計算の結果と比較してください。底が 2，$e$，10 以外の対数は，底の変換公式を使って計算してください。

    - $2^{10}$
    - $8^{2/3}$
    - $4^{-1/2}$
    - $\log_2 32$
    - $\log_{10} 0.001$
    - $\log_3\sqrt{3}$
    - $\log_2 12-\log_2 3$
    - $\log_4 8$

    NumPy の結果が手計算の値と完全には一致しない場合は，その理由を考察してください。

### 発展レベル（6問）
1. 関数の定義域と，定義域の外の値を与えたときの NumPy の挙動を確かめます。

    1. 次のコードの各要素の結果を予想してください。`exercise/exc03_10.py` で実行し，表示される警告と `inf`，`nan` の意味を説明してください。

        ```python
        x = np.array([4.0, 1.0, 0.25, 0.0, -1.0])
        print(np.log2(x))
        print(np.sqrt(x))
        ```

    2. 実数の範囲で考えたとき，次の関数の定義域を手計算で求めてください。

        - $\log(x-1)$
        - $\sqrt{4-x^2}$
        - $\log(\sin x)$（$0\le x\le4\pi$ の範囲で考えてください）

    3. 2 の 3 つの関数を，定義域より広い範囲の `x` に対して計算し，それぞれ別の subplot に描いて `outputs/exc03/exc03_10.png` に保存してください。グラフが描かれた範囲が 2 の答えと一致することを確認してください。

2. `plot` は点と点を線分で結ぶだけなので，点の数によって曲線の見え方が変わります。

    1. `np.linspace(0, 2 * np.pi, 3)` と `np.linspace(0, 2 * np.pi, 5)` の各点における $\sin x$ の値を手計算してください。これらの点を `plot` で結ぶと，それぞれどのような図になるか予想してください。
    2. `exercise/exc03_11.py` で，点数を 3，5，21，101 に変えて $\sin x$（$0\le x\le2\pi$）を描いてください。4 つのグラフを `plt.subplots(2, 2)` で 1 枚の図にまとめ，`outputs/exc03/exc03_11.png` に保存してください。
    3. 1 の予想と図を比較してください。また，点数が 3 のときに $\sin x$ が直線に見える理由を考察してください。

3. $\tan x$ のグラフで，未定義点の扱いを確かめます。

    1. $0\le x\le4\pi$ の範囲で $\tan x$ が定義されない点をすべて手計算で求めてください。
    2. `exercise/exc03_12.py` で，`np.linspace(0, 4 * np.pi, 1001)` に対する $\tan x$ を，そのまま描いた図と，解説の例と同様に絶対値が 10 を超える要素を `np.nan` に置き換えて描いた図を，上下 2 段の subplot にまとめてください。`outputs/exc03/exc03_12.png` に保存してください。
    3. 上段の図で縦に伸びる線が現れる位置と 1 の答えを比較し，その線が $\tan x$ のグラフの一部ではない理由を説明してください。

4. ブロードキャストの規則を使って，演算結果の形を予想します。

    1. 次の形を持つ配列 A と B について，`A + B` の結果の形を予想してください。演算できない場合は，その理由を答えてください。

        | 配列Aの形 | 配列Bの形 |
        | --- | --- |
        | `(4, 3)` | `(3,)` |
        | `(4, 3)` | `(4,)` |
        | `(4, 1)` | `(1, 3)` |
        | `(2, 3, 4)` | `(3, 1)` |
        | `(5,)` | `(5, 1)` |
        | `(2, 1, 4)` | `(3, 1)` |

    2. `exercise/exc03_13.py` で，`np.zeros` を使って各形の配列を作り，`A + B` の形を確かめてください。演算できない組み合わせは，他の組み合わせとは分けて実行してください。
    3. `k = np.arange(1, 5)` と `x = np.linspace(0, 2 * np.pi, 501)` から，`for` 文を使わずに $\sin(kx)$（$k=1,2,3,4$）を並べた形 `(4, 501)` の配列を作ってください。各行を 1 枚の図に重ねて描き，`outputs/exc03/exc03_13.png` に保存してください。

5. 指数関数と対数関数では，平行移動と拡大・縮小が同じグラフを表すことがあります。

    1. 指数法則を使って，$2^{x+1}=2\cdot2^x$ を示してください。この式から，$y=2^x$ のグラフを左に 1 平行移動したグラフと，縦方向に 2 倍したグラフが一致することを説明してください。
    2. 対数法則を使って，$\log_2(2x)=\log_2 x+1$（$x>0$）を示してください。この式から，$y=\log_2 x$ のグラフについて同様の関係を説明してください。
    3. `exercise/exc03_14.py` で 1 と 2 の両辺を計算し，差の絶対値の最大値を `np.max(np.abs(...))` で表示してください。また，両辺のグラフを重ねて描き，`outputs/exc03/exc03_14.png` に保存してください。

6. 第4回の数値微分で使えるように，関数を標本化する処理を関数にまとめてください。`exercise/exc03_15.py` に，Python の関数 `f`，区間の両端 `start`，`stop`，点数 `num` を受け取り，`np.linspace(start, stop, num)` で作った `x` と，それに対する `f(x)` の値 `y` を返す関数 `sample_function(f, start, stop, num)` を定義してください。

    この関数を使って，以下のようなコードが書けるものとします。

    ```python
    x, y = sample_function(np.sin, 0, 2 * np.pi, 101)
    print(x.shape, y.shape)  # => (101,) (101,)
    ```

    次に，この `x` と `y` を列として並べた形 `(101, 2)` の配列を作り，[`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html) で `outputs/exc03/exc03_15_sin_table.csv` に保存してください。列を並べるには [`np.column_stack`](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html) を使えます。`np.savetxt` には `delimiter=","`，`header="x,y"`，`comments=""` を指定してください。保存したファイルをテキストエディタで開き，1 行目が列名 `x,y`，2 行目以降がカンマ区切りの数値になっていることを確認してください。

## 詰まったときに見る資料
- [`../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md`](../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md)
- [`../textbook/markdown/ch08-applied-matplotlib.md`](../textbook/markdown/ch08-applied-matplotlib.md)
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)
