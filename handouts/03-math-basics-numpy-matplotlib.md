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

| 配列Aのサイズ | 配列Bのサイズ | 小さい方（B）の次元を補ったもの | 結果のサイズ |
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

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル（7問）
1. `scripts/`，`outputs/session03/`，`outputs/figures/`，`outputs/data/` を作成し，`scripts/session03_math_numpy_matplotlib.py` で `np.linspace(-5, 5, 501)` による配列 `x` を作ってください。
2. `y = x`，`y = x**2`，`y = np.sin(x)`，`y = np.exp(-x**2)` を計算し，それぞれの先頭 5 要素，shape，dtype を表示してください。
3. 4 つの関数を 1 枚の図に描き，凡例，軸ラベル，grid を付けて `outputs/figures/session03_elementary_functions.png` に保存してください。
4. `f(x) = x**2`，`g(x) = (x - 2)**2 + 1`，`h(x) = (x + 1)**2 - 2` を描き，平行移動の方向を確認してください。
5. `np.array([[1, 2, 3], [4, 5, 6]])` を作り，shape，dtype，1 行目，2 列目，右下の値を取り出してください。
6. Python のリストに対する `+` と NumPy 配列に対する `+` の違いを，短い例で確認してください。
7. 作成した図のファイル名，配列の shape，平行移動の読み取り結果を，コードコメントまたは既存の結果ファイルに記録してください。別レポートは作らないでください。

### 発展レベル（5問）
1. `np.log(x)` と `np.sqrt(x)` に定義域の外の値を与えたときに何が起きるかを確認し，警告や `nan` の意味を書いてください。
2. `np.linspace` の点数を 21，101，1001 に変え，`sin(x)` の図の滑らかさがどう変わるかを比較してください。
3. 同じ関数を複数の subplot に分けて描き，1 枚の図として保存してください。
4. `np.savetxt` を使って，`x` と `sin(x)` の対応表を `outputs/data/session03_sin_table.csv` に保存してください。
5. 第4回で数値微分に使えるように，`f(x)` を受け取って `x` と `y` を返す関数を 1 つ定義してください。

## 詰まったときに見る資料
- [`../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md`](../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md)
- [`../textbook/markdown/ch08-applied-matplotlib.md`](../textbook/markdown/ch08-applied-matplotlib.md)
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)
