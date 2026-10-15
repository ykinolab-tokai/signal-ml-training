# 第16回 連続最適化

## この回の目標

勾配を使ってパラメータを更新し，関数の値を小さくする方法を説明し，実装できる。

## 解説

この回では，第4回で扱った微分を多変数に広げ，関数の値を小さくするパラメータを数値計算で探す。
ここで扱う勾配降下法は，第17回以降で PyTorch を使ってモデルを学習させるときの基礎になる。
スクリプトの先頭には，第4回と同様に次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figure_dir = Path("outputs/exc16")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. 偏微分と勾配

#### 偏微分

複数の入力をとる関数を **多変数関数** という。
例えば，$f(x, y) = x^2 + 3y^2$ は，2 つの入力 $x$，$y$ をとる 2 変数関数である。
多変数関数で，1 つの変数以外を定数とみなし，その変数について微分することを **偏微分 (partial differentiation)** という。
$f$ を $x$ で偏微分した関数を $\dfrac{\partial f}{\partial x}$，$y$ で偏微分した関数を $\dfrac{\partial f}{\partial y}$ と書き，それぞれ次の極限で定義する。

$$
\frac{\partial f}{\partial x}(x, y) = \lim_{h\to0}\frac{f(x+h, y) - f(x, y)}{h},\qquad
\frac{\partial f}{\partial y}(x, y) = \lim_{h\to0}\frac{f(x, y+h) - f(x, y)}{h}
$$

例えば，$f(x, y) = x^2 + 3y^2$ を $x$ で偏微分するときは，$y$ を定数とみなすので，$3y^2$ も定数として扱う。
定数の導関数は 0 なので，$\dfrac{\partial f}{\partial x} = 2x$ である。
同様に，$x$ を定数とみなして $y$ で偏微分すると，$\dfrac{\partial f}{\partial y} = 6y$ である。
$xy$ のような変数の積も，$x$ で偏微分するときは $y$ を定数とみなし，$\dfrac{\partial}{\partial x}(xy) = y$ となる。
数値計算では，偏微分する変数だけを $\pm h$ 動かした中心差分で近似する（「勾配と等高線」のコード例の `grad_x`，`grad_y`）。

#### 勾配と等高線

各変数についての偏微分を並べたベクトルを **勾配 (gradient)** といい，$\nabla f$ と書く（$\nabla$ はナブラと読む）。

$$
\nabla f(x, y) = \left(\frac{\partial f}{\partial x}(x, y),\ \frac{\partial f}{\partial y}(x, y)\right)^\top
$$

勾配 $\nabla f(x, y)$ は，点 $(x, y)$ で関数の値が最も急に増加する向きを表す。
逆向きのベクトル $-\nabla f(x, y)$ は，関数の値が最も急に減少する向きを表す。
関数の値が等しい点を結んだ曲線を **等高線 (contour line)** という。
勾配 $\nabla f(x, y)$ は，点 $(x, y)$ を通る等高線と直交する。
グラフが滑らかな多変数関数の極小点では，勾配は零ベクトル（すべての成分が 0 のベクトル）になる。

```python
def f(x, y):
    return x**2 + 3 * y**2

x = np.linspace(-3, 3, 61)
y = np.linspace(-2, 2, 41)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
print(X.shape, Y.shape, Z.shape)  # (41, 61) (41, 61) (41, 61)

x0, y0, h = 1.5, 0.5, 1e-5
grad_x = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
grad_y = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)
print(grad_x, grad_y)  # 約 3.0 約 3.0

fig, ax = plt.subplots()
contours = ax.contour(X, Y, Z, levels=[0.5, 1, 2, 3, 5, 8])
ax.clabel(contours)
ax.scatter(x0, y0)
ax.quiver(x0, y0, 0.2 * grad_x, 0.2 * grad_y, angles="xy", scale_units="xy", scale=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
fig.savefig(figure_dir / "session16_example_gradient.png")
plt.show()
plt.close(fig)
```

`np.meshgrid(x, y)` は，`x` と `y` の値のすべての組合せを表す 2 次元配列 `X`，`Y` を返す。
`X[i, j]` は `x[j]` に，`Y[i, j]` は `y[i]` に等しく，配列の形は `(len(y), len(x))` になる。
行が $y$，列が $x$ に対応することに注意する。
`f` が配列を受け取れる関数なら，`f(X, Y)` で，すべての格子点（`x` と `y` の値の組で決まる点）での値をまとめて計算できる。
`ax.contour` は `levels` で指定した値の等高線を描き，`ax.clabel` は等高線に値のラベルを付ける。
`ax.quiver(x0, y0, u, v, angles="xy", scale_units="xy", scale=1)` は，点 $(x_0, y_0)$ から成分 $(u, v)$ の矢印を描く。
`angles="xy"`，`scale_units="xy"`，`scale=1` を指定すると，$(u, v)$ が座標軸の目盛りと同じ単位で扱われ，矢印の終点は $(x_0+u, y_0+v)$ になる。
この例では，見やすさのため，勾配を 0.2 倍した長さで描いている。
`ax.set_aspect("equal")` は，縦軸と横軸の目盛りの間隔をそろえる。
目盛りの間隔が縦と横で異なると，直交する勾配と等高線も，図の上では直角からずれて見える。

### 2. 勾配降下法

#### 更新式

第4回の第1節で見たように，傾きが正の点では左へ，負の点では右へ進むと，関数の値は小さくなる。
つまり，傾きの符号と逆の向きに進めばよい。
この考えに基づき，初期値 $x_0$ から次の更新を繰り返して極小点に近づける方法を **勾配降下法 (gradient descent)** という。

$$
x_{k+1} = x_k - \eta f'(x_k),\qquad k = 0, 1, 2, \ldots
$$

$\eta$（イータ）は 1 回の更新で進む量を決める正の定数で，**学習率 (learning rate)** と呼ばれる。
$\eta>0$ なので，$f'(x_k)>0$ のときは $x$ が減って左へ進み，$f'(x_k)<0$ のときは $x$ が増えて右へ進む。
多変数関数では，変数を並べたベクトルを $\boldsymbol{x}_k$ とし，導関数の代わりに勾配を使って次のように更新する。

$$
\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \eta\nabla f(\boldsymbol{x}_k)
$$

第1節で見たように $-\nabla f$ は関数の値が最も急に減少する向きなので，この更新は，その点で最も急に下る向きへ少しだけ進むことを表す。

```python
def f(x):
    return (x - 1)**2

def df(x):
    return 2 * (x - 1)

eta = 0.25
x = 3.0
history = [x]
for k in range(5):
    x = x - eta * df(x)
    history.append(x)
print(history)  # [3.0, 2.0, 1.5, 1.25, 1.125, 1.0625]
print(f(x))     # 0.00390625
```

この例では，$x_k$ が $f(x)=(x-1)^2$ の極小点 $x=1$ に近づいている。
$x_k$ と極小点 $x=1$ との距離は，1 回の更新ごとに半分になっている。
`history` には，初期値と各回の更新後の値を順に記録している。
このように途中の値を記録しておくと，更新の様子をグラフで確かめられる。

#### 学習率

学習率が小さすぎると，1 回の更新で進む量が小さく，極小点に近づくまでに多くの更新が必要になる。
逆に大きすぎると，極小点を飛び越えて振動したり，極小点から離れていったりする。
極小点から離れていくことを **発散 (divergence)** という。
上の例と同じ関数と初期値で，学習率を $\eta=1.25$ にすると次のようになる。

```python
eta = 1.25
x = 3.0
history = [x]
for k in range(4):
    x = x - eta * df(x)
    history.append(x)
print(history)  # [3.0, -2.0, 5.5, -5.75, 11.125]
```

$x_k$ は極小点 $x=1$ の左右を行き来しながら，極小点との距離が $2, 3, 4.5, 6.75, 10.125$ と 1.5 倍ずつ大きくなっており，発散している。
どの範囲の学習率なら極小点に近づくかは，関数の形によって決まる。
実際の機械学習では関数の形が事前に分からないため，いくつかの学習率で試し，関数の値の減り方を見て選ぶ。

#### 局所解と初期値

最適化では，極小点（第4回）のことを **局所解 (local optimum)** という。
考えている範囲全体で関数が最小となる点は，**大域的最適解 (global optimum)** という。
極小点が複数ある関数では，勾配降下法で到達する局所解が初期値によって変わり，到達した局所解で関数が最小値をとるとは限らない。
次の例では，2 つの極小点を持つ関数 $g(x)=x^4-2x^2+0.3x$ に対して，2 つの初期値から学習率 0.05 で勾配降下法を 100 回行う。

```python
def g(x):
    return x**4 - 2 * x**2 + 0.3 * x

def dg(x):
    return 4 * x**3 - 4 * x + 0.3

for x0 in [-2.0, 2.0]:
    x = x0
    for k in range(100):
        x = x - 0.05 * dg(x)
    print(x0, round(x, 3), round(g(x), 3))
```

```text
-2.0 -1.036 -1.305
2.0 0.96 -0.706
```

`round(x, 3)` は，`x` を小数点以下 3 桁に丸める組み込み関数である。
初期値 $-2$ からは左の極小点 $x\approx-1.036$ に，初期値 $2$ からは右の極小点 $x\approx0.960$ に到達している。
関数の値は左の極小点のほうが小さいので，左の極小点が大域的最適解であり，初期値 $2$ から到達した点は大域的最適解ではない局所解である。
勾配降下法は，各点の勾配，つまりその点の近くの情報だけを使って進むため，別の谷にあるより低い点には向かえない。
一方，極小点が 1 つだけの谷の形をした関数では，学習率が大きすぎなければ，どの初期値からでも同じ極小点に近づく。
第3節で扱う最小二乗法の損失は，この形をしている。

### 3. 最小二乗法

#### 問題設定

観測したデータの組 $(x_i, y_i)$（$i=1,\ldots,N$）に，直線 $y=ax+b$ を当てはめることを考える。
第15回と同様に，損失には次の平均二乗誤差を使う。

$$
L(a,b)=\frac{1}{N}\sum_{i=1}^{N}(ax_i+b-y_i)^2
$$

この損失を最小にする $(a, b)$ を求める方法を **最小二乗法 (least squares method)** という。
第15回では，候補の値をすべて調べるグリッドサーチで，損失が最小となる $(a, b)$ を探した。
最小二乗法の損失は $a$，$b$ の 2 次関数なので，勾配が零ベクトルになる条件から，解を直接求められる。

#### 行列による表記

データとパラメータを行列とベクトルで表すと，式と計算をまとめて書ける。
各行が $(x_i, 1)$ の $N$ 行 2 列の行列 $\boldsymbol{A}$，パラメータのベクトル $\boldsymbol{w}$，正解のベクトル $\boldsymbol{y}$ を

$$
\boldsymbol{A}=\begin{pmatrix} x_1 & 1 \\ x_2 & 1 \\ \vdots & \vdots \\ x_N & 1 \end{pmatrix},\qquad
\boldsymbol{w}=\begin{pmatrix} a \\ b \end{pmatrix},\qquad
\boldsymbol{y}=\begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{pmatrix}
$$

とすると，予測を並べたベクトルは $\boldsymbol{A}\boldsymbol{w}$ であり，その第 $i$ 成分は $ax_i+b$ である。
したがって，損失は，予測と正解の差 $\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y}$ のノルム（第5回）を使って次のように書ける。

$$
L(\boldsymbol{w})=\frac{1}{N}\|\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y}\|^2
$$

#### 勾配と正規方程式

$\boldsymbol{A}$ の第 $i$ 行を $\boldsymbol{a}_i^\top$，$i$ 行 $j$ 列の要素を $A_{ij}$ とすると，$L(\boldsymbol{w})=\frac{1}{N}\sum_{i}(\boldsymbol{a}_i^\top\boldsymbol{w}-y_i)^2$ である。
$\boldsymbol{a}_i^\top\boldsymbol{w}=\sum_j A_{ij}w_j$ を $w_j$ で偏微分すると $A_{ij}$ になるので，連鎖律（第4回）により次の式が成り立つ。

$$
\frac{\partial L}{\partial w_j}=\frac{2}{N}\sum_{i=1}^{N}(\boldsymbol{a}_i^\top\boldsymbol{w}-y_i)A_{ij}
$$

右辺の和は，$\boldsymbol{A}^\top$ の第 $j$ 行と $\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y}$ の内積である。
したがって，すべての $j$ についてまとめると，勾配は次のように書ける。

$$
\nabla L(\boldsymbol{w})=\frac{2}{N}\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y})
$$

極小点では勾配が零ベクトルになるので，次の式が成り立つ。

$$
\boldsymbol{A}^\top\boldsymbol{A}\boldsymbol{w}=\boldsymbol{A}^\top\boldsymbol{y}
$$

この連立一次方程式を **正規方程式 (normal equation)** という。
$\boldsymbol{A}^\top\boldsymbol{A}$ は 2 行 2 列の行列なので，第5回で扱った `np.linalg.solve` で解ける。

```python
x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([1.0, 3.0, 4.0, 8.0])
A = np.column_stack([x, np.ones_like(x)])
print(A.shape)                  # (4, 2)
w = np.linalg.solve(A.T @ A, A.T @ y)
print(w)                        # [2.2 0.7]
print(np.mean((A @ w - y)**2))  # 約 0.45
```

`np.column_stack` は，1 次元配列を列として並べた 2 次元配列を作る。
`np.ones_like(x)` は，`x` と同じ形で全要素が 1 の配列を返す。
したがって，`A` の各行は $(x_i, 1)$ になる。
この例で最小二乗法により求めた直線は，$y=2.2x+0.7$ である。

#### 勾配降下法による解法

同じ問題は，勾配 $\nabla L(\boldsymbol{w})$ を使った勾配降下法（第2節）でも解ける。

```python
w_gd = np.zeros(2)
eta = 0.1
for k in range(500):
    grad = 2 / len(y) * A.T @ (A @ w_gd - y)
    w_gd = w_gd - eta * grad
print(w_gd)  # 約 [2.2 0.7]
```

`np.zeros(2)` は，初期値 $(a,b)=(0,0)$ を表す。
500 回の更新の後，`w_gd` は正規方程式の解とほぼ一致する。
最小二乗法の損失は，極小点が 1 つだけの谷の形をしているため，学習率が大きすぎなければ，どの初期値から始めても同じ解に近づく。
式で解ける問題では直接解くほうが速く正確だが，ニューラルネットワークのように損失の極小点を式で求められない問題では，勾配降下法を使う。

#### ridge 回帰

損失に，パラメータの大きさを表す項 $\lambda\|\boldsymbol{w}\|^2$（$\lambda>0$）を加えた，次の損失を考える。

$$
L_\lambda(\boldsymbol{w})=\frac{1}{N}\|\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y}\|^2+\lambda\|\boldsymbol{w}\|^2
$$

$\|\boldsymbol{w}\|^2=\sum_j w_j^2$ を $w_j$ で偏微分すると $2w_j$ になるので，勾配は $\frac{2}{N}\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y})+2\lambda\boldsymbol{w}$ である。
これを零ベクトルとおき，両辺に $N/2$ を掛けて整理すると，次の方程式が得られる。

$$
(\boldsymbol{A}^\top\boldsymbol{A}+N\lambda\boldsymbol{I})\boldsymbol{w}=\boldsymbol{A}^\top\boldsymbol{y}
$$

ここで，$\boldsymbol{I}$ は単位行列である。
この方法を **ridge 回帰 (ridge regression)** という。
加えた項は，パラメータが大きくなりすぎるのを抑える働きをする。
損失を平均ではなく和 $\|\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y}\|^2+\lambda\|\boldsymbol{w}\|^2$ で定義した場合は，同じ計算により $(\boldsymbol{A}^\top\boldsymbol{A}+\lambda\boldsymbol{I})\boldsymbol{w}=\boldsymbol{A}^\top\boldsymbol{y}$ となる。

```python
lam = 0.1
N = len(y)
w_ridge = np.linalg.solve(A.T @ A + N * lam * np.eye(2), A.T @ y)
print(w_ridge)                                     # 約 [2.11988 0.74561]
print(np.linalg.norm(w), np.linalg.norm(w_ridge))  # 約 2.30868 約 2.24719
```

`np.eye(2)` は，2 行 2 列の単位行列である（第3回）。
$\lambda=0.1$ としたときの解のノルムは，項を加えない場合より小さくなっている。
この項を加える意味は，第18回で確率の観点から説明する。

### 4. 確率的勾配降下法

#### ミニバッチによる勾配の推定

データ数 $N$ が大きいと，全データでの損失の勾配を毎回計算するのに時間がかかる。
そこで，データからランダムに選んだ一部の集合 $\mathcal{B}$ を使い，その中だけで計算した損失

$$
L_{\mathcal{B}}(\boldsymbol{w})=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}(\boldsymbol{a}_i^\top\boldsymbol{w}-y_i)^2
$$

の勾配を使って更新を繰り返す。
この方法を **確率的勾配降下法 (stochastic gradient descent, SGD)** という。
$\mathcal{B}$ を **ミニバッチ (mini-batch)**，その要素数 $|\mathcal{B}|$ を **バッチサイズ (batch size)** という。
$\boldsymbol{A}$ と $\boldsymbol{y}$ からミニバッチに含まれる行だけを取り出したものを $\boldsymbol{A}_{\mathcal{B}}$，$\boldsymbol{y}_{\mathcal{B}}$ とすると，勾配は第3節と同じ形の式で計算できる。

$$
\nabla L_{\mathcal{B}}(\boldsymbol{w})=\frac{2}{|\mathcal{B}|}\boldsymbol{A}_{\mathcal{B}}^\top(\boldsymbol{A}_{\mathcal{B}}\boldsymbol{w}-\boldsymbol{y}_{\mathcal{B}})
$$

バッチサイズが $N$ のときは，第3節の勾配降下法と同じになる。

#### エポックと更新回数

全データを 1 回ずつ使い切るまでの更新のまとまりを **エポック (epoch)** という。
よく使われる実装では，各エポックの最初にデータの順番を `rng.permutation` でランダムに並べ替え，先頭からバッチサイズずつ取り出して更新する。
1 エポックあたりの更新回数は，$N$ をバッチサイズで割って切り上げた値になる。
次の例では，第3節の 4 点のデータに対して，バッチサイズ 2 で 3 エポック学習する。

```python
rng = np.random.default_rng(0)
w_sgd = np.zeros(2)
eta = 0.05
batch_size = 2
N = len(y)
for epoch in range(3):
    indices = rng.permutation(N)
    for start in range(0, N, batch_size):
        batch = indices[start:start + batch_size]
        A_b, y_b = A[batch], y[batch]
        grad = 2 / len(batch) * A_b.T @ (A_b @ w_sgd - y_b)
        w_sgd = w_sgd - eta * grad
    print(epoch, round(np.mean((A @ w_sgd - y)**2), 4))
```

```text
0 2.1755
1 0.5575
2 0.4907
```

`range(0, N, batch_size)` は，各ミニバッチの先頭の位置 `0`，`2` を順に生成する。
`indices[start:start + batch_size]` は並べ替えた添字からバッチサイズ分を取り出し，`A[batch]` はその添字の行だけを選択する（第3回の整数配列による選択）。
$N$ がバッチサイズで割り切れない場合，最後のミニバッチは小さくなるが，スライスは配列の末尾で止まるため，同じコードで扱える。
勾配の係数に `batch_size` ではなく `len(batch)` を使っているのは，このためである。
各エポックの終わりに全データでの損失を表示すると，損失が正規方程式の解での値 0.45 に向かって減っていることが分かる。

#### 勾配のばらつき

ミニバッチで求めた勾配は，全データでの勾配の推定値であり，どのデータを選んだかによって変わる。
ただし，同じ大きさのミニバッチをすべての選び方について平均すると，全データでの勾配に一致する。
次の例では，4 点から 2 点を選ぶ 6 通りのミニバッチのそれぞれについて，$\boldsymbol{w}=\boldsymbol{0}$ での勾配を計算する。

```python
from itertools import combinations

w0 = np.zeros(2)
full_grad = 2 / N * A.T @ (A @ w0 - y)
batch_grads = []
for batch in combinations(range(N), 2):
    batch = list(batch)
    A_b, y_b = A[batch], y[batch]
    batch_grads.append(2 / len(batch) * A_b.T @ (A_b @ w0 - y_b))
batch_grads = np.array(batch_grads)
print(full_grad)                 # [-17.5  -8. ]
print(batch_grads[:, 0])         # [ -3.  -8. -24. -11. -27. -32.]
print(batch_grads.mean(axis=0))  # [-17.5  -8. ]
```

`combinations(range(N), 2)` は，`0` から `N - 1` までの整数から 2 つを選ぶすべての組を順に返す，Python の標準ライブラリ `itertools` の関数である。
ミニバッチの勾配の第 1 成分は $-3$ から $-32$ まで大きくばらつくが，6 通りの平均は全データでの勾配と一致する。
このため，SGD の 1 回の更新では全データでの損失が小さくなるとは限らないが，更新を重ねると，平均的には全データでの勾配の向きに進む。
バッチサイズを大きくすると勾配のばらつきは小さくなるが，1 回の更新に必要な計算量は増える。

#### 実際に使われる最適化手法

実際の学習では，SGD を改良した方法がよく使われる。
過去の更新の向きを蓄積して振動を抑える方法を **momentum** といい，パラメータごとに更新量を調整する方法の代表例に **Adam** がある。
これらは，第17回以降で PyTorch の optimizer として使う。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第16回の第1問であれば，`exercise/exc16_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc16`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第1問の図であれば，`outputs/exc16/exc16_01.png` とします。
数値微分の方法や刻み幅の指定がない場合は，中心差分と刻み幅 `h = 1e-5` を使ってください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえで NumPy で計算した結果と比較してください。

### 基礎レベル（4問）
1. 2 変数関数の偏微分と勾配を求め，等高線と重ねて描きます。$q(x, y) = x^2 + xy + y^2$ とします。

    1. $\dfrac{\partial q}{\partial x}$ と $\dfrac{\partial q}{\partial y}$ を手計算し，点 $(1, 1)$，$(-2, 1)$，$(2, -2)$ における勾配 $\nabla q$ を求めてください。
    2. `exercise/exc16_01.py` で，3 つの点における 2 つの偏微分を中心差分で近似し，1 の結果と比較してください。
    3. $-3\le x\le3$，$-3\le y\le3$ の範囲で $q$ の等高線を描き，3 つの点における勾配ベクトルを，長さを 0.2 倍した矢印で重ねてください。`ax.set_aspect("equal")` で縦軸と横軸の目盛りの間隔をそろえ，`outputs/exc16/exc16_01.png` に保存してください。
    4. 図を見て，勾配ベクトルが等高線とどのような角度で交わるかを答えてください。また，勾配ベクトルが，関数の値が大きくなる向きと小さくなる向きのどちらを指しているかを答えてください。さらに，点 $(-2, 1)$ の勾配ベクトルが $x$ 軸と平行になる理由を，1 の結果と，この点の近くの等高線の向きから説明してください。

2. 勾配降下法で関数の極小点を求めます。$f(x) = (x-2)^2 + 1$，初期値 $x_0 = 0$，学習率 $\eta = 0.1$ とします。

    1. $f'(x)$ を求め，勾配降下法の更新式に代入して，$x_1$，$x_2$，$x_3$ を手計算してください。
    2. `exercise/exc16_02.py` で勾配降下法の更新を 20 回行い，各回の $x_k$ と $f(x_k)$ をリストに記録して表示してください。1 の手計算の結果と比較してください。
    3. `plt.subplots(1, 2)` を使い，左に $y=f(x)$ のグラフと点 $(x_k, f(x_k))$ の列を描いてください。右には，横軸を更新回数 $k$，縦軸を $x_k$ としたグラフを描いてください。図は `outputs/exc16/exc16_02.png` に保存してください。
    4. 更新式を変形し，$x_{k+1}-2$ を $x_k-2$ を使って表してください。この式から，1 回の更新で極小点 $x=2$ までの距離が何倍になるかを求め，$x_k$ が極小点に近づき続ける理由を説明してください。

3. 最小二乗法で直線を当てはめ，勾配降下法の結果と比べます。データには，第15回の基礎第3問と同じ方法で作った 40 個の $(x_i, y_i)$ をすべて使い，損失は $L(a,b)=\frac{1}{N}\sum_{i}(ax_i+b-y_i)^2$ とします。

    1. $L(a,b)$ の $a$ と $b$ についての偏微分を手計算してください。また，その結果が，解説の勾配 $\frac{2}{N}\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y})$ の 2 つの成分と一致することを，$\boldsymbol{A}^\top$ の各行が何を並べたものかに注目して確かめてください。
    2. `exercise/exc16_03.py` でデータを作り，1 で求めた 2 つの偏微分を計算する関数を定義してください。点 $(a, b) = (0.5, -0.3)$ で，その関数の値と，行列を使って計算した勾配が一致することを確かめてください。
    3. 正規方程式を `np.linalg.solve` で解いて $(a, b)$ を求めてください。求めた値を，データを作った式の係数 $(2, 1)$ と，第15回の基礎第3問でグリッドサーチにより求めた $(a, b)$ と比較してください。第15回では train の 30 個だけを使ったことに注意してください。
    4. 学習率 $\eta=0.1$，初期値 $(a, b)=(0, 0)$ で，$L$ を勾配降下法により 1000 回更新してください。各回の更新後の $L$ をリストに記録してください。最後の $(a, b)$ と 3 の解との差の絶対値の最大値を表示し，$10^{-5}$ 以下になっていることを確かめてください。
    5. `plt.subplots(1, 2, figsize=(10, 4))` を使い，左にデータと，2 つの方法で求めた直線を描いてください。右には，横軸を更新回数，縦軸を $L$ とした片対数グラフを描いてください。片対数グラフは `ax.semilogy` で描けます。右の図には，3 の解での $L$ の値を `ax.axhline` で水平線として重ねてください。図は `outputs/exc16/exc16_03.png` に保存してください。
    6. 勾配降下法の結果が正規方程式の解に近づく理由を，$L$ が $(a, b)$ の 2 次関数であることと，正規方程式が表す条件から説明してください。また，求めた $(a, b)$ がデータを作った式の係数 $(2, 1)$ と完全には一致しない理由と，右の図で $L$ が 0 まで下がらない理由を説明してください。

4. 確率的勾配降下法で，バッチサイズによる違いを調べます。基礎第3問と同じデータと損失を使い，学習率 $\eta=0.1$，初期値 $(a, b)=(0, 0)$ で 20 エポック学習します。各エポックの最初に，`rng.permutation` でデータの順番を並べ替えてください。並べ替えには，データを作った乱数生成器とは別に，条件ごとに `np.random.default_rng(1)` で作った乱数生成器を使ってください。

    1. バッチサイズを 1，8，40 としたときの，1 エポックあたりの更新回数を手計算してください。また，20 エポック後の全データでの $L$ が最も小さくなる条件と，エポックごとの $L$ の変化が最も揺れる条件を予想してください。
    2. `exercise/exc16_04.py` で，3 つのバッチサイズのそれぞれについて学習し，各エポックの終わりに全データでの $L$ を記録してください。更新回数の合計も数えて表示し，1 の手計算の結果と比較してください。
    3. 横軸をエポック，縦軸を $L$ とした片対数グラフに 3 つの条件の結果を重ねて描いてください。基礎第3問の正規方程式の解での $L$ の値も `ax.axhline` で重ね，`outputs/exc16/exc16_04.png` に保存してください。
    4. 1 の予想と結果を比較し，バッチサイズによって，1 エポックあたりの更新回数，損失の下がり方，損失の揺れ方がどう変わるかを説明してください。また，バッチサイズ 1 で $L$ が正規方程式の解での値の付近で揺れ続ける理由を，ミニバッチの勾配のばらつきから説明してください。

### 発展レベル（3問）
1. 勾配降下法の学習率によって，$x_k$ が極小点に近づく様子がどう変わるかを調べます。基礎第2問と同じく，$f(x) = (x-2)^2 + 1$，初期値 $x_0 = 0$ とします。

    1. 基礎第2問の 4 で求めた式を使い，学習率 $\eta = 0.01, 0.1, 0.5, 0.9, 1.0, 1.1$ のそれぞれについて，$x_k$ の変化が次の (a)〜(e) のどれになるかを予想してください。

        - (a) 2 の片側から単調に 2 に近づく
        - (b) 2 の両側を行き来しながら 2 に近づく
        - (c) 2 の両側を同じ幅で往復し続ける
        - (d) 2 から離れていく
        - (e) 1 回の更新で 2 に到達する

    2. `exercise/exc16_05.py` で，各学習率について更新を 30 回行ってください。横軸を更新回数 $k$，縦軸を $\lvert x_k-2\rvert$ とした片対数グラフに，6 つの学習率の結果を重ねて描き，`outputs/exc16/exc16_05.png` に保存してください。$\lvert x_k-2\rvert$ が 0 になる点は片対数グラフに描けないため，その学習率では $x_k$ の値を表示して確認してください。
    3. 1 の予想と結果を比較してください。また，$\eta = 0.1$ と $\eta = 0.9$ のグラフが重なる理由を説明してください。さらに，この関数で $x_k$ が 2 に近づくための学習率の範囲と，最も速く近づく学習率を答えてください。
    4. 実際の機械学習では関数の形が分からないため，最も速く近づく学習率を事前に計算できません。小さめの学習率と大きめの学習率のそれぞれについて，利点と欠点を 2 の結果を根拠に説明してください。

2. 極小点が複数ある関数で，グリッドサーチと勾配降下法を比べます。$f(x) = 3x^4 - 4x^3 - 12x^2 + 5$ とします。

    1. $f'(x)$ を因数分解して $f'(x) = 0$ となる $x$ をすべて求め，各点で極大と極小のどちらになるかを手計算してください。また，極小値のうち小さい方を答えてください。
    2. `exercise/exc16_06.py` で，$-2\le x\le3$ を 501 点に分けた各点で $f(x)$ を計算し，`np.argmin` で $f(x)$ が最小となる $x$ を求めて，1 の結果と比較してください。このように，候補の点をすべて調べて最小となる点を探す方法が，第15回で扱ったグリッドサーチです。
    3. 学習率 $\eta = 0.01$ の勾配降下法を，初期値 $x_0 = -2, -0.5, 0, 0.5, 3$ のそれぞれから 100 回更新し，最後の $x_k$ と $f(x_k)$ を表示してください。$y=f(x)$ のグラフに，各初期値から得た点 $(x_k, f(x_k))$ の列を重ねて描き，`outputs/exc16/exc16_06.png` に保存してください。
    4. 初期値によって到達する点が異なる理由と，$x_0 = 0$ から更新が進まない理由を説明してください。
    5. 変数が $d$ 個ある関数で，各変数の範囲を 100 点ずつに分けてグリッドサーチを行うとします。このときに関数を評価する回数を，$d$ の式で表してください。$d = 10$ の場合の回数を勾配降下法の 100 回の更新と比べ，グリッドサーチと勾配降下法をどのような場面で使い分けるとよいかを考察してください。

3. 勾配降下法で得られる点 $(x_k, y_k)$ を順に結んだ折れ線（軌跡）を等高線上に描き，学習率の影響を調べます。$f(x, y) = x^2 + 10y^2$，初期値 $(x_0, y_0) = (-4, 1)$ とします。

    1. $\nabla f(x, y)$ を手計算し，勾配降下法の更新式を $x$ と $y$ の成分ごとに書いてください。さらに，$x_{k+1}$ と $y_{k+1}$ がそれぞれ $x_k$ と $y_k$ の何倍になるかを，学習率 $\eta$ の式で表してください。
    2. 1 の結果から，$\eta = 0.02, 0.09, 0.11$ のそれぞれについて，軌跡がどのような形になるかを予想してください。また，$(x_k, y_k)$ が極小点 $(0, 0)$ に近づくための $\eta$ の範囲を答えてください。
    3. `exercise/exc16_07.py` で，各学習率について更新を 30 回行い，$f$ の等高線に軌跡を重ねて描いてください。3 つの学習率の図は，`plt.subplots(1, 3, figsize=(12, 4))` で 1 枚にまとめてください。各図では，`ax.set_xlim(-5, 5)` と `ax.set_ylim(-3, 3)` で表示範囲をそろえ，`ax.set_aspect("equal")` を指定してください。図は `outputs/exc16/exc16_07.png` に保存してください。
    4. 2 の予想と結果を比較してください。また，$x$ 方向と $y$ 方向で極小点に近づく速さが異なる理由と，学習率の上限が $y$ 方向で決まる理由を説明してください。

## 詰まったときに見る資料
- [`04-differentiation-and-integration.md`](04-differentiation-and-integration.md)
- [`../textbook/markdown/ch21-regression-analysis.md`](../textbook/markdown/ch21-regression-analysis.md)
- [`../textbook/markdown/ch23-basics-of-neural-networks.md`](../textbook/markdown/ch23-basics-of-neural-networks.md)
