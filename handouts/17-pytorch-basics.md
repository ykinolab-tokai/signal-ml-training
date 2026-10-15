# 第17回 PyTorch の基礎

## この回の目標

PyTorch の tensor と自動微分を使い，勾配降下法による学習ループを実装できる。

## 解説

この回では，第16回で NumPy を使って実装した勾配降下法を，深層学習のためのライブラリ **PyTorch** を使って書き直す。
第16回では，損失関数の勾配を手計算で導出してからプログラムに書いた。
PyTorch を使うと，損失を計算する処理を書くだけで，勾配を自動で計算できる。
スクリプトの先頭には，次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

figure_dir = Path("outputs/exc17")
figure_dir.mkdir(parents=True, exist_ok=True)
```

`torch` は PyTorch の本体であり，`torch.nn` はモデルや損失関数などの部品をまとめたモジュールである。
ここでは，`torch.nn` を `nn` という名前でインポートしている。
以下のコード例は，このコードに続けて書いて実行する。

### 1. tensor

#### tensor の作成と属性

PyTorch では，多次元配列を **tensor** という型で表す。
tensor は，第3回で扱った NumPy の `ndarray` と同様に，形を表す `shape` とデータ型を表す `dtype` を持つ。

```python
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = torch.tensor([1, 2, 3])
print(a.shape, a.dtype)   # torch.Size([2, 3]) torch.float32
print(b.shape, b.dtype)   # torch.Size([3]) torch.int64
print(a.ndim, a.numel())  # 2 6
```

`torch.tensor` は，リストなどから tensor を作る。
`shape` は `torch.Size` という型で表示されるが，NumPy の `shape` と同じくタプルとして扱える。
`ndim` は軸の数を，`numel()` は全要素数を返す（NumPy の `size` に当たる）。
小数を含むリストから作った tensor の `dtype` は `torch.float32` になり，NumPy の既定の `float64` とは異なる。
PyTorch のモデルは，計算の速さとメモリの使用量の観点から，`float32` で計算することが多い。
整数だけのリストから作った tensor の `dtype` は `torch.int64` になる。

NumPy と同様に，値や範囲を指定して tensor を作る関数が用意されている。

| 書き方 | 作られる tensor |
| --- | --- |
| `torch.zeros(2, 3)` | 全要素が 0 の 2 行 3 列の tensor |
| `torch.ones(2, 3)` | 全要素が 1 の 2 行 3 列の tensor |
| `torch.arange(3, 10, 2)` | 開始値 3 から刻み幅 2 で並べた `tensor([3, 5, 7, 9])`（終了値 10 を含まない） |
| `torch.linspace(-5, 5, 101)` | −5 から 5 までを，両端を含めて等間隔に並べた 101 点 |
| `torch.eye(3)` | 3 行 3 列の単位行列 |
| `torch.randn(2, 3)` | 標準正規分布（平均 0，標準偏差 1）に従う乱数を並べた 2 行 3 列の tensor |

`torch.zeros((2, 3))` のように，形をタプルで渡すこともできる。
PyTorch の乱数は，`torch.manual_seed(0)` のようにシードを指定すると再現できる。
これは第6回で扱った NumPy の乱数生成器 `rng` とは別の仕組みである。
このため，NumPy と PyTorch の両方で乱数を使う場合は，それぞれにシードを指定する。

#### NumPy との相互変換

`torch.from_numpy` は NumPy 配列から tensor を作り，tensor の `numpy()` メソッドは tensor から NumPy 配列を作る。

```python
x_np = np.linspace(0, 1, 3)
x = torch.from_numpy(x_np)
x32 = x.float()
print(x.dtype, x32.dtype)  # torch.float64 torch.float32

x_np[0] = 10.0
print(x)    # tensor([10.0000,  0.5000,  1.0000], dtype=torch.float64)
print(x32)  # tensor([0.0000, 0.5000, 1.0000])
```

`torch.from_numpy` で作った tensor は，元の NumPy 配列とデータを共有する。
このため，`x_np[0]` を変更すると，`x` の値も変わる。
一方，`float()` で `float32` に変換した `x32` は新しいデータを持つため，変わらない。
`torch.tensor(x_np)` のように `torch.tensor` で作った場合も，データはコピーされる。
逆に，`numpy()` で作った NumPy 配列は，元の tensor とデータを共有する。

`torch.from_numpy` は，元の配列の `dtype` を引き継ぐ。
NumPy の `float64` の配列を変換すると `float64` の tensor になるため，モデルに入力する前に `float()`（または `to(torch.float32)`）で `float32` に変換する。
`float64` の入力を `float32` のモデルに渡すと，データ型が一致しないことを示すエラーになる。

#### 演算と形の操作

同じ形の tensor 同士の `+`，`-`，`*`，`/` は要素ごとの演算であり，`@` は行列積である。
形の異なる tensor 同士の演算には，第3回で扱ったブロードキャストの規則がそのまま適用される。
和や平均などの集計では，NumPy の `axis` の代わりに `dim` で軸を指定する。

```python
a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
w = torch.tensor([1.0, 0.0, -1.0])
print(a.sum(dim=0))                       # tensor([3., 5., 7.])
print(a.mean(dim=1))                      # tensor([1., 4.])
print(a.mean(dim=1, keepdim=True).shape)  # torch.Size([2, 1])
print(a @ w)                              # tensor([-2., -2.])
print(a.unsqueeze(0).shape)               # torch.Size([1, 2, 3])
```

`a.sum(dim=0)` は軸 0 に沿って和をとり，各列に 1 つの値を残す。
`keepdim=True` を指定すると，集計した軸が長さ 1 の軸として残る（NumPy の `keepdims=True` に当たる）。
`a @ w` は形 `(2, 3)` の行列と長さ 3 のベクトルの積であり，結果の形は `(2,)` になる。
`unsqueeze(0)` は，位置 0 に長さ 1 の軸を挿入する（NumPy の `np.newaxis` に当たる）。
逆に，長さ 1 の軸を取り除くには `squeeze` を使う。
`reshape` は NumPy と同様に形を変える。
引数に `-1` を書いた軸の長さは，要素数が変わらないように自動で決まる。
なお，`mean` は浮動小数点数の tensor にしか使えず，`int64` の tensor に使うとエラーになる。

### 2. 自動微分

#### requires_grad と backward

PyTorch では，`requires_grad=True` を指定した tensor を使って計算すると，その計算の過程が記録される。
計算結果に対して `backward()` を呼ぶと，記録をたどって勾配が計算され，元の tensor の `grad` 属性に保存される。
この仕組みを **自動微分 (automatic differentiation)** という。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x
print(y)       # tensor(10., grad_fn=<AddBackward0>)
y.backward()
print(x.grad)  # tensor(7.)
```

$y = x^2 + 3x$ の導関数は $2x + 3$ なので，$x = 2$ における微分係数は 7 であり，`x.grad` と一致する。
`y` の表示にある `grad_fn` は，`y` がどの演算で作られたかの記録である。
`backward()` は，要素が 1 つの tensor（スカラー）に対して呼ぶ。
複数の要素を持つ tensor から勾配を求めるときは，損失関数のように和や平均をとって 1 つの値にしてから `backward()` を呼ぶ。

#### 計算グラフと連鎖律

自動微分は，計算の過程を **計算グラフ (computational graph)** として記録し，第4回で扱った連鎖律を使って勾配を求める。
例えば，$y = (3x+4)^2$ は，$u = 3x + 4$ と $y = u^2$ の合成関数である。

```python
x = torch.tensor(1.0, requires_grad=True)
u = 3 * x + 4
y = u**2
y.backward()
print(x.grad)  # tensor(42.)
```

入力から出力へ向かう計算を **順伝播 (forward propagation)** という。
この例の順伝播では，$x = 1$ から $u = 7$，$y = 49$ を計算し，途中の値を記録する。
`backward()` を呼ぶと，出力側から入力側へ向かって，各演算の微分係数を掛け合わせる。
この計算を **逆伝播 (backpropagation)** という。
この例では，$\dfrac{dy}{du} = 2u = 14$ と $\dfrac{du}{dx} = 3$ を掛けて，$\dfrac{dy}{dx} = 42$ を得る。
これは，第4回で求めた導関数 $6(3x+4)$ に $x = 1$ を代入した値と一致する。
`backward()` を呼ぶと，記録した途中の値は解放される。
このため，同じ計算結果に対して `backward()` をもう一度呼ぶとエラーになる。
もう一度勾配を求めるときは，順伝播の計算からやり直す。

#### 勾配の蓄積と torch.no_grad

`backward()` は，計算した勾配を `grad` に代入するのではなく，それまでの値に足し合わせる。

```python
x = torch.tensor(1.0, requires_grad=True)
for _ in range(2):
    y = x**2
    y.backward()
print(x.grad)  # tensor(4.)
x.grad.zero_()
print(x.grad)  # tensor(0.)
```

$x^2$ の $x = 1$ における微分係数は 2 だが，`backward()` を 2 回呼んだため，`x.grad` は $2 + 2 = 4$ になった。
勾配を使ってパラメータを更新するときは，更新のたびに勾配を 0 に戻す必要がある。
`zero_()` のように名前の末尾に `_` が付くメソッドは，その tensor 自身を書き換える。

パラメータの更新のように，勾配を求める必要がない計算は，`with torch.no_grad():` のブロックの中に書く。
このブロックの中の計算は，計算グラフに記録されない。
次の例は，第16回の第2節の勾配降下法の例（$f(x) = (x-1)^2$，学習率 0.25，初期値 3）の 1 回目の更新を，自動微分を使って書いたものである。

```python
w = torch.tensor(3.0, requires_grad=True)
loss = (w - 1)**2
loss.backward()
with torch.no_grad():
    w -= 0.25 * w.grad
w.grad.zero_()
print(w)         # tensor(2., requires_grad=True)
print(w.item())  # 2.0
```

`w.grad` は $f'(3) = 4$ であり，更新後の `w` は $3 - 0.25 \times 4 = 2$ になる。
これは第16回の例の `history` の 2 番目の値と一致する。
`item()` は，要素が 1 つの tensor から Python の数値を取り出す。
`requires_grad=True` の tensor に `numpy()` を使うとエラーになる。
グラフの描画などで NumPy 配列が必要なときは，`w.detach().numpy()` のように，`detach()` で計算の記録から切り離してから変換する。

### 3. 勾配の確認

自動微分で求めた勾配が正しいかどうかは，第4回の数値微分と比べて確かめられる。
パラメータ $w$ を $\pm h$ だけ動かして損失 $L$ を計算し，中心差分

$$
\frac{L(w+h) - L(w-h)}{2h}
$$

を自動微分で求めた勾配と比べる。
パラメータが複数ある場合は，1 つのパラメータだけを動かし，他のパラメータは固定して計算する。
数値微分では，パラメータ 1 つにつき損失を 2 回計算する必要がある。
ニューラルネットワークのパラメータは数百万個に及ぶこともあるため，学習には自動微分を使い，数値微分は勾配の確認だけに使う。

数値微分の刻み幅 $h$ は，tensor のデータ型に合わせて選ぶ。
`float64` 型の数値は 10 進数で約 16 桁の有効数字を持つが，`float32` 型の数値は約 7 桁しか持たない。
第4回で説明した桁落ちにより，$h$ を小さくしすぎると，差 $L(w+h) - L(w-h)$ の正しい有効数字が失われる。

```python
def g(w):
    return torch.sin(w) * w**2

def central_diff(f, w, h):
    return (f(w + h) - f(w - h)) / (2 * h)

for dtype, h in [(torch.float64, 1e-5), (torch.float32, 1e-5), (torch.float32, 1e-3)]:
    w = torch.tensor(1.5, dtype=dtype, requires_grad=True)
    g(w).backward()
    with torch.no_grad():
        numerical = central_diff(g, w, h)
    print(dtype, h, w.grad.item(), numerical.item())
```

```text
torch.float64 1e-05 3.151643663564495 3.151643663423797
torch.float32 1e-05 3.151643753051758 3.159046173095703
torch.float32 0.001 3.151643753051758 3.1517741680145264
```

`float64` で $h = 10^{-5}$ とした中心差分と自動微分の差は，約 $10^{-10}$ である。
`float32` で同じ $h$ を使うと，差は約 $7\times10^{-3}$ に広がる。
`float32` でも $h = 10^{-3}$ にすると，差は約 $10^{-4}$ に縮まる。
このように，`float32` のまま勾配を確かめるときは，第4回で使った $h = 10^{-5}$ より大きい刻み幅を使う。
より厳密に確かめたいときは，モデルとデータを `float64` に変換してから比べる。

### 4. モデル・損失関数・optimizer

#### nn.Linear

`nn.Linear(in_features, out_features)` は，長さ `in_features` の入力ベクトル $\boldsymbol{x}$ を，長さ `out_features` のベクトル

$$
\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}
$$

に変換する層である。
このように，入力のすべての要素が出力のすべての要素に関係する層を **全結合層 (fully connected layer)** という。
$\boldsymbol{W}$ を **重み (weight)**，$\boldsymbol{b}$ を **バイアス (bias)** と呼ぶ。
`nn.Linear(1, 1)` は，第15回と第16回で扱った直線のモデル $y = ax + b$ と同じである。

```python
torch.manual_seed(0)
model = nn.Linear(3, 2)
print(model.weight.shape, model.bias.shape)  # torch.Size([2, 3]) torch.Size([2])
x = torch.ones(4, 3)
print(model(x).shape)                        # torch.Size([4, 2])
for name, p in model.named_parameters():
    print(name, p.shape, p.requires_grad)
```

```text
weight torch.Size([2, 3]) True
bias torch.Size([2]) True
```

重み `model.weight` の形は `(out_features, in_features)` であり，入力と出力の順序が逆になることに注意する。
PyTorch のモデルには，複数のサンプルをまとめた tensor を入力する。
軸 0 はサンプルの番号を表し，形 `(N, in_features)` の入力から形 `(N, out_features)` の出力が得られる。
各行がそれぞれ 1 つのサンプルの入力であり，出力は $\boldsymbol{X}\boldsymbol{W}^\top + \boldsymbol{b}$ で計算される。
`model(x)` のようにモデルを関数のように呼び出すと，順伝播の計算が行われる。
重みとバイアスは `requires_grad=True` の tensor であり，作成時に乱数で初期化される。
このため，結果を再現するには，モデルを作る直前に `torch.manual_seed` でシードを指定する。
`named_parameters()` はパラメータの名前と値の組を，`parameters()` はパラメータの値だけを順に返す。
パラメータに値を設定するときは，`with torch.no_grad():` の中で `model.weight.fill_(0.5)` のように書く。

#### 損失関数

`nn.MSELoss()` は，予測値 $\hat{y}_i$ と正解 $y_i$ の平均二乗誤差

$$
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

を計算する損失関数である。

```python
loss_fn = nn.MSELoss()
pred = torch.tensor([[1.0], [2.0], [4.0]])
target = torch.tensor([[1.0], [3.0], [2.0]])
print(loss_fn(pred, target))  # tensor(1.6667)
```

この例では，$(0^2 + 1^2 + 2^2) / 3 \approx 1.6667$ である。
予測値と正解は同じ形にそろえる。
例えば，形 `(3, 1)` の予測値と形 `(3,)` の正解を渡すと，ブロードキャストにより形 `(3, 3)` の差が計算され，誤った値になる。
この場合は警告が表示されるので，見落とさないようにする。

#### optimizer

**optimizer** は，勾配を使ってパラメータを更新する規則を実装したものである。
`torch.optim.SGD(model.parameters(), lr=0.1)` は，学習率 0.1 の勾配降下法の optimizer を作る。
`optimizer.step()` は，渡されたすべてのパラメータ $\boldsymbol{\theta}$ を，それぞれの `grad` を使って

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla L(\boldsymbol{\theta})
$$

と更新する。
これは第16回の勾配降下法の更新式と同じである。
`optimizer.zero_grad()` は，渡されたすべてのパラメータの勾配を消去する（PyTorch 2 系では，`grad` を `None` にする）。

第16回で触れた momentum は，`torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)` のように指定する。
momentum を使うと，更新の向き $\boldsymbol{v}$ に過去の勾配が蓄積される。

$$
\boldsymbol{v}_{k+1} = \beta\boldsymbol{v}_k + \nabla L(\boldsymbol{\theta}_k),\qquad
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\boldsymbol{v}_{k+1}
$$

$\beta$（`momentum` の値，$0\le\beta<1$）は，過去の勾配をどれだけ残すかを決める。
勾配が毎回同じ向きを指す方向では $\boldsymbol{v}$ が大きくなり，更新が速くなる。
勾配の向きが毎回反転する方向では，過去の勾配と打ち消し合い，振動が抑えられる。
ただし，$\beta$ が大きすぎると，極小点を通り過ぎて振動しやすくなる。

### 5. 学習ループと scheduler

#### 学習ループ

モデル，損失関数，optimizer を組み合わせて，パラメータの更新を繰り返す処理を **学習ループ (training loop)** という。
次の例では，$y = -3x + 0.5$ を満たす 20 個のデータに，`nn.Linear(1, 1)` を当てはめる。

```python
x = torch.linspace(-1, 1, 20).reshape(-1, 1)
y = -3 * x + 0.5

torch.manual_seed(0)
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss_history = []
for step in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

print(model.weight.item(), model.bias.item())  # 約 -2.9986 約 0.5000

fig, ax = plt.subplots()
ax.semilogy(loss_history)
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.grid(True)
fig.savefig(figure_dir / "session17_example_loss.png")
plt.show()
plt.close(fig)
```

`reshape(-1, 1)` は，20 個の値を形 `(20, 1)`，つまり「20 個のサンプル × 1 個の特徴」に並べ替える。
ループの 1 回分は，次の 5 つの処理からなる。

1. `optimizer.zero_grad()`：前回の更新で求めた勾配を消去する。
2. `pred = model(x)`：順伝播で予測値を計算する。
3. `loss = loss_fn(pred, y)`：損失を計算する。
4. `loss.backward()`：逆伝播で各パラメータの勾配を求める。
5. `optimizer.step()`：勾配を使ってパラメータを更新する。

パラメータの 1 回の更新を **ステップ (step)** と呼ぶ。
損失の記録には `loss.item()` を使い，Python の数値として保存する。
`loss` の tensor をそのままリストに追加すると，計算グラフも一緒に保存され続けるため，メモリを無駄に使う。
この例では，100 ステップで重みが約 $-2.9986$，バイアスが約 $0.5000$ になり，データを作った直線 $y = -3x + 0.5$ に近づいている。
損失は更新のたびにほぼ一定の割合で小さくなるため，第16回と同様に，縦軸を対数目盛にした `semilogy` で描くと変化を読み取りやすい。

#### scheduler

学習の途中で学習率を変える仕組みを **scheduler** という。
`torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)` は，`scheduler.step()` が `step_size` 回呼ばれるたびに，学習率を `gamma` 倍にする。

```python
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

lr_history = []
for step in range(6):
    lr_history.append(scheduler.get_last_lr()[0])
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
print(lr_history)  # [0.1, 0.1, 0.05, 0.05, 0.025, 0.025]
```

`scheduler.get_last_lr()` は，現在の学習率をリストで返す（optimizer に渡したパラメータのグループごとに 1 つの値を持つため，リストになっている）。
`scheduler.step()` は，`optimizer.step()` の後に呼ぶ。
順序を逆にすると，最初の学習率が使われないまま変更されるため，警告が表示される。
学習率の値は，`0.010000000000000002` のように，丸め誤差を含んで表示されることがある。

optimizer は「勾配を使ってパラメータをどう更新するか」を決め，scheduler は「その更新に使う学習率を学習の進行に応じてどう変えるか」を決める。
学習の初めは大きな学習率で速く進み，後半は小さな学習率で極小点の近くに落ち着かせる，という使い方が多い。
ただし，学習率を下げるのが早すぎると，極小点に近づく前に更新がほとんど進まなくなる。
第20回以降のように，データ全体を 1 回ずつ使うエポックを単位として学習する場合は，`scheduler.step()` をエポックごとに 1 回呼ぶことが多い。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第17回の第1問であれば，`exercise/exc17_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc17`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第3問の図であれば，`outputs/exc17/exc17_03.png` とします。
乱数を使う問題では，指定がない限り，データの生成に `rng = np.random.default_rng(0)` を使い，モデルを作る直前に `torch.manual_seed(0)` を実行してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（6問）
1. tensor の作り方によって，データ型や，NumPy 配列とデータを共有するかどうかが変わることを確かめます。

    1. 次の各式が作る tensor の `shape` と `dtype` を予想してください。`exercise/exc17_01.py` に各式を書き，実際に表示して予想と比較してください。

        - `torch.tensor([1, 2, 3])`
        - `torch.tensor([1.0, 2.0])`
        - `torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)`
        - `torch.zeros(2, 3)`
        - `torch.arange(6).reshape(2, 3)`
        - `torch.linspace(0, 1, 5)`
        - `torch.from_numpy(np.linspace(0, 1, 5))`

    2. 次のコードを実行したとき，`t1`，`t2`，`t3` がそれぞれどう表示されるかを予想してください。実行して予想と比較し，それぞれの tensor が `a_np` とデータを共有しているかどうかを説明してください。

        ```python
        a_np = np.arange(6).reshape(2, 3)
        t1 = torch.from_numpy(a_np)
        t2 = torch.tensor(a_np)
        t3 = t1.float()
        a_np[0, 0] = 100
        print(t1)
        print(t2)
        print(t3)
        ```

    3. `model = nn.Linear(5, 1)` に `torch.from_numpy(np.ones((2, 5)))` を入力すると何が起きるかを予想してください。他の処理とは分けて実行し，表示されたエラーメッセージの意味を説明してください。そのうえで，入力を `float()` で `float32` に変換してから入力し，出力の `shape` が `(2, 1)` になることを確かめてください。

2. tensor の形の操作と軸ごとの集計を確かめます。`x = torch.arange(12).reshape(3, 4)` とします。

    1. 次の各式の結果の `shape` を予想してください。`exercise/exc17_02.py` で実際に表示し，予想と比較してください。

        - `x.sum(dim=0)`
        - `x.sum(dim=1)`
        - `x.sum(dim=1, keepdim=True)`
        - `x.unsqueeze(0)`
        - `x.reshape(-1)`
        - `x @ x.T`

    2. `x.mean(dim=0)` を実行すると何が起きるかを予想してください。他の処理とは分けて実行して確かめ，`x.float().mean(dim=0)` との違いを説明してください。
    3. `xf = x.float()` とします。各列からその列の平均を引いた tensor と，各行からその行の平均を引いた tensor を作ってください。前者の列ごとの平均と，後者の行ごとの平均がすべて 0 になることを確かめてください。
    4. 行ごとの平均を引くとき，`keepdim=True` を付けずに `xf - xf.mean(dim=1)` と書くと何が起きるかを予想してください。他の処理とは分けて実行して確かめ，その理由を第3回のブロードキャストの規則を使って説明してください。

3. 自動微分で求めた微分係数を，手計算の値と比べます。

    1. $f(x) = x^3 - 3x$ の導関数を手計算し，$x = 2, -1, 0.5$ における微分係数を求めてください（第4回の基礎第3問と同じ関数です）。`exercise/exc17_03.py` で，自動微分を使って同じ値を求め，手計算の値と比較してください。
    2. $q(x, y) = x^2 + xy + y^2$ の点 $(1, 1)$，$(-2, 1)$ における偏微分 $\dfrac{\partial q}{\partial x}$，$\dfrac{\partial q}{\partial y}$ を手計算してください（第16回の基礎第1問と同じ関数です）。`requires_grad=True` を指定した 2 つの tensor `x`，`y` から $q$ を計算して `backward()` を呼び，`x.grad` と `y.grad` を手計算の値と比較してください。
    3. $x = 2$ の tensor について，$f(x)$ の計算と `backward()` を 2 回繰り返したときの `x.grad` の値を予想してください。実行して確かめ，学習ループで更新のたびに勾配を消去する必要がある理由を説明してください。
    4. `x = torch.linspace(-2, 2, 101, requires_grad=True)` について `f(x).sum()` を計算し，`backward()` を呼んでください。`x.grad` の各要素が，対応する点における $f'(x)$ の値と一致することを `torch.allclose` で確かめてください。また，`plt.subplots(2, 1, sharex=True)` を使い，上段に $f(x)$ を，下段に `x.grad` を描いて，`outputs/exc17/exc17_03.png` に保存してください。和をとってから微分すると各点の微分係数が得られる理由を説明してください。

4. `nn.Linear` のモデルの損失について，自動微分で求めた勾配を，手計算と数値微分で確かめます。データを `x = torch.linspace(-1, 1, 10).reshape(-1, 1)`，`y = 2 * x + 1` とします。モデルを `model = nn.Linear(1, 1)` とし，`with torch.no_grad():` の中で `model.weight.fill_(0.5)` と `model.bias.fill_(0.0)` を実行して，重みを $w = 0.5$，バイアスを $b = 0$ に設定してください。損失関数は `nn.MSELoss()` とします。

    1. 損失 $L(w, b) = \dfrac{1}{N}\sum_{i=1}^{N}(wx_i + b - y_i)^2$ の偏微分 $\dfrac{\partial L}{\partial w}$，$\dfrac{\partial L}{\partial b}$ を，手計算で式に表してください。`exercise/exc17_04.py` で，その式に $w = 0.5$，$b = 0$ と 10 個のデータを代入した値を NumPy で計算してください。
    2. 損失を計算して `backward()` を呼び，`model.weight.grad` と `model.bias.grad` を 1 の値と比較してください。
    3. $w$ についての偏微分を，中心差分で近似してください。`with torch.no_grad():` の中で `model.weight` を $\pm h$ だけ動かして損失を計算し，計算後に元の値に戻してください。$b$ についても同様に，`model.bias` だけを動かして近似してください。刻み幅を $h = 10^{-1}, 10^{-3}, 10^{-5}$ として，自動微分の値との差の絶対値を表示してください。
    4. `model.double()` でモデルを，`x.double()` と `y.double()` でデータを `float64` に変換して，3 と同じ計算を行ってください。どのデータ型と刻み幅で差が大きくなるかを答え，その理由を第4回の桁落ちの説明を使って説明してください。また，この損失では $h = 10^{-1}$ でも差が小さい理由を，損失が $w$，$b$ の 2 次関数であることと，第4回の「刻み幅と誤差」で扱った $f(x) = x^2$ の中心差分の例を使って説明してください。

5. 第16回の勾配降下法を自動微分を使って書き，`torch.optim.SGD` の結果と比べます。基礎第4問と同じデータ，モデル，初期値（$w = 0.5$，$b = 0$）を使い，学習率を $\eta = 0.1$ とします。

    1. 基礎第4問の 1 で求めた勾配を使って，1 回目の更新後の $w$ と $b$ を手計算してください。
    2. `exercise/exc17_05.py` で，`torch.optim` を使わずに勾配降下法の更新を 5 回行ってください。各回では，損失を計算して `backward()` を呼び，`with torch.no_grad():` の中で各パラメータ `p` を `p -= 0.1 * p.grad` で更新してから，勾配を消去してください。各回の更新後の $w$ と $b$ を表示し，1 回目の値を 1 の手計算の結果と比較してください。
    3. 同じ初期値から，`torch.optim.SGD(model.parameters(), lr=0.1)` を使って 5 回更新してください。5 回後の $w$ と $b$ が 2 の結果と一致することを，`torch.allclose` で確かめてください（計算の順序の違いにより，最後の桁が異なることがあるためです）。また，`optimizer.zero_grad()`，`loss.backward()`，`optimizer.step()` が，2 のどの処理に当たるかをコメントに書いてください。
    4. このデータでは $\sum_i x_i = 0$ であり，$y$ の平均は 1 なので，$\dfrac{\partial L}{\partial b} = 2(b - 1)$ となります。これを使って $b_{k+1} - 1$ を $b_k - 1$ で表し，2 で表示した $b$ の値の変化を説明してください。

6. 学習ループと scheduler を組み合わせて，学習率の変化と損失の下がり方の関係を調べます。`rng = np.random.default_rng(0)` を使い，`x = rng.uniform(-1, 1, size=(50, 1))`，`y = 2 * x + 1 + rng.normal(0, 0.1, size=(50, 1))` で 50 個のデータを作り，`float32` の tensor に変換してください。`torch.manual_seed(0)` を実行してから `model = nn.Linear(1, 1)` を作り，損失関数を `nn.MSELoss()`，optimizer を `torch.optim.SGD(model.parameters(), lr=0.1)`，scheduler を `torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)` とします。各ステップでは全データを使って 1 回更新し，`scheduler.step()` は `optimizer.step()` の直後に呼んでください。

    1. 30 ステップ学習するとき，ステップ 0〜29 のそれぞれで使われる学習率を予想してください。
    2. `exercise/exc17_06.py` で 30 ステップ学習し，各ステップの損失を `loss_history` に，そのステップの更新に使った学習率を `lr_history` に記録してください。2 つのリストの長さがともに 30 であることを確かめてください。ステップ 0，9，10，19，20，29 の損失と学習率を表示し，学習率を 1 の予想と比較してください。
    3. `plt.subplots(2, 1, sharex=True)` を使い，上段に損失を，下段に学習率を，それぞれ `semilogy` で描いて，`outputs/exc17/exc17_06.png` に保存してください。
    4. 学習後の重みとバイアスを表示し，第16回と同様に正規方程式を `np.linalg.solve` で解いて求めた最小二乗解と比較してください。図から，損失の下がり方が変わるステップと学習率が変わるステップの対応を読み取り，この設定では 30 ステップで最小二乗解に近づけなかった理由を説明してください。また，optimizer と scheduler の役割の違いをコメントに書いてください。

### 発展レベル（3問）
1. 学習ループから `optimizer.zero_grad()` を除くと何が起きるかを調べます。基礎第6問と同じデータ，モデル，初期値を使い，scheduler は使わずに学習率 0.1 で 30 ステップ学習します。

    1. `zero_grad()` を除いた場合に，損失の変化がどうなるかを予想してください。
    2. `exercise/exc17_07.py` で，`zero_grad()` を呼ぶ場合と呼ばない場合の各ステップの損失を記録し，1 枚の図に描いて `outputs/exc17/exc17_07.png` に保存してください。
    3. `zero_grad()` を呼ばない場合，$k$ 回目の更新に使われる量は，それまでに求めた勾配の和 $\sum_{j=0}^{k}\nabla L(\boldsymbol{\theta}_j)$ になります。このことを使って，損失が単調に減少せず，増減を繰り返す理由を説明してください。

2. ミニバッチを使う確率的勾配降下法で，scheduler が損失の揺れをどう変えるかを調べます。基礎第6問と同じデータを使い，`torch.manual_seed(0)` を実行してから `nn.Linear(1, 1)` を作ります。`rng_batch = np.random.default_rng(1)` とし，第16回の基礎第4問と同様に，各エポックの最初に `perm = rng_batch.permutation(50)` でデータの順番を並べ替え，5 個ずつのミニバッチで 20 エポック（200 回）更新します。optimizer は `torch.optim.SGD(model.parameters(), lr=0.5)` とします。

    1. 学習率を 0.5 に固定した場合と，`StepLR(optimizer, step_size=50, gamma=0.1)` で 50 回の更新ごとに学習率を 1/10 にした場合とで，最後の 50 回の更新における損失の揺れ方がどう異なるかを予想してください。
    2. `exercise/exc17_08.py` で 2 つの条件を実行してください。各更新の直後に，全データでの損失を `with torch.no_grad():` の中で計算して記録してください。2 つの条件の損失を 1 枚の図に `semilogy` で描き，`outputs/exc17/exc17_08.png` に保存してください。
    3. 最後の 50 回の更新における損失の平均と標準偏差を条件ごとに表示し，基礎第6問の 4 で求めた最小二乗解での損失と比較してください。
    4. 2 つの条件の違いを，ミニバッチで求めた勾配のばらつき（第16回）と学習率の大きさに結びつけて説明してください。また，基礎第6問では scheduler によって学習が遅くなったことと合わせて，scheduler が役立つ場面を説明してください。

3. momentum を使うと，第16回の発展第3問で扱った関数の極小点への近づき方がどう変わるかを調べます。$f(x, y) = x^2 + 10y^2$，初期値を $(x_0, y_0) = (-4, 1)$ とします。パラメータを `p = torch.tensor([-4.0, 1.0], requires_grad=True)` とし，`torch.optim.SGD([p], lr=0.02, momentum=...)` で更新します。

    1. momentum を 0（通常の勾配降下法），0.5，0.9 とした 3 つの条件のうち，50 回の更新後に極小点 $(0, 0)$ に最も近づくのはどれかを予想してください。
    2. `exercise/exc17_09.py` で，各条件について 50 回更新し，点 $(x_k, y_k)$ の列を記録してください。`plt.subplots(1, 4, figsize=(16, 4))` を使い，左の 3 つに各条件の軌跡を $f$ の等高線と重ねて描き，右端に原点からの距離 $\sqrt{x_k^2 + y_k^2}$ を 3 条件重ねて `semilogy` で描いてください。軌跡の図では，`ax.set_xlim(-5, 5)`，`ax.set_ylim(-3, 3)`，`ax.set_aspect("equal")` を指定してください。図は `outputs/exc17/exc17_09.png` に保存してください。
    3. 1 の予想と結果を比較してください。momentum 0 で $x$ 方向の進みが遅い理由を，第16回の発展第3問の結果を使って説明してください。また，momentum 0.5 で極小点に速く近づく理由と，momentum 0.9 で軌跡が極小点のまわりを回るように振動する理由を，解説の momentum の更新式を使って説明してください。

## 詰まったときに見る資料
- [`16-continuous-optimization.md`](16-continuous-optimization.md)
- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch: Automatic differentiation package](https://docs.pytorch.org/docs/stable/autograd.html)
- [PyTorch: torch.optim](https://docs.pytorch.org/docs/stable/optim.html)
