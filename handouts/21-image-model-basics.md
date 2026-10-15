# 第21回 画像のための CNN

## この回の目標

小さな CNN で画像分類を学習させ，入力から logits までの shape と予測結果を説明できる。

## 解説

この回では，第12回で扱った 2 次元フィルタを学習できる層として使う **畳み込みニューラルネットワーク (convolutional neural network, CNN)** を扱う。
第19回の多層パーセプトロンと同じく，層を重ねたモデルを第17回の学習ループで学習させ，第20回の `Dataset` と `DataLoader` でデータを与える。
スクリプトの先頭には，次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

figure_dir = Path("outputs/exc21")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. 畳み込み層

#### 2 次元フィルタと畳み込み層

第12回では，画像とインパルス応答の 2 次元畳み込みで，平均化やエッジの検出を行った。
このときのフィルタの係数は，目的に合わせて人が決めていた。
**畳み込み層 (convolutional layer)** は，同じ形の計算を行う層であり，フィルタの係数を学習によって決める。
畳み込み層のフィルタの係数を並べた配列を **カーネル (kernel)** という。

PyTorch の `nn.Conv2d` は，形 `(N, C_in, H, W)` の tensor を受け取り，形 `(N, C_out, H', W')` の tensor を返す。
$N$ はバッチの大きさ，$H$ と $W$ は画像の高さと幅，$C_\mathrm{in}$ と $C_\mathrm{out}$ は入力と出力の **チャネル (channel)** の数である。
第12回の RGB 画像は，3 チャネルの 2 次元信号であった。
畳み込み層の出力の各チャネルは，入力の各位置がある模様にどれだけ反応したかを表す 2 次元配列であり，**特徴マップ (feature map)** と呼ばれる。

出力の第 $o$ チャネルの位置 $(i, j)$ の値は，次式で計算される。

$$
y_o[i, j] = b_o + \sum_{c=0}^{C_\mathrm{in}-1} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} w_{o,c}[u, v]\, x_c[i+u, j+v]
$$

$K$ はカーネルの大きさ，$w_{o,c}$ は出力の第 $o$ チャネルと入力の第 $c$ チャネルの組に対するカーネル，$b_o$ はバイアスである。
つまり，入力のすべてのチャネルにそれぞれ 2 次元のフィルタを適用し，その和にバイアスを加えたものが，出力の 1 つのチャネルになる。

```python
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
x = torch.zeros(2, 3, 28, 28)
y = conv(x)
print(conv.weight.shape)  # torch.Size([8, 3, 3, 3])
print(conv.bias.shape)    # torch.Size([8])
print(y.shape)            # torch.Size([2, 8, 26, 26])
```

重み `conv.weight` の形は `(C_out, C_in, K, K)`，バイアス `conv.bias` の形は `(C_out,)` である。
この例では，出力の高さと幅が 28 から 26 に減っている。
カーネルが画像からはみ出さない位置だけで計算するため，$K=3$ では上下左右の端の 1 画素ずつについて出力が得られないからである。

#### パディング・ストライドと出力の大きさ

入力の周囲に $P$ 画素ずつ 0 を補ってから計算する操作を **パディング (padding)** という。
また，カーネルを $S$ 画素ずつずらしながら計算するとき，$S$ を **ストライド (stride)** という。
高さ $H$ の入力に対する出力の高さ $H'$ は，次式で求められる（幅も同様である）。

$$
H' = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1
$$

$\lfloor\cdot\rfloor$ は，小数点以下を切り捨てる記号である。
$K=3$，$P=1$，$S=1$ とすると $H'=H$ となり，入力と出力の大きさがそろう。
$S=2$ とすると，出力の高さと幅はおよそ半分になる。

```python
x = torch.zeros(2, 3, 28, 28)
for k, s, p in [(3, 1, 0), (3, 1, 1), (3, 2, 1), (5, 1, 2)]:
    layer = nn.Conv2d(3, 8, kernel_size=k, stride=s, padding=p)
    print(k, s, p, tuple(layer(x).shape))
```

```text
3 1 0 (2, 8, 26, 26)
3 1 1 (2, 8, 28, 28)
3 2 1 (2, 8, 14, 14)
5 1 2 (2, 8, 28, 28)
```

例えば，3 行目は $\lfloor (28 + 2 - 3)/2 \rfloor + 1 = 13 + 1 = 14$ である。
出力のチャネルの数 `8` は `out_channels` で決まり，パディングとストライドには影響されない。

#### パラメータの数

畳み込み層のパラメータの数は，重みの要素数とバイアスの要素数の和 $C_\mathrm{out} C_\mathrm{in} K^2 + C_\mathrm{out}$ であり，入力の画像の大きさには依存しない。
同じ大きさの入力と出力を全結合層（`nn.Linear`）で結ぶ場合と比べると，その差が分かる。

```python
conv = nn.Conv2d(3, 8, kernel_size=3)
fc = nn.Linear(3 * 28 * 28, 8 * 26 * 26)
print(sum(p.numel() for p in conv.parameters()))  # 224
print(sum(p.numel() for p in fc.parameters()))    # 12725024
```

全結合層では，出力の各値が入力のすべての画素と，それぞれ別の重みで結ばれる。
畳み込み層では，次の 2 つの性質によって，パラメータの数が大幅に少なくなる。

- **局所性 (locality)**：出力の各値は，入力の $K\times K$ の近傍の画素だけから計算される。
- **重み共有 (weight sharing)**：画像のすべての位置で，同じカーネルを使う。

重み共有のため，ある模様に反応するカーネルは，その模様が画像のどこにあっても同じように反応する。

#### 畳み込み層が計算する相互相関

「2 次元フィルタと畳み込み層」の式では，カーネルを反転せずに入力に重ねて積和をとっている。
この計算は，信号処理では **相互相関 (cross-correlation)** と呼ばれる。
第10回と第12回の畳み込みでは，$h[n-k]$ のようにカーネルを反転してから積和をとった。
そのため，同じ係数を使っても，`nn.Conv2d` の出力は第12回の畳み込みの出力と一般には一致しない。
次の例では，中央の画素だけが 1 である画像（単位インパルス信号）を入力する。

```python
conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
impulse = torch.zeros(1, 1, 5, 5)
impulse[0, 0, 2, 2] = 1.0
with torch.no_grad():
    conv.weight[:] = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    print(conv(impulse)[0, 0])
```

```text
tensor([[0., 0., 0., 0., 0.],
        [0., 9., 8., 7., 0.],
        [0., 6., 5., 4., 0.],
        [0., 3., 2., 1., 0.],
        [0., 0., 0., 0., 0.]])
```

カーネル `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` に対して，出力の中央には上下左右を反転したカーネルが現れる。
第10回で見たように，畳み込みではインパルスに対する出力がカーネル（インパルス応答）そのものになるので，この点が畳み込みとの違いである。
学習ではカーネルの係数そのものを決めるため，反転の有無は問題にならない。
反転したカーネルを学習すれば，畳み込みと同じ処理を表せるからである。
ただし，`scipy.signal.convolve2d` などで計算した結果と比べるときは，反転の違いに注意する。
なお，重みへの代入は勾配の計算に含めないため，`with torch.no_grad():` の中で行う（第17回）。

### 2. pooling と受容野

#### pooling

特徴マップを小さな領域に分け，各領域の値を 1 つにまとめる操作を **pooling** という。
`nn.MaxPool2d(2)` は，2×2 の領域ごとの最大値をとり（ストライドも 2 になる），高さと幅を半分にする。
2×2 の領域ごとの平均をとる `nn.AvgPool2d(2)` もある。
pooling は，学習するパラメータを持たない。
max pooling では，模様の位置が領域の中で少しずれても，出力が変わりにくい。

```python
pool = nn.MaxPool2d(kernel_size=2)
gap = nn.AdaptiveAvgPool2d(1)
x = torch.tensor([[1.0, 2.0, 5.0, 0.0],
                  [3.0, 4.0, 1.0, 1.0],
                  [0.0, 0.0, 2.0, 2.0],
                  [1.0, 0.0, 3.0, 1.0]]).reshape(1, 1, 4, 4)
print(gap(x).shape, gap(x).item())  # torch.Size([1, 1, 1, 1]) 1.625
print(pool(x))
```

```text
tensor([[[[4., 5.],
          [1., 3.]]]])
```

左上の 2×2 の領域 `[[1, 2], [3, 4]]` の最大値は 4 である。
`nn.AdaptiveAvgPool2d(1)` は，各チャネルの特徴マップ全体の平均をとり，高さと幅を 1 にする。
この操作を **global average pooling** という。
入力の高さと幅によらず，出力の形は `(N, C, 1, 1)` になる。

#### 受容野

出力のある 1 つの値に影響を与えうる入力の画素の範囲を，その値の **受容野 (receptive field)** という。
3×3 の畳み込み層を 1 つ通した出力の受容野は，3×3 の範囲である。
層を重ねるほど受容野は広がり，pooling やストライドで特徴マップを縮小した後の層では，さらに速く広がる。
$l$ 番目の層の出力の受容野の幅 $r_l$ は，その層のカーネルの大きさ $K_l$ とストライド $S_l$ を使って，次の漸化式で計算できる。

$$
r_l = r_{l-1} + (K_l - 1)\,j_{l-1},\qquad
j_l = j_{l-1}\,S_l,\qquad
r_0 = 1,\quad j_0 = 1
$$

$j_l$ は，$l$ 番目の層の出力で隣り合う 2 つの値が，入力の画像の上で何画素離れた位置に対応するかを表す。
pooling 層も，カーネルの大きさ 2，ストライド 2 の層として同じ式で扱える。
例えば，3×3 の畳み込み層（ストライド 1）と 2×2 の max pooling を交互に 2 回ずつ重ねると，次のようになる。

| 層 | $K_l$ | $S_l$ | $j_l$ | $r_l$ |
| --- | --- | --- | --- | --- |
| 入力 | | | 1 | 1 |
| 3×3 の畳み込み | 3 | 1 | 1 | 3 |
| 2×2 の max pooling | 2 | 2 | 2 | 4 |
| 3×3 の畳み込み | 3 | 1 | 2 | 8 |
| 2×2 の max pooling | 2 | 2 | 4 | 10 |

最後の出力の各値は，入力の 10×10 の範囲の画素から計算される。
模様を 1 つの値で捉えるには，受容野がその模様を覆う大きさである必要がある。
パディングは，受容野の大きさを変えない。

受容野は，出力の 1 つの値の入力に関する勾配から確かめられる。
勾配が 0 でない画素は，その値に影響を与える画素だからである。
次の例では，3×3 の畳み込み層を 2 つ重ね（$r_2 = 5$），すべての重みを 1，バイアスを 0 にして，影響が打ち消し合わないようにしている。

```python
stack = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=3, padding=1),
    nn.Conv2d(1, 1, kernel_size=3, padding=1),
)
for layer in stack:
    nn.init.ones_(layer.weight)
    nn.init.zeros_(layer.bias)
x = torch.ones(1, 1, 16, 16, requires_grad=True)
y = stack(x)
y[0, 0, 8, 8].backward()
rows = torch.nonzero(x.grad[0, 0])[:, 0]
print(rows.min().item(), rows.max().item())  # 6 10
```

`nn.init.ones_` と `nn.init.zeros_` は，引数の tensor のすべての要素を 1 と 0 にする。
`torch.nonzero` は，0 でない要素の位置（行と列の番号の組）を並べた tensor を返す。
勾配が 0 でない行は，行番号 6 から 10 までの 5 行であり，$r_2 = 5$ と一致する。

### 3. CNN による画像分類

#### データセット

この回では，次の `LineDataset` を使う。
各画像は形 `(1, 32, 32)` のグレースケール画像であり，平均 0，標準偏差 0.1 の正規分布に従う雑音の上に，長さ 9 画素の線分が 1 本描かれている。
線分の向きは，横 (horizontal)，縦 (vertical)，右下がりの斜め (diagonal)，右上がりの斜め (anti-diagonal) の 4 種類であり，向きの番号 0〜3 がラベルである。
線分の中心の位置は乱数で決めるため，線分は画像のさまざまな位置に現れる。

```python
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
CLASS_NAMES = ["horizontal", "vertical", "diagonal", "anti-diagonal"]


def draw_line(image, label, row, col, half_length):
    dr, dc = DIRECTIONS[label]
    for t in range(-half_length, half_length + 1):
        image[row + t * dr, col + t * dc] = 1.0


class LineDataset(Dataset):
    def __init__(self, n_per_class, seed, size=32, half_length=4, noise_std=0.1):
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.arange(4), n_per_class)
        images = rng.normal(0.0, noise_std, size=(len(labels), 1, size, size))
        for i, label in enumerate(labels):
            row = rng.integers(half_length, size - half_length)
            col = rng.integers(half_length, size - half_length)
            draw_line(images[i, 0], label, row, col, half_length)
        self.images = torch.from_numpy(images.astype(np.float32))
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


train_set = LineDataset(n_per_class=250, seed=0)
x, y = train_set[0]
print(len(train_set), x.shape, y)  # 1000 torch.Size([1, 32, 32]) tensor(0)
```

`draw_line` は，中心 `(row, col)` から `DIRECTIONS[label]` の向きに前後 `half_length` 画素ずつ，画素値を 1 にする。
`rng.integers(low, high)` は，`low` 以上 `high` 未満の整数を 1 つ返す（第6回）。
線分が画像からはみ出さないように，中心は `half_length` 以上 `size - half_length` 未満の範囲から選ぶ。
ラベルは `np.repeat(np.arange(4), n_per_class)` によってクラスの順に並ぶため，学習では `DataLoader` の `shuffle=True` で順番を並べ替える（第20回）。
画像は `float32`，ラベルは `int64` の tensor であり，第18回の `nn.CrossEntropyLoss` にそのまま渡せる。
`seed` が同じなら，同じ画像とラベルが作られる。

#### モデルの構成

改良を比べる基準として最初に作る，単純で確実に動くモデルを **baseline** という。
この回では，次の `SmallCNN` を `LineDataset` の分類の baseline とする。

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
```

`features` は，畳み込み層・ReLU・max pooling の組を 2 回重ねた部分であり，画像から特徴マップを計算する。
この部分の受容野は，「受容野」の表のとおり 10×10 であり，長さ 9 画素の線分を覆う。
`head` は，global average pooling で各チャネルを 1 つの値にまとめ，`nn.Flatten` で形 `(N, 16)` にしてから，`nn.Linear` で 4 クラスの logits を計算する。
第19回と同様に，最後の層の出力は softmax をかける前の logits とし，損失は `nn.CrossEntropyLoss` で計算する。
global average pooling は位置について平均をとるため，同じ向きの線分なら，画像のどこにあっても同じような特徴が得られる。

#### shape の追跡

層を順に通しながら出力の形を表示すると，どこで何が変わるかを確かめられる。
`nn.Sequential` は，`for` 文で中の層を順に取り出せる。

```python
model = SmallCNN()
x = torch.zeros(8, 1, 32, 32)
for layer in model.features:
    x = layer(x)
    print(type(layer).__name__, tuple(x.shape))
for layer in model.head:
    x = layer(x)
    print(type(layer).__name__, tuple(x.shape))
print(sum(p.numel() for p in model.parameters()))  # 1316
```

```text
Conv2d (8, 8, 32, 32)
ReLU (8, 8, 32, 32)
MaxPool2d (8, 8, 16, 16)
Conv2d (8, 16, 16, 16)
ReLU (8, 16, 16, 16)
MaxPool2d (8, 16, 8, 8)
AdaptiveAvgPool2d (8, 16, 1, 1)
Flatten (8, 16)
Linear (8, 4)
```

`type(layer).__name__` は，層のクラスの名前を返す。
畳み込み層はチャネルの数を変え，max pooling は高さと幅を半分にする。
global average pooling と `nn.Flatten` で縦横の情報がなくなり，1 枚の画像が 16 個の数の並び（特徴ベクトル）になる。

#### 学習と評価

学習と評価には，次の関数を使う。

```python
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)


def confusion_matrix(model, loader, num_classes=4):
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            for t, p in zip(y, pred):
                matrix[t, p] += 1
    return matrix
```

`train_one_epoch` は，第17回の学習ループを `DataLoader` の 1 エポック分だけ実行し，データ全体での平均損失を返す。
`loss.item() * len(y)` でバッチ内の損失の和に戻してから足し合わせ，最後にデータの数 `len(loader.dataset)` で割るため，最後のバッチの大きさが異なっても正しい平均になる（第20回）。
`evaluate` は，予測クラス `argmax(dim=1)` が正解と一致した割合（正解率）を返す。
評価の前には `model.eval()` を呼び，`torch.no_grad()` の中で計算する（第19回）。

**混同行列 (confusion matrix)** は，正解と予測の組ごとにサンプルの数を数えた表である（第13回の音声区間検出のプロジェクトでは，2 クラスの混同行列を使った）。
`confusion_matrix` が返す行列の $t$ 行 $p$ 列の要素は，正解がクラス $t$ で，予測がクラス $p$ だったサンプルの数である。
対角要素は正解したサンプルの数であり，対角以外の要素から，どのクラスをどのクラスと間違えたかが分かる。

### 4. residual block

#### 層を深くしたときの問題

層を深く重ねると，表せる処理は増えるが，学習が進みにくくなることがある。
第19回の誤差逆伝播法では，勾配は出力側から入力側へ，各層の微分を掛け合わせながら伝わる。
多くの層を通ると，この積が 0 に近づき，入力に近い層の勾配がほとんど 0 になることがある。
この現象を **勾配消失 (vanishing gradient)** という。
勾配がほとんど 0 の層のパラメータは，勾配降下法でほとんど更新されない。

#### skip connection と residual block

**ResNet** と呼ばれるモデルでは，いくつかの層の出力に，その入力をそのまま足す。
入力をそのまま出力側へ渡す経路を **skip connection** といい，次の形の処理のまとまりを **residual block** という。

$$
\boldsymbol{y} = \boldsymbol{x} + F(\boldsymbol{x})
$$

$F$ は，畳み込み層などを重ねた処理である。
$F(\boldsymbol{x})=\boldsymbol{0}$ なら $\boldsymbol{y}=\boldsymbol{x}$ となるため，residual block は，入力をどれだけ変えるかという変化量 $F(\boldsymbol{x})$ を学習するとみなせる。
1 変数で考えると，$y = x + F(x)$ の微分は $1 + F'(x)$ である。
$F'(x)$ が 0 に近くても微分は 0 に近づかないため，多くの block を重ねても，入力側の層まで勾配が伝わりやすい。

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(torch.relu(self.conv1(x)))


block = ResidualBlock(4)
x = torch.randn(2, 4, 16, 16)
print(block(x).shape)  # torch.Size([2, 4, 16, 16])
```

この例の $F$ は，畳み込み層，ReLU，畳み込み層の順に重ねた処理である。
`torch.relu` は，`nn.ReLU` と同じ計算をする関数である。
`x + ...` の足し算には，`x` と $F(\boldsymbol{x})$ の形が一致している必要がある。
この例では，パディング 1 の 3×3 の畳み込み層を使い，チャネルの数も変えないため，形が保たれる。

#### 形が変わる場合の shortcut

ストライド 2 で高さと幅を半分にしたり，チャネルの数を増やしたりする block では，$\boldsymbol{x}$ と $F(\boldsymbol{x})$ の形が一致しないため，そのままでは足せない。
この場合は，$\boldsymbol{x}$ の形を $F(\boldsymbol{x})$ の形にそろえる変換 $P$ を通してから足す。

$$
\boldsymbol{y} = P(\boldsymbol{x}) + F(\boldsymbol{x})
$$

$P$ には，カーネルの大きさが 1 の畳み込み層（1×1 の畳み込み）を使うことが多い。
1×1 の畳み込みは，各位置で入力のチャネルの値を並べたベクトルに同じ行列を掛け（第5回の行列とベクトルの積），バイアスを加える層であり，チャネルの数を変えられる。
ストライドを $F$ と同じにすれば，高さと幅もそろう。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第21回の第1問であれば，`exercise/exc21_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc21`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第1問の図であれば，`outputs/exc21/exc21_01.png` とします。
解説の `DIRECTIONS`，`CLASS_NAMES`，`draw_line`，`LineDataset`，`SmallCNN`，`train_one_epoch`，`evaluate`，`confusion_matrix` は，解説の冒頭の `import` 文とともに `exercise/line_cnn.py` に保存してください。
各問題のファイルでは，`from line_cnn import LineDataset, SmallCNN` のように読み込んで使ってください。
`python exercise/exc21_04.py` のように実行すると，実行したファイルと同じディレクトリにある `line_cnn.py` を読み込めます。
`exercise/line_cnn.py` は第23回でも使います。
乱数の指定がない場合は，モデルを作る直前に `torch.manual_seed(0)` を実行し，学習用の `DataLoader` には `shuffle=True` と `generator=torch.Generator().manual_seed(0)` を指定してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（6問）
1. 畳み込み層と pooling の出力の形とパラメータの数を，式から求められるようにします。入力を `x = torch.zeros(2, 3, 32, 32)` とし，次の 5 つの層を考えます。

    - (a) `nn.Conv2d(3, 16, kernel_size=3)`
    - (b) `nn.Conv2d(3, 16, kernel_size=3, padding=1)`
    - (c) `nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)`
    - (d) `nn.Conv2d(3, 16, kernel_size=1)`
    - (e) `nn.MaxPool2d(2)`

    1. (a)〜(e) のそれぞれについて，`x` を入力したときの出力の形と，学習するパラメータの数を手計算してください。
    2. `exercise/exc21_01.py` で各層に `x` を入力し，出力の `shape` と `sum(p.numel() for p in layer.parameters())` を表示して，1 の結果と比較してください。
    3. (b) と同じ大きさの入力と出力を結ぶ全結合層 `nn.Linear(3 * 32 * 32, 16 * 32 * 32)` のパラメータの数を手計算してください。この層は大きいため，プログラムで作る必要はありません。
    4. 3 の結果を (b) のパラメータの数と比べてください。畳み込み層のパラメータの数がはるかに少ない理由を，解説の局所性と重み共有を使って説明してください。

2. `nn.Conv2d` が相互相関を計算することを，第12回で使った SciPy の関数と比べて確かめます。入力を `x = torch.arange(25.0).reshape(1, 1, 5, 5)` とします。`x[0, 0]` の $i$ 行 $j$ 列（0 から数える）の値は $5i+j$ です。カーネルを次の $\boldsymbol{k}$ とします。

    $$
    \boldsymbol{k} = \begin{pmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{pmatrix}
    $$

    1. `nn.Conv2d(1, 1, kernel_size=3, bias=False)` の重みを $\boldsymbol{k}$ にして `x` を入力したときの，出力の形と，出力の左上の値を手計算してください。
    2. `exercise/exc21_02.py` で，解説の例と同様に `torch.no_grad()` の中で重みに $\boldsymbol{k}$ を代入し，出力を表示して 1 の結果と比較してください。
    3. 同じ入力とカーネルを 2 次元の NumPy 配列にして（例：`x[0, 0].numpy()`），`scipy.signal.correlate2d` と `scipy.signal.convolve2d` に `mode="valid"` を指定して計算してください。どちらが `nn.Conv2d` の出力と一致するかを確かめてください。
    4. 出力のすべての値が同じになる理由を，`x` の値が右へ 1 画素進むごとに 1 増えることと，$\boldsymbol{k}$ の各行の形から説明してください。また，`convolve2d` の結果の符号が `nn.Conv2d` の出力と逆になる理由を，カーネルの反転を使って説明してください。
    5. カーネルを `torch.ones(3, 3) / 9` に変えて，3 と同じ比較を行ってください。浮動小数点数の比較には `np.allclose` を使ってください。このカーネルでは相互相関と畳み込みの結果が一致する理由を説明してください。

3. 層を重ねたときの受容野を，式と勾配の両方から求めます。次の 6 つの層をこの順に重ねたモデルを考えます。畳み込み層はすべて `nn.Conv2d(1, 1, kernel_size=3, padding=1)` とします。

    畳み込み → 畳み込み → `nn.AvgPool2d(2)` → 畳み込み → `nn.AvgPool2d(2)` → 畳み込み

    1. 解説の漸化式を使い，各層の出力の $j_l$ と $r_l$ を手計算して表にしてください。
    2. `exercise/exc21_03.py` で，このモデルを `nn.Sequential` で作ってください。解説の例と同様に，すべての畳み込み層の重みを 1 に，バイアスを 0 にしてください。pooling 層は重みを持たないため，各層が畳み込み層かどうかを `isinstance(layer, nn.Conv2d)` で判定してください。
    3. `x = torch.ones(1, 1, 64, 64, requires_grad=True)` を入力し，出力 `y` の形を確認してください。そのうえで，出力の中央付近の値 `y[0, 0, 8, 8]` に対して `backward()` を呼んでください。
    4. `x.grad[0, 0]` で 0 でない要素が並ぶ範囲の行数と列数を求め，1 で求めた最後の層の $r_l$ と比較してください。また，`x.grad[0, 0]` を `imshow` で表示し，`outputs/exc21/exc21_03.png` に保存してください。
    5. 2 つの `nn.AvgPool2d` を `nn.MaxPool2d` に変えると，4 の範囲がどう変わるかを予想してから確かめてください。max pooling では，勾配が各領域で最大値をとった 1 つの位置だけに伝わることと，この入力ではすべての値が等しいことから，結果を説明してください。

4. `LineDataset` が作る画像とラベル，`DataLoader` が作るバッチを確かめます。`train_set = LineDataset(n_per_class=250, seed=0)` と `test_set = LineDataset(n_per_class=50, seed=1)` を使います。

    1. `train_set` と `test_set` の要素数，`train_set[0]` の画像の形と dtype，ラベルの値と dtype を予想してください。`exercise/exc21_04.py` で表示して確かめてください。
    2. 各クラスから 2 枚ずつ，計 8 枚の画像を `plt.subplots(2, 4)` で並べてください。各画像のタイトルにはクラス名 `CLASS_NAMES[label]` を付け，`outputs/exc21/exc21_04.png` に保存してください。クラス $c$ の画像は，`train_set` の `250 * c` 番目から並んでいます。`imshow` には形 `(32, 32)` の配列（例：`image[0]`）を渡し，`cmap="gray"` を指定してください。
    3. `loader = DataLoader(train_set, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))` について，`len(loader)` の値を予想してから確かめてください。また，最初のバッチの画像とラベルの `shape` を表示してください。
    4. 最初のバッチに含まれる各クラスの枚数を `torch.bincount(labels, minlength=4)` で数えてください。`shuffle=False` にすると最初のバッチのラベルがどうなるかを予想してから確かめ，学習で `shuffle=True` を指定する理由を説明してください。

5. `SmallCNN` で `LineDataset` の 4 クラス分類を学習させ，損失・正解率・混同行列で結果を評価します。基礎第4問の `train_set` と `test_set` を使い，損失関数を `nn.CrossEntropyLoss()`，optimizer を `torch.optim.Adam(model.parameters(), lr=1e-2)`，バッチサイズを 32 とします。

    1. 4 クラスの logits がすべて等しいときの cross-entropy を手計算してください（第18回）。`exercise/exc21_05.py` で，学習前のモデルの `train_set` 全体に対する損失を `torch.no_grad()` の中で計算し，手計算の値と比較してください。全体の画像とラベルは，`train_set.images` と `train_set.labels` で取り出せます。
    2. `SmallCNN` を 5 エポック学習させてください。各エポックについて，`train_one_epoch` が返す平均損失と，`test_set` に対する正解率を表示してください。1 エポックの学習にかかった時間も `time.perf_counter()` で測って表示してください。
    3. `plt.subplots(1, 2)` を使い，左にエポックごとの平均損失を，右にエポックごとの正解率を描いて，`outputs/exc21/exc21_05.png` に保存してください。
    4. 第2エポックの終了時と第5エポックの終了時に，`test_set` に対する混同行列を表示してください。第2エポックの時点で，どのクラスの画像がどのクラスと予測されていたかを答えてください。また，正解率だけでは分からず，混同行列から分かることを説明してください。
    5. `test_set[0]`，`test_set[50]`，`test_set[100]`，`test_set[150]` の 4 枚について予測クラスを求め，画像を並べてタイトルに予測クラスと正解クラスを表示した図を `outputs/exc21/exc21_05_predictions.png` に保存してください。1 枚の画像を入力するときは，`image.unsqueeze(0)` で形を `(1, 1, 32, 32)` にしてください（第17回）。

6. skip connection の有無による違いを，同じ入力に対する出力で確かめます。解説の `ResidualBlock` から足し算 `x +` を除き，`self.conv2(torch.relu(self.conv1(x)))` を返す `PlainBlock` を作ってください。

    1. `PlainBlock(8)` と `ResidualBlock(8)` のパラメータの数を手計算してください。
    2. `exercise/exc21_06.py` で，`torch.manual_seed(0)` の直後に `x = torch.randn(2, 8, 32, 32)` を作り，続けて `PlainBlock(8)` と `ResidualBlock(8)` を作ってください。`x` を 2 つの block に入力し，出力の形とパラメータの数を表示して，1 の結果と比較してください。
    3. 各 block の出力 `out` について，入力との差の平均 `(out - x).abs().mean()` を表示してください。`x.abs().mean()` も表示してください。2 つの block の値の大小を予想してから確かめ，理由を `x + F(x)` の形から説明してください。
    4. `nn.init.zeros_` で `ResidualBlock` の `conv2` の重みとバイアスを 0 にしてから `x` を入力し，出力が `x` と完全に一致することを `torch.equal` で確かめてください。`PlainBlock` の `conv2` を同様に 0 にした場合の出力も表示してください。この結果から，residual block が入力からの変化量を学習するとみなせる理由を説明してください。

### 発展レベル（3問）
1. 全結合層だけのモデルと CNN を，学習時に現れなかった位置の線分で比べます。

    1. `exercise/line_cnn.py` の `LineDataset` に，線分の中心の列番号の範囲を指定する引数 `col_range=None` を追加してください。`col_range=(low, high)` のときは列番号を `rng.integers(low, high)` で決め，`None` のときは解説と同じ範囲から決めます。行番号は，これまでどおり列番号より先に決めてください。`col_range=None` のときに変更前と同じ画像が作られることを，`torch.equal` で確かめてください。
    2. `exercise/exc21_07.py` で，中心が画像の左半分にある `train_set = LineDataset(250, seed=0, col_range=(4, 16))` と `left_test = LineDataset(50, seed=1, col_range=(4, 16))`，右半分にある `right_test = LineDataset(50, seed=2, col_range=(16, 28))` を作ってください。
    3. 全結合層だけのモデル `nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 64), nn.ReLU(), nn.Linear(64, 4))` のパラメータの数を手計算し，プログラムで確かめてください。また，`SmallCNN` のパラメータの数と比べてください。
    4. 2 つのモデルを，基礎第5問と同じ条件で `train_set` を使って 5 エポック学習させてください。`train_set`，`left_test`，`right_test` に対する正解率を表にして表示してください。実行前に，各モデルの `right_test` に対する正解率を予想してください。
    5. 全結合層だけのモデルの `right_test` に対する正解率が低い理由を，そのモデルの重みが画素の位置ごとに異なることから説明してください。また，`SmallCNN` では学習時と異なる位置の線分も正しく分類できる理由を，重み共有と global average pooling を使って説明してください。

2. block を深く重ねたときの勾配の大きさを，skip connection の有無で比べます。

    1. `exercise/exc21_08.py` で，`depth` を 1，5，10，20 とし，基礎第6問の `PlainBlock(8)` を `depth` 個重ねたモデル `nn.Sequential(*[PlainBlock(8) for _ in range(depth)])` を作ってください。`*` は，リストの要素を別々の引数として渡す記法です。各 `depth` について，`torch.manual_seed(0)` の直後にモデルを作り，続けて `x = torch.randn(4, 8, 16, 16)` を作ってください。
    2. 出力 `out` から損失 `out.pow(2).mean()` を計算して `backward()` を呼び，入力に最も近い畳み込み層の重みの勾配のノルム `model[0].conv1.weight.grad.norm()` を表示してください。`ResidualBlock(8)` を重ねたモデルについても，同じ計算をしてください。
    3. 横軸を `depth`，縦軸を勾配のノルムとした片対数グラフ（`ax.semilogy`）に 2 種類の block の結果を描き，`outputs/exc21/exc21_08.png` に保存してください。
    4. `PlainBlock` を重ねると勾配のノルムがどう変化するかを，誤差逆伝播法で各層の微分が掛け合わされること（第19回）と結びつけて説明してください。また，`ResidualBlock` では勾配が小さくなりにくい理由を，解説の $1 + F'(x)$ を使って説明してください。

3. チャネルの数と解像度が変わる residual block を実装します。入力の形を `(2, 8, 32, 32)`，出力の形を `(2, 16, 16, 16)` とし，$F$ を `nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)`，ReLU，`nn.Conv2d(16, 16, kernel_size=3, padding=1)` の順に重ねた処理とします。

    1. `exercise/exc21_09.py` で，`torch.manual_seed(0)` の直後に `x = torch.randn(2, 8, 32, 32)` を作り，$F$ を `nn.Sequential` で作って `F(x)` の形を表示してください。`x + F(x)` を計算すると何が起きるかを予想してから確かめてください。エラーになる処理は，他の処理と分けて実行してください。
    2. `self.conv1`，`self.conv2`，`self.shortcut` を属性に持ち，`self.shortcut(x) + self.conv2(torch.relu(self.conv1(x)))` を返すクラス `DownBlock(in_channels, out_channels)` を，`nn.Module` を継承して実装してください。解説の $P$ にあたる `self.shortcut` は `nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)` とします。`DownBlock(8, 16)` の出力の形が `(2, 16, 16, 16)` になることを確かめてください。
    3. `DownBlock(8, 16)` のパラメータの数を，`conv1`，`conv2`，`shortcut` に分けて手計算し，プログラムで確かめてください。
    4. `P = block.shortcut` の重みの形を表示してください。`W = P.weight[:, :, 0, 0]`，`b = P.bias` として，`P(x)[0, :, 3, 5]` と `W @ x[0, :, 6, 10] + b` が一致することを `torch.allclose` で確かめてください。この結果を使って，1×1 の畳み込みが各位置で行う計算を，第5回の行列とベクトルの積で説明してください。また，出力の位置 `(3, 5)` が入力の位置 `(6, 10)` に対応する理由を，ストライドから説明してください。

## 詰まったときに見る資料
- [`12-image-signals.md`](12-image-signals.md)
- [`../textbook/markdown/ch23-basics-of-neural-networks.md`](../textbook/markdown/ch23-basics-of-neural-networks.md)
- [PyTorch: torch.nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch: torch.nn.MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
