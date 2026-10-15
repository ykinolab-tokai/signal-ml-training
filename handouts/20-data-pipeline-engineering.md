# 第20回 データパイプライン

## この回の目標

Dataset と DataLoader によるデータ処理を実装し，augmentation が入力とラベルに与える影響を説明できる。

## 解説

この回では，第19回で扱ったモデルの学習に，PyTorch でデータを扱う仕組みを組み合わせる。
データを 1 サンプルずつ返す Dataset と，サンプルをバッチにまとめる DataLoader を使うと，データの読み込み，前処理，augmentation を学習ループから分けて書ける。
スクリプトの先頭には，第19回の処理に Dataset と DataLoader の読み込みを加えた次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from pathlib import Path

figure_dir = Path("outputs/exc20")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. Dataset

#### データパイプライン

第19回では，すべてのデータを 1 つの tensor にまとめ，各更新でそのすべてを使った。
実際の学習では，データが多くて一度にメモリに載らない，ファイルを 1 つずつ読み込む，サンプルごとに異なる変換を加える，などの理由で，データを少しずつ取り出して使う。
データを読み込み，前処理を施し，バッチにまとめてモデルに渡すまでの一連の処理を **データパイプライン (data pipeline)** という。
PyTorch では，1 サンプルを返す Dataset と，複数のサンプルをバッチにまとめる DataLoader に役割を分けて書く。

#### Dataset の定義

**Dataset** は，番号を指定すると，その番号のサンプル（入力と正解の組）を返すオブジェクトである。
PyTorch では，`torch.utils.data.Dataset` を継承したクラス（第19回）を定義し，次の 2 つのメソッドを書く。

- `__len__(self)`：サンプル数を返す。`len(dataset)` と書くと呼ばれる。
- `__getitem__(self, index)`：番号 `index` のサンプルを返す。`dataset[index]` と書くと呼ばれる。

`__init__` と同様に，名前の前後に `__` が付いたメソッドは，`len(...)` や `[...]` などの Python の構文から自動的に呼ばれる特別なメソッドである。
次の例は，入力 $x=0, 1, \ldots, n-1$ と正解 $y=x^2$ の組を返す Dataset である。

```python
class SquareDataset(Dataset):
    def __init__(self, n):
        self.x = torch.arange(n, dtype=torch.float32).reshape(-1, 1)
        self.y = self.x**2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

dataset = SquareDataset(5)
print(len(dataset))  # 5
x, y = dataset[3]
print(x, y)          # tensor([3.]) tensor([9.])
print(x.shape)       # torch.Size([1])
```

`dataset[3]` は入力と正解の組（タプル）を返し，`x, y = dataset[3]` で 2 つの変数に分けて受け取れる。
1 サンプルの入力の shape は `(1,)` であり，サンプルを並べる軸を含まない。
入力と正解の tensor がすでにある場合は，`TensorDataset(x, y)` を使うと，同じ働きの Dataset をクラスを定義せずに作れる。

### 2. DataLoader

#### バッチの作成

**DataLoader** は，Dataset からサンプルを取り出し，指定した数ずつまとめたバッチを順に返す。

```python
loader = DataLoader(dataset, batch_size=2, shuffle=False)
print(len(loader))  # 3
for x_batch, y_batch in loader:
    print(x_batch.shape, y_batch.shape)
```

```text
torch.Size([2, 1]) torch.Size([2, 1])
torch.Size([2, 1]) torch.Size([2, 1])
torch.Size([1, 1]) torch.Size([1, 1])
```

`for x_batch, y_batch in loader:` と書くと，反復ごとに入力と正解のバッチを受け取る。
バッチの shape は，1 サンプルの shape の先頭にバッチの軸を加えた `(バッチサイズ, 1)` である。
サンプル数 $N$ がバッチサイズ $B$ で割り切れない場合，最後のバッチは $B$ より小さくなり，バッチの数は $N/B$ 以上の最小の整数 $\lceil N/B\rceil$ となる。
`len(loader)` はバッチの数を返す。
`drop_last=True` を指定すると，大きさが $B$ に満たない最後のバッチを使わない。

#### shuffle と乱数

`shuffle=True` を指定すると，エポックごとにサンプルの順番を並べ替えてからバッチにまとめる。
第16回の確率的勾配降下法で，エポックの最初にデータを並べ替えたのと同じ処理である。
並べ替えに使う乱数は，引数 `generator` に `torch.Generator().manual_seed(シード)` を渡すと固定できる。

```python
generator = torch.Generator().manual_seed(0)
loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
for epoch in range(2):
    order = [x_batch.flatten().tolist() for x_batch, _ in loader]
    print(epoch, order)
```

```text
0 [[3.0, 1.0], [0.0, 4.0], [2.0]]
1 [[1.0, 4.0], [3.0, 2.0], [0.0]]
```

`x_batch.flatten().tolist()` は，tensor を 1 次元にしてから Python のリストに変換する。
`_` は，受け取った値を使わないことを示す慣例的な変数名である。
1 つのエポックの中では，すべてのサンプルがちょうど 1 回ずつ現れる。
エポックごとに順番は変わるが，同じシードで作り直した DataLoader は，同じ順番を再現する。
DataLoader の引数 `num_workers` に 1 以上を指定すると，複数のプロセスで並行してサンプルを読み込めるが，乱数の扱いが変わり，実行環境によっては追加の設定が必要になる。
この資料では，既定値の 0 のまま使う。

### 3. collate_fn

#### 既定の collate 関数

DataLoader は，バッチに含めるサンプルのリストを **collate 関数** に渡し，その戻り値をバッチとする。
既定の collate 関数は，サンプルの要素（この例では入力と正解）ごとに，各サンプルの tensor を `torch.stack` で新しい先頭の軸に沿って積み重ねる。
Python の整数や浮動小数点数は，並べて 1 つの tensor にする。
このため，既定の collate 関数を使うには，すべてのサンプルの tensor の shape が等しい必要がある。

#### 長さの異なるデータ

音声の録音時間のように，サンプルごとに長さが異なるデータは，そのままでは積み重ねられない。
次の例では，長さ 2，4，3 の系列（値はそれぞれの長さ）と，その長さを返す Dataset を使う。

```python
class SequenceDataset(Dataset):
    def __init__(self, lengths):
        self.sequences = [torch.ones(n) * n for n in lengths]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], len(self.sequences[index])

seq_dataset = SequenceDataset([2, 4, 3])
loader = DataLoader(seq_dataset, batch_size=3)
```

`next(iter(loader))` と書くと，DataLoader から最初のバッチだけを取り出せる。
この `loader` で最初のバッチを取り出そうとすると，次のエラーが発生する。

```text
RuntimeError: stack expects each tensor to be equal size, but got [2] at entry 0 and [4] at entry 1
```

既定の collate 関数は，shape `(2,)` と `(4,)` の tensor を積み重ねられないためである。
このような場合は，バッチのまとめ方を書いた関数を `collate_fn` に渡す。

```python
def pad_collate(batch):
    sequences = [sequence for sequence, _ in batch]
    lengths = torch.tensor([length for _, length in batch])
    padded = torch.zeros(len(sequences), int(lengths.max()))
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence)] = sequence
    return padded, lengths

loader = DataLoader(seq_dataset, batch_size=3, collate_fn=pad_collate)
padded, lengths = next(iter(loader))
print(padded)
print(lengths)  # tensor([2, 4, 3])
```

```text
tensor([[2., 2., 0., 0.],
        [4., 4., 4., 4.],
        [3., 3., 3., 0.]])
```

`pad_collate` の引数 `batch` は，`seq_dataset[i]` が返す組を並べたリストである。
この関数は，バッチの中で最も長い系列に合わせて，短い系列の後ろに 0 を詰める。
このように長さをそろえるために値を詰めることを **パディング (padding)** という。
`enumerate(sequences)` は，番号 `i` と要素 `sequence` の組を順に返す。
パディングした要素は元のデータにないので，後で平均などを計算するときは，`lengths` を使ってこれらの要素を除く必要がある。

### 4. データの分割と前処理

#### Subset による分割

`Subset(dataset, indices)` は，元の Dataset のうち，`indices` で指定した番号のサンプルだけを返す Dataset を作る。
第6回で扱った `rng.permutation` で並べ替えた番号を 2 つに分ければ，train と validation に分割できる。

```python
dataset = SquareDataset(10)
rng = np.random.default_rng(0)
indices = rng.permutation(len(dataset))
print(indices)                      # [4 6 2 7 3 5 9 0 8 1]
train_set = Subset(dataset, indices[:7])
val_set = Subset(dataset, indices[7:])
print(len(train_set), len(val_set))  # 7 3
print(train_set[0])                 # (tensor([4.]), tensor([16.]))
```

`train_set[0]` は，元の Dataset の `dataset[indices[0]]`，つまり `dataset[4]` を返す。
Subset はサンプルを複製せず，番号の対応だけを保持する。
`torch.utils.data.random_split(dataset, [7, 3], generator=...)` でも，同様の分割ができる。

#### 前処理の統計量

第6回で扱った標準化のように，データから計算した統計量を使う前処理では，統計量を train のデータだけから計算し，validation と test にも同じ値を使う。
validation や test のデータから統計量を計算すると，評価に使うはずのデータの情報が学習の側に入り込み，評価が実際より良く見えるおそれがある。
これを **データリーク (data leakage)** という。

```python
x_train = torch.stack([x for x, _ in train_set])
print(x_train.shape)  # torch.Size([7, 1])
mean = x_train.mean(dim=0)
std = x_train.std(dim=0, correction=0)
print(mean, std)      # tensor([5.1429]) tensor([2.2315])

class StandardizedDataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return (x - self.mean) / self.std, y

train_scaled = StandardizedDataset(train_set, mean, std)
val_scaled = StandardizedDataset(val_set, mean, std)
x_val = torch.stack([x for x, _ in val_scaled])
print(x_val.flatten())  # tensor([-2.3047,  1.2804, -1.8565])
```

- `torch.stack([x for x, _ in train_set])`：train の入力をすべて取り出し，新しい先頭の軸に沿って積み重ねる。
- `x_train.std(dim=0, correction=0)`：$N$ で割る標準偏差（第6回の `ddof=0`）を計算する。PyTorch の `std` は既定で $N-1$ で割る（`ddof=1` に相当する）ので，NumPy の既定と同じ値にするには `correction=0` を指定する。
- `StandardizedDataset`：別の Dataset を受け取り，そのサンプルの入力に前処理を施して返す Dataset である。このように Dataset を包むと，前処理を元のデータの読み込みと分けて書ける。

標準化後の train の入力は，平均 0，標準偏差 1 になる。
一方，validation の入力の平均は 0 にならない（この例では約 $-0.96$）。
統計量を train から計算しているので，これは正しい結果である。

### 5. augmentation

#### ラベルを保つ変換

学習データの入力に，正解ラベルが変わらない範囲の変換を加えて，学習に使う入力の種類を増やす方法を **データ拡張 (data augmentation)** という。
画像では，左右反転，平行移動，ノイズの付加などがよく使われる。
ただし，変換がラベルを保つかどうかは，タスクの定義で決まる。
例えば，数字の画像の分類で上下反転を使うと，6 と 9 のようにラベルが変わってしまうことがある。

PyTorch では，画像の tensor の shape を `(チャネル, 高さ, 幅)` とし，軸 0 をチャネルとする。
第12回で扱った Pillow から得た配列の shape `(高さ, 幅, チャネル)` とは，軸の順番が異なる。

```python
image = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
print(torch.flip(image, dims=[2]))
print(torch.roll(image, shifts=1, dims=2))
print(torch.rot90(image, k=1, dims=[1, 2]))
```

```text
tensor([[[2., 1., 0.],
         [5., 4., 3.],
         [8., 7., 6.]]])
tensor([[[2., 0., 1.],
         [5., 3., 4.],
         [8., 6., 7.]]])
tensor([[[2., 5., 8.],
         [1., 4., 7.],
         [0., 3., 6.]]])
```

- `torch.flip(image, dims=[2])`：軸 2（横方向）の順番を逆にする（左右反転）。
- `torch.roll(image, shifts=1, dims=2)`：横方向に 1 画素ずらし，右端からはみ出した列を左端に戻す。このようにずらすことを **巡回シフト** という。
- `torch.rot90(image, k=1, dims=[1, 2])`：軸 1 と軸 2 がなす面の中で，画像を反時計回りに 90° 回転する。

いずれの変換も，画像の shape を変えない。

#### 学習時だけに使うランダムな変換

augmentation では，`__getitem__` が呼ばれるたびに，乱数で変換の量を決める。
このため，同じサンプルでも，エポックごとに異なる入力が学習に使われる。
validation と test では，実際のデータに対する性能を測るため，augmentation を使わない。

```python
class RandomShiftDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        shift = int(torch.randint(0, x.shape[-1], size=(1,)))
        return torch.roll(x, shifts=shift, dims=-1), y

torch.manual_seed(0)
shifted = RandomShiftDataset(TensorDataset(image.unsqueeze(0), torch.tensor([0])))
print(shifted[0][0])
print(shifted[0][0])
```

```text
tensor([[[1., 2., 0.],
         [4., 5., 3.],
         [7., 8., 6.]]])
tensor([[[0., 1., 2.],
         [3., 4., 5.],
         [6., 7., 8.]]])
```

`torch.randint(0, x.shape[-1], size=(1,))` は，0 以上 `x.shape[-1]` 未満の整数の乱数を 1 つ含む tensor を返す。
この乱数は `torch.manual_seed` で固定される PyTorch の乱数であり，DataLoader の `generator` とは別のものである。
同じ `shifted[0]` でも，呼ぶたびにずらす量が変わる（この例では 2 画素と 0 画素）。
`TensorDataset(image.unsqueeze(0), torch.tensor([0]))` は，`image` だけを含む 1 サンプルの Dataset である。

### 6. 学習ループへの組み込み

DataLoader を使う学習ループでは，エポックごとにすべてのバッチを順に取り出し，バッチごとにパラメータを更新する。

```python
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, size=(100, 1))
y = 3 * x + 1 + rng.normal(0, 0.1, size=(100, 1))
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
train_set = TensorDataset(x[:80], y[:80])
val_set = TensorDataset(x[80:], y[80:])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                          generator=torch.Generator().manual_seed(0))
val_loader = DataLoader(val_set, batch_size=16)
loss_fn = nn.MSELoss()

def evaluate(model, loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            total += loss_fn(model(x_batch), y_batch).item() * len(x_batch)
    model.train()
    return total / len(loader.dataset)

torch.manual_seed(0)
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(30):
    total = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x_batch)
    train_loss = total / len(train_set)

print(len(train_loader))                         # 5
print(train_loss, evaluate(model, val_loader))  # 約 0.0093 約 0.0102
print(model.weight.item(), model.bias.item())   # 約 2.98 約 1.00
```

- 1 エポックの更新回数はバッチの数 `len(train_loader)` に等しく，この例では $80/16=5$ 回である。
- `loss.item()` はバッチ内の平均の損失である。最後のバッチが小さい場合も正しく平均するため，バッチの大きさ `len(x_batch)` を掛けて足し合わせ，最後にサンプル数で割って，そのエポックの train の損失とする。
- `evaluate` は，第19回と同様に `model.eval()` と `torch.no_grad()` を使い，validation の損失を同じ方法で平均する。`loader.dataset` は，DataLoader に渡した Dataset である。
- 学習後の重みとバイアスは，データを作ったときの 3 と 1 に近い。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第20回の第1問であれば，`exercise/exc20_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc20`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第3問の図であれば，`outputs/exc20/exc20_03.png` とします。
他の問題で定義したクラスや関数を使う場合は，その定義を同じファイルにも書いてください。
モデルを作る直前には `torch.manual_seed(0)` を実行してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（6問）
1. Dataset の 2 つのメソッドの役割を確かめます。次の配列を使います。

    ```python
    features = np.arange(30).reshape(10, 3)
    labels = np.arange(10) % 2
    ```

    1. `exercise/exc20_01.py` で，`features` と `labels` を受け取り，それぞれを `torch.float32` と `torch.int64` の tensor として属性に保持する Dataset `ArrayDataset(features, labels)` を定義してください。`__getitem__` は，番号 `index` の入力と正解の組を返すようにします。
    2. `dataset = ArrayDataset(features, labels)` について，`len(dataset)` と `dataset[3]` の値，`dataset[3]` の入力と正解の shape を予想してから表示してください。
    3. `dataset[-1]` が返す値を予想してから表示してください。また，`dataset[10]` を実行すると何が起きるかを確かめてください（他の処理とは分けて実行してください）。
    4. `TensorDataset` で同じデータの Dataset を作り，`dataset[3]` と同じ値が返ることを確かめてください。最後に，`len(dataset)` と `dataset[3]` と書いたときに，`ArrayDataset` のどのメソッドがどの引数で呼ばれているかを説明してください。

2. DataLoader が作るバッチの数と shape，shuffle による順番を確かめます。基礎第1問の `dataset` を使います。

    1. `batch_size=4`，`shuffle=False` の DataLoader について，バッチの数と，各バッチの入力と正解の shape を予想してください。また，`drop_last=True` も指定した場合のバッチの数を予想してください。
    2. `exercise/exc20_02.py` で 2 つの DataLoader を作り，`len(loader)` と各バッチの shape を表示して，1 の予想と比較してください。
    3. `batch_size=4`，`shuffle=True`，`generator=torch.Generator().manual_seed(0)` の DataLoader で 2 エポック分のバッチを取り出し，各エポックで取り出されたサンプルの番号を順に表示してください。サンプルの番号は，入力の先頭の列を 3 で割ると求められます。各エポックの番号を小さい順に並べ替えると 0 から 9 になることを確かめてください。
    4. 同じシードの generator で DataLoader を作り直し，最初のエポックの順番が，3 の 1 エポック目と一致することを確かめてください。3 で 2 エポック目の順番が 1 エポック目と異なった理由と，作り直すと同じ順番が再現される理由を説明してください。

3. 縦縞と横縞の画像を返す Dataset を作ります。次の関数 `make_stripe` は，ラベル `label`（0 なら縦縞，1 なら横縞），周期 `period`，位相 `phase` を受け取り，値が 0 または 1 の shape `(1, 16, 16)` の画像を返します。

    ```python
    def make_stripe(label, period, phase, size=16):
        line = ((torch.arange(size) + phase) % period < period // 2).float()
        if label == 0:
            image = line.repeat(size, 1)
        else:
            image = line.reshape(-1, 1).repeat(1, size)
        return image.unsqueeze(0)
    ```

    1. `make_stripe(0, 4, 0)` の 1 行目（`image[0, 0]`）の 16 個の値を手計算してください。また，`make_stripe(1, 4, 0)` の 1 列目（`image[0, :, 0]`）の値を手計算し，1 行目（`image[0, 0]`）の値がどうなるかを答えてください。`line.repeat(size, 1)` は，`line` を縦に `size` 個並べた shape `(size, size)` の tensor を返します。
    2. `exercise/exc20_03.py` で，次の `__init__` を持つ Dataset `StripeDataset` を定義してください。`__getitem__` は，番号 `index` のラベル，周期，位相から `make_stripe` で作った画像と，Python の整数に変換したラベル `int(self.labels[index])` の組を返すようにします。

        ```python
        class StripeDataset(Dataset):
            def __init__(self, n, seed, random_phase=True):
                rng = np.random.default_rng(seed)
                self.labels = rng.integers(0, 2, size=n)
                self.periods = rng.choice([4, 6, 8], size=n)
                if random_phase:
                    self.phases = rng.integers(0, 8, size=n)
                else:
                    self.phases = np.zeros(n, dtype=np.int64)
        ```

    3. `StripeDataset(8, seed=0)` の 8 枚の画像を `plt.subplots(2, 4)` で並べ，各画像のタイトルにラベルと周期を書いて，`outputs/exc20/exc20_03.png` に保存してください。画像は `ax.imshow(image[0], cmap="gray")` で描けます。図を見て，ラベル 0 と 1 がそれぞれ縦縞と横縞になっていることを確かめてください。
    4. `batch_size=4` の DataLoader から取り出したバッチの入力と正解の shape と dtype を予想してから表示してください。入力の shape の 4 つの軸がそれぞれ何を表すかを説明してください。また，`__getitem__` で返した Python の整数が，バッチではどのような tensor になったかを答えてください。

4. augmentation が，縦縞と横縞のラベルを保つかを確かめます。基礎第3問の `make_stripe(0, 6, 1)`（縦縞）と `make_stripe(1, 6, 1)`（横縞）を使います。

    1. 次の 3 つの変換のそれぞれについて，変換後の 2 枚の画像が縦縞と横縞のどちらになるかを予想してください。

        - 左右反転 `torch.flip(image, dims=[2])`
        - 横方向に 3 画素の巡回シフト `torch.roll(image, shifts=3, dims=2)`
        - 90° 回転 `torch.rot90(image, k=1, dims=[1, 2])`

    2. `exercise/exc20_04.py` で，2 枚の画像と，それぞれに 3 つの変換を施した画像を，`plt.subplots(2, 4)`（各行が元の画像と 3 つの変換後の画像）で描き，`outputs/exc20/exc20_04.png` に保存してください。変換後の画像の shape も表示してください。
    3. 画像のすべての行が 1 行目と等しければ 0（縦縞），すべての列が 1 列目と等しければ 1（横縞），どちらでもなければ -1 を返す関数 `stripe_label(image)` を定義してください。すべての行が 1 行目と等しいかどうかは，ブロードキャスト（第3回）を使って `torch.all(image == image[:, :1, :])` で調べられます。6 枚の変換後の画像に適用し，1 の予想と比較してください。
    4. 縦縞と横縞を分類するタスクで，augmentation として使える変換と使えない変換を，3 の結果に基づいて答えてください。また，使えない変換を augmentation として学習に使うと，学習データにどのような問題が起こるかを説明してください。

5. 前処理の統計量を train のデータだけから計算し，validation のデータにも適用します。次のコードで，身長 [cm] と体重 [kg] を表す 100 人分の特徴量と，2 クラスのラベルを作り，番号を 80 個と 20 個に分けます。

    ```python
    rng = np.random.default_rng(0)
    features = np.stack([rng.normal(170, 6, size=100), rng.normal(60, 8, size=100)], axis=1)
    labels = rng.integers(0, 2, size=100)
    perm = rng.permutation(100)
    train_idx, val_idx = perm[:80], perm[80:]
    ```

    1. `exercise/exc20_05.py` で，`features[train_idx]` から列ごとの平均と標準偏差（`ddof=0`）を計算して表示してください。また，100 人全員から計算した列ごとの平均と標準偏差も表示して比較してください。
    2. 基礎第1問の `ArrayDataset` に，入力に適用する関数を受け取る引数 `transform`（既定値は `None`）を追加し，`transform` が `None` でなければ `__getitem__` で入力に適用するように変更してください。1 で train から計算した平均と標準偏差を tensor にして `mean` と `std` とし，`transform=lambda x: (x - mean) / std` を渡した Dataset を作ってください。`lambda x: 式` は，`x` を受け取って式の値を返す関数を 1 行で書く書き方です（`def` で定義した関数を渡してもかまいません）。この Dataset を `Subset` で train と validation に分けてください。
    3. 標準化後の train と validation の入力の，列ごとの平均と標準偏差を予想してから表示してください。validation の平均が 0，標準偏差が 1 にならないことを確かめ，それが誤りではない理由を説明してください。
    4. 1 で比べた 2 種類の統計量の差は小さくても，100 人全員から計算した統計量を前処理に使ってはいけない理由を，データリークと validation の役割から説明してください。

6. 長さの異なるデータをバッチにまとめる collate 関数を作ります。長さ $n=3, 5, 2, 4, 6$ の 5 個の系列 `torch.arange(1, n + 1, dtype=torch.float32)` と，ラベル `[0, 1, 0, 1, 1]` を順に返す Dataset を使います。

    1. `exercise/exc20_06.py` でこの Dataset を定義し，`batch_size=2` の DataLoader（collate 関数は既定のもの）からバッチを取り出すと何が起きるかを予想してから確かめてください（他の処理とは分けて実行してください）。
    2. 解説の `pad_collate` を，パディングした系列，長さ，ラベルの 3 つを返すように変更し，`collate_fn` に渡してください。各バッチのパディング後の shape と長さを予想してから表示してください。
    3. 5 個の系列の平均を手計算してください。次に，各バッチについて `padded.mean(dim=1)` と，パディングした要素を除いて求めた平均を表示し，手計算の結果と比較してください。パディングした要素を除くには，`torch.arange(padded.shape[1]) < lengths[:, None]` で作った真偽値の tensor（マスク）を `padded` に掛けて和をとり，長さで割ります。
    4. `padded.mean(dim=1)` が元の系列の平均と一致する系列と一致しない系列があることを確かめ，その理由を説明してください。また，マスクの shape がブロードキャスト（第3回）の規則でどのように決まるかを説明してください。

### 発展レベル（3問）
1. augmentation によって，train のデータにない位置の縞を分類できるようになるかを調べます。train には位相を 0 に固定した 16 枚の画像 `StripeDataset(16, seed=0, random_phase=False)` を，test には位相がランダムな 200 枚の画像 `StripeDataset(200, seed=100)` を使います（基礎第3問）。モデルは `nn.Sequential(nn.Flatten(), nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 2))` とします。`nn.Flatten()` は，shape `(N, 1, 16, 16)` の入力を shape `(N, 256)` に変換する層です。

    1. Dataset を受け取り，`__getitem__` で縦方向と横方向にそれぞれ 0 以上 16 未満のランダムな画素数だけ巡回シフトした画像を返す Dataset `RandomShiftDataset` を定義してください。ずらす量は `torch.randint` で決めます。2 つの軸は `torch.roll(x, shifts=(sy, sx), dims=(1, 2))` で同時にずらせます。
    2. augmentation を使わない場合と，train の Dataset を `RandomShiftDataset` で包んだ場合のそれぞれについて，test の正解率を予想してください。
    3. `exercise/exc20_07.py` で，2 つの条件のそれぞれについて，`batch_size=4`，`shuffle=True`，`generator=torch.Generator().manual_seed(0)` の DataLoader，`nn.CrossEntropyLoss`，`torch.optim.Adam(lr=1e-2)` を使って 100 エポック学習させてください。各エポックの train の損失（解説の第6節の方法で求めた平均）と test の正解率を記録し，左に train の損失（`ax.semilogy`），右に test の正解率を描いて，2 つの条件を重ねた図を `outputs/exc20/exc20_07.png` に保存してください。最後に，augmentation を使わない train の 16 枚に対する正解率と，test の正解率を表示してください。
    4. 2 の予想と結果を比較してください。augmentation を使わない場合に train の正解率が高くても test の正解率が低い理由と，augmentation によって test の正解率が上がる理由を，train と test の画像の違いから説明してください。また，augmentation を使った場合の train の損失が，使わない場合より大きく，不規則に変動する理由を説明してください。

2. データパイプラインで使う乱数と，結果の再現性を確かめます。

    1. `exercise/exc20_08.py` で，`StripeDataset(16, seed=0)` に対して，`batch_size=4`，`shuffle=True` の DataLoader を，`generator` のシードを 0，0，1 として 3 つ作ってください。それぞれから 1 エポック分のラベルの並びを表示し，どの 2 つが一致するかを確かめてください。
    2. 発展第1問の augmentation を使う学習を，エポック数を 5 として同じ設定で 2 回実行し，各エポックの test の正解率が完全に一致することを確かめてください。
    3. 2 の 2 回目だけ，`torch.manual_seed(0)` を `torch.manual_seed(1)` に変えて実行してください。学習を始める前のモデルの最初の層の重みと，最初のバッチの入力が，1 回目と一致するかを `torch.equal` で確かめてください。`RandomShiftDataset` のずらす量がどの乱数で決まっているかに着目して，結果を説明してください。
    4. この回のデータパイプラインとモデルの学習の結果を再現できるようにするために固定すべき乱数を，1〜3 の結果に基づいてすべて挙げてください。

3. augmentation がラベルを保つかどうかは，タスクの定義によって変わることを確かめます。縦縞と横縞を分類するタスクに加えて，縞の周期が 4 か 8 かを分類するタスクを考えます。

    1. `exercise/exc20_09.py` で，画像の縞の周期を返す関数 `stripe_period(image)` を作ってください。基礎第4問の `stripe_label` で縞の向きを調べ，値が変化する方向の 1 行（縦縞なら `image[0, 0]`，横縞なら `image[0, :, 0]`）を取り出して，同じ値が続く長さの最大値を 2 倍した値を返すようにします。隣り合う要素の値が異なる位置は，`line[1:] != line[:-1]` で求められます。`make_stripe` で作った周期 4，6，8 の画像に対して，正しい周期を返すことを確かめてください。
    2. 基礎第4問の 3 つの変換のそれぞれについて，向きのラベルと周期のラベルを保つかを予想してください。
    3. ラベル 0 と 1，周期 4 と 8，位相 0 から 7 のすべての組み合わせ（32 枚）の画像を `make_stripe` で作り，3 つの変換のそれぞれについて，変換の前後で向きが一致した画像の枚数と，周期が一致した画像の枚数を表にして表示してください。2 の予想と比較してください。
    4. 3 の結果から，90° 回転を augmentation として使えるタスクと使えないタスクを答え，変換がラベルを保つかどうかがタスクの定義で決まる理由を説明してください。
    5. 周期 6 の縦縞 `make_stripe(0, 6, 0)` に `torch.roll(image, shifts=3, dims=2)` を適用し，変換前と変換後の 1 行目の値を並べて表示してください。画像の幅 16 が周期 6 の倍数でないことに着目して，巡回シフトの後の縞がどこで乱れるかを説明してください。また，このような乱れが augmentation として問題になる場合があるかを考察してください。

## 詰まったときに見る資料
- [PyTorch tutorial: Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch docs: torch.utils.data](https://docs.pytorch.org/docs/stable/data.html)
- [`06-probability.md`](06-probability.md)：標準化と統計量
