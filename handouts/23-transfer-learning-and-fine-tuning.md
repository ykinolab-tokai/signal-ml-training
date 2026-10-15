# 第23回 transfer learning と fine-tuning

## この回の目標

転移学習で更新するパラメータの範囲を比較し，データと初期モデルに応じた選び方を説明できる。

## 解説

この回では，あるタスクで学習したモデルのパラメータを，別のタスクの学習に再利用する方法を扱う。
第21回の `LineDataset`，`SmallCNN`，`draw_line`，`train_one_epoch`，`evaluate` を使うため，第21回の演習で作った `exercise/line_cnn.py` から読み込む。
スクリプトの先頭には，次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from line_cnn import LineDataset, SmallCNN, draw_line, train_one_epoch, evaluate

figure_dir = Path("outputs/exc23")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書き，`line_cnn.py` と同じ `exercise/` の中のファイルとして実行する。

### 1. 転移学習の考え方

#### source task と target task

解きたいタスクの学習用データが少ないと，モデルを乱数の初期値から学習させても，学習用データにだけ当てはまり，新しいデータを正しく予測できないことがある（第19回の過学習）。
このようなとき，データの多い別のタスクで学習したモデルのパラメータを，学習の出発点として使う方法がある。
この方法を **転移学習 (transfer learning)** という。
パラメータを学習済みのタスクを **source task**，新しく解きたいタスクを **target task** という。
source task で学習した特徴が target task にも役立てば，target task の学習用データが少なくても高い性能を得やすい。

この回では，第21回の線分の向きの 4 クラス分類を source task とする。
target task は，横と縦の線分を組み合わせた十字形 `+` と，2 本の斜めの線分を組み合わせた斜めの十字形 `x` の 2 クラス分類とする。
target task の画像は source task より雑音が大きく（標準偏差 0.3），学習に使える画像は 1 クラスあたり 2 枚だけとする。
どちらのタスクの画像も同じ線分からできているため，source task で学習した線分の向きを捉える特徴が，target task でも役立つと期待できる。

```python
class CrossDataset(Dataset):
    def __init__(self, n_per_class, seed, size=32, half_length=4, noise_std=0.3):
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.arange(2), n_per_class)
        images = rng.normal(0.0, noise_std, size=(len(labels), 1, size, size))
        for i, label in enumerate(labels):
            row = rng.integers(half_length, size - half_length)
            col = rng.integers(half_length, size - half_length)
            for line_label in ([0, 1] if label == 0 else [2, 3]):
                draw_line(images[i, 0], line_label, row, col, half_length)
        self.images = torch.from_numpy(images.astype(np.float32))
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
```

`CrossDataset` は，第21回の `LineDataset` と同じ方法で中心を決め，ラベル 0 の画像には横と縦の線分を，ラベル 1 の画像には 2 本の斜めの線分を，同じ中心に描く。
`draw_line` の第2引数は，第21回の線分の向きの番号（0：横，1：縦，2：右下がり，3：右上がり）である。

#### backbone と head

画像分類のモデルは，入力から特徴ベクトルを計算する部分と，特徴ベクトルからクラスごとの logits を計算する部分に分けて考えられる。
前者を **backbone**，後者を **head** という。
第21回の `SmallCNN` では，`features` と global average pooling が backbone にあたり，最後の `nn.Linear(16, num_classes)` が head にあたる。
`SmallCNN` の属性 `head` には global average pooling と `nn.Flatten` も含まれるが，学習するパラメータを持つのは最後の `head[2]` だけである。
`model.head[:2]` のように `nn.Sequential` をスライスすると，先頭の 2 つの層だけを並べた `nn.Sequential` が得られる。

```python
model = SmallCNN()
x = torch.zeros(5, 1, 32, 32)
features = model.head[:2](model.features(x))
print(features.shape)  # torch.Size([5, 16])
```

backbone の特徴は，source task の 4 クラスに限らない，線分の向きのような一般的な情報を表すと期待できる。
一方，head の出力は source task の 4 クラスに対応するため，クラスの異なる target task にはそのまま使えない。
そこで，学習済みの backbone を残し，head を target task のクラスの数に合わせた新しい層に付け替える。

### 2. パラメータの凍結と更新の範囲

#### requires_grad による凍結

`requires_grad_(False)` を呼んだパラメータは，`backward()` で勾配が計算されず，`.grad` は `None` のままになる（第17回）。
勾配がないパラメータは，optimizer の `step()` で更新されない。
このように，一部のパラメータを学習中に変えないようにすることを **凍結 (freeze)** という。

```python
model = SmallCNN()
for p in model.features.parameters():
    p.requires_grad_(False)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(trainable, total)  # 68 1316

model(torch.zeros(1, 1, 32, 32)).sum().backward()
print(model.features[0].weight.grad)    # None
print(model.head[2].weight.grad.shape)  # torch.Size([4, 16])
```

`model.features.parameters()` は，`features` の中の層のパラメータだけを返す。
凍結しても forward の計算は変わらず，出力の形も同じである。
optimizer には，`[p for p in model.parameters() if p.requires_grad]` のように，学習するパラメータだけを渡す。

なお，`requires_grad_(False)` が止めるのはパラメータの更新だけである。
torchvision の ResNet などが含む batch normalization の層は，学習時の動作（`model.train()`）では内部の統計量を更新するため，凍結した部分の動作も変えたくない場合は，その部分を `eval()` にする。
`SmallCNN` は，そのような層を含まない。

#### 更新の範囲による 3 つの条件

学習済みのパラメータを初期値として，target task のデータでさらに学習することを **fine-tuning** という。
どの範囲のパラメータを更新するかによって，次の条件がある。
この回では，比較の基準として，乱数の初期値からすべてのパラメータを学習する条件（scratch）も使う。

| 条件 | 初期値 | 更新するパラメータ |
| --- | --- | --- |
| scratch | 乱数 | すべて |
| **linear probe** | source task の学習済みパラメータ（head は新しい層） | head だけ |
| **partial fine-tuning** | 同上 | 入力側の層を凍結し，残りの層と head |
| **full fine-tuning** | 同上 | すべて |

linear probe では，backbone が計算する特徴は変わらず，その特徴の線形変換でクラスを分けられるかだけを学習する。
更新するパラメータが少ないため，データが少なくても学習用データに過剰に合わせにくいが，backbone の特徴が target task に合っていなければ性能は上がらない。
full fine-tuning では，backbone の特徴も target task に合わせて変えられるが，更新するパラメータが多いため，データが少ないと学習用データに過剰に合わせてしまうおそれがある。
partial fine-tuning は，その中間である。
一般に，入力に近い層は線分や端のように多くのタスクに共通する特徴を，出力に近い層はタスクに固有の特徴を表すことが多いため，入力側の層を凍結する。

### 3. 学習済みパラメータの保存と読み込み

#### state_dict

モデルのパラメータは，`model.state_dict()` で，名前と tensor を対応させた辞書として取り出せる。

```python
model = SmallCNN()
for name, tensor in model.state_dict().items():
    print(name, tuple(tensor.shape))
```

```text
features.0.weight (8, 1, 3, 3)
features.0.bias (8,)
features.3.weight (16, 8, 3, 3)
features.3.bias (16,)
head.2.weight (4, 16)
head.2.bias (4,)
```

名前は，属性の名前と `nn.Sequential` の中の番号をつないだものである。
例えば `features.3.weight` は，`features` の 3 番目（0 から数える）の層，つまり 2 つ目の畳み込み層の重みである。
ReLU と pooling はパラメータを持たないため，名前に現れない。
第19回の early stopping では，`state_dict` を複製して，学習途中のモデルのパラメータを残した。

#### ファイルへの保存と読み込み

`torch.save` で `state_dict` をファイルに保存し，`torch.load` で読み込んだ辞書を `load_state_dict` でモデルに設定する。

```python
torch.save(model.state_dict(), figure_dir / "session23_example.pt")
loaded = SmallCNN()
loaded.load_state_dict(torch.load(figure_dir / "session23_example.pt"))
print(torch.equal(model.features[0].weight, loaded.features[0].weight))  # True

loaded.head[2] = nn.Linear(16, 2)
print(loaded.head[2])  # Linear(in_features=16, out_features=2, bias=True)
```

`load_state_dict` は，名前と形がすべて一致するときだけ値を設定し，形の異なる層があるとエラーになる。
そのため，source task のパラメータは，まず source task と同じ構造のモデル（4 クラスの `SmallCNN`）に読み込み，その後で head の線形層を付け替える。
`nn.Sequential` の要素は，`loaded.head[2] = ...` のように番号を指定して置き換えられる。
付け替えた層のパラメータは乱数で初期化されるため，結果を再現するには，その前に `torch.manual_seed` を実行する。
`torch.load` で読み込むファイルは，自分で保存したものなど，入手元が信頼できるものに限る。

### 4. 層ごとの学習率

head は乱数で初期化されるため大きく更新する必要があるが，backbone は学習済みの特徴を壊さないように少しずつ更新したい。
optimizer には，パラメータのグループごとに異なる学習率を指定できる。

```python
model = SmallCNN(num_classes=2)
optimizer = torch.optim.Adam([
    {"params": model.features.parameters(), "lr": 1e-3},
    {"params": model.head.parameters(), "lr": 1e-2},
])
print([group["lr"] for group in optimizer.param_groups])  # [0.001, 0.01]
```

optimizer に渡すリストの各要素は，パラメータのグループを表す辞書である。
`"params"` にそのグループのパラメータを，`"lr"` にそのグループの学習率を指定する。
`"lr"` を省略したグループには，`torch.optim.Adam(..., lr=...)` のように指定した既定の学習率が使われる。
`optimizer.param_groups` で，グループごとの設定を確かめられる。

### 5. 公開されている学習済みモデル

実際の画像認識では，大規模な画像データセットで学習したモデルを source とすることが多い。
torchvision には，ImageNet（約 128 万枚，1000 クラスの画像データセット）で学習した ResNet などのモデルが用意されている。
`torchvision.models.resnet18(weights="DEFAULT")` のように学習済みの重みを指定すると，初回の実行時に重みのファイル（約 45 MB）がダウンロードされる。
次の例では，構造だけを確かめるため，`weights=None`（乱数の初期値，ダウンロードなし）で作る。

```python
import torchvision

resnet = torchvision.models.resnet18(weights=None)
print(resnet.fc)  # Linear(in_features=512, out_features=1000, bias=True)
resnet.fc = nn.Linear(512, 2)
print(resnet(torch.zeros(1, 3, 224, 224)).shape)  # torch.Size([1, 2])
```

ResNet-18 の head は属性 `fc` の線形層であり，target task のクラスの数に合わせた層に付け替える。
学習済みの重みを使う場合は，入力の画像にも，source task の学習時と同じ前処理（大きさの変更と，チャネルごとの平均と標準偏差による標準化）を行う必要がある。
この回の演習では，ダウンロードを必要としないように，第21回の `SmallCNN` を使う。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第23回の第1問であれば，`exercise/exc23_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc23`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第3問の図であれば，`outputs/exc23/exc23_03.png` とします。
解説の `CrossDataset` は，第21回の `exercise/line_cnn.py` に追加し，`from line_cnn import CrossDataset` のように読み込んでください。
第21回の演習に取り組んでいない場合は，第21回の解説のコードから `exercise/line_cnn.py` を作ってください。
基礎第1問の `set_trainable` と基礎第4問の `run` は，後の問題でも使います。`exercise/line_cnn.py` に追加して読み込むか，各ファイルに書き写してください。
乱数の指定がない場合は，モデルを作る直前に `torch.manual_seed(0)` を実行してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（5問）
1. 更新の範囲による 3 つの条件で，学習するパラメータの数がどう変わるかを確かめます。モデルを `SmallCNN(num_classes=2)` とし，partial fine-tuning では最初の畳み込み層 `features[0]` だけを凍結するものとします。

    1. `features[0]`，`features[3]`，`head[2]` のパラメータの数と，モデル全体のパラメータの数を手計算してください。
    2. linear probe，partial fine-tuning，full fine-tuning のそれぞれについて，学習するパラメータの数を手計算してください。
    3. `exercise/exc23_01.py` に，モデルと条件名 `"linear_probe"`，`"partial"`，`"full"` を受け取り，条件に合わせて各パラメータの `requires_grad` を設定する関数 `set_trainable(model, mode)` を書いてください。関数の最初で，すべてのパラメータを `requires_grad_(True)` に戻してください。3 つの条件で学習するパラメータの数を表示し，2 の結果と比較してください。
    4. 各条件で `torch.randn(4, 1, 32, 32)` を入力し，logits の形がすべて `(4, 2)` であることを確かめてください。さらに，logits の和に `backward()` を呼んだ後，3 つの層の重みの `.grad` が `None` かどうかを表示してください。logits の形が同じでも，条件によって学習で変わるパラメータが異なることを説明してください。

2. source task で `SmallCNN` を学習させ，そのパラメータを保存して読み込みます。

    1. `exercise/exc23_02.py` で，第21回の基礎第5問と同じ条件（`LineDataset(n_per_class=250, seed=0)` で 5 エポック，Adam の学習率 0.01，バッチサイズ 32，`torch.manual_seed(0)`，`generator=torch.Generator().manual_seed(0)`）で `SmallCNN()` を学習させてください。`LineDataset(n_per_class=50, seed=1)` に対する正解率を表示してください。
    2. `model.state_dict()` の各要素の名前と形を表示し，それぞれがどの層のパラメータかを説明してください。
    3. `torch.save` で `state_dict` を `outputs/exc23/source_cnn.pt` に保存してください。第3問以降では，このファイルを source task の学習済みパラメータとして使います。
    4. `torch.manual_seed(1)` の直後に別の `SmallCNN()` を作ってください。このモデルについて，`outputs/exc23/source_cnn.pt` を読み込む前と後に，`LineDataset(n_per_class=50, seed=1)` の画像に対する logits を `torch.no_grad()` の中で計算してください。それぞれが 1 で学習したモデルの logits と一致するかを，`torch.equal` で確かめてください。
    5. `SmallCNN(num_classes=2)` に `outputs/exc23/source_cnn.pt` を `load_state_dict` で読み込むと何が起きるかを予想してから確かめ，エラーの内容から理由を説明してください。エラーになる処理は，他の処理と分けて実行してください。

3. target task のデータを確かめ，head の付け替えが必要な理由を考えます。

    1. `exercise/exc23_03.py` で，学習用の `train_set = CrossDataset(n_per_class=2, seed=0)` と評価用の `test_set = CrossDataset(n_per_class=100, seed=100)` を作り，それぞれの要素数とラベルを表示してください。学習用の 4 枚を `plt.subplots(1, 4)` で並べ，タイトルにラベル（0 は `+`，1 は `x`）を付けて `outputs/exc23/exc23_03.png` に保存してください。
    2. source task の学習済みパラメータを読み込んだ `SmallCNN()` で，`test_set` の 200 枚の予測クラス（source task の 4 クラスのどれか）を求めてください。target task のラベルごとに予測クラスの枚数を `torch.bincount(..., minlength=4)` で数え，2 行 4 列の表にして表示してください。実行前に，`+` と `x` の画像がそれぞれどのクラスと予測されるかを予想してください。
    3. 2 の結果から，source task の head の出力を target task の予測にそのまま使えない理由を説明してください。また，2 つのラベルで予測クラスの分布が異なるかどうかに注目し，backbone の特徴が target task に役立つかを考察してください。
    4. 読み込んだモデルの `head[2]` を，`torch.manual_seed(0)` の直後に作った `nn.Linear(16, 2)` に付け替えてください。`test_set` の画像に対する logits の形が `(200, 2)` になることと，付け替えた後も `features[0].weight` が読み込んだ値と一致することを確かめてください。

4. 学習用の画像が 1 クラスあたり 2 枚しかない target task で，4 つの条件を比べます。各条件の学習は，次の手順で行います。

    - `torch.manual_seed(seed)` の直後に `SmallCNN()` を作る。scratch 以外の条件では，`outputs/exc23/source_cnn.pt` を読み込む。
    - `head[2]` を `nn.Linear(16, 2)` に付け替え，基礎第1問の `set_trainable` で条件を設定する（scratch では `"full"` とする）。
    - 学習用データ `train_set = CrossDataset(n_per_class=2, seed=seed)` の 4 枚を 1 つのバッチとする `DataLoader(train_set, batch_size=4)` を作る。
    - 学習するパラメータだけを渡した `torch.optim.Adam`（学習率 0.01）と `nn.CrossEntropyLoss()` を使い，`train_one_epoch` を 100 回呼んで，パラメータを 100 回更新する。
    - 基礎第3問の `test_set` に対する正解率を `evaluate` で求める。

    1. 4 つの条件（scratch，linear probe，partial fine-tuning，full fine-tuning）のうち，正解率の平均が最も高い条件と，seed による正解率のばらつきが最も大きい条件を予想してください。
    2. `exercise/exc23_04.py` で，上の手順を，条件名と `seed` を受け取って正解率を返す関数 `run(mode, seed)` にまとめてください。`seed` を 0〜4 の 5 通りとして各条件の正解率を求め，条件ごとに 5 つの正解率と，その平均と標準偏差を表示してください。
    3. 横軸を条件，縦軸を正解率とし，条件ごとの 5 つの正解率を点で，その平均を横線で描いた図を `outputs/exc23/exc23_04.png` に保存してください。
    4. 1 の予想と結果を比較してください。scratch の正解率が seed によって大きくばらつく理由を，学習用の画像の枚数と，学習するパラメータの数から説明してください。
    5. seed ごとに比べると，scratch の正解率が他の条件と同程度かそれ以上になる場合があるかを確かめてください。条件を比べるときに，1 つの seed の結果だけで判断せず，複数の seed の平均とばらつきを見る理由を説明してください。

5. 凍結したパラメータが学習で変わらないことを確かめ，条件の選び方を考えます。

    1. `exercise/exc23_05.py` で，基礎第4問の手順に従い，`seed = 0` の linear probe と partial fine-tuning のモデルを作ってください。学習の前に `{name: p.detach().clone() for name, p in model.named_parameters()}` で各パラメータを複製しておき，学習の後に，各パラメータが学習の前と一致するかを `torch.equal` で調べて表示してください。
    2. 凍結したパラメータだけが学習の前後で一致することを確かめ，その理由を `requires_grad` と optimizer の関係から説明してください。
    3. 基礎第4問の結果と，各条件で学習するパラメータの数を根拠に，target task の学習用データが少ない場合に最初に試す条件と，その理由を書いてください。また，学習用データが十分に多い場合にはどの条件が有利になると予想されるかを，理由とともに書いてください（発展第1問で確かめられます）。

### 発展レベル（3問）
1. target task の学習用データの数によって，条件の優劣がどう変わるかを調べます。

    1. `exercise/exc23_06.py` で，基礎第4問の `run` に，1 クラスあたりの学習用の画像の数 `n_per_class` を引数として追加してください。バッチは学習用の画像全体とし，更新の回数は基礎第4問と同じ 100 回とします。
    2. `n_per_class` を 1，2，5，10，20 とし，scratch，linear probe，full fine-tuning のそれぞれについて，`seed` 0〜4 の正解率の平均と標準偏差を求めてください。
    3. 横軸を `n_per_class`（対数目盛り），縦軸を正解率の平均とし，標準偏差を `ax.errorbar` の `yerr` で示した図を `outputs/exc23/exc23_06.png` に保存してください。横軸を対数目盛りにするには，`ax.set_xscale("log")` を使います。
    4. 学習用の画像を増やしたときの，scratch と linear probe の正解率の変わり方の違いを説明してください。linear probe の正解率がある値で頭打ちになる理由を，更新されるパラメータの範囲から説明してください。
    5. full fine-tuning と scratch の結果の関係を説明し，学習用データの量に応じた条件の選び方をまとめてください。

2. backbone と head に異なる学習率を使う効果を調べます。target task は基礎第4問と同じ（1 クラスあたり 2 枚）とし，full fine-tuning で次の 3 つの optimizer を比べます。

    - (a) すべてのパラメータに学習率 0.01
    - (b) すべてのパラメータに学習率 0.001
    - (c) `features` に学習率 0.001，`head` に学習率 0.01（解説の「層ごとの学習率」）

    1. `exercise/exc23_07.py` で，`seed` 0〜4 について 3 つの optimizer でそれぞれ学習させ，基礎第3問の `test_set` に対する正解率の平均と標準偏差を表示してください。
    2. 学習の前後で最初の畳み込み層の重みがどれだけ変わったかを，相対的な変化量 $\lVert W_\mathrm{after} - W_\mathrm{before} \rVert / \lVert W_\mathrm{before} \rVert$ で求めてください。$\lVert\cdot\rVert$ は，すべての要素の 2 乗和の平方根（tensor の `.norm()`）とします。
    3. 学習後の `features` を source task の学習済みの head と組み合わせ，source task のデータ `LineDataset(n_per_class=50, seed=1)` に対する正解率を求めてください。このモデルは，source task の学習済みパラメータを読み込んだ `SmallCNN()` を作り，その `features` に `model.features.state_dict()` を `load_state_dict` で読み込むと作れます。
    4. 1〜3 の結果を，optimizer ごとに表にまとめてください。(b) の target task の正解率が低い理由と，(a) で source task の正解率が下がる理由を説明してください。target task の学習によって source task で得た性能が失われる現象は，**破滅的忘却 (catastrophic forgetting)** と呼ばれます。(c) の結果が 2 つの点でどうなったかを説明してください。

3. linear probe の性能が，backbone の特徴の質で決まることを確かめます。基礎第3問の `test_set` を使います。

    1. `exercise/exc23_08.py` で，source task の学習済みパラメータを読み込んだ `SmallCNN()` と，`torch.manual_seed(0)` の直後に作った学習していない `SmallCNN()` のそれぞれについて，`test_set` の画像の特徴ベクトル（解説の `model.head[:2](model.features(x))`，形 `(200, 16)`）を `torch.no_grad()` の中で計算してください。
    2. 特徴ベクトルの各成分について，ラベル 0 の画像での平均 $\mu_0$ と分散 $\sigma_0^2$，ラベル 1 の画像での平均 $\mu_1$ と分散 $\sigma_1^2$ を求め，次の値 $d$ を計算してください。tensor の `var` は既定では $N-1$ で割るため，第6回と同じく $N$ で割る分散を，第20回の `std` と同様に `var(dim=0, correction=0)` で求めてください。値がすべて 0 の成分では分母が 0 になるため，根号の中に $10^{-12}$ を加えてください。2 つの backbone について，16 個の $d$ の最大値を表示してください。

        $$
        d = \frac{\lvert \mu_0 - \mu_1 \rvert}{\sqrt{(\sigma_0^2 + \sigma_1^2)/2}}
        $$

    3. $d$ が何を表すかを，第6回の平均と標準偏差を使って説明してください。2 つの backbone の結果を比べ，どちらの特徴のほうが 2 つのラベルを分けやすいかを答えてください。
    4. 学習していない backbone を凍結して head だけを学習させる条件を，基礎第4問と同じ手順（`seed` 0〜4）で学習させ，正解率の平均と標準偏差を求めてください。`n_per_class` を 2 と 20 にして，source task の backbone を使う linear probe と比べてください。
    5. 学習用データを 10 倍に増やしても，学習していない backbone を使う条件の正解率があまり上がらない理由を，2 の結果と，head が特徴の線形変換しか学習できないことから説明してください。

## 詰まったときに見る資料
- [`21-image-model-basics.md`](21-image-model-basics.md)
- [PyTorch: Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [torchvision: Models and pre-trained weights](https://docs.pytorch.org/vision/stable/models.html)
