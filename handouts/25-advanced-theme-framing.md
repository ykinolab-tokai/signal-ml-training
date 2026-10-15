# 第25回 年度別発展テーマ：問題設定

## この回の目標

選択した発展テーマの入力・出力と評価指標を定め，最小実装が正しく動くことを確認して記録できる。

## 解説

第25回から第27回では，[`advanced/`](advanced/) の候補から年度ごとに 1 つ選んだ発展テーマを，3 回に分けて扱う。
この回では，テーマの入力・出力・評価指標を決め，入力から評価までを 1 回通す最小実装を作り，それが正しく動いていることを確かめる。
コード例では，第19回までに扱った PyTorch を使う。
スクリプトの先頭には，次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

figure_dir = Path("outputs/exc25")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. 発展テーマの進め方

#### 3 回の作業と保存物

発展テーマは，その年度に [`advanced/`](advanced/) の候補から 1 つだけを選ぶ。
画像・音響・共通基盤のテーマを同じ年度にすべて扱うわけではない。
第25回から第28回の作業と主な保存物は次のとおりである。

| 回 | 主な作業 | 主な保存物 |
| --- | --- | --- |
| 第25回 | 問題設定，評価指標の実装，最小実装と動作確認 | 問題設定（`exercise/exc25_01.md`），最小実装の script，実行記録（`exercise/exc25_06.md`） |
| 第26回 | 条件を変えた比較実験と失敗分析 | 結果の CSV と図，比較記録（`exercise/exc26_03.md`） |
| 第27回 | 実装の整理，再実行手順の記録，共有する資産の選択 | 整理した script，再実行手順（`exercise/exc27_04.md`） |
| 第28回 | 技術デモと code walkthrough | デモの説明メモ（`exercise/exc28_01.md`） |

選択テーマの実装は，候補 handout が指定するファイル（例：`advanced_image_segmentation_demo.py`）に書く。
テーマによらない練習は，各回の `exercise/excXX_YY.py` で行う。

#### 実行記録

第26回の比較実験と第27回の整理では，第25回に作った処理を何度も実行し直す。
このため，実行したコマンド，生成された保存物，まだ動かない処理を，その都度記録しておく。
コマンドは，[共通手順](README.md#python環境の準備更新と演習の実行)のとおり，作業リポジトリのルートで実行する形で書く。
記録に書くのは，後から読んだ人が同じ操作を再現するために必要な情報であり，作業の感想ではない。

### 2. 問題設定

#### 入力と出力

**問題設定 (problem formulation)** とは，何を入力し，何を出力し，出力の良し悪しをどう判断するかを決めることである。
入力と出力は，言葉だけでなく，shape，dtype，値の範囲，各軸の意味まで決める。
例えば，segmentation の入力を「画像」と書くだけでは，チャネルの軸があるか，画素値が 0〜255 か 0〜1 かが分からない。
「`(N, 1, 32, 32)` の `float32`，値の範囲は $[0, 1]$，軸は（サンプル，チャネル，縦，横）」のように書くと，実装と確認の基準になる。

候補テーマの入力と出力は，おおよそ次のとおりである。
`N` はサンプル数，`B` は batch size，`D` は入力の次元を表す。

| テーマ | 入力 | 出力 | 学習に使う損失 | 評価指標の例 |
| --- | --- | --- | --- | --- |
| 画像 segmentation | 画像 `(N, 1, 32, 32)` | 画素ごとの logits `(N, 1, 32, 32)` | `BCEWithLogitsLoss` | pixel accuracy，Dice |
| 音響 tagging | log-mel `(N, 1, n_mels, frames)` | クラスごとの logits `(N, 2)` | `BCEWithLogitsLoss` | threshold 後の precision，recall |
| representation learning | 入力の pair `(N, D)` と `(N, D)` | embedding `(N, 2)` | `CosineEmbeddingLoss` | 平均 cosine similarity |
| SVD と PCA | データ行列 `(N, D)` | 低次元表現 `(N, k)` と再構成 `(N, D)` | なし（学習しない） | 再構成誤差 |
| 効率化と export | model と入力 `(B, D)` | 推論時間，TorchScript のファイル | なし（学習しない） | 1 回あたり・1 サンプルあたりの時間 |

候補 handout に書かれていない値（データの個数，入力の次元など）は，自分で決めて理由とともに記録する。

#### 評価指標と良い方向

**評価指標 (evaluation metric)** は，出力の良し悪しを 1 つの数値で表す尺度である。
評価指標を決めるときは，次の 3 点を明記する。

- 定義式：何を数えて，何で割るか
- 良い方向：大きいほど良いか，小さいほど良いか
- 集計の単位：サンプルごとに計算して平均するか，全サンプルをまとめて数えてから計算するか

第13回の VAD プロジェクトで，評価対象のすべての信号から 1 つの混同行列を作ってから macro-F1 を計算すると定めていたのは，集計の単位を決めた例である。
集計の単位が異なると，同じ予測でも値が変わる。
例えば，Dice を画像ごとに計算して平均すると，前景の小さい画像も大きい画像も同じ重みで平均される。
一方，全画像の画素をまとめて数えてから計算すると，前景の大きい画像の影響が大きくなる。

学習に使う損失と評価指標は，同じとは限らない。
損失は勾配を計算するために微分できる必要があるが，評価指標は threshold で 0 か 1 に変換した後の値から計算してもよい。
例えば，segmentation では `BCEWithLogitsLoss` で学習し，threshold 後の mask の Dice で評価する。

### 3. 評価指標の実装

#### 混同行列

正解と予測がそれぞれ 1（正例）か 0（負例）である 2 クラスの予測は，次の 4 種類に分けられる。

| | 正解が 1 | 正解が 0 |
| --- | --- | --- |
| 予測が 1 | TP (true positive) | FP (false positive) |
| 予測が 0 | FN (false negative) | TN (true negative) |

この 4 種類の個数を並べた表を **混同行列 (confusion matrix)** という。
正解率 (accuracy) と，第13回で扱った precision，recall は，次の式で計算する。

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN},\qquad
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

precision は 1 と予測したもののうち正解も 1 だった割合，recall は正解が 1 のもののうち 1 と予測できた割合である。
F1 は，precision と recall の調和平均 $2\,\mathrm{Precision}\cdot\mathrm{Recall}/(\mathrm{Precision}+\mathrm{Recall})$ である。
segmentation では，画素ごとに予測と正解を比べて数える。
画素ごとの正解率を **pixel accuracy** という。

#### Dice と IoU

予測が 1 の画素の集合を $P$，正解が 1 の画素の集合を $T$ とし，集合の要素数を $|P|$ のように書く。
**Dice 係数 (Dice coefficient)** と **IoU (intersection over union)** は，2 つの集合の重なりを測る指標であり，次の式で定義される。

$$
\mathrm{Dice}=\frac{2|P\cap T|}{|P|+|T|}=\frac{2TP}{2TP+FP+FN},\qquad
\mathrm{IoU}=\frac{|P\cap T|}{|P\cup T|}=\frac{TP}{TP+FP+FN}
$$

どちらも $[0, 1]$ の値をとり，予測と正解が完全に一致すると 1，まったく重ならないと 0 になる。
どちらの式にも TN が現れないため，背景の画素が多くても値は大きくならない。

```python
pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
target = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

tp = np.sum((pred == 1) & (target == 1))
fp = np.sum((pred == 1) & (target == 0))
fn = np.sum((pred == 0) & (target == 1))
tn = np.sum((pred == 0) & (target == 0))
print(tp, fp, fn, tn)  # 2 2 1 5

accuracy = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
dice = 2 * tp / (2 * tp + fp + fn)
iou = tp / (tp + fp + fn)
print(accuracy, precision, recall)  # 0.7 0.5 約 0.667
print(dice, iou)                    # 約 0.571 0.4
```

`(pred == 1) & (target == 1)` は，2 つの条件を両方満たす要素で `True` となる真偽値の配列である。
`np.sum` は `True` を 1，`False` を 0 として足すので，条件を満たす要素の個数が得られる。
この数え方は配列の形によらないため，2 次元の mask や `(N, 1, H, W)` の配列にもそのまま使える。
この例では，FP が 2 個，FN が 1 個あるため，precision は 0.5，recall は約 0.667 と異なる値になる。
Dice は FP と FN の両方を分母に含むため，その間の値（約 0.571）になる。

#### 0 除算の扱い

正解にも予測にも 1 が 1 つもない場合，Dice と IoU の分母は 0 になる。
同様に，予測に 1 が 1 つもないと，precision の分母が 0 になる。
上の例のように NumPy の数で計算すると，0 で割ったところで警告が出て値は `nan` になる。
Python の `int` に変換してから割ると，`ZeroDivisionError` で実行が止まる。
segmentation では前景のない画像もあり得るため，このような場合の値を前もって決めておく。
例えば，正解も予測も空の場合は，完全に一致しているとみなして Dice を 1 とする定義がよく使われる。
どの定義を使ったかは，評価指標の説明に書いておく。

### 4. 最小実装

#### end-to-end の構成

**最小実装 (minimal implementation)** とは，性能は低くてもよいので，データの作成から評価と保存までが 1 回通る実装である。
このように，入力から最終的な出力までを通して動かすことを，**end-to-end** で動かすという。
最初から性能を上げようとすると，うまくいかないときに，どこが正しく動いていないのかを切り分けにくい。
最小実装では，データの作成，model の構築，学習，評価をそれぞれ関数に分けておく。
第26回の比較実験では，これらの関数の引数を変えるだけで条件を切り替えられる。

次の例では，2 次元の点 $(x_1, x_2)$ を，$x_1x_2>0$（第1象限か第3象限）なら 1，それ以外なら 0 に分類する。
2 つのクラスの境界は 2 本の座標軸であり，線形モデルの境界は 1 本の直線なので（第19回），線形モデルではこの分類を表せない。

```python
def make_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, 2)).astype(np.float32)
    y = (x[:, 0] * x[:, 1] > 0).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)

def build_model(hidden):
    return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 2))

def train(model, x, y, steps, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return (pred == y).float().mean().item()

x_train, y_train = make_data(200, seed=0)
x_test, y_test = make_data(1000, seed=1)
torch.manual_seed(0)
model = build_model(hidden=16)
history = train(model, x_train, y_train, steps=300, lr=1e-2)
print(len(history), history[-1])        # 300 約 0.043
print(evaluate(model, x_test, y_test))  # 約 0.98
```

`make_data` は，seed を指定した NumPy の乱数生成器（第6回）で点を作り，入力と正解を tensor で返す。
学習用と評価用で seed を変えているため，2 つのデータは別々に生成された点からなる。
model の初期値は，`build_model` を呼ぶ前の `torch.manual_seed(0)` で決まる。
`train` は損失の推移を，`evaluate` は正解率を返す。
結果を `print` するだけでなく返り値として返しておくと，第26回で複数の条件の結果を集めるときにそのまま使える。

#### shape の確認

```python
logits = model(x_train)
assert logits.shape == (200, 2), logits.shape
print(x_train.shape, x_train.dtype, y_train.dtype)  # torch.Size([200, 2]) torch.float32 torch.int64
```

`assert 条件, 式` は，条件が偽のときに `AssertionError` を発生させ，式の値をエラーメッセージとして表示する。
shape の予想を `assert` として書いておくと，予想と異なる shape が現れた時点で実行が止まり，原因の箇所が分かる。
`logits.shape` は `torch.Size([200, 2])` であり，タプル `(200, 2)` と `==` で比較できる。
`nn.CrossEntropyLoss` に渡す正解は `int64` のクラス番号であり（第18回），dtype も確認しておく。

### 5. 動作確認

#### 学習前の損失の理論値

学習を始める前の損失を予想し，実際の値と比べると，損失の計算の誤りを早く見つけられる。
$C$ クラス分類で model がどのクラスにも同じ logits を出力すると，softmax の確率はすべて $1/C$ になる（第18回）。
このとき，交差エントロピーは正解のクラスによらず，次の値になる。

$$
-\log\frac{1}{C}=\log C
$$

初期化した直後の model の logits は 0 に近い値になることが多いため，学習前の損失は $\log C$ に近くなる。
2 クラスでは $\log 2\approx0.693$，3 クラスでは $\log 3\approx1.099$ である。
`BCEWithLogitsLoss` でも，logits が 0 なら sigmoid の値が 0.5 になるので，損失は $\log 2$ になる。
`CosineEmbeddingLoss` で正例の pair だけを使う場合，損失は $1-\cos\theta$ の平均である（$\theta$ は pair の embedding のなす角）。
2 次元の embedding の向きがばらばらなら $\cos\theta$ の平均は 0 に近いため，学習前の損失は 1 に近くなる。

```python
torch.manual_seed(0)
model = build_model(hidden=16)
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    initial_loss = loss_fn(model(x_train), y_train).item()
print(initial_loss, np.log(2))  # 約 0.711 約 0.693
```

学習前の損失が理論値の数倍になる場合は，正解の形式（クラス番号か one-hot か），入力の値の範囲，損失関数の選び方を疑う。

#### 1 バッチでの過学習

少数のサンプルからなる 1 つのバッチだけを繰り返し学習すると，十分な大きさの model なら損失はほぼ 0 まで下がる。
この性質を使った確認を，**1 バッチでの過学習 (overfitting a single batch)** による確認という。
損失が下がらない場合は，性能を議論する以前に，学習の処理に誤りがある可能性が高い。
例えば，`zero_grad` の書き忘れ，optimizer に渡すパラメータの誤り，入力と正解の対応のずれ，学習率の設定の誤りなどである。

```python
torch.manual_seed(0)
model = build_model(hidden=16)
history = train(model, x_train[:8], y_train[:8], steps=200, lr=1e-2)
print(history[0], history[-1])                    # 約 0.639 約 0.0068
print(evaluate(model, x_train[:8], y_train[:8]))  # 1.0
```

1 バッチで過学習できることは，学習の処理が働いていることを示すだけであり，新しいデータに対する性能を示すものではない。

#### 自明なベースライン

学習をしない単純な予測で得られる評価指標の値を，**自明なベースライン (trivial baseline)** という。
例えば，学習データで最も多いクラスを常に予測する方法や，segmentation で全画素を背景と予測する方法である。
model の評価指標の値は，自明なベースラインの値と比べて初めて解釈できる。

```python
counts = torch.bincount(y_train)
majority = counts.argmax()
print(counts, majority)                            # tensor([103,  97]) tensor(0)
print((y_test == majority).float().mean().item())  # 約 0.509
```

`torch.bincount` は，0 以上の整数を値ごとに数える。
この例では，常にクラス 0 と予測したときの正解率は約 0.509 であり，前の例の model の正解率（約 0.98）はこれを大きく上回る。
一方，前景が小さい segmentation では，全画素を背景と予測するだけで pixel accuracy が 0.9 を超えることがある。

#### 学習しないテーマでの確認

学習を行わないテーマでは，数学的に成り立つ関係や，処理の前後で変わらないはずの値を確かめる。
SVD と PCA のテーマでは，`np.linalg.svd` で中心化したデータ行列 $\boldsymbol{X}_c$ を分解する。
特異値を大きい順に $s_1\ge s_2\ge\cdots$ とすると，上位 $k$ 個の主成分で再構成したときの二乗誤差の和は，残りの特異値の 2 乗の和 $\sum_{i>k}s_i^2$ に等しい。

```python
rng = np.random.default_rng(0)
X = rng.normal(0, 1, size=(5, 3))
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
for k in [1, 2, 3]:
    Xk = (Xc @ Vt[:k].T) @ Vt[:k]
    print(k, np.sum((Xc - Xk) ** 2), np.sum(S[k:] ** 2))
```

```text
1 3.001082559313517 3.001082559313521
2 0.1036549840038462 0.10365498400384625
3 1.0841180027577275e-30 0.0
```

`S` は特異値を大きい順に並べた配列，`Vt` の各行は主成分の向きを表す単位ベクトルである。
`Xc @ Vt[:k].T` が $k$ 次元の表現，それに `Vt[:k]` を掛けたものが元の空間での再構成になる。
$k=3$ ですべての成分を使うと，誤差は丸め誤差の大きさ（約 $10^{-30}$）になる。

効率化と export のテーマでは，trace した model が元の model と同じ出力を返すことを確かめる。

```python
torch.manual_seed(0)
model = build_model(hidden=16).eval()
example = torch.randn(4, 2)
traced = torch.jit.trace(model, example)
x_new = torch.randn(64, 2)
print(torch.allclose(traced(x_new), model(x_new)))  # True
```

`torch.jit.trace` は，例として与えた入力で model を 1 回実行し，そのときの計算の流れを TorchScript という形式で記録する。
`torch.allclose` は，2 つの tensor の全要素が誤差の範囲内で一致するかを返す。
trace に使った入力とは batch size の異なる入力で一致を確かめると，記録した計算が別の batch size でも使えることが分かる。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第25回の第2問であれば，`exercise/exc25_02.py` とします。
文章で答える問題は，同じ番号の `.md` ファイルに書いてください。第1問であれば，`exercise/exc25_01.md` とします。
各 `.py` ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc25`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第4問の図であれば，`outputs/exc25/exc25_04.png` とします。
選択テーマの実装は，候補 handout が指定するファイル（例：`advanced_image_segmentation_demo.py`）に書き，確かめた結果や説明はそのコードのコメントに残してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（6問）
1. 選択テーマの問題設定を，文章で具体的に決めます。[`advanced/README.md`](advanced/README.md) と，今年選んだテーマの候補 handout を読んでから取り組んでください。

    1. `exercise/exc25_01.md` を作成し，次の 6 つの見出しを順に書いてください。

        - `## 選択テーマ`：テーマ名と候補 handout のファイル名
        - `## 入力`：shape，dtype，値の範囲，各軸の意味
        - `## 出力`：shape，dtype，各要素の意味
        - `## 評価指標`：定義式，良い方向（大きいほど良いか小さいほど良いか），集計の単位
        - `## ベースライン`：比較の基準にする自明なベースライン（値は第3問で追記します）
        - `## 第27回までに残す保存物`：script，図，数値結果のファイル名

    2. 候補 handout に書かれていない値（データの個数，入力の次元，値の範囲など）があれば，自分で決めた値とその理由を，該当する見出しに書いてください。
    3. `## 評価指標` に，その指標が良い値になることが，テーマで解きたいことの何を意味するのかを 2 行以内で書いてください。学習に使う損失と評価指標が異なる場合は，異なる理由も書いてください。

2. 評価指標を計算する関数を実装し，手計算の結果と比べます。次の 4×4 の正解 mask `target` と予測 mask `pred` を使います。1 は前景，0 は背景を表します。

    ```python
    target = np.array([[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]])
    pred = np.array([[0, 0, 0, 0],
                     [0, 1, 1, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 0]])
    ```

    1. TP，FP，FN，TN を数え，pixel accuracy，precision，recall，Dice，IoU を手計算してください。
    2. `exercise/exc25_02.py` で，0 か 1 を要素とする配列 `pred` と `target` を受け取り，TP，FP，FN，TN を返す関数 `confusion_counts(pred, target)` を定義してください。さらに，その 4 つの値から 5 つの指標を計算し，`{"accuracy": ..., "precision": ..., "recall": ..., "dice": ..., "iou": ...}` という辞書で返す関数 `binary_metrics(pred, target)` を定義してください。2 次元の mask でも 1 次元の label でも使えるようにしてください。
    3. 2 の関数で 1 の値を計算し，手計算の結果と比較してください。
    4. `pred` と `target` がどちらもすべて 0 の場合（`np.zeros((4, 4), dtype=int)`）に，precision，recall，Dice，IoU の計算がどうなるかを予想してください。実行して確かめ，エラーや `nan` になる場合は，その場合の値の定義を決めて関数を直してください。決めた定義とその理由をコメントに書いてください。
    5. precision と recall の定義式を F1 の式に代入し，F1 が $2TP/(2TP+FP+FN)$ に等しいこと，つまり Dice と常に等しいことを示してください。

3. 前景が小さい mask について自明なベースラインの値を求め，評価指標の読み方を確かめます。次の 10 枚の正解 mask を使います。

    ```python
    target = np.zeros((10, 32, 32), dtype=np.int64)
    target[:, 12:18, 12:18] = 1
    ```

    1. 全画素を背景と予測する場合（`np.zeros_like(target)`）と，全画素を前景と予測する場合（`np.ones_like(target)`）のそれぞれについて，10 枚の全画素をまとめて数えたときの TP，FP，FN，TN と，pixel accuracy，Dice を手計算してください。
    2. `exercise/exc25_03.py` に第2問の 2 つの関数を書き，1 の値を確かめてください。
    3. `rng = np.random.default_rng(0)` を使い，各画素を確率 0.5 で前景とする予測 `(rng.random(target.shape) < 0.5).astype(np.int64)` を作って，pixel accuracy と Dice を計算してください。
    4. 全画素を背景と予測すると，pixel accuracy は高いのに Dice が 0 になる理由を，1 で求めた TP，FP，FN，TN を使って説明してください。
    5. 選択テーマについて自明なベースラインを 1 つ決め，その評価指標の値を計算してください。例えば，segmentation では全画素を背景とする予測，tagging ではクラスごとに学習データで多い方の値（0 か 1）を常に予測する方法，representation learning では学習前の encoder，PCA では平均だけを使う再構成（$k=0$），効率化と export では trace 前の model の計測時間を使えます。値と計算方法を `exercise/exc25_01.md` の `## ベースライン` に追記してください。

4. 1 バッチでの過学習を使って，学習の処理に誤りがないかを確かめます。次の 8 個のデータと model を使います。正解の label は乱数で決めたものであり，入力との間に規則はありません。

    ```python
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(0, 1, size=(8, 20)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 3, size=8))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 3))
    ```

    1. 学習前の `nn.CrossEntropyLoss` の値を，解説の第5節の理論値を使って予想してください。
    2. `exercise/exc25_04.py` で学習前の損失を計算し，1 の予想と比較してください。
    3. `torch.optim.Adam(model.parameters(), lr=1e-2)` を使い，同じ 8 個のデータで 200 回更新してください。各回の損失を記録し，最後の損失と，8 個のデータに対する正解率を表示してください。
    4. 正解の label を更新のたびに並べ替えてしまう誤りを入れた学習を行ってください。具体的には，`torch.manual_seed(0)` から model を作り直し，損失を `loss_fn(model(x), y[torch.randperm(8)])` で計算して，3 と同じ条件で 200 回更新してください。3 と 4 の損失の推移を 1 枚の図に描き，`outputs/exc25/exc25_04.png` に保存してください。
    5. 4 で損失が 0 に近づかない理由を説明してください。また，入力との間に規則がない label でも 3 で損失がほぼ 0 になったことから，1 バッチでの過学習が成功しただけでは model の良さを示せない理由を説明してください。

5. 選択テーマの最小実装を作り，shape と動作を確かめます。

    1. 候補 handout の基礎レベル第1問と第2問を，候補 handout が指定するファイルに実装してください。
    2. データの個数と主な配列の shape を，`exercise/exc25_01.md` の内容から予想してください。主な配列とは，model の入力と出力，PCA では `X`，`Z`，`X_recon`，効率化と export では計測に使う入力と model の出力です。予想を `assert` で書いて実行し，一致しなかった場合は理由をコメントに書いてください。
    3. 学習を行うテーマ（segmentation，tagging，representation learning）では，学習前の損失を解説の第5節の理論値から予想し，実際の値と比較してください。次に，dataset の先頭 4 サンプルだけを 1 つのバッチとして 200 回更新し，損失の推移を `outputs/exc25/exc25_05.png` に保存してください。損失が下がり続けない場合は，第4問と解説の第5節を参考に原因を調べてください。
    4. 学習を行わないテーマでは，次を確かめてください。PCA では，`Vt @ Vt.T` が単位行列に近いことと，第1主成分だけで再構成したときの二乗誤差の和が 2 番目以降の特異値の 2 乗の和（2 次元のデータでは 2 番目の特異値の 2 乗）に等しいことを確かめてください。効率化と export では，trace 前後の出力が `torch.allclose` で一致することと，同じ条件で 5 回計測した時間の平均と標準偏差を確かめてください。
    5. 確かめた内容と結果をコメントに書いてください。予想と異なる結果が出た場合は，入力，model，損失，評価のどこを疑ったかも書いてください。

6. 第26回以降に同じ処理を再実行できるように，実行記録を残します。

    1. `exercise/exc25_06.md` を作成し，次の 4 つの見出しを順に書いてください。

        - `## 実行したコマンド`：作業リポジトリのルートで実行するコマンドを，実行した順に書く
        - `## 生成された保存物`：ファイル名と内容（図，数値，model など）
        - `## まだ通っていない処理`：候補 handout の基礎レベルのうち，未実装またはエラーになる処理と，エラーメッセージの最終行
        - `## 次回最初に確認すること`：第26回の最初に確認すること

    2. 新しいターミナルを開き，`## 実行したコマンド` に書いたコマンドだけを使って最小実装を実行し直してください。同じ保存物が作られることを確かめ，足りなかった手順があれば追記してください。
    3. 記録したコマンドだけで再実行できることを確かめておくと，第26回と第27回の作業で何が楽になるかを説明してください。

### 発展レベル（3問）
1. 第26回で比較する条件を決めます。

    1. 候補 handout の発展レベルと解説の第2節を参考に，比較する条件を 2 つ選んでください。例えば，threshold，評価指標，batch size，前処理，model の幅，学習率，計測回数などから選べます。
    2. `exercise/exc25_07.md` に `## 比較条件の候補` を作り，条件ごとに「変える値（2〜3 通り）」「固定する条件」「評価指標の変化の予想」を書いてください。
    3. それぞれの条件を比較する価値がある理由を，最小実装で気づいたこと（損失の下がり方，評価指標の値，計測のばらつきなど）と結びつけて説明してください。

2. threshold を変えると評価指標がどう変わるかを調べます。8 個のサンプルについて，model が出力した正例の確率 `probs` と正解 `target` が次のように得られたとします。

    ```python
    probs = np.array([0.9, 0.8, 0.65, 0.55, 0.45, 0.35, 0.2, 0.1])
    target = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    ```

    1. threshold を 0.3，0.5，0.7 とし，それぞれで予測 `(probs >= threshold).astype(int)` を作って，precision，recall，F1 を手計算してください。
    2. `exercise/exc25_08.py` で，threshold を `np.linspace(0, 1, 101)` の 101 通りに変えて precision，recall，F1 を計算してください。F1 は，基礎レベル第2問の 5 で示した $2TP/(2TP+FP+FN)$ で計算してください。予測に 1 が 1 つもない threshold での precision には，基礎レベル第2問で決めた定義を使ってください。3 つの値を横軸 threshold の 1 枚の図に描き，`outputs/exc25/exc25_08.png` に保存してください。
    3. threshold を上げると precision と recall がそれぞれどう変わるかを，FP と FN の変化から説明してください。
    4. 正例の見逃しを避けたい場合と，誤検出を避けたい場合のそれぞれで，0.3，0.5，0.7 のどれを選ぶかを理由とともに答えてください。
    5. Dice と IoU の間に $\mathrm{Dice}=2\,\mathrm{IoU}/(1+\mathrm{IoU})$ が成り立つことを，TP，FP，FN を使って示してください。101 通りの threshold について両辺を計算し，差の絶対値の最大値を表示して確かめてください。

3. 役割に応じて，第26回の比較を進める準備をします。自分の学年に対応する項目に取り組み，`exercise/exc25_09.md` に書いてください。

    1. B4：発展レベル第1問の条件を切り替えるために変更するファイル，関数，引数を，`## 比較の準備` に 3 行以内で書いてください。
    2. M：B4 が第26回で比較を再実行できるように，必要な実行コマンドと変更箇所を `## 再実行の手順` に 3 行以内で整理してください。B4 の `## 比較の準備` を読み，不足している情報があれば `## 不足している情報` に書いてください。
    3. 学年にかかわらず：第26回の比較結果を読み誤りやすい点を 1 つ予想し，`## 読み誤りやすい点` に書いてください。例えば，乱数の seed による差を，条件の違いによる差と取り違える場合が考えられます。

## 詰まったときに見る資料
- [`advanced/README.md`](advanced/README.md)
- [`13-sp-projects/01_vad.md`](13-sp-projects/01_vad.md)
- [`24-experiment-code-engineering.md`](24-experiment-code-engineering.md)
- PyTorch docs: [`torch.nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [`torch.jit.trace`](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html)
