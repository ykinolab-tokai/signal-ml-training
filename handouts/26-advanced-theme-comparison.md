# 第26回 年度別発展テーマ：比較実験

## この回の目標

発展テーマで 2 つ以上の条件を比較し，失敗の原因を切り分けて改善方針を記録できる。

## 解説

この回では，第25回で作った最小実装を使い，条件を変えた比較実験を行う。
比較の結果には，条件の違いによる差だけでなく，乱数による差も含まれる。
この 2 つを区別して結果を読むことと，うまくいかなかった原因を切り分けることが，この回の中心である。
スクリプトの先頭には，次の処理を書く。

```python
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

figure_dir = Path("outputs/exc26")
figure_dir.mkdir(parents=True, exist_ok=True)
```

`csv` は，CSV ファイルを読み書きするための Python の標準ライブラリである。
以下のコード例は，このコードに続けて書いて実行する。

### 1. 比較実験の設計

#### 1 つずつ条件を変える

**比較実験 (comparative experiment)** では，条件 A と条件 B の結果の差から，条件の違いが結果に与える影響を調べる。
結果の差を条件の違いによるものと説明するには，比べたい条件以外をすべてそろえる必要がある。
例えば，model の幅と学習率を同時に変えると，結果に差が出ても，どちらの変更による差かを判断できない。
そこで，1 回の比較では 1 つの条件だけを変え，それ以外の条件は固定する。

| | 条件 A | 条件 B |
| --- | --- | --- |
| 変える条件（学習率） | `1e-3` | `1e-2` |
| 固定する条件 | データ，model の構造（隠れ層の幅 16），更新回数 300，seed 0〜4，評価データ | 条件 A と同じ |

ある要素を取り除いた条件と取り除かない条件を比べ，その要素の効果を調べる比較を **ablation** という。
例えば，augmentation（第20回）を使う条件と使わない条件の比較は ablation である。

#### 固定する条件を書き出す

結果に影響し得る条件には，次のものがある。

- データ：個数，分割，前処理
- model：構造，初期化
- 学習：損失関数，optimizer，学習率，更新回数，batch size
- 乱数：seed
- 評価：評価データ，評価指標，threshold

比較を始める前にこれらを書き出し，変える条件と固定する条件に分けて記録する。
第24回のように条件を config ファイルにまとめておくと，2 つの条件の違いを差分として確かめやすい。
書き出さなかった条件は，気づかないうちに変わっていても見落とされる。

### 2. 乱数による結果のばらつき

#### 乱数の発生源

学習の結果は，乱数の seed によって変わる。
model の初期値，データの生成と分割，ミニバッチの順序，augmentation などが乱数で決まるからである。
seed を 1 つだけ使って条件 A と B を比べると，条件の違いではなく，たまたま選んだ seed による差を見ている可能性がある。
評価の結果が良くても悪くても，それが seed による偶然でないことを確かめる必要がある。

#### 複数の seed での平均と標準偏差

そこで，各条件を複数の seed で実行し，結果の平均と標準偏差を比べる。
seed の数を $n$，各 seed での評価指標の値を $a_1,\ldots,a_n$ とすると，平均 $\bar{a}$ と標準偏差 $s$ は次の式で計算する。

$$
\bar{a}=\frac{1}{n}\sum_{i=1}^{n}a_i,\qquad
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(a_i-\bar{a}\right)^2}
$$

seed の数は 3〜10 程度と少ないことが多いため，標準偏差には第6回の不偏分散に基づく式（`ddof=1`）を使う。
次の例では，第25回の 2 次元の点の分類で，学習率 `1e-3` と `1e-2` を seed 0〜4 で比べる。
データは固定し，model の初期値だけを seed で変える。

```python
def make_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, 2)).astype(np.float32)
    y = (x[:, 0] * x[:, 1] > 0).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)

x_train, y_train = make_data(200, seed=0)
x_test, y_test = make_data(1000, seed=1)

def train_model(seed, hidden, lr, steps=300):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model

def accuracy(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()

rows = []
for lr in [1e-3, 1e-2]:
    for seed in range(5):
        model = train_model(seed, hidden=16, lr=lr)
        rows.append({"lr": lr, "seed": seed, "accuracy": accuracy(model, x_test, y_test)})

for lr in [1e-3, 1e-2]:
    values = np.array([row["accuracy"] for row in rows if row["lr"] == lr])
    print("lr={}: mean={:.3f}, std={:.3f}".format(lr, values.mean(), values.std(ddof=1)))
```

```text
lr=0.001: mean=0.907, std=0.045
lr=0.01: mean=0.981, std=0.001
```

`rows` は，1 回の実行の条件と結果を 1 つの辞書にまとめ，それを実行の回数だけ並べたリストである。
`train_model` は呼ばれるたびに `torch.manual_seed(seed)` を実行するため，同じ seed なら何回目の呼び出しでも同じ初期値になる。
学習率 `1e-3` では，300 回の更新では学習が終わりきらず，seed によって正解率が 0.83〜0.94 程度に散らばる。
学習率 `1e-2` では，5 つの seed がすべて約 0.98 になり，ばらつきも小さい。

### 3. 結果の記録と可視化

#### CSV への保存

比較の結果は，1 回の実行を 1 行とし，条件，seed，評価指標を列とする CSV ファイルに保存する。
CSV に保存しておくと，学習をやり直さずに集計や図を作り直せる。

```python
csv_path = figure_dir / "session26_example_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lr", "seed", "accuracy"])
    writer.writeheader()
    writer.writerows(rows)

with open(csv_path, newline="") as f:
    loaded = list(csv.DictReader(f))
print(len(loaded), loaded[0]["lr"], loaded[0]["seed"])  # 10 0.001 0
print(type(loaded[0]["accuracy"]))                      # <class 'str'>
```

`with open(...) as f:` は，ファイルを開き，ブロックを抜けるときに自動で閉じる書き方である。
`"w"` は書き込み用に開くことを表し，`newline=""` は `csv` モジュールで改行を正しく扱うための指定である。
`csv.DictWriter` は，`fieldnames` に指定した列名の順に，辞書を 1 行ずつ書き込む。
`writeheader()` で 1 行目に列名を書き，`writerows(rows)` で辞書のリストをまとめて書き込む。
`csv.DictReader` は，各行を列名をキーとする辞書として読み込む。
読み込んだ値はすべて文字列なので，計算に使うときは `float()` や `int()` で変換する。

#### 平均と標準偏差の図

条件ごとの平均を点で，標準偏差を誤差棒で描くと，差とばらつきを同時に比べられる。
各 seed の値も重ねて描くと，1 つの seed だけが外れているのか，全体が散らばっているのかが分かる。

```python
labels = ["0.001", "0.01"]
fig, ax = plt.subplots()
for i, lr in enumerate(labels):
    values = np.array([float(row["accuracy"]) for row in loaded if row["lr"] == lr])
    ax.scatter(np.full(len(values), i), values, color="gray", alpha=0.5)
    ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="o", capsize=5, color="black")
ax.set_xticks([0, 1], labels=["lr=1e-3", "lr=1e-2"])
ax.set_ylabel("test accuracy")
ax.grid(True)
fig.savefig(figure_dir / "session26_example_errorbar.png")
plt.show()
plt.close(fig)
```

`ax.errorbar(x, y, yerr=e)` は，点 $(x, y)$ と，$y-e$ から $y+e$ までの誤差棒を描く。
`fmt="o"` は点の形，`capsize` は誤差棒の端の横線の長さを指定する。
`np.full(len(values), i)` は，すべての要素が `i` の配列であり，各 seed の値を同じ横位置に並べるために使う。
`ax.set_xticks(位置, labels=名前)` は，横軸の目盛りの位置と名前を指定する。

### 4. 失敗分析

#### 症状と確認する箇所

失敗の原因を探すときは，最終的な評価指標だけでなく，途中の値も確かめる。
どこまで正しく動いているかを切り分けると，次に直す箇所が決まる。
よく現れる症状と，確認する箇所は次のとおりである。

| 症状 | 疑う箇所 | 確認方法 |
| --- | --- | --- |
| 損失が `nan` や `inf` になる | 学習率が大きすぎる，`log(0)` の計算 | 学習率を 1/10 にする，入力と出力の値の範囲を表示する |
| 学習用データの損失が下がらない | `zero_grad` の書き忘れ，optimizer に渡したパラメータ，学習率，正解の対応のずれ | 1 バッチでの過学習（第25回） |
| 学習用データでは良いが評価用データで悪い | 過学習，前処理の統計量の漏れ | 学習曲線（第19回），前処理の手順（第20回） |
| 評価指標の値が不自然に高い | クラスの偏り，評価指標の実装の誤り | 自明なベースライン（第25回），手計算できる例 |
| 実行するたびに結果が変わる | seed の指定漏れ | 同じ seed で 2 回実行して比べる（第24回） |

#### サンプルごとの誤差を調べる

評価指標はデータ全体の平均なので，どのサンプルで失敗しているかは分からない。
サンプルごとの損失を計算し，損失の大きいサンプルを調べると，失敗の傾向が見えることがある。

```python
model = train_model(seed=0, hidden=16, lr=1e-2)
loss_fn = nn.CrossEntropyLoss(reduction="none")
with torch.no_grad():
    losses = loss_fn(model(x_test), y_test)
print(losses.shape)  # torch.Size([1000])

worst = torch.argsort(losses, descending=True)[:20]
print(x_test[worst[:5]].abs().min(dim=1).values)  # tensor([0.0048, 0.0003, 0.0030, 0.0137, 0.0003])

fig, ax = plt.subplots()
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="coolwarm", s=5)
ax.scatter(x_test[worst, 0], x_test[worst, 1], marker="x", color="black", label="20 largest losses")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.set_aspect("equal")
fig.savefig(figure_dir / "session26_example_worst.png")
plt.show()
plt.close(fig)
```

`reduction="none"` を指定した損失関数は，平均をとらずにサンプルごとの損失を返す。
`torch.argsort(losses, descending=True)` は，損失の大きい順にサンプルの番号を並べた tensor を返す。
`x_test[worst[:5]].abs().min(dim=1).values` は，損失の大きい 5 サンプルについて，$|x_1|$ と $|x_2|$ の小さい方，つまり最も近い座標軸までの距離である。
どれも 0.02 未満であり，損失の大きいサンプルは，クラスの境界である座標軸のすぐ近くにある。
このように，失敗したサンプルに共通する性質を見つけると，「境界付近のデータを増やす」のような次の改善の候補が立てられる。

### 5. 改善の判断

#### 差とばらつきを比べる

2 つの条件の平均の差が，各条件の標準偏差より十分大きければ，その差は seed による偶然では説明しにくい。
逆に，平均の差が標準偏差と同じ程度以下なら，seed を変えると順位が入れ替わる可能性がある。
平均そのものの推定のばらつきは，**標準誤差 (standard error)** $s/\sqrt{n}$ で表され，seed の数 $n$ を増やすほど小さくなる（第6回の標本平均のばらつき）。
平均と標準偏差だけでなく，各 seed の値の分布も見て，一方の条件のすべての値が他方を上回っているかなども確かめる。

#### 評価指標以外の観点

改善の判断では，評価指標の平均だけでなく，次の観点も考える。

- seed によるばらつき（安定して同じ結果が得られるか）
- 計算時間と使用するメモリ
- 実装と説明の簡単さ

どの観点を優先したかを記録しておくと，第27回で条件を整理するときや，第28回のデモで判断の理由を説明するときに使える。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第26回の第1問であれば，`exercise/exc26_01.py` とします。
文章で答える問題は，同じ番号の `.md` ファイルに書いてください。第3問であれば，`exercise/exc26_03.md` とします。
各 `.py` ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc26`）を作成する処理を書いてください。
図と CSV は `figure_dir` に，問題番号を含むファイル名で保存してください。第1問の図であれば，`outputs/exc26/exc26_01.png` とします。

この回は複数条件の比較を残すため，`exercise/exc26_03.md` を第26回の比較記録とします。
候補 handout が比較レポート（例：`advanced_image_segmentation_report.md`）を指定している場合は，結果の解釈をそのレポートに書き，比較記録からはそのファイル名と見出しを参照してください。
同じ数値や解釈を二重に書かないでください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。

### 基礎レベル（5問）
1. 乱数の seed によって結果がどれだけ変わるかを調べます。解説の第2節と同じ `make_data` で作ったデータ（学習用は `make_data(200, seed=0)`，評価用は `make_data(1000, seed=1)`）を使い，隠れ層の幅を 4 とした model `nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))` を，`torch.optim.Adam(model.parameters(), lr=1e-2)` で 300 回更新します。各更新では学習用データをすべて使います。

    1. `torch.manual_seed` に渡す seed を 0〜9 の 10 通りに変えたとき，評価用データの正解率がすべて同じ値になるかを予想してください。
    2. `exercise/exc26_01.py` で，10 通りの seed のそれぞれについて学習と評価を行い，seed，学習用データでの最後の損失，評価用データの正解率を表示してください。正解率の平均と標準偏差（`ddof=1`）も表示してください。
    3. 結果を `seed`，`train_loss`，`accuracy` の 3 列の CSV として `outputs/exc26/exc26_01.csv` に保存してください。また，横軸を seed，縦軸を正解率とした図を `outputs/exc26/exc26_01.png` に保存してください。
    4. 正解率が最も低い seed と最も高い seed（同じ値の seed が複数ある場合はそのうちの 1 つ）のそれぞれについて，学習後の `model[0].weight` を表示してください。この 4 行 2 列の行列の各行は，隠れ層の各ユニットが入力 $(x_1, x_2)$ のどの向きを見ているかを表します。2 つの seed で 4 つの行の向きを比べ，正解率が低い理由を説明してください。学習用データでの最後の損失も比べ，その理由が過学習ではないと言える根拠も答えてください。
    5. seed を 1 つだけ使って 2 つの条件を比べると，どのような誤った結論を導く可能性があるかを，この結果を使って説明してください。

2. 2 つの条件を複数の seed で比較し，差が意味のある大きさかを判断します。第1問と同じデータと学習の設定で，隠れ層の幅 4 と 16 を比べます。

    1. 幅 16 の正解率の平均と標準偏差が，幅 4 と比べてどうなるかを予想してください。
    2. `exercise/exc26_02.py` で，2 つの幅のそれぞれについて seed 0〜4 で学習と評価を行い，`hidden`，`seed`，`accuracy` の 3 列の CSV を `outputs/exc26/exc26_02.csv` に保存してください。
    3. 保存した CSV を `csv.DictReader` で読み込み，幅ごとに正解率の平均と標準偏差（`ddof=1`）を計算して表示してください。解説の第3節と同じように，平均と標準偏差の誤差棒に各 seed の値を重ねた図を描き，`outputs/exc26/exc26_02.png` に保存してください。
    4. 平均の差と各条件の標準偏差を比べ，幅の違いによる差と言えるかを判断してください。各 seed の値の分布（一方の条件のすべての値が他方を上回っているかなど）にも触れて，判断の根拠を説明してください。
    5. この比較で，学習率や更新回数を変えずに幅だけを変えたことが重要な理由を説明してください。

3. 選択テーマで比較する条件を決め，比較の計画を立てます。

    1. 第25回の発展レベル第1問で選んだ条件（選んでいない場合は候補 handout の発展レベルの条件）から 1 つを選び，条件 A と条件 B を決めてください。
    2. `exercise/exc26_03.md` を作成し，`## 比較条件`，`## 固定した条件`，`## 結果の予想` の 3 つの見出しを書いてください。`## 固定した条件` には，解説の第1節の一覧を使って，固定する条件をすべて書いてください。
    3. 各条件を実行する回数を決め，`## 比較条件` に書いてください。学習を行うテーマでは seed を 3 つ以上，PCA ではデータを作る seed を 3 つ以上，効率化と export では計測を 5 回以上とします。
    4. `## 結果の予想` に，評価指標がどちらの条件で良くなるかと，その理由を書いてください。

4. 第3問の計画に従って比較を実行し，結果を記録します。

    1. 選択テーマの script で条件 A と条件 B を実行し，1 回の実行を 1 行とする CSV を `outputs/exc26/exc26_04.csv` に保存してください。列には，条件，seed（または計測の番号），評価指標を含めてください。候補 handout が保存物のファイル名を指定している場合は，そのファイルも保存してください。
    2. 条件ごとの評価指標の平均と標準偏差を計算し，第2問と同じ形式の図を `outputs/exc26/exc26_04.png` に保存してください。
    3. `exercise/exc26_03.md` に `## 結果` を追加し，CSV と図のファイル名，条件ごとの平均と標準偏差を書いてください。
    4. `## 解釈` を追加し，`## 結果の予想` と一致したかを書いてください。平均の差が seed や計測によるばらつきと比べて十分大きいかも判断し，その根拠を書いてください。

5. 失敗したサンプルを調べ，原因の仮説を立てます。

    1. `exercise/exc26_05.py` で，第1問で正解率が最も低かった seed を使って第1問と同じ学習を行い，その model について，評価用データのサンプルごとの損失を `nn.CrossEntropyLoss(reduction="none")` で計算してください。評価用データの散布図に，損失の大きい 20 サンプルを印を付けて重ね，`outputs/exc26/exc26_05.png` に保存してください。
    2. 1 の図を，解説の第4節の図（正解率約 0.98 の model）と比べ，損失の大きいサンプルの位置の違いを説明してください。
    3. 選択テーマでも，サンプルごとの値（segmentation では画像ごとの Dice，tagging ではサンプルごとの損失，representation learning では pair ごとの cosine similarity，PCA ではサンプルごとの再構成誤差，効率化と export では計測ごとの時間）を計算し，最も悪い 3 つを調べてください。
    4. `exercise/exc26_03.md` に `## 失敗例と原因の仮説` を追加し，3 で調べたサンプルに共通する性質と，原因の仮説を書いてください。仮説を確かめるために見た値（shape，値の範囲，損失の推移など）も，解説の第4節の表を参考にして書いてください。
    5. `## 次に直すこと` を追加し，仮説に基づいて次に試すことを 1 つ書いてください。

### 発展レベル（3問）
1. seed の数と，平均の推定のばらつきの関係を調べます。

    1. `exercise/exc26_06.py` で，第1問の CSV を読み込み，seed 0〜4 の 5 個の値と，seed 0〜9 の 10 個の値のそれぞれから，正解率の平均，標準偏差，標準誤差 $s/\sqrt{n}$ を計算してください。
    2. seed の数を 5 個から 10 個に増やすと，標準誤差がおよそ何倍になるかを予想してから，1 の結果と比べてください。
    3. 第2問の 2 つの条件について，平均の差と，それぞれの標準誤差を表示してください。差が標準誤差と比べて十分大きいかを判断してください。
    4. 選択テーマの結果についても同じ計算を行い，第27回で整理する条件を 1 つ選んでください。判断の理由を `exercise/exc26_03.md` の `## 第27回で整理する条件` に書いてください。評価指標の平均だけでなく，ばらつき，計算時間，実装の簡単さのうち重視した観点も書いてください。

2. 追加で 1 つの条件だけを変え，比較の結論が変わらないかを確かめます。

    1. 第4問で固定した条件から 1 つを選び，別の値に変えてください（例：更新回数，データの個数，評価データを作る seed）。
    2. 変えた条件のもとで条件 A と条件 B を同じ回数ずつ実行し，CSV を `outputs/exc26/exc26_07.csv` に保存してください。
    3. 条件 A と条件 B のどちらが良いかという結論が，第4問と同じかを確かめてください。結論が変わった場合は，その理由の仮説を書いてください。
    4. 結果を `exercise/exc26_03.md` の `## 追加の確認` に書いてください。

3. 比較結果の読み取りが過大な解釈になっていないかを，役割に応じて確かめます。自分の学年に対応する項目に取り組み，`exercise/exc26_08.md` に書いてください。

    1. M：B4 の `exercise/exc26_03.md` を読み，比較結果の読み取りで過大な解釈になりそうな点を 2 つ挙げてください。それぞれについて，確かめるために必要な追加の実験や確認を提案してください。
    2. B4：M の指摘に対して，追加の実験を行うかどうかと，その理由を書いてください。追加の実験を行った場合は，その結果のファイル名も書いてください。
    3. 学年にかかわらず：「今回の比較で言えること」と「今回の比較だけでは言えないこと」を 1 つずつ書いてください。

## 詰まったときに見る資料
- [`25-advanced-theme-framing.md`](25-advanced-theme-framing.md)
- [`06-probability.md`](06-probability.md)
- Python docs: [`csv`](https://docs.python.org/3/library/csv.html)
- Matplotlib docs: [`Axes.errorbar`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html)
