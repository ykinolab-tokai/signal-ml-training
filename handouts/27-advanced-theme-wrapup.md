# 第27回 年度別発展テーマ：整理

## この回の目標

発展テーマの実装を再利用できる形に整理し，再実行の手順と第28回で説明する対象を記録できる。

## 解説

この回では，第25回と第26回で作った実装と結果を，他の学生や将来の自分が再利用できる形に整理する。
整理では，コードの書き方を変えても結果が変わらないことを確かめ，再実行に必要な情報を記録する。
スクリプトの先頭には，次の処理を書く。

```python
import numpy as np
import torch
from torch import nn
from pathlib import Path

output_dir = Path("outputs/exc27")
output_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. 再利用可能な実装資産

#### 再利用できる状態

他の人が作者に質問しなくても実行でき，同じ結果を得られるコードと説明のまとまりを，**再利用可能な実装資産 (reusable asset)** と呼ぶ。
再利用するには，少なくとも次のことが分かる必要がある。

- 入力：どのデータを，どこに置けばよいか
- 実行コマンド：作業リポジトリのルートで何を実行すればよいか
- 出力：どのファイルが作られ，それぞれが何を表すか
- 前提条件：Python 環境，乱数の seed，所要時間，OS 側で必要な準備

自分の PC で一度動いたコードがあるだけでは，これらが分からず，他の人は再利用できない。

#### 整理の順序

整理は，次の順に進める。

1. 残すファイルを決める
2. スクリプトを関数に整理する
3. 整理の前後で結果が一致することを確かめる
4. 再実行手順を書き，新しい clone で確かめる
5. 共有する資産を選ぶ

不要なファイルを消すのは，残すファイルと再実行の条件が決まった後に行う。
先に消すと，再実行に必要だったファイルを失うおそれがある。

#### 残すファイルの確認

作業リポジトリにあるファイルは，Git のコマンドで確認できる。
次のコマンドは，作業リポジトリのルートで実行する。

```bash
git status --short
git ls-files
git grep -n "exc26_04.csv"
```

`git status --short` は，変更のあるファイルを 1 行ずつ表示する。
行頭の `??` は Git が管理していないファイル，`M` は commit 後に変更されたファイルを表す。
`git ls-files` は，Git が管理しているファイルの一覧を表示する。
`git grep -n 文字列` は，Git が管理しているファイルから文字列を探し，ファイル名と行番号を表示する。
あるファイルを消してよいか迷うときは，そのファイル名を `git grep` で探し，読み込んでいるスクリプトがないかを確かめる。

### 2. スクリプトの整理

#### 関数への分割

上から順に処理を並べただけのスクリプトは，一度実行するには手軽だが，条件を変えて実行し直したり，一部の処理を別のスクリプトで使ったりしにくい。
次のコードは，直線 $y=2x+1$ に雑音を加えたデータに，第17回の方法で直線を当てはめる整理前のスクリプトである。

```python
torch.manual_seed(0)
x = torch.linspace(0, 1, 50).reshape(-1, 1)
y = 2 * x + 1 + 0.1 * torch.randn(50, 1)
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.MSELoss()
for step in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
print(loss.item())  # 約 0.0113
```

このスクリプトを，処理のまとまりごとに関数へ分ける。
各関数は，必要な値を引数で受け取り，結果を返り値で返すようにする。
関数の外の変数を直接読み書きしないようにすると，関数を別の条件や別のスクリプトで使い回せる。

```python
def make_data(n, noise_std):
    """直線 y = 2x + 1 に正規乱数の雑音を加えたデータを作る。

    Args:
        n: サンプル数
        noise_std: 雑音の標準偏差
    Returns:
        x, y: どちらも shape (n, 1) の float32 の tensor
    """
    x = torch.linspace(0, 1, n).reshape(-1, 1)
    y = 2 * x + 1 + noise_std * torch.randn(n, 1)
    return x, y

def train(model, x, y, steps, lr):
    """平均二乗誤差を SGD で最小化し，最後の損失を返す。"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return loss.item()

def main():
    torch.manual_seed(0)
    x, y = make_data(50, noise_std=0.1)
    model = nn.Linear(1, 1)
    return train(model, x, y, steps=100, lr=0.5)

if __name__ == "__main__":
    print(main())  # 約 0.0113
```

`main()` は，データの作成，model の構築，学習を順に呼ぶ関数であり，スクリプト全体の処理の流れを表す。
学習率や更新回数は `main()` の中で引数として渡しているため，第24回の `argparse` で受け取った値に置き換えやすい。

#### docstring

関数の定義の直後に書いた文字列を **docstring** という。
docstring には，関数の目的，引数，返り値を書く。
tensor や配列を受け取ったり返したりする関数では，shape と dtype も書いておくと，使う人が関数の中を読まずに済む。
対話モードの Python で `help(make_data)` を実行すると，関数の引数と docstring が表示される。
docstring は `"""` で囲んだ複数行の文字列として書くことが多い。

#### `if __name__ == "__main__":`

Python は，ファイルを読み込むときに，そのファイルの名前を変数 `__name__` に入れる。
`python exercise/exc27_02.py` のようにファイルを直接実行すると，`__name__` は `"__main__"` になる。
一方，別のファイルから `import exc27_02` で読み込むと，`__name__` は `"exc27_02"` になる。
このため，`if __name__ == "__main__":` の下に書いた処理は，ファイルを直接実行したときだけ実行される。
この書き方にしておくと，第24回の test や第28回の walkthrough で関数だけを読み込んでも，学習が始まらない。

### 3. 整理の前後の結果の一致

#### 同じ seed で同じ結果になることを確かめる

処理の結果を変えずにコードの構造だけを整えることを **リファクタリング (refactoring)** という。
リファクタリングでは，同じ seed なら整理の前後で結果が完全に一致するはずである。
一致を確かめれば，整理の途中で処理を書き間違えていないことが分かる。

```python
print(loss.item() == main())  # True
```

`loss.item()` は整理前のスクリプトの最後の損失であり，`main()` は整理後の同じ値である。
同じ順序で同じ計算をしていれば，浮動小数点数の値も完全に一致するので，`==` で比べられる。
計算の順序を変えた場合など，丸め誤差の範囲で値が異なることが正しい場合は，`torch.allclose` や `np.isclose` で比べる。

#### 乱数を使う順序

`torch.randn` や `nn.Linear` の初期化は，同じ乱数生成器から順に乱数を取り出す。
このため，整理によって乱数を使う処理の順序が変わると，seed が同じでも結果が変わる。

```python
def main_swapped():
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    x, y = make_data(50, noise_std=0.1)
    return train(model, x, y, steps=100, lr=0.5)

print(main(), main_swapped())  # 約 0.0113 約 0.0078
```

`main_swapped()` は，model を作る処理とデータを作る処理の順序を入れ替えただけである。
しかし，model の初期値とデータの雑音に使われる乱数が入れ替わるため，最後の損失が変わる。
このような結果の変化は，処理の誤りではないが，整理の前後の比較を難しくする。

データの乱数を model の初期化と切り離すには，データ専用の乱数生成器 `torch.Generator` を使う。

```python
def make_data_with_generator(n, noise_std, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, 1, n).reshape(-1, 1)
    y = 2 * x + 1 + noise_std * torch.randn(n, 1, generator=generator)
    return x, y
```

`torch.randn(..., generator=generator)` は，`torch.manual_seed` で決まる共通の乱数生成器ではなく，`generator` から乱数を取り出す。
第25回と第26回の `make_data` のように NumPy の `np.random.default_rng(seed)`（第6回）を使う場合も，データの乱数は model の初期化と切り離される。
ただし，乱数の取り出し方を変えると，データの値そのものが変わるため，整理前の結果とは一致しなくなる。
乱数の扱いを変える変更は，リファクタリングとは別の変更として行い，結果が変わったことを記録する。

### 4. 再実行手順の記録

#### 再実行手順に書く内容

再実行手順には，次の項目を書く。

| 項目 | 書く内容 |
| --- | --- |
| 目的 | 何を確かめるための実行か |
| 実行環境 | Python 環境の準備方法（`uv sync --locked`），OS 側で必要な準備 |
| 入力 | 入力データの場所と作り方 |
| 実行コマンド | 作業リポジトリのルートで実行するコマンド |
| 出力 | 作られるファイル名と，それぞれの内容 |
| 所要時間 | 自分の PC での実行時間 |
| 既知の制限 | うまくいかない条件，確かめていない点 |

コマンドは，説明の文章ではなく，そのままコピーして実行できる形で書く。
自分の PC の絶対パス（例：`/home/ユーザー名/...`）を書くと，他の人の PC では動かないため，作業リポジトリのルートからの相対パスで書く。

#### 新しい clone での確認

自分の作業リポジトリでは，commit していないファイルや，以前の実行で作られたファイルが残っている。
このため，手順に書き漏らした操作があっても，自分の PC では気づかないことがある。
そこで，作業リポジトリを別のディレクトリに clone し直し，手順に書いた操作だけで実行できるかを確かめる。

```bash
cd ~/workspace
git clone REPOSITORY_URL signal-ml-work-check
cd signal-ml-work-check
uv sync --locked
source .venv/bin/activate
python advanced_image_segmentation_demo.py
```

`REPOSITORY_URL` は，[第1回](01-environment-and-workflow.md#作業ディレクトリの作成とリポジトリの-clone)と同じく，作業リポジトリの **Code** → **SSH** からコピーした URL に置き換える。
最後の行は，再実行手順の実行コマンドに置き換える。
clone されるのは push 済みの内容だけなので，確認の前に，残すファイルを commit して push しておく。
別の仮想環境が有効になっている場合は，`deactivate` で抜けてから `source .venv/bin/activate` を実行する。

### 5. 残すもの・共有するもの

#### Git で管理するもの

[共通手順](README.md#作業場所と保存先)のとおり，コードは `exercise/` や候補 handout が指定する場所に，図や数値の結果は `outputs/` に置き，作業リポジトリで commit する。
`.venv/` のように各自の PC で作り直せる環境は，Git で管理しない。
再実行で作り直せる大きな中間ファイル（学習途中の model など）は，commit する前に，結果の根拠として必要かを考える。
commit する前には `git status` と `git diff --cached` で，意図したファイルだけが含まれていることを確かめる。

#### 共有する資産を選ぶ

個人の試行錯誤の記録と，他の学生が再利用できる資産は，価値が異なる。
例えば，評価指標を計算する関数，shape を確かめる関数，結果を CSV に保存する関数は，他のテーマでも使える可能性がある。
共有する資産の候補には，入力と出力が docstring に書かれていること，手計算できる例で動作を確かめてあること，特定のテーマのファイル名や絶対パスに依存しないことが求められる。
教材リポジトリへの反映は，第28回で候補を整理したうえで，授業の指示に従って行う。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第27回の第2問であれば，`exercise/exc27_02.py` とします。
文章で答える問題は，同じ番号の `.md` ファイルに書いてください。第1問であれば，`exercise/exc27_01.md` とします。
各 `.py` ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `output_dir`（`outputs/exc27`）を作成する処理を書いてください。
選択テーマの script は，候補 handout が指定するファイルを整理してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。

### 基礎レベル（5問）
1. 第25回と第26回で作ったファイルを棚卸しし，残すファイルを決めます。

    1. 作業リポジトリのルートで `git status --short` と `git ls-files` を実行し，第25回と第26回で作った script，記録，保存物を一覧にしてください。
    2. `exercise/exc27_01.md` を作成し，`## 残すファイル`，`## 不要な一時ファイル`，`## 再実行に必要な条件` の 3 つの見出しを書いてください。`## 残すファイル` の各ファイルには，残す理由（再実行に必要，結果の根拠，第28回の説明に使う など）を 1 行で添えてください。
    3. `## 不要な一時ファイル` に挙げた各ファイルについて，`git grep -n "ファイル名"` で読み込んでいるスクリプトがないかを確かめ，結果を書き添えてください。
    4. `## 再実行に必要な条件` に，Python 環境，入力データ，乱数の seed，実行する順序を書いてください。
    5. 不要なファイルを消す前に残すファイルを決める理由を説明してください。

2. 上から順に処理を並べただけのスクリプトを関数に整理し，整理の前後で結果が一致することを確かめます。次のスクリプトを整理前のコードとします。

    ```python
    torch.manual_seed(0)
    x = torch.randn(200, 2)
    y = (x[:, 0] * x[:, 1] > 0).long()
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    print(loss.item(), accuracy)
    ```

    1. `exercise/exc27_02.py` に，整理前のコードを関数 `run_flat()` として書き，最後の損失と正解率を返すようにしてください。
    2. 同じ処理を，`make_data(n)`，`build_model(hidden)`，`train(model, x, y, steps, lr)`，`evaluate(model, x, y)` の 4 つの関数と，それらを順に呼ぶ `main()` に整理してください。`make_data` は入力 `x` と正解 `y` を，`train` は最後の損失を，`evaluate` は正解率を返すようにしてください。各関数には，引数と返り値（tensor の場合は shape と dtype）を説明する docstring を書いてください。`main()` は `torch.manual_seed(0)` を実行してから 4 つの関数を呼び，最後の損失と正解率を返すようにしてください。ファイルの最後では，`if __name__ == "__main__":` の下で `run_flat()` と `main()` の結果を表示してください。
    3. `run_flat()` と `main()` が返す最後の損失と正解率が，`==` で完全に一致することを確かめてください。一致しない場合は，乱数を使う処理の順序を見直してください。
    4. `main()` をコピーした関数を 3 つ作り，それぞれの次の位置に `torch.randn(1)` を 1 行だけ追加します。(a) `torch.manual_seed(0)` の直後，(b) `make_data` と `build_model` の間，(c) `build_model` の直後（学習の直前）。3 つの関数の結果が `run_flat()` と一致するかを，それぞれ予想してください。実行して確かめ，一致した場合と一致しなかった場合の違いを，乱数を使う処理の順序から説明してください。
    5. `torch.Generator().manual_seed(1)` で作った乱数生成器を使って入力を生成する関数 `make_data_with_generator(n)` を作ってください。`main()` の `make_data` をこの関数に置き換えた版で，`make_data_with_generator` と `build_model` を呼ぶ順序を入れ替えても結果が変わらないことを確かめてください。また，この版の結果が `run_flat()` と一致しなくなる理由を説明してください。

3. 選択テーマの script を整理し，整理の前後で結果が変わらないことを確かめます。

    1. 整理の前に，第26回で第27回の整理対象に選んだ条件で script を実行し，評価指標の値（PCA では再構成誤差，効率化と export では出力の shape と trace 前後の一致）を記録してください。
    2. 第2問と同じ方針で，選択テーマの script を関数と `main()` に整理し，各関数に docstring を書いてください。第24回の `argparse` を使い，少なくとも出力先を `--out` で指定できるようにしてください。乱数を使うテーマでは，seed も `--seed` で指定できるようにしてください。
    3. 整理後の script を 1 と同じ条件で実行し，評価指標の値が 1 と一致することを確かめてください。計測時間のように実行ごとに変わる値は，一致を確かめる対象から除き，その理由をコメントに書いてください。
    4. 作業リポジトリのルートで `python -c "import ファイル名"`（ファイル名は `.py` を除いたもの）を実行し，学習や計測が始まらないことを確かめてください。始まってしまう場合は，`if __name__ == "__main__":` の位置を見直してください。

4. 再実行手順を書き，新しい clone で手順どおりに再実行できるかを確かめます。

    1. `exercise/exc27_04.md` を作成し，第3問で整理した script について，解説の第4節の表の 7 項目を，`## 目的`，`## 実行環境`，`## 入力`，`## 実行コマンド`，`## 出力`，`## 所要時間`，`## 既知の制限` の見出しで順に書いてください。
    2. 残すファイルを commit して push してから，解説の第4節の方法で，作業リポジトリを `~/workspace/signal-ml-work-check` に clone してください。
    3. clone したディレクトリで，`exercise/exc27_04.md` に書いた操作だけを使って script を実行してください。作られた出力の評価指標の値が，元の作業リポジトリでの値と一致することを確かめてください。
    4. 手順に足りなかった操作や，元の作業リポジトリにしかなかったファイル（commit していない入力データなど）があれば，`exercise/exc27_04.md` を直してください。元の作業リポジトリで実行したときに，それらに気づかなかった理由も説明してください。
    5. 確認が終わったら，`deactivate` で仮想環境を抜けてください。`~/workspace/signal-ml-work-check` は削除してかまいません。

5. 第28回の code walkthrough で説明する対象を選びます。

    1. 選択テーマの script から，walkthrough で説明する関数または処理を 1 つ選んでください。例えば，データの作成，model の forward，損失の計算，評価指標の計算などから選べます。
    2. `exercise/exc27_05.md` を作成し，`## 説明対象`，`## 入力`，`## 処理`，`## 出力`，`## 聞かれそうな質問` の 5 つの見出しを書いてください。`## 入力` と `## 出力` には shape と dtype を，`## 聞かれそうな質問` には質問を 2 つ以上書いてください。
    3. それを選んだ理由を，「その処理を理解すると，テーマのどの部分が分かるか」の形で 2 行以内で書いてください。

### 発展レベル（4問）
1. `__name__` の値を確かめ，`if __name__ == "__main__":` が必要な理由を説明します。

    1. 第2問の `exercise/exc27_02.py` の `import` 文の後に `print("__name__ =", __name__)` を追加してください。`python exercise/exc27_02.py` で実行したときに表示される値を予想してから，確かめてください。
    2. `exercise/exc27_06.py` に `from exc27_02 import build_model` と，`build_model(16)` のパラメータ数を表示するコードを書いてください。`python exercise/exc27_06.py` で実行したときに何が表示されるかを予想してから，確かめてください。`python exercise/exc27_06.py` で実行すると，実行したファイルのある `exercise/` が読み込み先の候補に入るため，`exercise/exc27_02.py` を読み込めます。
    3. `exercise/exc27_02.py` の `if __name__ == "__main__":` の行を一時的に消し，その下の処理のインデントを戻してから，2 をもう一度実行してください。何が起きるかを確かめたら，元に戻してください。
    4. 1〜3 の結果を使って，他のファイルから関数を読み込むときに `if __name__ == "__main__":` が必要な理由を説明してください。

2. 他の学生が再実行するときに詰まりそうな点を整理します。

    1. `exercise/exc27_07.md` に `## 詰まりそうな点` を作り，第4問の確認を踏まえて 2 つ挙げてください。それぞれに，確かめるためのコマンドまたは確認すべき保存物を対応づけてください。
    2. 選択テーマの候補 handout で，書かれていなかったために自分で決めた値や，分かりにくかった記述があれば，`## 候補 handout への修正提案` に，該当する箇所と修正案を書いてください。

3. 共有する資産の候補を選びます。

    1. 自分の実装から，他のテーマや来年度の学生にも役立つ補助コード（評価指標の関数，shape を確かめる関数，結果を CSV に保存する関数など），テンプレート，説明を探してください。
    2. `exercise/exc27_08.md` に `## 共有資産の候補` を作り，候補ごとに，ファイル名と関数名，他の学生にとっての利点，共有する前に直す必要がある点を書いてください。
    3. 個人のメモとして残すものと共有資産にするものを分けた基準を，解説の第5節を参考に説明してください。

4. 来年度に同じテーマを扱う場合の条件を，役割に応じて整理します。自分の学年に対応する項目に取り組み，`exercise/exc27_09.md` に書いてください。

    1. M：来年度に同じテーマを扱う場合に固定すべき条件と，年度ごとに変えてよい条件を分け，`## 固定する条件` と `## 変えてよい条件` に書いてください。それぞれに理由を 1 行で添えてください。
    2. B4：M の提案を読み，今年度に自分が詰まった点から見て追加すべき条件があれば，`## B4 からの追加` に書いてください。

## 詰まったときに見る資料
- [`24-experiment-code-engineering.md`](24-experiment-code-engineering.md)
- [`26-advanced-theme-comparison.md`](26-advanced-theme-comparison.md)
- Python docs: [`__main__`](https://docs.python.org/3/library/__main__.html)
- PyTorch docs: [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
