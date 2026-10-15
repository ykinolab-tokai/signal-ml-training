# 第28回 技術デモと walkthrough

## この回の目標

自分の実装と再実行の条件を説明し，他者が再利用できる知見を整理できる。

## 解説

この回は，第15回以降に自分が実装した script を，技術デモと code walkthrough で説明する統合回である。
[README の時間配分](../README.md#4-各回の標準的な時間配分)では，第14回と第28回のような統合回は demo の比重を増やし，60 分程度を確保してよいとしている。
説明では，スライドを作らず，コード・図・出力を見せる。
スクリプトの先頭には，次の処理を書く。

```python
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

output_dir = Path("outputs/exc28")
output_dir.mkdir(parents=True, exist_ok=True)
```

`json`，`platform`，`subprocess`，`sys`，`datetime` は Python の標準ライブラリであり，第3節で実行時の情報を記録するために使う。
以下のコード例は，このコードに続けて，作業リポジトリの中で実行する。

### 1. 技術デモの構成

#### 説明の順序

**技術デモ (technical demo)** では，実装した script を実行し，何ができるかを見せながら説明する。
聞き手は，入力と出力が分からないと，結果の良し悪しを判断できない。
このため，「何を作ったか」の前に，「どの入力に対して，どの処理を通し，何が出力されるか」を説明する。
そのうえで，結果，失敗例，残っている課題，再実行の方法の順に説明する。
入力と出力は，shape，dtype，保存したファイル名で具体的に示す。

#### 5 分の構成例

5 分で説明する場合は，例えば次のように時間を割り当てる。

| 時間 | 内容 | 見せるもの |
| --- | --- | --- |
| 0:00〜0:30 | 何をする script か（タスクと評価指標） | 問題設定（`exercise/exc25_01.md`） |
| 0:30〜1:30 | 入力と出力 | 入力の例の図，shape |
| 1:30〜3:00 | 主な処理 | 該当する関数のコード |
| 3:00〜4:00 | 結果 | 比較の図，評価指標の値 |
| 4:00〜5:00 | 失敗例，残っている課題，再実行の方法 | 失敗例の図，再実行手順（`exercise/exc27_04.md`） |

#### 実演が失敗したときの備え

デモの前に script を一度実行し，出力をファイルに保存しておく。
デモの最中に実行が失敗したり時間がかかったりした場合は，その場で直そうとせず，保存しておいた出力を見せる。
失敗した内容は，後で原因を調べられるように記録しておく。

### 2. code walkthrough

#### 読む順序

**code walkthrough** とは，コードの処理の流れを順にたどりながら，他の人に説明することである。
全行を読み上げる必要はない。
次の順に，処理の分かれ目，前提としている条件，失敗しやすい箇所を説明する。

1. 実行の入口（`main()` と，コマンドライン引数）
2. データの作成または読み込み
3. model の構造
4. 損失と評価指標の計算
5. 結果の保存

各段階では，入力と出力の shape を示す。
shape を示すと，聞き手は「どこでデータがどの形に変わるか」をコードを読まずに追える。

#### forward hook による shape の追跡

PyTorch では，`nn.Module` に **forward hook** を登録すると，その層の forward が実行された直後に，指定した関数を呼び出せる。
hook として登録する関数は，層 `module`，層への入力 `inputs`，層の出力 `output` の 3 つを引数に受け取る。
これを使うと，model のコードを書き換えずに，各層の出力の shape を表示できる。

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8 * 16 * 16, 2),
)

def print_shape(module, inputs, output):
    print(type(module).__name__, tuple(inputs[0].shape), "->", tuple(output.shape))

handles = [layer.register_forward_hook(print_shape) for layer in model]
with torch.no_grad():
    model(torch.zeros(4, 1, 32, 32))
for handle in handles:
    handle.remove()
```

```text
Conv2d (4, 1, 32, 32) -> (4, 8, 32, 32)
ReLU (4, 8, 32, 32) -> (4, 8, 32, 32)
MaxPool2d (4, 8, 32, 32) -> (4, 8, 16, 16)
Flatten (4, 8, 16, 16) -> (4, 2048)
Linear (4, 2048) -> (4, 2)
```

`for layer in model` は，`nn.Sequential` に並べた層を順に取り出す。
`register_forward_hook` は登録を取り消すための handle を返し，`handle.remove()` で hook を外せる。
`inputs` は forward に渡された引数のタプルなので，1 つ目の入力を `inputs[0]` で取り出している。
`type(module).__name__` は，層のクラス名（`Conv2d` など）を表す文字列である。
hook の関数が何も返さなければ，層の出力は変わらない。

第19回のように `nn.Module` を継承して作った model では，層が入れ子になっている。
`model.named_modules()` は，入れ子の内側も含めて，すべての層を名前とともに取り出す。
その中から，子を持たない層（`list(module.children())` が空の層）だけに hook を登録すると，計算を行う層の出力を順に表示できる。

```python
class TwoBlockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * 8 * 8, 2))

    def forward(self, x):
        return self.head(self.features(x))

net = TwoBlockNet()
handles = []
for name, module in net.named_modules():
    if len(list(module.children())) == 0:
        hook = lambda module, inputs, output, name=name: print(name, tuple(output.shape))
        handles.append(module.register_forward_hook(hook))
with torch.no_grad():
    net(torch.zeros(2, 1, 8, 8))
for handle in handles:
    handle.remove()
```

```text
features.0 (2, 4, 8, 8)
features.1 (2, 4, 8, 8)
head.0 (2, 256)
head.1 (2, 2)
```

`features.0` は，属性 `features` の中の 0 番目の層という意味の名前である。
`lambda 引数: 式` は，式の値を返す名前のない関数を作る書き方である。
`name=name` は，登録した時点の層の名前を，その hook の中で使うための指定である。

### 3. 再実行可能性の記録

#### 記録する情報

デモで見せた結果を他の人が再現するには，その結果を作ったときの条件が必要である。
次の情報を，実行のたびに自動で記録する。

- コードの版：Git の commit の識別子（**commit hash**）と，commit していない変更があるか
- 実行環境：Python，NumPy，PyTorch のバージョン
- 実行の条件：コマンドライン引数，乱数の seed
- 実行した日時

commit していない変更がある状態で実行すると，commit hash が同じでも，実際に実行したコードは commit の内容と異なる。
このため，commit hash と一緒に，commit していない変更があるかも記録する。

#### 実行情報を自動で保存する

```python
def git_output(args):
    result = subprocess.run(["git"] + args, capture_output=True, text=True, check=True)
    return result.stdout.strip()

changes = git_output(["status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)outputs"])
run_info = {
    "commit": git_output(["rev-parse", "HEAD"]),
    "dirty": changes != "",
    "python": platform.python_version(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "argv": sys.argv,
    "seed": 0,
    "time": datetime.now().isoformat(timespec="seconds"),
}
info_path = output_dir / "session28_example_run_info.json"
info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
print(len(run_info["commit"]))               # 40
print(run_info["python"], run_info["torch"])  # 例：3.12.3 2.7.1+cpu
```

`subprocess.run` は，Python からコマンドを実行する関数である。
`capture_output=True` と `text=True` を指定すると，コマンドの出力を文字列として `result.stdout` で受け取れる。
`check=True` を指定すると，コマンドが失敗したときに `CalledProcessError` が発生する。
Git で管理していないディレクトリで実行すると `git rev-parse HEAD` が失敗するため，このコードは作業リポジトリの中で実行する。

`git rev-parse HEAD` は，現在の commit hash（40 文字の英数字）を表示する。
`git status --porcelain` は，変更のあるファイルを 1 行ずつ表示し，変更がなければ何も表示しない。
`--untracked-files=no` を付けると，Git が管理していないファイル（新しく作ったファイルなど）は数えない。
`-- . ":(exclude)outputs"` は，作業リポジトリ全体から `outputs/` を除いた範囲だけを調べる指定である。
`outputs/` のファイルは実行のたびに書き換わる結果であり，コードの版とは関係ないため数えない。
このため，`dirty` は，commit 済みのコードや設定のファイルに commit していない変更がある場合に `True` になる。
まだ一度も commit していない script は数えられないので，記録を残したい実行の前には，script を commit しておく。

`json.dumps(辞書, indent=2)` は，辞書を JSON 形式の文字列に変換する。
`ensure_ascii=False` は，日本語などの文字を `\u3042` のような記号の並びに置き換えずに，そのまま書き出す指定である。
`Path.write_text` は，文字列をファイルに書き込む。
JSON で保存した情報は，`json.loads(info_path.read_text(encoding="utf-8"))` で辞書として読み戻せる。

### 4. 知見の整理と共有

#### 個人のメモと共有する知見

作業中に残したメモの多くは，自分が試した順序の記録であり，他の人が読んでもすぐには役立たない。
一方，他の人が同じ問題に出会ったときに，すぐに原因と対処が分かる形に整理した知見は，共有する価値がある。
共有する知見は，次の 4 項目で書く。

- 問題：何が起きたか（エラーメッセージ，予想と異なる値）
- 原因：なぜ起きたか
- 対処：どう直したか
- 確認方法：直ったことをどう確かめたか（実行できるコマンドやコード）

例えば，第22回の STFT で frame 数が合わなかった経験は，次のように書ける。

- 問題：長さ $T=16000$ の信号を `n_fft=512`，`hop_length=128` で STFT すると，frame 数が $1+\lfloor(T-512)/128\rfloor=122$ ではなく 126 になる。
- 原因：`center=True`（既定）では，信号の両端に `n_fft // 2` 点ずつ padding してから frame に分けるため，frame 数は $1+\lfloor T/128\rfloor$ になる。
- 対処：frame 数を $1+\lfloor T/\mathrm{hop\_length}\rfloor$ で計算する。
- 確認方法：`torch.stft(torch.zeros(16000), n_fft=512, hop_length=128, window=torch.hann_window(512), return_complex=True).shape` が `torch.Size([257, 126])` になることを確かめる。

確認方法まで書いておくと，読んだ人が自分の環境で同じ結果になるかを確かめられる。

#### 共通資産への反映

README の第28回の内容には，共通資産への反映が含まれる。
第27回で選んだ共有資産の候補（補助コード，テンプレート，説明）は，この回で整理して提案する。
教材リポジトリへ反映するかどうかと反映の方法は，授業の指示に従う。
提案では，候補ごとに，他の学生にとっての利点，入力と出力，動作を確かめた方法を示す。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第28回の第2問であれば，`exercise/exc28_02.py` とします。
文章で答える問題は，同じ番号の `.md` ファイルに書いてください。第1問であれば，`exercise/exc28_01.md` とします。
各 `.py` ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `output_dir`（`outputs/exc28`）を作成する処理を書いてください。
出力ファイルは `output_dir` に，問題番号を含むファイル名で保存してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（5問）
1. デモで説明する script を選び，説明メモを作ります。

    1. 第15回以降に自分が実装した script から 1 つを選んでください。第27回で整理した選択テーマの script を選ぶことを推奨します。
    2. `exercise/exc28_01.md` を作成し，`## 選んだ script`，`## 入力`，`## 処理`，`## 出力`，`## 再実行コマンド`，`## 5 分の説明順` の 6 つの見出しを書いてください。`## 入力` と `## 出力` には shape，dtype，ファイル名を書いてください。`## 5 分の説明順` には，解説の第1節の表のように，時間と見せるものを書いてください。
    3. `## 再実行コマンド` に書いたコマンドを作業リポジトリのルートで実行し，`## 出力` に書いたファイルがすべて作られることを確かめてください。
    4. 説明の最初に入力と出力を示す理由を，自分の script に即して 2 行以内で説明してください。

2. forward hook を使って，model の各層の出力の shape を追跡します。次の model と入力を使います。

    ```python
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    )
    x = torch.zeros(2, 1, 28, 28)
    ```

    1. 各層の出力の shape と，各層のパラメータ数を手計算してください。`nn.AdaptiveAvgPool2d(1)` は，各チャネルについて縦横のすべての値の平均をとり，縦横の大きさを 1 にする層です。
    2. `exercise/exc28_02.py` で，解説の第2節と同じ方法で各層に forward hook を登録し，`x` を入力したときの各層の出力の shape を表示してください。1 の手計算の結果と比較してください。
    3. 各層のパラメータ数 `sum(p.numel() for p in layer.parameters())` を，shape と同じ行に表示してください。表示した各行を `outputs/exc28/exc28_02_shapes.txt` に保存してください。パラメータを持たない層を挙げ，持たない理由を説明してください。
    4. 第1問で選んだ script の model にも forward hook を登録し，1 バッチの入力に対する各層の出力の shape を確かめてください。`nn.Module` を継承した model では，解説の第2節のように子を持たない層だけに登録してください。model を使わないテーマ（PCA）では，主な配列の shape を処理の順に表示してください。結果を `exercise/exc28_01.md` の `## 処理` に反映してください。
    5. walkthrough で各層の shape を示すと，聞き手にとって何が分かりやすくなるかを説明してください。

3. 実行情報を自動で保存し，再実行に必要な情報を確かめます。

    1. `exercise/exc28_03.py` で，第27回の第2問で整理した関数を `from exc27_02 import make_data, build_model, train, evaluate` で読み込んでください。`python exercise/exc28_03.py` で実行すると，`exercise/` 内の別のファイルを読み込めます。第27回の第2問に取り組んでいない場合は，第27回の第2問の整理後の関数をこのファイルに書いてください。
    2. seed を 0 として，第27回の第2問の `main()` と同じ処理（`torch.manual_seed(seed)` の後に，200 個のデータ，隠れ層の幅 16 の model，学習率 `1e-2` で 200 回の更新）を行い，最後の損失と正解率を表示してください。値が第27回の第2問の `main()` と一致することを確かめてください。
    3. 解説の第3節の `run_info` に，最後の損失 `final_loss` と正解率 `accuracy` を加えた辞書を，`outputs/exc28/exc28_03_run_info.json` に保存してください。
    4. `exercise/exc28_03.py` を commit した直後と，commit 済みのファイル（例えば `exercise/exc28_03.py` のコメント）を変更して保存した後のそれぞれで，`dirty` の値を予想してください。実行して確かめ，確かめた後は変更を commit してください。
    5. commit hash だけでは実行したコードを特定できない場合があるのはなぜかを，4 の結果を使って説明してください。また，まだ一度も commit していない新しい script は `dirty` の判定に含まれません。このことによって，どのような場合に，記録した情報から実行したコードを再現できなくなるかを説明してください。

4. デモの準備ができているかを確認します。

    1. `exercise/exc28_04.md` を作成し，デモの説明に必要な項目を 6 つ挙げてください。少なくとも，入力，処理，出力，結果，失敗例，再実行の方法を含めてください。
    2. 各項目に，第1問の説明メモと保存物を見て `OK` または `NG` を付けてください。
    3. `NG` の項目には，デモまでに何を直すかを書いてください。直したら `OK` に書き換えてください。
    4. 実演が失敗したときに代わりに見せる保存物を 1 つ決め，そのファイル名と，それを選んだ理由を書いてください。

5. 個人のメモと共有する知見を分けて整理します。

    1. `exercise/exc28_05.md` に `## 個人のメモとして残すもの` と `## 共有資産として整えるもの` の 2 つの見出しを作り，第15回以降の作業から，それぞれに当てはまるものを 2 つ以上書いてください。各項目には，そう分けた理由を 1 行で添えてください。
    2. 共有する知見を 1 つ選び，解説の第4節の 4 項目（問題，原因，対処，確認方法）で，`## 共有する知見` に書いてください。確認方法には，実行すると結果を確かめられるコマンドまたはコードを書いてください。
    3. 2 の確認方法を自分で実行し，書いたとおりの結果になることを確かめてください。

### 発展レベル（3問）
1. 質疑の準備をし，デモで受けた質問を記録します。

    1. 聞き手から出そうな質問を 3 つ作り，それぞれに答えるために見るファイルや出力を，`exercise/exc28_06.md` の `## 想定質問` に書いてください。
    2. デモの後，実際に受けた質問と自分の回答を `## 質疑の記録` に書いてください。答えられなかった質問には，確認に使う結果ファイルまたはコードを添えてください。

2. 選んだ script を来年度の教材として再利用する場合を，役割に応じて考えます。自分の学年に対応する項目に取り組み，`exercise/exc28_07.md` に書いてください。

    1. B4 または M：選んだ script を来年度の教材として使う場合に，削るべき部分と補うべき部分を，`## 教材化の提案` に 3 行以内で書いてください。
    2. M：B4 の提案を読み，教材として使う前に確かめるべき点（再実行の手順，所要時間，前提とする知識など）を 2 つ，`## 教材化の前に確かめること` に書いてください。

3. 記録した実行情報から同じ処理を再実行し，結果が一致するかを確かめます。

    1. `exercise/exc28_08.py` で，第3問の `outputs/exc28/exc28_03_run_info.json` を読み込み，記録された commit hash と現在の commit hash（`git rev-parse HEAD`）が一致するかを表示してください。
    2. 記録された seed を使って第3問と同じ学習と評価を行い，最後の損失と正解率が記録された値と一致するかを確かめてください。
    3. commit hash が一致しない場合や，記録された `dirty` が `true` の場合に，再実行の結果が一致しなくなる可能性がある理由を説明してください。また，バージョンの異なる PyTorch で実行した場合に結果が一致しない可能性がある理由も考察してください。

## 詰まったときに見る資料
- [`27-advanced-theme-wrapup.md`](27-advanced-theme-wrapup.md)
- [`15-annual-policy-and-ml-overview.md`](15-annual-policy-and-ml-overview.md)
- PyTorch docs: [`Module.register_forward_hook`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
- Python docs: [`subprocess`](https://docs.python.org/3/library/subprocess.html), [`json`](https://docs.python.org/3/library/json.html)
