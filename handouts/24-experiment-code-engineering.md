# 第24回 実験コードの整備

## この回の目標

実験条件を設定ファイルとコマンドライン引数で管理し，test で確かめた変更を，確認方法と未確認事項を添えて draft PR にまとめられる。

## 解説

この回では，第17回で扱った学習ループを，条件を変えて何度も実行でき，その結果を後から確かめられる実験コードに整える。
具体的には，実験の条件を設定ファイルとコマンドライン引数で与え，実行の記録をログに残し，乱数シードで結果を再現できるようにし，動作を test で確かめる。
最後に，変更を pull request にまとめて確認してもらう流れを扱う。

解説では，直線のデータに線形モデルを当てはめる学習を，1 つのスクリプト `train_line.py` に少しずつまとめていく。
コード例を試す場合は，`train_line.py`，設定ファイル `line_fit.json`，test の `test_train_line.py` を同じディレクトリに置き，そのディレクトリで作業リポジトリ内の `.venv` を有効にして実行する。
`train_line.py` の先頭には，次の `import` 文を書く。

```python
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
```

`argparse`，`json`，`logging` は Python の標準ライブラリであり，追加のインストールなしで使える。
以下のコード例のうち，関数を定義するもの（第2節の `main` と最後の `if` 文を含む）と，第3節でロガーを作る行は，この `import` 文に続けて `train_line.py` に書いていく。
それ以外の，動作を確かめるためのコード例は，同じディレクトリの別のファイル（例えば `check_train_line.py`）に書く。
そのファイルには，上の `import` 文と，`from train_line import load_config` のように使う関数を読み込む行を先に書き，その後に続けてコード例を書いて実行する。

### 1. 設定ファイル

#### 実験の条件をコードから分ける

学習率やエポック数のように実験ごとに変える値をスクリプトの中に直接書くと，条件を変えるたびにコードを編集することになる。
編集を繰り返すと，ある結果がどの値で得られたのかが分からなくなりやすい。
そこで，実験の条件をコードとは別のファイルにまとめる。
このファイルを **設定ファイル (configuration file)** といい，config と略して呼ぶことも多い。
この回では，Python の標準ライブラリで読み書きできる **JSON** 形式の設定ファイルを使う。

次の内容を `line_fit.json` として保存する。

```json
{
  "lr": 0.1,
  "epochs": 20,
  "seed": 1
}
```

JSON では，全体を `{}` で囲み，`"名前": 値` の組をカンマで区切って並べる。
名前と文字列は二重引用符 `"` で囲む。
Python の辞書と似ているが，最後の組の後にカンマを付けたり，`#` でコメントを書いたりすることはできない。

#### 設定ファイルの読み込み

JSON 形式のファイルは，`json.load` で辞書として読み込める。

```python
config_path = Path("line_fit.json")
with config_path.open(encoding="utf-8") as f:
    config = json.load(f)
print(config)                                      # {'lr': 0.1, 'epochs': 20, 'seed': 1}
print(type(config["lr"]), type(config["epochs"]))  # <class 'float'> <class 'int'>
```

`with ... as f:` は，ファイルを開き，ブロックを抜けるときに自動で閉じる書き方である。
`json.load` は，JSON の値を Python の値に変換する。
小数点を含む数は `float`，含まない数は `int`，二重引用符で囲んだ値は `str`，`true` と `false` は `bool`，`null` は `None` になる。
例えば `"epochs": 20.0` と書くと `float` になり，学習で `range(config["epochs"])` を実行したときに `TypeError` が発生する。
JSON の書き方に誤りがあると，`json.load` は `json.JSONDecodeError` を発生させる。

#### 読み込むときの検査

設定ファイルは人が手で書くため，項目の書き忘れや型の誤りが起こりやすい。
誤りがそのまま学習のコードへ渡ると，エラーは学習の途中の，原因から離れた場所で発生する。
そこで，読み込んだ直後に必要な項目と値を確かめ，誤りがあれば原因の分かるメッセージで処理を止める。

```python
def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        config = json.load(f)
    for key in ["lr", "epochs", "seed"]:
        if key not in config:
            raise ValueError(f"{path} に {key} がありません")
    if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
        raise ValueError(f"epochs は正の整数にしてください: {config['epochs']!r}")
    return config
```

`raise ValueError(メッセージ)` は，値が不正であることを表す **例外 (exception)** `ValueError` を発生させ，関数の処理を中断する。
例外を発生させることを **送出する (raise)** という。
`isinstance(値, 型)` は，値がその型であれば `True` を返す。
`f"...{式}..."` は，`{}` の中の式の値を埋め込んだ文字列を作る **f 文字列** であり，第2回の `format()` と同じ役割を持つ。
`{config['epochs']!r}` の `!r` は値を `repr` で埋め込む指定であり，数の `2.5` と文字列の `'2.5'` の違いがメッセージから分かる。

送出された例外は，`try` 文で捕まえて処理を続けられる。

```python
Path("broken.json").write_text('{"lr": 0.1, "epochs": 20}', encoding="utf-8")
try:
    load_config("broken.json")
except ValueError as error:
    print("設定ファイルの誤り:", error)  # 設定ファイルの誤り: broken.json に seed がありません
```

`try` のブロックで，`except` に書いた種類の例外が発生すると，処理は `except` のブロックへ移る。
`as error` で受け取った例外を `print` すると，送出したときのメッセージが表示される。

#### 使った設定を結果と一緒に保存する

結果を保存するときは，その結果を得たときの設定も同じ場所に保存しておく。
設定ファイルは後から書き換えられることがあるため，結果の隣に残した設定が，その結果の条件を示す確かな記録になる。
辞書を JSON 形式のファイルに書き出すには，`json.dump` を使う。

```python
out_dir = Path("outputs/line_fit_run")
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "config.json").open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
```

`open` の `"w"` は書き込み用に開く指定であり，同じ名前のファイルがあれば上書きする。
`indent=2` を指定すると，人が読みやすいように改行と字下げを入れて書き出す。

### 2. コマンドライン引数

#### argparse による引数の定義

`python train_line.py --config line_fit.json` のように，スクリプトを実行するときに後ろに付ける文字列を **コマンドライン引数 (command-line argument)** という。
コマンドライン引数で動作を指定して使うプログラムの操作方法を **CLI (command-line interface)** と呼ぶ。
Python の標準ライブラリ `argparse` を使うと，受け取る引数を定義し，入力された文字列を指定した型に変換できる。

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="直線のデータに線形モデルを当てはめる")
    parser.add_argument("--config", type=Path, required=True, help="設定ファイル (JSON) のパス")
    parser.add_argument("--out", type=Path, required=True, help="結果を保存するディレクトリ")
    parser.add_argument("--seed", type=int, default=None, help="設定ファイルの seed の代わりに使う値")
    return parser.parse_args(argv)
```

`add_argument` の第 1 引数は引数の名前である。
`--` で始まる名前の引数は **オプション引数** と呼ばれ，`--config line_fit.json` のように名前と値を組にして指定する。
`type` は値の変換先の型，`required=True` は省略できないこと，`default` は省略したときの値，`help` は説明文を表す。
`parse_args` は，指定された値を属性として持つオブジェクトを返す。
属性の名前は，引数の名前から `--` を除いたものである。
`parse_args` の引数 `argv` に文字列のリストを渡すとそのリストを解釈し，`None` のときは実際のコマンドライン引数を解釈する。

```python
args = parse_args(["--config", "line_fit.json", "--out", "outputs/run2"])
print(args.config, args.out, args.seed)  # line_fit.json outputs/run2 None
print(type(args.config))                 # <class 'pathlib.PosixPath'>
```

`--seed` を省略したため，`args.seed` は `default` の `None` になる。
`type=Path` を指定したため，`args.config` は文字列ではなく `Path` になっている。
リストを渡して試せるので，test からも同じ関数を呼び出せる。

#### main 関数と実行の入口

スクリプトの処理は関数 `main` にまとめ，ファイルの最後で呼び出す。
次の `main` の中の `setup_logging` と `logger` は第3節で，`train` は第4節で定義する。
最後の `if __name__ == "__main__":` の 2 行は常にファイルの末尾に置き，第3節と第4節のコードはこの 2 行より前に書く。

```python
def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out / "train.log")
    logger.info("out=%s", args.out)
    for key, value in config.items():
        logger.info("%s=%s", key, value)

    history = train(config["lr"], config["epochs"], config["seed"])
    with (args.out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with (args.out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
```

`main` では，`--seed` が指定された場合に設定ファイルの `seed` を置き換え，実際に使う値を `config` にそろえてから，ログと学習に渡している。
保存先には，学習の結果 `history.json` と，実際に使った設定 `config.json` を並べて保存する。

`python train_line.py` として実行したファイルでは，特別な変数 `__name__` が文字列 `"__main__"` になる。
一方，別のファイルから `import train_line` や `from train_line import load_config` で読み込んだ場合は，`__name__` はモジュール名の `"train_line"` になる。
このため，`if __name__ == "__main__":` の中に書いた処理は，スクリプトとして実行したときだけ動き，`import` したときには動かない。
test などから `load_config` だけを使いたいときに，学習まで実行されてしまうことを防げる。

#### --help と引数の誤り

`argparse` を使ったスクリプトは，`--help`（または `-h`）を付けて実行すると，引数の定義から自動で作った使い方を表示する。

```bash
python train_line.py --help
```

```text
usage: train_line.py [-h] --config CONFIG --out OUT [--seed SEED]

直線のデータに線形モデルを当てはめる

options:
  -h, --help       show this help message and exit
  --config CONFIG  設定ファイル (JSON) のパス
  --out OUT        結果を保存するディレクトリ
  --seed SEED      設定ファイルの seed の代わりに使う値
```

1 行目の `usage` では，省略できる引数を `[]` で囲んで示す。
省略できない引数を付けずに実行すると，使い方とエラーの内容を表示して終了する。

```bash
python train_line.py
```

```text
usage: train_line.py [-h] --config CONFIG --out OUT [--seed SEED]
train_line.py: error: the following arguments are required: --config, --out
```

#### 設定ファイルと引数の使い分け

設定ファイルとコマンドライン引数は，どちらも実行の条件を与える手段だが，置く値を分けると整理しやすい。

| 置き場所 | 置く値 | 例 |
| --- | --- | --- |
| 設定ファイル | 実験の条件として比べたり記録したりする値 | 学習率，エポック数，乱数シード |
| コマンドライン引数 | 実行ごとに変わり，学習の結果を変えない値 | 設定ファイルのパス，保存先 |

同じ設定ファイルを使い，保存先だけを変えて実行すれば，同じ条件の結果を別の場所に作れる。
`--seed` のように設定ファイルの値を引数で置き換える場合は，置き換えた後の値を結果と一緒に保存し，どの値で実行したかが後から分かるようにする。

```bash
python train_line.py --config line_fit.json --out outputs/line_fit_run1
python train_line.py --config line_fit.json --out outputs/line_fit_seed2 --seed 2
```

2 つ目の実行では，`outputs/line_fit_seed2/config.json` の `seed` が `2` になる。

### 3. logging

#### print と logging

`print` は画面に表示するだけなので，ターミナルを閉じると表示は失われる。
後から結果を確かめるには，実行中に起きたことを時刻とともにファイルへ残しておく必要がある。
このような実行の記録を **ログ (log)** といい，Python では標準ライブラリ `logging` で出力できる。
`logging` では，メッセージに重要度を表す **レベル (level)** を付け，出力先と書式をまとめて設定できる。

| レベル | 使う場面 |
| --- | --- |
| `DEBUG` | 問題を調べるときだけ見る，詳しい途中経過 |
| `INFO` | 通常の進行（設定の値，エポックごとの損失など） |
| `WARNING` | 想定外だが，処理を続けられる状況 |
| `ERROR` | 処理を続けられない誤り |

#### ロガーの作成と出力先の設定

ログは，**ロガー (logger)** と呼ばれるオブジェクトを通して出力する。
ロガーはファイルの先頭（`import` 文の後）で作り，出力先と書式の設定は，実行の入口である `main` の中で 1 回だけ行う。

```python
logger = logging.getLogger(__name__)


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
```

`logging.getLogger(__name__)` は，そのモジュールのロガーを返す。
`logging.basicConfig` は，すべてのロガーから出力されたログの受け取り方をまとめて設定する。
`import` した別のファイルのロガーから出力したログも，この設定に従って出力される。
`level=logging.INFO` は `INFO` 以上のレベルを出力する指定であり，`DEBUG` のログは出力されない。
`format` の `%(asctime)s` は日時，`%(levelname)s` はレベル，`%(message)s` はメッセージに置き換えられる。
`handlers` には出力先を並べる。
`StreamHandler()` は画面（標準エラー出力）へ，`FileHandler(log_path)` は指定したファイルへ出力する。
`basicConfig` は，既定では 2 回目以降の呼び出しを無視する。
`force=True` を指定すると前の設定を置き換えるため，1 つのプロセスの中で保存先を変えて `main` を呼び出した場合も，新しいログファイルへ出力される。
なお，`basicConfig` で設定していない場合は，`INFO` のログは表示されない。

#### ログに残す内容

`logger.info(書式, 値, ...)` は，`INFO` レベルのログを出力する。
書式の `%s` には値の文字列が，`%d` には整数が，`%.4f` には小数点以下 4 桁の小数が入る。
第2節の `main` では，保存先と設定のすべての項目を `項目名=値` の形で 1 行ずつ出力している。
第4節の `train` では，エポックごとの平均損失を出力する。

```bash
python train_line.py --config line_fit.json --out outputs/line_fit_run1
```

```text
2026-10-15 15:41:11,136 INFO out=outputs/line_fit_run1
2026-10-15 15:41:11,137 INFO lr=0.1
2026-10-15 15:41:11,137 INFO epochs=20
2026-10-15 15:41:11,137 INFO seed=1
2026-10-15 15:41:13,688 INFO epoch=1 loss=7.0872
2026-10-15 15:41:13,690 INFO epoch=2 loss=2.5624
...
2026-10-15 15:41:13,734 INFO epoch=20 loss=0.0394
```

同じ内容が，画面と `outputs/line_fit_run1/train.log` の両方に出力される。
日時の部分は，実行するたびに変わる。
ログには，後からログだけを見て「いつ，どの条件で実行し，どこへ保存し，学習がどう進んだか」を追える内容を残す。
設定の値を残しておくと，2 つの結果の違いが条件の違いによるものかを，後から確かめられる。

### 4. 乱数シードと再現性

#### 学習で使われる乱数

ニューラルネットワークの学習では，いくつもの処理で乱数を使う。
例えば，`nn.Linear` のパラメータの初期値は乱数で決まり，`DataLoader` で `shuffle=True` とすると，エポックごとにデータの順番が乱数で並べ替えられる。
第6回で扱ったように，乱数の初期状態を決める **乱数シード (random seed)** を固定すると，同じ環境では同じ結果を再現できる。

| 乱数を使う処理 | 乱数を固定する方法 |
| --- | --- |
| データの生成（NumPy） | `np.random.default_rng(seed)` で作った乱数生成器を使う |
| パラメータの初期値（PyTorch） | モデルを作る前に `torch.manual_seed(seed)` を呼ぶ |
| ミニバッチの順番（`DataLoader`） | `generator=torch.Generator().manual_seed(seed)` を渡す |
| Python 標準の `random` を使う処理 | `random.seed(seed)` を呼ぶ |

ミニバッチの順番は，`DataLoader` に専用の乱数生成器（generator）を渡し，パラメータの初期値とは別の乱数で決める。

#### 乱数シードを指定した学習

```python
def train(lr, epochs, seed):
    data_rng = np.random.default_rng(0)
    x = data_rng.uniform(-1, 1, size=(64, 1))
    y = -3 * x + 2 + data_rng.normal(0, 0.2, size=(64, 1))
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))

    torch.manual_seed(seed)
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        total = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x_batch)
        history.append(total / len(dataset))
        logger.info("epoch=%d loss=%.4f", epoch + 1, history[-1])
    return history
```

この例では，データは常に同じシード `0` で作り，引数の `seed` はパラメータの初期値とミニバッチの順番だけに使う。
こうすると，seed を変えたときの結果の違いを，同じデータに対する学習のばらつきとして比べられる。
`TensorDataset` は，同じ長さの tensor を受け取り，`i` 番目の要素の組 `(x[i], y[i])` を返す Dataset である。
第20回のように Dataset のクラスを自分で定義する代わりに使える。
エポックごとの平均損失は，ミニバッチごとの損失にそのミニバッチの大きさを掛けて足し合わせ，データ数で割って求めている。

```python
history_a = train(lr=0.1, epochs=5, seed=1)
history_b = train(lr=0.1, epochs=5, seed=1)
history_c = train(lr=0.1, epochs=5, seed=2)
print(history_a == history_b)            # True
print(history_a == history_c)            # False
print([round(v, 4) for v in history_a])  # [7.0872, 2.5624, 1.1997, 0.645, 0.3731]
print([round(v, 4) for v in history_c])  # [5.9708, 2.1773, 1.0319, 0.5481, 0.3197]
```

同じ seed で 2 回学習すると，各エポックの平均損失は完全に一致する。
seed を変えると，パラメータの初期値とデータの順番が変わるため，損失の値も変わる。
ただし，どちらの seed でも損失は同じように減少している。

#### 再現できる範囲

乱数シードを固定しても，結果が一致するのは，同じコード，同じライブラリの版，同じ種類の計算機で実行した場合である。
ライブラリの版が変わると，同じ seed でも乱数の使い方や計算の順序が変わることがある。
GPU を使う計算には，同じ seed でも結果がわずかに変わる処理がある（`torch.use_deterministic_algorithms(True)` を実行しておくと，そのような処理を使ったときにエラーで知らせる）。
このため，ログには seed とともに実行の条件を残し，ライブラリの版は作業リポジトリの `uv.lock` で固定する。
また，seed は結果を再現するためのものであり，良い結果が出る seed を選ぶためのものではない。
seed による結果のばらつきは，第26回で複数の seed を使って調べる。

### 5. test

#### test の役割

コードを変更したときに，以前は正しく動いていた部分が動かなくなることを **regression** という。
変更のたびに手で実行して確かめるのは手間がかかり，確認を忘れやすい。
そこで，期待する動作を確かめるコードを書いておき，コマンド 1 つで繰り返し実行できるようにする。
このようなコードを **test** という。
test は，あらゆる入力で正しいことを証明するものではなく，決めた入力で期待どおりに動くことを素早く確かめるためのものである。
test の対象には，次のようなものを選ぶ。

- 答えが分かっている関数の結果（例：決まった設定ファイルを読み込んだときの値）
- 誤った入力に対する動作（例：項目のない設定ファイルで `ValueError` が送出される）
- スクリプト全体の動作（例：コマンドを実行すると，ログと結果のファイルが作られる）

#### unittest による test

Python の標準ライブラリ `unittest` では，`unittest.TestCase` を継承したクラスの中に，名前が `test` で始まるメソッドとして test を書く。
次の内容を `test_train_line.py` として保存する。

```python
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from train_line import load_config

HERE = Path(__file__).resolve().parent


class TestLoadConfig(unittest.TestCase):
    def test_valid_config(self):
        config = load_config(HERE / "line_fit.json")
        self.assertEqual(config["epochs"], 20)

    def test_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text(json.dumps({"lr": 0.1, "epochs": 20}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
```

`self.assertEqual(a, b)` は `a` と `b` が等しいことを確かめ，等しくなければその test を失敗にする。
`with self.assertRaises(ValueError):` は，ブロックの中で `ValueError` が送出されることを確かめ，送出されなければその test を失敗にする。
`tempfile.TemporaryDirectory()` は一時的なディレクトリを作り，`with` のブロックを抜けると中身ごと削除する。
test の中で作るファイルをここに置くと，作業リポジトリに不要なファイルが残らない。
`json.dumps` は，辞書を JSON 形式の文字列に変換する。
`Path(__file__).resolve().parent` は，この test ファイルが置かれたディレクトリの絶対パスである。
これを基準にファイルを指定すると，どのディレクトリから test を実行しても同じファイルを参照できる。

#### スクリプト全体の test

スクリプト全体の動作は，`subprocess.run` でコマンドを実行して確かめる。
次のクラスを，`test_train_line.py` の続きに書く。

```python
class TestCli(unittest.TestCase):
    def test_run_writes_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            command = [
                sys.executable, str(HERE / "train_line.py"),
                "--config", str(HERE / "line_fit.json"), "--out", str(out),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            log_text = (out / "train.log").read_text(encoding="utf-8")
            self.assertIn("lr=0.1", log_text)
```

`subprocess.run(コマンドのリスト)` は，ターミナルで入力するのと同じようにコマンドを実行し，終了するまで待つ。
`sys.executable` は，いま test を実行している Python のパスであり，作業リポジトリ内の `.venv` の Python をそのまま使える。
`capture_output=True, text=True` を指定すると，コマンドが画面へ出力した内容を，文字列として `result.stdout`（標準出力）と `result.stderr`（標準エラー出力）で受け取れる。
`result.returncode` は **終了コード** であり，正常に終了すると `0`，エラーで終了すると `0` 以外になる。
`assertEqual` の第 3 引数は失敗したときに表示するメッセージであり，ここではエラーの内容が分かるように標準エラー出力を渡している。
`self.assertIn(a, b)` は，`a` が `b` に含まれることを確かめる。

#### test の実行

`unittest` は，指定したディレクトリから test のファイルを探して実行できる。
コード例のファイルを置いたディレクトリで，次を実行する。

```bash
python -m unittest discover -s . -p "test_*.py" -v
```

```text
test_run_writes_log (test_train_line.TestCli.test_run_writes_log) ... ok
test_missing_key (test_train_line.TestLoadConfig.test_missing_key) ... ok
test_valid_config (test_train_line.TestLoadConfig.test_valid_config) ... ok

----------------------------------------------------------------------
Ran 3 tests in 6.204s

OK
```

`discover` は，`-s` で指定したディレクトリから，`-p` で指定したパターンに一致する名前のファイルを探し，その中の test をすべて実行する。
`-s` で指定したディレクトリは `import` の検索先にも加えられるため，同じディレクトリの `train_line.py` を `from train_line import load_config` で読み込める。
`-v` を付けると，test ごとの結果を 1 行ずつ表示する。
すべて成功すると最後に `OK` と表示される。
失敗した test があると，`FAIL` と test の名前，比べた値（例えば `AssertionError: 20 != 10`）が表示され，最後に `FAILED (failures=1)` のように失敗した数が表示される。
実行時間は計算機によって異なる。

### 6. PR と code review

#### ブランチと pull request

Git では，commit の履歴を **ブランチ (branch)** と呼ばれる流れに分けて管理できる。
これまでの演習では，`main` ブランチで commit と push を行ってきた。
変更を `main` とは別のブランチで行い，その変更を `main` に取り込んでよいかを確認してもらう依頼を，GitHub では **pull request (PR)** という。
PR では，取り込み先のブランチ（**base**）と変更を含むブランチ（**compare**）を選ぶと，2 つのブランチの差分（**diff**）が表示され，行ごとにコメントを付けて議論できる。
まだ取り込む段階ではないが共有したい変更は，**draft PR** として作成する。
draft PR は，作業の途中であることを明示したまま確認を依頼するために使う。

ブランチの作成から push までは，次のように行う。

```bash
git switch -c session24-pr-practice
git add exercise/exc24_01.py
git commit -m "第24回: 設定ファイルの読み込みと検査"
git push -u origin session24-pr-practice
```

`git switch -c 名前` は，新しいブランチを作り，そのブランチに切り替える。
以降の commit はこのブランチに記録され，`main` の履歴は変わらない。
`git push -u origin 名前` は，ブランチを GitHub へ送り，以後の `git push` の送信先をそのブランチに設定する。
push した後に GitHub で自分の作業リポジトリのページを開くと，**Compare & pull request** というボタンが表示される。
これを押し，base を `main`，compare を `session24-pr-practice` にして，作成ボタンの選択肢から **Create draft pull request** を選ぶと，draft PR を作成できる。
第1回で準備した `gh` コマンドを使う場合は，`gh pr create --draft --base main` を実行し，表示に従ってタイトルと本文を入力する。

#### PR の本文

PR の本文には，review する人が diff を読む前に知っておくべきことを書く。

```markdown
## 変更点
- 学習の条件を設定ファイル `line_fit.json` から読み込むようにした
- `--config`，`--out`，`--seed` で実行の条件を指定できるようにした

## 確認方法
- `python -m unittest discover -s . -p "test_*.py" -v` を実行し，3 件の test が OK になった
- 同じ seed で 2 回実行し，`history.json` が一致することを確かめた

## 未確認事項
- GPU での実行は確かめていない
- 設定ファイルに不要な項目がある場合の動作は決めていない
```

**変更点** には何をなぜ変えたか，**確認方法** には実行したコマンドとその結果，**未確認事項** にはまだ確かめていないことや判断を保留していることを書く。
未確認事項を書いておくと，review する人がどこを重点的に見ればよいかが分かり，完成したように見える変更に問題が隠れることを防げる。

#### code review

他の人が diff を読み，問題点や改善案をコメントすることを **code review** という。
review では，次の観点で diff を読む。

| 観点 | 確認すること |
| --- | --- |
| 正しさ | 変更の目的どおりに動くか，境界の値や誤った入力で壊れないか |
| 再現性 | 条件が設定ファイルとログに残るか，乱数シードが固定されているか |
| test | 変更した動作を確かめる test があり，実行して成功したか |
| 読みやすさ | 関数の名前と役割が対応しているか，同じ処理が重複していないか |

コメントには，対象の行，問題点，その理由，修正案を組にして書く。
例えば，「`lr` に負の値を書いてもエラーにならず，学習で損失が増え続けます。`load_config` で `lr` が正の数かを確かめてはどうでしょうか」のように書く。
コメントを受けた側は，修正した commit を同じブランチに push するか，修正しない理由を返信する。
同じブランチに push した commit は，同じ PR に自動で追加される。
PR の練習が終わった後は，[PR練習後の手順](README.md#pr練習後に通常作業へ戻る)に従って `main` に戻る。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第24回の第1問であれば，`exercise/exc24_01.py` とします。
設定ファイル，文章で答える問題，test は，問題文で指定した名前（`exercise/exc24_config.json`，`exercise/exc24_06.md`，`exercise/test_exc24_05.py` など）で保存してください。
各ファイルの先頭には，解説の冒頭にある `import` 文のうち，そのファイルで使うものを書いてください。
ログ，設定，損失の記録などの実行結果は，`outputs/exc24/` 以下に保存してください。
コマンドは，作業リポジトリのルートで，作業リポジトリ内の `.venv` を有効にして実行してください。

この回の演習は，第6問で draft PR にまとめるため，PR 練習用のブランチ `session24-pr-practice` で行います。
演習を始める前に，作業リポジトリのルートで次を実行し，最後に `session24-pr-practice` と表示されることを確認してください。
`git status` に commit していない変更が表示された場合は，先にその変更を `main` で commit してから実行してください。

```bash
git status
git switch main
git pull --ff-only origin main
git switch -c session24-pr-practice
git branch --show-current
```

基本的にはプログラム (Python) とコマンドを使って取り組むことを想定しています。
ただし，次の指示がある問題は，実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想して，その問題のファイルにコメントとして書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。

### 基礎レベル（6問）
1. 実験の条件を設定ファイルにまとめ，読み込むときに内容を検査します。

    1. `exercise/exc24_config.json` を作成し，次の内容を書いてください。

        ```json
        {
          "lr": 0.001,
          "epochs": 3,
          "name": "session24-demo",
          "seed": 0
        }
        ```

    2. `exercise/exc24_01.py` に，設定ファイルのパスを受け取り，読み込んだ辞書を返す関数 `load_config(path)` を定義してください。次の条件のどれかを満たさない場合は，どの項目がどう不正かが分かるメッセージを付けて `ValueError` を送出してください。

        - `lr`，`epochs`，`name`，`seed` の 4 項目がすべてある
        - `lr` は正の数（`int` または `float`）である
        - `epochs` は正の整数（`int`）である
        - `name` は文字列（`str`）である
        - `seed` は整数（`int`）である

    3. `if __name__ == "__main__":` の中で，1 のファイルを `load_config` で読み込み，各項目の値と型を表示してください。この `if` 文の中の最初の行には，`print("exc24_01.py を実行しました")` を書いてください（第2問で使います）。
    4. 1 の内容から `seed` を除いた設定ファイル `outputs/exc24/exc24_01_no_seed.json` と，`epochs` を `3.0` にした設定ファイル `outputs/exc24/exc24_01_float_epochs.json` を作ってください。それぞれを `load_config` に渡したときに表示されるメッセージを予想してください。`try` 文で `ValueError` を捕まえてメッセージを表示し，予想と比べてください。
    5. `exc24_01_float_epochs.json` を `json.load` で直接読み込んだときの `epochs` の型を確かめてください。検査をせずにこの値を学習の `range(config["epochs"])` に使った場合に起きるエラーも確かめ，読み込む時点で型を検査する理由を説明してください。

2. 設定ファイルのパスと保存先をコマンドライン引数で受け取り，実行の記録をログに残すスクリプトを作ります。

    1. `exercise/exc24_02.py` に，`argparse` を使って `--config`（設定ファイルのパス）と `--out`（保存先ディレクトリ）を受け取る関数 `parse_args(argv=None)` を定義してください。どちらも省略できない引数とし，`help` に説明を書いてください。`python exercise/exc24_02.py --help` を実行し，2 つの引数の説明が表示されることを確認してください。
    2. ログを画面とファイルの両方に出力するように設定する関数 `setup_logging(log_path)` を定義してください。
    3. 関数 `main(argv=None)` を定義し，次の処理を順に行ってください。第1問の `load_config` は，`from exc24_01 import load_config` で読み込んで使ってください。

        - 設定ファイルを `load_config` で読み込む
        - 保存先ディレクトリを作成し，`setup_logging` でログの保存先を `保存先/run.log` にする
        - 保存先のパスと，設定の 4 項目を `lr=0.001` のような `項目名=値` の形で，1 行ずつログに出力する
        - 読み込んだ設定を `保存先/config_used.json` に保存する

    4. 引数を付けずに `python exercise/exc24_02.py` を実行したときの表示を予想してください。実行して確かめてください。
    5. 同じ設定ファイル `exercise/exc24_config.json` を使い，`--out outputs/exc24/run_a` と `--out outputs/exc24/run_b` の 2 通りで実行してください。2 つの `run.log` と 2 つの `config_used.json` で，何が同じで何が異なるかを予想してから確かめてください。
    6. 実行したときに，第1問で書いた `exc24_01.py を実行しました` が表示されないことを確認し，その理由を説明してください。また，`lr` と `epochs` を設定ファイルに，保存先をコマンドライン引数に置いた理由を，5 の比較に基づいて説明してください。

3. 学習の結果が乱数によってどう変わるかを確かめ，乱数シードで同じ結果を再現できるようにします。データは第15回の基礎第3問と同じ直線のデータとし，`rng = np.random.default_rng(0)`，`x = rng.uniform(0, 1, size=40)`，`y = 2 * x + 1 + rng.normal(0, 0.1, size=40)` で作った 40 点すべてを学習に使います。

    1. `exercise/exc24_03.py` に，学習率 `lr`，エポック数 `epochs`，乱数シード `seed` を受け取り，エポックごとの平均損失のリストを返す関数 `train(lr, epochs, seed)` を定義してください。モデルは `nn.Linear(1, 1)`，損失関数は `nn.MSELoss()`，optimizer は `torch.optim.SGD` とします。`DataLoader` で大きさ 8 のミニバッチに分け，`shuffle=True` でエポックごとに順番を並べ替えてください。モデルを作る直前に `torch.manual_seed(seed)` を呼び，`DataLoader` には `generator=torch.Generator().manual_seed(seed)` を渡してください。エポックごとの平均損失は，`logger.info` で `epoch=1 loss=...` の形のログにも出力してください。
    2. `train(0.1, 5, 0)` を 2 回実行した結果と，`train(0.1, 5, 1)` の結果を表示してください。同じ seed の 2 回の結果が一致するか，seed を変えた結果と一致するかを予想してから，`==` で確かめてください。
    3. 次の (a)，(b) のように一時的に変更したそれぞれの場合に，`train(0.1, 5, 0)` を 2 回実行した結果が一致するかを予想してください。実行して確かめた後，元に戻してください。

        - (a) `torch.manual_seed(seed)` の行を削除する
        - (b) `DataLoader` に渡す `generator=...` を削除する（`torch.manual_seed(seed)` は残す）

    4. (b) の状態で，`model = nn.Linear(1, 1)` の直後に，使わない層を作る行 `unused = nn.Linear(1, 4)` を加えると，`train(0.1, 5, 0)` の結果が (b) のときから変わるかを予想してください。1 エポック目の最初のミニバッチの `x` も表示して確かめてください。次に，`generator` を渡す元のコードに同じ行を加えた場合も確かめてください。確かめた後は，加えた行を削除し，`generator` を渡す元のコードに戻してください。
    5. 3 と 4 の結果から，(b) で 2 回の結果が一致した理由と，`DataLoader` に専用の generator を渡す理由を説明してください。また，データの生成，パラメータの初期値，ミニバッチの順番のそれぞれで使われる乱数を，どのコードで固定しているかをコメントに書いてください。

4. 第2問のコマンドライン引数とログ，第3問の学習を組み合わせ，設定ファイルの条件で学習して結果を保存するスクリプトを作ります。

    1. `exercise/exc24_04.py` を作成し，第2問の `main` の処理に，次の処理を加えてください。第1問の `load_config`，第2問の `setup_logging`，第3問の `train` は `import` して使ってください。

        - 省略できる引数 `--seed`（整数）を加え，指定された場合は設定ファイルの `seed` の代わりに使う
        - 設定の `lr`，`epochs`，`seed` で `train` を実行する
        - 実際に使った設定（`--seed` で置き換えた後の値）を `保存先/config_used.json` に，エポックごとの平均損失のリストを `保存先/history.json` に保存する

    2. `python exercise/exc24_04.py --config exercise/exc24_config.json --out outputs/exc24/run_seed0` を実行してください。`run.log` に `lr=0.001`，`epochs=3`，`name=session24-demo`，`seed=0` と，3 エポック分の損失が記録されていることを確認してください。
    3. 2 と同じコマンドを `--out outputs/exc24/run_seed0_again` に変えて実行し，さらに `--seed 1 --out outputs/exc24/run_seed1` として実行してください。3 つの `history.json` のうち，どれとどれが一致するかを予想してから確かめてください。
    4. `run_seed0` の損失がほとんど減らない理由を，設定の値から説明してください。次に，`lr` を `0.1`，`epochs` を `20`，`name` を `session24-lr01` とした設定ファイル `exercise/exc24_04_lr01.json` を作り，`--out outputs/exc24/run_lr01` として実行して，損失が減ることを確かめてください。コードを変更せずに学習の条件を変えられたことを確認してください。
    5. `config_used.json` に，設定ファイルの内容ではなく実際に使った値を保存する理由を，`run_seed1` を例に説明してください。

5. 第1問と第4問の動作を自動で確かめる test を書き，変更しても壊れていないことをすぐに確認できるようにします。

    1. `exercise/test_exc24_05.py` に，`unittest` を使って次の 3 つの test を書いてください。test の中で作るファイルは，`tempfile.TemporaryDirectory()` で作った一時ディレクトリに置いてください。

        - `exercise/exc24_config.json` を `load_config` で読み込むと，`lr` が `0.001`，`epochs` が `3` になる
        - `seed` のない設定ファイルを `load_config` に渡すと，`ValueError` が送出される
        - `subprocess.run` で `exc24_04.py` を実行すると終了コードが `0` になり，保存先に `run.log`，`config_used.json`，`history.json` が作られ，`run.log` に `lr=0.001` が含まれ，`history.json` の要素数が `epochs` と同じ `3` になる

    2. 作業リポジトリのルートで次のコマンドを実行し，3 つの test がすべて `ok` になることを確認してください。実行したコマンドと結果の要約（`Ran 3 tests` と `OK`）を，test ファイルの先頭にコメントとして書いてください。

        ```bash
        python -m unittest discover -s exercise -p "test_exc24*.py" -v
        ```

    3. `exc24_04.py` で `history.json` に保存するリストを，一時的に `history[:-1]`（最後のエポックの損失を除いたリスト）に変えると，どの test がどのような表示で失敗するかを予想してください。実行して確かめ，元に戻してください。
    4. 3 の表示から，失敗の原因をどこまで特定できるかを説明してください。また，今後 `exc24_04.py` を変更するときに，この test があると何が楽になるかを説明してください。

6. 第1〜5問の変更を commit して push し，draft PR にまとめます。

    1. `exercise/exc24_06.md` を作成し，`## 変更目的`，`## 追加した test`，`## 確認方法`，`## review で見てほしい点` の 4 つの見出しを書いてください。`## 確認方法` には，第4問と第5問で実行したコマンドとその結果を書いてください。
    2. `git status` で変更されたファイルを確認してから，`git add exercise/exc24_* exercise/test_exc24_05.py outputs/exc24` で第1〜6問のファイルと実行結果を選んでください。`git diff --cached --stat` で，`__pycache__` などの意図しないファイルが含まれていないことを確認してから commit してください。
    3. `git push -u origin session24-pr-practice` で push してください。
    4. GitHub で，compare を `session24-pr-practice`，base を自分の作業リポジトリの `main` とする draft PR を作成してください。PR の本文には `## 変更点`，`## 確認方法`，`## 未確認事項` の見出しを書き，`## 未確認事項` には，確かめていないことを 1 つ以上書いてください。
    5. PR の **Files changed** に表示されるファイルを予想してから確認してください。表示されたファイルと `main` の関係を説明してください。

    PR を作成した後の追記や修正は，同じブランチに commit して push してください。発展レベルに取り組む場合も，同じブランチで行ってください。
    次回の通常作業を始める前に，[PR練習後の手順](README.md#pr練習後に通常作業へ戻る)で `main` へ切り替え，同期してください。review 待ちの変更は，練習ブランチに残ります。

### 発展レベル（4問）
1. 設定ファイルが見つからない場合や JSON として読めない場合に，原因の分かる 1 行のメッセージを表示して終了するようにします。

    1. `python exercise/exc24_04.py --config exercise/not_found.json --out outputs/exc24/run_missing` を実行したときの表示と，`outputs/exc24/run_missing` が作られるかを予想してから，実行して確かめてください。
    2. `exc24_04.py` を基に `exercise/exc24_07.py` を作り，設定ファイルが存在しない場合（`FileNotFoundError`），JSON として読めない場合（`json.JSONDecodeError`），`load_config` の検査で誤りが見つかった場合（`ValueError`）に，traceback ではなく，設定ファイルのパスと原因を含む 1 行のメッセージを表示して終了するようにしてください。終了には `sys.exit(メッセージ)` を使ってください。`sys.exit` に文字列を渡すと，その文字列を標準エラー出力に表示し，終了コード `1` で終了します。
    3. 存在しないパス，最後の項目の後にカンマがある JSON，`seed` のない JSON の 3 通りで実行し，それぞれのメッセージと終了コード（実行の直後に `echo $?` で表示できます）を確認してください。
    4. `except json.JSONDecodeError` を `except ValueError` より後に書いた場合に，最後の項目の後にカンマがある JSON でどのメッセージが表示されるかを予想し，確かめてください。`issubclass(json.JSONDecodeError, ValueError)` の結果と結びつけて，理由を説明してください。
    5. `exercise/test_exc24_07.py` に，存在しない設定ファイルを渡すと終了コードが `0` 以外になり，標準エラー出力に設定ファイルのパスが含まれ，`Traceback` が含まれないことを確かめる test を書いてください。第5問と同じコマンドで実行し，すべての test が成功することを確認してください。

2. 別の人が `run_seed0` と同じ結果を再現するために必要な情報を整理します。

    1. `exercise/exc24_08.md` に `## 記録されている情報` と `## まだ記録されていない情報` の 2 つの見出しを作り，次の 7 項目を分類してください。記録されている項目には，記録しているファイルの名前を書いてください。記録されていない項目には，その情報がないと再現できなくなる具体的な状況を書いてください。

        - 実行したコマンド
        - 設定の値
        - 乱数シード
        - コードの版（commit）
        - Python と主要なライブラリの版
        - データの作り方
        - 実行した計算機の種類（CPU か GPU か）

    2. まだ記録されていない情報のうち，`run.log` に加えると最も効果が大きいと考える項目を 1 つ選び，理由を書いてください。

3. 第5問の test では見つけられない不具合があることを確かめ，それを見つける test を加えます。

    1. `exc24_04.py` で `train` に渡す学習率を，一時的に設定の `lr` ではなく `0.1` に変えてください。この状態で第5問の test を実行した結果を予想し，確かめてください。
    2. `exercise/test_exc24_09.py` に，`exc24_04.py` を `subprocess.run` で実行して保存された `history.json` が，同じ設定で `train` を直接呼び出した結果と一致することを確かめる test を書いてください。1 の変更をしたまま実行し，この test が失敗することを確認してください。
    3. 1 の変更を元に戻し，`python -m unittest discover -s exercise -p "test_exc24*.py" -v` ですべての test が成功することを確認してください。
    4. 第5問の test が 1 の不具合を見つけられなかった理由を，第5問の test が確かめている内容に基づいて説明してください。
    5. `exercise/exc24_06.md` に `## 残っているリスク` を追加し，ここまでの test でもまだ見つけられない不具合を 2 つ書いてください。また，PR の本文の `## 未確認事項` に，追加した test が防ぐ regression と防げない regression を 3 行以内で書いてください。

4. 第6問の draft PR を review の観点から読み直し，指摘と修正を PR の上で行います。

    1. PR の **Files changed** で diff を読み，解説の code review の観点の表から 2 つ以上の観点を選んで，該当する行に review comment を 2 件以上書いてください。各 comment には，問題点，その理由，修正案を書いてください。
    2. そのうち 1 件を修正する commit を同じブランチに push し，その comment に修正の内容を返信してください。PR の **Commits** に修正の commit が追加されたことを確認してください。
    3. `exercise/exc24_06.md` に `## review で見つけた点と対応` を追加し，修正した点と修正しなかった点を，それぞれ理由とともに書いてください。

## 詰まったときに見る資料
- [PR練習後に通常作業へ戻る](README.md#pr練習後に通常作業へ戻る)
- Python docs: [`json`](https://docs.python.org/ja/3/library/json.html), [`argparse`](https://docs.python.org/ja/3/library/argparse.html), [`logging`](https://docs.python.org/ja/3/howto/logging.html), [`unittest`](https://docs.python.org/ja/3/library/unittest.html)
- [PyTorch: Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [GitHub Docs: プルリクエストの作成](https://docs.github.com/ja/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
