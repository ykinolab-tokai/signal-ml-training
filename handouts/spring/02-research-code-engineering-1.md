# 第02回 research code engineering I：project 構造・config・logging・CLI

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- 実験条件を `config` と CLI に分けて管理できる。
- 実行時に必要な値をログへ残し、後で再確認できる。
- 「何を変える情報を file に置き、何を実行時引数に置くか」を説明できる。

## ミニ講義
- 研究コードでは、設定値と実行コンテキストを分けると再現しやすい。学習率や epoch 数のような実験条件は `config`、出力先や実行対象のような実行ごとに変わる値は CLI に置くと整理しやすい。
- logging は「動いた証拠」ではなく、「どの条件で動いたかの記録」として使う。後からログだけを見ても実行条件が追えることが重要になる。
- `outputs/` のような生成物ディレクトリは、コード内で自動作成しておくと再実行しやすい。手作業依存を減らすことは、小さい実験でも重要である。

## 演習
### 基礎レベル
1. `session02_config.json` を作成し、`lr`, `epochs`, `name` を設定として保存する。値は `0.001`, `3`, `session02-demo` とする。
2. `session02_cli_logging_demo.py` を作成し、`argparse`, `json`, `logging`, `pathlib.Path` を使って `--config` と `--out` を受け取る CLI script にする。出力先ディレクトリを作成し、読み込んだ設定と出力先を log に残す。
3. 同じ config を使って、`outputs/session02_run` と `outputs/session02_run_alt` の 2 通りで実行する。2 回の実行で何が同じで何が変わったかを確認する。
4. `session02_run_log.md` を作成し、`## config の内容`, `## 実行コマンド`, `## ログに残った値`, `## config と CLI の分担` の 4 見出しを書く。`lr` と `epochs` を config に置き、`out` を CLI に置いた理由を自分の言葉で説明する。

### 発展レベル
1. config file が存在しない場合に、Python の traceback だけで終わらないよう明示的なエラーメッセージを追加する。どの例外を捕まえたかも report に書く。
2. `session02_run_log.md` に `## 再実行しやすさの確認` を追加し、別の人が同じ結果を再実行するために必要な情報と、まだ足りない情報を分けて書く。

## 確認ポイント
- `session02_config.json` が指定どおりの内容になっている。
- script が `--config` と `--out` を受け取り、`outputs/session02_run` と `outputs/session02_run_alt` を作成できる。
- ログに `lr=0.001`, `epochs=3`, `name=session02-demo` が含まれている。
- `session02_run_log.md` に、config と CLI の役割分担について自分の説明がある。

## 詰まったときに見る資料
- [`01-annual-policy-and-roles.md`](01-annual-policy-and-roles.md)
- Python docs: [`argparse`](https://docs.python.org/3/library/argparse.html), [`json`](https://docs.python.org/3/library/json.html), [`logging`](https://docs.python.org/3/library/logging.html), [`pathlib`](https://docs.python.org/3/library/pathlib.html)
