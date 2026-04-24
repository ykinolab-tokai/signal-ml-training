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
1. `session02_config.json` を作成し、次の内容をそのまま書く。
```json
{
  "lr": 0.001,
  "epochs": 3,
  "name": "session02-demo"
}
```
2. `session02_cli_logging_demo.py` を作成し、`--config` と `--out` を受け取る CLI script を実装する。
3. script には次の処理を入れる。
   - `argparse` で `--config` と `--out` を受け取る。
   - `json` で設定ファイルを読む。
   - `Path(args.out).mkdir(parents=True, exist_ok=True)` を実行する。
   - `logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")` を設定する。
   - `lr`, `epochs`, `name`, `out` を `logging.info()` で出力する。
4. 次のコマンドで script を実行する。
```bash
python3 session02_cli_logging_demo.py --config session02_config.json --out outputs/session02_run
```
5. `session02_run_log.md` を作成し、次の 3 見出しをこの順で書く。
   - `## config の内容`
   - `## 実行コマンド`
   - `## ログに残った値`
6. `## ログに残った値` には `lr`, `epochs`, `name`, `out` が表示されたことを書く。
7. `session02_run_log.md` の最後に 2 行追加し、`lr` と `epochs` を config に置き、`out` を CLI に置いた理由を書く。

### 発展レベル
1. `session02_cli_logging_demo.py` をそのまま使い、出力先だけを変えて 2 回目の実行を行う。
```bash
python3 session02_cli_logging_demo.py --config session02_config.json --out outputs/session02_run_alt
```
2. `session02_run_log.md` に `## 2 回の実行をどう区別するか` を追加し、2 回の実行で変わった値と変わらない値を 1 行ずつ書く。
3. `session02_run_log.md` に `## config と CLI の分担` を追加し、`name` を config 側に置く理由と、`out` を CLI 側に置く理由を 3 行以内で説明する。
4. 可能なら `session02_cli_logging_demo.py` に設定ファイルが存在しない場合の明示的なエラーメッセージを追加し、なぜそのチェックが必要かを 2 行で書く。

## 確認ポイント
- `session02_config.json` が指定どおりの内容になっている。
- script が `--config` と `--out` を受け取り、`outputs/session02_run` と `outputs/session02_run_alt` を作成できる。
- ログに `lr=0.001`, `epochs=3`, `name=session02-demo` が含まれている。
- `session02_run_log.md` に、config と CLI の役割分担について自分の説明がある。

## 詰まったときに見る資料
- [`01-annual-policy-and-roles.md`](01-annual-policy-and-roles.md)
- Python docs: [`argparse`](https://docs.python.org/3/library/argparse.html), [`json`](https://docs.python.org/3/library/json.html), [`logging`](https://docs.python.org/3/library/logging.html), [`pathlib`](https://docs.python.org/3/library/pathlib.html)
