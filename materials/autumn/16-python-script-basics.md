# 第16回 Python script の基礎：変数・list・dict・関数・例外・Path・ファイル保存

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- Python script を 1 本作り、コマンドラインから実行できる。
- list、dict、for、if、関数を使って、小さな処理を読みやすく分けられる。
- `try` / `except` で例外を確認し、`pathlib.Path` で出力先を作って結果を保存できる。

## ミニ講義
- この授業では、毎回 `.py` script を作り、実行コマンド、出力、保存物を残す。画面に表示して終わるのではなく、後から見直せる file を作ることが重要である。
- list は順序のある値の集まり、dict は名前と値の対応を持つ集まりである。for と if を組み合わせると、値の変換や集計を短く書ける。
- 関数は処理に名前を付けるために使う。例外処理は失敗を隠すためではなく、どこで何が失敗したかを確認して記録するために使う。
- `pathlib.Path` を使うと、出力先 directory の作成や file path の組み立てを読みやすく書ける。

## 演習
### 基礎レベル
1. `session16_python_script_basics.py` を作成し、Python の list に対して `len`, `max`, `min`, `sum`, `sorted` を使う。`a = [4, 8, 3, 4, 1]` を題材にし、各結果と型を表示する。
2. 同じ `a` について、先頭削除、末尾削除、末尾への `100` 追加、偶奇を 0/1 に変換する list comprehension、奇数の個数の計算を実装する。
3. `scores = {"alice": 82, "bob": 61, "carol": 95}` を使い、合格者だけを取り出す関数 `filter_passed(scores, threshold)` を定義する。`threshold=70` の結果を表示する。
4. `2 // 0` のように例外が出る処理を `try` / `except` で確認し、例外名と message を表示する。
5. `Path("outputs/text")` を作成し、list 操作、dict 操作、例外確認の結果を `outputs/text/session16_summary.txt` に保存する。
6. `session16_python_script_report.md` を作成し、`## 実行コマンド`, `## list 操作`, `## dict と関数`, `## 例外確認`, `## 保存した file` を書く。

### 発展レベル
1. `filter_passed` に空の dict を渡した場合の挙動を確認し、例外を出すべきか、空の結果を返すべきかを理由つきで書く。
2. `outputs/text/session16_summary.txt` をもう一度読み込み、保存した内容と画面に表示した内容が対応しているかを確認する。

## 確認ポイント
- list の組み込み関数、要素操作、list comprehension の結果が表示されている。
- dict を受け取る関数が定義され、`threshold=70` の結果が表示されている。
- 例外名と message が確認されている。
- `outputs/text/session16_summary.txt` が保存されている。
- report に、実行コマンド、処理結果、保存 file が書かれている。

## 詰まったときに見る資料
- [`../../latex/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md`](../../latex/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md)
- [`../../latex/markdown/ch01-basic-operations.md`](../../latex/markdown/ch01-basic-operations.md)
