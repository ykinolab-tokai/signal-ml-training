# 第02回 Python 基礎

- 変数に値を入れ，型と値を確認する。
- 関数，条件分岐，繰り返しを組み合わせて小さな処理を書く。
- list，dict，スライスを使って複数の値を扱う。
- 実行結果を report として残し，後から確認できる状態にする。

## 解説
- 変数は値に名前を付ける仕組みである。型を確認すると，数値，文字列，list，dict のどれとして処理されているかを切り分けやすい。
- 関数は処理に名前を付け，再利用しやすくするために使う。入力，引数，返り値を意識すると，処理の境界が明確になる。
- 条件分岐は場合分け，繰り返しは同じ処理の反復に使う。両者を list や dict と組み合わせると，データの変換や集計を簡潔に書ける。
- スライスは連続した範囲を取り出す表記である。開始，終了，刻み幅の意味を確認すると，NumPy 配列や信号処理の切り出しにもつながる。

## 演習
### 基礎レベル（7問）
1. `scripts/` と `outputs/session02/` を作成し，`scripts/session02_python_basics.py` に整数，浮動小数点数，文字列，真偽値を変数として書く。`type()` と `print()` で型と値を表示する。
2. `celsius_to_fahrenheit(celsius)` と `fahrenheit_to_celsius(fahrenheit)` を定義し，3 つ以上の入力で結果を確認する。
3. 点数を受け取り，80 点以上を `A`，60 点以上を `B`，それ未満を `C` と返す関数 `grade(score)` を作る。
4. `scores = [82, 61, 95, 48, 77]` に対して，for 文で合計，平均，最大値を計算する。組み込み関数を使った結果とも比較する。
5. `scores` から先頭3件，末尾2件，1つおきの値，逆順をスライスで取り出す。
6. `students = {"alice": 82, "bob": 61, "carol": 95}` を使い，70 点以上の学生名だけを list にする。
7. `outputs/session02/session02_report.md` に，実行コマンド，関数の入力と出力，list と dict の操作結果を記録する。

### 発展レベル（7問）
1. `grade(score)` に 0 未満または 100 超の値が渡された場合の扱いを決め，実装する。error にするか補正するか，理由も書く。
2. `scores` の平均以上の値だけを残す関数 `filter_above_average(values)` を作る。
3. `students` を `{"alice": {"score": 82, "passed": True}}` のような形に変換し，合否を dict の値として持たせる。
4. `enumerate` を使い，`scores` の index と値を並べて表示する。
5. `zip` を使い，`names = ["alice", "bob", "carol"]` と `scores = [82, 61, 95]` から dict を作る。
6. list を関数に渡して中身を変更する例と，新しい list を返す例を作り，違いを説明する。
7. ここまで作った関数に対して `assert` を 5 個以上書き，想定した入力で正しい値が返ることを確認する。

## 詰まったときに見る資料
- [`../textbook/markdown/ch02-basics-of-python.md`](../textbook/markdown/ch02-basics-of-python.md)
- [Python tutorial](https://docs.python.org/3/tutorial/)
