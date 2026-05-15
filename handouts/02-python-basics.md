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

この回の演習ファイルは，このリポジトリ直下の `exercise/` 以下に置く。ファイル名は `exc回2桁_問題番号2桁.py` の形式に統一する。第2回の第1問であれば，`exercise/exc02_01.py` とする。問題番号はこの回の通し番号とし，基礎レベルで `exc02_01.py` から `exc02_07.py`，発展レベルで `exc02_08.py` から `exc02_14.py` を使う。

### 基礎レベル（7問）
1. Python には数多くの組み込み関数が定義されている。組み込み関数を適切に使うことで，コードを簡潔に書ける。`a = [4, 8, 3, 4, 1]` というリストに対して以下の操作を行う組み込み関数を [公式ドキュメント](https://docs.python.org/ja/3/library/functions.html) からそれぞれ探し，適用するコードを `exercise/exc02_01.py` に書く。

    - リスト `a` の長さを求める。
    - リスト `a` に含まれる値の最大値を求める。
    - リスト `a` に含まれる値の最小値を求める。
    - リスト `a` に含まれる値の合計値を求める。
    - リスト `a` をソートして，`[1, 3, 4, 4, 8]` というリストを返す。

    ```python
    # 以下の FUNC という部分を組み込み関数に置き換える。

    a = [4, 8, 3, 4, 1]
    res = FUNC(a)
    print(res)
    ```

2. 以下の演算や関数をそれぞれ評価したとき，結果の値と型が何になるか予想する。`exercise/exc02_02.py` にコードを書き，実際に実行して，結果が予想と合うか確かめる。型の確認には `type()` 関数を使う。`2 // 0` は値を返さず例外を発生させるため，例外名とメッセージを確認する。

    - `1.2 + 3.8`
    - `10 // 100`
    - `1 >= 0`
    - `'Hello World' == 'Hello World'`
    - `not 'Chainer' != 'Tutorial'`
    - `all([True, True, False])`（`all` の定義は [公式ドキュメント](https://docs.python.org/ja/3/library/functions.html#all) を参照）
    - `any([True, True, False])`（`any` の定義は [公式ドキュメント](https://docs.python.org/ja/3/library/functions.html#any) を参照）
    - `abs(-3)`（`abs` の定義は [公式ドキュメント](https://docs.python.org/ja/3/library/functions.html#abs) を参照）
    - `2 // 0`

3. 機械学習では大量のデータを扱うことが多いため，リストの扱いに慣れておくと便利である。`a = [4, 8, 3, 4, 1]` というリストに対して以下の操作を行うコードを `exercise/exc02_03.py` に書く。

    - リスト `a` の先頭の要素を取り除いて，`[8, 3, 4, 1]` となるようにする。
    - リスト `a` の末尾の要素を取り除いて，`[4, 8, 3, 4]` となるようにする。
    - リスト `a` の末尾に `100` という値を追加して，`[4, 8, 3, 4, 1, 100]` となるようにする。

4. リスト内包表記は，リストを生成するための簡潔な手段である。例えば，平方数のリスト `[0, 1, 4, 9, 16, ...]` を構成したいとき，

    ```python
    squares = []
    for x in range(10):
        squares.append(x ** 2)
    ```

    と書く代わりに，

    ```python
    squares = [x ** 2 for x in range(10)]
    ```

    と書ける。リスト内包表記は一般に `[(式) for (変数) in (iterableオブジェクト)]` という構文を取る。リスト内包表記の説明は [公式ドキュメント](https://docs.python.org/ja/3/tutorial/datastructures.html#list-comprehensions) を参照する。`in` の後に続く iterable オブジェクトは `range` である必要はなく，例えばリストを指定できる。

    ```python
    a = [4, 8, 3, 4, 1]
    squares = [x ** 2 for x in a]
    print(squares)  # => [16, 64, 9, 16, 1]
    ```

    以下の問いの答えとなるコードを `exercise/exc02_04.py` に書く。

    1. `a = [4, 8, 3, 4, 1]` というリストに対し，要素が偶数なら `0`，奇数なら `1` に変換するコードをリスト内包表記で書く。この結果，リストは `[0, 0, 1, 0, 1]` に変換される。
    2. 1. で書いたコードと組み込み関数を組み合わせて，リスト `a` に含まれる奇数の個数を数えるコードを書く。
    3. リスト内包表記では，`if` 文を使うことで条件を満たす要素だけをリストに残せる。例えば，`b = [x for x in range(10) if x > 5]` と書くと，`b` は `[6, 7, 8, 9]` になる。リスト内包表記を使って，リスト `a` から奇数の要素だけを残すコードを書く。

5. Python では文字列型に対して便利な組み込み関数が多数定義されている。文字列型の機能については [公式ドキュメント](https://docs.python.org/ja/3/library/stdtypes.html#text-sequence-type-str) を参照する。以下の処理を行うプログラムを `exercise/exc02_05.py` に書く。

    1. [str.join()](https://docs.python.org/ja/3/library/stdtypes.html#str.join) を使って，`0` から `99` までの数をスペース区切りで並べた文字列 `"0 1 2 3 4 ... 99"` を構成する。
    2. [str.format()](https://docs.python.org/ja/3/library/stdtypes.html#str.format) を使って，`float` の値 `1.0 / 7.0` の小数点以下9桁までを表示する。

6. 関数と辞書を組み合わせる練習として，学生名と点数を扱う。`scores = {"alice": 82, "bob": 61, "carol": 95, "dave": 58}` に対して，以下の処理を行うコードを `exercise/exc02_06.py` に書く。

    1. 点数 `score` を受け取り，70 点以上なら `True`，70 点未満なら `False` を返す関数 `is_passed(score)` を定義する。
    2. `scores` の各学生について，合格した学生名だけをリストとして作る。
    3. `scores` の各学生について，`{"score": 点数, "passed": 合否}` という形の辞書に変換する。
    4. 作成した合格者リストと変換後の辞書を表示する。

7. NumPy Array，PyTorch Tensor，`nn.Module` などを理解するには，Python のクラスに慣れておく必要がある。クラスを実装する練習として，データを管理するクラスを実装する。次のメソッドをすべて持つクラス `DataManager` を `exercise/exc02_07.py` に書く。

    - `__init__(self, x, y, z)`: 3つの数 `x`，`y`，`z` を受け取り，インスタンスの属性としてそれぞれの値を記憶する。
    - `add_x(self, delta)`: `x` に `delta` だけ足して，値を更新する。
    - `add_y(self, delta)`: `y` に `delta` だけ足して，値を更新する。
    - `add_z(self, delta)`: `z` に `delta` だけ足して，値を更新する。
    - `sum(self)`: `x`，`y`，`z` の3つの数の合計値を返す。

    このクラスを使って，以下のようなコードが書けるものとする。

    ```python
    data_manager = DataManager(2, 3, 5)
    print(data_manager.sum())  # => 10
    data_manager.add_x(4)      # => data_manager.x の値が 2 から 6 に更新される
    print(data_manager.sum())  # => 14
    data_manager.add_y(0)      # => data_manager.y の値が 3 から 3 に更新される
    print(data_manager.sum())  # => 14
    data_manager.add_z(-9)     # => data_manager.z の値が 5 から -4 に更新される
    print(data_manager.sum())  # => 5
    ```

### 発展レベル（7問）
1. リストと辞書を使って，複数回の測定結果を整理する。次の `records` から，学生ごとの平均点を計算するコードを `exercise/exc02_08.py` に書く。

    ```python
    records = [
        {"name": "alice", "score": 82},
        {"name": "bob", "score": 61},
        {"name": "alice", "score": 90},
        {"name": "bob", "score": 70},
        {"name": "carol", "score": 95},
    ]
    ```

    出力は `{"alice": 86.0, "bob": 65.5, "carol": 95.0}` のように，名前を key，平均点を value とする辞書にする。for 文を使って合計と件数を集計し，最後に平均を計算する。

2. 以下で定義される関数 `f`，`g` があるとする。

    ```python
    def f(a):
        a = [6, 7, 8]

    def g(a):
        a.append(1)
    ```

    これらに対し，次のコードの実行結果がどうなるか予想する。`exercise/exc02_09.py` にコードを書き，実行結果が予想と一致しているか確認し，なぜそのような結果になったのか考察する。考察は `print()` で表示しても，コメントとして書いてもよい。

    ```python
    def somefunction():
        a0 = [1, 2, 3]
        f(a0)
        print(a0)

        a1 = [1, 2, 3]
        g(a1)
        print(a1)

    somefunction()
    ```

3. 2以上の整数 `p` が素数であるとは，「どんな `2` 以上 `p - 1` 以下の整数 `k` に対しても `p` は `k` で割り切れない」が成り立つことを指す。素数を小さい順から列挙すると，`2`，`3`，`5`，`7`，`11`，`13`，`17`，... となる。制御構文である `if` や `for` を用いて，`2` から `100` までに含まれる素数を列挙するプログラムを `exercise/exc02_10.py` に書く。

4. 第3問で作った素数列挙を関数に分ける。`exercise/exc02_11.py` に，整数 `n` を受け取り，`n` が素数なら `True`，そうでなければ `False` を返す関数 `is_prime(n)` を定義する。その関数を使って，`2` から `200` までの素数をリストとして表示する。さらに，`1`，`2`，`97`，`100` を入力したときの返り値を確認する。

5. 文字列のリストから出現回数を数える。`words = ["cat", "dog", "cat", "bird", "dog", "cat"]` に対して，各単語が何回現れたかを辞書として集計するコードを `exercise/exc02_12.py` に書く。出力は `{"cat": 3, "dog": 2, "bird": 1}` のような形にする。最後に，出現回数が2回以上の単語だけをリストとして表示する。

6. `DataManager` を拡張する。`exercise/exc02_13.py` に，`DataManager` の値を変更するたびに，変更前後の値を `history` というリストに記録するクラスを実装する。`add_x(4)` を実行した場合は，少なくとも「どの属性を」「いくつからいくつへ」変更したかが後から分かるようにする。最後に `history` の中身を表示する。

7. 第2回の内容を組み合わせた小さな集計プログラムを作る。`exercise/exc02_14.py` に，次の `scores` から平均点，最高点，最低点，合格者名，学生ごとの合否，入力値の妥当性を表示するコードを書く。

    ```python
    scores = {"alice": 82, "bob": 61, "carol": 95, "dave": 58, "ellen": 70, "frank": -5}
    ```

    合格条件は 70 点以上とする。平均点，最高点，最低点には組み込み関数を使い，合格者名と合否辞書の作成には for 文またはリスト内包表記を使う。ただし，点数が 0 以上 100 以下でない学生は無効データとして扱い，平均点，最高点，最低点，合否判定から除外する。最後に，無効データの学生名も表示する。

## 詰まったときに見る資料
- [`../textbook/markdown/ch02-basics-of-python.md`](../textbook/markdown/ch02-basics-of-python.md)
- [Python tutorial](https://docs.python.org/3/tutorial/)
