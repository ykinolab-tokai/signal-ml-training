# 第09回 線形時不変システムと畳み込み

- 線形時不変システムの線形性と時不変性を数値実験で確認する。
- 畳み込みを手計算に近い形と NumPy の関数で確認する。
- 畳み込み定理を FFT で確認する。
- フィルタと周波数応答の関係を理解する。

## 解説
- システムは入力信号を出力信号へ変換する規則である。線形性は重ね合わせが成り立つこと，時不変性は入力を遅らせると出力も同じだけ遅れることを表す。
- LTI システムでは，インパルス応答が分かれば任意の入力への出力を畳み込みで求められる。
- 畳み込みは，片方の信号を反転してずらしながら重なりを足し合わせる操作である。移動平均や差分フィルタも畳み込みとして書ける。
- 畳み込み定理は，時間領域の畳み込みが周波数領域の積に対応することを示す。フィルタの周波数応答を見ると，どの周波数が通るかを判断できる。

## 演習
### 基礎レベル（7問）
1. `scripts/`，`outputs/session09/`，`outputs/figures/` を作成し，`scripts/session09_lti_convolution.py` で短い信号 `x = [1, 2, 0, 1]` と impulse response `h = [1, -1, 0.5]` を定義する。
2. 畳み込みを for 文で実装し，`np.convolve(x, h)` と一致するか確認する。
3. `h = [1/3, 1/3, 1/3]` の移動平均フィルタを sine 波とノイズの和に適用し，前後の波形を描く。
4. 線形性 `T(a*x1 + b*x2) = a*T(x1) + b*T(x2)` を，移動平均フィルタで数値的に確認する。
5. 時不変性を確認するため，入力を5 sample 遅らせた場合の出力と，出力を5 sample 遅らせた場合を比較する。
6. FFT を使って畳み込みを周波数領域の積で計算し，時間領域の畳み込みと比較する。
7. 移動平均フィルタの周波数応答を描き，低周波と高周波のどちらを通しやすいか書く。

### 発展レベル（7問）
1. 差分フィルタ `h = [1, -1]` の周波数応答を描き，移動平均フィルタと比較する。
2. `np.convolve` の `full`，`same`，`valid` を比較し，出力長が変わる理由を書く。
3. 畳み込み定理を確認するときに zero padding が必要になる理由を，循環畳み込みとの違いに触れて説明する。
4. 複数の移動平均長を比較し，filter length が長くなると周波数応答がどう変わるか図にする。
5. impulse 入力を入れたとき，出力が impulse response そのものになることを確認する。
6. 周波数応答の振幅と位相を別々に描く。
7. 自分で `apply_fir_filter(x, h, mode="same")` を定義し，第10回のノイズ除去で再利用できる形にする。

## 詰まったときに見る資料
- [`../textbook/markdown/ch15-basics-of-lti-systems.md`](../textbook/markdown/ch15-basics-of-lti-systems.md)
- [`../textbook/markdown/ch14-basics-of-spectrum-analysis.md`](../textbook/markdown/ch14-basics-of-spectrum-analysis.md)
- [NumPy convolution](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
