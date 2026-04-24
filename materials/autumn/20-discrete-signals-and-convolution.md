# 第20回 離散信号・畳み込み・フィルタの基礎

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 1D / 2D 畳み込みを PyTorch で実行できる。
- kernel、padding、出力 shape の関係を説明できる。
- padding の違いが境界付近の結果へどう影響するかを説明できる。

## ミニ講義
- 畳み込みは、入力上を kernel が滑りながら局所的な加重和を取る操作と考えると理解しやすい。1D では時系列、2D では画像に適用できる。
- kernel は何を強調したいかで変わる。今回の 1D では差分的な edge 検出、2D では平均化による平滑化を扱う。
- padding を入れると出力 shape を保ちやすくなるが、境界では仮想的な値を使うことになる。そのため、shape だけでなく境界の見え方にも影響する。

## 演習
### 基礎レベル
1. `session20_convolution_basics.py` を作成し、指定された 1D signal と edge kernel に `F.conv1d` を適用する。入力、kernel、出力の shape を表示する。
2. 指定された 2D image と平均化 kernel に `F.conv2d` を適用する。入力、kernel、出力の shape を表示する。
3. 1D の入力と出力を `outputs/figures/session20_conv1d.png`、2D の入力と出力を `outputs/figures/session20_conv2d.png` に保存する。
4. `session20_convolution_report.md` に `## conv1d の結果`, `## conv2d の結果`, `## padding を 1 にした理由` を書く。

### 発展レベル
1. `padding=0` の条件でも 1D/2D convolution を計算し、output shape と境界付近の値を比較する。
2. report に `## padding あり / なしの比較` を追加し、padding によって保たれるものと、新たに仮定される境界の値を説明する。

## 確認ポイント
- `edge.shape` が `(1, 1, 5)`、`smoothed.shape` が `(1, 1, 3, 3)` である。
- `session20_conv1d.png` に 2 本の線、`session20_conv2d.png` に 2 枚の画像がある。
- report に、shape の比較だけでなく padding の意味に関する説明がある。

## 詰まったときに見る資料
- [`../../latex/markdown/ch13-basics-of-signal-processing.md`](../../latex/markdown/ch13-basics-of-signal-processing.md)
- [`../../latex/markdown/ch15-basics-of-lti-systems.md`](../../latex/markdown/ch15-basics-of-lti-systems.md)
