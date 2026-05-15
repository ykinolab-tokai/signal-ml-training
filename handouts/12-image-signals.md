# 第12回 画像信号

- 2次元信号として画像を扱い，shape と値域を確認する。
- Pillow と Matplotlib で画像を読み込み，表示，保存する。
- 幾何変換，2次元 DFT，2次元フィルタを実装する。
- 空間領域と周波数領域の見え方を対応づける。

## 解説
- 画像は2次元または3次元の配列として扱える。grayscale 画像は `(H, W)`，RGB 画像は `(H, W, C)` の形で表されることが多い。
- 読み込み時の dtype と値域を確認しないと，正規化や保存時に意図しない見え方になる。`uint8` の 0〜255 と float の 0〜1 を区別する。
- 幾何変換は，画素の位置を変える処理である。resize，crop，rotate では補間方法によって結果が変わる。
- 2次元 DFT は，画像を空間周波数の成分として見る方法である。低周波は滑らかな変化，高周波は細かい変化やエッジに対応しやすい。

## 演習
### 基礎レベル（7問）
1. `scripts/`，`outputs/session12/`，`outputs/images/`，`outputs/figures/` を作成し，`scripts/session12_image_signal.py` で縦横の勾配を持つ `128 x 128` の grayscale 画像を NumPy で作る。
2. 作成した画像を `Pillow` で `outputs/images/session12_gradient.png` に保存し，再度読み込んで shape，dtype，値域を表示する。
3. Matplotlib で grayscale 画像を表示し，colorbar 付きで `outputs/figures/session12_gradient_display.png` に保存する。
4. resize，crop，rotate を実行し，それぞれの結果を保存する。
5. 2次元 DFT を計算し，`fftshift` 後の log amplitude spectrum を保存する。
6. 周波数領域で単純な低域通過フィルタを作り，逆変換して画像を復元する。
7. 空間領域で平均化フィルタを適用し，周波数領域の低域通過フィルタと見え方を比較する。

### 発展レベル（7問）
1. RGB 画像を自作し，channel ごとの shape と値域を確認する。
2. resize の補間方法を nearest，bilinear，bicubic で比較し，境界の見え方を書く。
3. 回転角度を 15，45，90 度に変え，画像サイズと余白の扱いを確認する。
4. checkerboard pattern を作り，2次元 DFT の spectrum がどのように現れるか確認する。
5. 高域通過フィルタを作り，エッジが強調されるか確認する。
6. 2次元畳み込みで sharpening filter を実装し，平均化フィルタと比較する。
7. 画像の読み込み，変換，周波数解析，フィルタ処理の流れを，入力 file と出力 file の対応表として整理する。

## 詰まったときに見る資料
- [`../textbook/markdown/ch12-introduction-to-pillow.md`](../textbook/markdown/ch12-introduction-to-pillow.md)
- [`../textbook/markdown/ch14-basics-of-spectrum-analysis.md`](../textbook/markdown/ch14-basics-of-spectrum-analysis.md)
- [Pillow handbook](https://pillow.readthedocs.io/en/stable/handbook/index.html)
- [Matplotlib image tutorial](https://matplotlib.org/stable/tutorials/images.html)
