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
1. `session20_convolution_basics.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
```
2. 1D 信号と kernel を次の通り定義する。
```python
signal = torch.tensor([[[0.0, 1.0, 2.0, 1.0, 0.0]]])
kernel = torch.tensor([[[1.0, 0.0, -1.0]]])
edge = F.conv1d(signal, kernel, padding=1)
```
3. `signal.shape`, `kernel.shape`, `edge.shape` を表示する。
4. `signal.squeeze()` と `edge.squeeze()` を同じ図に描き、`outputs/figures/session20_conv1d.png` に保存する。
5. 2D 画像と blur kernel を次の通り定義する。
```python
image = torch.tensor([[[[0.0, 1.0, 0.0],
                        [1.0, 2.0, 1.0],
                        [0.0, 1.0, 0.0]]]])
blur = torch.tensor([[[[1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]]]]) / 9.0
smoothed = F.conv2d(image, blur, padding=1)
```
6. `image.shape`, `blur.shape`, `smoothed.shape` を表示する。
7. `image.squeeze()` と `smoothed.squeeze()` を横並び 2 枚の `imshow` で表示し、`outputs/figures/session20_conv2d.png` に保存する。
8. `session20_convolution_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## conv1d の結果`
   - `## conv2d の結果`
   - `## padding を 1 にした理由`

### 発展レベル
1. `padding=0` の条件でも `conv1d` と `conv2d` を計算する。
2. `session20_convolution_report.md` に `## padding あり / なしの比較` を追加し、1D と 2D のそれぞれで output shape がどう変わるかを書く。
3. そのうえで、「padding を 1 にすると何が保たれ、何が新たに仮定されるか」を 3 行以内で説明する。
4. 1D と 2D のどちらか一方について、境界付近の値が中央と違って見える理由を 2 行で書く。

## 確認ポイント
- `edge.shape` が `(1, 1, 5)`、`smoothed.shape` が `(1, 1, 3, 3)` である。
- `session20_conv1d.png` に 2 本の線、`session20_conv2d.png` に 2 枚の画像がある。
- report に、shape の比較だけでなく padding の意味に関する説明がある。

## 詰まったときに見る資料
- [`../../latex/markdown/ch13-basics-of-signal-processing.md`](../../latex/markdown/ch13-basics-of-signal-processing.md)
- [`../../latex/markdown/ch15-basics-of-lti-systems.md`](../../latex/markdown/ch15-basics-of-lti-systems.md)
