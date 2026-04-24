# 第22回 画像データの基礎：画像を配列として扱う・正規化・リサイズ・簡単なフィルタ

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- RGB 画像を自分で生成し、`(C, H, W)` tensor に変換できる。
- リサイズ、正規化、平均化フィルタを順番に適用できる。
- 補間方法やフィルタが画像の見え方にどう影響するかを説明できる。

## ミニ講義
- 画像は保存形式では `(H, W, C)`、PyTorch 側では `(C, H, W)` で扱うことが多い。shape を意識しないと前処理で混乱しやすい。
- 正規化は model に入れやすい値域へ揃える操作であり、画像そのものの意味を変えるためではない。今回は `(-1, 1)` に写す。
- リサイズでは補間方法によって境界の見え方が変わる。平均化フィルタも同様に、急な境界をなだらかにする方向へ働く。

## 演習
### 基礎レベル
1. `session22_image_data_basics.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
from torchvision.transforms.functional import to_pil_image

out_dir = Path("outputs/images")
out_dir.mkdir(parents=True, exist_ok=True)
```
2. 次の規則で 64x96 の RGB 画像を生成し、`outputs/images/session22_input.png` に保存する。
```python
img_np = np.zeros((64, 96, 3), dtype=np.uint8)
img_np[:, :48, 0] = 255
img_np[:32, 48:, 1] = 255
img_np[32:, 48:, 2] = 255
img = Image.fromarray(img_np, mode="RGB")
img.save(out_dir / "session22_input.png")
```
3. `x = PILToTensor()(img).float() / 255.0` で tensor に変換し、`x.shape`, `x.dtype`, `x.min()`, `x.max()` を表示する。
4. `F.interpolate(..., size=(128, 128), mode="bilinear", align_corners=False)` を使って `x_resized` を作る。
5. `x_norm = (x_resized - 0.5) / 0.5` で `(-1, 1)` へ正規化する。
6. 次の 3x3 平均化フィルタで `x_filtered` を計算する。
```python
kernel = torch.ones((3, 1, 3, 3), dtype=torch.float32) / 9.0
x_filtered = F.conv2d(x_norm.unsqueeze(0), kernel, padding=1, groups=3).squeeze(0)
```
7. `x_resized` を `outputs/images/session22_resized.png`、表示範囲へ戻した `x_filtered` を `outputs/images/session22_filtered.png` に保存する。
8. `session22_image_data_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 入力画像の shape と値域`
   - `## リサイズと正規化`
   - `## フィルタ後の変化`

### 発展レベル
1. `mode="nearest"` でも 128x128 リサイズを計算し、`x_resized_nearest` を作る。
2. `session22_image_data_report.md` に `## 補間方法の比較` を追加し、`bilinear` と `nearest` の見え方の違いを 3 行以内で書く。
3. そのうえで、「今回の 3 色ブロック画像ではどちらの補間が境界を保ちやすいか」を 2 行以内で説明する。
4. 平均化フィルタ後の画像を見て、補間による差とフィルタによる差が同じではない理由を 2 行で書く。

## 確認ポイント
- `session22_input.png` が赤・緑・青の 3 領域を持つ。
- `x.shape` が `(3, 64, 96)`、`x_resized.shape` が `(3, 128, 128)` である。
- `session22_resized.png` と `session22_filtered.png` が保存されている。
- report に、リサイズとフィルタの違いが見え方の変化として説明されている。

## 詰まったときに見る資料
- [`../../latex/markdown/ch12-introduction-to-pillow.md`](../../latex/markdown/ch12-introduction-to-pillow.md)
- Torchvision docs: [`PILToTensor`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.PILToTensor.html)
