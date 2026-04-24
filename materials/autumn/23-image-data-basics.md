# 第23回 画像データの基礎：画像を配列として扱う・正規化・リサイズ・簡単なフィルタ

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
1. `session23_image_data_basics.py` を作成し、指定された 64x96 RGB 画像を生成して `outputs/images/session23_input.png` に保存する。
2. `PILToTensor` で `(C, H, W)` tensor に変換し、shape, dtype, min, max を表示する。
3. bilinear resize、`(-1, 1)` 正規化、3x3 平均化フィルタを順に適用し、`outputs/images/session23_resized.png` と `outputs/images/session23_filtered.png` を保存する。
4. `session23_image_data_report.md` に `## 入力画像の shape と値域`, `## リサイズと正規化`, `## フィルタ後の変化` を書き、画像保存形式と PyTorch tensor 形式の違いを説明する。

### 発展レベル
1. nearest resize も実行し、bilinear との見え方の違いを比較する。
2. report に `## 補間方法の比較` を追加し、3 色ブロック画像で境界を保ちやすい補間方法と、平均化フィルタによる変化が補間と違う理由を説明する。

## 確認ポイント
- `session23_input.png` が赤・緑・青の 3 領域を持つ。
- `x.shape` が `(3, 64, 96)`、`x_resized.shape` が `(3, 128, 128)` である。
- `session23_resized.png` と `session23_filtered.png` が保存されている。
- report に、リサイズとフィルタの違いが見え方の変化として説明されている。

## 詰まったときに見る資料
- [`../../latex/markdown/ch12-introduction-to-pillow.md`](../../latex/markdown/ch12-introduction-to-pillow.md)
- Torchvision docs: [`PILToTensor`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.PILToTensor.html)
