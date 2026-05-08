# 発展テーマ候補：画像 segmentation

- 対象: B4・M
- 種別: 前期 11〜13 回の年度別発展テーマ候補
- 運用: このテーマを選んだ年度は、前期 11〜13 回の 3 回を使って `segmentation` を扱う。画像・音響・共通基盤を同じ年度にすべて扱う前提にはしない。

## この回の目標
- segmentation の入力画像、正解 mask、出力 mask、loss の対応を説明できる。
- 最小の segmentation pipeline を実装し、予測 mask を保存できる。
- pixel accuracy と Dice の違いを、予測 mask の評価観点として説明できる。

## ミニ講義
- segmentation では、1 枚の画像に対して 1 クラスを出すのではなく、各画素ごとに出力を持つ。だから model 出力も `(batch, channels, H, W)` になりやすい。
- `BCEWithLogitsLoss` は binary mask を扱うときによく使う。出力側は sigmoid 前の logits、正解側は 0/1 mask を用意する。
- 予測が良いかを見るとき、画素一致率だけだと背景優勢な場合に高く見えやすい。Dice のように重なりを見る指標も併せて確認すると解釈しやすい。

## 演習
### 基礎レベル
1. `advanced_image_segmentation_demo.py` を作成し、synthetic な画像と binary mask を返す dataset、最小の segmentation model、`BCEWithLogitsLoss` を使った学習 loop を実装する。dataset は 10 サンプルにする。
2. model 出力 shape が `(batch, 1, 32, 32)` になることを確認し、logits, sigmoid 後の確率, threshold 後の mask の違いをコード内で確認する。
3. 予測 mask を `outputs/figures/advanced_image_mask_prediction.png` に保存し、pixel accuracy と Dice を計算する。
4. `advanced_image_segmentation_report.md` に `## データと mask`, `## model 出力`, `## 予測 mask`, `## 評価指標` を書き、見た目・pixel accuracy・Dice の違いを説明する。

### 発展レベル
1. 背景が多い例を 1 つ追加し、pixel accuracy と Dice のどちらが変化を捉えやすいかを比較する。
2. report に `## 指標の比較` を追加し、背景優勢な segmentation で pixel accuracy だけを見る危険を 3 行以内で説明する。

## 確認ポイント
- dataset 長が `10` である。
- model 出力 shape が `(batch, 1, 32, 32)` である。
- `advanced_image_mask_prediction.png` が保存されている。
- report に、見た目だけでなく数値指標を使った評価が書かれている。

## 詰まったときに見る資料
- [`07-image-model-basics.md`](07-image-model-basics.md)
- [`../autumn/27-image-baseline-mini-implementation.md`](../autumn/27-image-baseline-mini-implementation.md)
