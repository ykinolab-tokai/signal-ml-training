# 発展テーマ候補：画像 segmentation

- 対象: B4・M
- 種別: 第 25〜27 回の年度別発展テーマ候補
- 運用: このテーマを選んだ年度は、第 25〜27 回の 3 回を使って `segmentation` を扱う。画像・音響・共通基盤を同じ年度にすべて扱う前提にはしない。

## この回の目標
- segmentation の入力画像、正解 mask、出力 mask、loss の対応を説明できる。
- 最小の segmentation pipeline を実装し、予測 mask を保存できる。
- pixel accuracy と Dice の違いを、予測 mask の評価観点として説明できる。

## 解説
- segmentation では、1 枚の画像に対して 1 クラスを出すのではなく、各画素ごとに出力を持つ。だから model 出力も `(batch, channels, H, W)` になりやすい。
- `BCEWithLogitsLoss` は binary mask を扱うときによく使う。出力側は sigmoid 前の logits、正解側は 0/1 mask を用意する。
- 予測が良いかを見るとき、画素一致率だけだと背景優勢な場合に高く見えやすい。Dice のように重なりを見る指標も併せて確認すると解釈しやすい。

### 固定データと学習条件

`import torch`, `from torch import nn`, `from torch.utils.data import Dataset, DataLoader` を使い、モデル作成前に `torch.manual_seed(11)` を設定する。
`SquareSegDataset` の各サンプルは、入力・正解ともに `float32` の `(1, 32, 32)` Tensorとする。
正解maskは0で初期化し、`mask[:, 10:22, 10:22] = 1` で中央12×12画素だけを1にする。入力は `mask.clone()` とする。
同じ内容の10件を返し、`DataLoader(dataset, batch_size=2, shuffle=True)` で学習する。
これは入出力とlossの接続確認用データであり、未知の画像への性能を測るデータではない。

モデルは `nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))` とする。
`nn.BCEWithLogitsLoss()` と `torch.optim.Adam(model.parameters(), lr=1e-2)` を用い、10 epoch学習する。
1 epochはDataLoaderを最後まで1巡することで、今回は5バッチの更新を10巡、計50 step行う。
各バッチで第19回の順序に従って勾配を消去し、順伝播、loss、逆伝播、更新を行う。
学習時は `model.train()`、推論時は `model.eval()` と `torch.no_grad()` を使う。

推論は先頭サンプルにバッチ軸を加えて行い、logitsに `torch.sigmoid` を適用した後、`>= 0.5` で二値化する。
予測と正解をbool配列として、全画素での一致数を画素数で割った値をpixel accuracyとする。
Diceは $2|P\cap Y|/(|P|+|Y|)$ とする（$P,Y$ は予測・正解の陽性画素集合）。両方が空なら1と定義する。
`outputs/figures/` を作成し、先頭サンプルの入力・正解・二値予測を同じ図に保存する。
発展では再学習せず、中央領域を `[14:18, 14:18]` の4×4画素にした入力・正解を追加し、同じモデルで比較する。

## 演習

作業場所・保存先・説明の残し方は [共通手順](../README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

このテーマでは，複数条件の結果と解釈をまとめて参照するため，下で指定する比較レポートを作る。数値や図は保存先を参照し，既存ファイルの内容を転記しない。第26回の比較記録からもこのレポートを参照する。

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
- [`21-image-model-basics.md`](../21-image-model-basics.md)
- [`19-autodiff-and-optimization.md`](../19-autodiff-and-optimization.md): 学習ループ
