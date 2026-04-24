# 第11回 画像深掘り回：segmentation（2026年度版）

- 対象: B4・M
- 種別: 前期発展枠 / ローテーション
- 今年度設定: この file では `segmentation` を扱う。別年度で差し替える場合は file 全体を書き換えてから配布する。

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
1. `session11_segmentation_demo.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

img_dir = Path("outputs/images")
img_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(11)
```
2. `SquareSegDataset` を作り、32x32 の 1 チャンネル画像と 32x32 の binary mask を返すようにする。
3. 各サンプルで中央 12x12 の正方形領域を `1`、それ以外を `0` にした mask を作る。
4. 入力画像は mask と同じ位置に 1 を置き、周囲は 0 にする。
5. dataset 長は `10` とする。
6. `DataLoader(dataset, batch_size=2, shuffle=True)` を作る。
7. model を次の通り定義する。
```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 1, kernel_size=1),
)
```
8. loss は `nn.BCEWithLogitsLoss()`、optimizer は `torch.optim.Adam(model.parameters(), lr=1e-2)` とする。
9. `for epoch in range(10):` で 10 epoch 学習する。
10. dataset の先頭サンプル 1 件で推論し、sigmoid 後の予測 mask を `outputs/images/session11_mask_prediction.png` に保存する。
11. `session11_segmentation_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 入力と正解 mask`
   - `## 学習設定`
   - `## 予測結果`
12. `## 予測結果` には、保存した予測 mask が正解 mask と比べてどう見えるかを 3 行以内で書く。

### 発展レベル
1. 予測 mask を 0.5 threshold で二値化し、pixel accuracy と Dice 係数を計算する。
2. `session11_segmentation_report.md` に `## 評価指標の比較` を追加し、2 つの指標の値を書く。
3. そのうえで、「今回のデータではどちらの指標が予測の良し悪しを表しやすいか」を 3 行以内で説明する。
4. M は、mask が極端に小さい場合に pixel accuracy が当てになりにくい理由を 2 行で書く。

## 確認ポイント
- dataset 長が `10` である。
- model 出力 shape が `(batch, 1, 32, 32)` である。
- `session11_mask_prediction.png` が保存されている。
- report に、見た目だけでなく数値指標を使った評価が書かれている。

## 詰まったときに見る資料
- [`07-image-model-basics.md`](07-image-model-basics.md)
- [`../autumn/26-image-baseline-mini-implementation.md`](../autumn/26-image-baseline-mini-implementation.md)
