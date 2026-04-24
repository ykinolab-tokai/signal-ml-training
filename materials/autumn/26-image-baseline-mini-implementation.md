# 第26回 画像ミニ実装：小規模 classification baseline

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- synthetic 画像データセット、Dataset、Model、training loop を 1 本につなげて実行できる。
- 入力 tensor と出力 logits の shape を説明できる。
- 学習後の予測結果を、loss と精度の両面から説明できる。

## ミニ講義
- baseline 実装では、データ生成、前処理、model、学習ループをまず 1 本で繋げることが重要である。小さくても end-to-end で動くことが後の改良の土台になる。
- classification では、各サンプルに 1 つの label を持たせ、model は class 数ぶんの logits を返す。shape を追うと、どこで画像が特徴ベクトルへ潰れているか見やすい。
- loss だけを見ると「下がった」で終わりやすい。実際に全サンプルでどれだけ当たるかを見ると、model が task を解けているか判断しやすい。

## 演習
### 基礎レベル
1. `session26_image_baseline.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

fig_dir = Path("outputs/figures")
img_dir = Path("outputs/images")
fig_dir.mkdir(parents=True, exist_ok=True)
img_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(26)
```
2. `StripeDataset` を作り、class 0 は中央縦線、class 1 は中央横線を持つ 32x32 RGB 画像を返すようにする。
3. 各 class を 20 枚、合計 40 枚作る。
4. 画像 tensor の shape を `(3, 32, 32)`、label を `0` または `1` にする。
5. `DataLoader(dataset, batch_size=8, shuffle=True)` を作る。
6. model を次の通り定義する。
```python
model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(8, 2),
)
```
7. `criterion = nn.CrossEntropyLoss()`、`optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)` を使い、`for epoch in range(5):` で 5 epoch 学習する。
8. 各 epoch の平均 loss を `loss_history` に記録し、`outputs/figures/session26_loss_curve.png` に保存する。
9. dataset の先頭サンプル 1 件を model に通し、画像を `outputs/images/session26_prediction_example.png` に保存する。
10. `session26_image_baseline_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## データセット仕様`
   - `## 学習設定と loss`
   - `## 推論例`

### 発展レベル
1. 学習後、全 40 サンプルに対して推論を行い、正解数を数える。
2. `session26_image_baseline_report.md` に `## 全体評価` を追加し、全体 accuracy と、class 0 / class 1 のどちらで間違えやすいかを書く。
3. そのうえで、「loss が下がっていても予測例 1 枚だけでは十分でない理由」を 2 行以内で説明する。

## 確認ポイント
- dataset 総数が `40` である。
- batch 入力 shape が `(8, 3, 32, 32)` である。
- `session26_loss_curve.png` と `session26_prediction_example.png` が保存されている。
- report に、単一の推論例だけでなく全体評価がある。

## 詰まったときに見る資料
- [`22-image-data-basics.md`](22-image-data-basics.md)
- [`25-training-loop-basics.md`](25-training-loop-basics.md)
