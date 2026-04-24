# 第10回 representation learning：contrastive learning の最小実装

- 対象: B4・M
- 種別: 前期発展枠 / 準固定

## この回の目標
- toy な pair データで contrastive learning の最小学習を実行できる。
- 学習前後の embedding を可視化できる。
- pair を近づける loss が、embedding 空間で何を起こしているかを説明できる。

## ミニ講義
- representation learning では、分類ラベルを直接当てる代わりに、似ているものは近く、異なるものは遠くに配置される表現を学ぶことが多い。
- 今回使う `CosineEmbeddingLoss` は、pair ごとの方向の近さを扱う。正例 pair だけを使うと「どれだけ近づいたか」は見えるが、「他とどう離れるべきか」は別に考える必要がある。
- embedding の散布図は見た目の確認に便利だが、見た目だけでは曖昧なこともある。平均 cosine similarity のような数値を添えると変化を説明しやすい。

## 演習
### 基礎レベル
1. `session10_representation_learning.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(10)
```
2. pair データを次の通り定義する。
```python
x1 = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 0.0, 1.0],
    [0.1, 0.0, 0.0, 0.9],
], dtype=torch.float32)
x2 = torch.tensor([
    [0.9, 0.1, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 1.0, 0.0],
    [0.1, 0.0, 0.0, 0.9],
    [0.0, 0.0, 0.0, 1.0],
], dtype=torch.float32)
target = torch.ones(8)
```
3. encoder を次の通り定義する。
```python
encoder = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
```
4. `nn.CosineEmbeddingLoss()` と `torch.optim.Adam(encoder.parameters(), lr=1e-2)` を使う。
5. 学習前の `z1_before`, `z2_before` を取得し、2 枚の点群を 1 枚に描いて `outputs/figures/session10_embeddings_before.png` に保存する。
6. `for step in range(50):` で 50 step 学習する。
7. 学習後の `z1_after`, `z2_after` を同様に描いて `outputs/figures/session10_embeddings_after.png` に保存する。
8. `session10_representation_learning_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 学習設定`
   - `## 学習前の embedding`
   - `## 学習後の embedding`
9. `## 学習後の embedding` には、pair がどう変化したかを 3 行以内で書く。

### 発展レベル
1. 学習前後で positive pair の平均 cosine similarity を計算する。
2. `session10_representation_learning_report.md` に `## cosine similarity の比較` を追加し、学習前後の数値を書く。
3. negative pair として、`x1` と並び順を 1 つずらした `x2_neg` を作り、学習後 embedding で平均 cosine similarity を計算する。
4. `## cosine similarity の比較` には、positive pair と negative pair のどちらが高くあるべきかを 2 行以内で説明する。
5. M は、negative pair を loss に本格的に入れるなら何を追加するかを 2 行で書く。

## 確認ポイント
- encoder の出力次元が `2` である。
- `session10_embeddings_before.png` と `session10_embeddings_after.png` が保存されている。
- report に、散布図の見た目と cosine similarity の両方の説明がある。
- 発展課題で、positive と negative の違いを loss の目的と結びつけて説明している。

## 詰まったときに見る資料
- [`../autumn/24-pytorch-basics.md`](../autumn/24-pytorch-basics.md)
- PyTorch docs: [`torch.nn.CosineEmbeddingLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)
