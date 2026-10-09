# 第24回 representation learning：contrastive learning の最小実装

- 対象: B4・M
- 種別: 前期発展枠 / 準固定

## この回の目標
- toy な pair データで contrastive learning の最小学習を実行できる。
- 学習前後の embedding を可視化できる。
- pair を近づける loss が、embedding 空間で何を起こしているかを説明できる。

## 解説
- representation learning では、分類ラベルを直接当てる代わりに、似ているものは近く、異なるものは遠くに配置される表現を学ぶことが多い。
- 今回使う `CosineEmbeddingLoss` は、pair ごとの方向の近さを扱う。正例 pair だけを使うと「どれだけ近づいたか」は見えるが、「他とどう離れるべきか」は別に考える必要がある。
- embedding の散布図は見た目の確認に便利だが、見た目だけでは曖昧なこともある。平均 cosine similarity のような数値を添えると変化を説明しやすい。

### pairデータと学習条件

同じグループの2つの入力をpositive pair（正例の組）とする。
次の `x1` の2行ずつが同じグループで、`x2` はその2行を交換したものである。
各行は4特徴量、行数は8である。準備コードを実行してから学習処理を実装する。

```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
torch.manual_seed(10)
x1 = torch.tensor([
    [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0.9],
], dtype=torch.float32)
x2 = x1[[1, 0, 3, 2, 5, 4, 7, 6]].clone()
target = torch.ones(8, dtype=torch.float32)
encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
```

両入力を同じencoderへ通し、`loss_fn(encoder(x1), encoder(x2), target)` を最小化する。
targetの `+1` は近づける組、`-1` は離す組を意味するが、基礎では `+1` だけを使う。
第19回の `zero_grad → forward → loss → backward → step` の順序で、8組すべてを使う更新を50回行う。
学習前後の推論は `encoder.eval()` と `torch.no_grad()`、学習時は `encoder.train()` を使う。
平均類似度は `torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean()` とする。
散布図は2次元出力の第0・第1成分を座標とし、同じグループを同色、`x1` と `x2` を異なるmarkerで描く。

発展のnegative pair（負例の組）は `x2_neg = torch.roll(x2, shifts=2, dims=0)` とする。
2行ずらすと別グループ同士になる。1行だけずらすと同じグループの組も含むため、すべてを負例とは呼べない。
正例だけの学習では全入力が同じ方向を向く解も許されるので、正例の類似度が高いだけで良い表現と判断しない。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

### 基礎レベル
1. `session24_representation_learning.py` を作成し、指定された positive pair データ、2 次元 encoder、`CosineEmbeddingLoss` を使って 50 step 学習する。
2. 学習前後の embedding をそれぞれ `outputs/figures/session24_embeddings_before.png`, `outputs/figures/session24_embeddings_after.png` に保存する。同じ pair が対応して見えるように色や marker を工夫する。
3. positive pair の平均 cosine similarity を学習前後で計算する。
4. 学習設定，学習前後の embedding の図のファイル名，cosine similarity を記録し，散布図と数値の対応をコードコメントまたは既存の結果ファイルに説明する。

### 発展レベル
1. `x2` の並びを2行ずらした `x2_neg` で negative pair を作り、学習後 embedding で平均 cosine similarity を計算する。
2. positive pair と negative pair の similarity を比較し、どちらが高くあるべきか、今回の loss だけで十分かを説明する。
3. M は negative pair を本格的に loss に入れる場合に追加したい設計上の注意を 2 つ書く。

## 確認ポイント
- encoder の出力次元が `2` である。
- `session24_embeddings_before.png` と `session24_embeddings_after.png` が保存されている。
- コードコメントまたは既存の結果ファイルに、散布図の見た目と cosine similarity の両方の説明がある。
- 発展課題で、positive と negative の違いを loss の目的と結びつけて説明している。

## 詰まったときに見る資料
- [`19-autodiff-and-optimization.md`](19-autodiff-and-optimization.md): 学習ループ
- PyTorch docs: [`torch.nn.CosineEmbeddingLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)
