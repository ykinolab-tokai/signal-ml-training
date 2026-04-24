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
1. `session10_representation_learning.py` を作成し、指定された positive pair データ、2 次元 encoder、`CosineEmbeddingLoss` を使って 50 step 学習する。
2. 学習前後の embedding をそれぞれ `outputs/figures/session10_embeddings_before.png`, `outputs/figures/session10_embeddings_after.png` に保存する。同じ pair が対応して見えるように色や marker を工夫する。
3. positive pair の平均 cosine similarity を学習前後で計算する。
4. `session10_representation_learning_report.md` に `## 学習設定`, `## 学習前の embedding`, `## 学習後の embedding`, `## cosine similarity の比較` を書き、散布図と数値が対応しているかを説明する。

### 発展レベル
1. `x2` の並びを 1 つずらした negative pair を作り、学習後 embedding で平均 cosine similarity を計算する。
2. positive pair と negative pair の similarity を比較し、どちらが高くあるべきか、今回の loss だけで十分かを説明する。
3. M は negative pair を本格的に loss に入れる場合に追加したい設計上の注意を 2 つ書く。

## 確認ポイント
- encoder の出力次元が `2` である。
- `session10_embeddings_before.png` と `session10_embeddings_after.png` が保存されている。
- report に、散布図の見た目と cosine similarity の両方の説明がある。
- 発展課題で、positive と negative の違いを loss の目的と結びつけて説明している。

## 詰まったときに見る資料
- [`../autumn/24-pytorch-model-dataset-basics.md`](../autumn/24-pytorch-model-dataset-basics.md)
- PyTorch docs: [`torch.nn.CosineEmbeddingLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)
