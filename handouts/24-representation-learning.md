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

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session24_representation_learning.py` を作成し、指定された positive pair データ、2 次元 encoder、`CosineEmbeddingLoss` を使って 50 step 学習してください。
2. 学習前後の embedding をそれぞれ `outputs/figures/session24_embeddings_before.png`, `outputs/figures/session24_embeddings_after.png` に保存してください。同じ pair が対応して見えるように色や marker を工夫してください。
3. positive pair の平均 cosine similarity を学習前後で計算してください。
4. 学習設定，学習前後の embedding の図のファイル名，cosine similarity を記録し，散布図と数値の対応をコードコメントまたは既存の結果ファイルに説明してください。

### 発展レベル
1. `x2` の並びを 1 つずらした negative pair を作り、学習後 embedding で平均 cosine similarity を計算してください。
2. positive pair と negative pair の similarity を比較し、どちらが高くあるべきか、今回の loss だけで十分かを説明してください。
3. M は negative pair を本格的に loss に入れる場合に追加したい設計上の注意を 2 つ書いてください。

## 確認ポイント
- encoder の出力次元が `2` である。
- `session24_embeddings_before.png` と `session24_embeddings_after.png` が保存されている。
- コードコメントまたは既存の結果ファイルに、散布図の見た目と cosine similarity の両方の説明がある。
- 発展課題で、positive と negative の違いを loss の目的と結びつけて説明している。

## 詰まったときに見る資料
- [`11-noise-and-signal-restoration.md`](11-noise-and-signal-restoration.md)
- PyTorch docs: [`torch.nn.CosineEmbeddingLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)
