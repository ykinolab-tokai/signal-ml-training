# 第09回 transfer learning と fine-tuning：freeze・linear probe・partial fine-tuning

- 対象: B4・M
- 種別: 前期発展枠 / 準固定

## この回の目標
- `linear_probe`, `partial_ft`, `full_ft` の違いをコードで比較できる。
- trainable parameter 数を数え、どこを学習対象にしたかを対応づけて説明できる。
- データ量や初期モデルへの信頼度に応じて、どの条件から試すかを説明できる。

## ミニ講義
- transfer learning では、すでに学習済みの表現をどこまで残すかが重要になる。`linear probe` は head だけ、`partial fine-tuning` は一部の層と head、`full fine-tuning` は全層を更新する。
- `requires_grad` を切り替えると、forward は同じでも backward で更新されるパラメータが変わる。だから output shape が同じでも、学習の自由度と過学習リスクは同じではない。
- trainable parameter 数は「どれだけ更新を許しているか」の粗い指標になる。条件比較では、更新対象、parameter 数、想定するデータ量の 3 つをセットで見ると判断しやすい。

## 演習
### 基礎レベル
1. `session09_transfer_learning_demo.py` を作成し、backbone と head からなる小さな `nn.Sequential` model を実装する。
2. `linear_probe`, `partial_ft`, `full_ft` の 3 条件を作り、各条件でどの layer の `requires_grad` を `True` にするかを明示する。
3. 各条件で trainable parameter 数と logits shape を確認する。logits shape が同じでも、更新される parameter が異なることを確認する。
4. `session09_transfer_learning_report.md` に `## 3 条件の定義`, `## trainable parameter 数`, `## 使い分けの考え方` を書き、データ量と backbone への信頼度に応じた選び方を説明する。

### 発展レベル
1. `## 最初に試す条件` を追加し、データが少ない場合と十分ある場合で、どの条件から試すかを理由付きで書く。
2. B4 は自分の研究テーマに近い task を仮定し、追加で比較したい観点を 1 つ書く。M は review で確認したい観点を、学習安定性・過学習・再現性のうち少なくとも 1 つに触れて書く。

## 確認ポイント
- 3 条件が `linear_probe`, `partial_ft`, `full_ft` の名前で実装されている。
- 3 条件とも logits shape が `(4, 2)` である。
- report に、trainable parameter 数だけでなく「どういう状況で使い分けるか」が書かれている。
- 発展課題では、条件選択の理由がデータ量や更新自由度に結びついている。

## 詰まったときに見る資料
- [`../autumn/24-pytorch-model-dataset-basics.md`](../autumn/24-pytorch-model-dataset-basics.md)
- [`../autumn/25-training-loop-basics.md`](../autumn/25-training-loop-basics.md)
