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
1. `session09_transfer_learning_demo.py` を作成し、次の import を書く。
```python
import torch
from torch import nn
```
2. backbone と head を次の通り定義する。
```python
backbone = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
)
head = nn.Linear(8, 2)
model = nn.Sequential(backbone, head)
```
3. 比較用入力を `x = torch.randn(4, 8)` として作る。
4. `linear_probe` 条件では backbone をすべて `requires_grad=False`、head を `True` にする。
5. `partial_ft` 条件では backbone の最後の `nn.Linear(16, 8)` と head を `True`、それ以前を `False` にする。
6. `full_ft` 条件ではすべて `requires_grad=True` にする。
7. 各条件で trainable parameter 数を数える関数を作る。
8. 各条件で `model(x)` を実行し、`logits.shape` を表示する。
9. `session09_transfer_learning_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 3 条件の定義`
   - `## trainable parameter 数`
   - `## 使い分けの考え方`
10. `## 使い分けの考え方` には、3 条件の違いを 4 行以内で書く。

### 発展レベル
1. `session09_transfer_learning_report.md` に `## 最初に試す条件` を追加し、次の 2 ケースでどの条件から始めるかを書く。
   - データが少なく、backbone をかなり信頼している場合
   - データが十分あり、backbone の表現が task に合わないかもしれない場合
2. 各ケースで、選んだ条件の理由を 2 行以内で書く。`parameter 数` か `更新対象` のどちらかに触れること。
3. B4 は、自分の研究テーマに近いと仮定したときに追加で見たい比較観点を 1 つ書く。
4. M は、条件比較に review で追加したい観点を 2 行書く。例として、学習安定性、過学習、再現性のどれを見るかを明記する。

## 確認ポイント
- 3 条件が `linear_probe`, `partial_ft`, `full_ft` の名前で実装されている。
- 3 条件とも logits shape が `(4, 2)` である。
- report に、trainable parameter 数だけでなく「どういう状況で使い分けるか」が書かれている。
- 発展課題では、条件選択の理由がデータ量や更新自由度に結びついている。

## 詰まったときに見る資料
- [`../autumn/24-pytorch-basics.md`](../autumn/24-pytorch-basics.md)
- [`../autumn/25-training-loop-basics.md`](../autumn/25-training-loop-basics.md)
