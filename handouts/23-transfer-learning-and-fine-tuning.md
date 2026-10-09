# 第23回 transfer learning と fine-tuning：freeze・linear probe・partial fine-tuning

- 対象: B4・M
- 種別: 前期発展枠 / 準固定

## この回の目標
- `linear_probe`, `partial_ft`, `full_ft` の違いをコードで比較できる。
- trainable parameter 数を数え、どこを学習対象にしたかを対応づけて説明できる。
- データ量や初期モデルへの信頼度に応じて、どの条件から試すかを説明できる。

## 解説
- transfer learning では、すでに学習済みの表現をどこまで残すかが重要になる。`linear probe` は head だけ、`partial fine-tuning` は一部の層と head、`full fine-tuning` は全層を更新する。
- `requires_grad` を切り替えると、forward は同じでも backward で更新されるパラメータが変わる。だから output shape が同じでも、学習の自由度と過学習リスクは同じではない。
- trainable parameter 数は「どれだけ更新を許しているか」の粗い指標になる。条件比較では、更新対象、parameter 数、想定するデータ量の 3 つをセットで見ると判断しやすい。

### 固定する入力とモデル

以下を実行して、比較に使うモデルと入力を用意する。

```python
import copy
import torch
from torch import nn

torch.manual_seed(9)
backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
head = nn.Linear(8, 2)
base_model = nn.Sequential(backbone, head)
x = torch.randn(4, 8)
```

`backbone` は特徴抽出部、`head` は特徴からクラスごとのスコア（logits）を求める部分である。
今回は学習済み重みをダウンロードせず、乱数初期化したモデルで「どの層を更新対象にするか」だけを確認する。
実際の転移学習で得られる精度や学習済み表現の有効性は、この演習からは判断できない。

各条件のモデルを `copy.deepcopy(base_model)` で作る。最初に全パラメータを `requires_grad_(False)` で凍結してから、次の層だけを `True` に戻す。

| 条件 | 学習対象の層 | 学習対象パラメータ数 |
| --- | --- | --- |
| `linear_probe` | `model[1]`（head） | 18 |
| `partial_ft` | `model[0][2]`（backbone最後のLinear）と `model[1]` | 154 |
| `full_ft` | 全層 | 298 |

学習対象数は `sum(p.numel() for p in model.parameters() if p.requires_grad)` で数える。
3条件に同じ `x` を渡すと出力はすべて `(4, 2)` であり、更新前の値も一致する。
`requires_grad=False` は値の計算を省く指定ではなく、そのパラメータの勾配を求めない指定である。
学習まで行う場合のoptimizerには `[p for p in model.parameters() if p.requires_grad]` を渡す。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

### 基礎レベル
1. `session23_transfer_learning_demo.py` を作成し、backbone と head からなる小さな `nn.Sequential` model を実装する。
2. `linear_probe`, `partial_ft`, `full_ft` の 3 条件を作り、各条件でどの layer の `requires_grad` を `True` にするかを明示する。
3. 各条件で trainable parameter 数と logits shape を確認する。logits shape が同じでも、更新される parameter が異なることを確認する。
4. 3条件の定義と trainable parameter 数を記録し，データ量と backbone への信頼度に応じた選び方をコードコメントまたは既存の結果ファイルに説明する。

### 発展レベル
1. データが少ない場合と十分ある場合で，どの条件から試すかを理由付きで基礎課題の記録に追記する。
2. B4 は自分の研究テーマに近い task を仮定し、追加で比較したい観点を 1 つ書く。M は review で確認したい観点を、学習安定性・過学習・再現性のうち少なくとも 1 つに触れて書く。

## 確認ポイント
- 3 条件が `linear_probe`, `partial_ft`, `full_ft` の名前で実装されている。
- 3 条件とも logits shape が `(4, 2)` である。
- コードコメントまたは既存の結果ファイルに、trainable parameter 数だけでなく「どういう状況で使い分けるか」が書かれている。
- 発展課題では、条件選択の理由がデータ量や更新自由度に結びついている。

## 詰まったときに見る資料
- [`19-autodiff-and-optimization.md`](19-autodiff-and-optimization.md): Module・勾配・更新
- PyTorch docs: [requires_gradとパラメータの凍結](https://docs.pytorch.org/docs/stable/notes/autograd.html#setting-requires-grad)
