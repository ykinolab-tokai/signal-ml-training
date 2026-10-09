# 第23回 transfer learning と fine-tuning：freeze・linear probe・partial fine-tuning

- 対象: B4・M
- 種別: 前期発展枠 / 準固定

## この回の目標

転移学習で更新するパラメータの範囲を比較し，データと初期モデルに応じた選び方を説明できる。

## 解説
- transfer learning では、すでに学習済みの表現をどこまで残すかが重要になる。`linear probe` は head だけ、`partial fine-tuning` は一部の層と head、`full fine-tuning` は全層を更新する。
- `requires_grad` を切り替えると、forward は同じでも backward で更新されるパラメータが変わる。だから output shape が同じでも、学習の自由度と過学習リスクは同じではない。
- trainable parameter 数は「どれだけ更新を許しているか」の粗い指標になる。条件比較では、更新対象、parameter 数、想定するデータ量の 3 つをセットで見ると判断しやすい。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session23_transfer_learning_demo.py` を作成し、backbone と head からなる小さな `nn.Sequential` model を実装してください。
2. `linear_probe`, `partial_ft`, `full_ft` の 3 条件を作り、各条件でどの layer の `requires_grad` を `True` にするかを明示してください。
3. 各条件で trainable parameter 数と logits shape を確認してください。logits shape が同じでも、更新される parameter が異なることを確認してください。
4. 3条件の定義と trainable parameter 数を記録し，データ量と backbone への信頼度に応じた選び方をコードコメントまたは既存の結果ファイルに説明してください。

### 発展レベル
1. データが少ない場合と十分ある場合で，どの条件から試すかを理由付きで基礎課題の記録に追記してください。
2. B4 は自分の研究テーマに近い task を仮定し、追加で比較したい観点を 1 つ書いてください。M は review で確認したい観点を、学習安定性・過学習・再現性のうち少なくとも 1 つに触れて書いてください。

## 確認ポイント
- 3 条件が `linear_probe`, `partial_ft`, `full_ft` の名前で実装されている。
- 3 条件とも logits shape が `(4, 2)` である。
- コードコメントまたは既存の結果ファイルに、trainable parameter 数だけでなく「どういう状況で使い分けるか」が書かれている。
- 発展課題では、条件選択の理由がデータ量や更新自由度に結びついている。

## 詰まったときに見る資料
- [`09-audio-signals.md`](09-audio-signals.md)
- [`11-noise-and-signal-restoration.md`](11-noise-and-signal-restoration.md)
