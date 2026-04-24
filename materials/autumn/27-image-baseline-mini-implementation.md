# 第27回 画像ミニ実装：小規模 classification baseline

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
1. `session27_image_baseline.py` を作成し、class 0 は中央縦線、class 1 は中央横線を持つ 32x32 RGB 画像を返す `StripeDataset` を実装する。各 class 20 枚、合計 40 枚にする。
2. `DataLoader(batch_size=8, shuffle=True)`、小さな CNN、`CrossEntropyLoss`、`Adam(lr=1e-2)` を使って 5 epoch 学習する。
3. epoch ごとの平均 loss を `outputs/figures/session27_loss_curve.png` に保存し、先頭サンプルの推論例を `outputs/images/session27_prediction_example.png` に保存する。
4. `session27_image_baseline_report.md` に `## データセット仕様`, `## 学習設定と loss`, `## 推論例` を書き、入力 shape、logits shape、予測 class を説明する。

### 発展レベル
1. 学習後に全 40 サンプルで推論し、全体 accuracy と class 別の誤り傾向を確認する。
2. report に `## 全体評価` を追加し、loss 曲線と 1 枚の推論例だけでは十分でない理由を説明する。

## 確認ポイント
- dataset 総数が `40` である。
- batch 入力 shape が `(8, 3, 32, 32)` である。
- `session27_loss_curve.png` と `session27_prediction_example.png` が保存されている。
- report に、単一の推論例だけでなく全体評価がある。

## 詰まったときに見る資料
- [`23-image-data-basics.md`](23-image-data-basics.md)
- [`25-training-loop-basics.md`](25-training-loop-basics.md)
