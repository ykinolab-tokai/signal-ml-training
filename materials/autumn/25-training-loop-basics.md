# 第25回 training loop の基礎：loss・optimizer・epoch・validation・overfitting

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- `zero_grad -> forward -> loss -> backward -> step` の流れを自分で書ける。
- train loss と validation loss を別々に記録できる。
- loss 曲線を見て、過学習の兆候を説明できる。

## ミニ講義
- training loop では、train と validation を混ぜないことが重要である。train は更新用、validation は更新せず様子を見るための系列である。
- loss 曲線は下がればよいわけではない。train だけ下がって validation が悪化するなら、モデルが train 側へ寄りすぎている可能性がある。
- overfitting を見るときは、最終 epoch だけでなく、どの epoch 付近から train / validation の差が開くかを見ると判断しやすい。

## 演習
### 基礎レベル
1. `session25_training_loop_basics.py` を作成し、指定された train/validation データ、model、criterion、optimizer を定義する。
2. 20 epoch の training loop を書き、各 epoch の train loss と validation loss を別々に記録する。validation では parameter を更新しないことを確認する。
3. train/validation の loss 曲線を `outputs/figures/session25_loss_curve.png` に保存する。
4. `session25_training_loop_report.md` に `## 学習設定`, `## train loss と validation loss`, `## loss 曲線の解釈` を書き、train と validation を分ける理由を説明する。

### 発展レベル
1. epoch 数を 100 に増やした条件も実行し、overfitting の兆候があるかを確認する。
2. report に `## overfitting の兆候` を追加し、train と validation の差が開き始めたと感じる epoch と、その判断理由を書く。
3. 最終 epoch が常に最良とは限らない理由を 2 行以内で説明する。

## 確認ポイント
- `train_losses` と `val_losses` が別々に記録されている。
- `session25_loss_curve.png` に train と validation の 2 本があり、凡例で区別できる。
- report に、loss の値だけでなく過学習の兆候に関する説明がある。

## 詰まったときに見る資料
- [`24-pytorch-model-dataset-basics.md`](24-pytorch-model-dataset-basics.md)
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
