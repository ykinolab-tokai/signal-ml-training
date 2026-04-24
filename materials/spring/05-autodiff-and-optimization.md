# 第05回 autodiff と最適化：backprop・optimizer・scheduler・勾配確認

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- autograd で得た勾配と数値微分を比較できる。
- optimizer と scheduler を入れた最小の学習ループを書ける。
- loss と learning rate の変化を見て、更新がどう進んだかを説明できる。

## ミニ講義
- autograd は計算グラフに沿って勾配を求める。一方、数値微分は値を少し動かして差分を見る。両者が近ければ、実装した loss と backward が大きく外れていないと確認しやすい。
- optimizer はパラメータ更新の規則、scheduler はその規則の中で学習率をどう変えるかを決める。どちらも学習ループに入るが、役割は異なる。
- loss 曲線を見るときは、単に下がったかではなく、「どこで下がり方が変わったか」「learning rate の変更と対応しているか」を見ると解釈しやすい。

## 演習
### 基礎レベル
1. `session05_autodiff_demo.py` を作成し、`y = 2x + 1` の toy データに対して `nn.Linear(1, 1)` の MSE loss を計算する。
2. 初期重みについて、autograd の勾配と中心差分による数値微分を比較する。数値微分は `eps = 1e-3` とし、どの parameter を動かしたかが分かるように書く。
3. `SGD(lr=0.1)` と `StepLR(step_size=10, gamma=0.1)` で 20 step 学習し、loss と learning rate を記録する。loss 曲線を `outputs/figures/session05_loss_curve.png` に保存する。
4. `session05_autodiff_report.md` に `## 勾配比較`, `## optimizer と scheduler`, `## loss の変化` を書き、勾配比較の数値、最初と最後の loss、learning rate が変わった step を説明する。

### 発展レベル
1. scheduler なしの条件も同じ初期化で実行し、scheduler あり・なしの loss 曲線を同じ図で比較する。
2. `session05_autodiff_report.md` に `## scheduler あり / なしの比較` を追加し、最終 loss、learning rate の変化、今回の toy 問題での安定性を比較する。
3. 「勾配が正しく計算できていても loss が下がらないことがある理由」を、learning rate または初期値の観点から 3 行以内で説明する。

## 確認ポイント
- autograd 勾配と数値微分が近い値になる。
- `loss_history` と `lr_history` の長さがともに `20` である。
- `session05_loss_curve.png` が保存されている。
- report に、loss の数値だけでなく scheduler の役割に関する説明がある。

## 詰まったときに見る資料
- [`../autumn/24-pytorch-model-dataset-basics.md`](../autumn/24-pytorch-model-dataset-basics.md)
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
