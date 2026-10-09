# 第19回 autodiff と最適化：backprop・optimizer・scheduler・勾配確認

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標

自動微分の勾配を検証し，optimizer と scheduler による学習の進み方を説明できる。

## 解説
- autograd は計算グラフに沿って勾配を求める。一方、数値微分は値を少し動かして差分を見る。両者が近ければ、実装した loss と backward が大きく外れていないと確認しやすい。
- optimizer はパラメータ更新の規則、scheduler はその規則の中で学習率をどう変えるかを決める。どちらも学習ループに入るが、役割は異なる。
- loss 曲線を見るときは、単に下がったかではなく、「どこで下がり方が変わったか」「learning rate の変更と対応しているか」を見ると解釈しやすい。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session19_autodiff_demo.py` を作成し、`y = 2x + 1` の toy データに対して `nn.Linear(1, 1)` の MSE loss を計算してください。
2. 初期重みについて、autograd の勾配と中心差分による数値微分を比較してください。数値微分は `eps = 1e-3` とし、どの parameter を動かしたかが分かるように書いてください。
3. `SGD(lr=0.1)` と `StepLR(step_size=10, gamma=0.1)` で 20 step 学習し、loss と learning rate を記録してください。loss 曲線を `outputs/figures/session19_loss_curve.png` に保存してください。
4. 勾配比較の数値，最初と最後の loss，learning rate が変わった step を記録し，optimizer と scheduler の役割をコードコメントまたは既存の結果ファイルに説明してください。

### 発展レベル
1. scheduler なしの条件も同じ初期化で実行し、scheduler あり・なしの loss 曲線を同じ図で比較してください。
2. scheduler あり・なしの最終 loss，learning rate の変化，今回の toy 問題での安定性を比較し，基礎課題の記録に追記してください。
3. 「勾配が正しく計算できていても loss が下がらないことがある理由」を、learning rate または初期値の観点から 3 行以内で説明してください。

## 確認ポイント
- autograd 勾配と数値微分が近い値になる。
- `loss_history` と `lr_history` の長さがともに `20` である。
- `session19_loss_curve.png` が保存されている。
- コードコメントまたは既存の結果ファイルに、loss の数値だけでなく scheduler の役割に関する説明がある。

## 詰まったときに見る資料
- [`11-noise-and-signal-restoration.md`](11-noise-and-signal-restoration.md)
- [`../textbook/markdown/ch23-basics-of-neural-networks.md`](../textbook/markdown/ch23-basics-of-neural-networks.md)
