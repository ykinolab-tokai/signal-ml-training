# 第17回 PyTorch の基礎：tensor・自動微分・学習ループ

## この回の目標

PyTorch の tensor と自動微分を使い，勾配降下法による学習ループを実装できる。

## 解説
- PyTorch の **tensor** は，NumPy の配列と同じように shape と dtype を持つ多次元配列である。`torch.from_numpy` と `Tensor.numpy()` で相互に変換できる。PyTorch のモデルは `float32` を使うことが多く，NumPy の既定の `float64` とは異なる。
- `requires_grad=True` を指定した tensor から計算した値に対して `backward()` を呼ぶと，勾配が自動で計算され，各 tensor の `.grad` に保存される。これを **自動微分 (automatic differentiation)** という。自動微分は計算の過程をたどり，第4回の連鎖律を繰り返し使って勾配を求める。
- autograd は計算グラフに沿って勾配を求める。一方、数値微分は値を少し動かして差分を見る。両者が近ければ、実装した loss と backward が大きく外れていないと確認しやすい。
- `torch.optim.SGD` の `step()` は，第16回の勾配降下法の更新 $\theta \leftarrow \theta - \eta\nabla L(\theta)$ を行う。勾配は `backward()` のたびに `.grad` へ足し合わされるため，更新の前に `zero_grad()` で 0 に戻す。
- optimizer はパラメータ更新の規則、scheduler はその規則の中で学習率をどう変えるかを決める。どちらも学習ループに入るが、役割は異なる。
- loss 曲線を見るときは、単に下がったかではなく、「どこで下がり方が変わったか」「learning rate の変更と対応しているか」を見ると解釈しやすい。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session17_tensor_basics.py` を作成し，`np.arange(6).reshape(2, 3)` から `torch.from_numpy` で tensor を作って，shape と dtype を表示してください。`float32` に変換した tensor を `numpy()` で配列に戻し，値と dtype を確認してください。
2. 1 の `float32` の tensor を `x` とし，`x.requires_grad_()` を実行してから $y=\sum_i x_i^2$ を計算して `backward()` を呼んでください。`x.grad` が手計算した勾配 $2x_i$ と一致することを確認してください。
3. `session17_autodiff_demo.py` を作成し、`y = 2x + 1` の toy データに対して `nn.Linear(1, 1)` の MSE loss を計算してください。
4. 初期重みについて、autograd の勾配と中心差分による数値微分を比較してください。数値微分は `eps = 1e-3` とし、どの parameter を動かしたかが分かるように書いてください。`float32` では，第4回で使った `h = 1e-5` より大きい刻み幅が必要になる理由も，第4回の桁落ちの説明を使って書いてください。
5. `SGD(lr=0.1)` と `StepLR(step_size=10, gamma=0.1)` で 20 step 学習し、loss と learning rate を記録してください。loss 曲線を `outputs/figures/session17_loss_curve.png` に保存してください。
6. 勾配比較の数値，最初と最後の loss，learning rate が変わった step を記録し，optimizer と scheduler の役割をコードコメントまたは既存の結果ファイルに説明してください。

### 発展レベル
1. scheduler なしの条件も同じ初期化で実行し、scheduler あり・なしの loss 曲線を同じ図で比較してください。
2. scheduler あり・なしの最終 loss，learning rate の変化，今回の toy 問題での安定性を比較し，基礎課題の記録に追記してください。
3. 「勾配が正しく計算できていても loss が下がらないことがある理由」を、learning rate または初期値の観点から 3 行以内で説明してください。
4. 基礎第5問の学習ループから `zero_grad()` を除いて実行し，loss の変化を比較してください。勾配が足し合わされることと結びつけて，違いを説明してください。

## 確認ポイント
- 基礎第1問で，`torch.from_numpy` で作った tensor と `float32` に変換した tensor の dtype の違いを確認している。
- 基礎第2問で，`x.grad` が $2x_i$ と一致している。
- autograd 勾配と数値微分が近い値になる。
- `loss_history` と `lr_history` の長さがともに `20` である。
- `session17_loss_curve.png` が保存されている。
- コードコメントまたは既存の結果ファイルに、loss の数値だけでなく scheduler の役割に関する説明がある。

## 詰まったときに見る資料
- [`16-continuous-optimization.md`](16-continuous-optimization.md)
- PyTorch docs: [Tensors](https://docs.pytorch.org/docs/stable/tensors.html), [Automatic differentiation package](https://docs.pytorch.org/docs/stable/autograd.html)
