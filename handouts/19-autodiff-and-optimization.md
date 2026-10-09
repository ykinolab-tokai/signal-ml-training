# 第19回 autodiff と最適化：backprop・optimizer・scheduler・勾配確認

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- autograd で得た勾配と数値微分を比較できる。
- optimizer と scheduler を入れた最小の学習ループを書ける。
- loss と learning rate の変化を見て、更新がどう進んだかを説明できる。

## 解説
- autograd は計算グラフに沿って勾配を求める。一方、数値微分は値を少し動かして差分を見る。両者が近ければ、実装した loss と backward が大きく外れていないと確認しやすい。
- optimizer はパラメータ更新の規則、scheduler はその規則の中で学習率をどう変えるかを決める。どちらも学習ループに入るが、役割は異なる。
- loss 曲線を見るときは、単に下がったかではなく、「どこで下がり方が変わったか」「learning rate の変更と対応しているか」を見ると解釈しやすい。

### PyTorchの最小構成と固定データ

第1回で用意したPython環境を使い、今回はCPU上で計算する。
TensorはNumPy配列と同様に `shape` と `dtype` を持つ多次元配列である。
`nn.Module` はパラメータと計算処理をまとめる基底クラスで、`nn.Linear` もその一種である。
`model(x)` は順伝播を行い、`model.parameters()` はモデルに登録されたパラメータを返す。
`nn.Linear(1, 1)` は各行に対して $wx+b$ を計算する。次は入力・モデルを用意する実行可能な例である。

```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
torch.manual_seed(5)
x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float32)
y = 2 * x + 1
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
print(x.shape, model(x).shape)  # どちらも torch.Size([5, 1])
```

`x` と `y` の第0軸は5サンプルのバッチ、第1軸は1特徴量を表す。
重みとバイアスは既定で `requires_grad=True` となり、`loss.backward()` によって各 `.grad` に勾配が蓄積される。
MSEは今回の5点の二乗誤差の平均である。

### 勾配確認と学習の手順

学習前に `model.zero_grad()`、lossの計算、`loss.backward()` を行い、`model.weight.grad[0, 0]` を記録する。
数値微分ではバイアスを固定し、重み `model.weight[0, 0]` だけを $w+\varepsilon$, $w-\varepsilon$ に変えて
$(L(w+\varepsilon)-L(w-\varepsilon))/(2\varepsilon)$ を求める。
パラメータの一時変更は `with torch.no_grad():` 内で行い、最後に元の値へ戻す。
`eps=1e-3` とし、勾配の絶対差が `1e-2` 未満になることを目安にする。

勾配確認後、次の学習ループを続けて実行できる。
1 stepはパラメータを1回更新することで、今回は毎回5点すべてを使う。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_history, lr_history = [], []
model.train()
for step in range(20):
    optimizer.zero_grad()  # 前のstepの勾配を消す
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss_history.append(loss.item())  # 更新前のloss
    lr_history.append(optimizer.param_groups[0]["lr"])  # この更新に使う学習率
    loss.backward()
    optimizer.step()
    scheduler.step()  # パラメータ更新後に、次のstepの学習率を設定
```

記録した学習率は1〜10回目が `0.1`、11〜20回目が `0.01` となる。
20回目の更新後のlossも `with torch.no_grad():` 内で計算し、最後の更新前の値とは区別する。
schedulerなしとの比較では、モデルを作る直前に同じ乱数seedを設定し、optimizerも新しく作り直す。

推論・検証では `model.eval()` と `with torch.no_grad():` を使い、`backward()` と `optimizer.step()` は行わない。
`eval()` はBatchNormなどの動作を切り替え、`no_grad()` は勾配の記録を止めるため、役割が異なる。
学習用データとは別の検証用データを使う場合は、検証用データでパラメータを更新しない。
今回は学習の仕組みを見る固定5点の例であり、未知データへの性能評価は行わない。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

### 基礎レベル
1. `session19_autodiff_demo.py` を作成し、`y = 2x + 1` の toy データに対して `nn.Linear(1, 1)` の MSE loss を計算する。
2. 初期重みについて、autograd の勾配と中心差分による数値微分を比較する。数値微分は `eps = 1e-3` とし、どの parameter を動かしたかが分かるように書く。
3. `SGD(lr=0.1)` と `StepLR(step_size=10, gamma=0.1)` で 20 step 学習し、loss と learning rate を記録する。loss 曲線を `outputs/figures/session19_loss_curve.png` に保存する。
4. 勾配比較の数値，最初と最後の loss，learning rate が変わった step を記録し，optimizer と scheduler の役割をコードコメントまたは既存の結果ファイルに説明する。

### 発展レベル
1. scheduler なしの条件も同じ初期化で実行し、scheduler あり・なしの loss 曲線を同じ図で比較する。
2. scheduler あり・なしの最終 loss，learning rate の変化，今回の toy 問題での安定性を比較し，基礎課題の記録に追記する。
3. 「勾配が正しく計算できていても loss が下がらないことがある理由」を、learning rate または初期値の観点から 3 行以内で説明する。

## 確認ポイント
- autograd 勾配と数値微分が近い値になる。
- `loss_history` と `lr_history` の長さがともに `20` である。
- `session19_loss_curve.png` が保存されている。
- コードコメントまたは既存の結果ファイルに、loss の数値だけでなく scheduler の役割に関する説明がある。

## 詰まったときに見る資料
- PyTorch公式チュートリアル: [学習ループとパラメータ更新](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [`../textbook/markdown/ch23-basics-of-neural-networks.md`](../textbook/markdown/ch23-basics-of-neural-networks.md)
