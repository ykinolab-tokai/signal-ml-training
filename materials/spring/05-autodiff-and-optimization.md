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
1. `session05_autodiff_demo.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(5)
```
2. データを次の通り定義する。
```python
x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
y = 2.0 * x + 1.0
```
3. `model = nn.Linear(1, 1)` を作成し、初期重みでの `weight.grad` を `loss.backward()` 後に記録する。
4. 数値微分は `eps = 1e-3` を使い、重みだけについて次の式で計算する。
```python
(loss(w + eps) - loss(w - eps)) / (2 * eps)
```
5. optimizer と scheduler を次の通り定義する。
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.MSELoss()
```
6. `for step in range(20):` で 20 step 学習し、各 step の loss を `loss_history`、learning rate を `lr_history` に記録する。
7. loss 曲線を `outputs/figures/session05_loss_curve.png` に保存する。
8. `session05_autodiff_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 勾配比較`
   - `## optimizer と scheduler`
   - `## loss の変化`
9. `## 勾配比較` には autograd 勾配と数値微分を並べて書く。
10. `## optimizer と scheduler` には `SGD`, `lr=0.1`, `step_size=10`, `gamma=0.1` を書く。
11. `## loss の変化` には最初と最後の loss を書き、どの step 付近で変化が緩やかになったかを 1 行添える。

### 発展レベル
1. scheduler を使わない条件も追加し、同じ初期化で 20 step 学習する。
2. `session05_autodiff_report.md` に `## scheduler あり / なしの比較` を追加し、次の 3 点を書く。
   - 最終 loss の比較
   - learning rate の変化
   - どちらが今回の toy 問題で安定して見えるか
3. 必要なら loss 曲線に 2 本目を追加し、`session05_loss_curve.png` で見分けられるよう凡例を付ける。
4. 最後に、「勾配が正しく計算できていても loss が下がらないことがあるのはなぜか」を 3 行以内で説明する。

## 確認ポイント
- autograd 勾配と数値微分が近い値になる。
- `loss_history` と `lr_history` の長さがともに `20` である。
- `session05_loss_curve.png` が保存されている。
- report に、loss の数値だけでなく scheduler の役割に関する説明がある。

## 詰まったときに見る資料
- [`../autumn/24-pytorch-basics.md`](../autumn/24-pytorch-basics.md)
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
