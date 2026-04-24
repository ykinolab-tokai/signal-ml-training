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
1. `session25_training_loop_basics.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(25)
```
2. train データを次の通り定義する。
```python
train_x = torch.tensor([
    [0.0, 1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0],
], dtype=torch.float32)
train_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
```
3. validation データを次の通り定義する。
```python
val_x = torch.tensor([
    [0.5, 1.5, 2.5, 3.5],
    [2.5, 3.5, 4.5, 5.5],
], dtype=torch.float32)
val_y = torch.tensor([0, 1], dtype=torch.long)
```
4. model, criterion, optimizer を次の通り定義する。
```python
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
```
5. `for epoch in range(20):` で学習し、各 epoch の train loss と validation loss を `train_losses`, `val_losses` に記録する。
6. `outputs/figures/session25_loss_curve.png` に 2 本の曲線を描いて保存する。
7. `session25_training_loop_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 学習設定`
   - `## train loss と validation loss`
   - `## loss 曲線の解釈`

### 発展レベル
1. epoch 数を `100` に増やした条件も実行し、loss 曲線を見比べる。
2. `session25_training_loop_report.md` に `## overfitting の兆候` を追加し、次の 2 点を書く。
   - train と validation の差が広がり始めたと感じる epoch
   - その判断理由
3. 「最終 epoch が常に最良とは限らない理由」を 2 行以内で説明する。

## 確認ポイント
- `train_losses` と `val_losses` が別々に記録されている。
- `session25_loss_curve.png` に train と validation の 2 本があり、凡例で区別できる。
- report に、loss の値だけでなく過学習の兆候に関する説明がある。

## 詰まったときに見る資料
- [`24-pytorch-basics.md`](24-pytorch-basics.md)
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
