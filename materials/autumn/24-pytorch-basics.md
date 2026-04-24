# 第24回 PyTorch の基礎：Tensor・Module・Dataset・DataLoader

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- `Dataset` が 1 サンプルを返し、`DataLoader` がバッチを作ることを説明できる。
- 最小の `nn.Module` を定義し、forward を通せる。
- batch size を変えたとき、どの次元が変わるかを説明できる。

## ミニ講義
- `Dataset` はサンプル単位、`DataLoader` はバッチ単位で考える。まず 1 サンプルの shape を確認してからバッチ shape を見ると、先頭次元の意味が分かりやすい。
- `nn.Module` は入力 tensor を受けて出力 tensor を返す部品である。forward の結果 shape は「入力のどの次元が保たれ、どの次元が変換されたか」で読む。
- batch size を変えても、通常は特徴量次元や class 数は変わらない。変わるのは先頭の batch 次元である。

## 演習
### 基礎レベル
1. `session24_pytorch_basics.py` を作成し、次の import を書く。
```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
```
2. 次の `ToyDataset` をそのまま実装する。
```python
class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
        ], dtype=torch.float32)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```
3. `loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)` を作り、`batch_x, batch_y = next(iter(loader))` を取り出す。
4. `batch_x.shape` と `batch_y.shape` を表示する。
5. 次の `ToyNet` をそのまま実装する。
```python
class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)
```
6. `model = ToyNet()` を作成し、`logits = model(batch_x)` を実行して `logits.shape` を表示する。
7. `session24_pytorch_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## Dataset が返すもの`
   - `## DataLoader が返すもの`
   - `## Model の出力`

### 発展レベル
1. `batch_size=1` の `DataLoader` でも最初のバッチを取り出し、`batch_x_1.shape`, `batch_y_1.shape`, `logits_1.shape` を確認する。
2. `session24_pytorch_report.md` に `## batch size の比較` を追加し、`batch_size=1` と `batch_size=2` で変わる次元と変わらない次元を書く。
3. そのうえで、「class 数に対応する次元が batch size を変えても変わらない理由」を 2 行以内で説明する。

## 確認ポイント
- `len(ToyDataset())` が `6` である。
- `batch_x.shape` が `(2, 4)`、`batch_y.shape` が `(2,)`、`logits.shape` が `(2, 2)` である。
- report に、1 サンプル shape とバッチ shape の違いが書かれている。
- 発展課題で、先頭次元が batch 次元であることを説明している。

## 詰まったときに見る資料
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
- PyTorch docs: [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
