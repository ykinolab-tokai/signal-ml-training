# 第07回 画像モデル基礎：CNN・ResNet・U-Net・receptive field・normalization

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- plain block と residual block の違いをコードで比較できる。
- 出力 shape と parameter 数を見て、見た目の違いと内部の違いを分けて説明できる。
- skip connection が表現と勾配の流れにどう関わるかを言葉で説明できる。

## ミニ講義
- CNN block は「畳み込みの並び」で考えると読みやすいが、ResNet ではそこに入力を足し戻す skip connection が入る。これにより、同じ shape を保ちながら振る舞いが変わる。
- output shape と parameter 数が同じでも、内部計算は同じとは限らない。特に residual block では `x + F(x)` の形になるため、「入力からどれだけ変えるか」を学ぶ見方ができる。
- 画像モデルを比較するときは、shape だけでなく「入力との差分がどう扱われるか」を見ると、block の意味を捉えやすい。

## 演習
### 基礎レベル
1. `session07_image_block_compare.py` を作成し、次の import を書く。
```python
import torch
from torch import nn
```
2. `torch.manual_seed(7)` を実行し、`x = torch.randn(2, 8, 32, 32)` を作る。
3. `PlainBlock` を次の通り実装する。
```python
class PlainBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
```
4. `ResidualBlock` を次の通り実装する。
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))
```
5. `plain_out = PlainBlock(8)(x)` と `res_out = ResidualBlock(8)(x)` を実行する。
6. `plain_out.shape` と `res_out.shape` を表示し、各 block の parameter 数を数える。
7. `session07_image_block_compare_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 入力と出力 shape`
   - `## parameter 数`
   - `## block の違い`
8. `## block の違い` には、skip connection の有無を 3 行以内で書く。

### 発展レベル
1. `plain_diff = (plain_out - x).abs().mean()` と `res_diff = (res_out - x).abs().mean()` を計算する。
2. `session07_image_block_compare_report.md` に `## 入力との差分` を追加し、`plain_diff` と `res_diff` の値を書く。
3. `parameter 数` と `入力との差分` を見比べて、「shape と parameter 数が似ていても block の意味が同じではない理由」を 3 行以内で説明する。
4. B4 は residual block を 2 個並べたときに期待する利点を 2 行書く。
5. M は shape が保てない場合に skip connection をそのまま足せない理由を 2 行書く。

## 確認ポイント
- `x.shape`, `plain_out.shape`, `res_out.shape` がすべて `(2, 8, 32, 32)` である。
- report に `入力と出力 shape`、`parameter 数`、`block の違い` があり、発展課題では入力との差分も説明されている。
- skip connection の説明が、単に「足している」ではなく、入力を保持しながら変化量を学ぶという見方に触れている。

## 詰まったときに見る資料
- [`../autumn/26-image-baseline-mini-implementation.md`](../autumn/26-image-baseline-mini-implementation.md)
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
