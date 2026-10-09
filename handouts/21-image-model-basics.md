# 第21回 画像モデル基礎：CNN・ResNet・U-Net・receptive field・normalization

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- plain block と residual block の違いをコードで比較できる。
- 出力 shape と parameter 数を見て、見た目の違いと内部の違いを分けて説明できる。
- skip connection が表現と勾配の流れにどう関わるかを言葉で説明できる。

## 解説
- CNN block は「畳み込みの並び」で考えると読みやすいが、ResNet ではそこに入力を足し戻す skip connection が入る。これにより、同じ shape を保ちながら振る舞いが変わる。
- output shape と parameter 数が同じでも、内部計算は同じとは限らない。特に residual block では `x + F(x)` の形になるため、「入力からどれだけ変えるか」を学ぶ見方ができる。
- 画像モデルを比較するときは、shape だけでなく「入力との差分がどう扱われるか」を見ると、block の意味を捉えやすい。

### 比較するモデルの仕様

`import torch` と `from torch import nn` を使い、モデル作成前に `torch.manual_seed(7)` を設定する。
各blockを `nn.Module` のサブクラスとして定義し、`__init__` の先頭で `super().__init__()` を呼ぶ。
その中で次の共通部分を `self.block = nn.Sequential(...)` として登録する。

| 順序 | 層 |
| --- | --- |
| 1 | `nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True)` |
| 2 | `nn.BatchNorm2d(8)` |
| 3 | `nn.ReLU()` |
| 4 | 1と同じ設定の新しい `nn.Conv2d` |
| 5 | `nn.BatchNorm2d(8)` |

`forward(self, x)` の戻り値は、`PlainBlock` では `torch.relu(self.block(x))`、`ResidualBlock` では `torch.relu(x + self.block(x))` とする。
`plain = PlainBlock()` と `residual = ResidualBlock()` を作る。
両方の `self.block` の初期値をそろえるため、`residual.block.load_state_dict(plain.block.state_dict())` でコピーする。
比較前に両モデルを `eval()` にし、`torch.no_grad()` 内で同じ `x` を入力する。
今回のBatchNormは学習前の統計量を使うため、これは学習性能ではなく構造の比較である。

入力の軸は `(バッチ, チャネル, 高さ, 幅)` である。paddingにより空間サイズを保ち、チャネル数も同じにすることで `x` を加算できる。
パラメータ数は `sum(p.numel() for p in model.parameters())` で求める。各blockは1200個で、加算自体には学習パラメータがない。
発展では新しい `ResidualBlock` を2個作って `nn.Sequential` でつなぎ、パラメータを共有しない。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

### 基礎レベル
1. `session21_image_block_compare.py` を作成し、同じ入力 `x = torch.randn(2, 8, 32, 32)` に対して `PlainBlock` と `ResidualBlock` の forward を通す。
2. 2 つの block の output shape、parameter 数、`(out - x).abs().mean()` を計算する。
3. 入力と出力の shape，parameter 数，block の違い，入力との差分をコードコメントまたは既存の結果ファイルに記録する。
4. shape と parameter 数が近くても、skip connection の有無によって block の意味が変わる理由を、`x + F(x)` という見方に触れて説明する。

### 発展レベル
1. residual block を 2 個並べた model を作り、shape と parameter 数がどう変わるかを確認する。
2. B4 は、block を深くする利点と起きやすい問題を 2 行ずつ書く。M は、channel 数や解像度が変わる場合に skip connection をそのまま足せない理由と対処案を 2 行ずつ書く。

## 確認ポイント
- `x.shape`, `plain_out.shape`, `res_out.shape` がすべて `(2, 8, 32, 32)` である。
- 基礎課題の記録に入力と出力の shape，parameter 数，block の違い，入力との差分がある。
- skip connection の説明が、単に「足している」ではなく、入力を保持しながら変化量を学ぶという見方に触れている。

## 詰まったときに見る資料
- [`19-autodiff-and-optimization.md`](19-autodiff-and-optimization.md): Module・推論モード
- PyTorch docs: [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
