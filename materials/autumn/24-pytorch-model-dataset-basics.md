# 第24回 PyTorch による model 実装の基礎：Module・Dataset・DataLoader・batch・logits

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- `Dataset` が 1 サンプルを返し、`DataLoader` がバッチを作ることを説明できる。
- 最小の `nn.Module` を定義し、forward を通せる。
- batch size を変えたとき、どの次元が変わるかを説明できる。

## ミニ講義
- Tensor の `shape` と `dtype` は第17回で確認済みとし、この回では model と data pipeline の接続に集中する。
- `Dataset` はサンプル単位、`DataLoader` はバッチ単位で考える。まず 1 サンプルの shape を確認してからバッチ shape を見ると、先頭次元の意味が分かりやすい。
- `nn.Module` は入力 tensor を受けて出力 tensor を返す部品である。forward の結果 shape は「入力のどの次元が保たれ、どの次元が変換されたか」で読む。
- batch size を変えても、通常は特徴量次元や class 数は変わらない。変わるのは先頭の batch 次元である。

## 演習
### 基礎レベル
1. `session24_pytorch_basics.py` を作成し、指定された `ToyDataset` を実装する。`__len__` と `__getitem__` が何を返すかを print で確認する。
2. `DataLoader(ToyDataset(), batch_size=2, shuffle=False)` から最初の batch を取り出し、`batch_x.shape` と `batch_y.shape` を表示する。
3. 指定された `ToyNet` を実装し、`logits = model(batch_x)` の shape を表示する。
4. `session24_pytorch_report.md` に `## Dataset が返すもの`, `## DataLoader が返すもの`, `## Model の出力` を書き、先頭次元が何を表すかを説明する。

### 発展レベル
1. `batch_size=1` と `batch_size=2` の batch shape と logits shape を比較する。
2. report に `## batch size の比較` を追加し、batch size で変わる次元と、class 数に対応して変わらない次元を説明する。

## 確認ポイント
- `len(ToyDataset())` が `6` である。
- `batch_x.shape` が `(2, 4)`、`batch_y.shape` が `(2,)`、`logits.shape` が `(2, 2)` である。
- report に、1 サンプル shape とバッチ shape の違いが書かれている。
- 発展課題で、先頭次元が batch 次元であることを説明している。

## 詰まったときに見る資料
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
- [`17-array-and-tensor-basics.md`](17-array-and-tensor-basics.md)
- PyTorch docs: [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
