# 第20回 data pipeline engineering：Dataset・DataLoader・前処理・augmentation

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- metadata、Dataset、collate、DataLoader の役割を分けて実装できる。
- augmentation の有無で入力値がどう変わるかを確認できる。
- augmentation が label を保つかどうかを、データの意味から説明できる。

## 解説
- `Dataset` は 1 サンプルを返し、`DataLoader` は複数サンプルをまとめてバッチにする。責務を分けると、前処理や augmentation をどこで行うか整理しやすい。
- augmentation は「見た目が変わる操作」ではなく、「task の label を保ったまま入力分布を広げる操作」と考える。だから、何でも足せばよいわけではない。
- collate 関数では、各サンプルをバッチへどう詰めるかを明示する。shape を追えるようにしておくと、後段の model 実装で迷いにくい。

### metadataとDatasetの仕様

第19回と同じPython環境を使う。次のデータは `(線の向き, 分類ラベル)` の4件で、ラベル `0` が縦線、`1` が横線を表す。

```python
import torch
from torch.utils.data import Dataset, DataLoader

metadata = [
    ("vertical", 0), ("horizontal", 1),
    ("vertical", 0), ("horizontal", 1),
]
```

`StripeDataset(Dataset)` の `__init__(self, metadata, augment=False)` で一覧と変換条件を保持する。
`__len__(self)` は一覧の長さ、`__getitem__(self, index)` は1件の `(image, label)` を返す。
各呼び出しで `torch.zeros(1, 16, 16, dtype=torch.float32)` を作り、縦線なら `image[0, :, 8] = 1`、横線なら `image[0, 8, :] = 1` とする。
添字は0始まりで、画像の軸は `(チャネル, 高さ, 幅)` である。ラベルはPythonの整数として返す。

`augment=False` では画像をそのまま、`augment=True` では `torch.flip(image, dims=[2])` を返す。
縦線の列は8から7へ移るが、横線の画像は左右反転しても同じ値になる。両方のサンプルで違いを確認する。
発展の `augment="shift"` は `torch.roll(image, shifts=2, dims=2)` とし、`True` の場合と文字列の場合を区別して分岐する。
`roll` は端からはみ出した値を反対側へ回す循環シフトである。

### 1サンプルからバッチへ

`collate_fn(batch)` に届くのは複数の `(image, label)` のリストである。
imagesを `torch.stack(images)`、labelsを `torch.tensor(labels, dtype=torch.long)` でまとめ、2つのTensorを返す。
`DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)` を作り、`next(iter(loader))` で最初のバッチを取り出す。
画像にはバッチ軸が加わって `(2, 1, 16, 16)`、ラベルは `(2,)` になる。最初のラベルは `[0, 1]` である。

## 演習
### 基礎レベル
1. `session20_dataloader_demo.py` を作成し、`metadata` から `vertical` / `horizontal` の stripe 画像 `(1, 16, 16)` と label を返す `StripeDataset` を実装する。
2. `augment=False` と `augment=True` の dataset を作り、左右反転によって入力 tensor のどこが変わり、shape と label がどう保たれるかを確認する。
3. `collate_fn` と `DataLoader(batch_size=2, shuffle=False)` を実装し、`batch_x.shape == (2, 1, 16, 16)`, `batch_y.shape == (2,)` になることを確認する。
4. `session20_dataloader_report.md` に `## metadata`, `## DataLoader の出力`, `## augmentation の有無` を書き、Dataset, collate, DataLoader の責務を混同しないように説明する。

### 発展レベル
1. `augment="shift"` を追加し、`torch.roll(image, shifts=2, dims=2)` による水平方向 shift を実装する。
2. `session20_dataloader_report.md` に `## label を保つ augmentation` を追加し、`flip` と `shift` が `vertical` / `horizontal` の label を保つかを task 定義に基づいて説明する。
3. 今回の stripe 分類で最初に採用する augmentation を 1 つ選び、採用理由と避けたい失敗例を 3 行以内で書く。

## 確認ポイント
- `len(metadata)` が `4` である。
- `batch_x.shape` が `(2, 1, 16, 16)`、`batch_y.shape` が `(2,)` である。
- report に、augmentation 後も label が保たれるかどうかの判断理由が書かれている。
- `Dataset`、`collate_fn`、`DataLoader` の役割が混ざっていない。

## 詰まったときに見る資料
- [`19-autodiff-and-optimization.md`](19-autodiff-and-optimization.md): Tensorとモデルの基礎
- PyTorch docs: [Dataset・DataLoader・collate](https://docs.pytorch.org/docs/stable/data.html)
