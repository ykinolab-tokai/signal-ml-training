# 第06回 data pipeline engineering：Dataset・DataLoader・前処理・augmentation

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- metadata、Dataset、collate、DataLoader の役割を分けて実装できる。
- augmentation の有無で入力値がどう変わるかを確認できる。
- augmentation が label を保つかどうかを、データの意味から説明できる。

## ミニ講義
- `Dataset` は 1 サンプルを返し、`DataLoader` は複数サンプルをまとめてバッチにする。責務を分けると、前処理や augmentation をどこで行うか整理しやすい。
- augmentation は「見た目が変わる操作」ではなく、「task の label を保ったまま入力分布を広げる操作」と考える。だから、何でも足せばよいわけではない。
- collate 関数では、各サンプルをバッチへどう詰めるかを明示する。shape を追えるようにしておくと、後段の model 実装で迷いにくい。

## 演習
### 基礎レベル
1. `session06_dataloader_demo.py` を作成し、次の import を書く。
```python
import torch
from torch.utils.data import Dataset, DataLoader
```
2. metadata を次の通り定義する。
```python
metadata = [
    ("vertical", 0),
    ("horizontal", 1),
    ("vertical", 0),
    ("horizontal", 1),
]
```
3. `StripeDataset` を作り、`mode` が `vertical` のときは中央縦線、`horizontal` のときは中央横線を持つ `(1, 16, 16)` tensor を返すようにする。
4. `augment=False` のときはそのまま返す。
5. `augment=True` のときは `torch.flip(image, dims=[2])` を適用して左右反転させる。
6. `collate_fn` を作成し、images を `torch.stack`、labels を `torch.tensor` にまとめる。
7. `loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)` を作り、`batch_x, batch_y = next(iter(loader))` を取り出す。
8. `dataset_no_aug` と `dataset_aug` の 0 番目サンプルを比較し、同じ shape で値だけが変わることを確認する。
9. `session06_dataloader_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## metadata`
   - `## DataLoader の出力`
   - `## augmentation の有無`
10. `## augmentation の有無` には、左右反転で何が変わり、何が変わらないかを 3 行以内で書く。

### 発展レベル
1. `StripeDataset` に `augment="shift"` を追加し、`torch.roll(image, shifts=2, dims=2)` で水平方向へずらす条件を実装する。
2. `session06_dataloader_report.md` に `## label を保つ augmentation` を追加し、`flip` と `shift` のそれぞれについて、`vertical` / `horizontal` の label が保たれるかを表ではなく文章で書く。
3. `flip` と `shift` のどちらが今回の task に向いているかを 3 行以内で説明する。shape ではなく、線の向きという task 定義に触れること。

## 確認ポイント
- `len(metadata)` が `4` である。
- `batch_x.shape` が `(2, 1, 16, 16)`、`batch_y.shape` が `(2,)` である。
- report に、augmentation 後も label が保たれるかどうかの判断理由が書かれている。
- `Dataset`、`collate_fn`、`DataLoader` の役割が混ざっていない。

## 詰まったときに見る資料
- [`../autumn/22-image-data-basics.md`](../autumn/22-image-data-basics.md)
- [`../autumn/24-pytorch-basics.md`](../autumn/24-pytorch-basics.md)
