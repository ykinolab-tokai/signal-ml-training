# 第19回 確率統計の基礎：平均・分散・分布・サンプリング・基本指標

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 固定データから平均、分散、標準偏差を計算できる。
- サンプリングしたデータの分布をヒストグラムで確認できる。
- 母分散と不偏分散の違いを、今回のデータに即して説明できる。

## ミニ講義
- 平均は中心、分散と標準偏差はばらつきの大きさを見る指標である。数値だけでなく、ヒストグラムと対応づけると理解しやすい。
- サンプリングでは、元の分布を完全には見られないので、有限個の標本から推定する。だから、固定データそのものの分散と、不偏補正付きの分散は意味が少し違う。
- ヒストグラムは bar の高さを見るだけでなく、「どの値に集まりやすいか」「ばらつきが広いか」を読むための図である。

## 演習
### 基礎レベル
1. `session19_probability_statistics.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
```
2. 固定データを次の 1 行で定義する。
```python
x = torch.tensor([1.0, 2.0, 2.0, 4.0, 7.0])
```
3. `mean_x`, `var_x`, `std_x` を次の通り計算し、表示する。
```python
mean_x = x.mean()
var_x = x.var(unbiased=False)
std_x = x.std(unbiased=False)
```
4. `x` のヒストグラムを 4 bins で描き、`outputs/figures/session19_hist_fixed.png` に保存する。
5. `torch.manual_seed(19)` を実行し、次のサンプルを生成する。
```python
samples = torch.randint(low=1, high=7, size=(1000,))
```
6. `samples.float().mean()` を計算し、`samples` のヒストグラムを 6 bins で描いて `outputs/figures/session19_hist_sampled.png` に保存する。
7. `session19_probability_statistics_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 固定データの統計量`
   - `## サンプリング結果`
   - `## 2 枚のヒストグラムの違い`

### 発展レベル
1. `var_x_unbiased = x.var(unbiased=True)` を追加で計算する。
2. `session19_probability_statistics_report.md` に `## 分散の計算方法の違い` を追加し、`unbiased=False` と `unbiased=True` の値を両方書く。
3. そのうえで、「固定データそのもののばらつきを見たいとき」と「標本から推定したいとき」でどちらを使うかを 3 行以内で説明する。
4. `samples.float().mean()` と `mean_x` を見比べて、同じとは限らない理由を 2 行で書く。

## 確認ポイント
- `mean_x` が `3.2`、`var_x` が `4.56`、`std_x` が約 `2.1354` である。
- 保存画像が 2 枚とも `outputs/figures/` 配下にある。
- report に、ヒストグラムの違いと分散の定義の違いが書かれている。

## 詰まったときに見る資料
- [`16-python-numpy-matplotlib-basics.md`](16-python-numpy-matplotlib-basics.md)
- [`../../latex/markdown/ch06b-basics-of-probability-statistics.md`](../../latex/markdown/ch06b-basics-of-probability-statistics.md)
