# 第04回 数値線形代数の実装：最小二乗・SVD・PCA

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- 固定データ行列に対して平均中心化、SVD、1 次元 PCA 射影、再構成を実行できる。
- `X`, `Z`, `X_recon` の shape の変化を追える。
- mean centering の有無が主成分と再構成に与える影響を説明できる。

## ミニ講義
- PCA は「分散が大きい方向へ低次元射影する」方法で、実装では平均中心化の有無が結果に強く効く。中心化しないと、原点からの位置そのものが主成分に混ざりやすい。
- `np.linalg.svd(X_centered, full_matrices=False)` では、行列の向きと shape を意識すると何が射影方向か追いやすい。今回の 2 次元データでは `Vt[0]` が第1主成分方向になる。
- 再構成は「1 次元表現 `Z` から元の空間へ戻した近似」であり、元データとの差を見ると、何を捨てて何を残したかが分かる。

## 演習
### 基礎レベル
1. `session04_pca_demo.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
```
2. データ行列を次の通り定義する。
```python
X = np.array([
    [2.0, 0.0],
    [0.0, 2.0],
    [3.0, 1.0],
    [1.0, 3.0],
    [-2.0, 0.0],
    [0.0, -2.0],
], dtype=np.float64)
```
3. `X_mean = X.mean(axis=0, keepdims=True)`、`X_centered = X - X_mean`、`U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)` を順に計算する。
4. `pc1 = Vt[0:1]`、`Z = X_centered @ pc1.T`、`X_recon = Z @ pc1 + X_mean` を計算する。
5. 元データ `X` と再構成データ `X_recon` を同じ散布図に描き、`outputs/figures/session04_pca_projection.png` に保存する。
6. `session04_pca_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## shape 一覧`
   - `## 特異値`
   - `## 再構成結果`
7. `## shape 一覧` には `X`, `X_centered`, `Z`, `X_recon` の shape を書く。
8. `## 特異値` には `S` の値を書く。
9. `## 再構成結果` には元データと再構成データの違いを 3 行以内で書く。

### 発展レベル
1. 同じ `X` を使い、平均中心化を行わない版 `X_no_center = X` でも SVD と 1 次元再構成を実行する。
2. `session04_pca_report.md` に `## mean centering の有無` を追加し、中心化あり・なしで次の 3 点を比較する。
   - 第1主成分方向
   - 再構成の見え方
   - 元データのどの情報が残りやすいか
3. 比較結果を 4 行以内で書き、「なぜ PCA の前に中心化することが多いのか」を自分の言葉で説明する。

## 確認ポイント
- `X.shape` が `(6, 2)`、`Z.shape` が `(6, 1)`、`X_recon.shape` が `(6, 2)` である。
- `session04_pca_projection.png` に元データと再構成データの 2 系列がある。
- report に特異値だけでなく、再構成の解釈が書かれている。
- 発展課題で、中心化の有無による違いを shape ではなく意味の違いとして説明している。

## 詰まったときに見る資料
- [`../autumn/18-linear-algebra-basics.md`](../autumn/18-linear-algebra-basics.md)
- [`../../latex/markdown/ch06a-basics-of-linear-algebra.md`](../../latex/markdown/ch06a-basics-of-linear-algebra.md)
