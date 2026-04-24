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
1. `session04_pca_demo.py` を作成し、指定された 2 次元データ `X` に対して平均中心化、SVD、1 次元 PCA 射影、再構成を実装する。
2. 元データ `X` と再構成データ `X_recon` を同じ散布図に描き、`outputs/figures/session04_pca_projection.png` に保存する。図には凡例と軸ラベルを付ける。
3. `session04_pca_report.md` を作成し、`## shape 一覧`, `## 特異値`, `## 再構成結果` を書く。`X`, `X_centered`, `Z`, `X_recon` の shape と、再構成で残った情報・失われた情報を説明する。
4. 再構成誤差 `((X - X_recon) ** 2).mean()` を計算し、数値と散布図の見え方を対応づけて 2 行で説明する。

### 発展レベル
1. 平均中心化しない条件でも同じ 1 次元再構成を実行し、中心化あり・なしの第1主成分方向と再構成の違いを比較する。
2. `session04_pca_report.md` に `## mean centering の有無` を追加し、「なぜ PCA の前に中心化することが多いのか」を今回のデータに即して説明する。

## 確認ポイント
- `X.shape` が `(6, 2)`、`Z.shape` が `(6, 1)`、`X_recon.shape` が `(6, 2)` である。
- `session04_pca_projection.png` に元データと再構成データの 2 系列がある。
- report に特異値だけでなく、再構成の解釈が書かれている。
- 発展課題で、中心化の有無による違いを shape ではなく意味の違いとして説明している。

## 詰まったときに見る資料
- [`../autumn/18-linear-algebra-basics.md`](../autumn/18-linear-algebra-basics.md)
- [`../../latex/markdown/ch06a-basics-of-linear-algebra.md`](../../latex/markdown/ch06a-basics-of-linear-algebra.md)
