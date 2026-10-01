# プロジェクト3：burst画像のノイズ除去

## 目的と入力

位置ずれのない連写画像を平均して，独立な雑音を減らす。
第13回の画像コースでは，以下の生成例と「基礎レベル」を用いる。
入力には教材repoの `data/cat.png` を使い，提出repoの `data/cat.png` へコピーしておく。
追加のデータやライブラリは使わない。

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

with Image.open("data/cat.png") as image:
    image = image.convert("RGB")
    image.thumbnail((256, 256))
    clean = np.asarray(image, dtype=np.float64) / 255.0

rng = np.random.default_rng(0)
noise = rng.normal(0.0, np.sqrt(0.1), size=(9, *clean.shape))
burst = np.clip(clean[None, ...] + noise, 0.0, 1.0)
figure_dir = Path("outputs/figures")
figure_dir.mkdir(parents=True, exist_ok=True)
print(clean.shape, burst.shape)
```

雑音は平均0，分散0.1の正規分布から画素・チャネル・フレームごとに独立に生成する。
最初に9枚まとめて生成し，枚数の比較にはその先頭を使う。
各観測画像を `[0,1]` にclipするため，境界付近には偏りが生じる。
clipしない独立な加法雑音なら平均後の雑音分散は $1/K$ になるが，この入力のMSEが厳密に $1/K$ になるとは限らない。

## 基準手法と評価

先頭 $K$ 枚の画素ごとの算術平均を復元画像とする。

$$
\hat{x}_K=\frac{1}{K}\sum_{k=1}^{K} y_k
$$

全RGB画素に対して平均二乗誤差を計算する。
画素値の範囲は `[0,1]` なので，PSNRは次の式を用いる。
MSEが0ならPSNRは無限大として扱う。

$$
\mathrm{MSE}=\operatorname{mean}((x-\hat{x})^2),\qquad
\mathrm{PSNR}=10\log_{10}(1/\mathrm{MSE})
$$

## 基礎レベル

1. 上のコードに続けて，`burst[:K].mean(axis=0)` で $K=1,3,9$ の復元画像を作る。平均する軸と出力のshapeを説明する。$K=1$ は未処理の観測画像，$K=3$ は基準手法，$K=9$ は比較条件である。
2. 各条件のMSEとPSNRを計算する。正解，$K=1,3,9$ の画像を同じ表示範囲で並べ，`outputs/figures/session13_comparison.png` に保存する。
3. [第13回](../13-image-baseline-mini-implementation.md)で指定したレポートに，数値，枚数を増やした影響，clipによって残る誤差を記す。

## 発展レベル（任意）

- 同じ9枚に対する中央値 `np.median(burst, axis=0)` と平均を比較する。
- 雑音の標準偏差を変えて，再度同じ条件で平均と中央値を比較する。

SSIMなど別の評価尺度は，定義・利用ライブラリを別途学んだ後の発展候補とする。
今回の完了条件はMSEとPSNRである。
