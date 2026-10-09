# 第03回 数学基礎と NumPy, Matplotlib

## この回の目標

- 初等関数を NumPy で計算し，Matplotlib でグラフにする。
- グラフの平行移動を式と図の両方で確認する。
- NumPy 配列の shape，dtype，スライスを確認する。
- 計算結果と図を file として保存する。

## 解説
- 初等関数は，入力 `x` に対して出力 `y` を返す対応として見る。NumPy 配列を入力にすると，多数の点で同時に関数値を計算できる。
- グラフの平行移動は，式の中の `x - a` と外側の `+ b` が，横方向と縦方向の移動に対応することを図で確認すると理解しやすい。
- NumPy 配列では，値だけでなく `shape`，`dtype`，次元数を確認する。後の信号処理と画像処理では，shape の読み間違いが実装ミスにつながる。
- Matplotlib の図は画面表示だけで終えず，`savefig` で保存する。保存された図と script を対応づけることで，結果を再現しやすくなる。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

`scripts/session03_math_numpy_matplotlib.py` に実装し，結果と説明はコードコメントまたは既存の結果ファイルに残す。

次の準備コードをファイルの先頭に置き，演習のコードを続ける。

```python
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
```

### 基礎レベル
1. `a = np.arange(12).reshape(3, 4)` の `shape`，`dtype`，`ndim`，`a[1, :]`，`a[:, 1]`，`a[::2, 1:3]` の値とshapeを予想してから確認する。`axis=0` は行を集約して列ごとの値を，`axis=1` は列を集約して行ごとの値を返すことを確かめる。
2. `x = np.linspace(-5, 5, 501)` で `f(x)=x**2` と `g(x)=(x-2)**2+1` を描く。移動方向を先に予想し，凡例・軸ラベル付きの図を `outputs/figures/session03_translation.png` へ保存する。
3. `centered = a - a.mean(axis=0)` を読み，引く配列のshapeと各行への適用を説明する。各列の平均が0になることを確認する。この処理を列ごとの中心化と呼び，後のPCAの前処理につながることを確認する。
4. `a_list = [1, 2, 3]` の `a_list + a_list` と，`np.array(a_list) + np.array(a_list)` を比較する。図・配列のshape・axisの予測結果・中心化の説明をコードコメントまたは既存の結果ファイルにまとめる。

### 発展レベル（1項目を選択）
1. `sin(x)` の描画点数を21，101，1001に変え，図の滑らかさと配列のshapeを比較する。
2. `np.log` と `np.sqrt` の定義域を確認する。負の値を入力したときのwarningと `nan` は意図的な失敗例として区別する。
3. `x` と `sin(x)` の2列を `np.savetxt` で `outputs/data/session03_sin_table.csv` に保存し，再読込でshapeと値を確認する。

## 確認ポイント
- スライスとaxisごとの出力shapeを，実行前の予測と比較している。
- 平行移動の向きと列平均を引く操作を，式・図・数値に対応づけて説明できる。
- スクリプトと図が指定場所にあり，予測・結果・説明をコードコメントまたは既存の結果ファイルから確認できる。

## 詰まったときに見る資料
- [`../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md`](../textbook/markdown/ch03-reviewing-elementary-math-with-numpy-and-matplotlib.md)
- [`../textbook/markdown/ch08-applied-matplotlib.md`](../textbook/markdown/ch08-applied-matplotlib.md)
