# 第12回 画像信号

## この回の目標

- 2次元信号として画像を扱い，shape と値域を確認する。
- Pillow と Matplotlib で画像を読み込み，表示，保存する。
- 空間領域と周波数領域の見え方を対応づける。

## 解説

### 0. 準備

この回では，画像の操作のために Python のライブラリ `Pillow` を使用する．

第1回の教材repoで `uv sync --locked` を行い，共通の `.venv` を使う。
作業場所と保存先は [共通方針](README.md#作業場所と保存先) に従う。

### 1. 画像の表現
画像は，縦横方向の空間座標を独立変数として，
その位置における光の強さ（明るさ）を表す
2次元信号として表現できる．
すなわち，画像は，座標 $(n_1, n_2)$ における
画素の明るさを表す信号 $x[n_1, n_2]$ として表される．
画像を扱う場合，座標 $n_1$ を行方向の空間座標，$n_2$ を列方向の空間座標とし，
原点 $(n_1, n_2) = (0, 0)$ を画像の左上隅に置くことが一般的である．

画素値 $x[n_1, n_2]$ は整数や浮動小数点数で表されるが，
整数を用いる場合には $[0, 255]$ の範囲で扱うことが多く，
浮動小数点数を用いる場合には $[0, 1]$ の範囲で扱うことが多い．
一方，信号処理の理論を考えるうえでは，
画素値を複素数とする方が便利である．

カラー画像の表現には，赤 (red)，緑 (green)，青 (blue) のそれぞれに対応する3つの
画素値 $x_r[n_1, n_2], x_g[n_1, n_2], x_b[n_1, n_2]$ を用いることが一般的である．
このように，RGBの組み合わせで表される画像をRGB画像と
いう．
RGB画像は，3チャネルの2次元信号である．
RGB画像に対して，1つの画素値 $x[n_1, n_2]$ だけを持つ
画像は **グレースケール (grayscale) 画像** と呼ばれる．

#### 画像の読み込み
画像をファイルから読み込むときには，Pillow の `Image.open` 関数を用いる．

```python
from PIL import Image
img = Image.open('data/cat.png').convert('RGB')  # 教材repoからコピーした入力
```
このとき，`img` は `PIL.Image.Image` クラスのオブジェクトである．

#### 画像の保存
画像をファイルに保存するときには，Pillow の `Image.save` 関数を用いる．

```python
from PIL import Image
img = Image.new('L', (128, 128))  # グレースケール画像を新規作成
from pathlib import Path
Path('outputs/images').mkdir(parents=True, exist_ok=True)
img.save('outputs/images/session12_blank.png')
```

#### NumPy 配列との相互変換
Pillow の `Image` オブジェクトと NumPy 配列は，次のようにして相互に変換できる．

```python
import numpy as np
from PIL import Image

# PIL → numpy (高さ, 幅, チャンネル) の配列
arr = np.array(img)             

# numpy → PIL
arr = np.clip(arr, 0, 255).astype(np.uint8) # 値域を [0, 255] にクリップし，uint8 型に変換
img2 = Image.fromarray(arr)
```

画像の高さを $H$、幅を $W$、チャンネル数を $C$ とすると，
Pillow から NumPy 配列に変換した場合，
カラー画像の配列は (H, W, C)，グレースケールは (H, W) の形状を持つ．
特に，RGB画像の配列は，$C=3$ であり， `arr[:, :, 0] ` が赤チャネル， `arr[:, :, 1] ` が緑チャネル，
 `arr[:, :, 2] ` が青チャネルの画素値を表す．
また，画素値は多くの場合 uint8 型で $[0, 255]$ の範囲で表される．
このルールは，Pillow に限ったものであり，用いるライブラリによっては異なる規則で画像が表されることもあるので注意する．

`fromarray` で配列を `Image` オブジェクトに変換する際は，渡す配列を原則 `uint8` 型にしておく．

#### 画像の表示
画像を表示するときには，Matplotlib の `imshow` 関数を用いる．

```python
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("data/cat.png").convert("RGB")

plt.imshow(img)      # PIL画像もnumpy配列もそのまま渡せる
plt.axis("off")      # 軸を非表示
from pathlib import Path
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/session12_input.png")
plt.close()
```

`imshow` のデフォルトでは，
グレースケール画像をその画素値の大小に応じたカラーマップで表示する．
グレースケール画像をグレースケールで表示したい場合は，次のように `cmap='gray'` を指定する．

```python
plt.imshow(img, cmap='gray')
```

### 2. 2次元信号

理論を考える上で重要な2次元信号をいくつか紹介する．

#### 単位インパルス信号

$$
\delta[n_1, n_2] = \begin{cases}
1 & n_1 = 0, n_2 = 0 \\
0 & \text{otherwise}
\end{cases}
$$

#### 複素正弦波

$$
e^{j (\omega_{1} n_1 + \omega_{2} n_2)}
$$
- $\omega_1, \omega_2$：それぞれ，$n_1, n_2$ 方向の空間角周波数

指数法則より，2次元複素正弦波は，
$n_1$ 方向の1次元複素正弦波と $n_2$ 方向の1次元複素正弦波の積で表せる．

$$
e^{j (\omega_{1} n_1 + \omega_{2} n_2)}
= e^{j \omega_{1} n_1} e^{j \omega_{2} n_2}
$$

また，オイラーの公式より，複素正弦波は次のように表せる．
$$
e^{j (\omega_{1} n_1 + \omega_{2} n_2)}
= \cos(\omega_{1} n_1 + \omega_{2} n_2)
+ j \sin(\omega_{1} n_1 + \omega_{2} n_2)
$$

$\cos(\omega_{1} n_1 + \omega_{2} n_2)$ や
$\sin(\omega_{1} n_1 + \omega_{2} n_2)$ を，
2次元正弦波という．

### 3. 2次元信号の演算

2次元信号に対しても，1次元信号と同様に基本演算が定義される．

#### シフト

信号を平行移動させる

$$
y[n_1, n_2] = x[n_1 - k_1, n_2 - k_2]
$$

- $k_1, k_2 \in \mathbb{Z}$：シフト量

#### 反転

信号を $n_1$ もしくは $n_2$ 軸に関して反転させる．

$$
y[n_1, n_2] = x[-n_1, n_2], \quad y[n_1, n_2] = x[n_1, -n_2]
$$

#### 拡大・縮小

信号を $n_1$ もしくは $n_2$ 軸に関して伸び縮みさせる．

$$
y[n_1, n_2] = x[an_1, an_2]
$$

- $a>0$ は出力座標から入力を参照する座標倍率であり，画像内容の拡大率は $1/a$ である。$a=2$ は縮小，$a=1/2$ は拡大に対応する。
- 抽象的な離散信号の操作として，参照座標が整数でないときに0を置く定義もできる。ただし，通常の画像拡大・回転では近傍の画素から補間する。Pillowの `resize` には出力サイズと補間法を指定する。

#### 回転
信号を原点を中心に $\theta$ だけ回転させる．

$$
y[n_1, n_2] = x[m_1, m_2]
$$

ただし，
$$
\begin{pmatrix}m_1 \\
m_2\end{pmatrix}
=
\begin{pmatrix}\cos \theta & \sin \theta \\
-\sin \theta & \cos \theta\end{pmatrix}
\begin{pmatrix}n_1 \\ n_2\end{pmatrix}
$$

#### 幾何変換

出力の整数画素座標から入力の参照座標への写像 $f: \mathbb{Z}^2 \to \mathbb{R}^2$ について，

$$
\begin{pmatrix}m_1 \\ m_2\end{pmatrix}
= f \begin{pmatrix}n_1 \\ n_2\end{pmatrix}
$$

としたとき，

$$
y[n_1, n_2] = x[m_1, m_2]
$$

で与えられる変換を一般に **幾何変換** と呼ぶ．

$f$ が以下の形で与えられるときを特に， **射影変換** と呼ぶ．

$$
\lambda\begin{pmatrix}m_1 \\ m_2 \\ 1\end{pmatrix}
=
\begin{pmatrix}a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}\end{pmatrix}
\begin{pmatrix}n_1 \\ n_2 \\ 1\end{pmatrix}
$$

右辺を $(u,v,w)^\top$ とすると，$w\ne0$ の位置で参照座標は $m_1=u/w$，$m_2=v/w$ となる。
ここで $\lambda=w$ である。$w=0$ は有限の参照座標を持たないので，出力を0などの背景値とする規則が必要になる。
非整数の参照座標には補間を用い，画像外には指定した背景値を用いる。
上の式は出力から入力を参照する向きであり，入力点を出力へ移す行列を使う場合は逆行列が必要である。

また，上式において $a_{31} = a_{32} = 0, \; a_{33} = 1$ の場合を，**アフィン変換** と呼ぶ．
アフィン変換における
$$
\boldsymbol{A} = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
$$
が実数 $s$と回転行列 $\boldsymbol{R}(\theta)$ を用いて $\boldsymbol{A} = s \boldsymbol{R}(\theta)$ の形で表される場合を，
**相似変換** と呼ぶ．

上記の包含関係をまとめると，以下のようになる．

$$
\text{相似変換} \subset \text{アフィン変換} \subset \text{射影変換} \subset \text{幾何変換}
$$

シフト，反転，拡大・縮小，回転はすべてアフィン変換の特別な場合である．


#### 振幅スケーリング・オフセット

$$
y[n_1, n_2] = a \cdot x[n_1, n_2] + b
$$

- $a$：振幅の倍率（負の数なら符号反転）
- $b$：オフセット（振幅方向の平行移動量）

#### 2つの信号の演算

2つの2次元信号 $x_1, x_2$ に対して：

- **加算**：$y[n_1, n_2] = x_1[n_1, n_2] + x_2[n_1, n_2]$

- **乗算**：$y[n_1, n_2] = x_1[n_1, n_2] \; x_2[n_1, n_2]$

- **内積**：$\displaystyle \langle x_1, x_2 \rangle = \sum_{n_1=-\infty}^{\infty} \sum_{n_2=-\infty}^{\infty} \overline{x_1[n_1, n_2]} x_2[n_1, n_2]$

    - ここで，$\overline{z}$ は複素数 $z$ の複素共役を表す．

- **畳み込み**：$\displaystyle y[n_1, n_2] = (x_1 * x_2)[n_1, n_2] = \sum_{k_1=-\infty}^{\infty} \sum_{k_2=-\infty}^{\infty} x_1[k_1, k_2] x_2[n_1 - k_1, n_2 - k_2]$

- **相関**：$\displaystyle y[n_1, n_2] = (x_1 \star x_2)[n_1, n_2] = \sum_{k_1=-\infty}^{\infty} \sum_{k_2=-\infty}^{\infty} \overline{x_1[k_1, k_2]} x_2[n_1 + k_1, n_2 + k_2]$


### 4. 2次元DFT

1次元の場合と同様に，2次元複素正弦波の直交性を利用して，
2次元離散時間信号を複素正弦波 $\phi_{k_1, k_2}[n_1, n_2] = e^{j 2 \pi (k_1 n_1 / N_1 + k_2 n_2 / N_2)}$
の線形結合として表すことができる（2次元離散フーリエ変換）．

$$
x[n_1, n_2] =
\frac{1}{N_1 N_2} \sum_{k_1=0}^{N_1-1} \sum_{k_2=0}^{N_2-1} X[k_1, k_2] e^{j 2 \pi (k_1 n_1 / N_1 + k_2 n_2 / N_2)}
\qquad \text{逆DFT}
$$

$$
X[k_1, k_2] = \langle \phi_{k_1, k_2}, x \rangle = \boldsymbol{\phi}_{k_1, k_2}^* \boldsymbol{x} = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2-1} \overline{\phi_{k_1, k_2}[n_1, n_2]} x[n_1, n_2] = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2-1} x[n_1, n_2] e^{-j 2 \pi (k_1 n_1 / N_1 + k_2 n_2 / N_2)}
\qquad \text{DFT}
$$

2次元DFT（逆DFT）は，1次元DFT（逆DFT）を行方向と列方向に順番に適用することで計算できる．

$$
X[k_1, k_2] = \sum_{n_1=0}^{N_1-1} \left( \sum_{n_2=0}^{N_2-1} x[n_1, n_2] e^{-j 2 \pi k_2 n_2 / N_2} \right) e^{-j 2 \pi k_1 n_1 / N_1}
\qquad \text{DFTの場合}
$$

### 5. 線形シフト不変システム
2次元の場合においても，信号 $x$ を $y$ に変換する信号処理システム $\mathcal{S}$ を考えることができる．

#### 線形性
任意の $a, b \in \mathbb{C}$ および
任意の信号 $x_1, x_2$ に対して，
以下の性質を満たすシステム $\mathcal{S}$ を，
**線形システム** と呼ぶ．

$$
\mathcal{S}\{a x_1 + b x_2\} = a \mathcal{S}\{x_1\} + b \mathcal{S}\{x_2\}
$$

#### シフト不変性

信号を $m_1, m_2$ サンプル遅延させる以下のシステム $\mathcal{D}_{m_1, m_2}$ を考える．

$$
y[n_1, n_2] = \mathcal{D}_{m_1, m_2}\{x\}[n_1, n_2] = x[n_1 - m_1, n_2 - m_2]
$$

このとき，任意の信号 $x$ および任意の $m_1, m_2$ に対して，
以下の性質を満たすシステム $\mathcal{S}$ を，
**シフト不変システム** と呼ぶ．

$$
\mathcal{S}\{\mathcal{D}_{m_1, m_2}\{x\}\} = \mathcal{D}_{m_1, m_2}\{\mathcal{S}\{x\}\}
$$

線形性とシフト不変性の両方を満たすシステムを，
**線形シフト不変システム**（Linear Shift-Invariant System, LSI システム）と呼ぶ．
線形シフト不変システムの入出力関係は，そのインパルス応答 $h[n_1, n_2]$ との畳み込みを用いて以下の通り表される．

$$
y[n_1, n_2] = (x * h)[n_1, n_2] = \sum_{k_1=-\infty}^{\infty} \sum_{k_2=-\infty}^{\infty} x[k_1, k_2] h[n_1 - k_1, n_2 - k_2]
$$


## 演習

作業場所は [提出repo](README.md#作業場所と保存先) のルートとする。

`data/cat.png` を教材repoからコピーする。`scripts/session12_image.py` と `outputs/session12/session12_report.md` を作り，画像・図を `outputs/images/` と `outputs/figures/` へ保存する。

### 基礎レベル
1. `Image.open(...).convert("RGB")` と `.convert("L")` をNumPy配列に変換し，shape，dtype，値域を比べる。RGBとグレースケールのチャネル数を説明する。
2. グレースケールをfloatへ変換して `255-x` のネガ画像を作り，保存する。`uint8` のまま加算する場合に起こる問題を説明する。さらに `[::2,::2]` で間引き，参照座標倍率2と出力サイズの関係を確認する。
3. `n1,n2=np.indices((128,128))` とし，$\cos(\pi n_1/8)$ の画像と `fftshift(fft2(...))` の振幅を描く。縞の向きとピーク位置を予想してから，周波数 $(\pm8,0)$ に対応することを確認する。
4. `[[1,2,3],[4,5,6],[7,8,9]]` に3×3平均化フィルタを掛けた中央1画素を手計算する。その後，グレースケール画像へ `scipy.signal.convolve2d(..., mode="same", boundary="symm")` を使い，出力shapeと輪郭の変化を確認する。境界を鏡映する指定も記録する。

### 発展レベル（1項目を選択）
1. RGBの各チャネル，明るさ+50，量子化2/4 bitから1つ選んで比較画像を作る。
2. seed=0の8×8配列で行・列の1次元FFTと2次元FFTが一致することを確認する。
3. 画像中心を回転中心として30度の回転を，出力から入力への座標参照と最近傍補間で実装する。画像外は0とし，同じ補間・出力サイズのPillow `rotate` と比較する。
4. ラプラシアンフィルタまたは周波数領域の低域通過フィルタを実装し，基礎の平均化と比較する。

## 確認ポイント
- 入力画像・出力画像のshapeとdtype，計算前後の値域を説明できる。
- 幾何変換を出力から入力への参照として理解し，射影座標では第3成分で割っている。
- 2次元正弦波の向きとDFTピーク，1画素の平均化計算が実装と対応している。

## 詰まったときに見る資料
- [Pillowによる画像操作](../textbook/markdown/ch12-introduction-to-pillow.md)
- [Pillowの座標変換](https://pillow.readthedocs.io/en/stable/reference/ImageTransform.html)
