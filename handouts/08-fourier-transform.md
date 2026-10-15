# 第08回 Fourier 変換

## この回の目標

DFT と逆変換を使い，信号の周波数成分と窓関数の影響を確認できる。

## 解説

この回では，第7回で扱った標本化と信号の内積，第5回で扱った線形独立と行列とベクトルの積を使い，信号を複素正弦波の重ね合わせに分解する離散フーリエ変換を計算する。
ここで扱う DFT と窓関数は，第9回の短時間フーリエ変換や，第10回の畳み込みの計算の基礎になる。
スクリプトの先頭には，第7回と同様に次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figure_dir = Path("outputs/exc08")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. Fourier (フーリエ) 変換

信号を（複素）正弦波の重ね合わせの形に分解することを **フーリエ変換 (Fourier transform)** という。
Fourier 変換には，対象とする信号の種類（連続時間か離散時間か，有限長か無限長か）によっていくつかの種類がある。
ここでは，コンピュータを用いた数値計算に適した離散フーリエ変換 (DFT) を扱う。

#### ベクトルの分解

$N$ 個の $N$ 次元ベクトル $\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ が線形独立（第5回）であるとき，任意の $N$ 次元ベクトル $\boldsymbol{x}$ は，それらの線形結合として次式のとおりただ 1 通りに表せる。

$$
\boldsymbol{x} = a_1 \boldsymbol{e}_1 + \cdots + a_N \boldsymbol{e}_N
$$

ここで，$a_1, \cdots, a_N$ はスカラーである。
このような $\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ を，$N$ 次元ベクトル全体の **基底 (basis)** という。

言い換えると，ベクトル $\boldsymbol{x}$ は，より単純なベクトル $\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ の成分に分解できる。
このことは，複雑な対象（ベクトル）を解析するために役立つ。
例えば，平面上を運動する物体の位置 $\boldsymbol{x} = (x_1, x_2)^\top$ は，$\boldsymbol{e}_1 = (1, 0)^\top$ と $\boldsymbol{e}_2 = (0, 1)^\top$ を用いて，水平方向の運動と垂直方向の運動に分解して考えることができる。

$$
\boldsymbol{x} = x_1 \boldsymbol{e}_1 + x_2 \boldsymbol{e}_2
$$

基底 $\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ が互いに直交し，さらに自分自身との内積がすべて同じ正の値 $\alpha$ であるとき，すなわち

$$
\langle \boldsymbol{e}_i, \boldsymbol{e}_j \rangle =
\begin{cases}
\alpha & i = j \\
0 & i \neq j
\end{cases}
$$

を満たすとき，$\boldsymbol{e}_1$ と $\boldsymbol{x}$ の内積をとると，$\boldsymbol{e}_1$ の係数 $a_1$ が得られる。

$$
\langle \boldsymbol{e}_1, \boldsymbol{x} \rangle
= \langle \boldsymbol{e}_1, a_1 \boldsymbol{e}_1 + \cdots + a_N \boldsymbol{e}_N \rangle
= a_1 \langle \boldsymbol{e}_1, \boldsymbol{e}_1 \rangle + \cdots + a_N \langle \boldsymbol{e}_1, \boldsymbol{e}_N \rangle
= a_1 \alpha
$$

ここで，内積は第7回で定義した $\langle \boldsymbol{u}, \boldsymbol{v} \rangle = \boldsymbol{u}^* \boldsymbol{v}$ であり，2 つ目の引数について和とスカラー倍を外に出せること（線形性）を使った。
よって，$a_1 = \langle \boldsymbol{e}_1, \boldsymbol{x} \rangle / \alpha$ となる。
他の係数も同様に，$a_i = \langle \boldsymbol{e}_i, \boldsymbol{x} \rangle / \alpha$ で求められる。
特に，$\alpha = 1$ のとき，$a_1 = \langle \boldsymbol{e}_1, \boldsymbol{x} \rangle$ となる。
このように，直交する基底を使うと，連立一次方程式を解かなくても，内積だけで係数が求められる。

#### 複素正弦波信号

複素数値離散時間信号 $e^{j \omega n}$ を，角周波数 $\omega$ の離散時間 **複素正弦波信号 (complex sinusoid)** という。
ここで，$j$ は虚数単位（$j^2 = -1$）である。
**オイラーの公式 (Euler's formula)** $e^{j \theta} = \cos\theta + j \sin\theta$ を使うと，複素正弦波信号は次のように表せる。

$$
e^{j \omega n} = \cos(\omega n) + j \sin(\omega n)
$$

すなわち，複素正弦波信号は，実部が角周波数 $\omega$ の余弦波，虚部が角周波数 $\omega$ の正弦波である。
複素平面の上では，$e^{j\omega n}$ は原点を中心とする半径 1 の円の上の点であり，$n$ が 1 増えるごとに角度 $\omega$ だけ回転する。
$\omega > 0$ なら反時計回り，$\omega < 0$ なら時計回りに回転するので，複素正弦波信号では正と負の角周波数を区別できる。
周波数 $F$ [Hz] の連続時間信号をサンプリング周波数 $F_s$ で標本化した場合，第7回で見たように $\omega = 2\pi F/F_s$ である。
$f = \omega / 2\pi = F / F_s$ を **正規化周波数 (normalized frequency)** といい，1 サンプルあたりに進む周期の数を表す。

離散時間の複素正弦波信号は，連続時間の場合と異なり，常に周期信号になるとは限らない。
すべての $n$ について $e^{j \omega (n + N)} = e^{j \omega n}$ が成り立つには，$e^{j\omega N} = 1$，すなわち $\omega N$ が $2\pi$ の整数倍でなければならない。
したがって，複素正弦波信号は $\omega / 2\pi$ が有理数となる場合に限り周期信号となる。
$\omega / 2\pi = k / N$（$k$ と $N$ は互いに素な整数，$N > 0$）と既約分数で表せるとき，基本周期は $N$ である。
例えば，$\omega = 2\pi \cdot 3/10$ のとき，$2\pi / \omega = 10/3$ は整数ではないが，基本周期は $N = 10$ である。
$\omega = 1$ のように $\omega / 2\pi$ が無理数の場合は，同じ値を繰り返すことはない。

```python
omega = 2 * np.pi * 3 / 10
n = np.arange(30)
x = np.exp(1j * omega * n)
print(x.dtype, x.shape)              # complex128 (30,)
print(2 * np.pi / omega)             # 3.3333333333333335
print(np.allclose(x[10:], x[:-10]))  # True
print(np.allclose(x[5:], x[:-5]))    # False

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].stem(n, x.real)
axes[0].set_ylabel("Real part")
axes[1].stem(n, x.imag)
axes[1].set_ylabel("Imaginary part")
axes[1].set_xlabel("n")
fig.savefig(figure_dir / "session08_example_complex_sinusoid.png")
plt.show()
plt.close(fig)
```

Python では，虚数単位を `1j` と書く。
`np.exp` に複素数の配列を渡すと，各要素の複素指数関数を計算し，`complex128` 型（実部と虚部がそれぞれ `float64` の複素数）の配列を返す。
`x.real` と `x.imag` は，それぞれ実部と虚部を並べた実数の配列である。
`ax.stem` などの描画関数は複素数の配列をそのまま描けないため，実部と虚部を分けて描く。
`x[10:]` と `x[:-10]` は，同じ位置の要素どうしが $x[n + 10]$ と $x[n]$ の組になるように，ずらした 2 つの配列である。
したがって，`np.allclose(x[10:], x[:-10])` は，周期 10 を持つかどうかを判定している。
$\omega \cdot 5 = 3\pi$ なので，5 サンプル後には符号が反転し，5 は周期ではない。

長さが $N$ であり，かつ周期 $N$ を持つ複素正弦波信号は，$\omega N$ が $2\pi$ の整数倍となる $\omega = 2\pi k / N$（$k$ は整数）のものに限られる。
$k$ と $k + N$ は同じ信号を表すので，異なるものは次の $N$ 個である。

$$
\phi_k[n] = e^{j 2 \pi k n / N}, \quad k, n \in \{0, 1, \cdots, N-1\}
$$

$\phi_k$ の基本周期は $N$ とは限らない（例えば，$\phi_0$ は定数である）が，いずれも $N$ を周期として持つ。
このとき，$\phi_0, \cdots, \phi_{N-1}$ は互いに直交し，

$$
\langle \phi_k, \phi_l \rangle = \sum_{n=0}^{N-1} e^{j 2\pi (l - k) n / N} =
\begin{cases}
N & k = l \\
0 & k \neq l
\end{cases}
$$

となる。
$k \neq l$ のとき，$r = e^{j 2\pi (l - k) / N} \neq 1$ とおくと，等比数列の和の公式から和は $(1 - r^N)/(1 - r)$ であり，$r^N = e^{j 2\pi (l - k)} = 1$ なので 0 になる。

#### 離散フーリエ変換 (discrete Fourier transform, DFT)

複素正弦波信号 $\phi_0, \cdots, \phi_{N-1}$ は互いに直交するので線形独立であり，$N$ 次元ベクトル全体の基底になる。
これらを基底として，長さ $N$ の離散時間信号 $x[n]$ を次のように分解できる。

$$
x[n] = \frac{1}{N} \left( X[0] \phi_0[n] + \cdots + X[N-1] \phi_{N-1}[n] \right)
$$

ここで，$\phi_k$ の係数を $X[k]/N$ と書いた。
$\phi_0, \cdots, \phi_{N-1}$ が互いに直交し，$\langle \phi_k, \phi_k \rangle = N$ であることを利用すると，「ベクトルの分解」と同様に係数は $\langle \phi_k, x \rangle / N$ となるので，

$$
X[k] = \langle \phi_k, x \rangle
$$

と表される。
$\frac{1}{N}$ は，$\langle \phi_k, \phi_k \rangle = N$ であることを考慮した正規化係数である。

このとき，$x[n]$ から $X[k]$ への変換を **離散フーリエ変換 (discrete Fourier transform, DFT)** という。

$$
X[k] = \langle \phi_k, x \rangle = \boldsymbol{\phi}_k^* \boldsymbol{x} = \sum_{n=0}^{N-1} \overline{\phi_k[n]} x[n] = \sum_{n=0}^{N-1} x[n] e^{-j 2 \pi k n / N}
$$

$k$ を **周波数インデックス (frequency index)** という。
$X[k]$ は，$x[n]$ に正規化周波数 $k/N$ の複素正弦波 $\phi_k$ がどれだけ含まれるかを表す。
$X[k]$ を並べたベクトル $\boldsymbol{X} = (X[0], \cdots, X[N-1])^\top$ を考えると，DFT は，各行に $\boldsymbol{\phi}_k^*$ を並べた行列

$$
\boldsymbol{F}_N =
\begin{pmatrix}
\boldsymbol{\phi}_0^* \\
\boldsymbol{\phi}_1^* \\
\vdots \\
\boldsymbol{\phi}_{N-1}^*
\end{pmatrix},
\quad
[\boldsymbol{F}_N]_{k,n} = e^{-j 2 \pi k n / N}
$$

を使って，

$$
\boldsymbol{X} = \boldsymbol{F}_N \boldsymbol{x}
$$

と表される。
この行列 $\boldsymbol{F}_N$ を $N$ 点の **DFT 行列 (DFT matrix)** という。
$[\boldsymbol{F}_N]_{k,n}$ は $\boldsymbol{F}_N$ の $k$ 行 $n$ 列の要素を表し，ここでは行と列の番号を，NumPy の添字と同じく 0 から数える。

また，$X[k]$ から $x[n]$ への変換を **逆離散フーリエ変換 (inverse discrete Fourier transform, IDFT)** という。

$$
x[n] = \frac{1}{N} \left( X[0] \phi_0[n] + \cdots + X[N-1] \phi_{N-1}[n] \right) = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j 2 \pi k n / N}
$$

IDFT も，DFT と同様に行列を使って表すことができる。

$$
\boldsymbol{x} = \boldsymbol{F}_N^{-1} \boldsymbol{X} = \frac{1}{N} \boldsymbol{F}_N^* \boldsymbol{X}
$$

ただし，DFT 行列の逆行列は，DFT 行列の複素共役転置を $1/N$ 倍した行列に一致することを利用した。
実際，$\boldsymbol{F}_N \boldsymbol{F}_N^*$ の $k$ 行 $l$ 列の要素は $\langle \phi_k, \phi_l \rangle$ なので，直交性から $\boldsymbol{F}_N \boldsymbol{F}_N^* = N\boldsymbol{I}$ が成り立つ。
NumPy では，DFT を `np.fft.fft`，IDFT を `np.fft.ifft` で計算する（計算方法は第3節で説明する）。

```python
x = np.array([2.0, 1.0, 0.0, 1.0, -1.0, 0.5])
N = len(x)
n = np.arange(N)
X1 = np.sum(x * np.exp(-2j * np.pi * 1 * n / N))  # 定義式で k = 1 を計算
X = np.fft.fft(x)
print(X.shape, X.dtype)      # (6,) complex128
print(np.isclose(X1, X[1]))  # True
print(np.round(X, 3))
x_back = np.fft.ifft(X)
print(np.allclose(x_back, x))       # True
print(np.max(np.abs(x_back.imag)))  # 0.0
```

```text
[ 3.5 +0.j     2.25-1.299j  2.75+0.433j -1.5 +0.j     2.75-0.433j
  2.25+1.299j]
```

`np.fft.fft(x)` は，長さ $N$ の配列 `x` の DFT を，同じ長さの `complex128` 型の配列として返す。
`X[0]` は，$e^{0} = 1$ なので $\sum_n x[n] = 3.5$ であり，信号の平均値の $N$ 倍に等しい。
`np.isclose(a, b)` は，2 つの値が丸め誤差程度の差の範囲で等しいかを判定する。
`np.fft.fft` は DFT の定義と同じく $1/N$ を掛けず，`np.fft.ifft` は IDFT の定義と同じく $1/N$ を掛ける。
`np.fft.ifft` の結果は，入力が実数の信号でも `complex128` 型であり，一般には丸め誤差による小さな虚部を含む（この例では 0 になった）。
虚部が無視できることを確かめてから `x_back.real` で実部を取り出して使う。

#### 振幅スペクトルと位相スペクトル

DFT の結果 $X[k]$ は複素数となる。
このとき，$X[k]$ の大きさ $|X[k]| = \sqrt{\mathrm{Re}(X[k])^2 + \mathrm{Im}(X[k])^2}$ を **振幅スペクトル (amplitude spectrum)** といい，$X[k]$ の偏角 $\angle X[k] = \mathrm{atan2}(\mathrm{Im}(X[k]), \mathrm{Re}(X[k]))$ を **位相スペクトル (phase spectrum)** という。
$\mathrm{atan2}$ は，偏角を $-\pi$ から $\pi$ の範囲で返す関数である。
NumPy では，振幅スペクトルを `np.abs(X)`，位相スペクトルを `np.angle(X)` で計算する。

サンプリング周波数 $F_s$ で標本化した $N$ 点の信号では，$\phi_k$ の正規化周波数が $k/N$ なので，$X[k]$ は次の周波数の成分に対応する。

$$
F_k = \frac{k}{N} F_s \ [\mathrm{Hz}]
$$

隣り合う周波数インデックスに対応する周波数の間隔は，$F_s / N$ である。
$k > N/2$ の成分は，第2節の周期性 $X[k] = X[k - N]$ により，負の周波数 $(k - N)F_s/N$ の成分とみなすこともできる。
`np.fft.fftfreq(N, d=1 / Fs)` は，各 $k$ に対応する周波数を，$k < N/2$ では $kF_s/N$，$k \ge N/2$ では $(k-N)F_s/N$ として返す。
`np.fft.fftshift` は，配列の後半を前に移し，周波数の小さい順に並べ替える。

```python
Fs = 20  # サンプリング周波数 [Hz]
N = 20
n = np.arange(N)
x = 1.0 + 2 * np.exp(1j * (2 * np.pi * 3 * n / Fs + np.pi / 4))
X = np.fft.fft(x)
amp = np.abs(X)
phase = np.angle(X)
phase[amp < 1e-6 * amp.max()] = 0.0  # 振幅がほぼ 0 の成分の位相は 0 とする
freq = np.arange(N) * Fs / N         # F_k = k Fs / N
print(freq[:5])                      # [0. 1. 2. 3. 4.]
print(np.round(amp[[0, 3]], 6))      # [20. 40.]
print(round(phase[3], 4))            # 0.7854
print(np.fft.fftfreq(N, d=1 / Fs)[[0, 1, 9, 10, 11, 19]])  # [  0.   1.   9. -10.  -9.  -1.]

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].stem(freq, amp)
axes[0].set_ylabel("Amplitude")
axes[1].stem(freq, phase)
axes[1].set_ylabel("Phase [rad]")
axes[1].set_xlabel("Frequency [Hz]")
fig.savefig(figure_dir / "session08_example_spectrum.png")
plt.show()
plt.close(fig)
```

この例の信号は，値 1 の定数（周波数 0 の成分）と，振幅 2，周波数 3 Hz，位相 $\pi/4$ の複素正弦波の和である。
周波数 3 Hz の複素正弦波は $\phi_3$ を $2e^{j\pi/4}$ 倍したものなので，直交性により $X[3] = 2e^{j\pi/4}\langle \phi_3, \phi_3 \rangle = 40e^{j\pi/4}$ となり，振幅は 40，位相は $\pi/4 \approx 0.7854$ になる。
同様に $X[0] = 20$ であり，それ以外の $X[k]$ は 0 になる。
振幅が 0 に近い成分では，丸め誤差だけで偏角が大きく変わるため，`np.angle` が返す値に意味はない。
このため，例では振幅が最大値の $10^{-6}$ 倍未満の成分の位相を 0 に置き換えている。
`phase[amp < 1e-6 * amp.max()] = 0.0` は，比較で得た真偽値の配列を添字に使った代入（第3回）である。
`np.fft.fftfreq` の結果では，$k = 10$ 以降が負の周波数として表されている。

### 2. DFT の性質

信号 $x[n], x_1[n], x_2[n]$ の $N$ 点 DFT をそれぞれ $X[k], X_1[k], X_2[k]$ とすると，DFT は次の性質を満たす。
以下では，$x[n]$ と $X[k]$ を，添字を $N$ で割った余りで読む周期的な列とみなす。
例えば，$X[-k]$ は $X[N-k]$ を，$x[n - n_0]$ は $x[(n - n_0) \bmod N]$ を表す。

#### 線形性

$$
a_1 x_1[n] + a_2 x_2[n] \overset{\mathrm{DFT}}{\longleftrightarrow} a_1 X_1[k] + a_2 X_2[k]
$$

DFT は行列とベクトルの積 $\boldsymbol{F}_N\boldsymbol{x}$ で表されるので，信号の線形結合の DFT は，それぞれの DFT の線形結合になる。

#### 周期性

$$
X[k + N] = X[k]
$$

$e^{-j 2\pi (k + N) n / N} = e^{-j 2\pi k n / N}$ なので，$k = 0, \ldots, N-1$ 以外の $k$ で DFT の式を計算しても，同じ値の繰り返しになる。
IDFT の式からも同様に $x[n + N] = x[n]$ が得られるので，DFT は長さ $N$ の信号を，それを周期 $N$ で繰り返した信号の 1 周期分として扱っている。

#### 共役性

$$
\overline{x[n]} \overset{\mathrm{DFT}}{\longleftrightarrow} \overline{X[-k]}
$$

信号の複素共役をとると，その DFT は，周波数インデックスの符号を反転して複素共役をとったものになる。

#### 対称性

$x[n]$ が実数値信号であるとき，$\overline{x[n]} = x[n]$ と共役性から，$X[k]$ は共役対称となる。
すなわち，

$$
X[k] = \overline{X[-k]}
$$

が成り立つ。
言い換えると，振幅スペクトルは偶対称となり，位相スペクトルは奇対称となる。

$$
|X[k]| = |X[-k]|, \quad \angle X[k] = -\angle X[-k]
$$

ただし，$X[k]$ が負の実数のとき，`np.angle` は $\pi$ を返すので，位相の等式は $2\pi$ の整数倍の違いを除いて成り立つ。
実数値信号の振幅スペクトルは $k$ と $N - k$ で等しいので，$k = 0, 1, \ldots, \lfloor N/2 \rfloor$ の範囲を見れば十分である。
第9回で使う `np.fft.rfft` は，この範囲の値だけを返す。

#### 時間シフト

$$
x[n - n_0] \overset{\mathrm{DFT}}{\longleftrightarrow} e^{-j 2 \pi k n_0 / N} X[k]
$$

ここでの $x[n - n_0]$ は，添字を $N$ で割った余りで読む巡回シフト（第7回）であり，`np.roll(x, n0)` にあたる。
$|e^{-j 2 \pi k n_0 / N}| = 1$ なので，時間シフトは振幅スペクトルを変えず，位相スペクトルに $k$ に比例する $-2\pi k n_0 / N$ を加える。
先頭を 0 で埋める遅延（第7回）は巡回シフトと異なるため，この式は一般には成り立たない。

#### 周波数シフト（変調）

$$
e^{j 2 \pi k_0 n / N} x[n] \overset{\mathrm{DFT}}{\longleftrightarrow} X[k - k_0]
$$

信号に複素正弦波 $\phi_{k_0}$ を掛けると，スペクトルは周波数インデックスの方向に $k_0$ だけ巡回シフトする。
信号に正弦波を掛けてスペクトルを移動させる操作を **変調 (modulation)** という。

#### 性質の数値的な確認

```python
rng = np.random.default_rng(0)
x = rng.standard_normal(8)  # 実数値の信号
N = len(x)
k = np.arange(N)
X = np.fft.fft(x)
X_minus = X[(-k) % N]       # X[-k] = X[N - k]
print((-k) % N)                                    # [0 7 6 5 4 3 2 1]
print(np.allclose(X, np.conj(X_minus)))            # True
n0 = 3
Y = np.fft.fft(np.roll(x, n0))                     # x[n - 3]（巡回シフト）の DFT
print(np.allclose(Y, np.exp(-2j * np.pi * k * n0 / N) * X))  # True
print(np.allclose(np.abs(Y), np.abs(X)))           # True
```

`(-k) % N` は，$-k$ を $N$ で割った余りであり，$X[-k]$ の添字を $0, \ldots, N-1$ の範囲で表したものである。
`X[::-1]` は $X[N-1-k]$ を並べたものであり，$X[-k]$ とは 1 つずれることに注意する。
1 つ目の `True` は実数値信号の対称性を，2 つ目と 3 つ目の `True` は時間シフトの性質と，時間シフトで振幅スペクトルが変わらないことを確かめている。
`np.exp(-2j * np.pi * k * n0 / N) * X` は，配列 `k` の各要素に対する係数と `X` の要素ごとの積である。

### 3. 高速フーリエ変換 (fast Fourier transform, FFT)

DFT を定義に従って直接計算すると，$N$ 個の $X[k]$ のそれぞれに $N$ 回の積と和が必要なので，$N$ 点の信号に対して $O(N^2)$ の計算量が必要となる。
しかし，計算を工夫することで，$O(N \log N)$ の計算量で同じ結果を得ることができる。
このように DFT を高速に計算するアルゴリズムの総称を **高速フーリエ変換 (fast Fourier transform, FFT)** という。
FFT は DFT を計算するためのアルゴリズムであるため，FFT を使って得られる結果は，丸め誤差を除いて DFT と同じである。
`np.fft.fft` と `np.fft.ifft` は，FFT を使って DFT と IDFT を計算する。
FFT は $N$ が 2 のべき乗のときに特に効率よく計算できるが，`np.fft.fft` は任意の $N$ で計算できる。

```python
import time

N = 2**16
x = np.random.default_rng(1).standard_normal(N)
start = time.perf_counter()
X = np.fft.fft(x)
elapsed = time.perf_counter() - start
print(X.shape)                    # (65536,)
print(N**2 / (N * np.log2(N)))    # 4096.0
print(f"{elapsed:.6f} s")         # 実行環境によって異なる
```

`time.perf_counter()` は，時間を測るための時計の値を秒単位で返す，Python の標準ライブラリ `time` の関数である。
処理の前後で呼び出した値の差が，その処理にかかった時間になる。
計算時間は PC や実行のたびに変わるため，比較するときは同じ処理を何回か測り，最小値などをまとめて使う。
$N = 2^{16}$ のとき，$N^2$ は $N\log_2 N$ の 4096 倍であり，$N$ が大きいほど FFT による計算量の削減の効果が大きくなる。

### 4. 有限長の信号と窓関数

#### 周波数分解能とゼロ埋め

第1節で見たように，$N$ 点 DFT の周波数の間隔は $F_s/N$ であり，これは観測時間 $NT_s$ の逆数 $1/(NT_s)$ に等しい。
周波数がこの間隔より近い 2 つの成分は，DFT では区別しにくい。
この意味で，周波数の間隔 $F_s/N$ を DFT の **周波数分解能 (frequency resolution)** という。
周波数分解能を細かくするには，観測時間を長くして $N$ を増やす必要がある。

`np.fft.fft(x, n=M)` のように $M > N$ を指定すると，信号の末尾に $M - N$ 個の 0 を加えてから $M$ 点 DFT を計算する。
これを **ゼロ埋め (zero padding)** という。
ゼロ埋めをすると周波数の間隔が $F_s/M$ に細かくなり，スペクトルの形を細かく描けるが，観測時間は変わらないので，近い 2 つの成分を区別する能力は向上しない。
逆に $M < N$ を指定すると，信号の先頭の $M$ 点だけを使う。

#### スペクトル漏れ

周波数 $F$ が $F_s/N$ の整数倍のとき，標本化した正弦波は $N$ 点の中にちょうど整数個の周期を含み，その DFT は対応する $k$ だけに値を持つ。
$F$ が $F_s/N$ の整数倍でないとき，$N$ 点の中に整数個の周期が収まらない。
第2節の周期性のとおり，DFT は $N$ 点を 1 周期として繰り返した信号を扱うので，繰り返しのつなぎ目で波形が不連続になり，その成分が多くの $k$ に広がって現れる。
この現象を **スペクトル漏れ (spectral leakage)** またはリーケージという。
無限に続く信号から $N$ 点を切り出すことは，$0 \le n < N$ で 1，それ以外で 0 となる **矩形窓 (rectangular window)** を掛けることに相当する。

#### 窓関数

切り出した信号に，両端で滑らかに 0 に近づく関数 $w[n]$ を掛けてから DFT を計算すると，つなぎ目の不連続が小さくなり，スペクトル漏れを抑えられる。
このように信号に掛ける関数を **窓関数 (window function)** という。
代表的な窓関数に，次の **ハン窓 (Hann window)** がある。

$$
w[n] = 0.5\left(1 - \cos\frac{2\pi n}{N - 1}\right), \quad 0 \le n < N
$$

NumPy では，ハン窓を `np.hanning(N)` で計算できる。
振幅スペクトルのピークを中心とする山を **メインローブ (main lobe)**，その周りに並ぶ小さな山を **サイドローブ (side lobe)** という。
ハン窓を掛けると，サイドローブが小さくなり，ピークから離れた周波数への漏れは大きく減る。
その代わりに，メインローブの幅（ピークの両側で最初に 0 になる周波数の間隔）は，矩形窓の約 $2F_s/N$ から約 $4F_s/N$ に広がる。
また，ハン窓の値の和 $\sum_n w[n]$ は約 $N/2$ なので，ピークの高さも約半分になる。
窓関数が異なる振幅スペクトルを比べるときは，$|X[k]|$ を $\sum_n w[n]$ で割って正規化する。
矩形窓では $\sum_n w[n] = N$ なので，正規化した振幅は $|X[k]|/N$ である。
第9回では，短時間フーリエ変換で信号を短い区間に区切るときに窓関数を使う。

```python
Fs, N = 64, 64                            # 周波数の間隔は Fs / N = 1 Hz
n = np.arange(N)
freq = n * Fs / N
x_on = np.cos(2 * np.pi * 8 * n / Fs)     # 8 Hz：1 Hz の整数倍
x_off = np.cos(2 * np.pi * 8.5 * n / Fs)  # 8.5 Hz：1 Hz の整数倍でない
w = np.hanning(N)

amp_on = np.abs(np.fft.fft(x_on)) / N
amp_off = np.abs(np.fft.fft(x_off)) / N
amp_hann = np.abs(np.fft.fft(x_off * w)) / np.sum(w)
print(np.round(amp_on[[8, 9, 20]], 4))    # [0.5 0.  0. ]
print(np.round(amp_off[[8, 9, 20]], 4))   # [0.3112 0.3254 0.0208]
print(np.round(amp_hann[[8, 9, 20]], 4))  # [4.266e-01 4.266e-01 1.000e-04]
half = N // 2 + 1                         # k = 0, ..., N/2
print(np.sum(amp_on[:half] > 0.01), np.sum(amp_off[:half] > 0.01), np.sum(amp_hann[:half] > 0.01))  # 1 33 6

fig, ax = plt.subplots()
ax.semilogy(freq[:half], amp_off[:half], "o-", label="8.5 Hz, rectangular")
ax.semilogy(freq[:half], amp_hann[:half], "s-", label="8.5 Hz, Hann")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Normalized amplitude")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session08_example_window.png")
plt.show()
plt.close(fig)
```

8 Hz の余弦波は 64 点の中にちょうど 8 周期を含むので，$k = 0, \ldots, 32$ の範囲では $k = 8$ だけが値 0.5 を持つ。
実数の余弦波は正と負の周波数の 2 つの複素正弦波に分かれるため，正規化した振幅は余弦波の振幅 1 の半分になる。
8.5 Hz の余弦波では，最も近い $k = 8, 9$ の値が 0.5 より小さくなり，$k = 20$（20 Hz）のように離れた周波数にも値が漏れている。
正規化した振幅が 0.01 を超える成分の数は，矩形窓では 33 個すべてだが，ハン窓では 6 個に減っている。
片対数グラフでは，ピークから離れるにつれて，ハン窓の振幅が矩形窓の振幅より速く小さくなっている。
この例では周波数の間隔が 1 Hz と粗いため，メインローブの幅の違いは図から読み取りにくい。
メインローブの形を細かく調べるには，ゼロ埋めをして周波数の間隔を細かくした DFT を使う。
`np.sum(amp_on[:half] > 0.01)` は，真偽値の配列の和が `True` の個数になることを使って，条件を満たす要素を数えている。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第8回の第1問であれば，`exercise/exc08_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc08`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第1問の図であれば，`outputs/exc08/exc08_01.png` とします。
信号を「$D$ 秒間標本化する」問題では，第7回と同様に，`N = round(D * Fs)`，`n = np.arange(N)`，`t = n / Fs` として，区間 $0 \le t < D$ の $N$ 点を使ってください。
振幅スペクトルと位相スペクトルは `ax.stem` で描いてください。
位相スペクトルを表示したり描いたりする問題では，解説の例と同様に，振幅が最大値の $10^{-6}$ 倍未満の成分の位相を 0 に置き換えてください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえで NumPy で計算した結果と比較してください。

### 基礎レベル（10問）
1. 正規化周波数と正規化角周波数を求め，標本化した正弦波の周期を確かめます。周期 $T = 1/4$ 秒の正弦波 $x(t) = \sin(2\pi t / T)$ を，サンプリング周波数 $F_s = 10$ Hz で標本化した信号を $x[n]$ とします。

    1. $x(t)$ の周波数 $F$ を求め，$x[n]$ の正規化周波数 $f$ と正規化角周波数 $\omega$ を手計算してください。
    2. $f$ を既約分数で表し，$x[n]$ の基本周期を手計算してください。また，その値を $2\pi/\omega$ と比較してください。
    3. `exercise/exc08_01.py` で，$n = 0, 1, \ldots, 19$ の $x[n]$ を作り，$f$，$\omega$，$2\pi/\omega$ を表示して 1 と 2 の結果と比較してください。また，2 で求めた基本周期を `p` として `np.allclose(x[p:], x[:-p])` が `True` になること，`p` より小さい正の整数ではすべて `False` になることを確かめてください。
    4. $x[n]$ を `ax.stem` で描き，`outputs/exc08/exc08_01.png` に保存してください。基本周期ごとに同じ値が繰り返すことを図で確認し，基本周期のサンプル数の間に $x(t)$ の周期がいくつ含まれるかを答えて，基本周期が $2\pi/\omega$ と一致しない理由を説明してください。

2. 複素正弦波信号を 3 次元の図で表し，周波数の符号による違いを確かめます。$x(t) = e^{j 4\pi t}$ と $y(t) = e^{-j 4\pi t}$ を，$F_s = 100$ Hz でそれぞれ 1 秒間標本化した信号を $x[n]$，$y[n]$ とします。

    1. $x(t)$ と $y(t)$ の周波数 [Hz] を答えてください。また，$t$ が増えるとき，複素平面の上で $x(t)$ と $y(t)$ がそれぞれどちら向きに回転するかを予想してください。
    2. `exercise/exc08_02.py` で $x[n]$ と $y[n]$ を作り，1 つの軸を時刻，残りの 2 つの軸を実部と虚部とした 3 次元の図に重ねて描いて，`outputs/exc08/exc08_02.png` に保存してください。3 次元の図は，`fig = plt.figure()` と `ax = fig.add_subplot(projection="3d")` で作った `ax` に対して，`ax.plot(t, x.real, x.imag)` のように描けます。軸ラベルは `ax.set_xlabel`，`ax.set_ylabel`，`ax.set_zlabel` で付けてください。
    3. $x[n]$ と $y[n]$ の実部どうしと虚部どうしがどのような関係になるかを予想し，`np.allclose` で確かめてください。また，$y[n] = \overline{x[n]}$ が成り立つことを確かめてください。
    4. 図と 3 の結果から，周波数が正のときと負のときで時間波形がどのように変わるかを説明してください。また，実部だけを観測した場合に 2 つの信号を区別できるかを，その理由とともに答えてください。

3. 周期 $N$ の複素正弦波信号が，互いに直交することを確かめます。$N = 8$ とし，$\phi_k[n] = e^{j 2 \pi k n / N}$（$k = 0, 1, \ldots, 7$，$n = 0, 1, \ldots, 7$）を考えます。

    1. $\langle \phi_1, \phi_1 \rangle$ と $\langle \phi_1, \phi_3 \rangle$ を手計算してください。$\langle \phi_1, \phi_3 \rangle$ は，和の各項 $e^{j\pi n/2}$（$n = 0, \ldots, 7$）が複素平面の上でどのような点になるかを考えて求めてください。
    2. `exercise/exc08_03.py` で，$\langle \phi_k, \phi_l \rangle$ を $k$ 行 $l$ 列の要素とする $8 \times 8$ の行列 `G` を，`np.vdot` を使って計算してください。`G` の形を表示し，`np.allclose(G, 8 * np.eye(8))` が `True` になることを確かめてください。
    3. `im = ax.imshow(np.abs(G))` で `G` の各要素の大きさを画像として描き，`fig.colorbar(im, ax=ax)` でカラーバーを付けて，`outputs/exc08/exc08_03.png` に保存してください。
    4. 2 の結果から，$\phi_0, \ldots, \phi_7$ が互いに直交することを説明してください。また，$n = 0, \ldots, 7$ の範囲で作った $\phi_9[n] = e^{j 2 \pi \cdot 9 n / 8}$ について，$\langle \phi_1, \phi_9 \rangle$ がいくつになるかを予想してから計算し，その値になる理由を説明してください。

4. $N = 4$ の DFT 行列を手で求め，DFT を手計算します。$\boldsymbol{x} = (1, 1, 0, 0)^\top$ とします。

    1. DFT 行列 $\boldsymbol{F}_4$ の 16 個の要素を，$e^{-j\pi/2} = -j$ などを使って手計算してください。
    2. $\boldsymbol{F}_4$ を使って，$\boldsymbol{X} = \boldsymbol{F}_4 \boldsymbol{x}$ を手計算してください。
    3. `exercise/exc08_04.py` で，1 の手計算の結果を `np.array` で入力した `F4` を作り，`F4 @ x` と `np.fft.fft(x)` が一致することを `np.allclose` で確かめてください。
    4. 2 の結果から，振幅スペクトル $|X[k]|$ と位相スペクトル $\angle X[k]$ を手計算し，`np.abs` と `np.angle` の結果と比較してください。$X[k] = 0$ となる成分の位相は，演習の冒頭の約束と同様に 0 とします。また，$X[1]$ と $X[3]$ が互いに複素共役になる理由を，DFT の対称性から説明してください。

5. $N = 4$ の逆 DFT 行列を手で求め，逆 DFT を手計算します。$\boldsymbol{X} = (1, 0, 0, 0)^\top$ とします。

    1. $\boldsymbol{F}_4^{-1} = \frac{1}{4}\boldsymbol{F}_4^*$ を，基礎第4問の $\boldsymbol{F}_4$ から手計算してください。
    2. $\boldsymbol{F}_4^{-1}$ を使って，$\boldsymbol{x} = \boldsymbol{F}_4^{-1}\boldsymbol{X}$ を手計算してください。
    3. `exercise/exc08_05.py` で `F4` と `F4_inv` を作り，`F4_inv @ X` と `np.fft.ifft(X)` が一致すること，`F4_inv @ F4` が単位行列になることを，`np.allclose` で確かめてください。
    4. $\boldsymbol{X} = (1, 0, 0, 0)^\top$ が $k = 0$ の成分だけを持つスペクトルであることを使って，2 の結果の 4 つの値がすべて等しくなる理由を説明してください。また，$\boldsymbol{X} = (0, 1, 0, 0)^\top$ の逆 DFT を予想し，`np.fft.ifft` で確かめてください。

6. 複素正弦波信号の DFT を計算し，振幅スペクトルの横軸を周波数に変換します。$x(t) = e^{j 4 \pi t}$ を $F_s = 16$ Hz で 1 秒間標本化した 16 点の信号を $x[n]$ とします。

    1. $x(t)$ の周波数と，$x[n]$ の正規化周波数を答えてください。$x[n]$ が $N = 16$ の $\phi_k$ のどれと一致するかを考え，16 点 DFT の振幅スペクトル $|X[k]|$ がどの $k$ で値を持ち，その値がいくつになるかを予想してください。
    2. `exercise/exc08_06.py` で `np.fft.fft` を使って $X[k]$ を計算し，`np.round(np.abs(X), 6)` を表示して 1 の予想と比較してください。
    3. `plt.subplots(2, 1)` を使い，上段に横軸を周波数インデックス $k$，下段に横軸を周波数 $F_k = kF_s/N$ [Hz] とした振幅スペクトルを描いて，`outputs/exc08/exc08_06.png` に保存してください。
    4. $|X[k]|$ が 1 つの $k$ だけで値を持つ理由を，$\phi_0, \ldots, \phi_{15}$ の直交性を使って説明してください。

7. 余弦波の DFT を計算し，スペクトルが 2 本立つ理由を考えます。$x(t) = \cos(4 \pi t)$ を $F_s = 16$ Hz で 1 秒間標本化した 16 点の信号を $x[n]$ とします。

    1. オイラーの公式を使って，$\cos(4\pi t)$ を 2 つの複素正弦波の和で表してください。この式と基礎第6問の結果から，16 点 DFT の振幅スペクトルがどの $k$ で値を持ち，その値がいくつになるかを予想してください。
    2. `exercise/exc08_07.py` で 16 点 DFT を計算し，振幅スペクトルと位相スペクトルを表示して，1 の予想と比較してください。
    3. `plt.subplots(2, 1, sharex=True)` を使い，横軸を周波数 $F_k = kF_s/N$ [Hz] として，上段に振幅スペクトル，下段に位相スペクトルを描いて，`outputs/exc08/exc08_07.png` に保存してください。
    4. スペクトルが 2 本立つ理由を，1 の式を使って説明してください。また，14 Hz の位置に現れる成分が $-2$ Hz の成分を表している理由を，DFT の周期性 $X[k] = X[k - N]$ を使って説明し，`np.fft.fftfreq(16, d=1 / 16)` で $k = 14$ に対応する周波数を表示して確かめてください。

8. 複素正弦波・余弦波・正弦波のスペクトルを比べ，オイラーの公式との関係を確かめます。$x_e(t) = e^{j 6 \pi t}$，$x_{\cos}(t) = \cos(6 \pi t)$，$x_{\sin}(t) = \sin(6 \pi t)$ を，$F_s = 8$ Hz でそれぞれ 1 秒間標本化した 8 点の信号を $x_e[n]$，$x_{\cos}[n]$，$x_{\sin}[n]$ とします。

    1. $x_{\cos}$ と $x_{\sin}$ を $e^{j6\pi t}$ と $e^{-j6\pi t}$ で表してください。この式から，3 つの信号の 8 点 DFT が値を持つ周波数 [Hz] と，その振幅と位相を予想してください。
    2. `exercise/exc08_08.py` で 3 つの信号の 8 点 DFT を計算してください。`plt.subplots(2, 3, figsize=(12, 6))` を使い，上段に振幅スペクトル，下段に位相スペクトルを，信号ごとに 1 列ずつ描いて，`outputs/exc08/exc08_08.png` に保存してください。横軸は `np.fft.fftshift(np.fft.fftfreq(8, d=1 / 8))`（$-4$ Hz から 3 Hz）とし，スペクトルも `np.fft.fftshift` で同じ順に並べ替えてください。
    3. 振幅と位相の値を表示して 1 の予想と比較してください。また，DFT の線形性から成り立つ関係 $X_e[k] = X_{\cos}[k] + jX_{\sin}[k]$ を，`np.allclose` で確かめてください。
    4. 3 つの信号の振幅スペクトルと位相スペクトルにどのような関係があるかを説明してください。特に，$x_{\sin}$ の位相が $\pm\pi/2$ になる理由を，$\frac{1}{2j} = -\frac{j}{2} = \frac{1}{2}e^{-j\pi/2}$ を使って説明してください。

9. DFT の点数によって，振幅スペクトルの見え方がどう変わるかを調べます。$x(t) = \cos(100 \pi t)$ を $F_s = 1000$ Hz で標本化した信号の先頭 $N$ 点（$n = 0, 1, \ldots, N-1$）について，$N = 1024, 256, 64, 16$ のそれぞれで $N$ 点 DFT を計算します。

    1. 各 $N$ について，周波数の間隔 $F_s/N$ と，50 Hz に最も近い $F_k$ を手計算してください。また，$N$ を小さくすると振幅スペクトルのピークの位置と広がりがどう変わるかを予想してください。
    2. `exercise/exc08_09.py` で各 $N$ の振幅スペクトルを計算し，$0 \le k \le N/2$ の範囲で $|X[k]|$ が最大となる $F_k$ を表示して，1 の手計算と比較してください。
    3. `plt.subplots(4, 1, figsize=(8, 10))` を使い，各 $N$ の振幅スペクトルを，横軸を周波数 $F_k$ として $k = 0, 1, \ldots, N/2$（$0 \le F_k \le F_s/2$）の範囲で描いて，`outputs/exc08/exc08_09.png` に保存してください。$N$ が異なっても比べられるように，振幅は $|X[k]|/N$ としてください。
    4. どの $N$ でも，50 Hz から離れた $F_k$ にまで値が広がっている理由を，$N$ 点の中に含まれる $x(t)$ の周期の数から説明してください。また，$0 \le k \le N/2$ の範囲でピークが 1 本だけになる $N$ を 1 つ挙げ（2 のべき乗でなくてもかまいません），その $N$ で計算して確かめてください。

10. DFT 行列を使って DFT を計算する関数を実装し，`np.fft.fft` の結果と比べます。

    1. `exercise/exc08_10.py` で，1 次元配列 `x` を受け取り，その長さ $N$ の DFT 行列 $\boldsymbol{F}_N$ を作って $\boldsymbol{F}_N\boldsymbol{x}$ を返す関数 `my_dft(x)` を定義してください。$[\boldsymbol{F}_N]_{k,n} = e^{-j2\pi kn/N}$ の指数に現れる $kn$ を並べた $N \times N$ の行列は，`np.outer(k, n)`（`k` と `n` はともに `np.arange(N)`）で作れます。`np.outer(a, b)` は，`a[i] * b[j]` を $i$ 行 $j$ 列の要素とする行列を返します。
    2. 長さ 8 の `x` について，`my_dft` の中で作る DFT 行列の形と，`my_dft(x)` の返り値の形と `dtype` を予想してから，表示して確かめてください。
    3. `rng = np.random.default_rng(0)` で作った乱数生成器から `x = rng.standard_normal(8)` を作り，`my_dft(x)` と `np.fft.fft(x)` の差の絶対値の最大値を表示してください。また，基礎第4問の $\boldsymbol{x} = (1, 1, 0, 0)^\top$ について，`my_dft` の結果が手計算の結果と一致することを `np.allclose` で確かめてください。
    4. 3 の差の絶対値の最大値が 0 ではなく $10^{-15}$ 程度の値になる理由を説明してください。

### 発展レベル（3問）
1. DFT 行列による計算と FFT の計算時間を比べ，点数 $N$ に対する増え方を調べます。基礎第10問と同じ `my_dft` を使い，$N = 256, 512, 1024, 2048, 4096$ とします。信号は `rng = np.random.default_rng(0)` で作った乱数生成器から `rng.standard_normal(N)` で作ってください。計算時間は解説の第3節と同様に `time.perf_counter()` の差で測り，各条件で 3 回測った最小値をその条件の計算時間とします。

    1. `complex128` 型の要素 1 つは 16 バイトです。$N = 4096$ と $N = 8192$ の DFT 行列が使うメモリ [バイト] を手計算してください。この結果から，自分の PC で `my_dft` を $N = 8192$ で実行してよいかを判断してください。
    2. $N$ を 2 倍にしたとき，`my_dft` と `np.fft.fft` の計算時間がそれぞれ約何倍になるかを，計算量 $O(N^2)$ と $O(N \log N)$ から予想してください。
    3. `exercise/exc08_11.py` で，各 $N$ について 2 つの方法の計算時間を測り，$N$，2 つの計算時間，計算時間の比（`my_dft` ÷ `np.fft.fft`）を 1 行ずつ表示してください。2 つの方法の結果が一致することも，`np.allclose` で確かめてください。
    4. 横軸を $N$，縦軸を計算時間とした両対数グラフ（`ax.loglog`）に，2 つの方法の結果を重ねて描き，`outputs/exc08/exc08_11.png` に保存してください。
    5. 隣り合う $N$ での計算時間の比を求め，2 の予想と比較してください。`np.fft.fft` の比が 2 から外れる理由を，計算時間が非常に短いときに計算以外の処理（関数の呼び出しや配列の確保など）が占める割合から考察してください。
    6. $N = 4096$ のとき，`my_dft` の中の DFT 行列を作る処理と，行列とベクトルの積の計算時間を別々に測り，どちらに時間がかかっているかを答えてください。その結果を，それぞれの処理で計算する値の個数から説明してください。

2. 窓関数によるスペクトルの違いを，メインローブの幅とスペクトル漏れの 2 つの観点で比べ，目的に応じて窓関数を選びます。$F_s = 1000$ Hz，$N = 256$ 点とし，矩形窓 `np.ones(N)` とハン窓 `np.hanning(N)` を比べます。スペクトルの形を細かく見るため，DFT は `np.fft.fft(x * w, n=4096)` で 4096 点にゼロ埋めして計算し，振幅は $|X[k]| / \sum_n w[n]$ で正規化してください（解説の第4節）。周波数は $F_k = kF_s/4096$ とします。

    1. 2 つの窓関数のうち，メインローブが広いのはどちらか，ピークから離れた周波数への漏れが大きいのはどちらかを予想してください。
    2. `exercise/exc08_12.py` で，次の 3 つの信号を 256 点ずつ作り，それぞれ 2 つの窓関数で正規化した振幅スペクトルを計算してください。

        - (a) $x_a(t) = \sin(2\pi \cdot 100 t)$
        - (b) $x_b(t) = \sin(2\pi \cdot 100 t) + \sin(2\pi \cdot 106 t)$
        - (c) $x_c(t) = \sin(2\pi \cdot 100 t) + 0.01\sin(2\pi \cdot 130 t)$

    3. (a) について，$0 \le F_k < F_s/2$ の範囲で振幅がピークの値の半分以上になる $F_k$ の最大値と最小値の差（半値幅）を，2 つの窓関数のそれぞれについて表示してください。その値を，周波数の間隔 $F_s/N$ と比較してください。
    4. `plt.subplots(3, 1, figsize=(8, 10))` を使い，(a)〜(c) の結果を 1 段ずつ，横軸を周波数 [Hz]（50〜200 Hz の範囲），縦軸を正規化した振幅とした片対数グラフ（`ax.semilogy`）に，2 つの窓関数を重ねて描いて，`outputs/exc08/exc08_12.png` に保存してください。
    5. (b) で，100〜106 Hz の範囲の振幅の最小値を 95〜111 Hz の範囲の振幅の最大値で割った値が 0.5 未満であれば，「2 つの成分を分離できた」とします。(c) で，125〜135 Hz の範囲の振幅の最大値が，(a) の同じ範囲の振幅の最大値の 2 倍以上であれば，「130 Hz の成分を検出できた」とします。それぞれの窓関数について，判定に使った値と判定結果を表示してください。
    6. 3〜5 の結果から，(b) のように振幅の近い 2 つの成分を分離したい場合と，(c) のように大きな成分の近くにある小さな成分を見つけたい場合のそれぞれで，どちらの窓関数を使うかを選んでください。選んだ理由を，メインローブの幅とスペクトル漏れの大きさを根拠に説明してください。

3. DFT の時間シフトの性質を使って，2 つの信号のずれ（シフト量）を推定します。この方法は位相相関と呼ばれ，第13回の[画像の位置合わせ](13-sp-projects/04_image_registration.md)のプロジェクトでは 2 次元に拡張して使います。`rng = np.random.default_rng(0)` で作った乱数生成器から `x = rng.standard_normal(64)` を作り，それを巡回シフトした `y = np.roll(x, 5)`（$y[n] = x[n - 5]$）を使います。$N = 64$ とします。

    1. 時間シフトの性質から，$R[k] = Y[k]\overline{X[k]} / |Y[k]\overline{X[k]}|$ が $e^{-j2\pi k \cdot 5/N}$ になることを示してください。さらに，$R[k]$ の逆 DFT $r[n]$ が，$n = 5$ で 1，それ以外の $n$ で 0 になることを，第1節の直交性を使って示してください。
    2. `exercise/exc08_13.py` で，$X[k]$，$Y[k]$，$R[k]$，$r[n]$ を計算してください。$|X[k]|$ の最小値を表示して 0 で割るおそれがないことを確かめてから，$r[n]$ の実部が最大となる $n$ とその値，$r[n]$ の虚部の絶対値の最大値を表示してください。
    3. `y` を `np.roll(x, -3)` に変えて同じ計算を行い，$r[n]$ の実部が最大となる $n$ を表示してください。その値から負のシフト量 $-3$ をどのように読み取ればよいかを，DFT の周期性を使って説明してください。
    4. 巡回シフトの代わりに，第7回と同様に先頭の 5 サンプルを 0 で埋めて遅延させた `y_zero` を使って同じ計算を行い，$r[n]$ の実部の最大値とその位置を表示してください。
    5. 2 と 4 の $r[n]$ の実部を，`plt.subplots(2, 1, sharex=True)` の上段と下段に `ax.stem` で描き，`outputs/exc08/exc08_13.png` に保存してください。4 で最大値の位置は 5 のままなのに，最大値が 1 より小さくなる理由を，時間シフトの性質が成り立つ条件から説明してください。

## 詰まったときに見る資料
- [`../textbook/markdown/ch14-basics-of-spectrum-analysis.md`](../textbook/markdown/ch14-basics-of-spectrum-analysis.md)
- [`07-signals-and-sampling.md`](07-signals-and-sampling.md)
- [`05-linear-algebra.md`](05-linear-algebra.md)
- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)
