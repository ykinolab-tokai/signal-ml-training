# 第08回 Fourier 変換

- DFT を使って，時間信号を周波数成分として確認する。
- 振幅スペクトルと位相スペクトルを描く。
- 窓関数がスペクトルに与える影響を確認する。
- 逆変換で時間信号に戻せることを確認する。

## 解説

### 1. Fourier (フーリエ) 変換

信号を、（複素）正弦波の重ね合わせの形に分解することを **フーリエ (Fourier) 変換** という。
Fourier 変換は、対象とする信号の種類によっていくつかの種類がある。
ここでは，コンピュータを用いた数値計算に適した **離散フーリエ変換 (discrete Fourier transform, DFT)** を紹介する。

#### ベクトルの分解

任意の $N$ 次元ベクトル $\boldsymbol{x}$ は，
$N$ 個の線形独立なベクトル $\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ の和として
次式のとおり表せる．

$$
\boldsymbol{x} = a_1 \boldsymbol{e}_1 + \cdots + a_N \boldsymbol{e}_N
$$

ここで，$a_1, \cdots, a_N$ はスカラーである．
このとき，$\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ を $\boldsymbol{x}$ の **基底 (basis)** という．

言い換えると， ベクトル $\boldsymbol{x}$ は，より単純なベクトル $\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ に分解できる．
このことは，複雑な対象（ベクトル）を解析するために役立つ．
例えば，平面上の運動 $\boldsymbol{x} = (x_1, x_2)^\top$ は，
$\boldsymbol{e}_1 = (1, 0)^\top$ と $\boldsymbol{e}_2 = (0, 1)^\top$ を用いて
水平方向の運動と垂直方向の運動に分解して考えることができる．

$$
\boldsymbol{x} = x_1 \boldsymbol{e}_1 + x_2 \boldsymbol{e}_2
$$

基底 $\boldsymbol{e}_1, \cdots, \boldsymbol{e}_N$ が互いに直交しているとき，
すなわち，

$$
\langle \boldsymbol{e}_i, \boldsymbol{e}_j \rangle =
\begin{cases}
\alpha & i = j \\
0 & i \neq j
\end{cases}
$$

を満たすとき，$\boldsymbol{e}_1$ と $\boldsymbol{x}$ の内積をとると，$\boldsymbol{e}_1$ の係数 $a_1$ を得られる．

$$
\langle \boldsymbol{e}_1, \boldsymbol{x} \rangle
= \langle \boldsymbol{e}_1, a_1 \boldsymbol{e}_1 + \cdots + a_N \boldsymbol{e}_N \rangle
= a_1 \langle \boldsymbol{e}_1, \boldsymbol{e}_1 \rangle + \cdots + a_N \langle \boldsymbol{e}_1, \boldsymbol{e}_N \rangle
= a_1 \alpha
$$

よって，$a_1 = \langle \boldsymbol{e}_1, \boldsymbol{x} \rangle / \alpha$ となる．
特に，$\alpha = 1$ のとき，$a_1 = \langle \boldsymbol{e}_1, \boldsymbol{x} \rangle$ となる．

#### 複素正弦波信号

複素数値離散時間信号 $e^{j \omega n}$ を，角周波数 $\omega$ の離散時間 **複素正弦波信号** という．
オイラーの公式 $e^{j \theta} = \cos(\theta) + j \sin(\theta)$ を使うと，複素正弦波信号は次のように表せる．

$$
e^{j \omega n} = \cos(\omega n) + j \sin(\omega n)
$$

すなわち，複素正弦波信号は，実部が角周波数 $\omega$ の余弦波，虚部が角周波数 $\omega$ の正弦波である．
複素正弦波信号は $2 \pi / \omega$ が有理数となる場合に限り周期信号となり，その時の周期は $N = 2 \pi / \omega$ である．

長さが $N$ であり，かつ，周期 $N$ を持つ複素正弦波信号は，次の式で表される．

$$
\phi_k[n] = e^{j 2 \pi k n / N}, \quad k, n \in \{0, 1, \cdots, N-1\}
$$

このとき，$\phi_0, \cdots, \phi_{N-1}$ は互いに直交し，

$$
\langle \phi_k, \phi_l \rangle =
\begin{cases}
N & k = l \\
0 & k \neq l
\end{cases}
$$

となる．

#### 離散フーリエ変換 (discrete Fourier transform, DFT)

複素正弦波信号 $\phi_0, \cdots, \phi_{N-1}$ は線形独立であるため，これらを基底として
離散時間信号 $x[n]$ を次のように分解できる．

$$
x[n] = \frac{1}{N} \left( X[0] \phi_0[n] + \cdots + X[N-1] \phi_{N-1}[n] \right)
$$

ここで， $\phi_k$ の係数 $X[k]$ は
$\phi_0, \cdots, \phi_{N-1}$ が互いに直交していることを利用して，

$$
X[k] = \langle \phi_k, x \rangle
$$

と表される．
また，$\frac{1}{N}$ は， $\langle \phi_k, \phi_k \rangle = N$ であることを考慮した正規化係数である．

このとき， $x[n]$ から $X[k]$ への変換を **離散フーリエ変換 (discrete Fourier transform, DFT)** という．

$$
X[k] = \langle \phi_k, x \rangle = \boldsymbol{\phi}_k^* \boldsymbol{x} = \sum_{n=0}^{N-1} \overline{\phi_k[n]} x[n] = \sum_{n=0}^{N-1} x[n] e^{-j 2 \pi k n / N}
$$

$X[k]$ を並べたベクトル $\boldsymbol{X} = (X[0], \cdots, X[N-1])^\top$ を考えると，
DFT は，各行に $\boldsymbol{\phi}_k^*$ を並べた行列

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
と表される．
この行列 $\boldsymbol{F}_N$ を $N$ 点の **DFT 行列** という．

また，$X[k]$ から $x[n]$ への変換を **逆離散フーリエ変換 (inverse discrete Fourier transform, IDFT)** という

$$
x[n] = \frac{1}{N} \left( X[0] \phi_0[n] + \cdots + X[N-1] \phi_{N-1}[n] \right) = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j 2 \pi k n / N}
$$

IDFTも，DFTと同様に行列を使って表すことができる．

$$
\boldsymbol{x} = \boldsymbol{F}_N^{-1} \boldsymbol{X} = \frac{1}{N} \boldsymbol{F}_N^* \boldsymbol{X}
$$

ただし，DFT行列の逆行列は，DFT行列の共役転置を $1/N$ 倍した行列に一致することを利用した．

#### 振幅スペクトルと位相スペクトル

DFTの結果 $X[k]$ は複素数となる．
このとき，$X[k]$ の大きさ $|X[k]| = \sqrt{\mathrm{Re}(X[k])^2 + \mathrm{Im}(X[k])^2}$ を **振幅スペクトル (amplitude spectrum)** といい，
$X[k]$ の偏角 $\angle X[k] = \mathrm{atan2}(\mathrm{Im}(X[k]), \mathrm{Re}(X[k]))$ を **位相スペクトル (phase spectrum)** という．


### 2. DFT の性質

信号 $x[n], x_1[n], x_2[n]$ の $N$ 点 DFT をそれぞれ $X[k], X_1[k], X_2[k]$ とすると，
DFT は次の性質を満たす．

#### 線形性

$$
a_1 x_1[n] + a_2 x_2[n] \xleftrightarrow{\mathrm{DFT}} a_1 X_1[k] + a_2 X_2[k]
$$

#### 周期性

$$
X[k + N] = X[k]
$$

#### 共役性

$$
\overline{x[n]} \xleftrightarrow{\mathrm{DFT}} \overline{X[-k]}
$$

#### 対称性
$x[n]$ が実数値信号であるとき， $X[k]$ は共役対称となる．すなわち，
$$
X[k] = \overline{X[-k]}
$$
言い換えると，振幅スペクトルは偶対称となり，位相スペクトルは奇対称となる．
$$
|X[k]| = |X[-k]|, \quad \angle X[k] = -\angle X[-k]]
$$

#### 時間シフト

$$
x[n - n_0] \xleftrightarrow{\mathrm{DFT}} e^{-j 2 \pi k n_0 / N} X[k]
$$

#### 周波数シフト（変調）

$$
e^{j 2 \pi k_0 n / N} x[n] \xleftrightarrow{\mathrm{DFT}} X[k - k_0]
$$

### 3. 高速フーリエ変換 (fast Fourier transform, FFT)

DFTを定義に従い直接計算すると，$N$ 点の信号に対して $O(N^2)$ の計算量が必要となる．
しかし，計算を工夫することで，$O(N \log N)$ の計算量で同じ結果を得ることができる．
このようにDFTを高速に計算するアルゴリズムの総称を **高速フーリエ変換 (fast Fourier transform, FFT)** という．
FFTは，DFTを計算するためのアルゴリズムであるため，FFTを使って得られる結果はDFTと同じである．


## 演習
基本的にはプログラム (Python) を使って取り組むことを想定しています．
ただし，手計算しなさいと書かれている問題は，コンピュータを使わずに計算してください．

### 基礎レベル
1. 周波数 $F$ Hz の連続時間信号をサンプリング周波数 $F_s$ Hz で標本化して得られる離散時間信号の周波数 $f$ は，
$f = F / F_s$ となる． $f$ を **正規化周波数 (normalized frequency)** という．
周期 $T = 1 / 4$ 秒の連続時間信号 $x(t)$ をサンプリング周波数 $F_s = 10$ Hz で標本化したとき，
得られる離散時間信号 $x[n]$ の周波数 $f$ を手計算しなさい．
また， $x[n]$ の角周波数 $\omega$ を手計算しなさい．

1. $x(t) = e^{j 4\pi t}$ をサンプリング周波数 $F_s = 100$ Hz で1秒間サンプリングした離散時間信号 $x[n]$ を
時間，実部，虚部の3次元でプロットしなさい．
また， $y(t) = e^{-j 4\pi t}$ を同様にプロットし，周波数が正のときと負のときで時間波形がどのように変わるか考察しなさい．

1. $N = 8$ のとき，複素正弦波信号 $\phi_k[n] = e^{j 2 \pi k n / N}, \quad k = 0, 1, \cdots, N-1$ 間の内積
$\langle \phi_k, \phi_l \rangle$ を計算し， $\langle \phi_k, \phi_l \rangle$ を $k$ 行 $l$ 列の要素とする行列を作りなさい．
この行列から $\phi_0, \cdots, \phi_{N-1}$ が互いに直交していることを確認しなさい．

1. $N = 4$ のとき，DFT行列 $\boldsymbol{F}_N$ を手で求めなさい．
また， $\boldsymbol{F}_N$ を使って，$x[n] = (1, 1, 0, 0)^\top$ の DFT を手計算しなさい．
得られた $X[k]$ を， `np.fft.fft` を使ってDFTを計算した結果と比較しなさい．

1. $N = 4$ のとき，逆DFT行列 $\boldsymbol{F}_N^{-1}$ を手で求めなさい．
また， $\boldsymbol{F}_N^{-1}$ を使って，$X[k] = (1, 0, 0, 0)^\top$ の逆DFT を手計算しなさい．
得られた $x[n]$ を， `np.fft.ifft` を使って逆DFTを計算した結果と比較しなさい．

1. 連続時間信号 $x(t) = e^{j 4 \pi t}$ をサンプリング周波数 $F_s = 16$ Hz で標本化した離散時間信号 $x[n]$ の16点 DFT を計算し，
横軸を周波数インデックス$k$として振幅スペクトルをプロットしなさい．
また，横軸を周波数$F$ [Hz] とした振幅スペクトルもプロットしなさい．

1. 連続時間信号 $x(t) = \cos(4 \pi t)$ をサンプリング周波数 $F_s = 16$ Hz で標本化した離散時間信号 $x[n]$ の16点 DFT を計算し，
横軸を周波数$F$ [Hz] として振幅スペクトルと位相スペクトルをプロットしなさい．
また，スペクトルが2本立つ理由を，オイラーの公式およびDFTの周期性を踏まえて考察しなさい．

1. 連続時間信号 $x_e(t) = e^{j 6 \pi t}, x_{\mathrm{cos}}(t) = \cos(6 \pi t), x_{\mathrm{sin}}(t) = \sin(6 \pi t)$ を
サンプリング周波数 $F_s = 8$ Hz で標本化した信号をそれぞれ $x_e[n], x_{\mathrm{cos}}[n], x_{\mathrm{sin}}[n]$ とする．
これらの信号の8点 DFT を計算し，振幅スペクトルと位相スペクトルを描きなさい．
また，これらの信号の振幅スペクトルと位相スペクトルを比較し，どのような関係があるか考察しなさい．

1. 連続時間信号 $x(t) = \cos(100 \pi t)$ をサンプリング周波数 $F_s = 1000$ Hz で標本化した離散時間信号 $x[n]$ に対し，
$N = 1024, 256, 64, 16$ 点 DFT を計算し，横軸を周波数$F$ [Hz] としてそれぞれの振幅スペクトルをプロットしなさい．
また，DFTの点数がスペクトルの見え方にどのような影響を与えるか考察しなさい．

1. DFT行列 $\boldsymbol{F}_N$ を使ってDFTを行う関数 `my_dft` を実装しなさい．
適当な信号 $x[n]$ に対して `my_dft` を使って $N = 8$ のときの DFT を計算し，
`np.fft.fft` を使ってDFTを計算した結果と比較しなさい．

1. $N = 8192$点の信号 $x[n]$ に対して， `my_dft` を使って$N$点 DFT を計算するのにかかる時間を計測しなさい．
また，同じ信号に対して `np.fft.fft` を使って$N$点 DFT を計算するのにかかる時間を計測しなさい．

## 詰まったときに見る資料
- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)
