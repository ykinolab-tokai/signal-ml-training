# 第08回 Fourier 変換

## この回の目標

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
複素正弦波信号が周期的になるのは，$\omega/(2\pi)$ が有理数のときである。
$\omega/(2\pi)=p/q$ を既約分数（$q>0$）で表すと，基本周期は整数 $q$ となる。
例えば $\omega=3\pi/4$ では $p/q=3/8$ なので基本周期は8である。
$\omega=0$ は定数列であり，離散時間では最小の正の整数周期を1とする。

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
|X[k]| = |X[-k]|, \quad \angle X[k] = -\angle X[-k] \pmod{2\pi}
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


### 4. 窓関数とスペクトル漏れ

有限区間を切り出すと，DFTではその区間を周期的に繰り返すとみなす。
端点の不連続があると，成分が周辺の周波数ビンへ広がる（スペクトル漏れ）。
Hann窓は端の値を小さくして遠くへの漏れを抑える一方，ピーク付近の主ローブを広げる。
振幅を比べるときは窓の和で規格化し，窓の大きさ自体の違いを除く。

## 演習

作業場所は [提出repo](README.md#作業場所と保存先) のルートとする。

`scripts/session08_fourier.py` と `outputs/session08/session08_report.md` を作る。図は `outputs/figures/` に保存する。

### 基礎レベル
1. $z=1+2j$ の共役と $\overline{z}z$ を手計算して `np.conj` で確認する。周期 $T=1/4$ 秒の信号を $F_s=10$ Hzで標本化したときの正規化周波数と角周波数を求める。$\omega=3\pi/4$ の基本周期も確認する。
2. $N=4$，$x=(1,1,0,0)$ のDFTを手計算し，`np.fft.fft(x)` と比較する。`np.fft.ifft` で元に戻し，最大絶対誤差を求める。
3. $F_s=16$ Hz，$N=16$で2 Hzの余弦波を生成し，`np.fft.fftfreq` を使った周波数軸で振幅・位相を描く。振幅が `1e-10` 未満のビンの位相は解釈しない。正負2つのピークがある理由を説明する。
4. **窓の比較。** $F_s=64$ Hz，$N=64$を固定し，8 Hzと8.5 Hzの余弦波に矩形窓 `np.ones(N)` とHann窓 `np.hanning(N)` を掛ける。`abs(np.fft.rfft(x*w))/w.sum()` を同じ軸で比較する。形を細かく見るためだけにFFT長を2048へ増やした図も作り，遠くへの漏れと主ローブ幅の違いを説明する。ゼロ埋めは観測時間や本来の分解能を増やさない。

### 発展レベル（1項目を選択）
1. $N=8$ のDFT行列を作り，基底間の共役内積が対角成分8，非対角成分0になることを確認する。自作 `my_dft` とFFTを比較する。
2. 自作DFTの時間を $N=32,128,256$ でFFTと比較する。`complex128` のDFT行列だけで $16N^2$ byte必要である。$N=8192$ は行列だけで1 GiBとなり，一時配列も必要なため本課題では作成しない。
3. 正負の複素正弦波を，時間・実部・虚部の3軸で図示し，回転方向を比較する。

## 確認ポイント
- 離散時間の周期を正の整数で判定している。
- DFTと逆DFTの数値が手計算・FFTと一致する。
- 窓の比較で標本化周波数，観測点数，規格化をそろえ，漏れと主ローブ幅を説明している。

## 詰まったときに見る資料
- [スペクトル解析の基礎](../textbook/markdown/ch14-basics-of-spectrum-analysis.md)
- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)
