# 第10回 線形時不変システムと畳み込み

- 線形時不変システムの線形性と時不変性を理解する。
- 畳み込みを手計算に近い形と NumPy の関数で確認する。
- 畳み込み定理を FFT で確認する。
- フィルタと周波数応答の関係を理解する。

## 解説

### 0. 準備

この回では，基本的な演算のために `numpy` を用い，より高度な信号処理のために `scipy.signal` を用いる．
- [SciPy documentation](https://docs.scipy.org/doc/scipy/index.html)

`SciPy` は，次のコマンドでインストールできる．
```bash
pip install scipy
```
Python仮想環境を作成している場合は，仮想環境を有効にしてから `pip install` を実行する．


### 1. 線形時不変システム

#### 信号処理システム
信号を別の信号へ変換する規則を，
**信号処理システム** と呼ぶ．
以下では，信号処理システムを単にシステムと呼ぶ．

数学的には，システムは信号から信号への写像である．
システム $\mathcal{S}$ により，
入力信号 $x$ が出力信号 $y$ に変換されることを，
次のように表す．

$$
y = \mathcal{S}\{x\} \qquad \text{または} \qquad y[n] = \mathcal{S}\{x\}[n]
$$

#### 線形性

任意の $a, b \in \mathbb{C}$ および
任意の信号 $x_1, x_2$ に対して，
以下の性質を満たすシステム $\mathcal{S}$ を，
**線形システム** と呼ぶ．

$$
\mathcal{S}\{a x_1 + b x_2\} = a \mathcal{S}\{x_1\} + b \mathcal{S}\{x_2\}
$$

#### 時不変性

信号を $n_0$ サンプル遅延させる以下のシステム $\mathcal{D}_{n_0}$ を考える．

$$
y[n] = \mathcal{D}_{n_0}\{x\}[n] = x[n-n_0]
$$

このとき，任意の信号 $x$ および任意の $n_0$ に対して，
以下の性質を満たすシステム $\mathcal{S}$ を，
**時不変システム** と呼ぶ．

$$
\mathcal{S}\{\mathcal{D}_{n_0}\{x\}\} = \mathcal{D}_{n_0}\{\mathcal{S}\{x\}\}
$$

#### 線形時不変システム

線形性と時不変性の両方を満たすシステムを，
**線形時不変システム**（Linear Time-Invariant System, LTI システム）と呼ぶ．
以下では，LTI システムを記号 $\mathcal{H}$ で表す．

### 2. インパルス応答

#### 単位インパルス信号

以下の離散時間信号 $\delta$ を，**単位インパルス信号** と呼ぶ．

$$
\delta[n] = \begin{cases}
1 & n = 0 \\
0 & n \neq 0
\end{cases}
$$

#### システムのインパルス応答

システム $\mathcal{S}$ に単位インパルス信号 $\delta$
を入力したときの出力 $\mathcal{S}\{\delta\}$ を，
**システムのインパルス応答** と呼ぶ．

### 3. 線形畳み込み

#### インパルス信号による離散時間信号の分解

任意の離散時間信号 $x$ は，単位インパルス信号 $\delta$
の遅延和として表すことができる．

$$
x = \sum_{k=-\infty}^{\infty} x[k] \mathcal{D}_k\{\delta\}
$$

時刻 $n$ における信号 $x$ の値は，次のように表すことができる．

$$
x[n] = \sum_{k=-\infty}^{\infty} x[k] \delta[n-k]
$$

#### 線形時不変システムの時間領域における入出力関係

線形時不変システム $\mathcal{H}$ に入力信号 $x$ を与えたときの出力 $y$ は，

$$
y = \mathcal{H}\{x\}
= \mathcal{H}\left\{\sum_{k=-\infty}^{\infty} x[k] \mathcal{D}_k\{\delta\}\right\}
\overset{\text{線形性}}{=} \sum_{k=-\infty}^{\infty} x[k] \mathcal{H}\{\mathcal{D}_k\{\delta\}\}
\overset{\text{時不変性}}{=} \sum_{k=-\infty}^{\infty} x[k] \mathcal{D}_k\{\mathcal{H}\{\delta\}\}
$$

とかける．

したがって， $y[n]$ は，
システムのインパルス応答 $h = \mathcal{H}\{\delta\}$ を用いて，次のように表すことができる．
$$
y[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]
$$

#### 線形畳み込みの定義

2つの信号 $x_1$ と $x_2$ に対する以下の演算 $x_1 * x_2$ を，**線形畳み込み (linear convolution)** ，
または単に **畳み込み (convolution)** と呼ぶ．

$$
(x_1 * x_2)[n] = \sum_{k=-\infty}^{\infty} x_1[k] x_2[n-k] = \sum_{k=-\infty}^{\infty} x_2[k] x_1[n-k] = (x_2 * x_1)[n]
$$

線形時不変システムの出力は，入力信号とシステムのインパルス応答の畳み込み $x * h$ に一致する．

畳み込みは， `numpy.convolve` 関数で計算できる．

```python
import numpy as np
x = [1, 2, 0, 1]
h = [1, -1, 0.5]
y = np.convolve(x, h, mode='full')
```

### 4. 畳み込み定理

#### 巡回畳み込み

$N$ 点周期の離散時間信号 $x_N$ を
線形時不変システム $\mathcal{H}$ に入力することを考える．
このとき，出力 $y$ は，

$$
y[n] = \sum_{k=0}^{N-1} x_N[k] h[n-k]
\overset{k=l+mN}{=} \sum_{l=0}^{N-1} \sum_{m=-\infty}^{\infty} x_N[l+mN] h[n-l-mN]
= \sum_{l=0}^{N-1} x_N[l] \sum_{m=-\infty}^{\infty} h[n-l-mN]
$$

ここで， $h_N[n] = \sum_{m=-\infty}^{\infty} h[n-mN]$ と定義すると，出力 $y$ は，

$$
y[n] = \sum_{l=0}^{N-1} x_N[l] h_N[n-l] = \sum_{l=0}^{N-1} h_N[l] x_N[n-l]
$$

と表せる．
この演算は， $x_N$ と $h_N$ の $N$点 **巡回畳み込み (circular convolution)** と呼ばれ，
$x_N \circledast_N h_N$ と表される．
また，非周期信号 $h$ から $N$ 点周期の信号 $h_N = \sum_{m=-\infty}^{\infty} h[n-mN]$ を得る操作を，
$h$ の $N$ 点 **周期化** と呼ぶ．

巡回畳み込みは，線形畳み込みの結果 $y$ を$N$点周期化したものと一致する．

信号の周期化は，次のように実装できる．

```python
import numpy as np

def periodize(x, N):
    x = np.asarray(x)
    y = np.zeros(N, dtype=x.dtype)
    np.add.at(y, np.arange(x.size) % N, x) # nearly equivalent to y[np.arange(x.size) % N] += x
    return y

x = np.arange(6)  # [0, 1, 2, 3, 4, 5]
print(periodize(x, 4))
```

また，巡回畳み込みは， `periodize` 関数を用いて次のように実装できる．

```python
import numpy as np

def circular_convolve_direct(x, h, N=None):
    x = np.asarray(x)
    h = np.asarray(h)

    if N is None:
        if x.size != h.size:
            raise ValueError("x と h の長さが異なる場合は N を指定してください。")
        N = x.size

    xN = periodize(x, N)
    hN = periodize(h, N)

    dtype = np.result_type(xN, hN, np.float64)
    y = np.zeros(N, dtype=dtype)

    for m in range(N):
        y += hN[m] * np.roll(xN, m)

    return y

x = np.array([1, 2, 3])
h = np.array([4, 5, 6])
y = circular_convolve_direct(x, h, N=3)
```

#### システムの周波数応答

$N$ 点周期の複素正弦波信号 $\phi_k = e^{j 2 \pi k n / N}, \quad k \in \{0, 1, \cdots, N-1\}$ を
システム $\mathcal{H}$ に入力したときの出力を，**システムの周波数応答** と呼ぶ．
このときの出力 $y$ は，巡回畳み込みを用いて次式で与えられる．
$$
y[n] = \mathcal{H}\{\phi_k\}[n] = \sum_{l=0}^{N-1} h_N[l] \phi_k[n-l]
= \sum_{l=0}^{N-1} h_N[l] e^{j 2 \pi k (n-l) / N}
= e^{j 2 \pi k n / N} \sum_{l=0}^{N-1} h_N[l] e^{-j 2 \pi k l / N}
= H_N[k] \phi_k[n]
$$
ここで，$\sum_{l=0}^{N-1} h_N[l] e^{-j 2 \pi k l / N}$ は $h_N$ の $N$ 点 DFT $H_N[k]$ であることを用いた．
この式から，システム $\mathcal{H}$ に複素正弦波信号 $\phi_k$ を入力したときの出力は，
$\phi_k$ を単に $H_N[k]$ 倍したものであることがわかる．
すなわち， $\mathcal{H}\{\phi_k\} = H_N[k] \phi_k$ なる関係が成り立っている．
よって，線形時不変システム $\mathcal{H}$ は，複素正弦波信号 $\phi_k$ の周波数は変化させず，
振幅を $| H_N[k] |$ 倍し，位相を $\angle H_N[k]$ だけ変化させるシステムであるといえる．

#### 畳み込み定理

$N$ 点の信号は周期 $N$ の複素正弦波信号の線形結合で表せる（逆DFT）ことを利用して，
システム $\mathcal{H}$ の入出力関係を考えてみよう．

$N$ 点の入力信号 $x$ は，そのDFT $X$ を用いて次のように表すことができる．

$$
\check{x} = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \phi_k
$$

ここで，複素正弦波の周期性より $\check{x}[n] = \check{x}[n+N]$ が成り立ち，
$\check{x}$ は周期 $N$ を持つことがわかる．
したがって， $\check{x}$ をシステム $\mathcal{H}$ に入力したときの出力は，
周期化されたインパルス応答との巡回畳み込みとして次の通り与えられる．

$$
y = \mathcal{H}\{\check{x}\} = \mathcal{H}\left\{\frac{1}{N} \sum_{k=0}^{N-1} X[k] \phi_k \right\}
= \frac{1}{N} \sum_{k=0}^{N-1} X[k] \mathcal{H}\{\phi_k\}
= \frac{1}{N} \sum_{k=0}^{N-1} X[k] H_N[k] \phi_k
$$

この式は， $X[k]$ と $H_N[k]$ の積の逆DFTが出力 $y$ になることを示している．
すなわち， $y$ のDFTは，

$$
Y[k] = X[k] H_N[k]
$$

となる．

以上のことから，次の **畳み込み定理** を得る．

$$
x \circledast_N h \; \xleftrightarrow{\mathrm{DFT}} \; X[k]H[k]
$$




## 演習
基本的にはプログラム (Python) を使って取り組むことを想定しています．
ただし，手計算しなさいと書かれている問題は，コンピュータを使わずに計算してください．

### 基礎レベル
1. 以下のシステム $y = \mathcal{S}\{x\}$ の線形性と時不変性を確認し，線形時不変システムをすべて挙げなさい．

    1. $y[n] = x[n] + 1$
    2. $y[n] = 2 x[n] + x[n-1]$
    4. $y[n] = x[-n]$
    5. $y[n] = x[n]^2$
    6. $y[n] = 0.9 y[n-1] + x[n], \quad y[n] = 0 \quad \text{for} \; n < 0$

2. 1の各システムのインパルス応答を求めなさい．また，インパルス応答の長さを求めなさい．

3. $x = (x[0], \cdots, x[3])^\top = (1, 2, 0, 1)^\top$ と $h = (h[0], h[1], h[2])^\top = (1/3, 1/3, 1/3)^\top$ を考える．

    1. 線形畳み込み $y = x * h$ を手計算しなさい．また，その結果得られる信号 $y$ の長さを求めなさい．
    2. 4点巡回畳み込み $y_4 = x \circledast_4 h$ を手計算しなさい．また，その結果得られる信号 $y_4$ の長さを求めなさい．
    3. `numpy.convolve(x, h)` を使って線形畳み込み $y = x * h$ を計算し， $y$ を `stem` プロットしなさい．
    4. `circular_convolve_direct(x, h, N=4)` を使って巡回畳み込み $y_4 = x \circledast_4 h$ を計算し， $y_4$ を `stem` プロットしなさい．
    5. $x$ の4点DFTと $h$ の4点DFTを計算し，それらの積の逆DFTを計算して，巡回畳み込み $y_4 = x \circledast_4 h$ の結果と比較しなさい．
    6. `circular_convolve_direct(x, h, N)` の $N$ を変化させて巡回畳み込み $y_N = x \circledast_N h$ を計算し， $y_N$ が $y$ と一致するための $N$ の条件を考察しなさい．

4. インパルス応答が $h = (h[0], h[1])^\top = (1, -1)^\top$ であるシステム $\mathcal{H}$ を考える．

    1. $x = (x[0], \cdots, x[3])^\top = (1, 2, 0, 1)^\top$ と $h$ の線形畳み込み $y = x * h$ を手計算しなさい．
       また，その結果得られる信号 $y$ の長さを求めなさい．
    2. チャープ信号 $x(t) = \sin(2\pi (f_0t + \frac{1}{2}kt^2))$ を，サンプリング周波数 $F_s = 16000$ Hz で
       サンプリングした信号 $x[n]$ を作成しなさい．
       ただし， $k = (f_1 - f_0) / T$ であり，
       $f_0 = 100$ [Hz]，$f_1 = 4000$ [Hz]，$T = 1$ [s] とする．
    3. システム $\mathcal{H}$ に入力信号 $x[n]$ を与えたときの出力 $y[n]$ を，線形畳み込みを用いて計算し，
       $y[n]$ を横軸時間 $t$ としてプロットしなさい．
    4. $x[n], h[n], y[n]$ のDFTを計算し，それらの振幅スペクトルに基づき2の結果を考察しなさい．
       ただし，DFTは16000点以上で計算すること．

5. 信号に含まれる所望の信号のみを通過し，それ以外を阻止するシステムを，**フィルタ (filter)** と呼ぶ．
   特に，低周波成分を通過し高周波成分を阻止するフィルタを **低域通過フィルタ (low-pass filter)** と呼び，
   高周波成分を通過し低周波成分を阻止するフィルタを **高域通過フィルタ (high-pass filter)** と呼ぶ．
   以下のインパルス応答 $h = (h[0], h[1], \cdots)^\top$ を持つシステム $\mathcal{H}$ が
   低域通過フィルタであるか高域通過フィルタであるかを，
   $h$ の振幅スペクトルをプロットして考察しなさい．

    1. $h = (1/3, 1/3, 1/3)^\top$
    2. $h = (1, -1)^\top$

6. $x[n]$, $h[n]$ をそれぞれ2048点の離散時間信号とし，これらの2048点巡回畳み込みを計算する．
   `circular_convolve_direct(x, h, N=2048)` を用いた場合の計算時間と，
   FFT を用いて畳み込み定理に基づいて計算した場合の計算時間をそれぞれ測定し，比較しなさい．

### 発展レベル

1. 次の行列 $\boldsymbol{C}$ を $4 \times 4$ の巡回行列と呼ぶ．以下の問いに答えなさい.

    $$
    \boldsymbol{C} = \begin{bmatrix}
    c_0 & c_3 & c_2 & c_1 \\
    c_1 & c_0 & c_3 & c_2 \\
    c_2 & c_1 & c_0 & c_3 \\
    c_3 & c_2 & c_1 & c_0
    \end{bmatrix}
    $$ 

    1. $\boldsymbol{C}$ の固有値・固有ベクトルを求め，$\boldsymbol{C}$ を対角化しなさい．
    2. $x = (x[0], \cdots, x[3])^\top$ と $h = (h[0], \cdots, h[3])^\top$ の巡回畳み込み $y = x \circledast_4 h$ を，
       $y = \boldsymbol{A} x$ と表すことができる．行列 $\boldsymbol{A}$ を求めなさい．
    3. DFT行列 $\boldsymbol{F}_4$ を用いて $Y[k] = X[k] H[k]$ となることをを導きなさい．

## 詰まったときに見る資料
- [NumPy convolution](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
- [SciPy documentation](https://docs.scipy.org/doc/scipy/index.html)
