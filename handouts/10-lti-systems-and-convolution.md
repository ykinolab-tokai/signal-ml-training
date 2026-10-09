# 第10回 線形時不変システムと畳み込み

## この回の目標

- 線形時不変システムの線形性と時不変性を理解する。
- 畳み込みを手計算に近い形と NumPy の関数で確認する。
- 畳み込み定理を FFT で確認する。
- フィルタと周波数応答の関係を理解する。

## 解説

### 0. 準備

この回では，基本的な演算のために `numpy` を用い，より高度な信号処理のために `scipy.signal` を用いる．
- [SciPy documentation](https://docs.scipy.org/doc/scipy/index.html)

Python環境の準備・更新は教材repoで行う。演習は作業repoで教材側の `.venv` を有効にして実行する。手順は [共有Python環境](README.md#python環境の準備更新と演習の実行)を参照する。
作業場所と保存先は [共通方針](README.md#作業場所と保存先) に従う。


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
この導出では $h$ が絶対可和（有限長なら自動的に満たす）であるとする。
このとき，出力 $y$ は，

$$
y[n] = \sum_{k=-\infty}^{\infty} x_N[k] h[n-k]
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
    np.add.at(y, np.arange(x.size) % N, x) # 重複する添字への加算も蓄積する
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
システム $\mathcal{H}$ に入力すると，出力は $y[n]=H_N[k]\phi_k[n]$ となる。
入力に掛かる複素利得 $H_N[k]$ を，この周波数での **周波数応答** と呼ぶ。出力波形 $y[n]$ とは区別する。
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

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

`scripts/session10_convolution.py` に実装し，結果と説明はコードコメントまたは既存の結果ファイルに残す。

### 基礎レベル
1. $y[n]=x[n]+1$ と $y[n]=2x[n]+x[n-1]$ の線形性・時不変性を定義で判定する。成立しない場合は反例を1つ示す。
2. $x=(1,2,0,1)$，$h=(1/3,1/3,1/3)$ の線形畳み込みを手計算する。範囲外を0とし，出力長が6になることを確認して `np.convolve` と比べる。
3. 同じ入力で4点FFTの積の逆FFTと，上の `circular_convolve_direct(..., N=4)` を比較する。次にFFT長を6にして線形畳み込みと比べる。線形畳み込みをFFTで計算するためのゼロ埋め長を説明する。
4. 移動平均 $h=(1/3,1/3,1/3)$ と差分 $h=(1,-1)$ の256点FFTの絶対値を，正規化周波数0〜0.5で描く。直流を通すかどうかを手計算でも確かめ，低域通過・高域通過の意味を説明する。

### 発展レベル（1項目を選択）
1. $y[n]=x[-n]$ と $y[n]=x[n]^2$ を判定し，基礎のシステムと比較する。
2. seed=0の長さ256の2配列で，直接巡回畳み込みとFFTを使う実装の結果・計算時間を比較する。
3. **巡回行列と固有値。** $Cv=\lambda v$ を満たす非零の $v$ を固有ベクトル，$\lambda$ を固有値と呼ぶ。独立な固有ベクトルを列に持つ $V$ では $V^{-1}CV$ が対角行列になる。まず `h=[1,2,0,0]` の巡回畳み込みを行列 $C$ で表し，DFTの基底を各列に持つ $V$ について `C@V` と `V@np.diag(np.fft.fft(h))` を比較する。固有値を手で一般形から導くことは必須にしない。

## 確認ポイント
- 線形性と時不変性を別々に判定し，畳み込みの出力長を説明できる。
- 4点では巡回畳み込み，6点ではゼロ埋めによる線形畳み込みを比較している。
- 周波数応答を複素利得として，出力波形と区別している。

## 詰まったときに見る資料
- [LTIシステムの基礎](../textbook/markdown/ch15-basics-of-lti-systems.md)
- [NumPy convolution](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
