# LTI 系の基礎

## システムとして信号処理を見る

信号を入力すると別の信号を出力する仕組みを**システム**と考えます．

$$
y[n] = \mathcal{T}\{x[n]\}
$$

ここで $\mathcal{T}$ は変換規則です．

その中でも特に重要なのが，**線形時不変 (Linear Time-Invariant; LTI) 系**です．

## 線形性

システム $\mathcal{T}$ が**線形**であるとは，任意の信号 $x_1[n], x_2[n]$ と定数 $a, b$ に対して

$$
\mathcal{T}\{a x_1[n] + b x_2[n]\}
= a\mathcal{T}\{x_1[n]\} + b\mathcal{T}\{x_2[n]\}
$$

が成り立つことです．

## 時不変性

システム $\mathcal{T}$ が**時不変**であるとは，入力を $n_0$ サンプル遅らせたとき，出力も同じだけ遅れることです．

$$
\mathcal{T}\{x[n-n_0]\} = y[n-n_0]
$$

## インパルス応答

LTI 系に単位インパルス $\delta[n]$ を入力したときの出力を

$$
h[n] = \mathcal{T}\{\delta[n]\}
$$

と定義し，これを**インパルス応答**と呼びます．

## 畳み込み

任意の入力に対する出力は

$$
y[n] = (x * h)[n] = \sum_{m=-\infty}^{\infty} x[m] h[n-m]
$$

で与えられます．これが**畳み込み和**です．

例えば

$$
h[0] = \frac{1}{3}, \quad h[1] = \frac{1}{3}, \quad h[2] = \frac{1}{3}
$$

なら

$$
y[n] = \frac{x[n] + x[n-1] + x[n-2]}{3}
$$

となり，3 点移動平均フィルタになります．

## 差分方程式による表現

例えば

$$
y[n] = x[n] + 0.5x[n-1]
$$

は FIR 系です．

一方，

$$
y[n] = 0.8 y[n-1] + x[n]
$$

のように過去の出力を使う系は IIR 系です．

## 因果性と BIBO 安定性

### 因果性

LTI 系が因果的であるためには，

$$
h[n] = 0 \quad (n < 0)
$$

であることが必要です．

### BIBO 安定性

離散時間 LTI 系では

$$
\sum_{n=-\infty}^{\infty} |h[n]| < \infty
$$

であれば BIBO 安定です．

## 周波数応答

複素指数信号

$$
x[n] = \exp(\mathrm{j}\omega n)
$$

を LTI 系に入れると，出力は

$$
y[n] = H(\mathrm{e}^{\mathrm{j}\omega}) \exp(\mathrm{j}\omega n)
$$

と書けます．

ここで

$$
H(\mathrm{e}^{\mathrm{j}\omega}) = \sum_{n=-\infty}^{\infty} h[n]\exp(-\mathrm{j}\omega n)
$$

を**周波数応答**と呼びます．

## Python で移動平均と IIR を比べる

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

output_dir = Path("outputs/ch15")
output_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
n = np.arange(200)
x = np.sin(2 * np.pi * 0.03 * n) + 0.4 * rng.standard_normal(size=n.shape)

h = np.ones(5) / 5
y_fir = np.convolve(x, h, mode="same")

y_iir = np.zeros_like(x)
alpha = 0.8
for i in range(len(x)):
    prev = y_iir[i - 1] if i > 0 else 0.0
    y_iir[i] = alpha * prev + (1 - alpha) * x[i]

plt.figure(figsize=(8, 3))
plt.plot(n, x, label="input", alpha=0.5)
plt.plot(n, y_fir, label="FIR moving average", linewidth=2)
plt.plot(n, y_iir, label="first-order IIR", linewidth=2)
plt.xlabel("sample index n")
plt.ylabel("amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fir_vs_iir.png", dpi=150)
plt.close()
```

## この章で押さえるべき点

- LTI 系は線形性と時不変性を持つシステムである
- LTI 系はインパルス応答 $h[n]$ だけで記述できる
- 任意入力への出力は畳み込み和 $y[n] = (x*h)[n]$ で与えられる
- 因果性は $h[n]=0$ for $n<0$，BIBO 安定性は $\sum_n |h[n]| < \infty$ と関係する
- 周波数応答を見ると，フィルタが各周波数をどう変えるか分かる
