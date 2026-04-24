# 積分の基礎

## この章の目的

微分が「ある点での変化の速さ」を扱うのに対し，積分は「小さな量を足し合わせて全体量を求める」ための道具です．
機械学習や信号処理では，

- 面積や総量の評価
- 平均値や期待値の計算
- 連続時間信号のエネルギー評価
- 確率密度関数の正規化

などで積分が現れます．

## 不定積分

関数 $F(x)$ が関数 $f(x)$ の原始関数であるとは，

$$
F'(x) = f(x)
$$

が成り立つことを言います．

このとき，

$$
\int f(x)\,dx = F(x) + C
$$

と書きます．ここで $C$ は積分定数です．

### 基本公式

$$
\int x^n\,dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)
$$

$$
\int \cos x\,dx = \sin x + C
$$

$$
\int \sin x\,dx = -\cos x + C
$$

$$
\int \frac{1}{x}\,dx = \log |x| + C
$$

### 線形性

$$
\int \left(af(x) + bg(x)\right)\,dx
= a\int f(x)\,dx + b\int g(x)\,dx
$$

が成り立ちます．

## 定積分

### 面積としての理解

$$
\int_a^b f(x)\,dx
$$

は，非負の関数ならグラフと $x$ 軸に挟まれた面積として理解できます．
より正確には，関数が負になる部分も含めた**符号付き面積**です．

### リーマン和

区間 $[a,b]$ を $N$ 分割し，

$$
\Delta x = \frac{b-a}{N}
$$

とすると，

$$
\sum_{k=0}^{N-1} f(x_k)\Delta x
$$

が定積分の近似になり，

$$
\int_a^b f(x)\,dx
= \lim_{N\to\infty}\sum_{k=0}^{N-1} f(x_k)\Delta x
$$

で定義されます．

### 微積分の基本定理

$F'(x)=f(x)$ を満たす原始関数 $F$ があるとき，

$$
\int_a^b f(x)\,dx = F(b) - F(a)
$$

が成り立ちます．

例えば

$$
\int_0^1 x^2\,dx
= \left[\frac{x^3}{3}\right]_0^1
= \frac{1}{3}
$$

です．

## 置換積分と部分積分の考え方

### 置換積分

例えば

$$
\int 2x \cos(x^2)\,dx
$$

では $u = x^2$ と置くと

$$
\int \cos u\,du = \sin u + C = \sin(x^2) + C
$$

となります．

### 部分積分

積の微分公式を変形すると

$$
\int f(x)g'(x)\,dx
= f(x)g(x) - \int f'(x)g(x)\,dx
$$

が得られます．

## 広義積分

積分区間が無限に広い，または被積分関数が途中で発散する場合には**広義積分**を考えます．

例えば

$$
\int_1^\infty \frac{1}{x^2}\,dx
= \lim_{R\to\infty}\int_1^R \frac{1}{x^2}\,dx
= 1
$$

です．

一方，

$$
\int_1^\infty \frac{1}{x}\,dx
$$

は発散します．

## 多重積分

2 変数関数 $f(x,y)$ に対しては

$$
\iint f(x,y)\,dx\,dy
$$

を考えます．

長方形領域では

$$
\iint_R f(x,y)\,dx\,dy
= \int_a^b \left(\int_c^d f(x,y)\,dy\right)dx
$$

のように 1 変数ずつ積分できます．

## Python で数値積分を確かめる

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

output_dir = Path("outputs/ch05")
output_dir.mkdir(parents=True, exist_ok=True)

x = np.linspace(0.0, 1.0, 1001)
y = x ** 2

approx = np.trapezoid(y, x)
print(f"approx integral = {approx:.6f}")

plt.figure(figsize=(4, 3))
plt.plot(x, y, label="y = x^2")
plt.fill_between(x, y, alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(output_dir / "integral_x_squared.png", dpi=150)
plt.close()
```

## この章で押さえるべき点

- 不定積分は原始関数を求める操作であり，積分定数が必要である
- 定積分は区間上の総和の極限であり，符号付き面積として理解できる
- 微積分の基本定理により，定積分は原始関数の差で計算できる
- 広義積分では無限区間や特異点を極限で扱う
- 多重積分は高次元での総和であり，画像・確率・信号処理で広く使う
