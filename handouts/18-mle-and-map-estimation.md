# 第18回 最尤推定と MAP 推定

## この回の目標

尤度に基づいてパラメータを推定し，損失関数と正則化の意味を確率の観点から説明できる。

## 解説

この回では，第6回で扱った確率分布を使って，データからモデルのパラメータを推定する方法を学ぶ。
第15〜17回では，平均二乗誤差を損失関数として使った。
第19回では，分類のために交差エントロピーと呼ばれる損失関数を使う。
この回では，これらの損失関数と，第16回の ridge 回帰で加えた項が，確率の考え方からどのように導かれるかを説明する。
スクリプトの先頭には，次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

figure_dir = Path("outputs/exc18")
figure_dir.mkdir(parents=True, exist_ok=True)
```

以下のコード例は，このコードに続けて書いて実行する。

### 1. 尤度と最尤推定

#### 確率モデルとパラメータ

表が出る確率が $\theta$ のコインを投げ，表を 1，裏を 0 で表す。
1 回の結果 $x\in\{0, 1\}$ が得られる確率は，次の式で書ける。

$$
p(x;\theta) = \theta^x(1-\theta)^{1-x}
$$

$x = 1$ を代入すると $\theta$ に，$x = 0$ を代入すると $1-\theta$ になる。
このように，0 と 1 の 2 つの値をとる確率変数の分布を **ベルヌーイ分布 (Bernoulli distribution)** という。
$p(x;\theta)$ の $;\theta$ は，確率がパラメータ $\theta$ によって決まることを表す。

各回の結果が互いに影響しないとき，それらの結果は **独立 (independent)** であるという。
独立なデータ $x_1, \ldots, x_N$ が得られる確率は，各回の確率の積になる。
例えば，$\theta = 0.5$ のコインを 3 回投げて，1，1，0 の順に結果が得られる確率は，$0.5 \times 0.5 \times 0.5 = 0.125$ である。

#### 尤度と対数尤度

データ $x_1, \ldots, x_N$ を固定し，それらが得られる確率をパラメータ $\theta$ の関数とみたもの

$$
L(\theta) = \prod_{i=1}^{N} p(x_i;\theta)
$$

を **尤度 (likelihood)** という。
$\prod$ は，和の記号 $\sum$ と同様に，積を表す記号である。
連続値のデータでは，確率の代わりに，第6回で扱った確率密度関数の値を掛け合わせる。

尤度の対数をとった

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{N}\log p(x_i;\theta)
$$

を **対数尤度 (log-likelihood)** という。
第3回の対数法則により，積の対数は対数の和になる。

```python
data = np.array([1, 1, 0])

def likelihood(theta, data):
    return np.prod(theta**data * (1 - theta)**(1 - data))

def log_likelihood(theta, data):
    return np.sum(data * np.log(theta) + (1 - data) * np.log(1 - theta))

print(likelihood(0.5, data), likelihood(2 / 3, data))          # 0.125 約 0.1481
print(log_likelihood(0.5, data), log_likelihood(2 / 3, data))  # 約 -2.0794 約 -1.9095
```

データ 1，1，0 は，$\theta = 0.5$ のコインよりも $\theta = 2/3$ のコインのほうが得られやすい。
尤度は，そのパラメータのもとでデータがどれだけ得られやすいかを表す。
ただし，尤度は $\theta$ の確率ではないので，$\theta$ について積分しても 1 になるとは限らない。

データの数が多いと，1 より小さい値を多数掛け合わせるため，尤度は極端に小さくなる。

```python
p = np.full(1100, 0.5)
print(np.prod(p))         # 0.0
print(np.sum(np.log(p)))  # 約 -762.46
```

$0.5^{1100}$ はおよそ $10^{-331}$ であり，`float64` で表せる最小の正の数（約 $5\times10^{-324}$）より小さいため，0 になってしまう。
このように，絶対値が小さすぎる値を表せずに 0 になる現象を **アンダーフロー (underflow)** という。
対数の和として計算すると，$1100\log0.5\approx-762.46$ を問題なく計算できる。
このため，実際の計算では尤度の代わりに対数尤度を使う。

#### 最尤推定

観測されたデータが最も得られやすくなるパラメータ，つまり尤度を最大にするパラメータを求めることを **最尤推定 (maximum likelihood estimation)** という。
最尤推定で求めた値を **最尤推定値** という。
$\log$ は単調増加関数なので，尤度を最大にする $\theta$ と，対数尤度を最大にする $\theta$ は一致する。

データ 1，1，0 の対数尤度は $\ell(\theta) = 2\log\theta + \log(1-\theta)$ である。
第4回の $(\log x)' = 1/x$ と連鎖律を使って $\theta$ で微分すると，次のようになる。

$$
\ell'(\theta) = \frac{2}{\theta} - \frac{1}{1-\theta}
$$

$\ell'(\theta) = 0$ を解くと $2(1-\theta) = \theta$ より $\theta = 2/3$ となり，これが最尤推定値である。
$0<\theta<2/3$ では $\ell'(\theta)>0$，$2/3<\theta<1$ では $\ell'(\theta)<0$ なので，$\theta = 2/3$ で最大になる。
同じ値は，$\theta$ を細かく動かして対数尤度を計算しても確かめられる。

```python
theta = np.linspace(0.01, 0.99, 99)
ll = np.array([log_likelihood(t, data) for t in theta])
print(theta[np.argmax(ll)])  # 0.67

fig, ax = plt.subplots()
ax.plot(theta, ll)
ax.set_xlabel("theta")
ax.set_ylabel("log-likelihood")
ax.grid(True)
fig.savefig(figure_dir / "session18_example_log_likelihood.png")
plt.show()
plt.close(fig)
```

`np.argmax` は，第3回で扱った `np.argmin` と同様に，最大値を持つ要素のインデックスを返す。
`theta` の刻み幅は 0.01 なので，$2/3$ に最も近い 0.67 が得られる。

機械学習では，損失関数を最小にする形で問題を書くことが多い。
対数尤度を最大にすることは，符号を反転した $-\ell(\theta)$ を最小にすることと同じである。
$-\ell(\theta)$ を **負の対数尤度 (negative log-likelihood)** という。
以下で見るように，回帰と分類でよく使われる損失関数は，負の対数尤度として導かれる。

### 2. 正規分布の最尤推定と二乗誤差

#### 平均と分散の最尤推定

第6回で扱った正規分布 $N(\mu, \sigma^2)$ の確率密度関数は，次の式で表される。

$$
p(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

データ $x_1, \ldots, x_N$ が独立に $N(\mu, \sigma^2)$ に従うとき，対数尤度は次のようになる。

$$
\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x_i-\mu)^2
$$

$\mu$ は第 2 項だけに含まれるので，$\ell$ を $\mu$ について最大にすることは，$\sum_{i}(x_i-\mu)^2$ を最小にすることと同じである。
$\ell$ を $\mu$ で偏微分して 0 とおくと，次の式が得られる。

$$
\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i=1}^{N}(x_i-\mu) = 0
\quad\Longrightarrow\quad
\hat{\mu} = \frac{1}{N}\sum_{i=1}^{N}x_i
$$

つまり，$\mu$ の最尤推定値は標本平均である。
同様に $\sigma^2$ で偏微分すると，分散の最尤推定値は次のようになる。

$$
\hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{\mu})^2
$$

これは，第6回で扱った `ddof=0` の分散である。

```python
def gaussian_log_likelihood(x, mu, sigma):
    return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2))

x = np.array([1.2, 0.8, 1.5, 0.9])
print(x.mean())  # 1.1
for mu in [1.0, 1.1, 1.2]:
    print(mu, gaussian_log_likelihood(x, mu, 1.0))
```

```text
1.0 -3.8457541328186906
1.1 -3.8257541328186906
1.2 -3.8457541328186906
```

$\sigma = 1$ に固定すると，対数尤度は標本平均の 1.1 で最大になり，そこから同じだけ離れた 1.0 と 1.2 では同じ値になる。
$\sum_{i}(x_i-\mu)^2$ が $\mu$ の 2 次関数であり，そのグラフが $\hat{\mu}$ を軸とする放物線になるためである。

#### 回帰と二乗誤差

回帰では，入力 $x_i$ に対する出力 $y_i$ が，モデルの予測 $f(x_i;\theta)$ に誤差 $\varepsilon_i$ が加わったものだと考える。

$$
y_i = f(x_i;\theta) + \varepsilon_i,\qquad \varepsilon_i \sim N(0, \sigma^2)
$$

$\varepsilon_i \sim N(0, \sigma^2)$ は，誤差 $\varepsilon_i$ が平均 0，分散 $\sigma^2$ の正規分布に従うことを表す。
このとき，$y_i$ は平均 $f(x_i;\theta)$，分散 $\sigma^2$ の正規分布に従う。
誤差が互いに独立なら，$\theta$ についての負の対数尤度は次のようになる。

$$
-\ell(\theta) = \frac{N}{2\sigma^2}\cdot\frac{1}{N}\sum_{i=1}^{N}\bigl(y_i - f(x_i;\theta)\bigr)^2 + \frac{N}{2}\log(2\pi\sigma^2)
$$

右辺の $\frac{1}{N}\sum_{i}(y_i - f(x_i;\theta))^2$ は，平均二乗誤差である。
$\sigma$ を固定すると，負の対数尤度は「平均二乗誤差の正の定数倍」に「$\theta$ によらない定数」を加えたものになる。
したがって，負の対数尤度を最小にする $\theta$ と，平均二乗誤差を最小にする $\theta$ は一致する。
つまり，平均二乗誤差を損失関数に使うことは，誤差が一定の分散の正規分布に従うと仮定して最尤推定を行うことに当たる。
この仮定が成り立たない場合（例えば，極端に外れた値がデータに多く含まれる場合）には，別の損失関数が使われることもある。

### 3. sigmoid と binary cross-entropy

#### sigmoid 関数

**sigmoid 関数 (sigmoid function)** は，次の式で定義される関数である。

$$
\mathrm{sigmoid}(z) = \frac{1}{1+e^{-z}}
$$

文献では $\sigma(z)$ と書くことが多いが，この資料では標準偏差の $\sigma$ と区別するため，$\mathrm{sigmoid}(z)$ と書く。
sigmoid 関数は次の性質を持つ。

- 任意の実数 $z$ に対して，値は 0 より大きく 1 より小さい。
- 単調増加であり，$\mathrm{sigmoid}(0) = 0.5$ である。
- $z$ が大きな正の数なら 1 に近づき，大きな負の数なら 0 に近づく。
- $\mathrm{sigmoid}(-z) = 1 - \mathrm{sigmoid}(z)$ が成り立つ。

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(np.array([-2.0, 0.0, 2.0])))  # [0.11920292 0.5        0.88079708]
```

$\mathrm{sigmoid}(-2) + \mathrm{sigmoid}(2) = 1$ となっていることが確かめられる。

#### 2 クラス分類の尤度

正解ラベルが $y\in\{0, 1\}$ の 2 クラス分類を考える。
モデルは入力 $x$ から実数 $z = f(x;\theta)$ を出力し，それを sigmoid 関数でクラス 1 の確率 $q = \mathrm{sigmoid}(z)$ に変換する。
確率に変換する前のモデルの出力 $z$ を **logit** という（複数形は logits）。
ラベル $y$ は確率 $q$ のベルヌーイ分布に従うと考えると，その確率は $q^y(1-q)^{1-y}$ である。
したがって，$N$ 個のサンプルの負の対数尤度の平均は，次の式になる。

$$
\mathrm{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\bigl(y_i\log q_i + (1-y_i)\log(1-q_i)\bigr)
$$

これを **binary cross-entropy (BCE)** という。
$y_i = 1$ のサンプルの項は $-\log q_i$ であり，$q_i$ が 1 に近いほど小さい。
$y_i = 0$ のサンプルの項は $-\log(1-q_i)$ であり，$q_i$ が 0 に近いほど小さい。
正解のクラスに高い確率を与えるほど，損失は小さくなる。

#### BCEWithLogitsLoss

PyTorch の `nn.BCEWithLogitsLoss` は，logits を受け取り，sigmoid 関数と binary cross-entropy をまとめて計算する。

```python
z = torch.tensor([0.0, 3.0])
y = torch.tensor([1.0, 0.0])
print(nn.BCEWithLogitsLoss()(z, y))                  # tensor(1.8709)
print(nn.BCEWithLogitsLoss(reduction="none")(z, y))  # tensor([0.6931, 3.0486])
```

1 つ目のサンプルは $q = \mathrm{sigmoid}(0) = 0.5$ なので，損失は $-\log0.5\approx0.6931$ である。
2 つ目のサンプルは正解が 0 なのに $q = \mathrm{sigmoid}(3)\approx0.95$ とクラス 1 を強く予測しているため，損失は約 3.0486 と大きい。
既定では，サンプルごとの損失の平均を返す。
`reduction="none"` を指定すると，サンプルごとの損失を返す。
正解ラベルは，logits と同じ形の浮動小数点数の tensor として渡す。

sigmoid 関数で確率を求めてから対数をとると，$|z|$ が大きいときに計算が破綻することがある。
例えば，`float32` では $\mathrm{sigmoid}(-200)$ が 0 になり，$\log 0$ を計算することになる。
`nn.BCEWithLogitsLoss` は，$-\log\mathrm{sigmoid}(z) = \log(1+e^{-z})$ のような式変形を使って，このような場合でも正しい値を計算する。
このため，2 クラス分類のモデルは最後に sigmoid 関数を入れずに logits を出力し，損失には `nn.BCEWithLogitsLoss` を使う。
予測の確率が必要なときだけ，`torch.sigmoid` で logits を確率に変換する。

### 4. softmax と cross-entropy

#### softmax 関数

$C$ クラスの分類では，モデルはクラスごとに 1 つずつ，合計 $C$ 個の logits $\boldsymbol{z} = (z_1, \ldots, z_C)$ を出力する。
**softmax 関数 (softmax function)** は，logits を各クラスの確率に変換する。

$$
\mathrm{softmax}(\boldsymbol{z})_c = \frac{e^{z_c}}{\sum_{j=1}^{C}e^{z_j}}
$$

各値は 0 より大きく 1 より小さく，$C$ 個の値の和は 1 になる。
logits の大小の順序は，確率の大小の順序と一致する。
すべての logits に同じ定数 $a$ を加えても，分子と分母に共通の $e^a$ が掛かって打ち消し合うため，結果は変わらない。
この性質を使い，最大の logit を引いてから $e^{z_c}$ を計算すると，大きな logits で $e^{z_c}$ が `inf` になるのを防げる。

```python
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

z = np.array([1.0, 2.0, 3.0])
print(softmax(z))        # [0.09003057 0.24472847 0.66524096]
print(softmax(z + 100))  # [0.09003057 0.24472847 0.66524096]
```

$C = 2$ の場合，$\mathrm{softmax}(\boldsymbol{z})_2 = \dfrac{1}{1+e^{-(z_2-z_1)}} = \mathrm{sigmoid}(z_2-z_1)$ となり，第3節の sigmoid 関数と同じ形になる。

#### 多クラス分類の負の対数尤度

クラス $c$ の確率を $\mathrm{softmax}(\boldsymbol{z})_c$ とすると，正解のクラスが $y$ のサンプルの負の対数尤度は $-\log\mathrm{softmax}(\boldsymbol{z})_y$ である。
$N$ 個のサンプルについて平均したもの

$$
\mathrm{CE} = -\frac{1}{N}\sum_{i=1}^{N}\log\mathrm{softmax}(\boldsymbol{z}_i)_{y_i}
$$

を **交差エントロピー (cross-entropy)** という。
ここで，$\boldsymbol{z}_i$ は $i$ 番目のサンプルの logits，$y_i$ はその正解のクラスである。
正解のクラスの確率が 1 に近いほど損失は 0 に近づき，0 に近いほど損失は大きくなる。

PyTorch の `nn.CrossEntropyLoss` は，logits を受け取り，softmax 関数と交差エントロピーをまとめて計算する。

```python
logits = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([2])
print(nn.CrossEntropyLoss()(logits, target))  # tensor(0.4076)
```

`nn.CrossEntropyLoss` には，形 `(N, C)` の logits と，各サンプルの正解のクラス番号（0 から $C-1$ までの整数）を並べた形 `(N,)` の tensor を渡す。
クラス番号の tensor の `dtype` は `int64` にする。
この例の損失は，正解のクラス 2 の確率 0.66524096 の対数をとって符号を反転した値 $-\log0.66524\approx0.4076$ である。
`nn.CrossEntropyLoss` は内部で softmax 関数を計算するため，分類のモデルの最後には softmax 関数を入れずに，logits を出力する。

### 5. MAP 推定と正則化

#### ベイズの定理と事後分布

最尤推定では，データだけからパラメータを決める。
データが少ないときは，データを得る前に持っている知識を推定に取り入れたい場合がある。
そこで，パラメータ $\theta$ を確率変数とみなし，データを得る前の $\theta$ の分布 $p(\theta)$ を考える。
これを **事前分布 (prior distribution)** という。
データ $D$ を得た後の $\theta$ の分布 $p(\theta\mid D)$ を **事後分布 (posterior distribution)** といい，次の **ベイズの定理 (Bayes' theorem)** で求められる。

$$
p(\theta\mid D) = \frac{p(D\mid\theta)\,p(\theta)}{p(D)}
$$

$p(D\mid\theta)$ は，パラメータが $\theta$ のときにデータ $D$ が得られる確率であり，尤度 $L(\theta)$ に等しい。
分母の $p(D)$ は $\theta$ によらないので，事後分布は尤度と事前分布の積に比例する。
このことを $p(\theta\mid D)\propto L(\theta)\,p(\theta)$ と書く（$\propto$ は比例することを表す）。
事後分布を最大にする $\theta$ を求めることを **MAP 推定 (maximum a posteriori estimation)** という。
対数をとると，MAP 推定は次の式を最大にする $\theta$ を求めることになる。

$$
\ell(\theta) + \log p(\theta)
$$

最尤推定の対数尤度に，事前分布の対数 $\log p(\theta)$ が加わった形である。

#### コイン投げの MAP 推定

コインの例で，事前分布を $p(\theta) = 30\theta^2(1-\theta)^2$（$0\le\theta\le1$）とする。
この事前分布は $\theta = 0.5$ で最大になり，「表と裏の出やすさはほぼ同じだろう」という知識を表す。
データ 1，1，0 に対しては，定数を除いて次の式を最大にすればよい。

$$
\ell(\theta) + \log p(\theta) = 2\log\theta + \log(1-\theta) + 2\log\theta + 2\log(1-\theta) + \log30 = 4\log\theta + 3\log(1-\theta) + \log30
$$

$\theta$ で微分して 0 とおくと $\dfrac{4}{\theta} - \dfrac{3}{1-\theta} = 0$ より，MAP 推定値は $\theta = 4/7\approx0.571$ である。
最尤推定値 $2/3$ より，事前分布が最大になる 0.5 に近い値になっている。
この事前分布は，表が 2 回，裏が 2 回出たデータを追加で観測したのと同じ働きをしている。
データが多くなると尤度の影響が大きくなり，MAP 推定値は最尤推定値に近づく。

```python
theta = np.linspace(0.001, 0.999, 999)
prior = 30 * theta**2 * (1 - theta)**2
like = theta**2 * (1 - theta)
posterior = like * prior
posterior = posterior / np.trapezoid(posterior, theta)
print(theta[np.argmax(posterior)])  # 約 0.571

fig, ax = plt.subplots()
ax.plot(theta, prior, label="prior")
ax.plot(theta, posterior, label="posterior")
ax.set_xlabel("theta")
ax.set_ylabel("density")
ax.legend()
ax.grid(True)
fig.savefig(figure_dir / "session18_example_map.png")
plt.show()
plt.close(fig)
```

`np.trapezoid` は，第4回で扱った台形公式による数値積分である。
尤度と事前分布の積を，その積分値で割ることで，$\theta$ について積分すると 1 になる事後分布が得られる。
これは，ベイズの定理の分母 $p(D)$ で割ることに当たる。

#### 正規分布の事前分布と L2 正則化

回帰のモデルのパラメータ $\boldsymbol{w} = (w_1, \ldots, w_d)$ に，各 $w_j$ が独立に正規分布 $N(0, \tau^2)$ に従うという事前分布を考える。
これは，「パラメータは 0 から大きく離れていないだろう」という知識を表す。
この事前分布の対数は，定数を除いて $-\dfrac{1}{2\tau^2}\|\boldsymbol{w}\|^2$ である。
第2節の正規分布の誤差のモデルと組み合わせると，MAP 推定は次の式を最小にする $\boldsymbol{w}$ を求めることになる。

$$
\frac{1}{2\sigma^2}\sum_{i=1}^{N}\bigl(y_i - f(x_i;\boldsymbol{w})\bigr)^2 + \frac{1}{2\tau^2}\|\boldsymbol{w}\|^2
$$

全体に $2\sigma^2/N$ を掛けると，次の式になる。

$$
\frac{1}{N}\sum_{i=1}^{N}\bigl(y_i - f(x_i;\boldsymbol{w})\bigr)^2 + \lambda\|\boldsymbol{w}\|^2,\qquad \lambda = \frac{\sigma^2}{N\tau^2}
$$

これは，第16回の ridge 回帰の損失 $L_\lambda(\boldsymbol{w})=\frac{1}{N}\|\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y}\|^2+\lambda\|\boldsymbol{w}\|^2$ と同じ形である。
つまり，ridge 回帰は，パラメータに正規分布の事前分布を仮定した MAP 推定である。
$\tau$ が小さい，つまり「パラメータは 0 に近い」という知識を強く信じるほど $\lambda$ は大きくなり，パラメータは 0 に近づけられる。
損失にパラメータの大きさを表す項を加えて，パラメータが大きくなりすぎるのを抑えることを **正則化 (regularization)** という。
特に，$\lambda\|\boldsymbol{w}\|^2$ を加えることを **L2 正則化 (L2 regularization)** という。

PyTorch の optimizer では，`weight_decay` 引数で L2 正則化を指定できる。
`torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.2)` は，各パラメータ $\boldsymbol{w}$ の勾配に $0.2\boldsymbol{w}$ を加えてから更新する。
$\dfrac{0.2}{2}\|\boldsymbol{w}\|^2$ を $\boldsymbol{w}$ で微分すると $0.2\boldsymbol{w}$ になるので，これは損失に $\dfrac{0.2}{2}\|\boldsymbol{w}\|^2$ を加えたのと同じ更新である。

```python
x = torch.tensor([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])
y = torch.tensor([[1.0], [0.0], [2.0]])
loss_fn = nn.MSELoss()

torch.manual_seed(0)
model_a = nn.Linear(2, 1)
optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1, weight_decay=0.2)
optimizer_a.zero_grad()
loss_fn(model_a(x), y).backward()
optimizer_a.step()

torch.manual_seed(0)
model_b = nn.Linear(2, 1)
optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
optimizer_b.zero_grad()
penalty = (model_b.weight**2).sum() + (model_b.bias**2).sum()
(loss_fn(model_b(x), y) + 0.2 / 2 * penalty).backward()
optimizer_b.step()

print(torch.allclose(model_a.weight, model_b.weight))  # True
print(torch.allclose(model_a.bias, model_b.bias))      # True
```

2 つのモデルは，同じシードで作ったため同じ初期値を持つ。
`weight_decay` を指定した更新と，損失に項を加えた更新の結果が一致している。
`model.parameters()` を渡した場合，`weight_decay` はバイアスにもかかることに注意する。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第18回の第1問であれば，`exercise/exc18_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc18`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第1問の図であれば，`outputs/exc18/exc18_01.png` とします。
乱数を使う問題では，問題で指定した乱数生成器を使ってください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（6問）
1. コイン投げの結果から，表が出る確率 $\theta$ を最尤推定します。結果を `data = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0])`（1 が表）とします。

    1. $N$ 回のうち $k$ 回表が出たデータについて，対数尤度 $\ell(\theta)$ を $k$，$N$，$\theta$ の式で表してください。$\ell'(\theta) = 0$ を解き，最尤推定値が $\hat{\theta} = k/N$ となることを手計算で示してください。そのうえで，このデータの最尤推定値を求めてください。
    2. `exercise/exc18_01.py` で，`theta = np.linspace(0.001, 0.999, 999)` の各値について，尤度 $L(\theta)$ と対数尤度 $\ell(\theta)$ を計算してください。`plt.subplots(2, 1, sharex=True)` を使い，上段に尤度を，下段に対数尤度を描いて，`outputs/exc18/exc18_01.png` に保存してください。
    3. `np.argmax` を使って，尤度と対数尤度が最大になる $\theta$ をそれぞれ求め，1 の結果と比較してください。2 つが一致する理由を説明してください。
    4. 10 回すべてが表だった場合の最尤推定値を，1 の式から答えてください。その値をもとに「このコインは裏が出ない」と判断することの問題点を，データの数の観点から説明してください。

2. 正規分布に従うデータから，平均と標準偏差を最尤推定します。`rng = np.random.default_rng(0)` を使い，`x = rng.normal(0.3, 1.0, size=50)` で 50 個のデータを作ってください。

    1. `exercise/exc18_02.py` で，標準偏差を $\sigma = 1$ に固定し，`mu = np.linspace(-1, 1, 201)`（刻み幅 0.01）の各値について対数尤度を計算してください。横軸を $\mu$，縦軸を対数尤度としたグラフを描き，`outputs/exc18/exc18_02.png` に保存してください。
    2. 対数尤度が最大になる $\mu$ を求め，標本平均 `x.mean()` と比較してください。差の絶対値が，刻み幅の半分の 0.005 以下であることを確かめてください。
    3. $\mu$ を標本平均に固定し，`sigma = np.linspace(0.5, 1.5, 101)` の各値について対数尤度を計算してください。対数尤度が最大になる $\sigma$ を，`x.std()`（`ddof=0`）と `x.std(ddof=1)` のそれぞれと比較し，どちらに近いかを答えてください。その結果を，解説の $\hat{\sigma}^2$ の式と，第6回の `ddof` の説明を使って説明してください。
    4. $\mu$ の最尤推定値が，データを作るときに指定した平均 0.3 と一致しない理由を説明してください。

3. 回帰で平均二乗誤差を損失関数に使うことが，正規分布を仮定した最尤推定とみなせることを確かめます。データは第15回の基礎第3問と同じ方法で，`rng = np.random.default_rng(0)`，`x = rng.uniform(0, 1, size=40)`，`y = 2 * x + 1 + rng.normal(0, 0.1, size=40)` で作り，第16回の基礎第3問と同じく 40 個すべてを使います。モデルを $f(x) = ax + b$ とします。

    1. `exercise/exc18_03.py` で，$a$ と $b$ をそれぞれ `np.linspace(0, 4, 41)` の値で動かし，すべての組 $(a, b)$ について，平均二乗誤差と，$\sigma = 0.1$ とした正規分布の負の対数尤度を計算してください。
    2. 2 つの値が最小になる $(a, b)$ をそれぞれ求め，一致することを確かめてください。また，負の対数尤度が，解説の式のとおり「平均二乗誤差の $N/(2\sigma^2)$ 倍」に「$\frac{N}{2}\log(2\pi\sigma^2)$」を加えた値になっていることを，`np.allclose` で確かめてください。
    3. $\sigma = 0.5$ に変えると，負の対数尤度が最小になる $(a, b)$ が変わるかどうかを予想してください。実行して確かめ，その理由を説明してください。
    4. $\sigma = 0.1$ のとき，負の対数尤度の最小値は負の値になります。確率は 1 以下なので，その対数の符号を反転した値は 0 以上になるはずです。それにもかかわらず負の値になる理由を，確率密度関数の値が 1 を超えることがある点から説明してください。

4. 2 クラス分類の損失を，sigmoid 関数と負の対数尤度から計算し，PyTorch の損失関数と比べます。3 個のサンプルの logits を `z = np.array([2.0, -1.0, 0.5])`，正解ラベルを `y = np.array([1, 0, 0])` とします。

    1. 3 個のサンプルのうち，損失（負の対数尤度）が最も大きくなるものと最も小さくなるものを予想してください。理由もコメントに書いてください。
    2. `exercise/exc18_04.py` で sigmoid 関数を定義し，各サンプルのクラス 1 の確率を計算してください。サンプルごとの負の対数尤度と，その平均である binary cross-entropy を NumPy で計算し，1 の予想と比較してください。
    3. `nn.BCEWithLogitsLoss` を使って同じ値を計算し，2 の結果と一致することを確かめてください。`reduction="none"` を指定して，サンプルごとの値も比較してください。logits と正解ラベルは，`float32` の tensor に変換して渡してください。
    4. 正解ラベルを `torch.tensor(y)`（`dtype` は `int64`）のまま渡すと何が起きるかを予想してください。他の処理とは分けて実行して確かめ，正解ラベルを浮動小数点数に変換する必要がある理由を説明してください。

5. 多クラス分類の損失を，softmax 関数と負の対数尤度から計算し，PyTorch の損失関数と比べます。2 個のサンプル，3 クラスの logits を `Z = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.0]])`，正解のクラス番号を `labels = np.array([0, 2])` とします。

    1. 2 個のサンプルのうち，損失が大きくなるのはどちらかを予想してください。
    2. `exercise/exc18_05.py` で，各行の最大値を引いてから softmax 関数を計算する関数を定義してください。各行の確率の和が 1 になることを確かめてください。各サンプルの負の対数尤度と，その平均である交差エントロピーを NumPy で計算し，1 の予想と比較してください。
    3. `nn.CrossEntropyLoss` に，logits（形 `(2, 3)`，`dtype` は `float32`）と正解のクラス番号（形 `(2,)`，`dtype` は `int64`）を渡し，2 の結果と一致することを確かめてください。
    4. 3 で作った logits の tensor に `torch.softmax(..., dim=1)` を適用して確率を求め，logits の代わりにその確率を `nn.CrossEntropyLoss` に渡すと，損失の値が変わることを確かめてください。分類のモデルの最後に softmax 関数を入れてはいけない理由を説明してください。また，すべての logits に 100 を加えても損失が変わらないことを確かめ，その理由を softmax 関数の式を使って説明してください。

6. 事前分布を使って，コインの表が出る確率を MAP 推定します。基礎第1問と同じデータを使い，事前分布を $p(\theta) = 12\theta(1-\theta)^2$（$0\le\theta\le1$）とします。

    1. $\int_0^1 12\theta(1-\theta)^2\,d\theta = 1$ となることと，$p(\theta)$ が最大になる $\theta$ を手計算してください。この事前分布が，コインについてどのような知識を表しているかを説明してください。
    2. 対数尤度と事前分布の対数の和を $\theta$ で微分して 0 とおき，MAP 推定値を手計算してください。
    3. `exercise/exc18_06.py` で，`theta = np.linspace(0.001, 0.999, 999)` の各値について，尤度，事前分布，事後分布を計算してください。尤度と，尤度と事前分布の積は，それぞれ `np.trapezoid` で求めた積分値で割り，積分が 1 になるようにしてください。3 つを 1 枚の図に描き，`outputs/exc18/exc18_06.png` に保存してください。
    4. 事後分布が最大になる $\theta$ を求め，2 の手計算の結果，および基礎第1問の最尤推定値と比較してください。MAP 推定値が最尤推定値からどちらへずれたかを答え，その理由を事前分布の形を使って説明してください。

### 発展レベル（3問）
1. 事前分布の強さとデータの数によって，MAP 推定値がどう変わるかを調べます。事前分布を $p(\theta)\propto\theta^{a}(1-\theta)^{2a}$ とします（$a = 1$ のとき，基礎第6問の事前分布になります）。

    1. $N$ 回中 $k$ 回表が出たデータに対する MAP 推定値を，$k$，$N$，$a$ の式で手計算してください。
    2. `exercise/exc18_07.py` で，基礎第1問のデータについて，$a = 1, 5, 20$ の MAP 推定値を数値計算で求め，1 の式の値と比較してください。3 つの事後分布を，基礎第6問と同様に積分が 1 になるようにして 1 枚の図に描き，`outputs/exc18/exc18_07.png` に保存してください。
    3. `rng = np.random.default_rng(0)` を使い，`(rng.random(1000) < 0.5).astype(int)` で 1000 回分のコイン投げの結果を作ってください。このデータについて，最尤推定値と，$a = 1, 5, 20$ の MAP 推定値を求めてください。
    4. 2 と 3 の結果を比べ，事前分布を強くしたとき（$a$ を大きくしたとき）と，データの数を増やしたときに，MAP 推定値がそれぞれどう変わるかを説明してください。

2. ridge 回帰の正則化の強さによって，求まるパラメータと予測曲線がどう変わるかを調べます。損失は，第16回の解説と同じく平均二乗誤差に $\lambda\|\boldsymbol{w}\|^2$ を加えた $L_\lambda(\boldsymbol{w})=\frac{1}{N}\|\boldsymbol{A}\boldsymbol{w}-\boldsymbol{y}\|^2+\lambda\|\boldsymbol{w}\|^2$ とし，その最小解を正規方程式 $(\boldsymbol{A}^\top\boldsymbol{A}+N\lambda\boldsymbol{I})\boldsymbol{w}=\boldsymbol{A}^\top\boldsymbol{y}$ を `np.linalg.solve` で解いて求めます。

    1. `exercise/exc18_08.py` で，第16回の基礎第3問のデータ（基礎第3問と同じ 40 個）について，$\lambda = 0, 0.1, 1$ の解 $\boldsymbol{w} = (a, b)^\top$ を求めてください。$\boldsymbol{A}$ の各行は $(x_i, 1)$ とします。各解のノルム $\|\boldsymbol{w}\|$ と平均二乗誤差も表示し，$\lambda$ を大きくすると解がどう変わるかを答えてください。
    2. 次に，9 次の多項式 $f(x) = \sum_{j=0}^{9}w_jx^j$ を 10 個のデータに当てはめます。`rng = np.random.default_rng(1)` を使い，train のデータを `x_train = np.linspace(0, 1, 10)`，`y_train = np.sin(2 * np.pi * x_train) + rng.normal(0, 0.2, size=10)` で作ってください。続けて，同じ `rng` で validation のデータを `x_val = rng.uniform(0, 1, size=50)`，`y_val = np.sin(2 * np.pi * x_val) + rng.normal(0, 0.2, size=50)` で作ってください。各行が $(1, x_i, x_i^2, \ldots, x_i^9)$ の行列 $\boldsymbol{A}$ は，`np.vander(x_train, 10, increasing=True)` で作れます。
    3. $\lambda = 0, 10^{-6}, 10^{-3}, 10^{-1}$ のそれぞれについて解を求め，train と validation の平均二乗誤差を表示してください。$0\le x\le1$ の範囲の予測曲線を train のデータと重ねて 1 枚の図に描き，`outputs/exc18/exc18_08.png` に保存してください。縦軸の表示範囲は `ax.set_ylim(-2, 2)` としてください。
    4. validation の平均二乗誤差が最も小さい $\lambda$ を答えてください。$\lambda$ が小さすぎる場合と大きすぎる場合の予測曲線の問題を，事前分布 $N(0, \tau^2)$ の $\tau$ の大きさと結びつけて説明してください。

3. sigmoid 関数や softmax 関数で確率を求めてから対数をとると，値によっては計算が破綻することを確かめます。

    1. logits を `z = torch.tensor([-200.0, 0.0, 200.0])`，正解ラベルを `y = torch.tensor([1.0, 1.0, 0.0])` とします。`exercise/exc18_09.py` で，次の 3 つの方法でサンプルごとの損失を計算して表示してください。

        - `torch.sigmoid(z)` で確率を求めてから，binary cross-entropy の式で計算する方法
        - `torch.sigmoid(z)` で求めた確率を `nn.BCELoss(reduction="none")` に渡す方法
        - logits を `nn.BCEWithLogitsLoss(reduction="none")` に渡す方法

    2. 1 つ目と 3 つ目のサンプルの損失の正しい値を，$-\log\mathrm{sigmoid}(z) = \log(1+e^{-z})$ を使って手計算してください（近似値でかまいません）。1 の 3 つの方法のうち，どれが正しい値を与えるかを答えてください。正しい値を与えない方法については，その理由を `torch.sigmoid(z)` の表示と [`nn.BCELoss` の説明](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) を使って説明してください。
    3. logits を `np.array([1000.0, 1001.0, 1002.0])` として，最大値を引かずに `np.exp` で softmax 関数を計算した結果と，最大値を引いてから計算した結果を表示してください。前者で `nan` が現れる理由を説明してください。
    4. 3 の logits について，正解のクラスを 0 としたときの損失 $-\log\mathrm{softmax}(\boldsymbol{z})_0 = -z_0 + \log\sum_j e^{z_j}$ を，`torch.logsumexp` を使って計算してください。その値が，`nn.CrossEntropyLoss` で計算した値と一致することを確かめてください。

## 詰まったときに見る資料
- [`../textbook/markdown/ch06b-basics-of-probability-statistics.md`](../textbook/markdown/ch06b-basics-of-probability-statistics.md)
- [`06-probability.md`](06-probability.md)
- [PyTorch: BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
