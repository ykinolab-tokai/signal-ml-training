# スペクトル解析の基礎

## なぜ周波数領域で考えるのか

時間領域で見た信号は，「いつ大きいか」は分かっても，「どのような速さで変動しているか」は必ずしも直感的ではありません．
スペクトル解析では，信号を複数の正弦波成分の重ね合わせとして見て，各周波数成分がどれだけ含まれているかを調べます．

## 正弦波は分解の基底になる

長さ $N$ の離散信号を考えるとき，

$$
\exp\left(\mathrm{j} 2\pi \frac{k}{N} n\right)
$$

の形をした複素指数関数を基底として使うと，信号を周波数成分に分解できます．

ここで

- $n = 0, 1, \ldots, N-1$ はサンプル番号
- $k = 0, 1, \ldots, N-1$ は周波数ビン番号
- $\mathrm{j}$ は虚数単位

です．

## 離散フーリエ変換

長さ $N$ の信号 $x[n]$ に対する**離散フーリエ変換 (DFT)** を

$$
X[k] = \sum_{n=0}^{N-1} x[n] \exp\left(-\mathrm{j}2\pi \frac{k}{N} n\right)
$$

と定義します．

逆変換は

$$
x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k] \exp\left(\mathrm{j}2\pi \frac{k}{N} n\right)
$$

です．

## 周波数ビンと実際の周波数

サンプリング周波数を $f_s$ Hz とすると，$k$ 番目のビンが表す周波数は

$$
f_k = \frac{k}{N} f_s
$$

です．

観測長 $N$ を長くすると周波数分解能は高くなります．

## DC 成分とナイキスト周波数

$k=0$ の成分は **DC 成分**で，信号の平均値に対応します．

離散時間信号で区別可能な最高周波数は

$$
\frac{f_s}{2}
$$

であり，これを**ナイキスト周波数**と呼びます．

これを超える成分があると**エイリアシング**が起こります．

## スペクトルを見るときの注意

### リーケージ

有限長の区間だけを切り出して DFT を取ると，1 つの周波数に集中していたはずのエネルギーが周辺ビンに広がって見えることがあります．
これを**スペクトルリーケージ**と呼びます．

### 窓関数

リーケージを和らげるために，切り出した信号に窓関数を掛けてから DFT を取ります．
代表例には Hann 窓や Hamming 窓があります．

### ゼロ埋め

信号の末尾に 0 を足して DFT 点数を増やすと，スペクトルの見た目は滑らかになります．
ただし，新しい情報が増えるわけではありません．

## Python で DFT を確かめる

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

output_dir = Path("outputs/ch14")
output_dir.mkdir(parents=True, exist_ok=True)

fs = 16000
duration = 0.064
t = np.arange(int(fs * duration)) / fs
x = 0.8 * np.sin(2 * np.pi * 440 * t)
x += 0.3 * np.sin(2 * np.pi * 1000 * t)

window = np.hanning(len(x))
xw = x * window

X = np.fft.rfft(xw)
freq = np.fft.rfftfreq(len(xw), d=1 / fs)
amp = np.abs(X)

plt.figure(figsize=(8, 3))
plt.plot(freq, amp)
plt.xlim(0, 2000)
plt.xlabel("frequency [Hz]")
plt.ylabel("magnitude")
plt.tight_layout()
plt.savefig(output_dir / "spectrum.png", dpi=150)
plt.close()
```

## この章で押さえるべき点

- DFT は信号を周波数成分へ分解する可逆変換である
- スペクトルの横軸は $f_k = kf_s/N$ で実周波数に変換して読む
- $k=0$ は平均値に対応する DC 成分である
- ナイキスト周波数を超える成分はエイリアシングを起こす
- リーケージ，窓関数，ゼロ埋めの意味を混同しない
