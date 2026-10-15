# 第22回 音響表現とモデル

## この回の目標

波形と log-mel を音響モデルに入力し，入力表現とモデル構造の関係を説明できる。

## 解説

この回では，第9回で扱った STFT とメルスペクトログラムを PyTorch の tensor として計算し，第20回の `Dataset`・`DataLoader` と第21回の CNN を使って，音響信号を分類するモデルを作る。
音響信号は，波形のままモデルに入力することも，時間と周波数の 2 つの軸を持つ表現に変換してから入力することもできる。
どの表現をどの構造のモデルに入力するかによって，モデルが利用できる情報が変わる。
スクリプトの先頭には，次の処理を書く。

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

figure_dir = Path("outputs/exc22")
figure_dir.mkdir(parents=True, exist_ok=True)
```

`torchaudio` は，PyTorch で音響信号を扱うためのライブラリであり，授業用の環境定義に含まれる。
以下のコード例は，このコードに続けて書いて実行する。

### 1. 音響信号の tensor 表現

#### 波形の shape

第9回で見たように，計算機上の音響信号は，サンプリング周波数 $F_s$ [Hz] で標本化した振幅の列 $x[n]$ である。
1 秒の音は $F_s$ 個のサンプルからなり，$F_s = 16000$ Hz なら 16000 個の数が並ぶ。
PyTorch で音響信号を扱うときは，形 `(チャネル数, サンプル数)` の tensor を使うのが慣例である。
複数の音をまとめたバッチは，形 `(バッチサイズ, チャネル数, サンプル数)` の 3 次元の tensor になる。
この並びは，第21回の画像の形 `(batch, channels, H, W)` から高さの軸を除いたものにあたる。

```python
fs = 16000
t = torch.arange(int(0.5 * fs)) / fs             # 0.5 秒分の時刻
wave = 0.5 * torch.sin(2 * torch.pi * 1000 * t)  # 1000 Hz の正弦波
print(wave.shape, wave.dtype)                    # torch.Size([8000]) torch.float32

stereo = np.zeros((8000, 2), dtype=np.float32)   # soundfile と同じ (サンプル数, チャネル数)
x = torch.from_numpy(stereo.T)                   # (チャネル数, サンプル数)
print(x.shape)                                   # torch.Size([2, 8000])

batch = wave.reshape(1, 1, -1)                   # (batch, channels, time)
print(batch.shape)                               # torch.Size([1, 1, 8000])
```

第9回の `soundfile` や `sounddevice` は，多チャネル信号を形 `(サンプル数, チャネル数)` の NumPy 配列として扱う。
PyTorch のモデルに入力するときは，`.T` で転置して `(チャネル数, サンプル数)` にそろえる。
転置を忘れると，2 チャネルで 8000 サンプルの音が，8000 チャネルで 2 サンプルの音として扱われる。
`reshape(1, 1, -1)` の `-1` は，要素数が合うようにその軸の長さを自動で決める指定である。
サンプリング周波数は tensor に含まれないため，別の変数で管理する。

### 2. STFT とメルスペクトログラム

#### STFT の計算と shape

第9回で扱った **短時間フーリエ変換 (short-time Fourier transform, STFT)** は，信号をフレーム長 $N$，ホップ長 $H$ で区切り，窓関数を掛けた各フレームを DFT する。
`torchaudio.transforms.Spectrogram` は，STFT の結果の絶対値の 2 乗 $\lvert X[k, m]\rvert^2$（パワースペクトログラム）を計算する。
引数 `n_fft` がフレーム長 $N$（DFT の点数），`hop_length` がホップ長 $H$ であり，窓関数は既定でハン窓である。

実数の信号の DFT は対称性を持つ（第8回）ため，周波数方向には $k = 0, 1, \ldots, N/2$ の $N/2 + 1$ 個の値だけが返る。
$k$ 番目の値は，周波数 $kF_s/N$ [Hz] に対応する。
また，既定では，信号の両端に $N/2$ サンプルずつ値を補ってからフレームに区切る。
このため，長さ $T$ サンプルの信号から得られるフレーム数 $L$ は，次式で求められる。

$$
L = \left\lfloor \frac{(T + N) - N}{H} \right\rfloor + 1 = \left\lfloor \frac{T}{H} \right\rfloor + 1
$$

$\lfloor a\rfloor$ は，$a$ 以下の最大の整数（小数点以下の切り捨て）を表す。

```python
spec = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256)
power = spec(wave)
print(power.shape)                 # torch.Size([513, 32])
print(power.mean(dim=1).argmax())  # tensor(64)
```

$N = 1024$ なので，周波数方向の長さは $1024/2 + 1 = 513$ である。
$T = 8000$，$H = 256$ なので，フレーム数は $\lfloor 8000/256\rfloor + 1 = 31 + 1 = 32$ である。
周波数の間隔は $F_s/N = 16000/1024 = 15.625$ Hz なので，1000 Hz は $k = 1000/15.625 = 64$ に対応する。
`power.mean(dim=1)` は各周波数のパワーをフレームについて平均した値であり，その最大値の位置 `argmax()` が 64 になっている。

`Spectrogram` は，`nn.Linear` と同じく `nn.Module` の一種であり，一度作ったものを関数のように呼び出して使う。
入力の最後の軸を時間とみなすため，形 `(batch, channels, time)` の tensor を渡すと，形 `(batch, channels, 周波数, フレーム)` の結果を返す。
第9回の `librosa.stft` の結果の絶対値を 2 乗すると，同じ値が得られる。
ただし，両端のフレームは，信号の外側に補う値の決め方の既定値が異なるため，値が少し異なる。

#### メルスペクトログラム

第9回で扱った **メル尺度 (mel scale)** は，聴覚の感じ方に合わせた周波数の尺度であり，周波数 $F$ [Hz] は次式でメル周波数 $\mathrm{mel}(F)$ に変換される。

$$
\mathrm{mel}(F) = 2595 \log_{10}\left(1 + \frac{F}{700}\right)
$$

`torchaudio.transforms.MelSpectrogram` は，パワースペクトログラムを計算した後，`n_mels` 個の三角形のフィルタで周波数方向の値を重み付きで足し合わせる。
フィルタの位置は，メル尺度で $0$ から $\mathrm{mel}(F_s/2)$ までを $n_{\mathrm{mels}} + 1$ 等分した $n_{\mathrm{mels}} + 2$ 個の点で決まる。
これらの点に下から $0, 1, 2, \ldots$ と番号を付けると，$m$ 番目（$m = 0, 1, \ldots, n_{\mathrm{mels}} - 1$）のフィルタは，$m$ 番目の点から $m + 2$ 番目の点までの範囲で正の重みを持ち，$m + 1$ 番目の点で重みが最大になる。
点はメル尺度で等間隔なので，Hz で見ると，低い周波数ほど細かく，高い周波数ほど粗く区切られる。
出力の形は `(n_mels, フレーム数)` であり，フレーム数は `Spectrogram` と同じである。

torchaudio の既定のメル尺度は，上の式と同じである。
一方，第9回の `librosa.feature.melspectrogram` は，既定で別の定義のメル尺度とフィルタの正規化を使うため，同じ引数を指定しても値は一致しない。
librosa と同じ値が必要な場合は，`MelSpectrogram` に `norm="slaney"` と `mel_scale="slaney"` を指定する。

#### 対数変換

音のパワーは，大きい成分と小さい成分の比が非常に大きい。
そこで，第9回と同様に，パワー $P$ を **デシベル (dB)** で表した値 $10\log_{10}P$ に変換してから扱う。
`torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)` は，パワーを dB に変換し，最大値より 80 dB 以上小さい値を「最大値 − 80 dB」に置き換える。
この資料では，dB に変換したメルスペクトログラムを **log-mel** と呼ぶ。

```python
mel = torchaudio.transforms.MelSpectrogram(sample_rate=fs, n_fft=1024, hop_length=256, n_mels=40)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
log_mel = to_db(mel(wave))
print(log_mel.shape)                  # torch.Size([40, 32])
print(log_mel.max() - log_mel.min())  # tensor(80.)
print(log_mel.mean(dim=1).argmax())   # tensor(13)

fig, ax = plt.subplots()
image = ax.imshow(log_mel.numpy(), origin="lower", aspect="auto", extent=[0, 0.5, 0, 40])
ax.set_xlabel("time [s]")
ax.set_ylabel("mel band")
fig.colorbar(image, ax=ax, label="dB")
fig.savefig(figure_dir / "session22_example_log_mel.png")
plt.show()
plt.close(fig)
```

`top_db=80` により，最大値と最小値の差は 80 dB になっている。
$\mathrm{mel}(1000) \approx 1000$，$\mathrm{mel}(8000) \approx 2840$ なので，40 個のフィルタの中心は，$2840/41 \approx 69.3$ メルの間隔で並ぶ。
1000 Hz に最も近い中心は $14 \times 69.3 \approx 970$ メル（$m = 13$）であり，`argmax()` の結果と一致する。

`imshow` の `origin="lower"` は配列の 0 行目を下に描く指定であり，低い周波数が下になる。
`aspect="auto"` は図の縦横比を配列の形に合わせない指定であり，`extent=[左端, 右端, 下端, 上端]` は軸の目盛りの範囲である。
ここでは横軸を秒で表すため，右端を信号の長さ 0.5 秒にしている。
`fig.colorbar` は，色と値の対応を示すカラーバーを付ける。

### 3. 1D CNN による波形の処理

#### nn.Conv1d

第21回の `nn.Conv2d` は，画像の縦と横の 2 方向に畳み込む層であった。
波形のように 1 つの軸（時間）を持つ信号には，時間方向にだけ畳み込む `nn.Conv1d` を使う。
`nn.Conv1d(in_channels, out_channels, kernel_size, stride)` は，形 `(batch, in_channels, T)` の入力から，形 `(batch, out_channels, T')` の出力を計算する。
重みの形は `(out_channels, in_channels, kernel_size)` であり，`nn.Conv2d` と同じく，カーネルを反転しない相互相関を計算する。
カーネルの長さを $K$，ストライドを $S$，両端に補う長さ（padding）を $P$ とすると，出力の長さ $T'$ は次式で求められる。

$$
T' = \left\lfloor \frac{T + 2P - K}{S} \right\rfloor + 1
$$

```python
conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=2)
x = torch.zeros(2, 1, 20)  # (batch, channels, time)
print(conv(x).shape)       # torch.Size([2, 4, 8])
print(conv.weight.shape)   # torch.Size([4, 1, 5])
```

$T = 20$，$K = 5$，$S = 2$，$P = 0$ なので，$T' = \lfloor 15/2\rfloor + 1 = 7 + 1 = 8$ である。

#### 受容野と時間の長さ

出力の 1 つの値の計算に使われる入力の範囲を，第21回と同様に **受容野 (receptive field)** という。
畳み込み層を重ねたときの $l$ 層目の受容野の長さ $r_l$ と，隣り合う出力の入力上での間隔 $j_l$ は，$l$ 層目のカーネルの長さ $K_l$ とストライド $S_l$ から次のように求められる。
ここで，$r_0 = j_0 = 1$ とする。

$$
r_l = r_{l-1} + (K_l - 1)\,j_{l-1},\qquad j_l = j_{l-1}\,S_l
$$

例えば，`kernel_size=5, stride=2` と `kernel_size=3, stride=2` の 2 層を重ねると，次のようになる。

| 層 $l$ | $K_l$ | $S_l$ | $r_l$ [サンプル] | $j_l$ [サンプル] |
| --- | --- | --- | --- | --- |
| 1 | 5 | 2 | $1 + 4 \times 1 = 5$ | $1 \times 2 = 2$ |
| 2 | 3 | 2 | $5 + 2 \times 2 = 9$ | $2 \times 2 = 4$ |

受容野はサンプル数で数えるため，時間に直すには $F_s$ で割る。
$F_s = 16000$ Hz では，9 サンプルは $9/16000$ 秒 $\approx 0.56$ ms にすぎない。
音響信号は画像に比べて 1 つの軸に並ぶ値の数が多いため，波形を入力する 1D CNN では，長いカーネルや大きなストライドを使って受容野を広げることが多い。

#### 時間方向の平均による集約

分類では，入力の長さによらず，クラスの数だけの logits を出力したい。
そこで，最後の畳み込み層の出力 `h`（形 `(batch, channels, frames)`）を時間方向に平均して形 `(batch, channels)` のベクトルにし，`nn.Linear` に入力することが多い。
この操作を，時間方向の **global average pooling** という。
平均をとるため，入力の長さが変わっても出力の形は変わらない。
一方，平均は足す順番によらないため，フレームの順序の情報は失われる。

```python
h = torch.tensor([[[1.0, 2.0, 3.0, 6.0],
                   [0.0, 0.0, 1.0, 5.0]]])  # (batch=1, channels=2, frames=4)
print(h.mean(dim=2))                        # tensor([[3.0000, 1.5000]])
print(h.flip(dims=[2]).mean(dim=2))         # tensor([[3.0000, 1.5000]])
```

`flip(dims=[2])` は，時間の軸の順序を逆にする。
フレームの順序を逆にしても，平均は変わらない。
したがって，時間方向の平均を使うモデルが利用できるのは，受容野に収まる局所的なパターンが「どれだけ現れたか」であり，「どの順番で現れたか」ではない。

### 4. log-mel を入力とする 2D CNN と CRNN

#### log-mel を画像として扱う

log-mel は，縦の軸がメル帯域，横の軸がフレームの 2 次元配列なので，チャネル数 1 の画像とみなして第21回の `nn.Conv2d` に入力できる。
このときの入力の形は `(batch, 1, n_mels, frames)` である。
`nn.Conv2d` の受容野は時間と周波数の両方向に広がり，「ある周波数の成分が始まる」「周波数が時間とともに上がる」といった，局所的な時間周波数のパターンを捉える。
ただし，画像の縦方向の移動と異なり，log-mel の縦方向の移動は音の高さの変化であり，意味が変わることに注意する。
最後に時間と周波数の両方向に平均をとる（`h.mean(dim=(2, 3))`）と，1D CNN と同じく入力の長さによらない形 `(batch, channels)` の特徴が得られるが，パターンが現れた順序の情報は失われる。

#### RNN と GRU

系列の各時刻の値を順に読み込み，それまでの内容を **隠れ状態 (hidden state)** と呼ぶベクトルにまとめながら処理するモデルを，**再帰型ニューラルネットワーク (recurrent neural network, RNN)** という。
最も基本的な RNN は，時刻 $t$ の入力 $\boldsymbol{x}_t$ と 1 つ前の時刻の隠れ状態 $\boldsymbol{h}_{t-1}$ から，次式で隠れ状態を更新する。
ここで，$\boldsymbol{h}_0 = \boldsymbol{0}$ とする。

$$
\boldsymbol{h}_t = \tanh\left(\boldsymbol{W}_x\boldsymbol{x}_t + \boldsymbol{W}_h\boldsymbol{h}_{t-1} + \boldsymbol{b}\right),\qquad t = 1, 2, \ldots, T
$$

$\boldsymbol{W}_x$，$\boldsymbol{W}_h$，$\boldsymbol{b}$ は学習するパラメータであり，すべての時刻で同じものを使う。
$\boldsymbol{h}_t$ は $\boldsymbol{h}_{t-1}$ を通して過去のすべての入力に依存するため，最後の隠れ状態 $\boldsymbol{h}_T$ は入力の順序によって変わる。
**GRU (gated recurrent unit)** は，前の時刻の隠れ状態をどれだけ残すかを入力に応じて調整する仕組み（ゲート）を加えた RNN であり，長い系列でも学習しやすい。

PyTorch では，`nn.GRU(input_size, hidden_size, batch_first=True)` で GRU を作る。
この GRU は，形 `(batch, time, input_size)` の系列を受け取り，各時刻の隠れ状態を並べた `out`（形 `(batch, time, hidden_size)`）と，最後の時刻の隠れ状態 `h_n`（形 `(層の数, batch, hidden_size)`）を返す。
この資料では，GRU を 1 層だけ使う。
`batch_first=True` を指定しないと，入力と `out` の形は `(time, batch, …)` の順になる。

```python
torch.manual_seed(0)
gru = nn.GRU(input_size=3, hidden_size=4, batch_first=True)
seq = torch.randn(2, 5, 3)                 # (batch, time, features)
out, h_n = gru(seq)
print(out.shape, h_n.shape)                # torch.Size([2, 5, 4]) torch.Size([1, 2, 4])
print(torch.allclose(out[:, -1], h_n[0]))  # True
_, h_rev = gru(seq.flip(dims=[1]))         # 時間の順序を逆にした系列
print(torch.allclose(h_n, h_rev))          # False
```

`out[:, -1]` は最後の時刻の隠れ状態であり，`h_n[0]` と一致する。
`torch.allclose` は，2 つの tensor の値がほぼ等しいかを判定する。
時間の順序を逆にした系列では，同じ値を同じ回数だけ読み込んでいても，最後の隠れ状態が変わる。
これが，時間方向の平均との違いである。

#### CRNN の構成

**CRNN (convolutional recurrent neural network)** は，畳み込み層で局所的な時間周波数のパターンを捉えた後，その特徴をフレームの順に RNN に入力するモデルである。
log-mel を入力とする CRNN では，tensor の形を次の順に変える。

1. `nn.Conv2d` と `nn.MaxPool2d` で，形 `(batch, 1, n_mels, frames)` の入力を形 `(batch, C, F', T')` の特徴にする。
2. `permute(0, 3, 1, 2)` で，軸を `(batch, T', C, F')` の順に並べ替える。
3. `reshape(batch, T', C * F')` で，各時刻の特徴を長さ $CF'$ のベクトルにする。
4. GRU に入力し，最後の隠れ状態 `h_n[-1]`（形 `(batch, hidden_size)`）を `nn.Linear` で logits にする。

```python
class TinyCRNN(nn.Module):
    def __init__(self, n_mels=40, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(input_size=8 * (n_mels // 2), hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):           # x: (batch, 1, n_mels, frames)
        h = self.conv(x)            # (batch, 8, n_mels // 2, frames // 2)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2)   # (batch, frames // 2, 8, n_mels // 2)
        h = h.reshape(b, t, c * f)  # (batch, time, features)
        out, h_n = self.gru(h)
        return self.fc(h_n[-1])     # (batch, n_classes)

model = TinyCRNN()
print(model(torch.zeros(4, 1, 40, 32)).shape)  # torch.Size([4, 3])
print(model(torch.zeros(4, 1, 40, 50)).shape)  # torch.Size([4, 3])
```

`h.shape` は 4 つの値の組なので，`b, c, f, t = h.shape` で各軸の長さを取り出せる。
`permute` は，軸の順序を入れ替える。
`batch_first=True` の GRU は 2 番目の軸を時間とみなすため，`reshape` の前に時間の軸を 2 番目へ移す必要がある。
`permute` を省いて `reshape` だけを使うと，形は同じでも，1 つの時刻の特徴に複数の時刻の値が混ざる（基礎レベル第3問で確かめる）。
GRU の `input_size` は $C \times F'$ であり，`n_mels` と pooling の回数から決まる。
一方，時間の長さ $T'$ は GRU が隠れ状態を何回更新するかを決めるだけなので，フレーム数が 32 でも 50 でも同じモデルを使える。

### 5. 入力表現とモデル構造の関係

#### 信号全体のスペクトルと順序

信号全体の振幅スペクトルは，信号にどの周波数の成分がどれだけ含まれるかを表すが，それらが現れた順序の情報をほとんど含まない。
例えば，実数の信号の順序を逆にしても，振幅スペクトル $\lvert X[k]\rvert$ は変わらず，位相だけが変わる。

```python
x = torch.tensor([1.0, 2.0, 0.0, -1.0, 3.0, 0.5, 0.0, 0.0])
X = torch.fft.rfft(x)
X_rev = torch.fft.rfft(x.flip(dims=[0]))
print(torch.allclose(X.abs(), X_rev.abs()))  # True
print(torch.allclose(X, X_rev))              # False
```

`torch.fft.rfft` は，第9回の `np.fft.rfft` と同じ計算を tensor に対して行う。
STFT はフレームごとにスペクトルを計算するため，時間の軸を残している。
しかし，その表現を受け取るモデルが時間方向に平均をとれば，順序の情報はモデルの中で失われる。

#### どの情報を残すか

入力表現とモデル構造の組み合わせによって，モデルが利用できる情報は次のように変わる。

| 入力 | モデル | 利用できる情報 | 入力の長さ |
| --- | --- | --- | --- |
| 波形 `(batch, 1, time)` | 1D CNN + 時間方向の平均 | 受容野に収まる波形のパターンと，その現れる頻度 | 任意 |
| log-mel `(batch, 1, n_mels, frames)` | 2D CNN + 時間・周波数方向の平均 | 受容野に収まる時間周波数のパターンと，その現れる頻度 | 任意 |
| log-mel | CRNN | 局所的なパターンと，それらが現れた順序 | 任意 |
| log-mel | 畳み込み層 + `nn.Flatten` + `nn.Linear` | 局所的なパターンと，それらが現れた位置 | 学習時と同じ長さだけ |

どの組み合わせがよいかは，課題に必要な情報によって決まる。
音色の違いのように短い区間のパターンに表れる性質を手がかりにする課題では，局所的なパターンの頻度だけで区別できる場合がある。
一方，音の並ぶ順序によって答えが決まる課題では，順序を残すモデルが必要になる。
演習では，同じデータに対してこれらの組み合わせを比べ，この違いを確かめる。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第22回の第1問であれば，`exercise/exc22_01.py` とします。
各ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc22`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第1問の図であれば，`outputs/exc22/exc22_01.png` とします。
前の問題で定義した関数やクラスを使う問題では，その定義をそのファイルにも書いてください（コピーしてかまいません）。
サンプリング周波数は，すべての問題で $F_s = 16000$ Hz とします。
学習の条件の指定がない場合は，`torch.optim.Adam`（学習率 `1e-3`）と `nn.CrossEntropyLoss` を使い，バッチサイズ 32 で 15 エポック学習してください。
モデルを作る直前に `torch.manual_seed(0)` を実行し，学習用の `DataLoader` には `shuffle=True` と `generator=torch.Generator().manual_seed(0)` を指定してください。
正解率は，`model.eval()` と `torch.no_grad()` のもとで計算してください。

基本的にはプログラム (Python) を使って取り組むことを想定しています。
ただし，次の 2 種類の指示がある問題は，プログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメントに書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。
- **手計算してください**：コンピュータを使わずに計算し，途中の式と答えをコメントに書いてください。そのうえでプログラムで計算した結果と比較してください。

### 基礎レベル（5問）
1. 1 秒の正弦波から STFT と log-mel を計算し，出力の形と図の見え方を確かめます。$x[n] = 0.5\sin(2\pi\cdot440\,n/F_s)$（$n = 0, 1, \ldots, 15999$）とします。

    1. `n_fft=512, hop_length=128` の `Spectrogram` と，`n_fft=1024, hop_length=256` の `Spectrogram` の出力の形を手計算してください。また，`n_fft=512, hop_length=128, n_mels=64` の `MelSpectrogram` の出力の形を手計算してください。
    2. `exercise/exc22_01.py` で `x` を tensor として作り，1 の 3 つの変換の出力の形を表示して，1 の答えと比較してください。log-mel は，`MelSpectrogram` の出力を `AmplitudeToDB(stype="power", top_db=80)` で変換して求めてください。
    3. `n_fft=512` の STFT で 440 Hz の成分が最も大きくなる周波数のインデックス $k$ を手計算してください。また，440 Hz と 8000 Hz のメル周波数を解説の式で計算し（`np.log10` を使ってかまいません），`n_mels=64` のメルスペクトログラムで 440 Hz の成分が最も大きくなるフィルタの番号 $m$（0 から数える）を予想してください。そのうえで，各周波数のパワーをフレームについて平均した値の最大値の位置を `argmax` で求め，答えと比較してください。
    4. `n_fft=512, hop_length=128` のパワースペクトログラムを `AmplitudeToDB(stype="power", top_db=80)` で dB にしたものと log-mel を，`plt.subplots(1, 2)` で横に並べて描いてください。横軸は時間 [s] とし，カラーバーを付けて `outputs/exc22/exc22_01.png` に保存してください。
    5. 2 つの図で，440 Hz の横線が縦軸の下から何割程度の高さに現れるかを比べてください。高さが異なる理由を，メル尺度でのフィルタの並び方から説明してください。

2. 波形を入力とする 1D CNN の各層の出力の長さと受容野を，式から求めて確かめます。次の畳み込み層を使います。

    ```python
    conv = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=64, stride=16),
        nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv1d(32, 32, kernel_size=8, stride=4),
        nn.ReLU(),
    )
    ```

    1. 形 `(8, 1, 16000)` の入力に対する，3 つの `nn.Conv1d` の出力の形を手計算してください。
    2. `exercise/exc22_02.py` で，`torch.zeros(8, 1, 16000)` を `conv` の各層に順に通し，各層の出力の形を表示して 1 の答えと比較してください。`for layer in conv:` と書くと，`nn.Sequential` の各層を順に取り出せます。
    3. 3 つの `nn.Conv1d` の受容野の長さ $r_l$ と出力の間隔 $j_l$ をサンプル数で手計算し，それぞれを ms に換算してください。
    4. 受容野をプログラムで確かめます。ReLU を除いた畳み込み層だけのモデル `nn.Sequential(conv[0], conv[2], conv[4])` に，`torch.randn(1, 1, 16000, requires_grad=True)` で作った入力 `x` を通し，出力を `y` とします。`y[0, :, 30].sum()` に対して `backward()` を実行し，`x.grad[0, 0]` が 0 でない要素のインデックスを `torch.nonzero` で求めてください。その最初と最後のインデックスと，範囲の長さを表示してください。範囲の長さが 3 の $r_3$ と一致することと，範囲の先頭が $30 \times j_3$ であることを確かめてください。
    5. 最後の層の出力の 1 つの値が見ている時間の長さを，ms で答えてください。また，第4問で作る音のように 2 つの音が 0.2 秒の無音を挟んで鳴る場合に，この出力の 1 つの値が 2 つの音を同時に見られるかを答えてください。

3. CRNN の中で，tensor の形がどう変わるかを確かめます。

    1. `nn.GRU(input_size=256, hidden_size=32, batch_first=True)` に形 `(8, 31, 256)` の入力を与えたときの，`out` と `h_n` の形を予想してください。
    2. `exercise/exc22_03.py` で，`torch.manual_seed(0)` を実行してから 1 の GRU と `torch.randn(8, 31, 256)` を作り，1 の予想を確かめてください。また，`out[:, -1]` と `h_n[0]` が一致することと，時間の順序を逆にした入力に対する `h_n` が元の `h_n` と一致しないことを，`torch.allclose` で確かめてください。
    3. 次の構成の CRNN を，`nn.Module` を継承したクラス `CRNN` として実装してください。

        - 畳み込み部分：`nn.Conv2d(1, 16, kernel_size=3, padding=1)`，`nn.ReLU()`，`nn.MaxPool2d(2)`，`nn.Conv2d(16, 16, kernel_size=3, padding=1)`，`nn.ReLU()`，`nn.MaxPool2d(2)` をこの順に並べた `nn.Sequential`
        - 解説の `TinyCRNN` と同様に軸を並べ替えてから，1 の GRU に入力する
        - 最後の隠れ状態を `nn.Linear(32, 2)` で 2 クラスの logits にする

        形 `(8, 1, 64, 126)` の log-mel を入力したときの，畳み込み部分の出力の形，GRU に入力する系列の形，logits の形を手計算してください。そのうえで，`forward` の中で各形を表示して確かめてください。
    4. 畳み込み部分の出力 `h` に対して，`permute` を省いて `h.reshape(b, t, c * f)` とした系列の形を予想してください。正しく並べ替えた系列と比べ，形と値がそれぞれ一致するかを `shape` と `torch.equal` で確かめてください。さらに，`permute` を省いた系列の最初の時刻の先頭 5 個の値と，`h[0, 0, 0, :5]` を表示して比べてください。`permute` を省くと 1 つの時刻の特徴として何が並ぶかを説明し，この誤りがエラーにならないために気付きにくい理由を書いてください。

4. 2 つの音が続けて鳴る音を分類するデータセットを作り，信号全体のスペクトルと log-mel を比べます。各音は長さ 1 秒で，0.1〜0.4 秒に 1 つ目の音，0.6〜0.9 秒に 2 つ目の音が鳴り，その間は無音です。2 つの音の一方は 200〜1000 Hz の低い音，もう一方は 2000〜4000 Hz の高い音です。低い音→高い音の順に鳴る音（上昇）をラベル 0，高い音→低い音の順に鳴る音（下降）をラベル 1 とします。音は次の関数で作ります。

    ```python
    def make_two_tones(f_first, f_second, duration=1.0, fs=16000):
        t = torch.arange(int(fs * duration)) / fs
        first = ((t >= 0.1 * duration) & (t < 0.4 * duration)).float()
        second = ((t >= 0.6 * duration) & (t < 0.9 * duration)).float()
        return (first * torch.sin(2 * torch.pi * f_first * t)
                + second * torch.sin(2 * torch.pi * f_second * t))
    ```

    `first` は，1 つ目の音が鳴る区間で 1，それ以外で 0 となる tensor です。`.float()` は，真偽値を 1.0 と 0.0 に変換します。

    1. `exercise/exc22_04.py` で，次の仕様の `Dataset` クラス `ToneOrderDataset(n, seed, representation, duration=1.0)` を実装してください。

        - `rng = np.random.default_rng(seed)` を作り，`n` 個の音のそれぞれについて，`f_low = rng.uniform(200, 1000)`，`f_high = rng.uniform(2000, 4000)`，`label = int(rng.integers(0, 2))`，雑音 `rng.normal(0, 0.05, size=サンプル数)` の順に乱数を生成する。
        - `label` が 0 なら `make_two_tones(f_low, f_high, duration)`，1 なら `make_two_tones(f_high, f_low, duration)` で音を作り，0.5 倍して雑音を加え，形 `(n, 1, サンプル数)` の tensor に格納する。
        - `representation` が `"wave"` なら波形を，`"logmel"` なら `MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=128, n_mels=64)` と `AmplitudeToDB(stype="power", top_db=80)` で全体を一度に変換した形 `(n, 1, 64, フレーム数)` の log-mel を入力とし，`__getitem__` で `(入力, ラベル)` を返す。
        - ラベルは `torch.int64` の tensor とする。

    2. train 用（`n=256, seed=0`）と test 用（`n=256, seed=1`）のデータセットを，`"wave"` と `"logmel"` のそれぞれについて作ってください。各データセットの長さ，入力 1 つの形，`torch.bincount` によるラベルごとの個数を表示し，2 つのクラスがおおよそ同じ数だけあることを確かめてください。
    3. 雑音を加えない 2 つの音 `0.5 * make_two_tones(500, 3000)`（上昇）と `0.5 * make_two_tones(3000, 500)`（下降）について，信号全体の振幅スペクトルがどう違うかを予想してください。そのうえで，`torch.fft.rfft` で 2 つの音の振幅スペクトルを計算し，`20 * torch.log10(振幅 + 1e-12)` で dB に変換してください。周波数 [Hz] は `torch.fft.rfftfreq(16000, d=1 / 16000)` で求められます。
    4. train 用の log-mel のデータセットから最初の上昇の音と最初の下降の音を選び，その log-mel と，3 の 2 つの振幅スペクトルを重ねたグラフを，`plt.subplots(1, 3)` で 1 枚の図にまとめてください。図は `outputs/exc22/exc22_04.png` に保存してください。
    5. 信号全体の振幅スペクトルでは上昇と下降を区別できない理由と，log-mel では区別できる理由を，図を根拠に説明してください。

5. 波形を入力とする 1D CNN と，log-mel を入力とする CRNN を学習させ，上昇と下降を区別できるかを比べます。1D CNN は，第2問の `conv` の出力を時間方向に平均し，`nn.Linear(32, 2)` で 2 クラスの logits にするクラス `WaveCNN` とします。CRNN は，第3問の `CRNN` を使います。

    1. 2 つのモデルの test の正解率が，それぞれ 0.5（でたらめに答えた場合の正解率）と 1 のどちらに近くなるかを予想してください。第2問の 5 の答えと，解説の第3節と第4節を手がかりにしてください。
    2. `exercise/exc22_05.py` で，第4問の train 用のデータセットを使って，2 つのモデルを学習させてください。エポックごとの平均の損失（各バッチの損失にバッチの大きさを掛けて足し，データ数で割った値）を記録し，2 つのモデルの学習曲線を 1 枚の図に重ねて，`outputs/exc22/exc22_05.png` に保存してください。
    3. 2 つのモデルについて，train と test の正解率を表示してください。また，`time.perf_counter()` で学習にかかった時間を測り，表示してください。
    4. 1 の予想と結果を比較してください。1D CNN の train の正解率が 0.5 を少し上回っても test の正解率が 0.5 前後にとどまる理由を，時間方向の平均によって失われる情報と結びつけて説明してください。また，CRNN が上昇と下降を区別できる理由を，GRU の隠れ状態の更新の仕組みから説明してください。

### 発展レベル（3問）
1. 第5問の 2 つのモデルの差が，入力表現とモデル構造のどちらによるものかを切り分けます。CRNN と同じ log-mel を入力とし，GRU の代わりに時間と周波数の両方向の平均をとる 2D CNN を考えます。このモデル `MelCNN` は，第3問の `CRNN` と同じ畳み込み部分の出力 `h` を `h.mean(dim=(2, 3))` で形 `(batch, 16)` にし，`nn.Linear(16, 2)` で 2 クラスの logits にします。

    1. `MelCNN` の畳み込み部分の時間方向の受容野を，フレーム数で手計算してください。`nn.MaxPool2d(2)` は，カーネルの長さ 2，ストライド 2 の層として扱えます。さらに，隣り合うフレームが `hop_length` の 128 サンプルずつずれ，各フレームが `n_fft` の 512 サンプルの区間から計算されることを使って，その受容野が元の波形の何サンプル（何 ms）にあたるかを求めてください。
    2. `MelCNN` の test の正解率を予想してください。
    3. `exercise/exc22_06.py` で，`MelCNN`，`CRNN`，`WaveCNN` を，`torch.manual_seed` と `DataLoader` の `generator` の seed を 0，1，2 に変えて，それぞれ 3 回ずつ学習させてください。各回の train と test の正解率と，モデルごとの test の正解率の平均と標準偏差を表示してください。
    4. `MelCNN` と `CRNN` の入力は同じ log-mel です。3 の結果から，第5問の 2 つのモデルの差が入力表現とモデル構造のどちらによるものかを判断し，その理由を 1 の受容野の長さと結びつけて説明してください。

2. 順序の情報が局所的なパターンに現れる場合を調べます。周波数が時間とともに連続的に変わる 1 秒のチャープ信号（第9回）を，次の関数で作ります。

    ```python
    def make_chirp(f_first, f_second, duration=1.0, fs=16000):
        t = torch.arange(int(fs * duration)) / fs
        k = (f_second - f_first) / duration
        return torch.sin(2 * torch.pi * (f_first * t + 0.5 * k * t**2))
    ```

    この信号の周波数は，時刻 0 の `f_first` から時刻 `duration` の `f_second` まで一定の割合で変わります。

    1. `exercise/exc22_07.py` で，第4問の `ToneOrderDataset` を，音を作る関数を引数で受け取れるように変更し，`make_two_tones` の代わりに `make_chirp` を使えるようにしてください。周波数とラベルの決め方は第4問と同じとします。このとき，ラベル 0 は周波数が上がるチャープ，ラベル 1 は周波数が下がるチャープになります。train 用（`n=256, seed=0`）と test 用（`n=256, seed=1`）の log-mel のデータセットを作ってください。
    2. 周波数が上がるチャープと下がるチャープの log-mel を 1 つずつ横に並べて描き，`outputs/exc22/exc22_07.png` に保存してください。
    3. `MelCNN` と `CRNN` の test の正解率を予想してから，それぞれを seed 0，1，2 で学習させて確かめてください。
    4. 発展第1問の 2 つの音では区別できなかった `MelCNN` が，チャープを区別できるかを結果から答えてください。その理由を，2 の図の中で，発展第1問の 1 で求めた受容野に収まる範囲のパターンに注目して説明してください。

3. 長さの異なる音をモデルに入力できるかを調べます。

    1. 長さ 0.5 秒，1 秒，2 秒の波形（形 `(4, 1, 8000)`，`(4, 1, 16000)`，`(4, 1, 32000)`）について，`WaveCNN` の出力の形，log-mel のフレーム数，`CRNN` の出力の形を予想してください。
    2. `exercise/exc22_08.py` で，`torch.zeros` で作った 3 種類の波形を使って，1 の予想を確かめてください。log-mel の計算には，第4問と同じ変換を使ってください。
    3. `CRNN` の畳み込み部分の後に `nn.Flatten()` と `nn.Linear(16 * 16 * 31, 2)` をつないだモデルを作り，1 秒と 2 秒の log-mel を入力したときに何が起きるかを予想してください。そのうえで確かめてください。2 秒の入力は，他の処理とは分けて実行してください。
    4. 第5問と同じ条件で `CRNN` を学習させた後，`ToneOrderDataset` の `duration` を 0.5，1.5，2.0 秒にした test 用のデータセット（`n=256, seed=2`）で正解率を計算してください。学習に使っていない長さの音でも分類できるかを答え，その理由を，`CRNN` の出力の形が入力の長さによらない仕組みと結びつけて説明してください。

## 詰まったときに見る資料
- [`09-audio-signals.md`](09-audio-signals.md)：STFT，窓関数，メルスペクトログラム
- [torchaudio transforms](https://docs.pytorch.org/audio/stable/transforms.html)
- PyTorch docs: [`torch.nn.Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html), [`torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
