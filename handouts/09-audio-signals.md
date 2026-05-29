# 第09回 音響信号

- 音を file として読み込み，波形，サンプリング周波数，長さを確認する．
- 音を再生・保存し，数値配列としての信号と実際の聞こえ方を対応づける．
- Fourier 変換によって振幅スペクトルを可視化する．
- STFT とメルスペクトログラムを計算し，時間周波数表現として可視化する．
- リサンプリングによってサンプル数，時間長，周波数成分がどう変わるか確認する．

## 解説

### 0. 準備

この回では，音信号の読み書きに `soundfile`，再生に `sounddevice`，STFT・メルスペクトログラム・リサンプリングに `librosa` を用いる．
- [soundfile documentation](https://python-soundfile.readthedocs.io/)
- [sounddevice documentation](https://python-sounddevice.readthedocs.io/)
- [librosa documentation](https://librosa.org/doc/latest/index.html)

Linux 環境であれば，次のコマンドで必要なライブラリと Python パッケージをインストールできる．
```bash
sudo apt update
sudo apt install -y libportaudio2 pulseaudio-utils alsa-utils libasound2-plugins libsndfile1 ffmpeg
pip install soundfile sounddevice librosa
```
Python仮想環境を作成している場合は，仮想環境を有効にしてから `pip install` を実行する．

### 1. sounddevice を用いた音信号の再生と収録

コンピュータ上の音信号は離散時間信号として扱われる．
すなわち，音信号は，離散時間 $n$ における音の振幅（強さ）を表す信号 $x[n]$ と，
サンプリング周波数 $F_s$ [Hz] の組で表される．
振幅値は整数や浮動小数点数で表されるが，
浮動小数点数を用いる場合には $[-1, 1]$ の範囲で扱うことが多い．

音信号には，左右のスピーカー（やイヤホン）のそれぞれに対応する2つの振幅 $x_L[n], x_R[n]$ を持たせる場合がある．
このような信号を **ステレオ信号** ，または，2チャネル信号という．
**チャネル (channel)** とは，各時刻 $n$ における音信号の振幅の数のことをいう．
ステレオ信号に対して，1つの振幅 $x[n]$ だけを持つ信号は **モノラル信号** という．
また，一般に，チャネル数が2以上の信号をまとめて **多チャネル信号** という．

音の再生や収録など，スピーカーやマイクなどのデバイスを用いた処理のためのライブラリとして， `sounddevice` がある．
`sounddevice` は，音信号を NumPy 配列として扱う．
デバイスの操作は環境に依存するため，（特にWSL上では）正常に実行できないことがある．


#### 音信号の再生

音信号を再生するには， `sounddevice.play` を用いる．
音信号の再生の完了を待つには， `sounddevice.wait` を用いる．
音を再生する際には，意図しない大きな音が出ないように，
まずはコンピュータの設定で音量を小さめにしておくことをおすすめする．
問題がなければ音量を大きくすると良い．

```python
import numpy as np
import sounddevice as sd

fs = 48000
duration = 3

# 440 Hzの正弦波を作って再生
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
x = 0.2 * np.sin(2 * np.pi * 440 * t)

sd.play(x, fs)
sd.wait()
```

ライブラリを用いる場合には，公式ドキュメントを参照して，その仕様を理解することが重要である．
公式ドキュメントで `play` 関数の仕様を確認しよう．
今後も，新しいライブラリを使うときには，まずは公式ドキュメントを確認する習慣をつけると良い．

ドキュメントから分かる通り， `sounddevice` は多チャネル信号を
`(num_samples, num_channels)` の形を持つ二次元の NumPy 配列として扱う．
よって，多チャネル信号は，次のように作成・再生できる．

```python
import numpy as np
import sounddevice as sd

fs = 48000
duration = 3

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
x_left = 0.2 * np.sin(2 * np.pi * 440 * t)
x_right = 0.2 * np.sin(2 * np.pi * 880 * t)
x_stereo = np.stack([x_left, x_right], axis=-1)

sd.play(x_stereo, fs)
sd.wait()
```

`numpy.stack` 等は初めて見る人もいるかもしれないが，
これも NumPy の公式ドキュメントを見れば用途や仕様が分かるので確認しよう．

#### 音信号の収録

音信号を収録するには， `sounddevice.rec` を用いる．
この関数は，録音時間，サンプリング周波数，チャネル数などを指定して，録音した音信号を NumPy 配列として返す．
多チャネル信号の収録には，それに対応したマイクが必要になるため注意しよう．

```python
import sounddevice as sd
import matplotlib.pyplot as plt

fs = 48000          # サンプリング周波数 [Hz]
duration = 5        # 録音時間 [秒]
channels = 1        # 1: mono, 2: stereo

print("録音開始")
audio = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=channels,
    dtype="float32",
)
sd.wait()           # 録音終了まで待つ
print("録音終了")
```

### 2. soundfile を用いた音信号の読み書き

音信号をファイルとして記録したり，ファイルとして保存されている音信号を読み込んだりするには， `soundfile` が便利である．
`soundfile` も， `sounddevice` 同様に，音信号を NumPy 配列として扱う．

#### 音の保存

音信号の保存には，`soundfile.write` を利用する．
音信号のためのファイル形式には，WAV，FLAC，MP3 などがある．
ファイル形式によっては，ファイルサイズを小さくするために，
音信号の情報を削減した上で圧縮符号化することがある（**非可逆圧縮**）．
研究用途では，音信号の内容をできるだけ忠実に保存するために，
非可逆圧縮を行わないWAVなどのファイル形式を選ぶことが多い．

```python
from pathlib import Path
import numpy as np
import soundfile as sf

output_dir = Path("outputs/audio")
output_dir.mkdir(parents=True, exist_ok=True)

fs = 16000
duration = 1.0
t = np.arange(int(fs * duration)) / fs
x = 0.5 * np.sin(2 * np.pi * 440 * t)

sf.write(output_dir / "09_sine_440hz.wav", x, fs)
```

#### 音の読み込み

ファイルに保存されている音信号を読み込むには， `soundfile.read` を利用する．
この関数は，音信号を NumPy 配列として返すとともに，そのサンプリング周波数も返す．
```python
from pathlib import Path

import numpy as np
import soundfile as sf

audio_path = Path("outputs/audio/09_input.wav")

data, sr = sf.read(audio_path, dtype="float32")

print("shape:", data.shape)
print("dtype:", data.dtype)
print("samplerate:", sr)
print("duration [s]:", len(data) / sr)
```

### 3. フーリエ変換によるスペクトルの可視化

波形 $x[n]$ は時間方向の変化を表す．
一方，Fourier 変換を用いると，信号にどの周波数成分がどれだけ含まれるかを確認できる．
音信号は実数値であり，そのフーリエ変換は対称性を持つため，
正の周波数側だけを返す `numpy.fft.rfft` を使うと冗長性を排除して効率的にスペクトルを計算できる．
得られる配列の各要素がどの周波数に対応するかは， `numpy.fft.rfftfreq` を使って計算できる．

```python
segment = mono[:sr]  # 先頭 1 秒を使う
window = np.hanning(len(segment))

X = np.fft.rfft(segment * window)
freq = np.fft.rfftfreq(len(segment), d=1 / sr)
magnitude = np.abs(X)
magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))

plt.figure(figsize=(8, 3))
plt.plot(freq, magnitude_db, linewidth=0.8)
plt.xlabel("frequency [Hz]")
plt.ylabel("magnitude [dB]")
plt.xlim(0, sr / 2)
plt.tight_layout()
plt.savefig(figure_dir / "09_spectrum.png", dpi=150)
plt.close()
```

### 4. 短時間フーリエ変換

音声や音楽のように時間とともに内容が変化する信号では，
信号の周波数成分と時間変化の両方を解析することが重要である．
しかし，信号全体に対するフーリエ変換では，それに含まれる周波数成分を解析することはできるが，
信号の時間変化を解析できなくなってしまう．
この問題を解決するために，信号を短い時間区間（**フレーム**）に分割して，
各フレームに対してフーリエ変換を行う方法がある．
この方法を **短時間フーリエ変換 (short-time Fourier transform, STFT)** という．

#### 窓関数 (window function)

信号からフレームを切り出すための関数を **窓関数 (window function)** という．
最も単純な窓関数として次式で与えられる **矩形窓 (rectangular window)** がある．

$$
w[n] = \begin{cases}
1 & 0 \leq n < N \\
0 & \text{otherwise}
\end{cases}
$$

ここで， $N$ は切り出したいフレームの長さを表すパラメータであり，
**フレーム長 (frame length)** や **窓長 (window length)** と呼ばれる．
信号 $x$ から $H$ 点間隔で長さ $N$ のフレームを切り出す操作は次式で与えられる．

$$
x_m[n] = x[n + mH] w[n], \quad m \in \mathbb{Z}
$$

$H$ は，**ホップ長 (hop length)** や **フレームシフト (frame shift)** とも呼ばれるパラメータである．
$H < N$ のとき，隣り合うフレームは重複する区間を持つ．
このとき，$N - H$ は，フレームが重なる幅を表し，**オーバーラップ (overlap) 幅** などと呼ばれる．
ホップ長は，窓長に対する割合で与えることも多い．
$H = N / 4$ や $H = N / 2$ などがよく使われる．

矩形窓以外にも様々な窓関数があり，代表的なものとして次のものがある．
- ハン窓 (Hann window)

   $\displaystyle w[n] = \begin{cases}
   0.5 \left(1 - \cos\left(\frac{2\pi n}{N - 1}\right)\right), & 0 \leq n < N \\
   0, & \text{otherwise}
   \end{cases}$

- ハミング窓 (Hamming window)

   $\displaystyle w[n] = \begin{cases}
   0.54 - 0.46 \cos\left(\frac{2\pi n}{N - 1}\right), & 0 \leq n < N \\
   0, & \text{otherwise}
   \end{cases}$

#### STFT の定義

ホップ長 $H$，窓関数を $w[n]$ としたとき，
STFT は， $x_m[n]$ のフーリエ変換を各$m$について計算することで得られる．
すなわち，

$$
X[k, m] = \sum_{n=0}^{N-1} x_m[n] e^{-j 2\pi kn / N} = \sum_{n=0}^{N-1} x[n + mH] w[n] e^{-j 2\pi kn / N}
$$

で与えられる．
ここで， $k$ は周波数インデックスを表す．
STFTにおけるDFTの点数は，
上式の通りフレーム長 $N$ と一致させることが多い．

STFT によって得られる $X[k, m]$ の 絶対値 $|X[k, m]|$ やその2乗 $|X[k, m]|^2$ は，
**スペクトログラム (spectrogram)** と呼ばれる（文献によって指す対象が異なるので注意）．

Python を用いたSTFTの計算には，`librosa.stft` を用いることができる．

```python
import librosa
import librosa.display

n_fft = 1024
hop_length = 256

D = librosa.stft(
    mono,
    n_fft=n_fft,
    hop_length=hop_length,
    window="hann",
)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(8, 4))
librosa.display.specshow(
    D_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="hz",
)
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.savefig(figure_dir / "session09_stft.png", dpi=150)
plt.close()
```

音信号に含まれる高周波成分は低周波成分と比べて小さいことが多いため，
スペクトログラムを対数スケール ( $20 \log_{10}(|X|)$ [dB] ) で可視化することが多い．
`librosa.amplitude_to_db` は，スペクトルの振幅を対数スケールに変換する関数である．

#### メルスペクトログラム

スペクトログラムは，線形（等間隔）な周波数軸を持つ時間周波数表現である．
一方，人間の聴覚は，低い周波数の違いには敏感で，高い周波数の違いには鈍感という，周波数に関して非線形な性質を持つ．
**メル尺度 (mel scale)** は，このような知覚特性を加味した周波数尺度である．
周波数 $F$ [Hz] からのメル周波数 $M$ への変換は，次式で与えられる．

$$
M = 2595 \log_{10}\left(1 + \frac{F}{700}\right)
$$

時間とメル周波数を軸とするスペクトログラムを **メルスペクトログラム (mel spectrogram)** という．
メルスペクトログラムは，音声認識や音響イベント認識などの入力特徴量としてよく使われる．

メルスペクトログラムの計算には，`librosa.feature.melspectrogram` を用いることができる．

```python
import librosa
import librosa.display

n_mels = 80

mel = librosa.feature.melspectrogram(
    y=mono,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    power=2.0,
)
mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(8, 4))
librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel",
)
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.savefig(figure_dir / "session09_mel_spectrogram.png", dpi=150)
plt.close()
```

## 演習
基本的にはプログラム (Python) を使って取り組むことを想定しています．
音声 file の読み書きには `soundfile`，再生には `sounddevice`，STFT・メルスペクトログラム・リサンプリングには `librosa` を用いてください．

### 基礎レベル

1. 代表的な音律である12平均律において，
   音階（ド，ド＃，レ，レ＃，ミ，ファ，ファ＃，ソ，ソ＃，ラ，ラ＃，シ）の音の周波数は，
   ラ（A4）の音を $440$ Hz として隣り合う音の周波数の比が $2^{1/12}$ となるよう定められている
   （1オクターブ＝12音離れると周波数が2倍になる）．
   ド・レ・ミ・ファ・ソ・ラ・シの音の周波数を計算しなさい．

1. ファに対応する周波数を持つ正弦波を
   サンプリング周波数 $F_s = 16000$ Hz で 3 秒間サンプリングした信号を作り，
   保存しなさい．

1. 周波数 $100$ Hz, $200$ Hz, $4000$ Hz, $4100$ Hz の正弦波を
   サンプリング周波数 $F_s = 16000$ Hz で 5 秒間サンプリングした信号を作り，保存しなさい．
   作成した $100$ Hz の信号と $200$ Hz の信号を聴き比べなさい．
   また， $4000$ Hz の信号と $4100$ Hz の信号を聴き比べなさい．
   周波数が高いときと低いときにおける $100$ Hz の差の聞こえ方の違いを観察しなさい．

1. $100$ Hz と $200$ Hz をそれぞれメル周波数に変換し，それらの差 $M_1$ を計算しなさい．
   また， $4000$ Hz と $4100$ Hz をそれぞれメル周波数に変換し，それらの差 $M_2$ を計算しなさい．
   $4000$ Hz よりメル尺度で $M_1$ だけ大きい周波数を計算し，その周波数を持つ正弦波を 3 と同様に作り，保存しなさい．
   $4000$ Hz の音とその音の高さの差は，$100$ Hz の音と $200$ Hz の音の高さの差と同じように感じられるか？

1. $x[n] = \sin(\pi n / 8)$ に対して $N = 128$ 点の矩形窓をかけた信号 $x[n] w[n]$ を，
   横軸を$n$ として `stem` プロットしなさい．また，窓関数 $w[n]$ を折れ線グラフとして重ねてプロットしなさい．
   $N = 128$ 点のハン窓を用いた場合についても，窓をかけた信号と窓関数を同様にプロットしなさい．

1. 信号 $x(t) = \sin(2\pi (f_0t + \frac{1}{2}kt^2))$ を考える（この信号は **チャープ信号 (chirp signal)** と呼ばれる）．
   ただし，$T$ [s] を信号長とし， $k = (f_1 - f_0) / T$ とする．
   $f_0 = 100$ [Hz]，$f_1 = 4000$ [Hz]，$T = 5$ [s] として，
   $x(t)$ をサンプリング周波数 $F_s = 16000$ Hz でサンプリングし，保存しなさい．

1. 6で作成したチャープ信号に対してDFT (`numpy.fft.rfft`で良い) を施し，
   横軸を周波数 [Hz] として振幅スペクトルをプロットしなさい．

1. 6で作成したチャープ信号に対してSTFTを施し，
   横軸を時間 [s]，縦軸を周波数 [Hz] としてスペクトログラムをプロットしなさい．
   ただし，窓は $N = 1024$ 点のハン窓，ホップ長は $H = 512$ 点とする．

1. 6で作成したチャープ信号のメルスペクトログラムを，
   横軸時間 [s]，縦軸メル周波数としてプロットしなさい．
   ただし，窓は $N = 1024$ 点のハン窓，ホップ長は $H = 512$ 点，
   メル周波数ビン数は $n_\mathrm{mels} = 80$ とする．

### 発展レベル

1. `piano.wav` は，ド・レ・ミ・ファ・ソ・ラ・シのうちの3つの音からなる和音である．
この信号を周波数解析して，どの音が含まれているかを推測しなさい．

1. 基礎レベル 6 で作成したチャープ信号を，
   $N = 1024, 256, 64, 16$ 点のハン窓を用いて，
   ホップ長を $H = \frac{N}{2}$ としてSTFTし，
   スペクトログラムをそれぞれプロットしなさい．
   また，この結果から，
   時間分解能と周波数分解能の関係について考察しなさい．

1. STFTの逆変換は `librosa.istft` で計算できる．
   $1024$ 点のガウス雑音を作成し，
   以下の条件でSTFTした後に逆STFTを計算しなさい．
   再構成された信号が元の信号と一致するか確認し，
   結果を考察しなさい．

   - 条件1: $128$ 点の矩形窓，ホップ長 $128$ 点
   - 条件2: $128$ 点のハン窓，ホップ長 $128$ 点
   - 条件3: $128$ 点のハン窓，ホップ長 $64$ 点

1. 逆STFTにより元の信号を完全に再構成するための条件を示しなさい．

## 詰まったときに見る資料
- [soundfile documentation](https://python-soundfile.readthedocs.io/)
- [sounddevice documentation](https://python-sounddevice.readthedocs.io/)
- [librosa documentation](https://librosa.org/doc/latest/index.html)
- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)
