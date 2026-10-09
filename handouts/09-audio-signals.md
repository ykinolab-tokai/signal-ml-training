# 第09回 音響信号

## この回の目標

- 音を file として読み込み，波形，サンプリング周波数，長さを確認する．
- 音を保存・再読込し，波形と保存形式による誤差を確認する．
- Fourier 変換によって振幅スペクトルを可視化する．
- STFT とメルスペクトログラムを計算し，時間周波数表現として可視化する．

## 解説

### 0. 準備

この回では，音信号の読み書きに `soundfile`，再生に `sounddevice`，STFT・メルスペクトログラムに `librosa` を用いる．
- [soundfile documentation](https://python-soundfile.readthedocs.io/)
- [sounddevice documentation](https://python-sounddevice.readthedocs.io/)
- [librosa documentation](https://librosa.org/doc/latest/index.html)

Pythonパッケージの準備・更新は教材repoで行う。演習は作業repoで教材側の `.venv` を有効にして実行する。手順は [共有Python環境](README.md#python環境の準備更新と演習の実行)を参照する。
保存・再読込・周波数解析は音声デバイスなしで実施できる。
再生・収録の節は，対応する機器が使える場合の任意の例であり，基礎演習の完了条件には含めない。
Linuxで `sounddevice` がPortAudio不足を報告する場合は，担当者と環境を確認してから必要なシステムライブラリを用意する。これはPython環境の同期とは別の操作である。

```bash
sudo apt update
sudo apt install -y libportaudio2 pulseaudio-utils alsa-utils libasound2-plugins libsndfile1 ffmpeg
```

以下の「音の保存」から「メルスペクトログラム」までのPythonコードは，同じ `.py` ファイルへ順に追記する。
作業場所は作業repoのルートとし，入力・保存先は各コードの `Path` で指定する。

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
x = 0.2 * np.sin(2 * np.pi * 440 * t)

sf.write(output_dir / "09_sine_440hz.wav", x, fs, subtype="PCM_16")
```

#### 音の読み込み

ファイルに保存されている音信号を読み込むには， `soundfile.read` を利用する．
この関数は，音信号を NumPy 配列として返すとともに，そのサンプリング周波数も返す．
```python
from pathlib import Path

import numpy as np
import soundfile as sf

audio_path = Path("outputs/audio/09_sine_440hz.wav")  # 直前の例で保存したファイル

data, sr = sf.read(audio_path, dtype="float32")

print("shape:", data.shape)
print("dtype:", data.dtype)
print("samplerate:", sr)
print("duration [s]:", len(data) / sr)
mono = data.mean(axis=1) if data.ndim == 2 else data

import matplotlib.pyplot as plt
figure_dir = Path("outputs/figures")
figure_dir.mkdir(parents=True, exist_ok=True)
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

#### 周波数と音名の対応

12平均律ではA4を440 Hzとし、半音上がるごとに周波数が $2^{1/12}$ 倍になる。
A4から半音で $m$ 個離れた音の周波数は $F=440\times2^{m/12}$ [Hz] である（下の音は $m<0$）。
音名は C, C♯, D, D♯, E, F, F♯, G, G♯, A, A♯, B の順で、Bの次のCでオクターブ番号が増える。
C4, D4, E4, F4, G4, A4, B4 は、それぞれ $m=-9,-7,-5,-4,-2,0,2$ に対応する。

スペクトルの正のピーク周波数 $F>0$ からは $m=12\log_2(F/440)$ を計算し、最も近い整数と比較する。
ピアノ音には基音の整数倍付近に倍音のピークも現れるため、各ピークをそのまま別の音名と判断しない。
基音候補と倍音の対応も確認して、含まれる音の候補を挙げる。

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
`librosa.amplitude_to_db` は，スペクトルの振幅を対数スケールに変換する関数である。
上のコードでは `ref=np.max` により最大振幅を0 dBとし，`center=True`（既定）の端点ゼロ埋めを使う。
この窓指定は周期形のHann窓であり，上で示した対称形（分母 $N-1$）とは端の扱いが異なる。

#### メルスペクトログラム

スペクトログラムは，線形（等間隔）な周波数軸を持つ時間周波数表現である．
一方，人間の聴覚は，低い周波数の違いには敏感で，高い周波数の違いには鈍感という，周波数に関して非線形な性質を持つ．
**メル尺度 (mel scale)** は，このような知覚特性を加味した周波数尺度である．
この回ではHTK方式のメル尺度を使う。周波数 $F$ [Hz] からメル値 $M$ への変換は次式で与えられる。
librosaの既定はSlaney方式なので，計算と表示の両方に `htk=True` を指定する。
フィルタの面積規格化を表す `norm` とメル尺度の選択は別の設定である。

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
    htk=True,
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
    htk=True,
)
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.savefig(figure_dir / "session09_mel_spectrogram.png", dpi=150)
plt.close()
```

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

`scripts/session09_audio.py` に実装し，結果と説明はコードコメントまたは既存の結果ファイルに残す。

### 基礎レベル
1. 振幅0.2，440 Hz，1秒，$F_s=16000$ Hzの正弦波を作り，`outputs/audio/09_sine_440hz.wav` へ `subtype="PCM_16"` で保存する。同じファイルを読み直し，shape `(16000,)`，標本化周波数，時間長，元の浮動小数点配列との最大絶対誤差を確認する。量子化幅 $1/32768$ と比較する。
2. 読み込んだ配列から上の例に沿って振幅スペクトルを保存し，440 Hz付近のピークを確認する。ステレオを扱う場合だけチャネル平均でモノラル化する。
3. $x(t)=0.2\sin(2\pi(f_0t+kt^2/2))$，$f_0=100$ Hz，$f_1=4000$ Hz，$T=2$ 秒，$k=(f_1-f_0)/T$ のチャープを同じ $F_s$ で生成する。Hann窓，`n_fft=1024`，`hop_length=512`，`center=True` でSTFTを計算し，時間・周波数軸付きで保存する。
4. 同じチャープとSTFT条件で80帯域のメルスペクトログラムを作る。計算・`specshow` とも `htk=True` にする。各出力shapeと軸の意味，STFTとの見え方の違いをコードコメントまたは既存の結果ファイルに記録する。メル軸の配置は非線形だが，`specshow(y_axis="mel")` の目盛ラベルはHzである。

### 発展レベル（1項目を選択）
1. 教材repoからコピーした `data/piano.wav` のスペクトルのピークを調べ，含まれる音の候補を挙げる。周波数と音名の対応を根拠にする。
2. チャープの窓長を1024，256，64と変え，ホップ長をその半分にして時間・周波数分解能を比較する。
3. **逆STFTの足場。** 長さ128の矩形窓とHann窓を，ホップ長128または64で重ねたときの窓の二乗和を描く。ゼロになる点があるか調べる。その後seed=0の1024点のガウス雑音を `center=True` でSTFTし，同じ窓・ホップ長，`length=1024` の逆STFTで最大誤差を比較する。有限信号の端点を含め，窓の二乗和が非零であることが再構成に必要な理由を説明する。
4. 再生機器が使える場合，振幅0.2の100/200 Hzと4000/4100 Hzの音を保存して聴き比べ，HTK式でのメル値の差と対応づける。

## 確認ポイント
- 保存したファイルと読込ファイルが同じで，PCM16の丸め誤差を確認した。
- `mono`，`sr`，`figure_dir` を定義した後にスペクトルの例を実行している。
- メル尺度とSTFT条件を計算・描画で統一した。再生・録音は必須にしていない。

## 詰まったときに見る資料
- [soundfileの入出力](../textbook/markdown/ch11-introduction-to-soundfile.md)
- [librosa 0.11 melspectrogram](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html)
