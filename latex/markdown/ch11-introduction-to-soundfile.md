<!-- Re-authored after migration because the original notebook content was too thin for self-study. -->

# PySoundFile 入門

## この章の目的

音響信号処理では，波形データを NumPy 配列として読み書きできることが出発点になります．
PySoundFile はそのための軽量なライブラリであり，

- WAV や FLAC などの音声ファイルを読む
- NumPy 配列として波形を扱う
- 加工結果を音声ファイルとして保存する

という基本作業をシンプルに行えます．

## PySoundFile とは何か

PySoundFile は libsndfile を土台にした Python 用の音響ファイル入出力ライブラリです．
WAV，FLAC，OGG など複数の形式に対応し，音声データを NumPy 配列として返します．

詳細は [公式ドキュメント](https://pysoundfile.readthedocs.io/en/latest/) を参照してください．

## インストール

通常は `pip` でインストールできます．

```bash
pip install soundfile
```

WSL や Linux 環境で libsndfile が不足している場合は，先にシステム側のライブラリを入れます．

```bash
sudo apt update
sudo apt install libsndfile1
```

```python
import soundfile as sf
```

## 音声を読む

```python
data, samplerate = sf.read("path/to/audio.wav")
```

ここで `data` は NumPy 配列，`samplerate` はサンプリング周波数です．

## モノラルとステレオの配列形状

PySoundFile が返す配列形状はチャネル数で変わります．

- モノラル: `(num_samples,)`
- ステレオ: `(num_samples, 2)`
- 一般の多チャネル: `(num_samples, num_channels)`

確認の基本は次です．

```python
print(data.shape)
print(data.dtype)
print(samplerate)
```

## dtype と振幅範囲

既定では浮動小数点配列として読むことが多いですが，整数型を指定することもできます．

```python
data_int16, samplerate = sf.read("path/to/audio.wav", dtype="int16")
```

## 一部分だけ読む

```python
seconds = 3
frames = 16000 * seconds
segment, sr = sf.read("path/to/audio.wav", frames=frames)
```

途中から読むときは `start` を使います．

```python
start = 16000 * 5
frames = 16000 * 2
segment, sr = sf.read("path/to/audio.wav", start=start, frames=frames)
```

## 音声を書き出す

```python
sf.write("path/to/output.wav", data, samplerate)
```

例えば FLAC として保存するなら

```python
sf.write("denoised.flac", denoised, samplerate)
```

です．

## Python で最小の読み書きと検証を行う

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

input_path = "path/to/audio.wav"
output_dir = Path("outputs/ch11")
output_dir.mkdir(parents=True, exist_ok=True)

data, sr = sf.read(input_path, dtype="float32")

if data.ndim == 2:
    mono = data.mean(axis=1)
else:
    mono = data

segment = mono[:sr]
sf.write(output_dir / "segment.wav", segment, sr)

t = np.arange(len(segment)) / sr
plt.figure(figsize=(8, 3))
plt.plot(t, segment, linewidth=1)
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.tight_layout()
plt.savefig(output_dir / "segment_waveform.png", dpi=150)
plt.close()
```

## 情報だけ見たいときは SoundFile オブジェクトを使う

```python
with sf.SoundFile("path/to/audio.wav") as f:
    print(f.samplerate)
    print(f.channels)
    print(len(f))
```

## この章で押さえるべき点

- PySoundFile は音声ファイルを NumPy 配列として読み書きするための基本ライブラリである
- `sf.read` は波形配列とサンプリング周波数を返す
- モノラルは 1 次元，ステレオは `(num_samples, 2)` のような 2 次元配列になる
- `start` や `frames` を使うと必要な区間だけ読める
- 保存時は dtype，shape，samplerate を意識して検証する
