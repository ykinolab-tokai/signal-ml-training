# 第23回 音響データの基礎：波形・spectrogram・mel・簡単な augmentation

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 1 秒の合成波形を作り、wav と特徴量画像を保存できる。
- spectrogram と log-mel の shape を確認できる。
- time shift が時間周波数表現のどこに効き、どこに効きにくいかを説明できる。

## ミニ講義
- 波形は時間方向の列で、spectrogram や mel は時間と周波数の 2 軸を持つ表現である。shape を追うと、どの軸が増減しているか見やすい。
- mel は周波数軸を人の知覚に近い刻みでまとめた表現で、log を取ると弱い成分も見やすくなる。
- time shift は波形を時間方向へずらす augmentation であり、音の高さそのものを変える操作ではない。そのため、時間位置は変わっても卓越周波数は大きく変わりにくい。

## 演習
### 基礎レベル
1. `session23_audio_data_basics.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import math
import matplotlib.pyplot as plt
import torch
import torchaudio

out_dir = Path("outputs/audio")
out_dir.mkdir(parents=True, exist_ok=True)
```
2. 次のコードで 1 秒の合成波形を作る。
```python
sr = 16000
t = torch.arange(0, sr, dtype=torch.float32) / sr
wave = (torch.sin(2 * math.pi * 440 * t) + 0.5 * torch.sin(2 * math.pi * 660 * t)).unsqueeze(0)
```
3. `wave.shape`, `wave.min()`, `wave.max()` を表示する。
4. `torchaudio.save(out_dir / "session23_original.wav", wave, sr)` で元波形を保存する。
5. spectrogram と mel を次の通り計算する。
```python
spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128)(wave)
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=512,
    hop_length=128,
    n_mels=64,
)(wave)
log_mel = (mel + 1e-6).log()
```
6. augmentation として次の time shift を適用し、保存する。
```python
shifted = torch.roll(wave, shifts=800, dims=1)
torchaudio.save(out_dir / "session23_shifted.wav", shifted, sr)
```
7. `log_mel.squeeze(0).numpy()` を `imshow` で描き、`outputs/audio/session23_log_mel.png` に保存する。
8. `session23_audio_data_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 波形と shape`
   - `## 特徴量設定`
   - `## augmentation の内容`
9. `## augmentation の内容` には `800 sample の time shift` を適用したことを書く。

### 発展レベル
1. `shifted` に対しても同じ設定で `log_mel_shifted` を計算し、`outputs/audio/session23_shifted_log_mel.png` に保存する。
2. `session23_audio_data_report.md` に `## time shift 前後の比較` を追加し、元の log-mel と shifted 後の log-mel で何が変わり、何が変わりにくいかを 3 行以内で書く。
3. そのうえで、「time shift が周波数そのものを大きく変えにくい理由」を 2 行以内で説明する。

## 確認ポイント
- `wave.shape` が `(1, 16000)` である。
- `outputs/audio/session23_original.wav` と `outputs/audio/session23_shifted.wav` が保存されている。
- `outputs/audio/session23_log_mel.png` が保存されている。
- report に、time shift が時間側と周波数側へ与える影響の違いが書かれている。

## 詰まったときに見る資料
- [`../../latex/markdown/ch11-introduction-to-soundfile.md`](../../latex/markdown/ch11-introduction-to-soundfile.md)
- [`../../latex/markdown/ch14-basics-of-spectrum-analysis.md`](../../latex/markdown/ch14-basics-of-spectrum-analysis.md)
