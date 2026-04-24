# 第08回 音響表現とモデル：STFT・mel・1D CNN・CRNN の入口

- 対象: B4・M
- 種別: 前期発展枠 / 準固定

## この回の目標
- 波形、STFT、log-mel の shape を比較できる。
- waveform 入力の 1D CNN と log-mel 入力の CRNN の forward を通せる。
- 入力表現と model 構造の相性を、時間軸・周波数軸の扱いから説明できる。

## ミニ講義
- 音響データは同じ音でも表現を変えられる。波形は時間軸だけを持つ 1 次元列、STFT や log-mel は時間と周波数の 2 軸を持つ表現になる。
- 1D CNN は時間方向へ畳み込む構造なので、波形の近傍パターンを扱いやすい。CRNN は 2D 畳み込みで局所的な時間周波数パターンを見た後、RNN で時間方向の並びを追う構造と考えられる。
- どの表現がよいかは「何を軸として残したいか」で変わる。shape を追うだけでなく、どの軸を model が読むのかまで把握すると設計判断しやすい。

## 演習
### 基礎レベル
1. `session08_audio_model_input_demo.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import math
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
```
2. 次のコードで波形を生成する。
```python
sr = 16000
t = torch.arange(0, sr, dtype=torch.float32) / sr
wave = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)
```
3. STFT と log-mel を次の通り計算する。
```python
stft = torch.stft(
    wave.squeeze(0),
    n_fft=512,
    hop_length=128,
    window=torch.hann_window(512),
    return_complex=True,
)
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=512,
    hop_length=128,
    n_mels=64,
)(wave)
log_mel = (mel + 1e-6).log()
```
4. `wave.shape`, `stft.shape`, `log_mel.shape` を表示する。
5. 1D CNN を次の通り定義し、`wave.unsqueeze(0)` を入力して forward を通す。
```python
cnn1d = nn.Sequential(
    nn.Conv1d(1, 4, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(8),
    nn.Flatten(),
    nn.Linear(32, 2),
)
```
6. CRNN を次の通り定義し、`log_mel.unsqueeze(0)` を入力して forward を通す。
```python
class TinyCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=64 * 4, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv(x)
        b, c, f, tt = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, tt, c * f)
        _, h = self.rnn(x)
        return self.fc(h[-1])
```
7. `log_mel.squeeze(0).numpy()` を `imshow` で描き、`outputs/figures/session08_log_mel.png` に保存する。
8. `session08_audio_model_input_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 表現ごとの shape`
   - `## 1D CNN の入力と出力`
   - `## CRNN の入力と出力`
9. `## 1D CNN の入力と出力` と `## CRNN の入力と出力` には、どの軸を時間として扱っているかを 1 行ずつ書く。

### 発展レベル
1. `session08_audio_model_input_report.md` に `## 表現と model の相性` を追加し、`waveform + 1D CNN` と `log-mel + CRNN` の組を比較する。
2. それぞれについて、次の 2 点を 1 行ずつ書く。
   - どの軸情報を直接使いやすいか
   - どの前処理が model 側の負担を減らしているか
3. 最後に、「波形をそのまま CRNN に入れない理由」または「log-mel をそのまま 1D CNN に入れるときに追加で整えたいこと」を 3 行以内で説明する。

## 確認ポイント
- `wave.shape` が `(1, 16000)` である。
- `cnn1d` と `TinyCRNN` の logits shape がどちらも `(1, 2)` である。
- `session08_log_mel.png` が保存されている。
- report に、shape の列挙だけでなく、軸の意味と表現と model の相性に関する説明がある。

## 詰まったときに見る資料
- [`../autumn/21-fourier-and-stft-basics.md`](../autumn/21-fourier-and-stft-basics.md)
- [`../autumn/23-audio-data-basics.md`](../autumn/23-audio-data-basics.md)
