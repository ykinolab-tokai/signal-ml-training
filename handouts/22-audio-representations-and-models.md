# 第22回 音響表現とモデル：STFT・mel・1D CNN・CRNN の入口

- 対象: B4・M
- 種別: 前期発展枠 / 準固定

## この回の目標
- 波形、STFT、log-mel の shape を比較できる。
- waveform 入力の 1D CNN と log-mel 入力の CRNN の forward を通せる。
- 入力表現と model 構造の相性を、時間軸・周波数軸の扱いから説明できる。

## 解説
- 音響データは同じ音でも表現を変えられる。波形は時間軸だけを持つ 1 次元列、STFT や log-mel は時間と周波数の 2 軸を持つ表現になる。
- 1D CNN は時間方向へ畳み込む構造なので、波形の近傍パターンを扱いやすい。CRNN は 2D 畳み込みで局所的な時間周波数パターンを見た後、RNN で時間方向の並びを追う構造と考えられる。
- どの表現がよいかは「何を軸として残したいか」で変わる。shape を追うだけでなく、どの軸を model が読むのかまで把握すると設計判断しやすい。

### 波形と時間周波数表現の条件

第1回のPython環境で、次の準備コードを実行する。サンプリング周波数は16000 Hz、時間は0秒以上1秒未満である。

```python
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
torch.manual_seed(8)
sr = 16000
t = torch.arange(sr, dtype=torch.float32) / sr
wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
stft = torch.stft(
    wave, n_fft=512, hop_length=128, win_length=512,
    window=torch.hann_window(512), center=True,
    pad_mode="reflect", return_complex=True,
)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=512, win_length=512, hop_length=128,
    n_mels=64, window_fn=torch.hann_window, power=2.0,
    center=True, pad_mode="reflect", mel_scale="htk", norm=None,
)
mel = mel_transform(wave)
log_mel = torch.log(mel + 1e-6)
print(wave.shape, stft.shape, log_mel.shape)
```

出力形状は順に `(1, 16000)`, `(1, 257, 126)`, `(1, 64, 126)` となる。
STFTは複素振幅、melはパワー、log-melはその自然対数であり、dBとは区別する。
`center=True` ではフレーム中心の時刻は `フレーム番号 * 128 / sr` 秒となる。
図の横軸はこの時刻、縦軸はメル帯域番号 `0`〜`63` とし、単なる帯域番号をHzと表示しない。

### 1D CNNとTinyCRNNの仕様

1D CNNは `nn.Sequential` で次の順につなぐ。
`nn.Conv1d(1, 4, kernel_size=5, padding=2)` → `nn.ReLU()` → `nn.AdaptiveAvgPool1d(8)` → `nn.Flatten()` → `nn.Linear(32, 2)`。
`wave.unsqueeze(1)` を入力すると、軸は `(バッチ, チャネル, 時間)`、形状は `(1, 1, 16000)` になる。

`TinyCRNN(nn.Module)` は `__init__` で `super().__init__()` を呼び、次の3層を属性として登録する。

| 属性 | 層 |
| --- | --- |
| `conv` | `nn.Conv2d(1, 4, kernel_size=3, padding=1)` |
| `rnn` | `nn.GRU(input_size=4 * 64, hidden_size=8, batch_first=True)` |
| `head` | `nn.Linear(8, 2)` |

入力は `log_mel.unsqueeze(1)` とし、形状を `(B, 1, 64, T)` にする。
`B` はバッチ数、`T` はフレーム数で、今回はそれぞれ1と126である。
`forward(self, x)` では受け取った `x` に `conv` と `torch.relu` を適用する。
得られる `(B, 4, 64, T)` を `permute(0, 3, 1, 2).reshape(B, T, 4 * 64)` で `(バッチ, 時間, 特徴量)` に並べ替える。
`sequence, hidden = self.rnn(features)` とし、`sequence[:, -1, :]` を `head` に渡す。
両モデルを `eval()` にし、`torch.no_grad()` 内で出力形状を比較する。今回は学習せず、logitsをクラス確率とは解釈しない。

## 演習
### 基礎レベル
1. `session22_audio_model_input_demo.py` を作成し、1 秒の 440 Hz 波形から waveform, STFT, log-mel を作る。`wave.shape`, `stft.shape`, `log_mel.shape` を表示する。
2. waveform を入力する 1D CNN と、log-mel を入力する `TinyCRNN` を実装し、それぞれ logits shape が `(1, 2)` になることを確認する。
3. log-mel を `outputs/figures/session22_log_mel.png` に保存する。図には時間方向と周波数方向が分かるように軸ラベルを付ける。
4. `session22_audio_model_input_report.md` に `## 表現ごとの shape`, `## 1D CNN の入力と出力`, `## CRNN の入力と出力` を書き、各 model がどの軸を時間として扱っているかを説明する。

### 発展レベル
1. `session22_audio_model_input_report.md` に `## 表現と model の相性` を追加し、`waveform + 1D CNN` と `log-mel + CRNN` を、直接使いやすい軸情報と前処理の負担の観点で比較する。
2. 波形をそのまま CRNN に入れる場合、または log-mel を 1D CNN に入れる場合に、入力 shape をどう整える必要があるかを 3 行以内で説明する。

## 確認ポイント
- `wave.shape` が `(1, 16000)` である。
- `cnn1d` と `TinyCRNN` の logits shape がどちらも `(1, 2)` である。
- `session22_log_mel.png` が保存されている。
- report に、shape の列挙だけでなく、軸の意味と表現と model の相性に関する説明がある。

## 詰まったときに見る資料
- [`09-audio-signals.md`](09-audio-signals.md): STFTとメルスペクトログラム
- PyTorch docs: [MelSpectrogram](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.transforms.MelSpectrogram.html), [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
