# 第21回 Fourier 変換と STFT の基礎

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 合成サイン波の spectrum を計算して保存できる。
- STFT を計算して時間周波数表現を保存できる。
- `n_fft` の違いが周波数分解能と図の見え方へどう関わるかを説明できる。

## ミニ講義
- Fourier 変換は、時間波形を周波数成分へ分解して見る方法である。全体を 1 回で見ると spectrum になり、時間ごとに区切って見ると STFT になる。
- spectrum は「どの周波数が含まれるか」を見るのに向き、STFT は「いつその周波数が出ているか」を見るのに向く。
- `n_fft` を変えると、どれだけ細かく周波数を分けて見るかが変わる。細かく見るほど計算窓も大きくなり、時間方向の見え方とのバランスが変わる。

## 演習
### 基礎レベル
1. `session21_fourier_stft_basics.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import math
import matplotlib.pyplot as plt
import torch

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
```
2. 次のコードで合成波形を作る。
```python
sr = 16000
t = torch.arange(0, sr, dtype=torch.float32) / sr
wave = torch.sin(2 * math.pi * 440 * t) + 0.5 * torch.sin(2 * math.pi * 880 * t)
```
3. `torch.fft.rfft` で `spec`、`freq`、`magnitude` を計算する。
4. `torch.topk(magnitude, k=2)` を使ってピーク 2 つの index を取り、その周波数を表示する。
5. spectrum 図を `outputs/figures/session21_spectrum.png` に保存する。
6. `torch.stft(..., n_fft=512, hop_length=128, return_complex=True)` で STFT を計算し、`stft.abs()` を `outputs/figures/session21_stft.png` に保存する。
7. `session21_fourier_stft_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 波形と周波数ピーク`
   - `## spectrum 図`
   - `## STFT 図`

### 発展レベル
1. `n_fft=256` の条件でも STFT を計算する。
2. `session21_fourier_stft_report.md` に `## n_fft の比較` を追加し、`256` と `512` のどちらが周波数の区別を細かく見やすいかを 2 行以内で書く。
3. そのうえで、「なぜ `n_fft` を大きくすると周波数側が細かく見えやすくなるか」を 3 行以内で説明する。
4. spectrum と STFT を見比べて、「どちらの図で何が分かり、何が分からないか」を 2 行で書く。

## 確認ポイント
- `wave.shape` が `(16000,)` である。
- ピーク周波数として 440 Hz と 880 Hz 付近が抽出される。
- `session21_spectrum.png` と `session21_stft.png` が保存されている。
- report に、`n_fft` の違いと spectrum / STFT の役割差が書かれている。

## 詰まったときに見る資料
- [`../../latex/markdown/ch13-basics-of-signal-processing.md`](../../latex/markdown/ch13-basics-of-signal-processing.md)
- [`../../latex/markdown/ch14-basics-of-spectrum-analysis.md`](../../latex/markdown/ch14-basics-of-spectrum-analysis.md)
