# 第27回 音響ミニ実装：小規模 classification baseline

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 合成音データセット、特徴量抽出、Model、training loop を 1 本につなげて実行できる。
- `waveform -> log-mel -> CNN -> logits` の流れを説明できる。
- 学習後の予測結果を、loss と全体精度の両面から説明できる。

## ミニ講義
- 音響 baseline では、波形をそのまま入れる代わりに log-mel を特徴量として使うと、周波数構造を 2 次元の画像のように扱いやすくなる。
- classification なので最終出力は class 数ぶんの logits になる。今回の task では `440 Hz` と `880 Hz` の 2 クラスである。
- 1 枚の log-mel 画像だけ見ても model 全体の性能は分からない。全サンプルに対する予測を確認して、task を本当に解けているかを見る必要がある。

## 演習
### 基礎レベル
1. `session27_audio_baseline.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import math
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

fig_dir = Path("outputs/figures")
audio_dir = Path("outputs/audio")
fig_dir.mkdir(parents=True, exist_ok=True)
audio_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(27)
```
2. `ToneDataset` を作り、class 0 は `440 Hz`、class 1 は `880 Hz` の 1 秒波形を返すようにする。
3. 各 class を 20 本、合計 40 本作る。
4. 各波形に対して次の log-mel 変換を適用し、入力 shape を `(1, 64, time)` にする。
```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=128,
    n_mels=64,
)
```
5. `DataLoader(dataset, batch_size=8, shuffle=True)` を作る。
6. model を次の通り定義する。
```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(8, 2),
)
```
7. `criterion = nn.CrossEntropyLoss()`、`optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)` を使い、`for epoch in range(5):` で 5 epoch 学習する。
8. 各 epoch の平均 loss を `loss_history` に記録し、`outputs/figures/session27_loss_curve.png` に保存する。
9. dataset の先頭サンプル 1 件の log-mel を `outputs/audio/session27_example_log_mel.png` に保存し、そのサンプルを model に通して予測 class を得る。
10. `session27_audio_baseline_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## データセット仕様`
   - `## 学習設定と loss`
   - `## 推論例`

### 発展レベル
1. 学習後、全 40 サンプルに対して推論を行い、全体 accuracy を計算する。
2. `session27_audio_baseline_report.md` に `## 全体評価` を追加し、class 0 / class 1 のどちらが区別しやすそうかを 2 行以内で書く。
3. そのうえで、「log-mel の見た目が分かれていても、全体評価を確認する必要がある理由」を 2 行以内で説明する。

## 確認ポイント
- dataset 総数が `40` である。
- batch 入力 shape の先頭 2 次元が `(8, 1)` である。
- `session27_loss_curve.png` と `session27_example_log_mel.png` が保存されている。
- report に、推論例 1 件だけでなく全体評価がある。

## 詰まったときに見る資料
- [`23-audio-data-basics.md`](23-audio-data-basics.md)
- [`25-training-loop-basics.md`](25-training-loop-basics.md)
