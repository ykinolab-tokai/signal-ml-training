# 第12回 音響深掘り回：tagging（2026年度版）

- 対象: B4・M
- 種別: 前期発展枠 / ローテーション
- 今年度設定: この file では `tagging` を扱う。別年度で差し替える場合は file 全体を書き換えてから配布する。

## この回の目標
- clip-level audio tagging における multi-label target と logits の対応を説明できる。
- log-mel 入力の最小 tagging pipeline を実装できる。
- sigmoid 後の確率と threshold の関係を説明できる。

## ミニ講義
- audio tagging は「どのクラスが含まれるか」を複数同時に判定することがある。そのため target は one-hot ではなく multi-hot になり、各次元は独立に解釈する。
- `BCEWithLogitsLoss` は各クラスごとの二値判定をまとめて扱うときに使いやすい。出力 logits に sigmoid をかけると各クラスの確率らしい値になるが、採用する threshold は task 次第で変わる。
- 確率が高いことと threshold を超えることは別である。threshold を動かすと、取りこぼしと誤検出のどちらを重く見るかが変わる。

## 演習
### 基礎レベル
1. `session12_audio_tagging_demo.py` を作成し、次の import と出力ディレクトリ作成コードを書く。
```python
from pathlib import Path
import math
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

fig_dir = Path("outputs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(12)
```
2. `TaggingDataset` を作り、1 秒波形と長さ 2 の multi-hot target を返すようにする。
3. 4 パターン `[0,0]`, `[1,0]`, `[0,1]`, `[1,1]` を 2 回ずつ繰り返し、合計 8 サンプル作る。
4. 波形は target に応じて `440 Hz` と `880 Hz` のサイン波を足し合わせて作る。
5. log-mel を次の設定で計算する。
```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=128,
    n_mels=64,
)
```
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
7. loss は `nn.BCEWithLogitsLoss()`、optimizer は `torch.optim.Adam(model.parameters(), lr=1e-2)` とする。
8. `for epoch in range(10):` で 10 epoch 学習する。
9. dataset の先頭サンプル 1 件の log-mel を `outputs/figures/session12_log_mel_example.png` に保存する。
10. `session12_audio_tagging_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## target の定義`
   - `## 学習設定`
   - `## 予測結果`
11. `## 予測結果` には、先頭サンプルの target と logits を書く。

### 発展レベル
1. 全 8 サンプルについて logits に sigmoid をかけ、threshold を `0.3`, `0.5`, `0.7` の 3 つで比較する。
2. `session12_audio_tagging_report.md` に `## threshold の比較` を追加し、どの threshold で positive 判定が増えやすいかを 2 行以内で書く。
3. そのうえで、「今回の toy task ならどの threshold を最初に使うか」を 2 行以内で説明する。
4. M は、実データで threshold を決めるときに validation 側で何を見るべきかを 2 行で書く。

## 確認ポイント
- dataset 長が `8` である。
- target shape が `(batch, 2)`、model 出力 shape が `(batch, 2)` である。
- `session12_log_mel_example.png` が保存されている。
- report に、logits と threshold の違いに関する説明がある。

## 詰まったときに見る資料
- [`08-audio-representations-and-models.md`](08-audio-representations-and-models.md)
- [`../autumn/27-audio-baseline-mini-implementation.md`](../autumn/27-audio-baseline-mini-implementation.md)
