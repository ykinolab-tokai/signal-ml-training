# 第13回 効率化回：CPU profiling と TorchScript export（2026年度版）

- 対象: B4・M
- 種別: 前期発展枠 / ローテーション
- 今年度設定: この file では `CPU profiling` と `TorchScript export` を扱う。別年度で差し替える場合は file 全体を書き換えてから配布する。

## この回の目標
- baseline 推論時間を測定し、平均値を記録できる。
- `torch.jit.trace` で TorchScript 化して保存できる。
- batch size が推論時間の見え方にどう影響するかを説明できる。

## ミニ講義
- profiling では 1 回だけ測るとばらつきが大きい。warm-up の後で複数回測り、平均を取ると比較しやすい。
- export は保存形式の変換だけではなく、「その model が指定した入力形状で辿れるか」を確認する作業でもある。trace 前後で shape を比べるのは最低限の整合性確認になる。
- 推論時間は batch size によって見え方が変わる。1 回あたり時間と 1 サンプルあたり時間は別物なので、どちらを比較しているか意識する必要がある。

## 演習
### 基礎レベル
1. `session13_profile_export.py` を作成し、次の import を書く。
```python
import time
import torch
from torch import nn
```
2. model と入力を次の通り定義する。
```python
torch.manual_seed(13)
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
x = torch.randn(64, 256)
```
3. warm-up として `model(x)` を 10 回実行する。
4. `time.perf_counter()` を使って 100 回の推論時間を測定し、平均時間をミリ秒で計算する。
5. 平均時間を `session13_profile_summary.txt` に 1 行で書く。
6. `traced_model = torch.jit.trace(model, x)` を実行し、`session13_traced_model.pt` として保存する。
7. `model(x).shape` と `traced_model(x).shape` を表示する。
8. `session13_profile_export_report.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 測定条件`
   - `## 推論時間`
   - `## export 結果`
9. `## 推論時間` には平均時間をミリ秒で書く。
10. `## export 結果` には trace 前後の出力 shape と保存ファイル名を書く。

### 発展レベル
1. `batch_size=1` の入力 `x_small = torch.randn(1, 256)` でも同じように 100 回測定する。
2. `session13_profile_export_report.md` に `## batch size による違い` を追加し、`batch_size=1` と `batch_size=64` の平均時間を書く。
3. そのうえで、「1 回あたり時間」と「1 サンプルあたり時間」のどちらを見るべきかを、今回の比較に即して 3 行以内で説明する。
4. M は、再現性を上げるために測定条件へ追加したい項目を 2 行書く。

## 確認ポイント
- `session13_profile_summary.txt` が作成されている。
- `session13_traced_model.pt` が作成されている。
- trace 前後の出力 shape がどちらも `(64, 10)` である。
- report に、batch size による見え方の違いが説明されている。

## 詰まったときに見る資料
- PyTorch docs: [`torch.jit.trace`](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html)
- Python docs: [`time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter)
