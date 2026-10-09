# 発展テーマ候補：音響 tagging

- 対象: B4・M
- 種別: 第 25〜27 回の年度別発展テーマ候補
- 運用: このテーマを選んだ年度は、第 25〜27 回の 3 回を使って `tagging` を扱う。画像・音響・共通基盤を同じ年度にすべて扱う前提にはしない。

## この回の目標
- clip-level audio tagging における multi-label target と logits の対応を説明できる。
- log-mel 入力の最小 tagging pipeline を実装できる。
- sigmoid 後の確率と threshold の関係を説明できる。

## 解説
- audio tagging は「どのクラスが含まれるか」を複数同時に判定することがある。そのため target は one-hot ではなく multi-hot になり、各次元は独立に解釈する。
- `BCEWithLogitsLoss` は各クラスごとの二値判定をまとめて扱うときに使いやすい。出力 logits に sigmoid をかけると各クラスの確率らしい値になるが、採用する threshold は task 次第で変わる。
- 確率が高いことと threshold を超えることは別である。threshold を動かすと、取りこぼしと誤検出のどちらを重く見るかが変わる。

### 波形とラベルの対応

第22回と同じPython環境とimportを使い、`from torch.utils.data import Dataset, DataLoader` を加える。モデル作成前に `torch.manual_seed(12)` を設定する。
targetは `[0,0]`, `[1,0]`, `[0,1]`, `[1,1]` をこの順に2回繰り返した8件で、各targetを `float32` の形状 `(2,)` とする。
第0成分は440 Hz、第1成分は880 Hzの音の有無を表す。
サンプリング周波数16000 Hz、1秒、`t = torch.arange(16000, dtype=torch.float32) / 16000` とし、
波形は `0.5 * target[0] * torch.sin(2 * torch.pi * 440 * t) + 0.5 * target[1] * torch.sin(2 * torch.pi * 880 * t)` で作る。
`[0,0]` は無音、`[1,1]` は2音の和である。

第22回と同じ `MelSpectrogram` 条件（FFT長512、ホップ長128、Hann窓、64帯域、`power=2.0`, `center=True`, `mel_scale="htk"`, `norm=None`）を使う。
`torch.log(mel + 1e-6)` にチャネル軸を加え、Datasetは `(log_mel, target)` を返す。
1件の形状は `(1, 64, 126)` と `(2,)`、`DataLoader(dataset, batch_size=2, shuffle=True)` の出力は `(2, 1, 64, 126)` と `(2, 2)` になる。

### モデルと学習条件

`nn.Sequential` で `nn.Conv2d(1, 8, 3, padding=1)` → `nn.ReLU()` → `nn.AdaptiveAvgPool2d((1, 1))` → `nn.Flatten()` → `nn.Linear(8, 2)` をつなぐ。
`nn.BCEWithLogitsLoss()` と `torch.optim.Adam(model.parameters(), lr=1e-2)` を使い、10 epoch（4バッチを10巡、計40 step）学習する。
lossへはsigmoid前のlogitsを渡す。学習時は `model.train()`、予測時は `model.eval()` と `torch.no_grad()` を使う。
第19回の学習ループをバッチごとに実行する。

基礎のthresholdは `0.5` とし、`torch.sigmoid(logits) >= threshold` で予測ラベルを得る。
全8件の正解・確率・予測を並べて比較する。短い学習で全件正解になることは完了条件にしない。
図の保存先 `outputs/figures/` を作成し、`[1,1]` のサンプル（添字3）のlog-melを保存する。
軸は第22回と同じく時間 [秒] とメル帯域番号とする。この固定8件での結果を未知の音声への性能とは解釈しない。

## 演習

作業場所・保存先・説明の残し方は [共通手順](../README.md#作業場所と保存先) に従う。以下の相対パスは，自分の作業repoのルートを基準とする。

このテーマでは，複数条件の結果と解釈をまとめて参照するため，下で指定する比較レポートを作る。数値や図は保存先を参照し，既存ファイルの内容を転記しない。第26回の比較記録からもこのレポートを参照する。

### 基礎レベル
1. `advanced_audio_tagging_demo.py` を作成し、multi-label target を持つ synthetic audio tagging dataset と log-mel 入力の小さな model を実装する。dataset は 8 サンプルにする。
2. target shape と model 出力 shape がどちらも `(batch, 2)` になることを確認し、`BCEWithLogitsLoss` で10 epoch学習する。
3. log-mel 例を `outputs/figures/advanced_audio_log_mel_example.png` に保存し、sigmoid 後の確率と threshold 後の予測 label を表示する。
4. `advanced_audio_tagging_report.md` に `## multi-label target`, `## logits と確率`, `## threshold の影響` を書き、確率が高いことと採用 label になることの違いを説明する。

### 発展レベル
1. threshold を `0.3`, `0.5`, `0.7` に変え、予測 label がどう変わるかを比較する。
2. report に `## threshold 比較` を追加し、取りこぼしと誤検出のどちらを重く見るかで threshold の選び方が変わる理由を説明する。

## 確認ポイント
- dataset 長が `8` である。
- target shape が `(batch, 2)`、model 出力 shape が `(batch, 2)` である。
- `advanced_audio_log_mel_example.png` が保存されている。
- report に、logits と threshold の違いに関する説明がある。

## 詰まったときに見る資料
- [`22-audio-representations-and-models.md`](../22-audio-representations-and-models.md)
- [`20-data-pipeline-engineering.md`](../20-data-pipeline-engineering.md): Datasetとバッチ化
