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
1. `session08_audio_model_input_demo.py` を作成し、1 秒の 440 Hz 波形から waveform, STFT, log-mel を作る。`wave.shape`, `stft.shape`, `log_mel.shape` を表示する。
2. waveform を入力する 1D CNN と、log-mel を入力する `TinyCRNN` を実装し、それぞれ logits shape が `(1, 2)` になることを確認する。
3. log-mel を `outputs/figures/session08_log_mel.png` に保存する。図には時間方向と周波数方向が分かるように軸ラベルを付ける。
4. `session08_audio_model_input_report.md` に `## 表現ごとの shape`, `## 1D CNN の入力と出力`, `## CRNN の入力と出力` を書き、各 model がどの軸を時間として扱っているかを説明する。

### 発展レベル
1. `session08_audio_model_input_report.md` に `## 表現と model の相性` を追加し、`waveform + 1D CNN` と `log-mel + CRNN` を、直接使いやすい軸情報と前処理の負担の観点で比較する。
2. 波形をそのまま CRNN に入れる場合、または log-mel を 1D CNN に入れる場合に、入力 shape をどう整える必要があるかを 3 行以内で説明する。

## 確認ポイント
- `wave.shape` が `(1, 16000)` である。
- `cnn1d` と `TinyCRNN` の logits shape がどちらも `(1, 2)` である。
- `session08_log_mel.png` が保存されている。
- report に、shape の列挙だけでなく、軸の意味と表現と model の相性に関する説明がある。

## 詰まったときに見る資料
- [`../autumn/21-fourier-and-stft-basics.md`](../autumn/21-fourier-and-stft-basics.md)
- [`../autumn/22-audio-data-basics.md`](../autumn/22-audio-data-basics.md)
