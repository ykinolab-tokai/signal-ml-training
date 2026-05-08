# 第26回 音響ミニ実装：小規模 classification baseline

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
1. `session26_audio_baseline.py` を作成し、class 0 は 440 Hz、class 1 は 880 Hz の 1 秒波形を返す `ToneDataset` を実装する。各 class 20 本、合計 40 本にする。
2. 各波形を log-mel `(1, 64, time)` に変換し、`DataLoader(batch_size=8, shuffle=True)` で小さな CNN に入力する。
3. `CrossEntropyLoss` と `Adam(lr=1e-2)` で 5 epoch 学習し、loss 曲線を `outputs/figures/session26_loss_curve.png` に保存する。先頭サンプルの log-mel を `outputs/audio/session26_example_log_mel.png` に保存する。
4. `session26_audio_baseline_report.md` に `## データセット仕様`, `## 学習設定と loss`, `## 推論例` を書き、`waveform -> log-mel -> CNN -> logits` の流れを説明する。

### 発展レベル
1. 学習後に全 40 サンプルで推論し、全体 accuracy と class 別の誤り傾向を確認する。
2. report に `## 全体評価` を追加し、log-mel の見た目が分かれていても全体評価を確認する必要がある理由を説明する。

## 確認ポイント
- dataset 総数が `40` である。
- batch 入力 shape の先頭 2 次元が `(8, 1)` である。
- `session26_loss_curve.png` と `session26_example_log_mel.png` が保存されている。
- report に、推論例 1 件だけでなく全体評価がある。

## 詰まったときに見る資料
- [`22-audio-data-basics.md`](22-audio-data-basics.md)
- [`25-training-loop-basics.md`](25-training-loop-basics.md)
