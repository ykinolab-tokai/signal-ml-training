# 第22回 音響データの基礎：波形・spectrogram・mel・簡単な augmentation

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
1. `session22_audio_data_basics.py` を作成し、440 Hz と 660 Hz を含む 1 秒の合成波形を作る。shape, min, max を表示し、`outputs/audio/session22_original.wav` に保存する。
2. spectrogram, mel, log-mel を計算し、log-mel を `outputs/audio/session22_log_mel.png` に保存する。
3. `torch.roll` で 800 sample の time shift を行い、`outputs/audio/session22_shifted.wav` に保存する。
4. `session22_audio_data_report.md` に `## 波形と shape`, `## 特徴量設定`, `## augmentation の内容` を書き、波形と時間周波数表現の shape の違いを説明する。

### 発展レベル
1. shifted 波形にも同じ log-mel 変換を適用し、`outputs/audio/session22_shifted_log_mel.png` に保存する。
2. report に `## time shift 前後の比較` を追加し、時間方向に変わるもの、周波数方向に大きく変わりにくいものを分けて説明する。

## 確認ポイント
- `wave.shape` が `(1, 16000)` である。
- `outputs/audio/session22_original.wav` と `outputs/audio/session22_shifted.wav` が保存されている。
- `outputs/audio/session22_log_mel.png` が保存されている。
- report に、time shift が時間側と周波数側へ与える影響の違いが書かれている。

## 詰まったときに見る資料
- [`../../latex/markdown/ch11-introduction-to-soundfile.md`](../../latex/markdown/ch11-introduction-to-soundfile.md)
- [`../../latex/markdown/ch14-basics-of-spectrum-analysis.md`](../../latex/markdown/ch14-basics-of-spectrum-analysis.md)
