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
1. `session21_fourier_stft_basics.py` を作成し、440 Hz と 880 Hz を含む 1 秒の合成波形を生成する。
2. `torch.fft.rfft` で spectrum を計算し、magnitude のピーク 2 つの周波数を表示する。spectrum 図を `outputs/figures/session21_spectrum.png` に保存する。
3. `torch.stft` で STFT を計算し、`stft.abs()` を `outputs/figures/session21_stft.png` に保存する。
4. `session21_fourier_stft_report.md` に `## 波形と周波数ピーク`, `## spectrum 図`, `## STFT 図` を書き、どの図で何が分かるかを説明する。

### 発展レベル
1. `n_fft=256` と `n_fft=512` の STFT を比較する。
2. report に `## n_fft の比較` を追加し、周波数側の見え方と時間方向の見え方の trade-off を説明する。

## 確認ポイント
- `wave.shape` が `(16000,)` である。
- ピーク周波数として 440 Hz と 880 Hz 付近が抽出される。
- `session21_spectrum.png` と `session21_stft.png` が保存されている。
- report に、`n_fft` の違いと spectrum / STFT の役割差が書かれている。

## 詰まったときに見る資料
- [`../../latex/markdown/ch13-basics-of-signal-processing.md`](../../latex/markdown/ch13-basics-of-signal-processing.md)
- [`../../latex/markdown/ch14-basics-of-spectrum-analysis.md`](../../latex/markdown/ch14-basics-of-spectrum-analysis.md)
