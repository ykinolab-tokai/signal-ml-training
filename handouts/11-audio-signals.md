# 第11回 音響信号

- 音を file として読み込み，波形，サンプリング周波数，長さを確認する。
- 音を再生し，リサンプリングによる変化を確認する。
- STFT とメルスペクトログラムを計算し，時間周波数表現として保存する。
- 波形，スペクトログラム，メルスペクトログラムの対応を説明する。

## 解説
- 音響信号は，サンプリング周波数と振幅列で表される。file を読むときは，波形の shape，dtype，サンプリング周波数を必ず確認する。
- 再生は人間が結果を確認する重要な手段だが，環境によって使える方法が異なる。再生できない場合でも，波形と保存 file から処理の正しさを確認する。
- リサンプリングはサンプリング周波数を変える処理である。時間長，サンプル数，周波数成分の関係を崩さないように注意する。
- STFT は短い窓ごとに Fourier 変換することで，時間と周波数の両方を見る表現である。メルスペクトログラムは，人の聴覚に近い周波数軸へ変換した表現である。

## 演習
### 基礎レベル（7問）
1. `scripts/`，`outputs/session11/`，`outputs/audio/`，`outputs/figures/` を作成し，`scripts/session11_audio_signal.py` で `fs = 16000`，長さ 1 秒の 440 Hz sine 波を作る。
2. `soundfile.write` で `outputs/audio/session11_sine_440hz.wav` に保存し，`soundfile.read` で読み戻して shape，dtype，サンプリング周波数を表示する。
3. 利用環境で可能なら保存した wav を再生し，聞こえ方を `outputs/session11/playback_notes.md` に記録する。再生できない場合は，その理由を書く。
4. 440 Hz と 880 Hz を足した信号を作り，波形を `outputs/figures/session11_waveform.png` に保存する。
5. `torchaudio.functional.resample` を使い，16 kHz から 8 kHz へリサンプリングする。サンプル数と時間長を比較する。
6. `torch.stft` または `torchaudio.transforms.Spectrogram` で STFT magnitude を計算し，画像として保存する。
7. `torchaudio.transforms.MelSpectrogram` でメルスペクトログラムを計算し，STFT との shape の違いを report に書く。

### 発展レベル（7問）
1. `n_fft` を 256，512，1024 に変え，周波数方向と時間方向の見え方を比較する。
2. `hop_length` を変え，時間フレーム数がどう変化するか表にする。
3. 16 kHz から 4 kHz へリサンプリングした場合，880 Hz 成分がどう見えるか確認する。
4. 音量を 0.2，0.8，1.5 倍し，clipping が起きる場合の波形を確認する。
5. 左右で異なる sine 波を持つ stereo wav を作り，読み込んだときの shape を確認する。
6. メルバンド数を 40，80，128 に変え，図の解像度と shape を比較する。
7. 波形で分かること，STFT で分かること，メルスペクトログラムで分かることを3列の表にまとめる。

## 詰まったときに見る資料
- [`../textbook/markdown/ch11-introduction-to-soundfile.md`](../textbook/markdown/ch11-introduction-to-soundfile.md)
- [`../textbook/markdown/ch14-basics-of-spectrum-analysis.md`](../textbook/markdown/ch14-basics-of-spectrum-analysis.md)
- [soundfile documentation](https://python-soundfile.readthedocs.io/)
- [torchaudio transforms](https://docs.pytorch.org/audio/stable/transforms.html)
