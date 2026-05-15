# 第08回 Fourier 変換

- DFT を使って，時間信号を周波数成分として確認する。
- 振幅スペクトルと位相スペクトルを描く。
- 窓関数がスペクトルに与える影響を確認する。
- 逆変換で時間信号に戻せることを確認する。

## 解説
- Fourier 変換は，信号を複数の周波数成分の重ね合わせとして見るための道具である。離散信号では DFT を使う。
- DFT の結果は複素数であり，大きさが振幅，角度が位相に対応する。振幅だけを見ると，時間的なずれの情報を見落とすことがある。
- 周波数 bin は，サンプリング周波数とサンプル数から決まる。`np.fft.fftfreq` を使うと，各 bin が何 Hz に対応するか確認できる。
- 有限長の信号を切り出して DFT するため，窓の端で不連続があると spectral leakage が起きる。窓関数はその影響を調整する。

## 演習
### 基礎レベル（7問）
1. `scripts/`，`outputs/session08/`，`outputs/figures/` を作成し，`scripts/session08_fourier_transform.py` で `fs = 1000`，`duration = 1.0` の 50 Hz sine 波を作る。
2. `np.fft.fft` と `np.fft.fftfreq` を使い，振幅スペクトルを描く。
3. 50 Hz と 120 Hz を足した信号を作り，振幅スペクトルの peak がどこに出るか確認する。
4. 位相を 0，`pi/2` に変えた 50 Hz sine 波を作り，時間波形と位相スペクトルを比較する。
5. 55.5 Hz の sine 波を DFT し，周波数が bin に一致しない場合の leakage を確認する。
6. 矩形窓と Hann 窓を比較し，振幅スペクトルの広がりを図にする。
7. `np.fft.ifft` で時間信号を復元し，元信号との最大絶対誤差を計算する。

### 発展レベル（7問）
1. 小さい `N = 8` について DFT 行列を自分で作り，`np.fft.fft` と同じ結果になるか確認する。
2. 振幅スペクトルの正規化方法を調べ，片側スペクトルで振幅を読み取りやすい形にする。
3. zero padding を入れた場合，周波数 peak の見え方がどう変わるか比較する。
4. 位相スペクトルを `np.unwrap` ありとなしで描き，違いを説明する。
5. 窓関数を Hann，Hamming，Blackman で比較する。
6. 短い信号と長い信号で周波数分解能がどう変わるか確認する。
7. 第7回の aliasing の例を Fourier 変換し，時間波形だけでは分かりにくい変化を周波数で説明する。

## 詰まったときに見る資料
- [`../textbook/markdown/ch14-basics-of-spectrum-analysis.md`](../textbook/markdown/ch14-basics-of-spectrum-analysis.md)
- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)
