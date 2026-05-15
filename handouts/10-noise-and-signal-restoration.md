# 第10回 ノイズと信号復元

- 加法性ノイズを含む信号を作り，時間波形とスペクトルで確認する。
- 信号とノイズのスペクトルの違いを観察する。
- 移動平均フィルタと Wiener filter による復元を比較する。
- 復元前後の誤差を数値で評価する。

## 解説
- 加法性ノイズは，観測信号を `観測 = 真の信号 + ノイズ` と見る単純なモデルである。実データでは真の信号は分からないが，合成実験では復元方法を評価しやすい。
- ノイズは時間波形だけでなくスペクトルで見ると特徴が分かりやすい。信号とノイズが異なる周波数帯にある場合，フィルタで分けられることがある。
- 移動平均フィルタは高周波成分を抑える単純な復元方法である。ただし，信号の急な変化も弱める可能性がある。
- Wiener filter は，信号とノイズのパワー比に基づいて周波数成分を調整する考え方である。ここでは合成データで信号とノイズのスペクトルを既知として扱う。

## 演習
### 基礎レベル（7問）
1. `scripts/`，`outputs/session10/`，`outputs/figures/` を作成し，`scripts/session10_noise_restoration.py` で 5 Hz と 40 Hz の sine 波を足した clean signal を作る。
2. 平均 0 の Gaussian noise を加え，noisy signal を作る。clean，noise，noisy を同じ図に描く。
3. clean，noise，noisy の振幅スペクトルを描き，どの周波数帯に成分があるか確認する。
4. moving average filter を実装し，window length 3，9，21 で復元結果を比較する。
5. clean と復元信号の平均二乗誤差を計算し，window length ごとに表にする。
6. 合成データで得られる clean と noise のパワースペクトルを使い，周波数領域の Wiener filter を実装する。
7. noisy，moving average，Wiener filter の結果を比較し，`outputs/session10/session10_report.md` に図と誤差をまとめる。

### 発展レベル（7問）
1. ノイズの標準偏差を 0.1，0.5，1.0 に変え，復元の難しさがどう変わるか確認する。
2. 高周波ノイズだけでなく低周波ノイズを加え，移動平均が効きにくい例を作る。
3. Wiener filter で信号パワーまたはノイズパワーの推定を意図的にずらし，復元結果への影響を見る。
4. 時間領域の移動平均と周波数領域の低域通過フィルタを比較する。
5. 復元結果の MSE だけでなく，最大絶対誤差も計算する。
6. clean signal を知らない場合に，ノイズの性質をどう推定できるか，今回の合成実験に基づいて案を書く。
7. 過度に強いフィルタで信号まで失われる例を作り，スペクトルと時間波形の両方で説明する。

## 詰まったときに見る資料
- [`../textbook/markdown/ch13-basics-of-signal-processing.md`](../textbook/markdown/ch13-basics-of-signal-processing.md)
- [`../textbook/markdown/ch14-basics-of-spectrum-analysis.md`](../textbook/markdown/ch14-basics-of-spectrum-analysis.md)
- [`../textbook/markdown/ch15-basics-of-lti-systems.md`](../textbook/markdown/ch15-basics-of-lti-systems.md)
