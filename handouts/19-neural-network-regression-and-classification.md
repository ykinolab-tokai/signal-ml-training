# 第19回 ニューラルネットワークによる回帰と分類

## この回の目標

多層パーセプトロンで回帰と分類のモデルを学習させ，学習曲線から過学習を判断できる。

## 解説
- 入力の線形変換だけを使うモデル（線形モデル）は，回帰では直線（多次元では平面）しか表せず，分類ではクラスの境界を直線でしか表せない。線形変換の後に非線形な **活性化関数**（例：ReLU $\max(0, z)$）を挟んで層を重ねると，曲線や曲がった境界も表せる。このようなモデルを **多層パーセプトロン (multilayer perceptron, MLP)** という。
- 実数値を予測する **回帰** では，最後の層の出力をそのまま予測値とし，`nn.MSELoss`（平均二乗誤差）を損失に使う。これは，誤差が正規分布に従うと仮定した最尤推定にあたる（第18回）。
- クラスを予測する **分類** では，最後の層がクラスごとのスコア（logits）を出力する。`nn.CrossEntropyLoss` は，logits を softmax で確率に変換したときの負の対数尤度を計算する（第18回）。そのため，モデルの最後に softmax を入れる必要はない。
- 回帰と分類の違いは，最後の層の出力の意味と損失関数だけである。隠れ層の構造と学習ループは共通である。
- 各層のパラメータについての損失の勾配は，出力側から入力側へ連鎖律を繰り返し使って求める。これを **誤差逆伝播法 (backpropagation)** といい，PyTorch では第17回の自動微分が行う。`torch.optim.Adam` は，パラメータごとに更新量を調整する勾配降下法の一種である（第16回）。
- 学習の進み具合は，train と validation の損失を更新回数ごとに並べた **学習曲線** で確認する。train の損失が下がり続けるのに validation の損失が上がり始めた状態を **過学習 (overfitting)** という。validation の損失が最小になった時点のパラメータを採用する方法を **early stopping** という。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session19_mlp_regression.py` を作成し，`rng = np.random.default_rng(0)` を使って $[0, 1]$ の一様乱数 $x$ を 100 個と，$y=\sin(2\pi x)+\varepsilon$（$\varepsilon$ は平均 0，標準偏差 0.1 の正規乱数）を作ってください。前半 70 個を train，残り 30 個を validation とします。
2. 線形モデル `nn.Linear(1, 1)` と，MLP `nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))` を，`nn.MSELoss` と `torch.optim.Adam(lr=1e-2)` で 2000 回更新してください。各更新では train の全データを使います。実行前に，どちらの validation の平均二乗誤差が小さくなるかを予想してください。データと 2 つのモデルの予測曲線を 1 枚の図に描き，`outputs/figures/session19_regression_fit.png` に保存してください。train と validation の平均二乗誤差も表示してください。
3. `session19_mlp_classification.py` を作成し，2 クラスの 2 次元データを作ってください。`rng = np.random.default_rng(0)` を使い，各クラス 200 点を，半径 $r$ と角度 $\phi$ から点 $(r\cos\phi, r\sin\phi)$ として作ります。クラス 0 は $r = 1$，クラス 1 は $r = 2$ に，平均 0，標準偏差 0.2 の正規乱数を加えます。$\phi$ は $[0, 2\pi)$ の一様乱数とします。400 点の順番を `rng.permutation` で並べ替え，前半 70 % を train，残りを validation としてください。データの散布図を `outputs/figures/session19_classification_data.png` に保存してください。
4. 線形モデル `nn.Linear(2, 2)` と，MLP `nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))` を，`nn.CrossEntropyLoss` と `torch.optim.Adam(lr=1e-2)` で 200 回更新してください。各更新では train の全データを使います。train と validation の損失を更新回数ごとに記録して学習曲線を描き，`outputs/figures/session19_classification_curves.png` に保存してください。最後の train と validation の正解率も表示してください。
5. 線形モデルが回帰と分類のどちらでもうまくいかない理由を，予測曲線とクラスの境界の形に結びつけて説明してください。また，基礎第2問と基礎第4問で，モデルの最後の層の出力の数と損失関数が異なる理由を，第18回の内容を使って説明してください。

### 発展レベル
1. 基礎第1問のデータで，train を先頭 10 点だけに減らし，隠れ層を 2 層（各 256 ユニット）にした MLP を 5000 回更新してください。予測曲線と学習曲線を描き，過学習が起きているかを判断してください。
2. 発展第1問で validation の損失が最小になった更新回数を求め，early stopping でその時点のパラメータを採用した場合と，最後まで学習した場合の validation の平均二乗誤差を比較してください。
3. 基礎第4問の MLP の決定境界を，格子点での予測クラスを塗り分けて描き，`outputs/figures/session19_decision_boundary.png` に保存してください。線形モデルの境界との違いを説明してください。

## 確認ポイント
- 回帰のデータが train 70 点，validation 30 点，分類のデータが train 280 点，validation 120 点である。
- `session19_regression_fit.png` に，データと 2 つのモデルの予測曲線が描かれている。
- 学習曲線の図に，分類の 2 つのモデルの train と validation の損失が描かれている。
- MLP の validation の誤差（回帰）と正解率（分類）が線形モデルより良いことを確認し，その理由が予測曲線や境界の形と結びつけて説明されている。
- 発展課題に取り組んだ場合は，過学習の判断が train と validation の損失の推移に基づいている。

## 詰まったときに見る資料
- [`18-mle-and-map-estimation.md`](18-mle-and-map-estimation.md)
- [`../textbook/markdown/ch23-basics-of-neural-networks.md`](../textbook/markdown/ch23-basics-of-neural-networks.md)
