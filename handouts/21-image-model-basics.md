# 第21回 画像のための CNN：畳み込み層・画像分類・residual block

## この回の目標

小さな CNN で画像分類を学習させ，入力から logits までの shape と予測結果を説明できる。

## 解説
- 畳み込み層（`nn.Conv2d`）は，第12回の2次元フィルタ処理と同じ計算を行う。ただし，フィルタの係数を手で決めるのではなく，学習で決める。入力と出力はチャネル（色や特徴の種類）を持ち，shape は `(batch, channels, H, W)` になる。
- pooling や stride によって画像の縦横のサイズを小さくしながら層を重ねると，1 つの出力が入力画像のより広い範囲を見るようになる。この範囲を **receptive field** という。
- baseline 実装では、データ生成、前処理、model、学習ループをまず 1 本で繋げることが重要である。小さくても end-to-end で動くことが後の改良の土台になる。
- classification では、各サンプルに 1 つの label を持たせ、model は class 数ぶんの logits を返す。shape を追うと、どこで画像が特徴ベクトルへ潰れているか見やすい。
- loss だけを見ると「下がった」で終わりやすい。実際に全サンプルでどれだけ当たるかを見ると、model が task を解けているか判断しやすい。
- ResNet では、畳み込みの並びに入力を足し戻す skip connection が入る。residual block は `x + F(x)` の形になるため、「入力からどれだけ変えるか」を学ぶ見方ができる。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session21_image_baseline.py` を作成し、class 0 は中央縦線、class 1 は中央横線を持つ 32x32 RGB 画像を返す `StripeDataset` を実装してください。各 class 20 枚、合計 40 枚にしてください。
2. `DataLoader(batch_size=8, shuffle=True)`、小さな CNN、`CrossEntropyLoss`、`Adam(lr=1e-2)` を使って 5 epoch 学習してください。
3. epoch ごとの平均 loss を `outputs/figures/session21_loss_curve.png` に保存し、先頭サンプルの推論例を `outputs/images/session21_prediction_example.png` に保存してください。
4. データセット仕様，学習設定と loss，推論例について，入力 shape，各層の出力 shape，logits shape，予測 class をコードコメントまたは既存の結果ファイルに記録してください。

### 発展レベル
1. 学習後に全 40 サンプルで推論し、全体 accuracy と class 別の誤り傾向を確認してください。loss 曲線と 1 枚の推論例だけでは十分でない理由を説明してください。
2. `session21_image_block_compare.py` を作成し、同じ入力 `x = torch.randn(2, 8, 32, 32)` に対して `PlainBlock` と `ResidualBlock` の forward を通してください。2 つの block の output shape、parameter 数、`(out - x).abs().mean()` を計算し，skip connection の有無によって block の意味が変わる理由を、`x + F(x)` という見方に触れて説明してください。
3. B4 は、block を深くする利点と起きやすい問題を 2 行ずつ書いてください。M は、channel 数や解像度が変わる場合に skip connection をそのまま足せない理由と対処案を 2 行ずつ書いてください。

## 確認ポイント
- dataset 総数が `40` である。
- batch 入力 shape が `(8, 3, 32, 32)` である。
- `session21_loss_curve.png` と `session21_prediction_example.png` が保存されている。
- 基礎課題の記録にデータセット仕様・学習設定・推論例があり，発展課題に取り組んだ場合は全体評価もある。
- 発展第2問に取り組んだ場合は，`x.shape`, `plain_out.shape`, `res_out.shape` がすべて `(2, 8, 32, 32)` であり、skip connection の説明が、入力を保持しながら変化量を学ぶという見方に触れている。

## 詰まったときに見る資料
- [`12-image-signals.md`](12-image-signals.md)
- [`../textbook/markdown/ch23-basics-of-neural-networks.md`](../textbook/markdown/ch23-basics-of-neural-networks.md)
