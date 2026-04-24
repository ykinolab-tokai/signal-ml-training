# 第07回 画像モデル基礎：CNN・ResNet・U-Net・receptive field・normalization

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- plain block と residual block の違いをコードで比較できる。
- 出力 shape と parameter 数を見て、見た目の違いと内部の違いを分けて説明できる。
- skip connection が表現と勾配の流れにどう関わるかを言葉で説明できる。

## ミニ講義
- CNN block は「畳み込みの並び」で考えると読みやすいが、ResNet ではそこに入力を足し戻す skip connection が入る。これにより、同じ shape を保ちながら振る舞いが変わる。
- output shape と parameter 数が同じでも、内部計算は同じとは限らない。特に residual block では `x + F(x)` の形になるため、「入力からどれだけ変えるか」を学ぶ見方ができる。
- 画像モデルを比較するときは、shape だけでなく「入力との差分がどう扱われるか」を見ると、block の意味を捉えやすい。

## 演習
### 基礎レベル
1. `session07_image_block_compare.py` を作成し、同じ入力 `x = torch.randn(2, 8, 32, 32)` に対して `PlainBlock` と `ResidualBlock` の forward を通す。
2. 2 つの block の output shape、parameter 数、`(out - x).abs().mean()` を計算する。
3. `session07_image_block_compare_report.md` に `## 入力と出力 shape`, `## parameter 数`, `## block の違い`, `## 入力との差分` を書く。
4. shape と parameter 数が近くても、skip connection の有無によって block の意味が変わる理由を、`x + F(x)` という見方に触れて説明する。

### 発展レベル
1. residual block を 2 個並べた model を作り、shape と parameter 数がどう変わるかを確認する。
2. B4 は、block を深くする利点と起きやすい問題を 2 行ずつ書く。M は、channel 数や解像度が変わる場合に skip connection をそのまま足せない理由と対処案を 2 行ずつ書く。

## 確認ポイント
- `x.shape`, `plain_out.shape`, `res_out.shape` がすべて `(2, 8, 32, 32)` である。
- report に `入力と出力 shape`、`parameter 数`、`block の違い` があり、発展課題では入力との差分も説明されている。
- skip connection の説明が、単に「足している」ではなく、入力を保持しながら変化量を学ぶという見方に触れている。

## 詰まったときに見る資料
- [`../autumn/27-image-baseline-mini-implementation.md`](../autumn/27-image-baseline-mini-implementation.md)
- [`../../latex/markdown/ch23-basics-of-neural-networks.md`](../../latex/markdown/ch23-basics-of-neural-networks.md)
