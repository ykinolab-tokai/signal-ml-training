# 第21回 画像モデル基礎：CNN・ResNet・U-Net・receptive field・normalization

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標

plain block と residual block を比較し，skip connection の役割を説明できる。

## 解説
- CNN block は「畳み込みの並び」で考えると読みやすいが、ResNet ではそこに入力を足し戻す skip connection が入る。これにより、同じ shape を保ちながら振る舞いが変わる。
- output shape と parameter 数が同じでも、内部計算は同じとは限らない。特に residual block では `x + F(x)` の形になるため、「入力からどれだけ変えるか」を学ぶ見方ができる。
- 画像モデルを比較するときは、shape だけでなく「入力との差分がどう扱われるか」を見ると、block の意味を捉えやすい。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session21_image_block_compare.py` を作成し、同じ入力 `x = torch.randn(2, 8, 32, 32)` に対して `PlainBlock` と `ResidualBlock` の forward を通してください。
2. 2 つの block の output shape、parameter 数、`(out - x).abs().mean()` を計算してください。
3. 入力と出力の shape，parameter 数，block の違い，入力との差分をコードコメントまたは既存の結果ファイルに記録してください。
4. shape と parameter 数が近くても、skip connection の有無によって block の意味が変わる理由を、`x + F(x)` という見方に触れて説明してください。

### 発展レベル
1. residual block を 2 個並べた model を作り、shape と parameter 数がどう変わるかを確認してください。
2. B4 は、block を深くする利点と起きやすい問題を 2 行ずつ書いてください。M は、channel 数や解像度が変わる場合に skip connection をそのまま足せない理由と対処案を 2 行ずつ書いてください。

## 確認ポイント
- `x.shape`, `plain_out.shape`, `res_out.shape` がすべて `(2, 8, 32, 32)` である。
- 基礎課題の記録に入力と出力の shape，parameter 数，block の違い，入力との差分がある。
- skip connection の説明が、単に「足している」ではなく、入力を保持しながら変化量を学ぶという見方に触れている。

## 詰まったときに見る資料
- [`13-image-baseline-mini-implementation.md`](13-image-baseline-mini-implementation.md)
- [`../textbook/markdown/ch23-basics-of-neural-networks.md`](../textbook/markdown/ch23-basics-of-neural-networks.md)
