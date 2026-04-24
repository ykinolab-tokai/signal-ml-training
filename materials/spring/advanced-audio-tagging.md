# 発展テーマ候補：音響 tagging

- 対象: B4・M
- 種別: 前期 11〜13 回の年度別発展テーマ候補
- 運用: このテーマを選んだ年度は、前期 11〜13 回の 3 回を使って `tagging` を扱う。画像・音響・共通基盤を同じ年度にすべて扱う前提にはしない。

## この回の目標
- clip-level audio tagging における multi-label target と logits の対応を説明できる。
- log-mel 入力の最小 tagging pipeline を実装できる。
- sigmoid 後の確率と threshold の関係を説明できる。

## ミニ講義
- audio tagging は「どのクラスが含まれるか」を複数同時に判定することがある。そのため target は one-hot ではなく multi-hot になり、各次元は独立に解釈する。
- `BCEWithLogitsLoss` は各クラスごとの二値判定をまとめて扱うときに使いやすい。出力 logits に sigmoid をかけると各クラスの確率らしい値になるが、採用する threshold は task 次第で変わる。
- 確率が高いことと threshold を超えることは別である。threshold を動かすと、取りこぼしと誤検出のどちらを重く見るかが変わる。

## 演習
### 基礎レベル
1. `advanced_audio_tagging_demo.py` を作成し、multi-label target を持つ synthetic audio tagging dataset と log-mel 入力の小さな model を実装する。dataset は 8 サンプルにする。
2. target shape と model 出力 shape がどちらも `(batch, 2)` になることを確認し、`BCEWithLogitsLoss` で数 step 学習する。
3. log-mel 例を `outputs/figures/advanced_audio_log_mel_example.png` に保存し、sigmoid 後の確率と threshold 後の予測 label を表示する。
4. `advanced_audio_tagging_report.md` に `## multi-label target`, `## logits と確率`, `## threshold の影響` を書き、確率が高いことと採用 label になることの違いを説明する。

### 発展レベル
1. threshold を `0.3`, `0.5`, `0.7` に変え、予測 label がどう変わるかを比較する。
2. report に `## threshold 比較` を追加し、取りこぼしと誤検出のどちらを重く見るかで threshold の選び方が変わる理由を説明する。

## 確認ポイント
- dataset 長が `8` である。
- target shape が `(batch, 2)`、model 出力 shape が `(batch, 2)` である。
- `advanced_audio_log_mel_example.png` が保存されている。
- report に、logits と threshold の違いに関する説明がある。

## 詰まったときに見る資料
- [`08-audio-representations-and-models.md`](08-audio-representations-and-models.md)
- [`../autumn/26-audio-baseline-mini-implementation.md`](../autumn/26-audio-baseline-mini-implementation.md)
