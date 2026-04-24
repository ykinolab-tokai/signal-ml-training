# 第28回 口頭技術確認・code walkthrough・repo 整理

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 第26回と第27回の baseline を、入力から出力まで口頭で説明できる。
- 提出物の不足や命名ミスを自分で点検できる。
- 説明が弱い箇所を洗い出し、どの file を見直せば補強できるか判断できる。

## ミニ講義
- 口頭確認では、断片的な用語説明よりも「入力」「前処理」「model 出力」「推論例」の順で話すと伝わりやすい。
- repo 整理は掃除ではなく検証である。必要な file が揃っているか、名前が指定どおりか、後から他人が追えるかを確認する。
- 「答えられない質問」をそのままにせず、どの script や保存物を見直せば確認できるかを結びつけると、自力で補強しやすい。

## 演習
### 基礎レベル
1. `session28_oral_check_notes.md` を作成し、次の 6 見出しをこの順で書く。
   - `## 画像 baseline の入力`
   - `## 画像 baseline の model 出力`
   - `## 画像 baseline の推論例`
   - `## 音響 baseline の入力`
   - `## 音響 baseline の model 出力`
   - `## 音響 baseline の推論例`
2. `## 画像 baseline の入力` には第26回で使った dataset の class 定義、入力 shape、前処理を書く。
3. `## 画像 baseline の model 出力` には logits の shape と class 数を書く。
4. `## 画像 baseline の推論例` には保存画像の file 名、正解 label、予測 label を書く。
5. `## 音響 baseline の入力` には第27回で使った waveform 周波数、log-mel shape、特徴量設定を書く。
6. `## 音響 baseline の model 出力` には logits の shape と class 数を書く。
7. `## 音響 baseline の推論例` には保存画像の file 名、正解 label、予測 label を書く。
8. `session28_repo_checklist.md` を作成し、次の 8 行をそのまま書き、各行の末尾に `OK` または `NG` を追記する。
```text
第15回の提出物がある
第16回の提出物がある
第17回の提出物がある
第18回から第25回までの script がある
第26回の loss 曲線と推論例がある
第27回の loss 曲線と推論例がある
不要な一時ファイルを提出対象に含めていない
各回のファイル名が handout 指定と一致している
```
9. `session28_oral_check_notes.md` の末尾に `## 想定質問` を追加し、次の 6 問に対する答えを書く。
   - `画像 baseline の入力 shape は何か`
   - `画像 baseline の class 0 と class 1 は何か`
   - `画像 baseline の logits shape は何か`
   - `音響 baseline の入力特徴量は何か`
   - `音響 baseline の class 0 と class 1 の周波数は何か`
   - `音響 baseline の logits shape は何か`

### 発展レベル
1. `session28_oral_check_notes.md` に `## 説明が弱い箇所` を追加し、自分が口頭で詰まりそうな点を 2 つ書く。
2. 各項目について、どの file を見直せば補強できるかを 1 行ずつ書く。
3. `session28_repo_checklist.md` で `NG` になった項目がある場合は、その原因と修正方針を 1 行ずつ追記する。
4. 最後に、「保存物の file 名まで説明に含める理由」を 2 行以内で書く。

## 確認ポイント
- `session28_oral_check_notes.md` に 6 見出しと `## 想定質問` がある。
- `session28_repo_checklist.md` の 8 行すべてに `OK` / `NG` が付いている。
- 第26回と第27回の保存物 file 名が note 内に書かれている。
- 発展課題で、自分の弱点と見直すべき file が対応づいている。

## 詰まったときに見る資料
- [`26-image-baseline-mini-implementation.md`](26-image-baseline-mini-implementation.md)
- [`27-audio-baseline-mini-implementation.md`](27-audio-baseline-mini-implementation.md)
