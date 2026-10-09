# 第14回 口頭技術確認・code walkthrough・リポジトリ整理・PR / code review 体験

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標

音響・画像の実装を入力から出力まで説明し，提出物と変更差分をレビューできる形に整えられる。

## 解説
- 口頭確認では、断片的な用語説明よりも、音響信号処理では「入力 file」「波形とサンプリング周波数」「時間周波数表現」「保存物」、画像 baseline では「入力」「前処理」「model 出力」「推論例」の順で話すと伝わりやすい。
- リポジトリ整理は掃除ではなく検証である。必要な file が揃っているか、名前が指定どおりか、後から他人が追えるかを確認する。
- PR は毎週の通常提出ではなく、変更点を他人に review してもらうための形式である。本文には、変更点、確認方法、未確認事項を書くと、reviewer が見るべき範囲を判断しやすい。
- 「答えられない質問」をそのままにせず、どの script や保存物を見直せば確認できるかを結びつけると、自力で補強しやすい。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。以下の相対パスは，自分の作業リポジトリのルートを基準とします。

### 基礎レベル
1. `session14_oral_check_notes.md` を作成し、第09回の音響信号処理と第13回の画像 baseline について、`## 音響信号の入力`, `## 音響信号の処理と表現`, `## 音響信号の保存物`, `## 画像 baseline の入力`, `## 画像 baseline の model 出力`, `## 画像 baseline の推論例` の 6 見出しで整理してください。
2. 音響信号処理では waveform 周波数、サンプリング周波数、STFT shape、mel shape、保存した wav と図 file 名を書いてください。画像 baseline では class 定義、入力 shape、logits shape、保存画像 file 名、正解 label、予測 label を書いてください。
3. `session14_repo_checklist.md` を作成し、第1回から第13回までの提出物、保存物、不要な一時ファイル、handout 指定の file 名を点検して `OK` / `NG` を付けてください。
4. 練習用 branch `pr-practice-session14` を作成し、`session14_oral_check_notes.md` と `session14_repo_checklist.md` を commit / push してください。
5. GitHub 上で `session14 walkthrough practice` というタイトルの draft PR を 1 件作成し、本文に `## 変更点`, `## 確認方法`, `## 未確認事項` を書いてください。
6. `session14_oral_check_notes.md` に `## 想定質問` を追加し、画像 baseline 3 問、音響信号処理 3 問の答えを書いてください。

PRの宛先は自分の作業リポジトリの `main`，比較元は `pr-practice-session14` とします。練習の追記は同じブランチにcommit・pushしてください。次回の通常作業の前に [PR練習後の手順](README.md#pr練習後に通常作業へ戻る)で `main` へ切り替え，同期してください。レビュー待ちの変更は練習ブランチに残ります。

### 発展レベル
1. `session14_oral_check_notes.md` に `## 説明が弱い箇所` を追加し、口頭で詰まりそうな点を 2 つ書いてください。各項目には、どの file を見直せば補強できるかを添えてください。
2. `session14_repo_checklist.md` で `NG` になった項目がある場合は、原因と修正方針を 1 行ずつ追記してください。
3. PR 本文の `## 未確認事項` に、reviewer に見てほしい点を 2 つ書いてください。
4. 保存物の file 名まで説明に含める理由を、再現性と review しやすさの観点から説明してください。

## 確認ポイント
- `session14_oral_check_notes.md` に 6 見出しと `## 想定質問` がある。
- `session14_repo_checklist.md` の 8 行すべてに `OK` / `NG` が付いている。
- PR タイトルが `session14 walkthrough practice` であり、本文に `変更点`、`確認方法`、`未確認事項` がある。
- 第09回と第13回の保存物 file 名が note 内に書かれている。
- 発展課題で、自分の弱点と見直すべき file が対応づいている。

## 詰まったときに見る資料
- [`09-audio-signals.md`](09-audio-signals.md)
- [`13-image-baseline-mini-implementation.md`](13-image-baseline-mini-implementation.md)
