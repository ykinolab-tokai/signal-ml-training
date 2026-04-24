# 第17回 Git / GitHub の基礎：clone・commit・push・PR

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- `clone -> add -> commit -> push` を通常提出の流れとして実行できる。
- 練習用 branch を作成し、pull request を 1 件作成できる。
- 通常提出と PR 練習の違いを、自分の言葉で説明できる。

## ミニ講義
- 通常提出では、自分の提出履歴を積み重ねることが目的なので、`submission repo` の default branch に `commit` と `push` を行う。一方、PR は変更内容を review 可能な単位で見せる練習になる。
- branch を切る理由は、変更単位を分けて比較しやすくするためである。練習用 branch を残すと、main の通常提出と混ざらずに試行できる。
- PR 本文には、変更点、確認方法、未確認事項を書く。これにより、reviewer は何を見ればよいか判断しやすくなる。

## 演習
### 基礎レベル
1. 自分の `submission repo` を clone し、`git status` の結果を確認する。
2. default branch 上で `session17_git_practice.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 今回の通常提出で追加したもの`
   - `## 実行したコマンド`
   - `## default branch に push した理由`
3. `git add session17_git_practice.md`、`git commit -m "session17-main-submit"`、`git push origin main` をこの順で実行する。
4. `git checkout -b pr-practice-session17` を実行して練習用 branch へ移る。
5. `session17_git_commands.md` を作成し、次の 5 行をそのまま書く。
```text
git status
git add session17_git_commands.md
git commit -m "session17-pr-practice"
git push origin pr-practice-session17
pull request を作成した
```
6. `git add session17_git_commands.md`、`git commit -m "session17-pr-practice"`、`git push origin pr-practice-session17` をこの順で実行する。
7. GitHub 上で `pr-practice-session17` から default branch への pull request を 1 件作成し、タイトルを `session17 pr practice` とする。
8. PR 本文には次の 3 見出しをこの順で書き、その下に 1 行ずつ記入する。
   - `## 変更点`
   - `## 確認方法`
   - `## 未確認事項`

### 発展レベル
1. `session17_git_practice.md` の `## default branch に push した理由` では、通常提出が PR ではなく main である理由を 2 行以内で書く。
2. `session17_git_commands.md` の末尾に 2 行追加し、練習用 branch を main に merge しない理由を書く。
3. PR 本文の `## 未確認事項` には、「まだ review が必要な点」または「この PR が練習用であること」が reviewer に伝わるように書く。
4. 最後に、通常提出と PR 練習の違いを `提出先`, `目的`, `残し方` の 3 観点で 3 行以内にまとめる。

## 確認ポイント
- default branch への通常提出と、練習用 branch からの PR 作成が両方できている。
- 通常提出の commit message が `session17-main-submit`、PR 練習の commit message が `session17-pr-practice` である。
- PR タイトルが `session17 pr practice` であり、本文に `変更点`、`確認方法`、`未確認事項` がある。
- 通常提出と PR 練習の違いが、単に「branch が違う」ではなく運用目的の違いとして書かれている。

## 詰まったときに見る資料
- [`../../README.md`](../../README.md)
- [`../../latex/markdown/ch01-basic-operations.md`](../../latex/markdown/ch01-basic-operations.md)
