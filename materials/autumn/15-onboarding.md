# 第15回 研究室オンボーディング：Linux・SSH・ディレクトリ・仮想環境・repo clone・提出用 Git 最小運用

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 研究室サーバへ SSH 接続し、現在地確認、ディレクトリ作成、ファイル作成を自力で実行できる。
- 仮想環境を作成し、有効化後の Python を確認できる。
- 共通 repo と自分の `submission repo` を clone し、通常提出の `commit` / `push` を実行できる。
- 共通 repo と `submission repo` の違い、生成 AI 利用申告の意味を説明できる。

## ミニ講義
- Linux の基本操作では、`pwd` で現在地、`ls` で中身、`cd` で移動先を確認する。今どこで作業しているかを把握しないまま進めると、提出物や環境を見失いやすい。
- 仮想環境は、プロジェクトごとに使う Python と package を分離するための仕組みである。`which python` が `.venv` 配下を指しているかを見ると、本当に切り替わったか確認できる。
- 通常提出では、毎回の成果物を自分の `submission repo` の default branch に残す。まず `git status` で状態を見て、必要な file を `git add` し、`git commit` と `git push` で履歴を残す。
- この授業では、教材は共通 repo、通常提出は `submission repo` に分ける。生成 AI は使うこと自体よりも、「どこに使い、何を採用したか」が追えることが重要になる。

## 演習
### 基礎レベル
1. `session15_basic_commands.txt` を作成し、`pwd`, `ls`, `cd`, `mkdir`, `cat` をそれぞれ 1 回以上使う。各コマンドについて、実行した目的と出力の意味を 1 行ずつ記録する。
2. `q01.txt` から `q08.txt` を作成し、絶対パス、相対パス、`pwd`, ホームディレクトリ, `ls`, `cd`, `mkdir`, `cat` を自分の言葉で説明する。既存の演習問題と同じ題材だが、丸写しではなく今回の作業場所に結びつけて書く。
3. `.venv` を作成して有効化し、`session15_environment_checklist.md` に `## 作成した仮想環境`, `## python -V`, `## which python`, `## .venv を使う理由` の 4 見出しを書いて結果を記録する。
4. 共通 repo と自分の `submission repo` をそれぞれ clone する。`submission repo` で `session15_git_commands.md` を作成し、`git status`, `git add`, `git commit`, `git push`, `git log --oneline` を実行した順に記録する。
5. `session15_ai_policy_check.md` を作成し、共通 repo と submission repo の違い、通常提出の置き場、生成 AI を使った場合の申告項目を書く。

### 発展レベル
1. `session15_environment_checklist.md` に、`.venv` を使わない場合に起きる問題を 2 つ書く。単に「環境が汚れる」ではなく、再現性や他人の実行環境への影響に触れること。
2. `session15_git_commands.md` に `## status を見る理由` を追加し、`add` 前、`commit` 後、`push` 後で何を確認したいかを書く。
3. `session15_ai_policy_check.md` に、生成 AI 申告が必要な例と不要な例を 1 つずつ追加し、判断理由を書く。

## 確認ポイント
- `session15_basic_commands.txt` に現在地確認、移動後の現在地、仮想環境有効化後の `python -V` と `which python` がある。
- `q01.txt` から `q08.txt` の内容が file 名と対応している。
- `session15_environment_checklist.md` に 4 見出しが順番通りある。
- `session15_git_commands.md` に実行した git command と目的が順番に記録されている。
- `session15_ai_policy_check.md` に、repo の役割差と AI 申告判断の理由が書かれている。
- `.venv` を使う理由が「慣習だから」ではなく、環境分離の観点で説明されている。

## 詰まったときに見る資料
- [`../../README.md`](../../README.md)
- [`../../latex/markdown/ch00-installing-requirements.md`](../../latex/markdown/ch00-installing-requirements.md)
