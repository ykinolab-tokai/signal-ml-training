# 第15回 研究室オンボーディング：Linux・SSH・ディレクトリ・仮想環境・生成AI運用

- 対象: B3
- 種別: 後期固定ブートキャンプ

## この回の目標
- 研究室サーバへ SSH 接続し、現在地確認、ディレクトリ作成、ファイル作成を自力で実行できる。
- 仮想環境を作成し、有効化後の Python を確認できる。
- 共通 repo と `submission repo` の違い、生成 AI 利用申告の意味を説明できる。

## ミニ講義
- Linux の基本操作では、`pwd` で現在地、`ls` で中身、`cd` で移動先を確認する。今どこで作業しているかを把握しないまま進めると、提出物や環境を見失いやすい。
- 仮想環境は、プロジェクトごとに使う Python と package を分離するための仕組みである。`which python` が `.venv` 配下を指しているかを見ると、本当に切り替わったか確認できる。
- この授業では、教材は共通 repo、通常提出は `submission repo` に分ける。生成 AI は使うこと自体よりも、「どこに使い、何を採用したか」が追えることが重要になる。

## 演習
### 基礎レベル
1. SSH 接続後、`session15_practice` ディレクトリを作成し、その中へ移動する。
2. `session15_basic_commands.txt` を作成し、次のコマンドの結果を順に貼る。
   - `pwd`
   - `ls`
   - `cd session15_practice` 後の `pwd`
   - 仮想環境有効化後の `python -V`
   - 仮想環境有効化後の `which python`
3. `q01.txt` から `q08.txt` を作成し、各 file に次の 1 行を書く。
```text
q01.txt: pwd は現在の作業ディレクトリを表示する。
q02.txt: ls は現在のディレクトリにあるファイルとディレクトリを表示する。
q03.txt: cd は移動先のディレクトリへ移動する。
q04.txt: mkdir は新しいディレクトリを作成する。
q05.txt: touch は空ファイルを作成するか既存ファイルの時刻を更新する。
q06.txt: cat はファイルの中身を表示する。
q07.txt: submission repo は通常提出を保存する場所である。
q08.txt: 共通 repo は教材と見本コードを参照する場所である。
```
4. `python3 -m venv .venv` を実行し、`source .venv/bin/activate` の後に `python -V` と `which python` を確認する。
5. `session15_environment_checklist.md` を作成し、次の 4 見出しをこの順で書く。
   - `## SSH 接続確認`
   - `## ディレクトリ操作確認`
   - `## 仮想環境確認`
   - `## つまずいた点`
6. 各見出しの下に、実行したコマンドと結果を 1 行ずつ書く。

### 発展レベル
1. `session15_ai_policy_check.md` を作成し、次の 3 見出しをこの順で書く。
   - `## 共通 repo と submission repo の違い`
   - `## 今回の提出物で生成AI申告が必要か`
   - `## README.md で確認したルール`
2. `README.md` を読み、`## README.md で確認したルール` には次の 2 文を必ず含める。
   - `通常提出は submission repo の default branch に commit と push を行う。`
   - `生成 AI を使った場合は用途と採用箇所を申告する。`
3. `## 今回の提出物で生成AI申告が必要か` には、今回 AI を使ったかどうかに加え、「なぜその判断になるか」を 2 行以内で書く。
4. `session15_environment_checklist.md` の最後に 2 行追加し、`which python` が `.venv` を指していないと何が困るかを書く。

## 確認ポイント
- `session15_basic_commands.txt` に現在地確認、移動後の現在地、仮想環境有効化後の `python -V` と `which python` がある。
- `q01.txt` から `q08.txt` の内容が file 名と対応している。
- `session15_environment_checklist.md` に 4 見出しが順番通りある。
- `session15_ai_policy_check.md` に、repo の役割差と AI 申告判断の理由が書かれている。
- `.venv` を使う理由が「慣習だから」ではなく、環境分離の観点で説明されている。

## 詰まったときに見る資料
- [`../../README.md`](../../README.md)
- [`../../latex/markdown/ch00-installing-requirements.md`](../../latex/markdown/ch00-installing-requirements.md)
