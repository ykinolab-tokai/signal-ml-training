# Issue #64: 作業リポジトリ導入後の共通手順

日付: 2026-10-09
対象: https://github.com/ykinolab-tokai/signal-ml-training/issues/64
教材の基準: `e2782790b0e9255e9fd29d9c276a3df8106f2686`
作業テンプレートの基準: `2fecb8e314ee1066ac4bfc750a25add0aed65f26`

## 決定と変更

- ユーザーの指定「別レポートは必要な課題のみ」を反映。通常の数値・shape・図の説明はコードコメントまたは既存の結果ファイルに残し、転記を求めない。
- 第03・06・11・13・16・18〜24回の別レポート／転記用記録を整理。既存の計算・図・解釈の要求は維持し、記録先の変更に伴う確認ポイントも同期した。
- 発展テーマ候補3件と第26回は、複数条件をまとめて参照する比較記録を残す。候補レポート・第26回記録・数値出力は相互参照でよく、同じ内容を転記しない。PRチェックリストや再実行手順も既存の結果を参照できる。
- 作業リポジトリは研究室所有の非公開 `signal-ml-work-<GitHubユーザー名>`、PC上は `~/workspace/signal-ml-work`。標準コードは `exercise/excXX_YY.py`。各回の明示指定を優先し、フォルダなしで指定したファイルは作業リポジトリのルートに置く。
- 第1回の教材リポジトリ内 `exercises/exc01_01.py` は環境確認用の例外。第2回以降・補足の課題は作業リポジトリを基準にする。
- 環境の準備・更新は教材リポジトリ、実行は作業リポジトリで教材側 `.venv` を有効化。第09・10・12回と補足教材の独立環境／個別pip導入を共通手順へ接続した。
- 第14・17回と補足ch01でPRの宛先を同じ作業リポジトリのmainと明示。未commit変更、レビュー待ち、マージ後、次回が未マージ変更に依存する場合を説明した。
- 補足ch00・01・02・03・11・12の変更箇所をMarkdownとLaTeXで同期。全28回の構成、既存の評価・AI利用方針は変更しない。

## 関連リポジトリ

[作業テンプレートリポジトリ](https://github.com/ykinolab-tokai/signal-ml-work-template)のREADMEも修正した。
初期設定の前提を教材リポジトリ取得・Python環境準備までに限定し、第1回で作成したリポジトリをcloneする手順、保存規則、レポート方針、環境更新、PR後の復帰を同期した。
教材リポジトリと作業テンプレートリポジトリは別々のPRで反映する。Issue #64の完了には両方の変更が必要。関連PRは各PRの本文から参照する。

## 検証

- `python3 -m unittest discover -s tests -v`: 成功（HTML変換テスト1件）。
- `python3 scripts/build_handouts.py`: 成功（37資料とindex、ZIP）。生成先は `build/handouts/` と `build/handouts.zip`。
- `cd textbook && latexmk -interaction=nonstopmode -halt-on-error main.tex`: PDF生成成功。管理対象の `textbook/main.pdf` を更新。
- 追加した相対リンク53件の参照先、および第1回・共通手順・READMEへのGitHub形式の見出しアンカーを照合。
- 一時作業リポジトリで教材の `.venv` を有効化し、作業テンプレートの `exercise/check_environment.py` を実行。Pythonのパス、`sine.png`、`sine.csv` の生成を確認。NumPy、SciPy、soundfile、sounddevice、librosa、Pillowのimport成功。
- 一時bare リポジトリと作業・レビュー用cloneで補足ch01のGit操作を検証。レビュー待ちの変更を練習ブランチに保持してmainへ戻る場合、マージ後に `git pull --ff-only origin main` で取得する場合とも成功。
- HTMLの内容・リンクと、PDFの作業場所／共有環境／PR後の復帰手順の表示を確認。
- 両リポジトリの `git diff --check` を確認。

## 検証範囲と残る別課題

学生アカウントでのGitHub権限、Windows/MacのGUI、音声機器、全演習の実行は未検証。
HTMLは生成内容とリンクを検査し、ブラウザでの目視確認は行っていない。
第09回の入力ファイル・未定義変数、第13回の主題、第15回の未整備な運用規定等は既存の #15・#19・#21 で扱う。今回の修正だけでこれらのIssue全体が解消したとは扱わない。
