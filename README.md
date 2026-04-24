# 画像・音響信号処理と深層学習の技術基盤ゼミ

このリポジトリは、ゼミを **「画像・音響信号処理と深層学習の技術基盤を身につけ、実装できるようにする場」** として運営するための共通基盤です。研究設計・研究評価・文章作法・発表作法は別枠で扱い、このリポジトリでは **数学、信号処理、プログラミング、PyTorch 実装、モデル理解、コード品質、再利用可能な実装資産の整備** のみを扱います。`signal-ml-training` は教材配布・見本コード・共通資産のための repo であり、B3 の通常提出先そのものとしては使いません。

年間 28 回は、**前期 14 回を B4・M 向けの発展枠、後期 14 回を新規配属 B3 向けの固定ブートキャンプ** として設計します。前期は、立て直しが完了した B4・M が共通基盤の上に発展的な実装力を積む場です。後期は、新しく入ってくる B3 に毎年同じコアを与え、B4・M は mentor / reviewer として参加します。

## 配布資料とテンプレート

- 学生に直接配布する配布資料は、すべて Markdown ファイルとして [`materials/spring`](materials/spring) と [`materials/autumn`](materials/autumn) に配置します。
- 前期の固定配布資料は `materials/spring/01-...md` から `10-...md` と `14-...md`、後期配布資料は `materials/autumn/15-...md` から `28-...md` とし、回番号と一致させます。
- 前期 11〜13 回の発展枠は、年度ごとに画像・音響・共通基盤のうち 1 テーマだけを選び、3 回に分けて扱います。候補 handout は回番号を付けず、テーマ別の記録として `materials/spring/advanced-...md` に置きます。
- 雛形は [`templates/spring-session-template.md`](templates/spring-session-template.md) と [`templates/autumn-session-template.md`](templates/autumn-session-template.md) に置きます。
- README から各ディレクトリへの案内は相対リンクで統一します。GitHub 上でも clone 後でも追いやすい構成を保つためです。

## 1. 年間構成の基本方針

- 後期 14 回は毎年固定します。
- 前期 14 回は、B4・M が共通基盤の上で発展的な実装力を積む枠とします。
- 前期の具体的な題材は、その年度に使う配布資料で確定します。

## 1.5. 提出と Git 運用の基本方針

- `signal-ml-training` は、教材配布、見本コード、テンプレート、共通資産の管理に使う共通 repo とします。
- B3 の通常提出は、この共通 repo ではなく、学生ごとの private な submission repo に残す方針とします。
- B3 は学期の最初に共通 repo と自分の submission repo をそれぞれ 1 回 clone し、その後の通常提出では `edit -> commit -> push` を反復します。
- B3 の毎週の提出で `fork` や `pull request` は必須にしません。まずは CUI で `clone`、`pull`、`commit`、`push` を確実に使えることを優先します。
- `pull request` は第 28 回の code walkthrough と repo 整理の題材として扱いますが、B3・B4・M のいずれにも継続的な共通 repo への PR 提出は課しません。
- 教員・TA は、B3 の採点時には学生の submission repo の default branch を確認します。

## 2. 各学年の責務

### B3 の責務

B3 の責務は、理解したふりをせず、基礎を確実に固めることです。研究上の独自性ではなく、技術基盤の確立を求めます。

- 毎回の事前課題を前日までに完了する
- 冒頭小テストに参加する
- 配布された `.py` script を自力で再実行できる
- 各回の基礎演習を、自分の submission repo 上に残す
- 後期の終わりまでに、画像 1 件、音響 1 件の baseline を動かし、前処理から forward までを口頭で説明できる
- 生成 AI を使った場合は、用途と採用箇所を申告する

B3 に mentor 業務は課しません。最優先は自分の基礎形成です。

### B4 の責務

B4 の責務は、自分でできることを増やす段階から、後輩に説明できる段階へ移ることです。

- 前期では毎回、基礎課題に加えて extension task を 1 つ担当する
- 後期では B3 の演習をレビューし、少なくとも週 1 回はコメントを返す
- 後期では B3 の「詰まり」を技術的に切り分ける
- 生成 AI を使う場合でも、後輩には「答え」ではなく「確認の仕方」を示す

### M1 の責務

M1 は、技術理解の整理役と、後輩指導の設計役を担います。単なる TA ではなく、研究室の実装文化を維持する役目です。

- 前期で少なくとも 2 回、発展回の演習設計または補助資料作成を担当する
- B4 が担当する B3 mentor 業務を監督し、必要に応じて介入する
- B4 より一段深い extension task に取り組む
- 年間で少なくとも 1 件、研究室全体で再利用できる技術ノートやテンプレートを残す

### M2 の責務

M2 の責務は、個別の課題遂行よりも、研究室全体の品質管理と知識の継承です。

- 前期の発展回で少なくとも 2 回、技術デモまたは code walkthrough を主導する
- B4・M1 が作った資料、課題、共通コードを品質確認する
- 教員不在でも回る最低限の運営手順を点検し、必要に応じて更新を提案する
- B3 ではなく B4 / M1 の mentor として振る舞う

M2 がいない年度は、この役割を最上級の M1 が代替します。

## 3. 評価基準

このゼミは、研究テーマの新規性や研究成果ではなく、**技術基盤の獲得と研究室への貢献** で評価します。評価は相対評価ではなく到達評価です。

### 共通の必須条件

- 出席率 80% 以上
- 提出物はすべて Git 管理されていること
- B3 の通常提出は自分の submission repo で管理し、共通 repo への直コミットや通常提出目的の PR は行わないこと
- 生成 AI 利用時は申告欄を埋めること
- 指名時に、自分のコードまたは保存済み出力ファイルの一部をその場で説明できること

これを満たさない場合は、点数にかかわらず不合格扱いとします。

### B3 の評価基準

- 事前課題・小テスト：25 点
- 毎回の基礎演習提出：35 点
- ミニ実装 2 件（画像 1、音響 1）：25 点
- 口頭説明・live debug：15 点

合格条件は 60 点以上に加え、後期末までに以下を満たすことです。

- Linux / SSH、Python、Git、PyTorch の基本操作ができる
- 画像 baseline と音響 baseline を一つずつ動かせる
- 前処理、モデル入力、出力の意味を自分の言葉で説明できる

### B4 の評価基準

- 前期の extension task：30 点
- 週次参加・小テスト：15 点
- 後期の B3 mentor・レビュー：25 点
- 口頭説明・live modification：30 点

合格条件は 70 点以上に加え、以下を満たすことです。

- B3 の演習レビューを継続的に行う
- 自分が担当した extension task を後輩に説明できる

### M1 の評価基準

- 回の設計・進行担当：25 点
- 発展課題または技術ノート：25 点
- B4 mentor の監督とレビュー支援：30 点
- 口頭説明・技術デモ：15 点

合格条件は 75 点以上に加え、以下を満たすことです。

- 少なくとも 2 回、進行または技術説明の中心を担う
- 年間で少なくとも 1 件、研究室全体で再利用可能な資料を残す

### M2 の評価基準

- 運営・品質管理：25 点
- 技術デモ・code walkthrough：25 点
- B4 / M1 への mentor：25 点
- 運営手順の点検と引き継ぎ整備：10 点
- 口頭確認・技術面談：15 点

合格条件は 80 点以上に加え、以下を満たすことです。

- 年間で少なくとも 2 回、研究室全体に向けた技術デモを主導する
- handoff 可能な運営資産を 1 件以上残す

## 4. 各回の標準的な時間配分

100 分は原則として次の配分で固定します。

1. 0〜10 分: 冒頭小テスト。前回内容の再確認。原則として生成 AI 不使用。
2. 10〜20 分: 前回の要点と典型的な詰まりの共有。担当学生が短く報告。
3. 20〜40 分: 中核概念に関する mini lecture。教員が話す時間はここに集中。
4. 40〜55 分: guided exercise。配布済みの `.py` script や最小コードを使い、その概念を自分で確かめる。この前半も原則として生成 AI 不使用。
5. 55〜80 分: レイヤ別演習。B3 は基礎課題、B4 は extension task、M は extension の遂行と mentor を担当。ここは生成 AI 利用可だが、利用時は必ず申告。
6. 80〜95 分: 2〜3 組の short demo または口頭確認。スライド不要、コード・図・出力のみを使う。
7. 95〜100 分: exit ticket の記入と次回課題の確認。

第 14 回と第 28 回のような統合回だけは、demo の比重を増やして 60 分程度確保してよいものとします。

## 5. 全 28 回の内容

### 前期 14 回：B4・M 向け発展枠

| 回 | 内容 | 配布資料 |
| -- | -- | -- |
| 1 | 年間方針、役割分担、共通 repo、生成 AI 運用、コーディング規約 | [01 年間方針と役割](materials/spring/01-annual-policy-and-roles.md) |
| 2 | research code engineering I：project 構造、config、logging、CLI | [02 research code engineering I](materials/spring/02-research-code-engineering-1.md) |
| 3 | research code engineering II：Git flow、PR、code review、簡単な test | [03 research code engineering II](materials/spring/03-research-code-engineering-2.md) |
| 4 | 数値線形代数の実装：最小二乗、SVD、PCA | [04 数値線形代数](materials/spring/04-numerical-linear-algebra.md) |
| 5 | autodiff と最適化：backprop、optimizer、scheduler、勾配確認 | [05 autodiff と最適化](materials/spring/05-autodiff-and-optimization.md) |
| 6 | data pipeline engineering：Dataset、DataLoader、前処理、augmentation | [06 data pipeline engineering](materials/spring/06-data-pipeline-engineering.md) |
| 7 | 画像モデル基礎：CNN、ResNet、U-Net、receptive field、normalization | [07 画像モデル基礎](materials/spring/07-image-model-basics.md) |
| 8 | 音響表現とモデル：STFT、mel、1D CNN、CRNN、Conformer の入口 | [08 音響表現とモデル](materials/spring/08-audio-representations-and-models.md) |
| 9 | transfer learning と fine-tuning：freeze、linear probe、adapter 的発想 | [09 transfer learning と fine-tuning](materials/spring/09-transfer-learning-and-fine-tuning.md) |
| 10 | representation learning：contrastive、masked prediction、embedding | [10 representation learning](materials/spring/10-representation-learning.md) |
| 11 | 年度別発展テーマ 第 1 回：選択テーマの問題設定、最小実装、評価観点 | [発展テーマ handout 一覧](materials/spring/advanced-themes.md) |
| 12 | 年度別発展テーマ 第 2 回：選択テーマの比較実験、失敗分析、改善 | [発展テーマ handout 一覧](materials/spring/advanced-themes.md) |
| 13 | 年度別発展テーマ 第 3 回：選択テーマの整理、再利用可能な実装資産化、共有 | [発展テーマ handout 一覧](materials/spring/advanced-themes.md) |
| 14 | 技術デモ・code walkthrough・共通資産反映 | [14 技術デモと walkthrough](materials/spring/14-technical-demo-and-walkthrough.md) |

前期 11〜13 回は、その年度に選んだ 1 つの発展テーマを 3 回に分けて扱う枠です。画像、音響、共通基盤を同じ年度にすべて扱う枠ではありません。候補 handout は記録として残しますが、回番号は付けません。

### 後期 14 回：B3 向け固定ブートキャンプ

| 回 | 内容 | 配布資料 |
| -- | -- | -- |
| 15 | 研究室オンボーディング：Linux、SSH、ディレクトリ、仮想環境、repo clone、提出用 Git 最小運用 | [15 研究室オンボーディング](materials/autumn/15-onboarding.md) |
| 16 | Python script の基礎：変数、list、dict、関数、例外、Path、ファイル保存 | [16 Python script の基礎](materials/autumn/16-python-script-basics.md) |
| 17 | NumPy・PyTorch Tensor・matplotlib の基礎 | [17 配列と tensor の基礎](materials/autumn/17-array-and-tensor-basics.md) |
| 18 | 線形代数の基礎：ベクトル、内積、行列積、ノルム | [18 線形代数の基礎](materials/autumn/18-linear-algebra-basics.md) |
| 19 | 確率統計の基礎：平均、分散、分布、サンプリング、基本指標 | [19 確率統計の基礎](materials/autumn/19-probability-and-statistics-basics.md) |
| 20 | 離散信号、畳み込み、フィルタの基礎 | [20 離散信号と畳み込み](materials/autumn/20-discrete-signals-and-convolution.md) |
| 21 | Fourier 変換と STFT の基礎 | [21 Fourier 変換と STFT](materials/autumn/21-fourier-and-stft-basics.md) |
| 22 | 音響データの基礎：波形、spectrogram、mel、簡単な augmentation | [22 音響データの基礎](materials/autumn/22-audio-data-basics.md) |
| 23 | 画像データの基礎：画像を配列として扱う、正規化、リサイズ、簡単なフィルタ | [23 画像データの基礎](materials/autumn/23-image-data-basics.md) |
| 24 | PyTorch による model 実装の基礎：Module、Dataset、DataLoader、batch、logits | [24 PyTorch model と Dataset の基礎](materials/autumn/24-pytorch-model-dataset-basics.md) |
| 25 | training loop の基礎：loss、optimizer、epoch、validation、overfitting | [25 training loop の基礎](materials/autumn/25-training-loop-basics.md) |
| 26 | 音響ミニ実装：小規模 classification baseline | [26 音響 baseline ミニ実装](materials/autumn/26-audio-baseline-mini-implementation.md) |
| 27 | 画像ミニ実装：小規模 classification baseline | [27 画像 baseline ミニ実装](materials/autumn/27-image-baseline-mini-implementation.md) |
| 28 | 口頭技術確認・code walkthrough・repo 整理・PR / code review 体験 | [28 口頭確認と repo 整理](materials/autumn/28-oral-check-and-repo-wrapup.md) |

## 6. 前期と後期での学年ごとの役割の違い

前期は B4 と M が主役です。B4 は extension の実装担当、M は設計とレビュー担当です。後期は B3 が学習主体になり、B4 は direct mentor、M は supervisor になります。つまり、**B4 は「学ぶ側」から「教える側」へ、M は「教える側」から「場を設計する側」へ移る** ことになります。

## 7. 生成 AI の扱い

- 小テストと guided exercise 前半では使わせない
- 演習後半のデバッグ、英語確認、API 確認では使用可
- コードや文章に採用した場合は申告必須
- public な生成 AI に未公開データや未公開コードを入力しない
- AI を使った提出物でも、その場で説明できなければ評価しない

## 8. この設計の要点

- 後期は毎年固定にして、新配属者の入口を安定させる
- 前期は発展枠として、基盤の上に応用実装を積む
- 学年差は内容ではなく責務で処理する

B3 は学ぶ、B4 は学びつつ教える、M は設計し支える、という役割分担により、教員 1 名体制でも回しやすい運営を目指します。
