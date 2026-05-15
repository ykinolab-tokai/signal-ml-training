# 画像・音響信号処理と深層学習の技術基盤ゼミ

このリポジトリは、前提知識のない学生が 1 年で画像・音響信号処理と深層学習の技術基盤を身につけ、
自力で実装できるようにするための教材集です。
複数年にわたって繰り返し取り組むことで、基礎を定着させ、理解と実装の精度を高めることも狙います。

扱う範囲は、数学、信号処理、機械学習の理論とそれらの Python 実装です。
研究設計、研究評価、文章作法、発表作法などの研究スキルは、このリポジトリでは扱いません。

全28回の構成で設計します。

# 全 28 回の内容

### 第 1〜14 回

| 回 | 主題 | 内容 | 資料 |
| -- | -- | -- | -- |
| 1 | 環境構築とワークフロー | ターミナル，ファイルパス，ディレクトリ操作，Python 仮想環境，VS Code, Git | [01 環境構築とワークフロー](handouts/01-environment-and-workflow.md) |
| 2 | Python 基礎 | 変数、関数，条件分岐，繰り返し，list、dict、スライス | [02 Python 基礎](handouts/02-python-basics.md) |
| 3 | 数学基礎とNumpy, Matplotlib | 初等関数とグラフ，グラフの平行移動，Numpy配列，Matplotlibによるグラフの描画 | [03 数学基礎とNumpy, Matplotlib](handouts/03-math-basics-numpy-matplotlib.md) |
| 4 | 微分と積分 | 傾き，数値微分，偏微分，勾配，関数の極値，区分求積法，定積分と数値積分 | [04 微分と積分](handouts/04-differentiation-and-integration.md) |
| 5 | 線形代数 | ベクトル・行列の和，内積，行列積，ノルム，射影，線形変換，逆行列 | [05 線形代数](handouts/05-linear-algebra.md) |
| 6 | 確率 | 平均，分散，共分散，一様分布，正規分布，最尤推定 | [06 確率](handouts/06-probability.md) |
| 7 | 信号とサンプリング | 連続時間と離散時間，サンプリング周波数，サンプリング定理，信号に対する演算，量子化 | [07 信号とサンプリング](handouts/07-signals-and-sampling.md) |
| 8 | Fourier 変換 | DFT, 振幅スペクトルと位相スペクトル，窓関数 | [08 Fourier 変換](handouts/08-fourier-transform.md) |
| 9 | 線形時不変システムと畳み込み | 線形時不変システム，畳み込み，畳み込み定理，フィルタと周波数応答 | [09 線形時不変システムと畳み込み](handouts/09-lti-systems-and-convolution.md) |
| 10 | ノイズと信号復元 | 加法性ノイズ，信号とノイズのスペクトル，フィーナーフィルタ | [10 ノイズと信号復元](handouts/10-noise-and-signal-restoration.md) |
| 11 | 音響信号 | 音の読み込み，再生，リサンプリング，STFT，メルスペクトログラム | [11 音響信号](handouts/11-audio-signals.md) |
| 12 | 画像信号 | 2次元信号，画像の読み込み，表示，幾何変換，2次元DFT，2次元フィルタ | [12 画像信号](handouts/12-image-signals.md) |
| 13 | ミニプロジェクト | 画像: ノイズ除去，鮮鋭化，トーンマッピング，色変換，音: ノイズ除去，簡易イコライザ作成，音声のピッチ推定 | [13 画像 baseline ミニ実装](handouts/13-image-baseline-mini-implementation.md) |
| 14 | 統合確認 | 口頭技術確認、code walkthrough、repo 整理、PR / code review 体験 | [14 口頭確認と repo 整理](handouts/14-oral-check-and-repo-wrapup.md) |

### 第 15〜28 回

| 回 | 主題 | 内容 | 資料 |
| -- | -- | -- | -- |
| 15 | 年間方針と役割 | 年間方針、役割分担、共通 repo、生成 AI 運用、コーディング規約 | [15 年間方針と役割](handouts/15-annual-policy-and-roles.md) |
| 16 | research code engineering I | project 構造、config、logging、CLI | [16 research code engineering I](handouts/16-research-code-engineering-1.md) |
| 17 | research code engineering II | Git flow、PR、code review、簡単な test | [17 research code engineering II](handouts/17-research-code-engineering-2.md) |
| 18 | 数値線形代数の実装 | 最小二乗、SVD、PCA | [18 数値線形代数](handouts/18-numerical-linear-algebra.md) |
| 19 | autodiff と最適化 | backprop、optimizer、scheduler、勾配確認 | [19 autodiff と最適化](handouts/19-autodiff-and-optimization.md) |
| 20 | data pipeline engineering | Dataset、DataLoader、前処理、augmentation | [20 data pipeline engineering](handouts/20-data-pipeline-engineering.md) |
| 21 | 画像モデル基礎 | CNN、ResNet、U-Net、receptive field、normalization | [21 画像モデル基礎](handouts/21-image-model-basics.md) |
| 22 | 音響表現とモデル | STFT、mel、1D CNN、CRNN、Conformer の入口 | [22 音響表現とモデル](handouts/22-audio-representations-and-models.md) |
| 23 | transfer learning と fine-tuning | freeze、linear probe、adapter 的発想 | [23 transfer learning と fine-tuning](handouts/23-transfer-learning-and-fine-tuning.md) |
| 24 | representation learning | contrastive、masked prediction、embedding | [24 representation learning](handouts/24-representation-learning.md) |
| 25 | 年度別発展テーマ 第 1 回 | 選択テーマの問題設定、最小実装、評価観点 | [25 年度別発展テーマ：問題設定](handouts/25-advanced-theme-framing.md) |
| 26 | 年度別発展テーマ 第 2 回 | 選択テーマの比較実験、失敗分析、改善 | [26 年度別発展テーマ：比較実験](handouts/26-advanced-theme-comparison.md) |
| 27 | 年度別発展テーマ 第 3 回 | 選択テーマの整理、再利用可能な実装資産化、共有 | [27 年度別発展テーマ：整理](handouts/27-advanced-theme-wrapup.md) |
| 28 | 統合デモ | 技術デモ、code walkthrough、共通資産反映 | [28 技術デモと walkthrough](handouts/28-technical-demo-and-walkthrough.md) |

第 25〜27 回は、その年度に選んだ 1 つの発展テーマを 3 回に分けて扱う枠です。画像、音響、共通基盤を同じ年度にすべて扱う枠ではありません。候補テーマ別 handout は [`handouts/advanced/`](handouts/advanced/) に記録として残し、学生は 25〜27 の入口 handout から順に作業します。

## 配布資料とテンプレート

- 学生に直接配布する資料は、すべて Markdown ファイルとして [`handouts/`](handouts/) に配置します。
- 第 25〜27 回の発展テーマ枠は、年度ごとに画像・音響・共通基盤のうち 1 テーマだけを選び、3 回に分けて扱います。各回の入口 handout は `handouts/25-...md` から `handouts/27-...md` に置き、候補テーマ別 handout は [`handouts/advanced/`](handouts/advanced/) に置きます。
- 雛形は [`templates/session-template.md`](templates/session-template.md) に置きます。
- README から各資料へのリンクは相対リンクで統一します。GitHub 上でも clone 後でも追いやすい構成を保つためです。

## 4. 各回の標準的な時間配分

100 分は原則として次の配分で固定します。

1. 0〜10 分：冒頭小テストと前回内容の再確認。原則として生成 AI は使用しない。
2. 10〜20 分：前回の要点と典型的な詰まりを共有する。担当学生が短く報告する。
3. 20〜40 分：中核概念に関する mini lecture。教員の説明はここに集中する。
4. 40〜55 分：guided exercise。配布済みの `.py` script や最小コードを使い、その概念を自分で確かめる。前半は原則として生成 AI を使用しない。
5. 55〜80 分：レイヤ別演習。B3 は基礎課題、B4 は extension task、M は extension の遂行と mentor を担当する。生成 AI は利用可とするが、利用時は必ず申告する。
6. 80〜95 分：2〜3 組の short demo または口頭確認。スライドは不要とし、コード・図・出力のみを使う。
7. 95〜100 分：exit ticket の記入と次回課題の確認。

第 14 回と第 28 回のような統合回だけは、demo の比重を増やし、60 分程度を確保してよいものとします。
