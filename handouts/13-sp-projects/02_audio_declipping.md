# プロジェクト2：音響信号のDeclipping

# タスク説明

録音機器や増幅回路の許容振幅を超えると，音響信号の正負のピークが一定値で切り取られることがある．この非線形な劣化を **clipping** といい，clippingによって失われた波形を推定する処理を **declipping** という．

清浄信号を $x[n]$，正のclippingしきい値を $T>0$ とすると，hard clippingされた観測信号 $y[n]$ は

$$
y[n]=
\begin{cases}
T & x[n]\geq T\\
x[n] & -T<x[n]<T\\
-T & x[n]\leq -T
\end{cases}
$$

で表される．本プロジェクトの目的は，$y[n]$ と既知の $T$ から，清浄信号 $x[n]$ に近い復元信号 $\hat{x}[n]$ を推定することである．

次のサンプル集合を区別して扱う．

$$
\mathcal{R}=\{n\mid |y[n]|<T\}
$$

$$
\mathcal{C}_{+}=\{n\mid y[n]=T\},\qquad
\mathcal{C}_{-}=\{n\mid y[n]=-T\}
$$

$\mathcal{R}$ は値が信頼できるサンプル，$\mathcal{C}_{+}$ と $\mathcal{C}_{-}$ は復元が必要なサンプルである．未clippingサンプルを不必要に変更せず，clippingの符号としきい値に矛盾しない復元を行うことが重要である．

関連する既習事項は，振幅スケーリング，MSE，DFT，STFT，逆STFTである．ベースラインでは，未習内容である自己回帰モデルを簡単に導入する．

# データ

配布データは，モノラルの清浄音響信号から人工的にhard clippingを生成したものである．音声，単音楽器，複数楽器を含み，サンプリング周波数は全ファイルで統一されている．振幅値は浮動小数点数で扱い，clipping後に正規化してはならない．

想定するディレクトリ構成は次のとおりである．

```text
data/declipping/
├── train/
│   ├── clean/*.wav
│   ├── clipped/*.wav
│   └── metadata.csv
├── dev/
│   ├── clean/*.wav
│   ├── clipped/*.wav
│   └── metadata.csv
├── test/
│   ├── clipped/*.wav
│   └── metadata.csv
└── evaluate.py
```

`metadata.csv` には，ファイル名，信号種別，サンプリング周波数，clippingしきい値 $T$，clipped sample ratioが記録されている．`test` の清浄信号は最終評価スクリプトだけが参照する．

clippingの強さは，清浄信号の絶対値 $|x[n]|$ の分位点から $T$ を決めることで制御する．標準条件では，clipped sample ratioをおおむね次の4段階とする．

- 1 %
- 5 %
- 10 %
- 20 %

`train`，`dev`，`test` には，同じ原音から切り出した区間を重複して含めない．処理は原則として各ファイルの元の振幅スケールのまま行い，復元後に正解へ合わせるためのゲイン調整を行ってはならない．

# 評価方法

主評価尺度は，clippingされたサンプル集合

$$
\mathcal{C}=\mathcal{C}_{+}\cup\mathcal{C}_{-}
$$

だけを対象とした **clipped-region SDR** とする．

$$
\mathrm{SDR}_{\mathcal{C}}(\hat{x})
=10\log_{10}
\frac{\sum_{n\in\mathcal{C}}x[n]^2}
{\sum_{n\in\mathcal{C}}(x[n]-\hat{x}[n])^2+\varepsilon}
$$

ここで，$\varepsilon=10^{-12}$ とする．入力clipped信号 $y$ に対する改善量も計算する．

$$
\Delta\mathrm{SDR}_{\mathcal{C}}
=\mathrm{SDR}_{\mathcal{C}}(\hat{x})
-\mathrm{SDR}_{\mathcal{C}}(y)
$$

併せて，次の値を報告すること．

- 全サンプルを対象としたSDR
- clippingされたサンプルだけのRMSE

$$
\mathrm{RMSE}_{\mathcal{C}}
=\sqrt{\frac{1}{|\mathcal{C}|}
\sum_{n\in\mathcal{C}}(x[n]-\hat{x}[n])^2}
$$

- 信頼できるサンプルの最大変更量

$$
e_{\mathcal{R}}
=\max_{n\in\mathcal{R}}|\hat{x}[n]-y[n]|
$$

- clipping制約違反率

$$
r_{\mathrm{viol}}
=\frac{
|\{n\in\mathcal{C}_{+}\mid \hat{x}[n]<T\}|
+|\{n\in\mathcal{C}_{-}\mid \hat{x}[n]>-T\}|}
{|\mathcal{C}|}
$$

- 1秒の音響信号当たりの平均処理時間

主結果は，ファイルごとの $\mathrm{SDR}_{\mathcal{C}}$ を計算した後，条件ごとに平均して示す．信号種別別，clipped sample ratio別の結果も必ず示すこと．復元信号は正解信号に合わせてスケーリングせず，そのまま評価する．

波形全体だけでは差が分かりにくいため，成功例と失敗例について，clipping区間を含む20〜50 ms程度の波形拡大図，誤差信号，STFTスペクトログラムを示すこと．

# ベースライン

参照値として，何も処理しないclipped信号 $\hat{x}_{\mathrm{B0}}=y$ を必ず評価する．採点対象となるベースラインは，**双方向自己回帰補間**とする．

自己回帰（autoregressive，AR）モデルは，現在のサンプルを過去 $p$ サンプルの線形結合で近似するモデルである．

$$
x[n]\approx\sum_{k=1}^{p}a_kx[n-k]
$$

AR係数 $a_1,\ldots,a_p$ は，clipping区間の周辺にある信頼できるサンプルを用いて最小二乗法で推定する．過学習や不安定化を抑えるため，次のridge付き最小二乗問題を解く．

$$
\hat{\boldsymbol{a}}
=\arg\min_{\boldsymbol{a}}
\|\boldsymbol{A}\boldsymbol{a}-\boldsymbol{b}\|_2^2
+\lambda\|\boldsymbol{a}\|_2^2
$$

ベースラインでは，次の手順を用いる．

1. 連続するclippingサンプルを1つのclipping区間として検出する．
2. 各区間の左側と右側から，それぞれ最大 $M=8p$ 個の信頼できるサンプルを取得する．
3. 左側文脈から順方向ARモデル，時間反転した右側文脈から逆方向ARモデルを推定する．
4. 順方向予測 $f[j]$ と逆方向予測 $b[j]$ を計算する．区間長を $L$ とし，$j=0,\ldots,L-1$ に対して

$$
\tilde{x}[j]
=(1-w_j)f[j]+w_jb[j],\qquad
w_j=\frac{j+1}{L+1}
$$

で混合する．
5. 信頼できるサンプルは必ず観測値に戻す．
6. 正方向にclippingされたサンプルは $\hat{x}[n]\geq T$，負方向にclippingされたサンプルは $\hat{x}[n]\leq -T$ となるよう射影する．

標準パラメータは $p=32$，$M=256$，$\lambda=10^{-6}$ とする．文脈が不足する区間では，利用可能な範囲に応じてAR次数を下げる．ARモデルの推定が数値的に失敗した場合は，clipped信号をそのまま使用し，その区間をログに記録する．

`dev` では，$p\in\{8,16,32,64\}$ だけを探索してよい．その他のベースライン設定は固定する．

# 課題

1. 清浄信号とclipped信号を読み込み，clippingしきい値 $T$ と集合 $\mathcal{R}$，$\mathcal{C}_{+}$，$\mathcal{C}_{-}$ を求めなさい．波形，振幅ヒストグラム，clipping区間を可視化しなさい．
2. 処理なしの参照値B0について，$\mathrm{SDR}_{\mathcal{C}}$，全体SDR，$\mathrm{RMSE}_{\mathcal{C}}$ を計算し，clipped sample ratioとの関係を示しなさい．
3. 双方向AR補間のベースラインを実装しなさい．短い人工正弦波やチャープ信号を用いて，AR予測，順逆予測の混合，clipping制約への射影が意図どおり動作することを確認しなさい．
4. `dev` を用いてAR次数 $p$ を選択しなさい．$p$ と $\mathrm{SDR}_{\mathcal{C}}$，処理時間の関係を示し，選択理由を説明しなさい．
5. ベースラインを `test` に適用し，主評価尺度と副評価尺度を報告しなさい．信号種別別，clipped sample ratio別に結果を分け，短い区間と長い区間で性能がどう変わるか分析しなさい．
6. ベースラインの失敗例を少なくとも3件抽出し，ARモデルの不一致，区間長，周期性，過渡音，文脈不足などの観点から原因を説明しなさい．
7. ベースラインに対する改良を1つ以上実装しなさい．例として，区間長に応じたAR次数選択，予測の信頼度に基づく混合，境界のcrossfade，STFT係数の反復しきい値処理，複数窓長の統合，正弦波モデルによる補間などがある．STFTを用いる場合は，逆STFT後に必ず信頼サンプルとclipping制約へ再射影すること．
8. 改良手法のアブレーションを行い，どの要素がどの信号種別・clipping強度で有効だったかを示しなさい．全体SDRだけでなく，$\mathrm{SDR}_{\mathcal{C}}$ と制約違反率を必ず比較すること．
9. ベースラインと改良手法について，評価表，波形拡大図，誤差信号，スペクトログラム，処理時間をまとめなさい．復元音もWAV形式で保存し，コード，実行手順，乱数シードとともに提出しなさい．
