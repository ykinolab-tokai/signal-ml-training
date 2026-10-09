# 配布用HTMLの生成

Python 3.12 と Pandoc を使い，`handouts/` 以下の全MarkdownファイルをHTMLに変換する。
Pythonの追加パッケージは不要。Pandoc 3.1.3で動作確認済み。

リポジトリのルートで実行する。

```bash
python3 scripts/build_handouts.py
```

生成物は次の場所に保存される。

- `build/handouts/index.html`: 全資料への入口
- `build/handouts/`: 各回・補足資料を単独で配布できるHTML
- `build/handouts.zip`: 全資料をまとめたZIP（展開すると `handouts/` ディレクトリになる）

各回を配布するときは，対応するHTMLファイルを1つだけ渡す。CSS・画像・ライセンス表記はHTML内に埋め込まれており，別ファイルの添付は不要。
たとえば第1回は `build/handouts/01-environment-and-workflow.html` だけを配布する。
全資料をまとめて配布する場合はZIPを渡し，展開後の `index.html` を入口として使う。
現在のChrome・Edge・Firefox・Safariなど，MathMLに対応するブラウザを使用する。
本文・目次・数式の表示にインターネット接続は不要。他の回や教科書，公式ドキュメントへの参照リンクを開くには接続が必要。

## 変換時の扱い

- Markdownを原稿として維持する。コードブロックは表示用に変換し，実行しない。
- 同じ資料の見出しへのリンクはHTML内のリンクとして保持し，存在を検査する。
- 他の回・教科書・リポジトリのREADMEへのリンクはGitHub上の `main` の該当資料に向ける。単独配布時に隣のHTMLファイルを必要としない。
- 一覧ページの資料リンクは，まとめて閲覧するためのローカルHTMLへの相対リンクとする。
- Pandocの `--embed-resources` でCSSと画像などの表示用リソースをHTML内に埋め込む。外部URLの画像を原稿に追加した場合は，生成時に取得できる必要がある。
- 数式はMathMLに変換する。Pandoc 3.1.3で未対応の `\xleftrightarrow{上ラベル}` は，変換時に同じ意味の矢印と上ラベルの定義を補う。原稿は変更しない。
- 共通の見た目は `scripts/html/handouts.css`，ページ構造は `scripts/html/page.html` で管理する。
- 変換警告，存在しない参照先，配布物内の見出しリンク切れがあれば失敗する。外部サイトの到達性は検査しない。
- 再生成時は，前回の生成物であることを確認して `build/handouts/` を置き換える。削除された原稿のHTMLも除去するため，生成先に手作業でファイルを追加しない。
- `build/` はGitの管理対象外。HTMLを修正したい場合は原稿またはテンプレートを修正して再生成する。

## 検証

```bash
python3 -m unittest discover -s tests -v
python3 scripts/build_handouts.py
```

テストではHTMLを1枚だけ別のディレクトリにコピーし，CSS・画像・ライセンスの埋め込み，数式・日本語見出しへのリンク，コードの非実行，再生成，失敗時の前回出力の保持を確認する。
全資料の変換では，各HTMLのローカルリンクと見出しの存在を検査する。

参考：[Pandocの導入](https://pandoc.org/installing.html)，[MathMLとブラウザ対応](https://developer.mozilla.org/en-US/docs/Web/MathML)。
