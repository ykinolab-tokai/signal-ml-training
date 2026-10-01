# 環境構築と作業場所

環境構築の手順は [第1回 環境構築とワークフロー](../../handouts/01-environment-and-workflow.md) にまとめています。
Windows・WSL・macOSの分岐，GitHub認証，VS Codeの任意設定も第1回を参照してください。

## 教材repoと共有環境

作業の順序は，教材repoをcloneし，そのディレクトリに移動してから，第1回で指定したPython環境を用意し，実行確認する，です。
`uv sync --locked` は教材repoの `pyproject.toml` と `uv.lock` がある場所で実行します。
clone前の空のディレクトリで実行しないでください。
依存関係は教材repoの `.venv` にそろえ，以後の回でも同じ環境を使います。

## 演習の保存先

第1回の動作確認用 `exercises/` は教材repoに置きます。
第2〜13回の提出物は授業で案内されたprivateな提出repoに置き，`scripts/`，`data/`，`outputs/` を使います。
提出repoのURLは授業の案内に従ってください。
具体的な実行方法と入力データのコピー先は [handouts共通案内](../../handouts/README.md#作業場所と保存先) を参照してください。
