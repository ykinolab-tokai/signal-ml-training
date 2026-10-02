# 第01回 環境構築とワークフロー

## この回の目標

- ターミナルで現在地とファイルの保存先を確認する。
- 教材repoにPython 3.12の仮想環境を用意し，そのPythonでスクリプトを実行する。
- VS Codeでファイルを編集し，出力画像とGitの変更状態を確認する。

## 解説

OS別の準備を済ませてから，共通の手順で「認証 → 教材の取得 → Python環境の構築 → 動作確認」と進める。
WindowsではUbuntu 24.04 LTSのWSL2環境，MacではApple Silicon（Mシリーズ）を使う。
Intel Macはこの教材の固定依存関係に対応しないため，担当者に対応環境を相談する。
すでに設定済みの項目は，確認コマンドが成功すれば再設定しなくてよい。

ターミナルはコマンドの入力先，VS Codeはファイルの編集に使う。
`pwd` は現在地，`ls` はファイル一覧，`cd` は移動，`mkdir -p` はディレクトリ作成のコマンドである。
`~` は自分のホームディレクトリ，`.` は現在地を表す。エラーが出たら，次へ進む前に入力先と現在地を確認する。
用語・各操作の意味・トラブル対処は [環境構築の詳細](../textbook/markdown/ch00-installing-requirements.md) にまとめてある。

### 1. OS別の準備

#### Windows

**管理者として開いたPowerShell**で実行する。

```powershell
winget install --id Microsoft.VisualStudioCode --exact
winget install --id Microsoft.WindowsTerminal --exact
wsl --install --no-distribution
```

再起動を求められたら再起動し，PowerShellを開き直す。

```powershell
wsl --update
wsl --version
```

WSLが2.4.10以上であることを確認してから，Ubuntuを導入する。

```powershell
wsl --install -d Ubuntu-24.04
wsl -l -v
```

`Ubuntu-24.04` の `VERSION` が `2` であることを確認する。
Ubuntuの初回起動でユーザー名とパスワードを設定する。パスワード入力中は文字が表示されない。
VS Codeをインストールした後の**PowerShell**で，WSL拡張機能を入れる。

```powershell
code --install-extension ms-vscode-remote.remote-wsl
wsl ~ -d Ubuntu-24.04
```

ここからは**Ubuntuのターミナル**に入力する。

```bash
cat /etc/os-release
sudo apt update
sudo apt install git curl openssh-client
```

`VERSION_ID="24.04"` を確認する。続けてGitHub CLI（`gh`）を導入する。
以下は公式の配布元をaptに登録し，`gh` をインストールする手順である。各コマンドの成功を確認して進める。

```bash
sudo mkdir -p -m 755 /etc/apt/keyrings
sudo curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  -o /etc/apt/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
gh --version
```

以降もUbuntuのターミナルを使う。Macの節を飛ばし，「2. GitHubの認証とGitの設定」へ進む。

#### Mac（Apple Silicon）

**Macのターミナル**でHomebrewを導入する。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

終了時に表示される `Next steps` のPATH設定コマンドを実行してから，次へ進む。

```bash
brew --version
brew install --cask visual-studio-code
brew install git gh
gh --version
```

VS Codeで `Cmd+Shift+P` を押し，`Shell Command: Install 'code' command in PATH` を実行する。
ターミナルを開き直す。以降の共通手順はこのターミナルで行う。

### 2. GitHubの認証とGitの設定

GitHubアカウントを用意し，**UbuntuまたはMacのターミナル**で実行する。

```bash
gh auth login
```

表示される質問に対して，**接続先，Git通信の方式，使用するSSH鍵，認証方法**を次のように選ぶ。
質問の順序は，`gh` のバージョンや設定済みの項目によって変わる。

- 接続先：`GitHub.com`
- Git通信の方式：`SSH`
- SSH公開鍵：使用する鍵を選ぶ。鍵がなければ，案内に従って作成・登録する。秘密鍵は共有しない。
- 認証方法：ブラウザでのログインを選び，表示されたコードをGitHubの画面へ入力する。

```bash
gh auth status
ssh -T git@github.com
```

初回接続でホスト鍵の確認が出た場合の手順は [認証の詳細](../textbook/markdown/ch00-installing-requirements.md#github-と-git-の初期設定) を参照する。
自分のGitHubユーザー名を含む認証成功メッセージを確認する。`ssh -T` は成功時も終了コード1になる。
次の名前・メールアドレスを自分の情報に置き換え，commitに記録する著者情報を設定する。

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

### 3. 教材の取得とPython環境の構築

同じターミナルで教材repo（教材を管理するリポジトリ）を取得する。
すでに取得済みの場合は `git clone` を省略し，そのディレクトリへ移動する。

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:ykinolab-tokai/signal-ml-training.git
cd signal-ml-training
pwd
ls
git status
```

現在地が教材repoのルートで，`pyproject.toml` と `uv.lock` があることを確認する。
Python環境の管理ツール `uv` を導入する。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

終了時のPATH設定の案内に従い，ターミナルを開き直すか，表示された `source` コマンドを実行する。
教材repoで次を実行し，固定された依存関係から仮想環境 `.venv` を作る。

```bash
cd ~/workspace/signal-ml-training
uv --version
uv python install 3.12
uv sync --locked
source .venv/bin/activate
python --version
which python
```

Pythonが3.12系で，実行ファイルのパスが教材repoの `.venv/bin/python` であることを確認する。

### 4. VS Codeで編集する準備

同じターミナルで実行する。

```bash
code .
mkdir -p exercises
```

VS Codeに教材repoが開き，`pyproject.toml` と `uv.lock` が見えることを確認する。
Windowsではウィンドウが `WSL: Ubuntu-24.04` に接続していることも確認する。
ファイルはVS Codeで編集し，実行はターミナルから行う。
実行ボタン・デバッグ機能を使う場合だけ，[任意の設定](../textbook/markdown/ch00-installing-requirements.md#vs-code-の実行デバッグ機能で-venv-を使う任意) を行う。

## 演習

動作確認として，次のコードを教材repo内の `exercises/exc01_01.py` に保存する。追加の演習はない。

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

output_dir = Path("outputs/setup_check")
output_dir.mkdir(parents=True, exist_ok=True)

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.tight_layout()
plt.savefig(output_dir / "sin.png", dpi=150)
plt.close()

print("setup check completed")
```

**UbuntuまたはMacのターミナル**で教材repoへ移動し，仮想環境を有効にして実行する。

```bash
cd ~/workspace/signal-ml-training
source .venv/bin/activate
python exercises/exc01_01.py
git status
```

`setup check completed` と表示され，`outputs/setup_check/sin.png` に正弦波が保存されれば完了である。
画像はVS Codeで開いて確認する。`git status` の出力から，Gitが認識している変更の有無も確認する。
第1回のコードと画像は教材repoに置く。第2回以降の解答は，授業で指定された学生ごとの非公開の提出repoへ保存する。

## 確認ポイント

- WindowsではUbuntu 24.04 LTSがWSL2で動いている。
- GitHubの認証が成功し，教材repoを取得できた。
- Python 3.12と教材repoの `.venv/bin/python` を使っている。
- VS Codeで保存したスクリプトを実行し，完了メッセージと正弦波の画像を確認できた。
- `git status` が示す教材repoの変更状態を説明できる。

## 詰まったときに見る資料

- [環境構築とワークフローの詳細](../textbook/markdown/ch00-installing-requirements.md)：用語，OS別手順，認証，任意のVS Code設定，トラブル対処。
- [共通の作業場所と保存先](README.md#作業場所と保存先)：教材repoと提出repoの使い分け。
