# Windows
## Visual Studio Code と Windowsターミナルのインストール

1. Microsoft Storeから`アプリ インストーラー`をインストール

    - https://github.com/microsoft/winget-cli

1. 画面左下の検索から`PowerShell`と検索し，クリックしてPowerShellを起動

1. PowerShellで以下のコマンドを1行ずつ実行

    ```powershell
    winget install vscode
    winget install "Windows Terminal"
    ```

1. 画面左下の検索から`ターミナル`と検索し，クリックしてターミナルを開く
    - ターミナル上で`PowerShell`が起動し操作できる

1. 以降，PowerShellを起動する場合は，Windowsターミナルを介すると便利

    - タスクバーなどにピンどめしておくと良い

## WSL2 (Ubuntu) のインストール
1. PowerShellを管理者権限で起動し，以下のコマンドを実行
    
    ```powershell
    wsl --install
    ```

    - 2026年4月時点の標準的な LTS は Ubuntu 24.04 LTS です．明示的に選びたい場合は `wsl --list --online` で候補を確認し，`wsl --install -d Ubuntu-24.04` を使います．
    
2. PCを再起動する
3. windowsターミナルを起動し，画面上部のv字ボタンから Ubuntu を選択して起動する
4. ユーザ名とパスワードの入力を求められるので，好きなものを入力

    - パスワード入力の際，画面には入力した文字が全く表示されないが，きちんと入力されているので気にせず続けること
    - 必要なら `wsl -l -v` で Ubuntu が WSL2 で動いていることを確認する

## Python3 のインストール
1. Ubuntuを起動し，以下のコマンドを実行

    ```bash
    sudo apt update
    sudo apt install python3 python3-venv python3-pip
    ```

    - WSL 上の Ubuntu 24.04 LTS では標準の `python3` は 3.12 系です

## Git のインストール

1. Ubuntuを起動し，以下のコマンドを実行

    ```bash
    sudo apt update
    sudo apt install git 
    ```

## GitHub CLIのインストール

1. Ubuntuを起動し，以下のコマンドを実行

    ```bash
    type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt update \
    && sudo apt install gh -y
    ```


# Mac

## Homebrew のインストール
1. ターミナルを起動し，以下のコマンド (最新のコマンドは[HomebrewのWebページ](https://brew.sh/ja/)を参照) を実行
    ```
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

    1. パスワードを入力
    2. インストールされるもののリストが表示されるので，続ける場合はEnterキーを入力
    3. インストールが完了すると実行すべきコマンドが表示されるので，それらを実行

## Visual Studio Code のインストール
1. ターミナルを起動し，以下のコマンドを実行

    ```
    brew install --cask visual-studio-code
    ```
1. ターミナルからVSCodeを起動できるように，VSCodeへのPATHを `.zshrc` に追加

    ```
    echo 'export PATH="/Applications/Visual Studio Code.app/Contents/Resources/app/
bin":$PATH' >> ~/.zshrc
    ```

## Python3 のインストール
1. ターミナルを起動し，以下のコマンドを実行

    ```
    brew install python@3.12
    ```

    - この repo では Python 3.12 系を標準にします．`brew install python` で上流最新の 3.14 系を入れる方法もありますが，まずは共有環境と揃える方を優先してください

## Git のインストール
1. Macは，Gitがデフォルトでインストールされている．ターミナルで以下のコマンドを実行し，Gitのバージョンが表示されれば問題ない
    ```
    git --version
    ```
1. もし，Gitがインストールされていない場合は，以下のコマンドを実行
    ```
    brew install git
    ```

## GitHub CLIのインストール
1. ターミナルを起動し，以下のコマンドを実行

    ```
    brew install gh
    ```

# 共通

## GitHub の設定
1. [GitHub](https://github.co.jp/)から、「[GitHubに登録する](https://github.com/join)」という項目に飛ぶことができるのでそこでGitHubに登録する。
注）ユーザアカウント名は英語で登録する。
2. https://github.co.jp/pricing の一番左の項目である「Developer」から、「学生の方は、[Student Developer Pack](https://education.github.com/)の一環として、無料で利用できます。」 という項目を選択する。
3. そこのページから、「[Get Your Pack](https://education.github.com/pack)」という項目を選択し、自分のアカウント情報、学校名など必要な情報を入力する。
4. GitHubに登録したメールアドレスに「Powerup get! Welcome to the Student Developer Pack.」というタイトルのメールが届いたら無料の学生パックに登録が完了したこととなる。
5. 管理者にお願いして ykinolab-tokai のメンバーに招待してもらう。
1. Ubuntuを起動し，以下のコマンドを実行

    ```bash
    gh auth login
    ```

## 作業ディレクトリの準備

まずはホームディレクトリの下に授業用の作業ディレクトリを作り，その中にリポジトリを置くようにします．

```bash
mkdir -p ~/workspace
cd ~/workspace
```

## Python 仮想環境と依存関係

Python では，プロジェクトごとに依存関係を分けるために仮想環境を使います．
この repo では，個別に `pip install` を並べるよりも，`pyproject.toml` と `uv.lock` から共有環境を再現する方を標準にします．

上流の最新 stable は 2026年4月時点で Python 3.14.4 ですが，この repo は `pyproject.toml` の `requires-python = ">=3.12,<3.13"` に合わせて Python 3.12 系を使います．
NumPy や matplotlib，pandas，scikit-learn，Pillow，soundfile，PyTorch 系も更新が速いため，本文に固定値を並べるのではなく，正確な依存関係は `pyproject.toml` と `uv.lock` を参照してください．

1. `uv` を入れていない場合は，まずインストールします

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. repo ルートで共有環境を作ります

```bash
uv python install 3.12
uv sync
source .venv/bin/activate
```

作業を終えたら

```bash
deactivate
```

で抜けられます．

## 最初の動作確認

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

## Git の初期設定

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global --list
```

## リポジトリを clone して動作確認する

```bash
git clone <repository-url>
cd <repository-name>
pwd
ls
git status
```

## VS Code から開く

```bash
code .
```

## 詰まりやすい点

- 仮想環境に入っていないまま `pip install` している
- Windows 側と WSL 側のどちらで作業しているか分からなくなっている
- `python` と `python3` が別物で混乱している
- ターミナルでいる場所と VS Code で開いている場所が一致していない
- 図は表示されたが保存されておらず，提出物になっていない

まずは次を確認します．

```bash
which python
python --version
pwd
git status
```
