# 第01回 環境構築とワークフロー

- ターミナルで現在地，ファイルパス，ディレクトリ構造を確認する。
- Python 仮想環境を作成し，実行に使われる Python を確認する。
- VS Code で repo を開き，ターミナルとエディタを行き来しながら作業する。
- Git で作業状態を確認し，変更を記録する最小手順を身につける。

## 解説

### キーワード
- **Operating System**

    コンピュータ全体を管理するための基本ソフトウェア。
    代表例として、Windows、macOS、Linuxがある。

- **ファイルシステム**

    ファイルやフォルダを保存し、整理し、読み書きするための仕組み。

- **Command line interface (CLI)**
    
    コマンドを入力してコンピュータを操作する仕組み．
    操作を自動化しやすいため、開発や研究でよく使われる。

- **ターミナル**

    CLIでコンピュータを操作するためのソフトウェア

- **テキストエディタ**

    文字情報を編集するためのソフトウェア。
    メモを書くための簡単なものから、
    プログラムを書くための高機能なものまである。
    代表例として、Visual Studio Code、Vim、Emacsなどがある。

- **Pythonと仮想環境**

    Pythonは、機械学習などで広く使われるプログラミング言語。
    仮想環境は、プロジェクトごとにPythonの実行環境を分けて管理する仕組みである。

- **Git**

    ソースコードや文書の変更履歴を管理するためのバージョン管理システム。
    いつ、誰が、どのような変更を加えたかを記録できる。
    Gitで管理されたプロジェクトをオンラインで記録・共有
    するためのサービスとしてGitHubがある．

### CLI操作における注意

- むやみにコマンドを実行しない
- コマンドの実行結果をよく確認する
- エラーメッセージを無視しない

### 環境構築 (Windows)

この資料では，
Windows 上に Ubuntu 24.04 LTS の WSL2 環境を作成し，
その Ubuntu 上で Python と Git 操作を行うものとする。
そのため，まずはWSL2環境を構築する．

まず PowerShell を管理者権限で開き，
VS Code，Windows Terminal，WSL 本体をインストールする。

```powershell
winget install vscode
winget install "Windows Terminal"
wsl --install --no-distribution
wsl --update
wsl --version
```

- `wsl --install --no-distribution` 後は PC の再起動が必要になる場合がある。
- Ubuntu 24.04 LTS 以降は新しい WSL distro format で配布されるため，`wsl --version` で WSL 2.4.10 以上になっていることを確認する。
    - 古い場合は `wsl --update` を実行し，Windows Terminal を開き直す。

次に，インストール可能な distro 名を確認し，`Ubuntu-24.04` を明示してインストールする。

```powershell
wsl --list --online
wsl --install Ubuntu-24.04
```

Ubuntu 24.04 LTS が WSL2 として入っているか，PowerShell 側で確認する。

```powershell
wsl -l -v
```

`Ubuntu-24.04` の `VERSION` が `2` であれば WSL2 として動く。必要なら既定の distro に設定する。

```powershell
wsl --set-default Ubuntu-24.04
wsl ~ -d Ubuntu-24.04
```

インストール後，Windows Terminal から
Ubuntu 24.04 LTS を起動する。
初回起動時には Ubuntu 側の user name と password を設定する。
password 入力中は画面に文字が表示されないが，入力自体は受け付けられている。

Ubuntu 側に入ったら，Ubuntu の release 情報も確認する。

```bash
cat /etc/os-release
```

`VERSION_ID="24.04"` が確認できれば，この授業で使う Ubuntu 24.04 LTS 環境になっている。

Ubuntu を起動したら，Python，Git，`curl` を入れる。
以降の作業は，原則としてこの Ubuntu のターミナルで行う。

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git curl
```

GitHub CLI (`gh`) は，GitHub への login や repository 操作をターミナルから行うために使う。
Ubuntu 24.04 LTS では，GitHub CLI の apt repository を追加してから install する。

```bash
sudo mkdir -p -m 755 /etc/apt/keyrings
wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
gh --version
```

WSL 上のディレクトリを VS Code で開くには，
Windows 側の VS Code に WSL extension が入っている必要がある。
インストールには，Powershell で次を実行する。

```powershell
code --install-extension ms-vscode-remote.remote-wsl
```

その後，Ubuntu 側のターミナルを開き，
次を実行できることを確認する。

```bash
code --version
code .
```

初回の `code .` では，WSL 側に VS Code Server が自動 install される。
`code: command not found` になる場合は，Windows Terminal と Ubuntu を開き直し，
Windows 側の VS Code と WSL extension が入っているか確認する。

### 環境構築 (Mac)

Mac では，まず，ソフトウェアをインストールするための
package manager として，Homebrew を導入する．
Terminal を開き，
Homebrew 公式ページに掲載されている以下のコマンドを実行する。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- install の途中で password の入力や Enter の入力を求められることがある。
- install 完了時に `Next steps` として PATH 設定用の コマンドが表示された場合は，それをそのまま実行する。

次のコマンドで Homebrew が使えることを確認する。

```bash
brew --version
```

Homebrew が使えるようになったら，VS Code，Python 3.12，Git，GitHub CLI をそろえる。
Mac に Git が入っている場合でも，`git --version` で確認してから進める。

```bash
brew install --cask visual-studio-code
brew install python
brew install git gh
```

Mac で terminal から VS Code を開くには，
`code` command が PATH に入っている必要がある。
まず次で確認する。

```bash
code --version
```

`command not found: code` などと表示される場合は，
VS Code を開き，Command Palette から
`Shell Command: Install 'code' command in PATH` を実行する。
その後，Terminal を開き直して，もう一度 `code --version` を確認する。
確認できたら，作業 directory で次を実行すると VS Code が開く。

```bash
code .
```

### ターミナル操作とパス

ターミナルでの作業には，常に「現在地」となるディレクトリ（**カレントディレクトリ**）がある。
すべての操作はこの現在地を基準に行われるため，
まずどこにいるかを把握することが重要である。

| コマンド | 役割 |
|---|---|
| `pwd` | 現在地のパスを表示する |
| `ls` | 現在地にあるファイルやディレクトリの一覧を表示する |

**パス（path）** は，ファイルや
ディレクトリ（フォルダ）の場所を表す文字列である。
住所が分かれば建物にたどり着けるのと同じように，
パスが分かれば目的のファイルにたどり着ける。
パスは，ディレクトリ名を **`/`（スラッシュ）** でつないで書く。

| 記号 | 意味 |
|---|---|
| `/` | ディレクトリの区切り。先頭の `/` はルート（最上位） |
| `.` | カレントディレクトリ（現在地） |
| `..` | 親ディレクトリ（1つ上） |
| `~` | ホームディレクトリ (/home/username) |

**絶対パス**：ルートディレクトリ `/` から始まるパス

```
/home/yourname/workspace/data.csv
```

**相対パス**：現在地を基準として書くパス（`/` で始めない）

```
data.csv                    # 現在地にある data.csv
outputs/log.txt             # 現在地にある outputs ディレクトリの中にある log.txt
../README.md                # 1つ上のディレクトリにある README.md
../../shared/config.yml     # 2つ上のディレクトリにある shared/ 内 の config.yml
```

現在地を別のディレクトリへ移動するには `cd` を使う。

```bash
cd workspace      # workspace ディレクトリへ移動
cd ..             # 1つ上のディレクトリへ戻る（.. は親ディレクトリを表す）
```

新しいディレクトリを作るには `mkdir` を使う。
`-p` オプションを付けると，
途中のディレクトリもまとめて作成される。

```bash
mkdir -p outputs/session01
```

### GitHub と Git の初期設定

GitHub を使うには，GitHub account,
GitHub CLI の認証が必要になる。
この資料では GitHub との通信に SSH を使う前提にする。

```bash
gh auth login
```

`gh auth login` の途中では，次のように選ぶ。

- `What account do you want to log into?`: `GitHub.com`
- `What is your preferred protocol for Git operations?`: `SSH`
- `Authenticate Git with your GitHub credentials?`: `Yes`
- browser を使う認証を選び，表示された code を GitHub の画面に入力する

SSH を選ぶと，`gh` は既存の SSH 鍵を探す。
適切な鍵がない場合は，新しい鍵の作成と GitHub への登録を促す。
このとき GitHub に登録されるのは公開鍵であり，
秘密鍵は自分の PC または WSL 環境の中に残る。
秘密鍵を他人に見せたり，GitHub に貼り付けたりしてはいけない。

認証と SSH key の設定が終わったら，次で状態を確認する。

```bash
gh auth status
ssh -T git@github.com
```

GitHub user 名を含む認証成功メッセージが出ればよい。

Git の著者情報は次で設定する。

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global --list
```

`user.name` と `user.email` は commit に記録される。
自身の GitHub account と対応する情報にしておく。

### 作業ディレクトリの作成とリポジトリの clone

この教材では，
作業場所をホームディレクトリの下にまとめる。
例として `~/workspace` を作り，
その中にこのリポジトリや提出用リポジトリを置く。

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:ykinolab-tokai/signal-ml-training.git
```

リポジトリを clone したら，まず次を確認する。

```bash
pwd
ls
cd signal-ml-training
ls
git status
```

`pwd` は現在地，`ls` は現在地のファイル，
`git status` は Git が認識している変更状態を
確認するために使う。

### Python 環境の構築

この repo は Python 3.12 系を標準にする。
Python環境の管理には `uv` を用いる．

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

repo ルートで次を実行すると，
`pyproject.toml` と `uv.lock` に従い
Python 3.12 環境と `.venv` が用意される。

```bash
uv python install 3.12
uv sync
source .venv/bin/activate
python --version
which python
```

`python --version` が 3.12 系であり，
`which python` が `.venv` 配下を指していれば，
授業用の環境に入っている。


### VS Code からこのリポジトリを開く

次を実行して，VS Code からこのリポジトリを開くことができる。

```bash
cd ~/workspace/signal-ml-training
code .
```

WSL の場合は，VS Code の左下やウィンドウ名に
`WSL: Ubuntu-24.04` のように表示されていることを確認する。

VS Code で Python のプログラムを実行するには，
Microsoft の Python 拡張機能が必要である。
入っていない場合は，VS Code の拡張機能画面で `Python` を検索し，
Microsoft が提供しているものをインストールする。

このリポジトリでは，リポジトリ内の `.venv` を
Python の実行環境として使う。
VS Code で Command Palette を開き，
`Python: Select Interpreter` を実行して，
`.venv/bin/python` を選ぶ。

選択後，VS Code のターミナルで次を実行する。

```bash
which python
python --version
```

`which python` がこのリポジトリ内の `.venv/bin/python` を指していれば，
VS Code から実行する Python も
先ほど作成した仮想環境を使っている。

### 最初の動作確認

環境構築後は，Pythonを実行できることまで確認する。
次の Python スクリプトを
`signal-ml-training/exercises/exc00_01.py` として
作成し，
実行の結果 `singal-ml-training/outputs/setup_check/sin.png` が作られれば正しく環境構築できている。

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

## 演習
今回は環境構築を主とするため，演習はない．

## 詰まったときに見る資料
- [`../README.md`](../README.md)
- [`../textbook/markdown/ch01-basic-operations.md`](../textbook/markdown/ch01-basic-operations.md)
- [uv documentation](https://docs.astral.sh/uv/)
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
