# 環境構築とワークフロー

本章は，環境構築の手順とその意味を，初めて取り組む学生が確認できるように説明する。
授業用のhandoutと重なる操作も含め，OS別の導入からPythonスクリプトの実行までを本章内で扱う。
Windowsでは「環境構築 (Windows)」，Apple SiliconのMacでは「環境構築 (Mac)」を行い，その後は共通手順へ進む。
設定済みの項目は確認コマンドで状態を確かめ，成功していれば再設定しなくてよい。

## キーワード
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

## CLI操作における注意

- むやみにコマンドを実行しない
- コマンドの実行結果をよく確認する
- エラーメッセージを無視しない

## 環境構築 (Windows)

この資料では，
Windows 上に Ubuntu 24.04 LTS の WSL2 環境を作成し，
その Ubuntu 上で Python と Git 操作を行うものとする。
そのため，まずはWSL2環境を構築する．

まず PowerShell を管理者権限で開き，
VS Code，Windows Terminal，WSL 本体をインストールする。

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

- `wsl --install --no-distribution` 後は PC の再起動が必要になる場合がある。
- Ubuntu 24.04 LTS 以降は新しい WSL distro format で配布されるため，`wsl --version` で WSL 2.4.10 以上になっていることを確認する。
    - 古い場合は `wsl --update` を実行し，Windows Terminal を開き直す。

次に，インストール可能な distro 名を確認し，`Ubuntu-24.04` を明示してインストールする。

```powershell
wsl --list --online
wsl --install -d Ubuntu-24.04
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

Ubuntu を起動したら，Git，`curl`，SSHクライアントを入れる。Python 3.12は後述の `uv` で管理する。
以降の作業は，原則としてこの Ubuntu のターミナルで行う。

```bash
sudo apt update
sudo apt install git curl openssh-client
```

GitHub CLI (`gh`) は，GitHub への login や repository 操作をターミナルから行うために使う。
Ubuntu 24.04 LTS では，GitHub CLI の公式apt配布元を登録してからインストールする。
公開鍵ファイルはダウンロードしたパッケージの署名確認に使い，`signed-by` でそのファイルを指定する。
以下の各コマンドの成功を確認してから次の行へ進む。

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

WSL 上のディレクトリを VS Code で開くには，
Windows 側の VS Code に WSL extension が入っている必要がある。
インストールには，PowerShell で次を実行する。

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

## 環境構築 (Mac)

この手順は Apple Silicon（M シリーズ）の Mac を対象にする。
この repo で固定している PyTorch 2.7.1 には Intel Mac 用の wheel がないため，
Intel Mac ではこの手順のまま環境をそろえることはできない。
Intel Mac を使う場合は，授業用に別の対応環境を用意する必要がある。

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

Homebrew が使えるようになったら，VS Code，Git，GitHub CLI をそろえる。
Python 3.12 は，後述する `uv` でインストールして管理する。
Mac に Git が入っている場合でも，`git --version` で確認してから進める。

```bash
brew install --cask visual-studio-code
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

## ターミナル操作とパス

ターミナルでの作業には，常に「現在地」となるディレクトリ（**カレントディレクトリ**）がある。
相対パスで指定する操作はこの現在地を基準に行われるため，
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
| `~` | ホームディレクトリ（Ubuntuでは `/home/username`，Macでは `/Users/username`） |

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

## GitHub と Git の初期設定

GitHub を使うには，GitHub account,
GitHub CLI の認証が必要になる。
この資料では GitHub との通信に SSH を使う前提にする。

```bash
gh auth login
```

`gh auth login` の質問に対して，接続先，Git通信の方式，使用するSSH鍵，認証方法を次のように選ぶ。
表示される質問や順序は，`gh` のバージョンや既存の認証・SSH 鍵の設定によって変わる。

- `What account do you want to log into?`: `GitHub.com`
- `What is your preferred protocol for Git operations?`: `SSH`
- SSH 公開鍵の登録を求められたら，使用する鍵を選ぶ。鍵がない場合は，案内に従って作成する。
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

初回のSSH接続では，接続先のホスト鍵を信頼するか確認されることがある。
表示されたfingerprint（鍵の識別値）を [GitHub公式の公開鍵一覧](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints) と照合し，一致した場合に `yes` を入力する。
自分のGitHubユーザー名を含む認証成功メッセージが出ればよい。
GitHubはSSHでのシェル操作を提供しないため，`ssh -T git@github.com` は認証成功時にも終了コード1を返す。
`Permission denied (publickey)` は成功ではない。使用する公開鍵が自分のGitHubアカウントに登録されているかを確認する。

Git の著者情報は次で設定する。

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global --list
```

`user.name` と `user.email` は commit に記録される。
自身の GitHub account と対応する情報にしておく。

## 作業ディレクトリの作成とリポジトリの clone

この教材では，
作業場所をホームディレクトリの下にまとめる。
例として `~/workspace` を作り，
その中にこのリポジトリや提出用リポジトリを置く。

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:ykinolab-tokai/signal-ml-training.git
```

すでに同じ場所へ取得済みの場合は `git clone` を省略する。リポジトリを取得したら，まず次を確認する。

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

## Python 環境の構築

この repo は Python 3.12 系を標準にする。
Python環境の管理には `uv` を用いる．

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール末尾に表示される PATH の案内に従い，
ターミナルを開き直すか，案内された `source` コマンドを実行する。
次のコマンドでバージョンが表示されることを確認してから先へ進む。

```bash
uv --version
```

repo ルートで次を実行すると，
`pyproject.toml` と `uv.lock` に従い
Python 3.12 環境と `.venv` が用意される。
`pyproject.toml` は必要なパッケージの条件，`uv.lock` は解決済みのバージョンを記録する。
`--locked` はlockfileを書き換えず，不整合があれば停止する指定である。
取得前の空のディレクトリで実行せず，教材repoのルートへ移動してから実行する。

```bash
uv python install 3.12
uv sync --locked
source .venv/bin/activate
python --version
which python
```

`python --version` が 3.12 系であり，
`which python` が `.venv` 配下を指していれば，
授業用の環境に入っている。


## VS Code からこのリポジトリを開く

この repo を VS Code で開くには，
Ubuntu（WSL）または Mac のターミナルで次を実行する。

```bash
cd ~/workspace/signal-ml-training
code .
```

VS Code の Explorer に `pyproject.toml` と `uv.lock` が見えていることを確認する。

WSL の場合は，VS Code の左下やウィンドウ名に
`WSL: Ubuntu-24.04` のように表示されていることを確認する。
このウィンドウのターミナルも Ubuntu 側で動作する。

ターミナルからプログラムを実行する場合は，
前節で準備した `uv` を使って repo の環境で実行できる。
VS Code のターミナルでも，通常の Ubuntu や Mac のターミナルでも同じ操作でよい。
次の例は，「最初の動作確認」のサンプルファイルを保存した後に実行する。

```bash
cd ~/workspace/signal-ml-training
uv run --locked python exercises/exc01_01.py
```

`uv run --locked` はlockfileを更新せずにrepoの環境を確認・同期してから実行するため，
事前に `source .venv/bin/activate` を実行する必要はない。
一方，`python exercises/exc01_01.py` と直接実行する場合は，
そのターミナルで `.venv` が有効になっていることを先に確認する。
新しいターミナルで `python` を直接使うときは，その都度 `.venv` を有効にする。
詳しくは [uv のプロジェクト操作](https://docs.astral.sh/uv/guides/projects/) を参照する。

## VS Code の実行・デバッグ機能で `.venv` を使う（任意）

以下は，VS Code の実行ボタンやデバッグ機能から，
repo の `.venv` にある Python を使いたい場合の任意の設定である。
repo を VS Code で開いて編集するだけの場合や，ターミナルで `uv run` を使う場合には，
この設定は必要ない。「最初の動作確認」へ進んでよい。
授業用の Python 環境は，どの実行方法でも「Python 環境の構築」で説明した `uv` で管理する。

この設定を行う場合は，まず repo のルートで次を実行し，
`pyproject.toml` と `uv.lock` に従って `.venv` を準備する。

```bash
cd ~/workspace/signal-ml-training
uv sync --locked
```

VS Code の Python 実行・デバッグ機能を使うには，
Microsoft の Python 拡張機能が必要である。
入っていない場合は，VS Code の拡張機能画面で `Python` を検索し，
Microsoft が提供しているものをインストールする。
WSL の場合は，拡張機能画面で Ubuntu 側でも有効になっていることを確認し，
`Install in WSL: Ubuntu-24.04` と表示される場合はそのボタンでインストールする。

次に，この repo で使う Python を明示的に選ぶ。

1. Windows では `Ctrl+Shift+P`，Mac では `Cmd+Shift+P` で Command Palette を開く。
2. `Python: Select Interpreter` を実行する。
3. この repo 内の `.venv/bin/python` に対応する Python 3.12 を選ぶ。

選択するパスの例は次のとおりである。

- WSL：`/home/yourname/workspace/signal-ml-training/.venv/bin/python`
- Mac：`/Users/yourname/workspace/signal-ml-training/.venv/bin/python`

`yourname` は自分の user name に読み替える。
候補に `.venv` が見つからない場合は，repo のルートで `uv sync --locked` が成功したことを確認し，
Command Palette の `Developer: Reload Window` でウィンドウを再読み込みしてから選び直す。

選択後，既存のターミナルをゴミ箱ボタンで終了し，
メニューの `Terminal` → `New Terminal` から新しいターミナルを開く。
次を実行して，そのターミナルの現在地と Python を確認する。

```bash
cd ~/workspace/signal-ml-training
pwd
which python
python --version
```

`pwd` が repo のルート，`which python` がこの repo 内の `.venv/bin/python`，
`python --version` が 3.12 系を示すことを確認する。
新しいターミナルで仮想環境が自動的に有効にならない場合は，
repo のルートで `source .venv/bin/activate` を実行し，もう一度確認する。

VS Code の実行ボタンで使われる Python も確認する。

次の 2 行を repo 直下の `interpreter_check.py` として保存する。

```python
import sys
print(sys.executable)
```

このファイルをエディタで開いた状態で，右上の `Run Python File in Terminal` を押す。
同じ操作は，エディタ内を右クリックし，`Run` → `Python File in Terminal` からも選べる。
表示されるパスがこのリポジトリ内の `.venv/bin/python` であることを確認する。
異なる場合は `Python: Select Interpreter` でこの repo の `.venv` を選び直す。

`Run Python File in Terminal` は，VS Code で選択した Python を使ってファイルを実行する。
デバッグ機能も，実行環境を個別に指定していなければ，この選択を使う。
教材を更新して依存パッケージが変わった場合は，実行前に repo のルートで `uv sync --locked` を行う。
詳しくは [VS Code の Python 実行手順](https://code.visualstudio.com/docs/python/run) を参照する。

## 最初の動作確認

環境構築後は，Pythonを実行できることまで確認する。
まず repo のルートに移動し，スクリプトの保存先を作る。

```bash
cd ~/workspace/signal-ml-training
mkdir -p exercises
```

次の Python スクリプトを
repo 直下の `exercises/exc01_01.py` として保存する。

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

repo のルートで仮想環境を有効にして，保存したスクリプトを実行する。

```bash
cd ~/workspace/signal-ml-training
source .venv/bin/activate
python exercises/exc01_01.py
```

`setup check completed` と表示され，repo 直下に
`outputs/setup_check/sin.png` が作られることを確認する。
この画像を開いて正弦波が描かれていれば，動作確認は完了である。

`git status` も実行し，Gitが認識している変更の有無を確認する。
無視対象のファイルは通常の `git status` に表示されない。出力画像の有無は実際にファイルを開いて確認する。

## 教材repoと作業repoの使い分け

教材repoは教材・見本・共通の依存関係を読む場所で，第1回の動作確認もここで行う。
第2回以降の解答は自分の作業repo（提出repo／submission repo）へ保存する。
[第1回の作成手順](../../handouts/01-environment-and-workflow.md#自分用リポジトリの作成)に従い，
作業用テンプレートから `ykinolab-tokai` 所有の非公開 `signal-ml-work-<GitHubユーザー名>` を作成する。
作成したrepoのREADMEに従って `~/workspace/signal-ml-work` にcloneする。
取得済みならそのフォルダを開き，二重にcloneしない。
教材repoは `~/workspace/signal-ml-training`，作業repoは `~/workspace/signal-ml-work` に並ぶ。

第2回以降と補足の課題の相対パスは，特記がなければ作業repoのルートを基準とする。

- コードの標準は `exercise/excXX_YY.py`（回番号・問題番号は2桁）。各回が `scripts/` 以下などを指定した場合はその指定を優先する。フォルダなしのファイル名だけを指定した場合は作業repoのルートに置く。
- 図・音声・数値結果の標準は `outputs/` 以下。説明はコードコメントまたは既存の結果ファイルに残し，別レポートは必要な課題が明示した場合だけ作る。同じ内容を転記しない。
- `data/cat.png` や `data/piano.wav` を使う場合は，教材repoの同名ファイルを作業repoの `data/` へコピーする。

環境の初回準備は本章の前半に従う。教材更新後の依存関係の同期は教材repoで行う。

```bash
cd ~/workspace/signal-ml-training
uv sync --locked
```

演習時は作業repoへ移動し，教材側の仮想環境を有効にする。

```bash
cd ~/workspace/signal-ml-work
source ../signal-ml-training/.venv/bin/activate
pwd
python --version
which python
```

`pwd` が作業repo，`which python` が教材側の `.venv/bin/python` を指すことを確認する。
例えば第2回第1問は `python exercise/exc02_01.py` で実行する。
配置を変えた場合は実際のパスへ読み替える。`.venv` は作業repoへコピーせず，毎回作り直さない。
保存前に `Path(...).mkdir(parents=True, exist_ok=True)` などで出力ディレクトリを作る。
保存規則とPR練習後の通常作業への戻り方は [共通手順](../../handouts/README.md#作業場所と保存先)を参照する。

## うまく進まないとき

- WSLのバージョンが2.4.10未満：PowerShellで `wsl --update` を実行し，Windows Terminalを開き直して `wsl --version` を再確認する。
- `wsl -l -v` でUbuntuの `VERSION` が1：PowerShellで `wsl --set-version Ubuntu-24.04 2` を実行し，`wsl -l -v` で2になったことを確認する。
- `code` や `uv` が見つからない：インストール完了時のPATH設定を確認してターミナルを開き直し，`code --version` または `uv --version` を再実行する。
- `uv sync --locked` がプロジェクトを見つけられない：`pwd` と `ls` で教材repoの `pyproject.toml` と `uv.lock` がある場所か確認する。
- lockfileと設定の不整合で停止する：授業で指定された教材の版か確認し，担当者にエラーを伝える。自分だけ依存関係を変更しない。
- Pythonのパスが `.venv` ではない：教材repoで `source .venv/bin/activate` を実行し，`which python` と `python --version` を再確認する。
- 画像が見つからない：実行時の現在地を `pwd` で確認する。サンプルはその現在地を基準に `outputs/setup_check/sin.png` を作る。

## 公式資料

- [Microsoft：WSLのインストール](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Ubuntu：WSL2への導入と必要なWSLバージョン](https://ubuntu.com/wsl/docs/stable/howto/install-ubuntu-wsl2/)
- [GitHub CLI：Linuxへの導入](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)
- [GitHub CLI：認証](https://cli.github.com/manual/gh_auth_login)
- [GitHub：SSH接続の確認](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)
- [Homebrew：導入](https://brew.sh/)
- [uv：導入](https://docs.astral.sh/uv/getting-started/installation/)
- [uv：lockfileと環境の同期](https://docs.astral.sh/uv/concepts/projects/sync/)
- [VS Code：WSL](https://code.visualstudio.com/docs/remote/wsl)
- [VS Code：Python環境](https://code.visualstudio.com/docs/python/environments)
