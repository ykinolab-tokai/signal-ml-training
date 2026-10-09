# 第01回 環境構築とワークフロー

## この回の目標

ターミナル・VS Code・Git を使い，Python の実行環境を整えて作業できる。

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
    ファイルとその変更履歴を管理する単位をリポジトリと呼ぶ。
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

GitHub CLI (`gh`) は，GitHub への login やリポジトリの操作をターミナルから行うために使う。
Ubuntu 24.04 LTS では，GitHub CLI のパッケージ配布元（aptリポジトリ）を追加してから install する。

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

この手順は Apple Silicon（M シリーズ）の Mac を対象にする。
`signal-ml-training` リポジトリでバージョンを固定している PyTorch 2.7.1 には Intel Mac 用の wheel がないため，
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
確認できたら，作業ディレクトリで次を実行すると VS Code が開く。

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

GitHub を使うには，個人用アカウントの作成と，
GitHub CLI の認証が必要になる。

#### GitHub アカウントの作成

すでに個人用の GitHub アカウントを持っている場合は，そのアカウントを使う。
ブラウザでサインインできることと，メールアドレスの確認が済んでいることを確認し，
次の「GitHub CLI の認証」へ進む。

アカウントを持っていない場合は，次の手順で無料の個人用アカウントを作成する。

1. ブラウザで [GitHub の登録ページ](https://github.com/signup)を開く。
2. メールアドレス，パスワード，ユーザー名，国／地域を入力する。
   メールアドレスには，自分で確認メールを受け取れるものを使う。
   ユーザー名には半角英数字とハイフンを使えるが，ハイフンの連続使用や先頭・末尾への使用はできない。
   パスワードは画面に表示される条件を満たすものを設定する。
3. 画面の案内に従って確認操作を行い，`Create account` を選ぶ。
4. 登録したメールアドレスに届く確認メールの案内に従い，メールアドレスの確認を完了する。
   確認が済んでいないと，リポジトリの作成などの操作ができない。
5. GitHub にサインインできることを確認し，自分のユーザー名を把握しておく。

登録画面では，Google または Apple アカウントを使う方法も選べる。
上記はメールアドレスで登録する場合の手順である。
画面の表示や手順が異なる場合は，[GitHub 公式のアカウント作成ガイド](https://docs.github.com/ja/account-and-profile/how-tos/account-management/creating-an-account-on-github)を確認する
（登録画面と公式ガイドの確認日：2026年10月2日）。

#### GitHub CLI の認証

アカウントの準備ができたら，ターミナルで次のコマンドを実行する。
この資料では GitHub との通信に SSH を使う前提にする。

```bash
gh auth login
```

`gh auth login` の途中では，次の方針で選ぶ。
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

GitHub user 名を含む認証成功メッセージが出ればよい。

Git の著者情報は次で設定する。

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global --list
```

`user.name` と `user.email` は commit に記録される。
自身の GitHub account と対応する情報にしておく。

### 自分用作業リポジトリの作成

環境構築と各回の解答の保存に使う，自分用の非公開作業リポジトリをテンプレートから作成する。

1. GitHub にサインインし，[作業用テンプレート](https://github.com/ykinolab-tokai/signal-ml-work-template)を開く。
2. **Use this template** → **Create a new repository** を選ぶ。
3. **Owner** を `ykinolab-tokai`，**Repository name** を `signal-ml-work-<GitHubユーザー名>`（例：ユーザー名が `taro-lab` なら `signal-ml-work-taro-lab`），公開範囲を **Private** にして作成する。

GitHub上の名前は `signal-ml-work-<GitHubユーザー名>`，PC上のフォルダ名は `signal-ml-work` とする。
すでに作成・取得済みなら同じリポジトリを使う。

テンプレートを開けない，Owner に `ykinolab-tokai` が表示されない，または作成できない場合は，担当教員に連絡する。

### 作業ディレクトリの作成とリポジトリの clone

この教材では，ホームディレクトリの下に作業場所 `~/workspace` を作り，
その中に先ほど作成した自分用の作業リポジトリを以下のコマンドで clone する。

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone REPOSITORY_URL signal-ml-work
```

ここで `REPOSITORY_URL` は，
作業リポジトリの **Code** → **SSH** ボタンからコピーしたURLに置き換える。

このコマンドで `~/workspace/signal-ml-work` ディレクトリが作られ，
作業用テンプレートのファイルとGitの管理情報が保存される。
以降，この教材ではこの場所を `signal-ml-work` ディレクトリと呼ぶ。
リポジトリを clone したら，次のコマンドでカレントディレクトリとファイルを確認し，このディレクトリへ移動する。

```bash
pwd
ls
cd signal-ml-work
ls
git status
```

`pwd` は現在地，`ls` は現在地のファイル，
`git status` は Git が認識している変更状態を
確認するために使う。

### Python 環境の構築

この教材では Python 3.12 系を標準にする。
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

作業用テンプレートには，環境定義ファイル `pyproject.toml` と `uv.lock` が含まれている。
作業用リポジトリのルートに両方あることを確認する。

```bash
cd ~/workspace/signal-ml-work
ls pyproject.toml uv.lock
```

続いて，`signal-ml-work` ディレクトリで次を実行する。
`pyproject.toml` と `uv.lock` に従い，作業用リポジトリ内に Python 3.12 環境 `.venv` が用意される。

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


### VS Code で作業用リポジトリを開く

`signal-ml-work` ディレクトリを VS Code で開くには，
Ubuntu（WSL）または Mac のターミナルで次を実行する。

```bash
cd ~/workspace/signal-ml-work
code .
```

VS Code の Explorer に `pyproject.toml` と `uv.lock` が見えていることを確認する。

WSL の場合は，VS Code の左下やウィンドウ名に
`WSL: Ubuntu-24.04` のように表示されていることを確認する。
このウィンドウのターミナルも Ubuntu 側で動作する。

### VS Code の実行・デバッグ機能で `.venv` を使う（任意）

以下は，VS Code の実行ボタンやデバッグ機能から，
`signal-ml-work` ディレクトリ内の仮想環境 `.venv` にある Python を使いたい場合の任意の設定である。
`signal-ml-work` ディレクトリを VS Code で開いてファイルを編集するだけの場合や，ターミナルで `uv run` を使う場合には，
この設定は必要ない。「最初の動作確認」へ進んでよい。

VS Code の Python 実行・デバッグ機能を使うには，
Microsoft の Python 拡張機能が必要である。
入っていない場合は，VS Code の拡張機能画面で `Python` を検索し，
Microsoft が提供しているものをインストールする。
WSL の場合は，拡張機能画面で Ubuntu 側でも有効になっていることを確認し，
`Install in WSL: Ubuntu-24.04` と表示される場合はそのボタンでインストールする。

作業用リポジトリの `.vscode/settings.json` に `python.defaultInterpreterPath` がある場合は，
その値を `${workspaceFolder}/.venv/bin/python` に変更する。
次に，コードを実行する Python を明示的に選ぶ。

1. Windows では `Ctrl+Shift+P`，Mac では `Cmd+Shift+P` で Command Palette を開く。
2. `Python: Select Interpreter` を実行する。
3. `signal-ml-work` ディレクトリ内の `.venv/bin/python` に対応する Python 3.12 を選ぶ。

選択するパスの例は，WSL では
`/home/yourname/workspace/signal-ml-work/.venv/bin/python`，
Mac では `/Users/yourname/workspace/signal-ml-work/.venv/bin/python` である。
`yourname` は自分の user name に読み替える。
候補に `.venv` が見つからない場合は，`signal-ml-work` ディレクトリで `uv sync --locked` が成功したことを確認し，
Command Palette の `Developer: Reload Window` でウィンドウを再読み込みしてから選び直す。

選択後，既存のターミナルをゴミ箱ボタンで終了し，
メニューの `Terminal` → `New Terminal` から新しいターミナルを開く。
次を実行して，そのターミナルの現在地と Python を確認する。

```bash
cd ~/workspace/signal-ml-work
pwd
which python
python --version
```

`pwd` が `signal-ml-work` ディレクトリのパス，`which python` がその中の `.venv/bin/python` のパス，
`python --version` が 3.12 系を示すことを確認する。
新しいターミナルで仮想環境が自動的に有効にならない場合は，
`signal-ml-work` ディレクトリで `source .venv/bin/activate` を実行し，もう一度確認する。

VS Code の実行ボタンで使われる Python も確認するため，
次の 2 行を `signal-ml-work` ディレクトリ直下の `interpreter_check.py` として保存する。

```python
import sys
print(sys.executable)
```

このファイルをエディタで開いた状態で，右上の `Run Python File in Terminal` を押す。
同じ操作は，エディタ内を右クリックし，`Run` → `Python File in Terminal` からも選べる。
表示されるパスが `signal-ml-work` ディレクトリ内の `.venv/bin/python` であることを確認する。
異なる場合は `Python: Select Interpreter` で `signal-ml-work` ディレクトリ内の `.venv` にある Python を選び直す。

`Run Python File in Terminal` は，VS Code で選択した Python を使ってファイルを実行する。
デバッグ機能も，実行環境を個別に指定していなければ，この選択を使う。
作業用リポジトリの環境定義ファイルを更新した場合は，実行前に `signal-ml-work` ディレクトリで `uv sync --locked` を行う。
詳しくは [VS Code の Python 実行手順](https://code.visualstudio.com/docs/python/run) を参照する。

### 最初の動作確認

環境構築後は，Pythonを実行できることまで確認する。
まず `signal-ml-work` ディレクトリに移動し，スクリプトの保存先を作る。

```bash
cd ~/workspace/signal-ml-work
mkdir -p exercise
```

次の Python スクリプトを
`signal-ml-work` ディレクトリ内の `exercise/exc01_01.py` として保存する。

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

`signal-ml-work` ディレクトリで仮想環境を有効にして，保存したスクリプトを実行する。

```bash
cd ~/workspace/signal-ml-work
source .venv/bin/activate
python exercise/exc01_01.py
```

`setup check completed` と表示され，`signal-ml-work` ディレクトリ内に
`outputs/setup_check/sin.png` が作られることを確認する。
この画像を開いて正弦波が描かれていれば，動作確認は完了である。

### 変更の記録と GitHub への push

動作確認で作成したコードと画像を，Git で記録して自分の作業用リポジトリへ送る。
**commit** は変更を手元の履歴に記録する操作，**push** はその履歴を GitHub へ送る操作である。

まず，作業用リポジトリで現在のブランチと送信先を確認する。

```bash
cd ~/workspace/signal-ml-work
git status
git remote -v
```

ブランチが `main`，`origin` のURLが自分の `signal-ml-work-<GitHubユーザー名>` を指していることを確認する。
続いて，記録するファイルを `git add` で選ぶ。

```bash
git add exercise/exc01_01.py outputs/setup_check/sin.png
```

任意の VS Code 設定で `interpreter_check.py` を作成した場合は，
`git add interpreter_check.py .vscode/settings.json` も実行する。
`.venv/` は Git の管理対象に含めない。

次のコマンドで，commitするファイルと変更内容を確認する。

```bash
git status
git diff --cached
```

`git diff --cached` は，`git add` で選んだ変更を表示する。画像については，画像ファイルが追加されたことを確認する。
差分の閲覧画面に切り替わった場合は，`q` を押すとコマンド入力へ戻れる。
内容を確認したら，変更内容を表すメッセージを付けてcommitし，pushする。

```bash
git commit -m "第1回: 環境構築と動作確認"
git push origin main
git status
```

pushが成功したら，ブラウザで自分の作業用リポジトリの `main` を開く。
commitメッセージと `exercise/exc01_01.py`，`outputs/setup_check/sin.png` が反映されていることを確認する。
pushに失敗した場合は，エラーメッセージを確認し，解決できなければ担当教員に相談する。

## 演習
今回は環境構築を主とするため，追加の演習はありません。上の動作確認とGitHubへのpushを完了してください。

第1回の `exercise/exc01_01.py` と `outputs/setup_check/`，第2回以降の解答は，
すべて自分の作業用リポジトリ `signal-ml-work` 内に保存してください。
保存先と実行環境の使い方は [共通の作業場所](README.md#作業場所と保存先) を参照してください。

## 確認ポイント
- Ubuntu 24.04 LTS（Windowsの場合）とPython 3.12の環境を確認した。
- `signal-ml-work` ディレクトリで `uv --version`，`python --version`，`which python` を確認し，`.venv` のPythonを使っている。
- `exercise/exc01_01.py` を `signal-ml-work` ディレクトリから実行し，完了メッセージと `outputs/setup_check/sin.png` の正弦波を確認した。
- `git status` で `signal-ml-work` リポジトリの変更状態を説明できる。
- 動作確認のコードと画像をcommit・pushし，自分の作業用リポジトリの `main` に反映されたことをGitHub上で確認した。
- テンプレートから自分用の非公開リポジトリを作成し，この資料に従って作業用リポジトリ内で環境構築と動作確認を完了した。
- VS Codeの実行・デバッグ設定を選んだ場合だけ，`sys.executable` も確認した。設定を省略しても本回は完了できる。

## 詰まったときに見る資料
- [`../README.md`](../README.md)
- [`../textbook/markdown/ch01-basic-operations.md`](../textbook/markdown/ch01-basic-operations.md)
- [uv documentation](https://docs.astral.sh/uv/)
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
