# 第01回 環境構築とワークフロー

## この回の目標

ターミナル・VS Code・Git を使い，Python の実行環境を整えて作業できる。

## 解説

この回は最初の回であり，第2回以降のすべての回で使う作業環境を準備する。
Windows では WSL2 上の Ubuntu 24.04 LTS のターミナルを，Mac（Apple Silicon）では Mac のターミナルを使い，次の順に作業する。

1. OS ごとに，VS Code，Git，GitHub CLI などのソフトウェアを準備する（第2節）。
2. GitHub のアカウントと GitHub CLI の認証を準備し，Git の著者情報を設定する（第4節）。
3. 自分用の作業リポジトリを GitHub 上に作り，PC に clone する（第5節）。
4. 作業リポジトリの中に Python 3.12 の環境を作る（第6節）。
5. VS Code で作業リポジトリを開き，Python のスクリプトを実行して図を保存する（第7節と第8節）。
6. 変更を Git で記録し，GitHub へ push する（第9節）。

第1節と第3節では，これらの作業に必要な用語とターミナルの操作を説明する。
各コマンドを実行する場所（Windows の PowerShell，Ubuntu のターミナル，Mac のターミナル）は，コマンドの直前に示す。
コマンドは 1 行ずつ実行し，結果を確かめてから次へ進む。

この回の Python のスクリプトは，第6節で Python の環境を作った後に実行する。
スクリプトの先頭には，次の処理を書く。

```python
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figure_dir = Path("outputs/exc01")
figure_dir.mkdir(parents=True, exist_ok=True)
```

`sys` は，実行中の Python 自身の情報を得るための標準ライブラリである。
`numpy` は数値計算のためのライブラリ，`matplotlib.pyplot` は図を描くためのモジュールであり，どちらも第3回で詳しく扱う。
`Path` は，ファイルやディレクトリのパスを扱うためのクラスである。
`figure_dir.mkdir(parents=True, exist_ok=True)` は，保存先のディレクトリ `outputs/exc01` を，途中の `outputs` も含めて作り，すでにある場合は何もしない。
以下のコード例は，このコードに続けて書いて実行する。

### 1. コンピュータとコマンドライン操作の基本

#### 基本的な用語

**オペレーティングシステム (operating system, OS)** は，コンピュータ全体を管理するための基本ソフトウェアである。
代表例として，Windows，macOS，Linux がある。
この授業で使う Ubuntu は，Linux を基にした OS の 1 つである。
**ファイルシステム (file system)** は，ファイルやフォルダを保存し，整理し，読み書きするための仕組みである。
この資料では，フォルダのことを **ディレクトリ (directory)** と呼ぶ。

**コマンドラインインターフェース (command line interface, CLI)** は，文字でコマンドを入力してコンピュータを操作する仕組みである。
マウスで画面を操作する方法と比べて，同じ操作を正確に繰り返したり，手順を記録して共有したりしやすいため，開発や研究でよく使われる。
**ターミナル (terminal)** は，CLI でコンピュータを操作するためのソフトウェアである。
ターミナルの中では，入力したコマンドを解釈して実行する **シェル (shell)** が動いている。
標準のシェルは Ubuntu では bash，Mac では zsh であり，この資料のコマンドはどちらでも同じ書き方で実行できる。

**テキストエディタ (text editor)** は，文字情報を編集するためのソフトウェアである。
メモを書くためのものから，プログラムの作成を助ける機能を備えたものまであり，代表例として Visual Studio Code，Vim，Emacs がある。
この授業では，**Visual Studio Code（VS Code）** を使う。

**Python** は，機械学習などで広く使われるプログラミング言語である。
**仮想環境 (virtual environment)** は，Python の実行ファイルと追加のパッケージ（ライブラリ）を，プロジェクトごとに分けて管理する仕組みである。
この授業では，作業リポジトリの中の `.venv` ディレクトリに仮想環境を作る。

**Git** は，ソースコードや文書の変更履歴を管理するための **バージョン管理システム (version control system)** である。
Git を使うと，いつ，誰が，どのような変更を加えたかを記録できる。
ファイルとその変更履歴を管理する単位を **リポジトリ (repository)** という。
**GitHub** は，Git で管理したリポジトリをインターネット上に保存し，共有するためのサービスである。

#### コマンドを実行するときの注意

コマンドは，コンピュータへの直接の指示である。
ファイルの削除やシステムの設定の変更も 1 行で実行できるため，意味の分からないコマンドをむやみに実行しない。
特に，管理者の権限で実行する `sudo` 付きのコマンドや，インターネットから取得したスクリプトを実行するコマンド（`curl ... | sh` の形のもの）は，システム全体に影響する。
これらは，この資料や公式の資料に書かれたものだけを実行し，実行前に入力内容を確かめる。

コマンドを実行したら，その結果をよく確認する。
多くのコマンドは，成功したときには何も表示しないか短い表示だけを出し，失敗したときにはエラーメッセージを表示する。
エラーメッセージ（`error`，`not found`，`Permission denied` などを含む表示）を無視して次のコマンドへ進むと，その後の手順も失敗し，原因が分かりにくくなる。
エラーが表示されたらそこで止まってメッセージを読み，解決できない場合は，実行したコマンドと表示されたメッセージをそのまま担当教員に伝えて相談する。

### 2. OS ごとの準備

この節では，Windows と Mac のそれぞれで，授業に必要なソフトウェアを準備する。
自分の OS に対応する小見出しの手順だけを行い，終わったら第3節へ進む。
すでに準備済みの項目は，確認のコマンドで状態を確かめ，成功していれば設定し直さなくてよい。

#### 環境構築 (Windows)

この資料では，Windows 上に Ubuntu 24.04 LTS の **WSL2 (Windows Subsystem for Linux 2)** 環境を作り，その Ubuntu 上で Python と Git の操作を行う。
WSL2 は，Windows の中で Linux を動かすための仕組みである。
そのため，まずは WSL2 の環境を構築する。

スタートメニューで PowerShell を検索して「管理者として実行」を選び，管理者権限の PowerShell を開く。
次のコマンドで，VS Code，Windows Terminal，WSL 本体をインストールする。

```powershell
winget install --id Microsoft.VisualStudioCode --exact
winget install --id Microsoft.WindowsTerminal --exact
wsl --install --no-distribution
```

`winget` は Windows のパッケージ管理ツールであり，`--id` と `--exact` は，インストールするパッケージを ID の完全一致で指定するオプションである。
`wsl --install --no-distribution` は，Linux の配布形態（distro）を入れずに，WSL 本体だけをインストールする。
この後，PC の再起動が必要になる場合がある。
再起動を求められたら再起動し，管理者権限の PowerShell を開き直して，WSL を更新してからバージョンを確認する。

```powershell
wsl --update
wsl --version
```

Ubuntu 24.04 LTS 以降は新しい WSL distro format で配布されるため，`wsl --version` で WSL のバージョンが 2.4.10 以上になっていることを確認する。
古い場合は，もう一度 `wsl --update` を実行し，Windows Terminal を開き直す。

次に，インストールできる distro の名前を確認し，`Ubuntu-24.04` を明示してインストールする。

```powershell
wsl --list --online
wsl --install -d Ubuntu-24.04
```

Ubuntu 24.04 LTS が WSL2 として入っているかを，PowerShell 側で確認する。

```powershell
wsl -l -v
```

`Ubuntu-24.04` の `VERSION` が `2` であれば，WSL2 として動いている。
`VERSION` が `1` の場合は，`wsl --set-version Ubuntu-24.04 2` を実行し，もう一度 `wsl -l -v` で `2` になったことを確認する。
必要なら，次のコマンドで `Ubuntu-24.04` を既定の distro に設定し，Ubuntu のホームディレクトリで起動する。

```powershell
wsl --set-default Ubuntu-24.04
wsl ~ -d Ubuntu-24.04
```

インストール後は，Windows Terminal から Ubuntu 24.04 LTS を起動する。
初回の起動時には，Ubuntu 側の user name と password を設定する。
password の入力中は画面に文字が表示されないが，入力自体は受け付けられている。
この password は，後で `sudo` を使うときに入力する。

Ubuntu 側に入ったら，Ubuntu の release 情報を確認する。
`cat` は，ファイルの内容を表示するコマンドである。

```bash
cat /etc/os-release
```

`VERSION_ID="24.04"` が表示されれば，この授業で使う Ubuntu 24.04 LTS の環境になっている。

続いて，Ubuntu のターミナルで Python，Git，`curl` を入れる。
以降の作業は，原則としてこの Ubuntu のターミナルで行う。

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git curl
```

`apt` は Ubuntu のパッケージ管理ツールであり，`apt update` で入手できるパッケージの一覧を更新してから，`apt install` で指定したパッケージをインストールする。
`sudo` を付けたコマンドでは，Ubuntu の password を求められることがある。
途中で `Do you want to continue? [Y/n]` と表示された場合は，`Y` を入力して Enter キーを押す。

**GitHub CLI**（`gh` コマンド）は，GitHub への login やリポジトリの操作をターミナルから行うためのツールである。
Ubuntu 24.04 LTS では，GitHub CLI のパッケージの配布元（apt リポジトリ）を追加してからインストールする。

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

このコマンド列は，配布元のパッケージの署名を確かめるための鍵ファイルを `/etc/apt/keyrings/` に保存し，配布元を `/etc/apt/sources.list.d/github-cli.list` に登録してから，`gh` をインストールする。
`\` で終わる行は次の行に続くため，その 2 行で 1 つのコマンドになる。
各コマンドの成功を確かめてから次へ進み，最後の `gh --version` でバージョンが表示されることを確認する。

WSL 上のディレクトリを VS Code で開くには，Windows 側の VS Code に WSL extension が入っている必要がある。
インストールするには，PowerShell で次を実行する。
`code` が見つからないと表示された場合は，PowerShell を開き直してから実行する。

```powershell
code --install-extension ms-vscode-remote.remote-wsl
```

その後，Ubuntu 側のターミナルを開き，次を実行できることを確認する。

```bash
code --version
code .
```

`code .` は，現在のディレクトリを VS Code で開くコマンドである（`.` の意味は第3節で説明する）。
初回の `code .` では，WSL 側に VS Code Server が自動でインストールされる。
`code: command not found` と表示される場合は，Windows Terminal と Ubuntu を開き直し，Windows 側の VS Code と WSL extension が入っているかを確認する。

#### 環境構築 (Mac)

この手順は Apple Silicon（M シリーズ）の Mac を対象にする。
Apple メニューの「この Mac について」で，「チップ」の欄に Apple M1 などと表示されていれば，Apple Silicon の Mac である。
授業の環境定義ファイル（第6節の `pyproject.toml` と `uv.lock`）でバージョンを固定している PyTorch 2.7.1 には，Intel Mac 用の wheel（インストール用のパッケージファイル）がない。
このため，Intel Mac ではこの手順のまま環境をそろえることはできない。
Intel Mac を使う場合は，授業用に別の対応環境を用意する必要がある。

Mac では，まず，ソフトウェアをインストールするための **パッケージ管理ツール (package manager)** として，Homebrew を導入する。
Terminal を開き，Homebrew の公式ページに掲載されている次のコマンドを実行する。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

インストールの途中で，password の入力や Enter キーの入力を求められることがある。
インストールの完了時に `Next steps` として PATH を設定するコマンドが表示された場合は，それをそのまま実行する（PATH については第6節で説明する）。
次のコマンドで，Homebrew が使えることを確認する。

```bash
brew --version
```

Homebrew が使えるようになったら，VS Code，Git，GitHub CLI をそろえる。
Python 3.12 は，第6節で説明する `uv` でインストールして管理する。
Mac に Git が入っている場合でも，`git --version` で確認してから進める。

```bash
brew install --cask visual-studio-code
brew install git gh
```

Mac のターミナルから VS Code を開くには，`code` コマンドが PATH に入っている必要がある。
まず，次で確認する。

```bash
code --version
```

`command not found: code` などと表示される場合は，VS Code を開き，Command Palette（`Cmd+Shift+P`）から `Shell Command: Install 'code' command in PATH` を実行する。
その後，Terminal を開き直して，もう一度 `code --version` を確認する。
確認できたら，作業ディレクトリで次を実行すると，そのディレクトリが VS Code で開く。

```bash
code .
```

### 3. ターミナル操作とパス

#### 現在地を確かめるコマンド

ターミナルでの作業には，常に「現在地」となるディレクトリがあり，これを **カレントディレクトリ (current directory)** という。
相対パス（後述）で指定する操作はこの現在地を基準に行われるため，まずどこにいるかを把握することが重要である。

| コマンド | 役割 |
| --- | --- |
| `pwd` | 現在地のパスを表示する |
| `ls` | 現在地にあるファイルやディレクトリの一覧を表示する |

`ls` の後にディレクトリのパスを書くと，現在地の代わりにそのディレクトリの中身を表示する。
ファイルのパスを書くと，そのファイルがあればパスをそのまま表示し，なければエラーを表示する。
`ls` は，名前が `.` で始まるファイルやディレクトリ（`.gitignore` や `.venv` など）を表示しない。
これらも表示するには，`ls -a` を使う。

#### パスの書き方

**パス (path)** は，ファイルやディレクトリの場所を表す文字列である。
住所が分かれば建物にたどり着けるのと同じように，パスが分かれば目的のファイルにたどり着ける。
パスは，ディレクトリ名を `/`（スラッシュ）でつないで書く。
次の記号は，パスの中で特別な意味を持つ。

| 記号 | 意味 |
| --- | --- |
| `/` | ディレクトリの区切り。先頭の `/` はルートディレクトリ（最上位） |
| `.` | カレントディレクトリ（現在地） |
| `..` | 親ディレクトリ（1 つ上） |
| `~` | ホームディレクトリ（Ubuntu では `/home/yourname`，Mac では `/Users/yourname`） |

`yourname` は，自分の user name に読み替える。
**絶対パス (absolute path)** は，ルートディレクトリ `/` から始まるパスであり，現在地によらず同じ場所を指す。

```
/home/yourname/workspace/data.csv
```

**相対パス (relative path)** は，現在地を基準として書くパスであり，`/` で始めない。
次の例は，現在地が `/home/yourname/workspace` のときの相対パスである。

```
data.csv                    # 現在地にある data.csv
outputs/log.txt             # 現在地にある outputs ディレクトリの中にある log.txt
../README.md                # 1 つ上のディレクトリにある README.md
../../shared/config.yml     # 2 つ上のディレクトリにある shared/ の中の config.yml
```

1 行目の `data.csv` は，上の絶対パスと同じファイルを指す。
同じファイルでも，現在地が変わると相対パスの書き方は変わるが，絶対パスは変わらない。
`~/workspace` のように `~` で始まるパスは，シェルがホームディレクトリの絶対パスに置き換えるため，どの現在地からでも同じ場所を指す。

#### ディレクトリの移動と作成

現在地を別のディレクトリへ移動するには，`cd` を使う。
`cd ~`，または引数を付けない `cd` を実行すると，ホームディレクトリへ移動する。

```bash
cd workspace      # workspace ディレクトリへ移動
cd ..             # 1 つ上のディレクトリへ戻る（.. は親ディレクトリを表す）
```

`cd workspace` は相対パスで指定しているため，現在地に `workspace` ディレクトリがない場合は `No such file or directory` のようなエラーになる。

新しいディレクトリを作るには，`mkdir` を使う。
`-p` オプションを付けると，途中のディレクトリもまとめて作り，すでにある場合もエラーにしない。

```bash
mkdir -p outputs/exc01
```

このコマンドは，現在地に `outputs` がなければ作り，その中に `exc01` を作る。
`-p` を付けずに `mkdir outputs/exc01` を実行すると，`outputs` がない場合は `cannot create directory` というエラーになる。
解説の冒頭の `figure_dir.mkdir(parents=True, exist_ok=True)` は，Python の中で `mkdir -p` と同じ処理を行う。
この小見出しのコマンドは説明のための例であり，ここで実行する必要はない。
実際の作業場所は，第5節で作る。

### 4. GitHub と Git の初期設定

GitHub を使うには，個人用のアカウントの作成と，GitHub CLI の認証が必要になる。
この節の後半では，Git の著者情報も設定する。

#### GitHub アカウントの作成

すでに個人用の GitHub アカウントを持っている場合は，そのアカウントを使う。
ブラウザでサインインできることと，メールアドレスの確認が済んでいることを確かめ，次の「GitHub CLI の認証」へ進む。

アカウントを持っていない場合は，次の手順で無料の個人用アカウントを作成する。

1. ブラウザで [GitHub の登録ページ](https://github.com/signup)を開く。
2. メールアドレス，パスワード，ユーザー名，国／地域を入力する。
   メールアドレスには，自分で確認メールを受け取れるものを使う。
   ユーザー名には半角英数字とハイフンを使えるが，ハイフンの連続使用や先頭・末尾への使用はできない。
   パスワードは，画面に表示される条件を満たすものを設定する。
3. 画面の案内に従って確認操作を行い，`Create account` を選ぶ。
4. 登録したメールアドレスに届く確認メールの案内に従い，メールアドレスの確認を完了する。
   確認が済んでいないと，リポジトリの作成などの操作ができない。
5. GitHub にサインインできることを確認し，自分のユーザー名を把握しておく。

登録画面では，Google または Apple のアカウントを使う方法も選べる。
上記は，メールアドレスで登録する場合の手順である。
画面の表示や手順が異なる場合は，[GitHub 公式のアカウント作成ガイド](https://docs.github.com/ja/account-and-profile/how-tos/account-management/creating-an-account-on-github)を確認する（登録画面と公式ガイドの確認日：2026 年 10 月 2 日）。

#### GitHub CLI の認証

アカウントの準備ができたら，Ubuntu（WSL）または Mac のターミナルで次のコマンドを実行する。
この資料では，GitHub との通信に **SSH (Secure Shell)** を使う前提にする。

```bash
gh auth login
```

`gh auth login` の途中では，次の方針で選ぶ。
表示される質問や順序は，`gh` のバージョンや，既存の認証・SSH 鍵の設定によって変わる。

- `Where do you use GitHub?`（古い版では `What account do you want to log into?`）：`GitHub.com`
- `What is your preferred protocol for Git operations on this host?`：`SSH`
- SSH 公開鍵の登録を求められたら（`Upload your SSH public key to your GitHub account?`），使用する鍵を選ぶ。鍵がない場合は，`Generate a new SSH key to add to your GitHub account?` で `Yes` を選び，案内に従って作成する。
- `How would you like to authenticate GitHub CLI?`：`Login with a web browser` を選び，表示された code を GitHub の画面に入力する。ブラウザが自動で開かない場合は，表示された URL をブラウザで開く。

SSH では，**公開鍵 (public key)** と **秘密鍵 (private key)** の組を使って，接続する人が本人であることを確かめる。
SSH を選ぶと，`gh` は既存の SSH 鍵を探し，適切な鍵がない場合は，新しい鍵の作成と GitHub への登録を促す。
このとき GitHub に登録されるのは公開鍵であり，秘密鍵は自分の PC または WSL 環境の中に残る。
秘密鍵を他人に見せたり，GitHub に貼り付けたりしてはいけない。

認証と SSH 鍵の設定が終わったら，次のコマンドで状態を確認する。

```bash
gh auth status
ssh -T git@github.com
```

`gh auth status` では，自分の GitHub のユーザー名で login していることと，`Git operations protocol` が `ssh` であることを確認する。
初回の SSH 接続では，接続先のホスト鍵を信頼するかを確認されることがある。
表示された fingerprint（鍵の識別値）を [GitHub 公式の公開鍵の一覧](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)と照合し，一致した場合に `yes` と入力する。
`ssh -T git@github.com` では，次のように，自分の GitHub のユーザー名を含む認証成功のメッセージが出ればよい。

```text
Hi <GitHubユーザー名>! You've successfully authenticated, but GitHub does not provide shell access.
```

コマンドは，終了するときに，成功したかどうかを表す整数を返す。
この整数を **終了コード (exit status)** といい，直前に実行したコマンドの終了コードは `echo $?` で表示できる。
一般に，終了コードは成功なら 0，失敗なら 0 以外である。
ただし，`ssh -T git@github.com` が成功したかどうかは，終了コードではなく，表示されたメッセージで判断する。
`Permission denied (publickey)` と表示された場合は認証の失敗であり，使用する公開鍵が自分の GitHub アカウントに登録されているかを確認する。

#### Git の著者情報

Git の著者情報は，次のコマンドで設定する。

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global --list
```

`user.name` と `user.email` は，commit（第9節）に著者の情報として記録される。
GitHub は，commit に記録されたメールアドレスを使って，その commit を GitHub のアカウントと対応付ける。
このため，自分の GitHub アカウントに登録したメールアドレスと対応する情報にしておく。
`--global` を付けた設定は，その PC（Windows では WSL の Ubuntu）で自分のユーザーが使うすべてのリポジトリに適用される。
`git config --global --list` で，設定した値が表示されることを確認する。

### 5. 作業リポジトリの準備

この授業では，役割の異なる 2 つのリポジトリを使う。
**教材リポジトリ**（`signal-ml-training`）は，この資料などの教材を置く場所であり，GitHub 上や配布された資料で読む。
**作業リポジトリ** は，自分の演習のコード，図，結果を保存する非公開のリポジトリであり，この節で作る。
2 つのリポジトリの使い分けは，[共通手順](README.md#作業場所と保存先)にまとめている。

#### 自分用作業リポジトリの作成

環境構築と各回の解答の保存に使う，自分用の非公開の作業リポジトリを，テンプレートから作成する。

1. GitHub にサインインし，[作業用テンプレート](https://github.com/ykinolab-tokai/signal-ml-work-template)を開く。
2. **Use this template** → **Create a new repository** を選ぶ。
3. **Owner** を `ykinolab-tokai`，**Repository name** を `signal-ml-work-<GitHubユーザー名>`（例：ユーザー名が `taro-lab` なら `signal-ml-work-taro-lab`），公開範囲を **Private** にして作成する。

GitHub 上の名前は `signal-ml-work-<GitHubユーザー名>`，PC 上のフォルダ名は `signal-ml-work` とする。
すでに作成・取得済みなら，同じリポジトリを使う。

テンプレートを開けない，Owner に `ykinolab-tokai` が表示されない，または作成できない場合は，担当教員に連絡する。

#### 作業ディレクトリの作成とリポジトリの clone

GitHub 上のリポジトリを，変更履歴ごと手元の PC にコピーすることを **clone** という。
clone の元になる GitHub 上のリポジトリを，**リモートリポジトリ (remote repository)** と呼ぶ。
この教材では，ホームディレクトリの下に作業場所 `~/workspace` を作り，その中に先ほど作成した自分用の作業リポジトリを，次のコマンドで clone する。

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone REPOSITORY_URL signal-ml-work
```

ここで `REPOSITORY_URL` は，作業リポジトリの **Code** → **SSH** ボタンからコピーした URL に置き換える。

このコマンドで `~/workspace/signal-ml-work` ディレクトリが作られ，作業用テンプレートのファイルと Git の管理情報が保存される。
以降，この教材ではこの場所を `signal-ml-work` ディレクトリ，または作業リポジトリのルートと呼ぶ。
リポジトリを clone したら，次のコマンドで現在地とファイルを確認し，このディレクトリへ移動する。

```bash
pwd
ls
cd signal-ml-work
ls
git status
```

`pwd` は現在地，`ls` は現在地のファイル，`git status` は Git が認識している変更の状態を確認するために使う。
clone した直後の `git status` は，`On branch main` と `nothing to commit, working tree clean` を表示する（表示の意味は第9節で説明する）。

### 6. Python の実行環境

#### Python 環境の構築

この授業では，Python 3.12 系を標準にする。
Python の環境の管理には，**uv** を用いる。
uv は，Python 本体とパッケージのインストールや，仮想環境の作成をまとめて行うツールである。
Ubuntu（WSL）または Mac のターミナルで次を実行し，uv をインストールする。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストールの末尾に表示される PATH の案内に従い，ターミナルを開き直すか，案内された `source` コマンドを実行する。
次のコマンドでバージョンが表示されることを確認してから，先へ進む。

```bash
uv --version
```

作業用テンプレートには，環境定義ファイル `pyproject.toml` と `uv.lock` が含まれている。
`pyproject.toml` は Python のバージョンと必要なパッケージの条件を，`uv.lock` はその条件を満たすように決めた各パッケージの具体的なバージョンを記録したファイルである。
作業リポジトリのルートに両方があることを確認する。

```bash
cd ~/workspace/signal-ml-work
ls pyproject.toml uv.lock
```

続いて，作業リポジトリのルートで次を実行する。
`pyproject.toml` と `uv.lock` に従い，作業リポジトリの中に Python 3.12 の仮想環境 `.venv` が作られる。

```bash
uv python install 3.12
uv sync --locked
source .venv/bin/activate
python --version
which python
```

`uv python install 3.12` は Python 3.12 をインストールし，`uv sync --locked` は `.venv` を作って，`uv.lock` に記録されたバージョンのパッケージをインストールする。
`--locked` は，`uv.lock` を書き換えずに使い，`pyproject.toml` と食い違っていれば停止する指定である。
`uv sync` は `pyproject.toml` のあるディレクトリで実行する。
別のディレクトリで実行すると，`pyproject.toml` が見つからないというエラーになる。
`python --version` が 3.12 系であり，`which python` が `.venv` の中の `python` を指していれば，授業用の環境に入っている。

#### 仮想環境の有効化と PATH

**環境変数 (environment variable)** は，シェルと，そこから起動したプログラムが参照する設定値である。
環境変数 `PATH` には，コマンドを探すディレクトリが `:` で区切って並んでいる。
`python` のようにディレクトリを含まないコマンド名を入力すると，シェルは `PATH` に並んだディレクトリを先頭から順に探し，最初に見つかったファイルを実行する。
`which python` は，このとき実行されるファイルのパスを表示するコマンドである。
見つからない場合は，何も表示しないか，`python not found` のように表示する（シェルによって異なる）。
現在の `PATH` は，`echo $PATH` で表示できる。

`source .venv/bin/activate` は，現在のターミナルの `PATH` の先頭に `.venv/bin` を加えて，仮想環境を有効にする。
このため，有効にした後に `python` と入力すると `.venv/bin/python` が実行され，`.venv` にインストールしたパッケージを使える。
仮想環境が有効なターミナルでは，プロンプト（コマンドの入力を待つ表示）の先頭に `(signal-ml-work)` が付く。
この変更は `source` を実行したターミナルだけに及び，新しく開いたターミナルや，`deactivate` で仮想環境を無効にした後には及ばない。
`python` を実行して `command not found` と表示された場合や，`which python` が `.venv` 以外を指す場合は，現在のターミナルで仮想環境が有効になっているかを確認する。

新しいターミナルで演習を行うときは，毎回，次のように作業リポジトリのルートへ移動して仮想環境を有効にする（[Python 環境の準備・更新と演習の実行](README.md#python環境の準備更新と演習の実行)）。

```bash
cd ~/workspace/signal-ml-work
source .venv/bin/activate
```

`.venv` は毎回作り直す必要はない。
授業で環境定義ファイルの更新が案内された場合は，2 つのファイルを更新してから，作業リポジトリのルートで `uv sync --locked` を実行する。

### 7. VS Code での作業

#### VS Code で作業リポジトリを開く

作業リポジトリを VS Code で開くには，Ubuntu（WSL）または Mac のターミナルで次を実行する。

```bash
cd ~/workspace/signal-ml-work
code .
```

VS Code の Explorer に `pyproject.toml` と `uv.lock` が見えていることを確認する。
WSL の場合は，VS Code の左下やウィンドウ名に `WSL: Ubuntu-24.04` のように表示されていることを確認する。
このウィンドウのターミナルも，Ubuntu 側で動作する。

#### VS Code の実行・デバッグ機能で `.venv` を使う（任意）

以下は，VS Code の実行ボタンやデバッグ機能から，作業リポジトリの仮想環境 `.venv` にある Python を使いたい場合の任意の設定である。
作業リポジトリを VS Code で開いてファイルを編集するだけの場合や，ターミナルで仮想環境を有効にして `python` を実行する場合には，この設定は必要ない。
その場合は，第8節へ進んでよい。

VS Code の Python の実行・デバッグ機能を使うには，Microsoft の Python 拡張機能が必要である。
入っていない場合は，VS Code の拡張機能の画面で `Python` を検索し，Microsoft が提供しているものをインストールする。
WSL の場合は，拡張機能の画面で Ubuntu 側でも有効になっていることを確認し，`Install in WSL: Ubuntu-24.04` と表示される場合はそのボタンでインストールする。

作業用テンプレートの `.vscode/settings.json` では，VS Code が使う Python の初期候補として，`python.defaultInterpreterPath` に `${workspaceFolder}/.venv/bin/python` が設定されている。
`${workspaceFolder}` は VS Code で開いたフォルダを表すため，この値は作業リポジトリの中の `.venv/bin/python` を指す。
値が異なる場合は，この値に変更する。
次に，コードを実行する Python を明示的に選ぶ。

1. Windows では `Ctrl+Shift+P`，Mac では `Cmd+Shift+P` で Command Palette を開く。
2. `Python: Select Interpreter` を実行する。
3. 作業リポジトリの中の `.venv/bin/python` に対応する Python 3.12 を選ぶ。

選択するパスの例は，WSL では `/home/yourname/workspace/signal-ml-work/.venv/bin/python`，Mac では `/Users/yourname/workspace/signal-ml-work/.venv/bin/python` である。
候補に `.venv` が見つからない場合は，作業リポジトリのルートで `uv sync --locked` が成功したことを確認し，Command Palette の `Developer: Reload Window` でウィンドウを再読み込みしてから選び直す。

選択後，既存のターミナルをゴミ箱のボタンで終了し，メニューの `Terminal` → `New Terminal` から新しいターミナルを開く。
次を実行して，そのターミナルの現在地と Python を確認する。

```bash
cd ~/workspace/signal-ml-work
pwd
which python
python --version
```

`pwd` が作業リポジトリのルート，`which python` がその中の `.venv/bin/python`，`python --version` が 3.12 系を示すことを確認する。
新しいターミナルで仮想環境が自動的に有効にならない場合は，作業リポジトリのルートで `source .venv/bin/activate` を実行し，もう一度確認する。

VS Code の実行ボタンを使うには，Python のファイルをエディタで開いた状態で，右上の `Run Python File in Terminal` を押す。
同じ操作は，エディタ内を右クリックし，`Run` → `Python File in Terminal` からも選べる。
`Run Python File in Terminal` は，VS Code で選択した Python を使ってファイルを実行する。
デバッグ機能も，実行環境を個別に指定していなければ，この選択を使う。
テンプレートの `.vscode/settings.json` では `python.terminal.executeInFileDir` が `false` であり，この場合は通常，ファイルのあるディレクトリではなく，開いているフォルダ（作業リポジトリのルート）を現在地としてファイルが実行される。
実行ボタンで使われる Python と現在地は，第8節のコード例のように `sys.executable` と `Path.cwd()` を表示して確かめられる（演習の発展レベル第3問）。
作業リポジトリの環境定義ファイルを更新した場合は，実行前に作業リポジトリのルートで `uv sync --locked` を行う。
詳しくは，[VS Code の Python 実行手順](https://code.visualstudio.com/docs/python/run)を参照する。

### 8. Python スクリプトの実行

#### スクリプトの作成と実行

**スクリプト (script)** は，Python のプログラムを書いた，拡張子が `.py` のテキストファイルである。
スクリプトは，VS Code の Explorer で新しいファイルを作るか，ターミナルで `code exercise/exc01_01.py` のように `code` の後にファイルのパスを書いて開き，編集してから保存する。
演習のスクリプトは，作業リポジトリの `exercise/` に置く（[共通手順](README.md#作業場所と保存先)）。
スクリプトを実行するには，作業リポジトリのルートで仮想環境を有効にしてから，`python` の後にスクリプトのパスを書く。
例えば，演習の基礎第1問のスクリプトは，次のように実行する。

```bash
cd ~/workspace/signal-ml-work
source .venv/bin/activate
python exercise/exc01_01.py
```

`exercise/exc01_01.py` は現在地からの相対パスなので，作業リポジトリのルート以外で同じコマンドを実行すると，`can't open file` というエラーになる。
仮想環境を有効にせずに実行すると，`python` が見つからないか，`.venv` 以外の Python で実行されて `ModuleNotFoundError: No module named 'numpy'` のようなエラーになる。

#### 実行中の Python と保存先の確認

次のコード例は，実行中の Python と現在地を表示し，余弦波のグラフを保存する。
コメントには，作業リポジトリのルートで，仮想環境を有効にして実行したときの表示を示す。

```python
print(sys.executable)  # 作業リポジトリの .venv/bin/python の絶対パス
print(Path.cwd())      # 作業リポジトリのルートの絶対パス

t = np.linspace(0, 1, 201)
v = np.cos(2 * np.pi * 3 * t)
print(t.shape, v.shape)  # (201,) (201,)

fig, ax = plt.subplots()
ax.plot(t, v)
ax.set_xlabel("t")
ax.set_ylabel("cos(6 pi t)")
fig.savefig(figure_dir / "session01_example_cosine.png")
plt.show()
plt.close(fig)

saved_path = figure_dir / "session01_example_cosine.png"
print(saved_path)            # outputs/exc01/session01_example_cosine.png
print(saved_path.exists())   # True
print(saved_path.resolve())  # 作業リポジトリのルートの絶対パスに outputs/exc01/session01_example_cosine.png が続く
```

`sys.executable` は，このスクリプトを実行している Python の実行ファイルの絶対パスである。
仮想環境を有効にしたターミナルで実行すると，`which python` と同じ `.venv/bin/python` のパスになる。
`Path.cwd()` は，実行時の現在地（カレントディレクトリ）を返す。
`np.linspace(0, 1, 201)` は 0 から 1 までを等間隔に分けた 201 点の配列であり，形 `(201,)` は要素数が 201 の 1 次元配列であることを表す（第3回で詳しく扱う）。
`plt.subplots()` は図 `fig` と描画領域 `ax` を作り，`ax.plot` は線を描き，`ax.set_xlabel` と `ax.set_ylabel` は軸のラベルを付ける。
`fig.savefig` は図をファイルに保存し，`plt.show` は図を画面に表示し，`plt.close` は図を閉じる。
`plt.show` で図のウィンドウが開いた場合は，ウィンドウを閉じるとスクリプトの続きが実行される。
画面に表示できない環境ではウィンドウは開かないため，保存した PNG ファイルを VS Code の Explorer で開いて確かめる。

`figure_dir / "session01_example_cosine.png"` のように，`Path` と文字列を `/` でつなぐと，パスを連結できる。
`figure_dir` は相対パス `outputs/exc01` なので，保存先は実行時の現在地を基準に決まる。
`saved_path.resolve()` は，現在地と相対パスをつないだ絶対パスを返す。
作業リポジトリのルートで実行すれば，図は `~/workspace/signal-ml-work/outputs/exc01/` に保存される。
別のディレクトリで実行すると，そのディレクトリの下に `outputs/exc01` が作られて図が保存されるため，実行時の現在地は作業リポジトリのルートにそろえる（[共通手順](README.md#作業場所と保存先)）。

### 9. 変更の記録と GitHub への push

#### Git の基本操作

Git では，作業中のファイルが置かれたディレクトリを **作業ツリー (working tree)** という。
作業ツリーで行った変更のうち，次の記録に含めるものを置いておく場所を **ステージングエリア (staging area)** という。
ステージングエリアの内容を，メッセージを付けて手元のリポジトリの履歴に記録する操作を **commit** といい，記録された 1 回分の変更のまとまりも commit と呼ぶ。
**push** は，手元で作った commit を GitHub 上のリポジトリへ送る操作である。
clone したリポジトリでは，clone 元のリモートリポジトリが `origin` という名前で登録されている。
commit の履歴の流れを **ブランチ (branch)** といい，この授業の通常の作業は `main` ブランチで行う。

これらの操作には，次のコマンドを使う。
コマンドは，作業リポジトリのルートで実行する。

| コマンド | 役割 |
| --- | --- |
| `git status` | 現在のブランチと，作業ツリー・ステージングエリアの状態を表示する |
| `git remote -v` | 登録されているリモートリポジトリ（`origin`）の URL を表示する |
| `git add <ファイルのパス>` | 指定したファイルの変更をステージングエリアに加える |
| `git diff --cached` | ステージングエリアに加えた変更，つまり次の commit に含まれる変更を表示する |
| `git commit -m "<メッセージ>"` | ステージングエリアの内容を，メッセージを付けて commit する |
| `git push origin main` | `main` ブランチの commit を `origin` へ送る |

commit のメッセージには，変更の内容を表す文を書く（例：`第1回: 環境構築と動作確認`）。
push が終わったら，ブラウザで GitHub 上の作業リポジトリを開き，commit のメッセージとファイルが反映されていることを確かめる。
push に失敗した場合は，エラーメッセージを確認し，解決できなければ担当教員に相談する。

#### `git status` の読み方

`git status` は，ファイルを状態ごとの見出しに分けて表示する。
例えば，commit 済みの `README.md` を編集し，新しいファイル `memo.md` と `draft.txt` を作ってから，`git add memo.md` だけを実行した状態では，次のように表示される。

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   memo.md

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        draft.txt
```

`Changes to be committed` には，`git add` でステージングエリアに加えた変更が表示され，これらが次の commit に含まれる。
`new file` は，新しく追加するファイルであることを表す。
`Changes not staged for commit` には，Git が管理しているファイルのうち，変更したがステージングエリアに加えていないものが表示され，`modified` は内容が変更されたことを表す。
`Untracked files` には，Git がまだ管理していないファイルが表示される。
後の 2 つの見出しの下にあるファイルの変更は，`git add` するまで commit に含まれない。
作業ツリーのすべての変更を commit した後は，`nothing to commit, working tree clean` と表示される。

1 行目は現在のブランチを，2 行目は手元の `main` と GitHub 上の `main`（`origin/main`）の関係を表す。
`Your branch is up to date with 'origin/main'.` は，両者の commit が一致していることを表す。
commit した後，push する前には，`Your branch is ahead of 'origin/main' by 1 commit.` のように，GitHub へまだ送っていない commit の数が表示される。
ただし，比べる相手は最後に GitHub と通信した時点（clone，push，pull など）の `origin/main` であり，`git status` 自体は GitHub と通信しない。

Git の管理対象から外すファイルは，リポジトリの `.gitignore` に書く。
作業用テンプレートの `.gitignore` には，仮想環境の `.venv/` や，入力データを置く `/data/` などが書かれている。
これらは `git status` に表示されず，`git add` で指定しても `The following paths are ignored` というメッセージが出て追加されない。
`.venv` は，それぞれの PC で `uv sync --locked` によって作り直せるため，Git では管理しない。

#### 変更内容と履歴の確認

`git diff` は，作業ツリーの変更のうち，まだステージングエリアに加えていないものを行単位で表示する。
`git add` した後の変更を見るには，前に説明した `git diff --cached` を使う。
例えば，commit 済みの `memo.md` の 3 行目を `first line` から `updated line` に書き換えると，`git diff memo.md` は次のように表示する。

```text
diff --git a/memo.md b/memo.md
index 8ad32d8..b945956 100644
--- a/memo.md
+++ b/memo.md
@@ -1,3 +1,3 @@
 # memo

-first line
+updated line
```

`---` の行は変更前のファイルを，`+++` の行は変更後のファイルを表す。
`@@ -1,3 +1,3 @@` は，変更前の 1 行目からの 3 行と，変更後の 1 行目からの 3 行を表示していることを表す。
先頭が `-` の行は削除された行，`+` の行は追加された行であり，1 行を書き換えると，削除と追加の 2 行として表示される。
先頭が空白の行は，変更の前後にある変更されていない行である。
画像のようにテキストでないファイルの変更は，行単位ではなく `Binary files ... differ` のように表示される。
表示が長いときに閲覧用の画面に切り替わった場合は，`q` を押すとコマンドの入力に戻れる。

`git log --oneline` は，commit の履歴を新しい順に 1 行ずつ表示する。
上の例で `memo.md` を追加した commit の後に実行すると，次のように表示される。

```text
20db6a7 Add memo
d82d435 Initial commit
```

各行の先頭の英数字は commit を識別する値（commit ID の先頭部分）であり，commit ごとに異なる。
`git log --oneline -n 3` は最新の 3 件を，`git log --oneline -- <ファイルのパス>` はそのファイルを変更した commit だけを表示する。

## 演習

作業場所・保存先・説明の残し方は [共通手順](README.md#作業場所と保存先) に従ってください。
演習ファイルは，自分の作業リポジトリ（ `~/workspace/signal-ml-work` ）直下の `exercise/` 以下に置いてください。
以下の相対パスは，自分の作業リポジトリを基準とします。
ファイル名は `exc授業回2桁_問題番号2桁.py` の形式に統一してください。第1回の第1問であれば，`exercise/exc01_01.py` とします。
文章で答える問題は，拡張子を `.md` とした同じ形式のファイルに書いてください。第1回の第2問であれば，`exercise/exc01_02.md` とします。
各 `.py` ファイルの先頭には，解説の冒頭にある `import` 文と，保存先ディレクトリ `figure_dir`（`outputs/exc01`）を作成する処理を書いてください。
図は `figure_dir` に，問題番号を含むファイル名で保存してください。第1問の図であれば，`outputs/exc01/exc01_01.png` とします。
コマンドは，特に指定がない限り，Ubuntu（WSL）または Mac のターミナルで，作業リポジトリのルートを現在地とし，`source .venv/bin/activate` で仮想環境を有効にしてから実行してください。
コマンドの結果を記録するときは，表示された行をそのまま貼り付けてください。
パスの `yourname` と `<GitHubユーザー名>` は，自分の user name と GitHub のユーザー名に読み替えてください。
パスワード，token，秘密鍵の内容は，どのファイルにも書かないでください。

基本的にはコマンドとプログラム (Python) を使って取り組むことを想定しています。
ただし，次の指示がある問題は，コマンドやプログラムを実行する前に自分で考えてください。

- **予想してください**：実行前に結果を予想してコメント（文章で答える問題では `.md` ファイル）に書き，実行後に予想と一致したか，一致しなかった場合はその理由を追記してください。

### 基礎レベル（4問）
1. 作業リポジトリの Python 環境で，NumPy と Matplotlib を使うスクリプトを実行できることを確かめます。解説の第6節までの手順を終えた作業リポジトリを使い，ターミナルは VS Code の中ではなく，Windows Terminal の Ubuntu または Mac の Terminal で新しく開いたものとします。

    1. `exercise/exc01_01.py` を作成し，先頭に解説の冒頭の処理を書いてください。続けて，`x = np.linspace(0, 2 * np.pi, 400)` と `y = np.sin(x)` を計算し，`sys.executable` と `x.shape` を表示する処理を書いてください。さらに，横軸のラベルを `x`，縦軸のラベルを `sin(x)` とした $y=\sin x$ のグラフを，`fig.savefig(figure_dir / "exc01_01.png", dpi=150)` で `outputs/exc01/exc01_01.png` に保存して図を閉じる処理と，最後に `setup check completed` と表示する処理を書いてください。グラフの描き方は，解説の第8節のコード例を参考にしてください。
    2. 新しいターミナルで `cd ~/workspace/signal-ml-work` を実行した後，`source .venv/bin/activate` を実行する前と後のそれぞれで，`which python` と `python --version` が何を表示するかを予想してください。何も表示されない，またはエラーになると考える場合は，そのように書いてください。予想は，`exercise/exc01_01.py` の先頭にコメントで書いてください。
    3. 2 の 4 回の実行を実際に行い，結果を予想と比較してください。仮想環境を有効にする前の表示が予想と異なった場合は，`which python3` も実行して比べてください。有効にした後の `python --version` が 3.12 系であることを確認してください。また，仮想環境を有効にすると `which python` の表示が変わる理由を，解説の `PATH` の説明を使ってコメントに書いてください。
    4. 仮想環境を有効にしたまま，作業リポジトリのルートで `python exercise/exc01_01.py` を実行してください。`sys.executable` の表示が 3 の `which python` の表示と一致すること，`x.shape` が `(400,)` であること，最後に `setup check completed` が表示されることを確認してください。2 つのパスが一致する理由も，コメントに書いてください。
    5. VS Code の Explorer で `outputs/exc01/exc01_01.png` を開き，横軸の 0 から約 6.28（$2\pi$）の範囲に，正弦波が 1 周期分描かれていることを確認してください。

2. ターミナルの現在地と，相対パスの基準の関係を確かめます。基礎第1問を終えた作業リポジトリを使い，結果は `exercise/exc01_02.md` に書くこととします。

    1. `exercise/exc01_02.md` を作成し，`## 予想`，`## 結果`，`## 説明` の 3 つの見出しをこの順に書いてください。
    2. 次のコマンドを上から順に実行したときに，4 回の `pwd` が表示するパスと，`ls ../outputs/exc01` が表示するファイル名を予想し，`## 予想` に書いてください。パスの `yourname` の部分は，自分の user name にしてください。

        ```bash
        cd ~/workspace/signal-ml-work/exercise
        pwd
        ls ../outputs/exc01
        cd ../..
        pwd
        cd signal-ml-work/outputs/exc01
        pwd
        cd ~
        pwd
        ```

    3. 2 のコマンドを実行し，表示を `## 結果` に貼り付けて，予想と比較してください。
    4. `outputs/exc01/exc01_01.png` を，(a) 絶対パス，(b) 作業リポジトリのルートからの相対パス，(c) `exercise/` からの相対パスの 3 通りで書いてください。(a) はホームディレクトリで，(b) は作業リポジトリのルートで，(c) は `exercise/` で，それぞれ `ls` の後に書いて実行し，エラーにならずにパスが表示されることを確認してください。
    5. 作業リポジトリのルートで `python exc01_01.py` を実行するとどうなるかを予想してから実行し，表示されたメッセージを `## 結果` に記録してください。同じ場所で `python exercise/exc01_01.py` なら実行できる理由を，相対パスがどこを基準に解釈されるかという観点から `## 説明` に書いてください。また，4 の (a)〜(c) のうち，現在地を変えても同じ書き方で使えるものを答え，その理由を説明してください。

3. この授業の環境と GitHub への接続が，解説の手順どおりに準備できたことを確かめて記録します。結果は `exercise/exc01_03.md` に書き，パスワード，token，秘密鍵の内容は書かないこととします。

    1. `exercise/exc01_03.md` を作成し，`## OS とツールのバージョン`，`## GitHub への接続`，`## 作業リポジトリ` の 3 つの見出しをこの順に書いてください。
    2. Windows の場合は，PowerShell で `wsl --version` と `wsl -l -v` を実行し，WSL のバージョンを表す行と `Ubuntu-24.04` の行を `## OS とツールのバージョン` に書いてください。WSL のバージョンが 2.4.10 以上であることと，`VERSION` が `2` であることを確認してください。さらに，Ubuntu のターミナルで `cat /etc/os-release` を実行し，`VERSION_ID` の行を書いてください。Mac の場合は，`sw_vers -productVersion` と `uname -m` の結果を書き，`uname -m` が `arm64` であることを確認してください。
    3. どちらの OS でも，`git --version`，`gh --version` の 1 行目，`uv --version`，`code --version` の 1 行目を，`## OS とツールのバージョン` に書いてください。
    4. `gh auth status` を実行し，表示された GitHub のユーザー名と `Git operations protocol` の値を `## GitHub への接続` に書いてください。次に，`ssh -T git@github.com` を実行した直後に `echo $?` を実行したときに表示される終了コードを予想してから，2 つのコマンドを実行してください。表示されたメッセージと終了コードを記録して予想と比較し，終了コードが 0 でなくても接続に成功したと判断できる理由を，表示されたメッセージの内容から説明してください。
    5. 作業リポジトリのルートで `git remote -v`，`git branch --show-current`，`git config user.name`，`git config user.email` を実行し，結果を `## 作業リポジトリ` に書いてください。`origin` の URL が `git@github.com:ykinolab-tokai/signal-ml-work-<GitHubユーザー名>.git` であることと，ブランチが `main` であることを確認してください。また，ブラウザで作業リポジトリのページを開き，所有者が `ykinolab-tokai` であることと，リポジトリ名の横に `Private` と表示されていることを確認してください。
    6. push のときに，GitHub がどのようにして本人の操作であることを確かめるかを，`origin` の URL の形式，`gh auth login` で選んだ通信方式，公開鍵と秘密鍵の置き場所から説明してください。

4. 作業リポジトリに加えた変更を commit して GitHub へ push し，各段階の `git status` の表示を確かめます。対象は，基礎第1〜4問で作る `exercise/exc01_01.py`，`outputs/exc01/exc01_01.png`，`exercise/exc01_02.md`，`exercise/exc01_03.md`，`exercise/exc01_04.md` の 5 つのファイルとします。

    1. `exercise/exc01_04.md` を作成し，`## 予想`，`## 結果`，`## 説明` の 3 つの見出しをこの順に書いてください。作業リポジトリのルートで `git status` を実行したときに，5 つのファイルと `.venv/` が，`Changes to be committed`，`Changes not staged for commit`，`Untracked files` のどの見出しの下に表示されるか，または表示されないかを予想してください。さらに，4 で commit した直後と push した直後の `git status` の 2 行目がそれぞれどうなるかも予想し，すべて `## 予想` に書いてください。
    2. `git status` と `git remote -v` を実行し，表示を `## 結果` に貼り付けて，1 の予想と比較してください。ブランチが `main` であることと，`origin` の URL が自分の作業リポジトリを指していることを確認してください。`outputs/exc01/exc01_01.png` が予想どおりに表示されない場合は，`git status --untracked-files=all` も実行して，表示を比べてください。
    3. 次のコマンドで 5 つのファイルをステージングエリアに加え，`git status` で 5 つのファイルが `Changes to be committed` に表示されることと，`git diff --cached --stat` で commit に含まれるファイルの一覧を確認してください。5 つ以外のファイル（解説のコード例で保存した図や，作業用テンプレートの `exercise/check_environment.py` を実行して作られる `outputs/setup/` など）が `Untracked files` に表示された場合は，何のファイルかを確かめ，commit に含めるかどうかを決めてください。含める場合は，そのファイルも `git add` してください。

        ```bash
        git add exercise/exc01_01.py outputs/exc01/exc01_01.png exercise/exc01_02.md exercise/exc01_03.md exercise/exc01_04.md
        git status
        git diff --cached --stat
        ```

    4. 次のコマンドで commit と push を行い，2 回の `git status` の表示を 1 の予想と比較してください。

        ```bash
        git commit -m "第1回: 環境構築と動作確認"
        git status
        git push origin main
        git status
        ```

    5. ブラウザで作業リポジトリの `main` を開き，commit のメッセージと 5 つのファイルが反映されていることを確認してください。push に失敗した場合は，エラーメッセージを記録し，担当教員に相談してください。
    6. 3〜5 の表示と，3 で決めた追加のファイルの扱いとその理由を，ここで `## 結果` と `## 説明` に追記してください。3 の `git add` の後に `exercise/exc01_04.md` を編集すると，その編集は 4 の commit に含まれないため，ここでまとめて追記します。`## 説明` には，4 の 2 回の `git status` で 2 行目の表示が変わった理由を，手元の commit と GitHub 上の commit の関係から書いてください。
    7. 追記した後の `git status` で `exercise/exc01_04.md` がどの見出しの下に表示されるかを予想してから，確かめてください。その後，`git add exercise/exc01_04.md`，`git commit -m "第1回: commit と push の記録"`，`git push origin main` を実行し，最後の `git status` で `Your branch is up to date with 'origin/main'.` と `nothing to commit, working tree clean` が表示されることを確認してください。

### 発展レベル（3問）
1. コードを変更したときに，Git が変更をどのように表示し，記録するかを `git diff` と `git log` で確かめます。基礎第4問で push した後の作業リポジトリを使い，予想と結果は `exercise/exc01_05.md` に書くこととします（差分を調べる `exercise/exc01_05.py` にコメントを書くと，差分の内容が変わるためです）。

    1. `cp exercise/exc01_01.py exercise/exc01_05.py` で基礎第1問のスクリプトをコピーし，`exercise/exc01_05.py` の中の保存先のファイル名を `exc01_05.png` に変更してください。`python exercise/exc01_05.py` を実行して `outputs/exc01/exc01_05.png` が作られることを確認してから，この 2 つのファイルを `git add` し，`git commit -m "第1回: 発展第1問の準備"` と `git push origin main` を実行してください。
    2. `exercise/exc01_05.md` を作成し，`## 予想`，`## 結果`，`## 説明` の 3 つの見出しを書いてください。`exercise/exc01_05.py` の `np.linspace(0, 2 * np.pi, 400)` を `np.linspace(0, 2 * np.pi, 9)` に変更したとき，`git status` で `exercise/exc01_05.py` と `exercise/exc01_05.md` がそれぞれどの見出しの下に表示されるかを予想してください。また，`git diff` の表示で，先頭が `-` の行と `+` の行（`---` と `+++` で始まる行を除く）がそれぞれ何行あり，どのような内容になるかを予想してください。
    3. 2 の変更を保存し，`git status` と `git diff` を実行して予想と比較してください。続けて `python exercise/exc01_05.py` を実行し，`git status` に新たに `modified` として表示されるファイルを確かめてください。`outputs/exc01/exc01_05.png` を開き，基礎第1問の図と比べて線の形がどう変わったかと，その理由を書いてください。
    4. `git add exercise/exc01_05.py outputs/exc01/exc01_05.png` を実行した後に，`git diff` と `git diff --cached` がそれぞれ何を表示するかを予想してから実行してください。`git add` の後に `git diff` が何も表示しない理由を，作業ツリーとステージングエリアの関係から説明してください。
    5. `git commit -m "第1回: 発展第1問 点数を 9 に変更"` を実行してから，`git log --oneline -n 3` と `git log --oneline -- exercise/exc01_05.py` を実行してください。後者が表示する行数を実行前に予想し，結果と比較してください。
    6. コードと，そのコードで作った図を同じ commit に含めておくと，後から履歴を調べるときに何が分かるかを，この問題の 2 つの commit を例に説明してください。最後に `exercise/exc01_05.md` を commit し，push してください。

2. 同じスクリプトでも，実行時の現在地によって相対パスで指定した保存先が変わることを確かめ，保存先の決め方を比べます。予想と説明は `exercise/exc01_06.py` のコメントに書き，仮想環境は有効にしたままとします。

    1. `exercise/exc01_06.py` に，解説の冒頭の処理に続けて，`Path.cwd()` を表示する処理を書いてください。続けて，`x = np.arange(5)` と `y = x**2` の折れ線グラフを，タイトルを `Path.cwd().name`（現在地のディレクトリ名）として `figure_dir / "exc01_06.png"` に保存し，保存した図の絶対パス `(figure_dir / "exc01_06.png").resolve()` を表示する処理を書いてください。
    2. 次の (a)〜(c) の 3 通りで実行したときに，図が保存される場所を絶対パスで予想してください。

        - (a) 作業リポジトリのルートで `python exercise/exc01_06.py` を実行する。
        - (b) `cd exercise` で `exercise/` へ移動してから，`python exc01_06.py` を実行する。
        - (c) `cd ~` でホームディレクトリへ移動してから，`python workspace/signal-ml-work/exercise/exc01_06.py` を実行する。

    3. (a)〜(c) を実行し，表示された絶対パスを予想と比較してください。作業リポジトリのルートに戻って `git status` を実行し，(a)〜(c) で保存された図のうち，どれがどのように表示され，どれが表示されないかを確かめて，その理由を説明してください。
    4. `figure_dir` の定義を次の 2 行に変更し，(a)〜(c) をもう一度実行して保存先を確かめてください。`__file__` は実行中のスクリプトのパスを表し，`Path(__file__).resolve().parent.parent` は，`exercise/` の 1 つ上，つまり作業リポジトリのルートの絶対パスになります。

        ```python
        repository_dir = Path(__file__).resolve().parent.parent
        figure_dir = repository_dir / "outputs" / "exc01"
        ```

    5. 3 で作業リポジトリの外と `exercise/` の下に保存された図を `rm` で 1 つずつ削除してから，空になったディレクトリを `rmdir` で削除してください（例：`rm exercise/outputs/exc01/exc01_06.png` の後に `rmdir exercise/outputs/exc01 exercise/outputs`）。`rmdir` は空でないディレクトリを削除しないため，ほかのファイルを誤って消すことを防げます。`git status` で，`exercise/exc01_06.py` と `outputs/exc01/exc01_06.png` 以外の新しいファイルが表示されないことを確認してください。
    6. この授業では，[共通手順](README.md#作業場所と保存先) のとおり，実行時の現在地を作業リポジトリのルートにそろえます。そのうえで，1 のように現在地を基準とする書き方と，4 のようにスクリプトの位置を基準とする書き方について，それぞれの利点と注意点を 1 つ以上挙げてください。入力ファイル（例：第12回で使う `data/cat.png`）の読み込みも考えて，第2回以降の演習でどちらの書き方を使うかを，理由とともに答えてください。最後に `exercise/exc01_06.py` と `outputs/exc01/exc01_06.png` を commit し，push してください。

3. VS Code の実行ボタンで使われる Python と現在地を確かめ，実行に使う Python を何が決めているかを整理します。解説の第7節の任意の設定（Python 拡張機能のインストールと `Python: Select Interpreter`）を済ませていることとし，予想と説明は `exercise/exc01_07.py` のコメントに書きます。

    1. VS Code で `.vscode/settings.json` を開き，`python.defaultInterpreterPath` と `python.terminal.executeInFileDir` の値を確認してください。
    2. `exercise/exc01_07.py` に，解説の冒頭の処理に続けて，`sys.executable`，`sys.version`，`Path.cwd()` を表示する処理を書いてください。このファイルをエディタで開き，`Run Python File in Terminal` で実行したときに表示される 3 つの値を予想してから実行してください。続けて，仮想環境を有効にしたターミナルで，作業リポジトリのルートから `python exercise/exc01_07.py` を実行し，2 つの方法の表示を比べてください。
    3. メニューの `Terminal` → `New Terminal` で新しいターミナルを開き，プロンプトの先頭に `(signal-ml-work)` が付いているかと，`which python` の表示を確認してください。付いていない場合は，`source .venv/bin/activate` を実行してから確認してください。
    4. `Python: Select Interpreter` で，候補に表示される `.venv` 以外の Python（例：`/usr/bin/python3`）を一時的に選び，`Run Python File in Terminal` で `exercise/exc01_07.py` を実行してください。実行前に，エラーになるかどうかと，なる場合はどの行で何というエラーになるかを予想してください。エラーになった場合は，その原因を説明してください。確認が終わったら `.venv` の Python を選び直し，2 と同じ表示に戻ることを確かめてください。
    5. 実行ボタンを使う場合とターミナルで `python` を使う場合のそれぞれについて，実行に使われる Python を決めているもの（選択した interpreter，`PATH` など）をまとめてください。`git status` で `.vscode/settings.json` が変更されていないことも確認してください。変更されていた場合は，`git diff .vscode/settings.json` で内容を確かめ，`python.defaultInterpreterPath` が `${workspaceFolder}/.venv/bin/python` のままであることを確認してから commit に含めてください。最後に `exercise/exc01_07.py` を commit し，push してください。

## 詰まったときに見る資料
- [`../README.md`](../README.md)
- [`../textbook/markdown/ch01-basic-operations.md`](../textbook/markdown/ch01-basic-operations.md)
- [uv documentation](https://docs.astral.sh/uv/)
- [Pro Git（日本語版）](https://git-scm.com/book/ja/v2)
