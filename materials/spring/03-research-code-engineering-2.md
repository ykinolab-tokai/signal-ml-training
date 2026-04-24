# 第03回 research code engineering II：Git flow・PR・code review・簡単な test

- 対象: B4・M
- 種別: 前期発展枠 / 固定

## この回の目標
- 1 つの変更を branch、commit、draft PR として切り出せる。
- 最小の `unittest` を追加し、CLI script の回帰確認を自分で行える。
- PR に「何を変えたか」「どう確かめるか」「何がまだ不確実か」を書ける。

## ミニ講義
- PR は diff を見せるだけでは不十分で、変更目的、確認方法、未確認事項が揃って初めて review しやすくなる。
- test は「正しさを完全に証明するもの」ではなく、「壊していないことを素早く確認するもの」と考えると設計しやすい。今回のような CLI script では、実行できることとログに期待値が出ることが最低限の回帰確認になる。
- draft PR は、まだ設計や確認が途中でも共有したいときに使う。最終版のふりをせず、未確定部分を先に明示することが重要である。

## 演習
### 基礎レベル
1. 第02回の files があるリポジトリで `git checkout -b session03-pr-practice` を実行する。
2. `test_session02_cli_logging_demo.py` を作成し、次の内容をそのまま書く。
```python
import subprocess
import unittest


class TestSession02CliLoggingDemo(unittest.TestCase):
    def test_cli_runs(self):
        result = subprocess.run(
            [
                "python3",
                "session02_cli_logging_demo.py",
                "--config",
                "session02_config.json",
                "--out",
                "outputs/session03_test_run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("lr=0.001", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
```
3. `python3 -m unittest test_session02_cli_logging_demo.py` を実行し、成功を確認する。
4. `session03_pr_checklist.md` を作成し、次の 4 見出しをこの順で書く。
   - `## 変更目的`
   - `## 追加した test`
   - `## 確認方法`
   - `## review で見てほしい点`
5. `## 変更目的` には `CLI script が実行できることを test で確認する` と書く。
6. `## 確認方法` には `python3 -m unittest test_session02_cli_logging_demo.py` と書く。
7. `## review で見てほしい点` には、reviewer に特に見てほしい観点を 2 行以内で書く。
8. `git add test_session02_cli_logging_demo.py session03_pr_checklist.md`、`git commit -m "session03 add cli test"`、`git push origin session03-pr-practice` を順に実行する。
9. draft pull request を 1 件作成し、タイトルを `session03 pr practice` とする。

### 発展レベル
1. `test_session02_cli_logging_demo.py` に、`outputs/session03_test_run` が作られることを確認する assertion を 1 つ追加する。
2. `session03_pr_checklist.md` に `## 残っているリスク` を追加し、この test ではまだ確認できていない点を 2 行以内で書く。
3. PR 本文には次の 3 見出しをこの順で書く。
   - `## 変更点`
   - `## 確認方法`
   - `## 未確認事項`
4. `## 未確認事項` には、なぜ draft PR のまま残すのかが reviewer に伝わるように 1 行書く。
5. 最後に、`session03_pr_checklist.md` または PR 本文で「この test が regression を防げる理由」を 2 行で説明する。

## 確認ポイント
- branch 名が `session03-pr-practice` である。
- `python3 -m unittest test_session02_cli_logging_demo.py` が成功する。
- draft PR が作成され、本文に `変更点`、`確認方法`、`未確認事項` がある。
- `review で見てほしい点` と `残っているリスク` が書かれており、test の限界が明示されている。

## 詰まったときに見る資料
- [`02-research-code-engineering-1.md`](02-research-code-engineering-1.md)
- Python docs: [`unittest`](https://docs.python.org/3/library/unittest.html), [`subprocess`](https://docs.python.org/3/library/subprocess.html)
