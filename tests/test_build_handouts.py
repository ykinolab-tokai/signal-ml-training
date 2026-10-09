"""Distribution behavior, checked with the installed Pandoc."""

import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.build_handouts import build, validate_links


class BuildHandoutsTest(unittest.TestCase):
    def test_distribution_preserves_code_math_and_links_without_network_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(Path(__file__).resolve().parents[1] / "scripts/html", root / "scripts/html")
            (root / "LICENSE").write_text("Test license", encoding="utf-8")
            (root / "README.md").write_text("# Policy", encoding="utf-8")
            sources = root / "handouts"
            (sources / "advanced").mkdir(parents=True)
            source = """# 第1回 テスト

## 数式

$x^2$ と次の式。

$$
\\frac{1}{2} \\xleftrightarrow{\\mathrm{DFT}} X
$$

[次](advanced/example.md#日本語の見出し) / [方針](../README.md)
[この式](01-example.md#数式)

![図](figure.svg)

```python
raise RuntimeError("This example must not execute")
```
"""
            (sources / "01-example.md").write_text(source, encoding="utf-8")
            (sources / "figure.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
                '<circle cx="10" cy="10" r="8"/></svg>', encoding="utf-8"
            )
            (sources / "advanced/example.md").write_text(
                "# 発展\n\n## 日本語の見出し\n\n[戻る](../01-example.md#数式)\n",
                encoding="utf-8",
            )
            build(root)
            output = root / "build/handouts"
            html = (output / "01-example.html").read_text(encoding="utf-8")
            self.assertIn('<math ', html)
            self.assertIn('<mover>', html)
            self.assertIn('/blob/main/handouts/advanced/example.md#', html)
            self.assertIn('href="#数式"', html)
            self.assertIn('https://github.com/ykinolab-tokai/signal-ml-training/blob/main/README.md', html)
            self.assertIn('This example must not execute', html)
            self.assertNotIn('<script', html)
            self.assertTrue((output / "index.html").is_file())
            self.assertNotIn('rel="stylesheet"', html)
            self.assertIn('--accent:', html)
            self.assertIn('data:image/svg+xml;base64,', html)
            self.assertIn('Test license', html)
            self.assertFalse((output / "assets").exists())
            self.assertFalse((output / "LICENSE.txt").exists())
            # A renamed HTML file remains usable with no neighboring files.
            with tempfile.TemporaryDirectory() as isolated:
                shutil.copyfile(output / "01-example.html", Path(isolated) / "lesson.html")
                validate_links(Path(isolated))
            self.assertTrue((root / "build/handouts.zip").is_file())
            self.assertEqual((sources / "01-example.md").read_text(encoding="utf-8"), source)

            # A deleted source must not survive in the next distribution.
            (sources / "01-example.md").write_text("# 第1回 更新\n", encoding="utf-8")
            (sources / "advanced/example.md").unlink()
            build(root)
            self.assertFalse((output / "advanced/example.html").exists())

            # Failed builds preserve the last usable distribution.
            previous = (output / "01-example.html").read_bytes()
            (sources / "01-example.md").write_text(
                "# 第1回\n\n[missing](missing.md)\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing.md"):
                build(root)
            self.assertEqual((output / "01-example.html").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
