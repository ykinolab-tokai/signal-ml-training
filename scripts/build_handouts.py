#!/usr/bin/env python3
"""Build offline handout pages and a ZIP using Python's standard library and Pandoc."""

import json
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from html import escape
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

GITHUB = "https://github.com/ykinolab-tokai/signal-ml-training"
MARKER = ".generated-by-build-handouts"


def pandoc(*arguments: str, text: str | None = None) -> str:
    """Run Pandoc without executing any examples in the source document."""
    result = subprocess.run(
        ["pandoc", "--fail-if-warnings", *arguments],
        input=text, text=True, encoding="utf-8", capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Pandoc failed")
    return result.stdout


def nodes(value):
    """Visit Pandoc AST nodes, including nodes inside lists and table cells."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


def rewrite_links(document, source: Path, root: Path):
    """Use online references; resolve image paths for Pandoc resource embedding."""
    for node in nodes(document):
        if node.get("t") not in {"Link", "Image"}:
            continue
        target = node["c"][-1]
        url = urlsplit(target[0])
        if url.scheme or url.netloc or not url.path:
            continue
        local = (source.parent / unquote(url.path)).resolve()
        if not local.is_relative_to(root) or not local.exists():
            raise ValueError(f"{source.relative_to(root)}: missing or invalid link: {target[0]}")
        if local.is_dir() and (local / "README.md").is_file():
            local /= "README.md"
        if node["t"] == "Image":
            path = local.as_uri()
        elif local == source:
            path = ""
        else:
            kind = "tree" if local.is_dir() else "blob"
            path = f"{GITHUB}/{kind}/main/{quote(local.relative_to(root).as_posix())}"
        target[0] = urlunsplit(("", "", path, url.query, url.fragment))


class PageLinks(HTMLParser):
    """Collect the generated IDs and resource references for local validation."""

    def __init__(self, text: str):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append(attributes[key])


def validate_links(stage: Path) -> None:
    """Fail on missing local files or heading anchors before publishing output."""
    pages = {
        path.resolve(): PageLinks(path.read_text(encoding="utf-8"))
        for path in stage.rglob("*.html")
    }
    for page, parsed in pages.items():
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            if not target.is_relative_to(stage.resolve()) or not target.is_file():
                raise ValueError(f"{page.name}: missing local target: {link}")
            if url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                raise ValueError(f"{page.name}: missing heading: {link}")


def render(document, title, output: Path, root: Path) -> None:
    # Pandoc 3.1's texmath lacks this mathtools command used in sessions 08/10.
    # Define its upper-label form locally, leaving the Markdown source intact.
    for node in nodes(document):
        if node.get("t") == "Math" and r"\xleftrightarrow" in node["c"][1]:
            node["c"][1] = (
                r"\newcommand{\xleftrightarrow}[1]{\overset{#1}{\longleftrightarrow}} "
                + node["c"][1]
            )
    document["meta"]["title"] = {"t": "MetaInlines", "c": title}
    output.parent.mkdir(parents=True, exist_ok=True)
    pandoc(
        "--from=json", "--to=html5", "--standalone", "--embed-resources", "--mathml",
        "--toc", "--toc-depth=3",
        f"--template={root / 'scripts/html/page.html'}",
        f"--css={root / 'scripts/html/handouts.css'}",
        f"--variable=index-url:{GITHUB}/blob/main/README.md",
        f"--variable=license-text:{escape((root / 'LICENSE').read_text(encoding='utf-8'))}",
        f"--output={output}", text=json.dumps(document, ensure_ascii=False),
    )


def build(root: Path) -> Path:
    """Build all handouts under root; return the distribution directory.

    The previous generated directory is replaced only after all conversions and
    local link checks succeed. Source Markdown and its code examples are read only.
    """
    root = root.resolve()
    sources = sorted((root / "handouts").rglob("*.md"))
    if not sources:
        raise ValueError("No Markdown files found in handouts/")
    output = root / "build/handouts"
    if output.is_symlink() or (output.exists() and not (output / MARKER).is_file()):
        raise ValueError(f"Refusing to replace a directory not generated by this script: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pages = {source: source.relative_to(root / "handouts").with_suffix(".html") for source in sources}
    with tempfile.TemporaryDirectory(prefix="handouts-", dir=output.parent) as temporary:
        stage = Path(temporary)
        groups = {"第1〜14回": [], "第15〜28回": [], "補足資料・発展テーマ": []}
        for source in sources:
            document = json.loads(pandoc("--from=gfm+tex_math_dollars", "--to=json", str(source)))
            title = next(
                (block["c"][2] for block in document["blocks"] if block["t"] == "Header"),
                [{"t": "Str", "c": source.stem}],
            )
            rewrite_links(document, source, root)
            render(document, title, stage / pages[source], root)
            group = "補足資料・発展テーマ"
            if source.parent == root / "handouts" and source.name[:2].isdigit():
                group = "第1〜14回" if int(source.name[:2]) <= 14 else "第15〜28回"
            groups[group].append([{"t": "Plain", "c": [{
                "t": "Link", "c": [["", [], []], title, [quote(pages[source].as_posix()), ""]],
            }]}])

        index = json.loads(pandoc("--from=gfm", "--to=json", text=(
            "# 配布資料一覧\n\n"
            "各回の資料を選んでください。各HTMLファイルは単独で配布でき，本文と数式はオフラインでも読めます。"
            "他の回・教科書・公式ドキュメントへの参照リンクを開くにはインターネット接続が必要です。\n"
        )))
        for number, (heading, entries) in enumerate(groups.items()):
            if entries:
                index["blocks"].extend([
                    {"t": "Header", "c": [2, [f"group-{number}", [], []], [{"t": "Str", "c": heading}]]},
                    {"t": "BulletList", "c": entries},
                ])
        render(index, [{"t": "Str", "c": "配布資料一覧"}], stage / "index.html", root)
        validate_links(stage)
        (stage / MARKER).write_text("Generated by scripts/build_handouts.py\n", encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(stage), output)
    shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(f"Built {len(sources)} handouts + index: {output}")
    print(f"ZIP: {output.with_suffix('.zip')}")
    return output


if __name__ == "__main__":
    try:
        build(Path(__file__).resolve().parents[1])
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"HTML build failed: {error}")
