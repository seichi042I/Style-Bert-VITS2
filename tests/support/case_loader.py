"""テストケースファイルの読み込み。形式: ファイル名:漢字かな交じり:カタカナ（1行1件、コロン区切り）。"""

from pathlib import Path


def load_cases(path: Path) -> list[tuple[str, str, str]]:
    """テキストファイルから (filename, kanji_kana, kata) のリストを返す。

    空行と # で始まる行はスキップする。
    1行は filename:kanji_kana:kata の形式（コロンは先頭から2つのみ分割）。
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    cases: list[tuple[str, str, str]] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        cases.append((parts[0], parts[1], parts[2]))
    return cases
