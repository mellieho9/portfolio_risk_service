#!/usr/bin/env python3
"""Fail if any Python file has a comment-to-code ratio above MAX_RATIO.

Counts lines starting with # (after stripping whitespace) as comments.
Blank lines and docstrings are excluded from both numerator and denominator.
Run: python scripts/check_comments.py [files...]
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_RATIO = 0.20  # 20% — more than 1 comment per 5 lines of code is verbose


def _ratio(path: Path) -> tuple[int, int]:
    """Return (comment_lines, code_lines) for a single file."""
    comment_lines = 0
    code_lines = 0
    in_docstring = False
    docstring_char = ""

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        # Track triple-quoted docstrings (skip their content)
        if not in_docstring:
            if line.startswith('"""') or line.startswith("'''"):
                docstring_char = line[:3]
                in_docstring = not line.endswith(docstring_char) or len(line) == 3
                continue
        else:
            if docstring_char in line:
                in_docstring = False
            continue

        if line.startswith("#"):
            comment_lines += 1
        else:
            code_lines += 1

    return comment_lines, code_lines


def main(files: list[str]) -> int:
    violations: list[str] = []

    for f in files:
        path = Path(f)
        if path.suffix != ".py":
            continue
        comments, code = _ratio(path)
        total = comments + code
        if total == 0:
            continue
        ratio = comments / total
        if ratio > MAX_RATIO:
            violations.append(
                f"{path}: {comments}/{total} lines are comments "
                f"({ratio:.0%} > {MAX_RATIO:.0%} limit)"
            )

    if violations:
        print("Comment density too high — write self-documenting code:\n")
        for v in violations:
            print(f"  {v}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
